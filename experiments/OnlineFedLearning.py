import gc
import math

import torch

from src.clients import ByzantineClient
from src.server import Server
from src.utils import evaluate, k_schedule


def online_federated_averaging(
    server: Server,
    tot_num_loc_rounds: int,
    loc_round_alpha: float,
    methode: str,
    batchsize: int,
    learning_rate: float,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    output: bool = True,
    history: bool = False,
    attackers: ByzantineClient | None = None,
    aggeg_func: str = "Mean",
    pre_aggreg: bool = False,
    pre_agg_method: str = "NNM",
    decay_gd: bool = False,
    decay_factor_gd: float = 0.66,
    decay_constant_gd: float = 0.1,
) -> None:
    """
    Train a federated model using online federated averaging.

    Distribute the current global model to all clients, perform local
    training, optionally apply Byzantine attacks and pre-aggregation,
    aggregate the client model updates, and evaluate the resulting
    global model after each communication round.

    Parameters
    ----------
    server : Server
        Federated learning server coordinating the clients and maintaining
        the global model.
    tot_num_loc_rounds : int
        Maximum total number of local training steps.
    loc_round_alpha : float
        Exponent controlling the growth of the number of local training
        steps between communication rounds.
    methode : str
        Local optimization method used by the clients. Supported methods
        depend on the implementation of ``Client.online_get_model_update``.
    batchsize : int
        Number of samples per mini-batch during local training.
    learning_rate : float
        Learning rate used for local optimization.
    X_test : torch.Tensor
        Test input features used to evaluate the global model.
    y_test : torch.Tensor
        Ground-truth labels corresponding to ``X_test``.
    output : bool, default=True
        If ``True``, print the progress and accuracy after each global
        communication round.
    history : bool, default=False
        If ``True``, store test accuracy and cumulative local training
        steps in ``server.loss_history`` and ``server.local_steps``.
    attackers : ByzantineClient or None, default=None
        Byzantine client used to generate malicious model updates.
        If ``None``, no Byzantine attacks are applied.
    aggeg_func : str, default="Mean"
        Aggregation method used by the server to combine client updates.
    pre_aggreg : bool, default=False
        If ``True``, apply the configured pre-aggregation defense before
        the final model aggregation.
    pre_agg_method : str, default="NNM"
        Pre-aggregation method used when ``pre_aggreg`` is enabled.
    decay_gd : bool, default=False
        Whether to apply learning-rate decay during local training.
    decay_factor_gd : float, default=0.66
        Exponent controlling the rate of learning-rate decay.
    decay_constant_gd : float, default=0.1
        Constant used in the learning-rate decay calculation.

    Returns
    -------
    None
        The server's global model is updated in place. If ``history`` is
        enabled, evaluation results and cumulative local training steps
        are also stored on the server.

    Raises
    ------
    ValueError
        May be raised by client or server methods if an unsupported
        optimization or aggregation method is specified.
    """
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    server.global_model.to(device)

    num_attackers = attackers.f if attackers is not None else 0

    if num_attackers > 0 and num_attackers > math.ceil(
        (len(server.clients) + num_attackers) / 2
    ):
        print("Warning: Impossible to train, more Attackers then Honest Clients.")
        return

    max_datasize = max(client.features.shape[0] for client in server.clients)

    k_sched = k_schedule(max_datasize, loc_round_alpha)

    if tot_num_loc_rounds >= max_datasize:
        print(
            "Warning: Reduce the maximum amount of local steps, "
            "as there is no enough data for a complete trainig."
        )
        return

    k = 0
    local_steps = 0

    if history:
        server.loss_history = []
        server.local_steps = []

    while local_steps + k_sched[k] <= tot_num_loc_rounds:
        if output:
            print(f"Global Round {k + 1}, Local Steps: {k_sched[k]}")

        # Server distributes the global model.
        server.distribute_model()

        # Clients train locally.
        client_models_updates = []
        template_state_dict = server.global_model.state_dict()

        for client in server.clients:
            update = client.online_get_model_update(
                client.idx,
                local_steps,
                local_steps + k_sched[k],
                methode,
                batchsize,
                learning_rate,
                decay=decay_gd,
                decay_factor=decay_factor_gd,
                decay_constant=decay_constant_gd,
            )

            update_cpu = {key: value.cpu() for key, value in update.items()}
            client_models_updates.append(update_cpu)

            gc.collect()
            torch.cuda.empty_cache()

        if attackers is not None:
            attacked_state_dict = attackers.apply_attack_to_model(
                client_models_updates,
                template_state_dict,
            )
            client_models_updates.extend(attacked_state_dict)

        # Server aggregates updates.
        if pre_aggreg:
            client_models_updates = server.pre_aggregate_method(
                client_models_updates,
                num_attackers,
                pre_agg_method,
            )

        server.aggregate_model_updates(
            client_models_updates,
            num_attackers,
            aggeg_func,
        )

        del client_models_updates
        gc.collect()
        torch.cuda.empty_cache()

        # Evaluate the global model on the test set.
        local_steps += k_sched[k]

        test_accuracy = evaluate(
            server.global_model,
            X_test,
            y_test,
            device,
        )

        if history:
            server.loss_history.append(test_accuracy)
            server.local_steps.append(local_steps)

        if output:
            print(f"Global model accuracy {k + 1}: {test_accuracy:.4f}")
            print(
                f"Total local steps so far: {local_steps:.4f}, "
                f"compared to {max_datasize} samples in the biggest client."
            )

        k += 1
