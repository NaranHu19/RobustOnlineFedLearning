import numpy as np
import torch

from src.clients import ByzantineClient
from src.server import Server
from src.utils import evaluate, k_schedule


def online_federated_averaging_randAttack(
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
    Train a federated model with randomly assigned Byzantine attackers.

    During each communication round, the server distributes the current
    global model to all clients. Clients perform local training and,
    optionally, a subset of updates is replaced with Byzantine updates
    before aggregation. The server then aggregates the client updates,
    evaluates the global model, and optionally records the training
    history.

    Parameters
    ----------
    server : Server
        Federated learning server coordinating the participating clients.
    tot_num_loc_rounds : int
        Total number of local training steps to perform.
    loc_round_alpha : float
        Exponent controlling the growth of the local-step schedule.
    methode : str
        Local optimization method (e.g., ``"SGD"`` or ``"MBGD"``).
    batchsize : int
        Mini-batch size used during local training.
    learning_rate : float
        Initial learning rate for client optimization.
    X_test : torch.Tensor
        Test features used for evaluating the global model.
    y_test : torch.Tensor
        Test labels corresponding to ``X_test``.
    output : bool, default=True
        If ``True``, print training progress after each communication round.
    history : bool, default=False
        If ``True``, store the evaluation history and cumulative local
        training steps in the server.
    attackers : ByzantineClient or None, default=None
        Byzantine attack model used to generate malicious client updates.
        If ``None``, no attacks are applied.
    aggeg_func : str, default="Mean"
        Aggregation rule used to combine client model updates.
    pre_aggreg : bool, default=False
        Whether to apply a pre-aggregation defense before aggregation.
    pre_agg_method : str, default="NNM"
        Name of the pre-aggregation method.
    decay_gd : bool, default=False
        Whether to use a decaying learning rate during local training.
    decay_factor_gd : float, default=0.66
        Exponent controlling the learning-rate decay.
    decay_constant_gd : float, default=0.1
        Constant used to compute the decaying learning rate.

    Returns
    -------
    None
        This function updates the server model in place and optionally
        stores training statistics in ``server.loss_history`` and
        ``server.local_steps``.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    server.global_model.to(device)

    num_attackers = attackers.f if attackers is not None else 0

    if num_attackers > 0 and 2 * num_attackers >= len(server.clients):
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
            client_models_updates.append(
                client.online_get_model_update(
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
            )

        if attackers is not None:
            attacked_state_dict = attackers.apply_attack_to_model(
                client_models_updates,
                template_state_dict,
            )

            attackers_ids = np.random.choice(
                np.arange(1, len(server.clients)),
                size=attackers.f,
                replace=False,
            )

            if output:
                print(
                    "The Byzantine Clients are: ",
                    *attackers_ids,
                )

            for pos, val in zip(attackers_ids, attacked_state_dict):
                client_models_updates[pos] = val

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
