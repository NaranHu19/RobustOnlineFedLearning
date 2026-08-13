from copy import deepcopy

import byzfl
import torch
from torch import nn

from src.clients import Client
from src.utils import flat_updates_avg, unflat_updates_avg


class Server:
    """Coordinate clients and aggregate their model updates."""

    def __init__(self, model: nn.Module) -> None:
        """
        Initialize the server with a template model.

        Parameters
        ----------
        model : nn.Module
            Model to use as the initial global model.
        """
        self.global_model = deepcopy(model)
        self.clients: list[Client] = []
        self.loss_history: list[float] = []
        self.local_steps: list[int] = []

    def add_client(self, client: Client) -> None:
        """
        Register a client with the server.

        Parameters
        ----------
        client : Client
            Client to add to the server's list of participating clients.
        """
        self.clients.append(client)

    def distribute_model(self) -> None:
        """
        Distribute the current global model to all registered clients.

        Each client receives a deep copy of the current global model so that
        local training can be performed independently.
        """
        for client in self.clients:
            client.set_global_model(deepcopy(self.global_model))

    def pre_aggregate_method(
        self,
        client_models_updates: list[dict[str, torch.Tensor]],
        num_attackers: int = 0,
        pre_agg_method: str = "NNM",
    ) -> list[dict[str, torch.Tensor]]:
        """
        Preprocess client updates before final aggregation.

        Client updates are processed layer by layer using the selected
        preprocessing method. Currently, NNM (Nearest Neighbor Mixing)
        is supported through the ByzFL library.

        Parameters
        ----------
        client_models_updates : list[dict[str, torch.Tensor]]
            Model state dictionaries containing the updates produced by
            participating clients.
        num_attackers : int, default=0
            Number of Byzantine attackers participating in the aggregation.
        pre_agg_method : str, default="NNM"
            Pre-aggregation method to apply. Currently, only ``"NNM"`` is
            supported.

        Returns
        -------
        list[dict[str, torch.Tensor]]
            Preprocessed client model updates.

        Raises
        ------
        ValueError
            If ``pre_agg_method`` is not supported.
        """
        if not client_models_updates:
            print("No client model updates received for pre-aggregation.")
            return client_models_updates

        n_clients = len(client_models_updates)

        if pre_agg_method == "NNM":
            agg = byzfl.NNM(num_attackers)
        else:
            raise ValueError(
                f"Pre-aggregation method '{pre_agg_method}' not supported."
            )

        keys = list(client_models_updates[0].keys())

        # Create list of empty dictionaries for the pre-aggregated updates.
        pre_aggregated_updates: list[dict[str, torch.Tensor]] = [
            {} for _ in range(n_clients)
        ]

        for key in keys:
            # Flatten all client tensors for this key.
            flattened = [
                client_update[key].flatten()
                for client_update in client_models_updates
            ]
            stacked = torch.stack(flattened, dim=0)

            agg_result = agg(stacked)

            # Ensure proper shape.
            if agg_result.ndim == 1:
                agg_result = agg_result.unsqueeze(0)

            n_buckets = agg_result.shape[0]

            for client_idx in range(n_clients):
                bucket_idx = client_idx % n_buckets
                pre_aggregated_updates[client_idx][key] = (
                    agg_result[bucket_idx]
                    .view(client_models_updates[0][key].shape)
                    .detach()
                    .clone()
                    .to(client_models_updates[0][key].device)
                    .type(client_models_updates[0][key].dtype)
                )

        # Sanity check.
        for i in range(n_clients):
            for key in keys:
                assert (
                    pre_aggregated_updates[i][key].shape
                    == client_models_updates[0][key].shape
                ), f"Shape mismatch for key {key}, client {i}"

        return pre_aggregated_updates

    def aggregate_model_updates(
        self,
        client_models_updates: list[dict[str, torch.Tensor]],
        num_attackers: int = 0,
        aggeg_func: str = "Mean",
    ) -> None:
        """
        Aggregate client model updates using a robust aggregation rule.

        The client model updates are flattened into vectors, processed using
        the selected aggregation method, and reshaped before updating the
        global model.

        Parameters
        ----------
        client_models_updates : list[dict[str, torch.Tensor]]
            Model state dictionaries containing the updates produced by
            participating clients.
        num_attackers : int, default=0
            Number of Byzantine attackers participating in the aggregation.
        aggeg_func : str, default="Mean"
            Aggregation method to use. Supported methods are ``"Mean"``,
            ``"TriMean"``, ``"GeoMed"``, and ``"MultiKrum"``.

        Raises
        ------
        ValueError
            If ``aggeg_func`` is not supported.
        """
        if not client_models_updates:
            print("No client model updates received for aggregation.")
            return

        template_state_dict = client_models_updates[0]

        # Flatten all client model parameters.
        flattened_client_params: list[torch.Tensor] = []

        for state_dict in client_models_updates:
            flattened_params = flat_updates_avg(list(state_dict.values()))
            flattened_client_params.append(flattened_params)

        aggregated_flattened_params: torch.Tensor

        if aggeg_func == "Mean":
            agg = byzfl.Average()
            aggregated_flattened_params = agg(
                torch.stack(flattened_client_params)
            )

        elif aggeg_func == "TriMean":
            agg = byzfl.TrMean(num_attackers)
            aggregated_flattened_params = agg(
                torch.stack(flattened_client_params)
            )

        elif aggeg_func == "GeoMed":
            agg = byzfl.GeometricMedian(nu=0.0, T=100)
            aggregated_flattened_params = agg(
                torch.stack(flattened_client_params)
            )

        elif aggeg_func == "MultiKrum":
            agg = byzfl.MultiKrum(num_attackers)
            aggregated_flattened_params = agg(
                torch.stack(flattened_client_params)
            )

        else:
            raise ValueError(
                f"Aggregation function '{aggeg_func}'"
                " not supported for model aggregation."
            )

        # Unflatten the aggregated parameters.
        aggregated_parameters = unflat_updates_avg(
            aggregated_flattened_params,
            list(template_state_dict.values()),
        )

        # Update the global model.
        aggregated_state_dict: dict[str, torch.Tensor] = {}

        for i, key in enumerate(template_state_dict.keys()):
            aggregated_state_dict[key] = aggregated_parameters[i]

        self.global_model.load_state_dict(aggregated_state_dict)
