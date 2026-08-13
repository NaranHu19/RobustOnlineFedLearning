from copy import deepcopy
from typing import Any

import byzfl
import torch

from src.utils import flat_updates_avg, unflat_updates_avg


class Client:
    """Represent an honest participant performing local computation."""

    def __init__(
        self,
        dataloader: Any,
        model: Any,
        optimizer: Any,
        idx: int,
        device: torch.device,
    ) -> None:
        """
        Initialize a client with its data, model, optimizer, and device.

        Parameters
        ----------
        dataloader : Any
            DataLoader containing the client's private training data.
        model : Any
            Model used for local training.
        optimizer : Any
            Optimizer responsible for updating the local model parameters.
        idx : int
            Unique identifier assigned to the client.
        device : torch.device
            Device on which the model and client data are stored.
        """
        self.idx = idx
        self.device = device
        self.dataloader = dataloader
        self.local_model = deepcopy(model).to(self.device)
        self.optimizer = optimizer
        self.optimizer.model = self.local_model
        self.global_model = None
        self.features, self.targets = self._fetch_all_data()
        self.features = self.features.to(self.device)
        self.targets = self.targets.to(self.device)

    def _fetch_all_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fetch and concatenate all batches from the client's dataloader.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple containing all input features and corresponding targets.
        """
        all_features: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        for inputs, targets in self.dataloader:
            all_features.append(inputs.to(self.device))
            all_targets.append(targets.to(self.device))

        return torch.cat(all_features, dim=0), torch.cat(all_targets, dim=0)

    def set_global_model(self, global_model: Any) -> None:
        """
        Synchronize the local model with the current global model.

        Parameters
        ----------
        global_model : Any
            Global model whose parameters are copied to the local model.

        Notes
        -----
        The optimizer is updated to reference the synchronized local model.
        """
        self.global_model = global_model
        self.local_model.load_state_dict(global_model.state_dict())
        self.optimizer.model = self.local_model

    def get_model_update(
        self,
        methode: str,
        batchsize: int,
        num_local_rounds: int,
        learning_rate: float,
        decay: bool = False,
        decay_factor: float = 1.0,
        decay_constant: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """
        Perform local training using the selected optimization method.

        Parameters
        ----------
        methode : str
            Optimization method to use. Supported methods are ``"GD"``,
            ``"SGD"``, and ``"MBGD"``.
        batchsize : int
            Number of samples per mini-batch when using ``"MBGD"``.
        num_local_rounds : int
            Number of local optimization rounds to perform.
        learning_rate : float
            Learning rate used during local optimization.
        decay : bool, default=False
            Whether to apply learning-rate decay.
        decay_factor : float, default=1.0
            Exponent controlling the learning-rate decay.
        decay_constant : float, default=1.0
            Constant used in the learning-rate decay schedule.

        Returns
        -------
        dict[str, torch.Tensor]
            State dictionary containing the locally trained model parameters.

        Raises
        ------
        ValueError
            If the global model has not been set or if an unsupported
            optimization method is specified.
        """
        if self.global_model is None:
            print(
                "Debug (Client.get_model_update): Global model not set for the client."
            )
            raise ValueError("Global model not set for the client.")

        initial_global_model_state_dict = deepcopy(self.global_model.state_dict())

        self.local_model.load_state_dict(self.global_model.state_dict())
        self.local_model.to(self.device)

        X = self.features
        y = self.targets

        if X.numel() == 0 or y.numel() == 0:
            print("Debug (Client.get_model_update): Warning: Client has empty data!")

            zero_update_vector = {
                key: torch.zeros_like(param)
                for key, param in initial_global_model_state_dict.items()
            }

            print(
                "Debug (Client.get_model_update):"
                "Returning zero update vector due to empty data."
            )
            return zero_update_vector

        for _ in range(num_local_rounds):
            if methode == "GD":
                self.optimizer.gradient_descent(
                    X,
                    y,
                    lr=learning_rate,
                    max_iters=1,
                    client=False,
                    decay=decay,
                    decay_factor=decay_factor,
                    decay_constant=decay_constant,
                )
            elif methode == "SGD":
                self.optimizer.stochastic_gd(
                    X,
                    y,
                    lr=learning_rate,
                    max_iters=1,
                    client=False,
                    decay=decay,
                    decay_factor=decay_factor,
                    decay_constant=decay_constant,
                )
            elif methode == "MBGD":
                self.optimizer.mini_batch_gd(
                    X,
                    y,
                    lr=learning_rate,
                    batch_size=batchsize,
                    max_iters=1,
                    client=False,
                    decay=decay,
                    decay_factor=decay_factor,
                    decay_constant=decay_constant,
                )
            else:
                raise ValueError("Invalid gradient method specified.")

        return self.local_model.state_dict()

    def get_model_update_decay(
        self,
        idx: int,
        methode: str,
        batch_size: int,
        start: int,
        end: int,
        learning_rate: float,
        decay: bool = False,
        decay_factor: float = 1.0,
        decay_constant: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """
        Perform online local training over a specified data range.

        Parameters
        ----------
        idx : int
            Identifier of the client performing local training.
        methode : str
            Optimization method to use. Supported methods are ``"SGD"``
            and ``"MBGD"``.
        batch_size : int
            Number of samples per mini-batch when using ``"MBGD"``.
        start : int
            Starting index of the data range.
        end : int
            Exclusive ending index of the data range.
        learning_rate : float
            Learning rate used during local optimization.
        decay : bool, default=False
            Whether to apply learning-rate decay.
        decay_factor : float, default=1.0
            Exponent controlling the learning-rate decay.
        decay_constant : float, default=1.0
            Constant used in the learning-rate decay schedule.

        Returns
        -------
        dict[str, torch.Tensor]
            State dictionary containing the locally trained model parameters.

        Raises
        ------
        ValueError
            If the global model has not been set or if an unsupported
            optimization method is specified.
        """
        if self.global_model is None:
            print(
                "Debug (Client.get_model_update): Global model not set for the client."
            )
            raise ValueError("Global model not set for the client.")

        initial_global_model_state_dict = deepcopy(self.global_model.state_dict())

        self.local_model.load_state_dict(self.global_model.state_dict())
        self.local_model.to(self.device)

        X = self.features
        y = self.targets

        if X.numel() == 0 or y.numel() == 0:
            print("Debug (Client.get_model_update): Warning: Client has empty data!")

            zero_update_vector = {
                key: torch.zeros_like(param)
                for key, param in initial_global_model_state_dict.items()
            }

            print(
                "Debug (Client.get_model_update):"
                "Returning zero update vector due to empty data."
            )
            return zero_update_vector

        if methode == "SGD":
            self.optimizer.online_stochastic_gd(
                idx,
                X,
                y,
                start,
                end,
                lr=learning_rate,
                client=False,
                decay=decay,
                decay_factor=decay_factor,
                decay_constant=decay_constant,
            )
        elif methode == "MBGD":
            self.optimizer.online_mini_batch_gd(
                idx,
                X,
                y,
                start,
                end,
                batchsize=batch_size,
                lr=learning_rate,
                client=False,
                decay=decay,
                decay_factor=decay_factor,
                decay_constant=decay_constant,
            )
        else:
            raise ValueError("Invalid gradient method specified.")

        return self.local_model.state_dict()

    def online_get_model_update(
        self,
        idx: int,
        k_sched1: int,
        k_sched2: int,
        methode: str,
        batchsize: int,
        learning_rate: float,
        decay: bool,
        decay_factor: float,
        decay_constant: float,
    ) -> dict[str, torch.Tensor]:
        """
        Perform local training using a scheduled data slice.

        Parameters
        ----------
        idx : int
            Identifier of the client performing local training.
        k_sched1 : int
            Starting index of the scheduled data range.
        k_sched2 : int
            Exclusive ending index of the scheduled data range.
        methode : str
            Optimization method to use. Supported methods are ``"SGD"``
            and ``"MBGD"``.
        batchsize : int
            Number of samples per mini-batch when using ``"MBGD"``.
        learning_rate : float
            Learning rate used during local optimization.
        decay : bool
            Whether to apply learning-rate decay.
        decay_factor : float
            Exponent controlling the learning-rate decay.
        decay_constant : float
            Constant used in the learning-rate decay schedule.

        Returns
        -------
        dict[str, torch.Tensor]
            State dictionary containing the locally trained model parameters.

        Raises
        ------
        ValueError
            If the global model has not been set or if an unsupported
            optimization method is specified.
        """
        if self.global_model is None:
            print(
                "Debug (Client.get_model_update): Global model not set for the client."
            )
            raise ValueError("Global model not set for the client.")

        self.local_model.load_state_dict(self.global_model.state_dict())

        X = self.features
        y = self.targets

        if methode == "SGD":
            self.optimizer.online_stochastic_gd(
                idx,
                X,
                y,
                k_sched1,
                k_sched2,
                lr=learning_rate,
                client=False,
                decay=decay,
                decay_factor=decay_factor,
                decay_constant=decay_constant,
            )
        elif methode == "MBGD":
            self.optimizer.online_mini_batch_gd(
                idx,
                X,
                y,
                k_sched1,
                k_sched2,
                batchsize,
                lr=learning_rate,
                client=False,
                decay=decay,
                decay_factor=decay_factor,
                decay_constant=decay_constant,
            )
        else:
            raise ValueError("Invalid gradient method specified.")

        return self.local_model.state_dict()


class ByzantineClient(byzfl.ByzantineClient):  # type: ignore[misc]
    """
    Simulate malicious clients that perform Byzantine attacks.

    Extend the byzfl ByzantineClient implementation to apply attacks to
    model updates before they are returned to the server.
    """

    def __init__(self, attack_params: dict[str, Any]) -> None:
        """
        Initialize the Byzantine client with attack parameters.

        Parameters
        ----------
        attack_params : dict[str, Any]
            Configuration parameters defining the Byzantine attack.

        Notes
        -----
        The parameters are passed directly to the parent
        ``byzfl.ByzantineClient`` implementation.
        """
        super().__init__(attack_params)

    def apply_attack_to_model(
        self,
        list_of_model_state_dicts: list[dict[str, torch.Tensor]],
        template_model_state_dict: dict[str, torch.Tensor],
    ) -> list[dict[str, torch.Tensor]]:
        """
        Apply the configured Byzantine attack to model state dictionaries.

        Parameters
        ----------
        list_of_model_state_dicts : list[dict[str, torch.Tensor]]
            Model state dictionaries received from participating clients.
        template_model_state_dict : dict[str, torch.Tensor]
            State dictionary defining the expected model parameter structure.

        Returns
        -------
        list[dict[str, torch.Tensor]]
            State dictionaries containing the attacked model updates.
        """
        list_of_flattened_params: list[torch.Tensor] = []

        for model_state_dict in list_of_model_state_dicts:
            model_parameters = list(model_state_dict.values())
            flattened_params = flat_updates_avg(model_parameters)
            list_of_flattened_params.append(flattened_params)

        attacked_flattened_params_list = self.apply_attack(list_of_flattened_params)

        attacked_state_dicts: list[dict[str, torch.Tensor]] = []
        template_parameters = list(template_model_state_dict.values())

        for attacked_flattened_params in attacked_flattened_params_list:
            attacked_parameters = unflat_updates_avg(
                attacked_flattened_params,
                template_parameters,
            )

            attacked_state_dict = {}

            for i, key in enumerate(template_model_state_dict.keys()):
                attacked_state_dict[key] = attacked_parameters[i]

            attacked_state_dicts.append(attacked_state_dict)

        return attacked_state_dicts
