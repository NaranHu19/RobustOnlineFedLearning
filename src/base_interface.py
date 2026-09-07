import collections
from typing import Any, cast

import byzfl.fed_framework.models as models
import torch
from byzfl.utils.conversion import (
    flatten_dict,
    unflatten_dict,
    unflatten_generator,
)


class BaseInterface:
    """
    Provide a common interface for federated learning clients and servers.

    Parameters
    ----------
    params : dict[str, Any]
        Configuration parameters used to initialize the model, device,
        optimizer, and learning-rate scheduler.
    """

    def __init__(self, params: dict[str, Any]) -> None:
        # Input validation
        self._validate_params(params)

        # Initialize model
        model_name = params["model_name"]
        self.device = params["device"]

        model = getattr(models, model_name)()

        if self.device == "cuda" and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(model)
        else:
            self.model = model

        self.model.to(self.device)

        # Initialize optimizer. If set to None, the client does not need
        # this information.
        optimizer_name = params["optimizer_name"]
        if optimizer_name is not None:
            optimizer_class = getattr(torch.optim, optimizer_name, None)
            if optimizer_class is None:
                raise ValueError(
                    f"Optimizer '{optimizer_name}' is not supported by PyTorch."
                )

            self.optimizer = optimizer_class(
                self.model.parameters(),
                lr=params["learning_rate"],
                weight_decay=params["weight_decay"],
            )

            gamma = params["learning_rate_decay"]
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda t: (t + 1) ** (-gamma),
            )

    def _validate_params(self, params: dict[str, Any]) -> None:
        """
        Validate the input parameters for correct types and values.

        Parameters
        ----------
        params : dict[str, Any]
            Dictionary of input parameters.

        Raises
        ------
        KeyError
            If a required parameter is missing.
        TypeError
            If ``model_name`` or ``device`` has an invalid type.
        ValueError
            If a numerical parameter has an invalid value.
        """
        # Required keys for Server and Client
        required_keys = ["model_name", "device"]
        if params.get("isServer", False):
            # Required keys for Server, optional for Client
            required_keys += [
                "learning_rate",
                "weight_decay",
                "learning_rate_decay",
            ]

        for key in required_keys:
            if key not in params:
                raise KeyError(f"Missing required parameter: {key}")

        # Validate types and ranges
        if not isinstance(params["model_name"], str):
            raise TypeError("Parameter 'model_name' must be a string.")

        if not isinstance(params["device"], str):
            raise TypeError("Parameter 'device' must be a string.")

        if params["learning_rate"] is not None:
            if (
                not isinstance(params["learning_rate"], float)
                or params["learning_rate"] <= 0
            ):
                raise ValueError("Parameter 'learning_rate' must be a positive float.")

        if params["weight_decay"] is not None:
            if (
                not isinstance(params["weight_decay"], float)
                or params["weight_decay"] < 0
            ):
                raise ValueError(
                    "Parameter 'weight_decay' must be a non-negative float."
                )

        if params["learning_rate_decay"] is not None:
            if (
                not isinstance(params["learning_rate_decay"], float)
                or params["learning_rate_decay"] <= 0
                or params["learning_rate_decay"] > 1.0
            ):
                raise ValueError(
                    "Parameter 'learning_rate_decay' must be a positive "
                    "float smaller than or equal to 1.0."
                )

    def get_flat_parameters(self) -> torch.Tensor:
        """
        Return model parameters as a flat tensor.

        Returns
        -------
        torch.Tensor
            Flattened model parameters.
        """
        return flatten_dict(self.model.state_dict())

    def get_flat_gradients(self) -> torch.Tensor:
        """
        Return model gradients as a flat tensor.

        Returns
        -------
        torch.Tensor
            Flattened model gradients.
        """
        return flatten_dict(self.get_dict_gradients())

    def get_dict_parameters(self) -> dict[str, torch.Tensor]:
        """
        Return model parameters in dictionary form.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing the model parameters.
        """
        return cast(
            dict[str, torch.Tensor],
            self.model.state_dict(),
        )

    def get_dict_gradients(
        self,
    ) -> collections.OrderedDict[str, torch.Tensor]:
        """
        Return model gradients in dictionary form.

        Returns
        -------
        collections.OrderedDict[str, torch.Tensor]
            Ordered dictionary containing the model gradients.
        """
        new_dict: collections.OrderedDict[str, torch.Tensor] = collections.OrderedDict()

        for key, value in self.model.named_parameters():
            if value.grad is None:
                raise RuntimeError(f"Gradient for parameter '{key}' is None.")
            new_dict[key] = value.grad

        return new_dict

    def set_parameters(self, flat_vector: list[torch.Tensor]) -> None:
        """
        Set model parameters from a flat tensor list.

        Parameters
        ----------
        flat_vector : list[torch.Tensor]
            Flat list of parameters to set.
        """
        new_dict = unflatten_dict(
            self.model.state_dict(),
            flat_vector,
        )
        self.model.load_state_dict(new_dict)

    def set_gradients(self, flat_vector: list[torch.Tensor]) -> None:
        """
        Set model gradients from a flat tensor list.

        Parameters
        ----------
        flat_vector : list[torch.Tensor]
            Flat list of gradients to set.
        """
        new_dict = unflatten_generator(
            self.model.named_parameters(),
            flat_vector,
        )

        for key, value in self.model.named_parameters():
            value.grad = new_dict[key].clone().detach()

    def set_model_state(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> None:
        """
        Set the model state dictionary.

        Parameters
        ----------
        state_dict : dict[str, torch.Tensor]
            Dictionary containing the model state.
        """
        self.model.load_state_dict(state_dict)
