from collections.abc import Iterator
from typing import Any, cast

import numpy as np
import torch
from byzfl.fed_framework import ModelBaseInterface
from byzfl.utils.conversion import flatten_dict


class OnlineClient(ModelBaseInterface):  # type: ignore[misc]
    """
    Represent a client performing local online federated learning.

    Parameters
    ----------
    params : dict[str, Any]
        Configuration parameters used to initialize the client, model,
        optimizer, loss function, and training dataloader.
    """

    def __init__(self, params: dict[str, Any]) -> None:
        # Check for correct types and values in params
        if not isinstance(params, dict):
            raise TypeError(
                f"'params' must be of type dict, " f"but got {type(params).__name__}"
            )

        if not isinstance(params["loss_name"], str):
            raise TypeError(
                "'loss_name' must be of type str, "
                f"but got {type(params['loss_name']).__name__}"
            )

        if not isinstance(params["LabelFlipping"], bool):
            raise TypeError(
                "'LabelFlipping' must be of type bool, "
                f"but got {type(params['LabelFlipping']).__name__}"
            )

        if not isinstance(params["nb_labels"], int) or params["nb_labels"] <= 1:
            raise ValueError("'nb_labels' must be an integer greater than 1")

        if not isinstance(
            params["training_dataloader"],
            torch.utils.data.DataLoader,
        ):
            raise TypeError(
                "'training_dataloader' must be a DataLoader, "
                "but got "
                f"{type(params['training_dataloader']).__name__}"
            )

        # Initialize Client instance
        super().__init__(
            {
                # Required parameters
                "model_name": params["model_name"],
                "device": params["device"],
                # Optional parameters
                "learning_rate": params.get("learning_rate", None),
                "weight_decay": params.get("weight_decay", None),
                "milestones": params.get("milestones", None),
                "learning_rate_decay": params.get("learning_rate_decay", None),
                "optimizer_name": params.get("optimizer_name", None),
                "optimizer_params": params.get("optimizer_params", {}),
            }
        )

        self.initial_learning_rate = params["learning_rate"]
        self.learning_rate_decay = params["learning_rate_decay"]

        self.criterion = getattr(torch.nn, params["loss_name"])()
        self.gradient_LF = torch.Tensor([0])
        self.labelflipping = params["LabelFlipping"]
        self.nb_labels = params["nb_labels"]

        self.momentum_gradient = torch.zeros_like(
            torch.cat(tuple(tensor.view(-1) for tensor in self.model.parameters())),
            device=params["device"],
        )

        self.training_dataloader = params["training_dataloader"]
        self.train_iterator: Iterator[tuple[torch.Tensor, torch.Tensor]] = iter(
            self.training_dataloader
        )

        self.store_per_client_metrics = params["store_per_client_metrics"]
        self.loss_list: list[float] = list()
        self.train_acc_list: list[float] = list()

    def _sample_train_batch(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve the next training batch.

        Reinitialize the dataloader iterator when the end of the dataset
        is reached.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Input data and corresponding target labels for the current
            batch.
        """
        try:
            return next(self.train_iterator)
        except StopIteration:
            self.train_iterator = iter(self.training_dataloader)
            return next(self.train_iterator)

    def compute_gradients(self) -> float:
        """
        Compute gradients for the current training batch.

        Compute the local model loss and gradients. When label flipping
        is enabled, also compute and store gradients using flipped labels.
        Optionally record the training loss and accuracy.

        Returns
        -------
        float
            Loss value for the current training batch.
        """
        inputs, targets = self._sample_train_batch()
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        if self.labelflipping:
            self.model.eval()
            targets_flipped = targets.sub(self.nb_labels - 1).mul(-1)

            self._backward_pass(
                inputs,
                targets_flipped,
            )
            self.gradient_LF = self.get_dict_gradients()
            self.model.train()

        train_loss_value = self._backward_pass(
            inputs,
            targets,
            train_acc=self.store_per_client_metrics,
        )

        if self.store_per_client_metrics:
            self.loss_list.append(train_loss_value)

        return train_loss_value

    def _backward_pass(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        train_acc: bool = False,
    ) -> float:
        """
        Perform a backward pass through the model.

        Compute the loss and gradients for the provided inputs and
        targets. Optionally compute and store the training accuracy.

        Parameters
        ----------
        inputs : torch.Tensor
            Input data for the batch.
        targets : torch.Tensor
            Target labels for the batch.
        train_acc : bool, optional
            Whether to compute and store training accuracy. The default
            is False.

        Returns
        -------
        float
            Loss value for the current batch.
        """
        self.model.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss_value = loss.item()
        loss.backward()

        if train_acc:
            _, predicted = torch.max(outputs.data, 1)
            total = targets.size(0)
            correct = (predicted == targets).sum().item()
            acc = correct / total
            self.train_acc_list.append(acc)

        return float(loss_value)

    def compute_model_update(self, num_rounds: int, start_time: int) -> float:
        """
        Perform multiple local model update rounds.

        Sample training batches, compute gradients, update model
        parameters, and optionally record training metrics.

        Parameters
        ----------
        num_rounds : int
            Number of local updates to perform during the interval.
        start_time : int
            Global local-update time t_k at the beginning of the interval.

        Returns
        -------
        float
            Mean loss across all local training rounds.
        """
        losses = np.zeros(num_rounds)

        for i in range(num_rounds):
            t = start_time + i

            learning_rate = self.initial_learning_rate * (
                (t + 1) ** (-self.learning_rate_decay)
            )

            for param_group in self.optimizer.param_groups:
                param_group["lr"] = learning_rate

            inputs, targets = self._sample_train_batch()
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            train_loss_value = self._backward_pass(
                inputs,
                targets,
                train_acc=self.store_per_client_metrics,
            )

            losses[i] = train_loss_value

            self.optimizer.step()

            if self.store_per_client_metrics:
                self.loss_list.append(train_loss_value)

        return float(losses.mean())

    def get_flat_flipped_gradients(self) -> torch.Tensor:
        """
        Return gradients computed with flipped targets.

        Returns
        -------
        torch.Tensor
            Flattened tensor containing gradients computed from flipped
            target labels.
        """
        return cast(
            torch.Tensor,
            flatten_dict(self.gradient_LF),
        )

    def get_loss_list(self) -> list[float]:
        """
        Return recorded training losses.

        Returns
        -------
        list[float]
            Training losses recorded during local training.
        """
        return self.loss_list

    def get_train_accuracy(self) -> list[float]:
        """
        Return recorded training accuracies.

        Returns
        -------
        list[float]
            Training accuracies recorded for processed batches.
        """
        return self.train_acc_list

    def set_model_state(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> None:
        """
        Set the client model state.

        Load the provided model state dictionary into the local model.
        This can be used to synchronize the client with a global model.

        Parameters
        ----------
        state_dict : dict[str, torch.Tensor]
            Model state dictionary containing parameters and buffers.

        Raises
        ------
        TypeError
            If `state_dict` is not a dictionary.
        """
        if not isinstance(state_dict, dict):
            raise TypeError(
                "'state_dict' must be of type dict, "
                f"but got {type(state_dict).__name__}"
            )

        self.model.load_state_dict(state_dict)
