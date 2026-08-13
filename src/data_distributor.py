import random
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader, Dataset, Subset


class DataDistributor:
    """Distribute a dataset among clients using configurable strategies."""

    def __init__(
        self,
        dataset: Dataset,
        num_clients: int,
        batch_size: int,
        num_classes: int,
        distribution: str = "EqStd",
        dist_param: float = 0.5,
    ) -> None:
        """
        Initialize the data distributor with the dataset and configuration.

        Parameters
        ----------
        dataset : Dataset
            Dataset to distribute among the clients.
        num_clients : int
            Number of clients participating in federated learning.
        batch_size : int
            Number of samples in each client data loader batch.
        num_classes : int
            Number of classes in the dataset.
        distribution : str, default="EqStd"
            Distribution strategy to use.
        dist_param : float, default=0.5
            Parameter controlling the degree of data heterogeneity.
        """
        self.dataset: Dataset[Any] = dataset
        self.num_clients = num_clients
        self.batch_size = batch_size
        self.distribution = distribution
        self.distribution_parameter = dist_param
        self.num_classes = num_classes

    def iid_idx(self, idx: list[int]) -> list[npt.NDArray[np.int_]]:
        """
        Split shuffled indices uniformly among clients.

        Parameters
        ----------
        idx : list[int]
            Dataset indices to distribute.

        Returns
        -------
        list[numpy.typing.NDArray[numpy.int_]]
            List of index arrays, with one array assigned to each client.
        """
        random.shuffle(idx)
        splits = np.array_split(idx, self.num_clients)
        return [np.asarray(split, dtype=np.int_) for split in splits]

    def extreme_niid_idx(
        self,
        idx: list[int],
    ) -> list[npt.NDArray[np.int_]]:
        """
        Split indices according to sorted class labels.

        Samples are grouped by their class labels before being divided
        among clients. This produces an extreme Non-IID distribution in
        which clients may receive samples from only a limited number of
        classes.

        Parameters
        ----------
        idx : list[int]
            Dataset indices to distribute.

        Returns
        -------
        list[numpy.typing.NDArray[numpy.int_]]
            List of index arrays, with one array assigned to each client.
        """
        if len(idx) == 0:
            return [
                np.array([], dtype=np.int_) for _ in range(self.num_clients)
            ]

        targets = self.dataset.targets
        sorted_idx = np.array(sorted(zip(targets[idx], idx)))[:, 1]

        splits = np.array_split(sorted_idx, self.num_clients)
        return [np.asarray(split, dtype=np.int_) for split in splits]

    def distribute_data(self) -> list[DataLoader[Any]]:
        """
        Distribute the dataset according to the selected strategy.

        Apply the configured distribution method and create one PyTorch
        DataLoader for each client.

        Returns
        -------
        list[DataLoader[Any]]
            Data loaders containing the data assigned to each client.

        Raises
        ------
        ValueError
            If the configured distribution strategy is not supported.
        """
        data_size = len(self.dataset)
        indices = list(range(data_size))
        np.random.shuffle(indices)

        if self.distribution == "EqStd":
            split_size = data_size // self.num_clients
            client_indices = [
                indices[i * split_size : (i + 1) * split_size]
                for i in range(self.num_clients)
            ]

        elif self.distribution == "GammaNiid":
            nb_similarity = int(
                len(indices) * self.distribution_parameter
            )
            iid = self.iid_idx(indices[:nb_similarity])
            niid = self.extreme_niid_idx(indices[nb_similarity:])

            split_idx = [
                np.concatenate((iid[i], niid[i]))
                for i in range(self.num_clients)
            ]
            client_indices = [
                node_idx.astype(int).tolist()
                for node_idx in split_idx
            ]

        elif self.distribution == "Dirich":
            sample = np.random.dirichlet(
                np.repeat(
                    self.distribution_parameter,
                    self.num_clients,
                ),
                size=self.num_classes,
            )

            class_indices_dict: dict[int, torch.Tensor] = {}
            indices_tensor = torch.tensor(indices)

            if isinstance(self.dataset.targets, list):
                targets_tensor = torch.tensor(self.dataset.targets)
            else:
                targets_tensor = self.dataset.targets

            for k in range(self.num_classes):
                class_k_indices = targets_tensor[indices_tensor] == k
                class_k = indices_tensor[class_k_indices]

                if class_k.numel() > 0:
                    class_indices_dict[k] = class_k
                else:
                    class_indices_dict[k] = torch.tensor(
                        [],
                        dtype=torch.long,
                    )

            client_indices = [[] for _ in range(self.num_clients)]

            for k in range(self.num_classes):
                class_k_indices = class_indices_dict[k]
                num_class_k_samples = len(class_k_indices)

                if num_class_k_samples > 0:
                    client_class_k_counts = torch.tensor(
                        sample[k] * num_class_k_samples
                    ).long()

                    diff = (
                        num_class_k_samples
                        - torch.sum(client_class_k_counts).item()
                    )

                    if diff != 0:
                        client_class_k_counts[: abs(diff)] += torch.sign(
                            torch.tensor(diff)
                        ).long()

                    split_indices = torch.split(
                        class_k_indices,
                        client_class_k_counts.tolist(),
                    )

                    for client_idx in range(self.num_clients):
                        client_indices[client_idx].extend(
                            split_indices[client_idx].tolist()
                        )

        else:
            raise ValueError(
                f"Distribution strategy '{self.distribution}' "
                "is not supported."
            )

        client_dataloaders: list[DataLoader[Any]] = []

        for idx in client_indices:
            subset = Subset(self.dataset, idx)
            dataloader = DataLoader(
                subset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=4 if torch.cuda.is_available() else 0,
            )
            client_dataloaders.append(dataloader)

        return client_dataloaders
