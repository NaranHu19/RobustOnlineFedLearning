"""
Run federated learning experiments with Byzantine attacks and defenses.
"""

import os
import pickle
from typing import Any

import numpy as np
import torch
from byzfl.utils.misc import set_random_seed
from OnlineFedLearning_RandomAttacks import (
    online_federated_averaging_randAttack,
)
from torchvision import datasets, transforms

from src.clients import ByzantineClient, Client
from src.data_distributor import DataDistributor
from src.models import CNN_MNIST
from src.optimizer import CustomOptimizer
from src.server import Server


# Data generation parameters

num_clients = 7
num_attackers = 3
batch_size = 32
num_classes = 10


# Training parameters

decay_factor = 0.1
decay_coeff = 0.1
learning_rate = 0.1
tot_num_loc_rounds = 8450


# Data

SEED = 42
set_random_seed(SEED)


# Data preparation

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

train_dataset = datasets.MNIST(
    root="XXXX",
    train=True,
    download=True,
    transform=transform,
)

test_dataset = datasets.MNIST(
    root="XXXX",
    train=False,
    download=True,
    transform=transform,
)

test_dataset.targets = torch.Tensor(test_dataset.targets).long()

X_test = torch.stack(
    [test_dataset[i][0] for i in range(len(test_dataset))]
)

y_test = torch.tensor(
    [test_dataset[i][1] for i in range(len(test_dataset))]
)


def run_experiment(
    k_alpha: float,
    attack: str,
    defense: str,
    alpha: float,
) -> tuple[
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    list[int],
]:
    """
    Run one experimental configuration and repeat the training five times.

    Initialize the data distributor, server, and clients, execute the
    federated training loop, and calculate the mean and standard deviation
    of the resulting accuracies.

    Parameters
    ----------
    k_alpha : float
        Growth factor used by the local-round scheduling strategy.
    attack : str
        Name of the Byzantine attack applied to malicious clients.
    defense : str
        Name of the aggregation method used to defend against Byzantine
        updates.
    alpha : float
        Dirichlet distribution parameter controlling data heterogeneity
        among clients.

    Returns
    -------
    tuple[numpy.ndarray[Any, Any], numpy.ndarray[Any, Any], list[int]]
        Tuple containing the mean accuracy across repeated experiments,
        the standard deviation of the accuracy across repeated experiments,
        and the local training step schedule.
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Using device:", device)

    data_distributor = DataDistributor(
        train_dataset,
        num_clients,
        batch_size,
        num_classes,
        distribution="Dirich",
        dist_param=alpha,
    )

    client_dataloaders = data_distributor.distribute_data()

    byz_worker = ByzantineClient(
        {
            "name": attack,
            "f": num_attackers,
            "parameters": {"tau": 1.5},
        }
    )

    results: list[list[float]] = []

    for _ in range(5):
        model = CNN_MNIST()
        server = Server(model=model)

        for i, dataloader in enumerate(client_dataloaders):
            client_model = CNN_MNIST()

            client_optimizer = CustomOptimizer(
                client_model,
                lambd=0.1,
                device=device,
            )

            client = Client(
                dataloader=dataloader,
                model=client_model,
                optimizer=client_optimizer,
                idx=i,
                device=device,
            )

            server.add_client(client)

        online_federated_averaging_randAttack(
            server,
            tot_num_loc_rounds,
            k_alpha,
            "MBGD",
            32,
            learning_rate,
            X_test,
            y_test,
            output=False,
            history=True,
            attackers=byz_worker,
            aggeg_func=defense,
            pre_aggreg=False,
            pre_agg_method="NNM",
            decay_gd=True,
            decay_factor_gd=decay_factor,
            decay_constant_gd=decay_coeff,
        )

        results.append(server.loss_history)

    results_array = np.array(results)
    final_means = np.mean(results_array, axis=0)
    final_stds = np.std(results_array, axis=0)

    return (
        final_means,
        final_stds,
        server.local_steps,
    )


# Parameters

checkpoint_file = "XXXXX.pkl"
checkpoint_file2 = "XXXXX.pkl"

alphas = [1 + 1e-3, 1.5, 2, 2.5]
attacks = [
    "SignFlipping",
    "InnerProductManipulation",
    "ALittleIsEnough",
]
agg_func = ["Mean", "TriMean", "GeoMed", "MultiKrum"]
alpha_hetero = [0.2, 1, 100]


# Load or initialize results_dict

if os.path.exists(checkpoint_file):
    with open(checkpoint_file, "rb") as checkpoint_reader:
        results_dict = pickle.load(checkpoint_reader)

    print(f"Loaded checkpoint from {checkpoint_file}")
else:
    results_dict = {}

    for local_alpha in alphas:
        results_dict[local_alpha] = {}

        for attack in attacks:
            results_dict[local_alpha][attack] = {}

            for agg in agg_func:
                results_dict[local_alpha][attack][agg] = {}

                for al in alpha_hetero:
                    results_dict[local_alpha][attack][agg][al] = {}


if os.path.exists(checkpoint_file2):
    with open(checkpoint_file2, "rb") as checkpoint_reader:
        k_schedules = pickle.load(checkpoint_reader)

    print(f"Loaded checkpoint from {checkpoint_file2}")
else:
    k_schedules = []


# Parallel execution

if __name__ == "__main__":
    for local_alpha in alphas:
        for attack in attacks:
            for defense in agg_func:
                for al in alpha_hetero:
                    # Skip if already computed.
                    if results_dict[local_alpha][attack][defense][al]:
                        print(
                            f"Skipping {local_alpha}, {attack}, {defense}, "
                            f"{al} (already done)"
                        )
                        continue

                    means, stds, k_sched = run_experiment(
                        local_alpha,
                        attack,
                        defense,
                        al,
                    )

                    results_dict[local_alpha][attack][defense][al] = {
                        "mean": means,
                        "std": stds,
                    }

                    k_schedules.append(k_sched)

                    with open(
                        checkpoint_file2,
                        "wb",
                    ) as checkpoint_writer:
                        pickle.dump(
                            k_schedules,
                            checkpoint_writer,
                        )

                    with open(
                        checkpoint_file,
                        "wb",
                    ) as checkpoint_writer:
                        pickle.dump(
                            results_dict,
                            checkpoint_writer,
                        )

                    print(
                        f"Saved checkpoint for {local_alpha}, "
                        f"{attack}, {defense}, {al}"
                    )
