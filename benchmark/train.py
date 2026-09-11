import time
from typing import Any

import numpy as np
from byzfl import ByzantineClient, DataDistributor, Server
from byzfl.utils.misc import set_random_seed
from torch import Tensor
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from benchmark.managers import FileManager, ParamsManager
from src.aggregation_time import k_schedule
from src.clients import OnlineClient

transforms_hflip = transforms.Compose(
    [transforms.RandomHorizontalFlip(), transforms.ToTensor()]
)
transforms_mnist = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
transforms_cifar_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)
transforms_cifar_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

# Supported datasets
dict_datasets = {
    "mnist": ("MNIST", transforms_mnist, transforms_mnist),
    "fashionmnist": ("FashionMNIST", transforms_hflip, transforms_hflip),
    "emnist": ("EMNIST", transforms_mnist, transforms_mnist),
    "cifar10": ("CIFAR10", transforms_cifar_train, transforms_cifar_test),
    "cifar100": ("CIFAR100", transforms_cifar_train, transforms_cifar_test),
    "imagenet": ("ImageNet", transforms_hflip, transforms_hflip),
}


def start_training(params: dict[str, Any]) -> None:
    """
    Run a federated learning training experiment.

    Initialize the dataset, clients, server, attack, and aggregation
    strategy from the provided configuration. Train the global model,
    evaluate it at regular intervals, and store the configured results.

    Parameters
    ----------
    params : dict[str, Any]
        Configuration parameters defining the benchmark experiment.
    """
    params_manager = ParamsManager(params)

    # <----------------- File Manager  ----------------->
    file_manager = FileManager(
        {
            "result_path": params_manager.get_results_directory(),
            "model_path": params_manager.get_models_directory(),
            "dataset_name": params_manager.get_dataset_name(),
            "model_name": params_manager.get_model_name(),
            "nb_clients": params_manager.get_nb_clients(),
            "nb_byz": params_manager.get_f(),
            "declared_nb_byz": params_manager.get_tolerated_f(),
            "data_distribution_name": params_manager.get_name_data_distribution(),
            "distribution_parameter": (
                None
                if params_manager.get_name_data_distribution()
                in ["iid", "extreme_niid"]
                else params_manager.get_parameter_data_distribution()
            ),
            "aggregation_name": params_manager.get_aggregator_name(),
            "pre_aggregation_names": [
                dict["name"] for dict in params_manager.get_preaggregators()
            ],
            "attack_name": params_manager.get_attack_name(),
            "learning_rate": params_manager.get_learning_rate(),
            "learning_rate_decay": params_manager.get_learning_rate_decay(),
            "weight_decay": params_manager.get_weight_decay(),
            "aggreg_freq_scale": params_manager.get_training_algorithm_parameters()[
                "aggreg_freq_scale"
            ],
            "aggreg_mult_scale": params_manager.get_training_algorithm_parameters()[
                "aggreg_mult_scale"
            ],
        }
    )

    file_manager.save_config_dict(params_manager.get_data())

    # <----------------- Federated Framework ----------------->

    # Configurations
    nb_clients = params_manager.get_nb_clients()
    nb_byz_clients = params_manager.get_f()
    nb_training_steps = params_manager.get_nb_steps()
    batch_size = params_manager.get_honest_clients_batch_size()

    dd_seed = params_manager.get_data_distribution_seed()
    training_seed = params_manager.get_training_seed()
    set_random_seed(dd_seed)

    # Data Preparation
    key_dataset_name = params_manager.get_dataset_name()
    dataset_name = dict_datasets[key_dataset_name][0]
    dataset = getattr(datasets, dataset_name)(
        root=params_manager.get_data_folder(), train=True, download=True, transform=None
    )
    dataset.targets = Tensor(dataset.targets).long()

    train_size = int(params_manager.get_size_train_set() * len(dataset))
    val_size = len(dataset) - train_size

    # Split Train set into Train and Validation
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Apply transformations to each dataset
    train_dataset.dataset.transform = dict_datasets[key_dataset_name][1]
    val_dataset.dataset.transform = dict_datasets[key_dataset_name][2]

    # Prepare Validation and Test data
    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=params_manager.get_batch_size_evaluation(),
            shuffle=False,
        )
    else:
        val_loader = None

    test_dataset = getattr(datasets, dataset_name)(
        root=params_manager.get_data_folder(),
        train=False,
        download=True,
        transform=dict_datasets[key_dataset_name][2],
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=params_manager.get_batch_size_evaluation(),
        shuffle=False,
    )

    # Distribute data among clients using non-IID Dirichlet distribution
    data_distributor = DataDistributor(
        {
            "data_distribution_name": params_manager.get_name_data_distribution(),
            "distribution_parameter": params_manager.get_parameter_data_distribution(),
            "nb_honest": nb_clients,
            "data_loader": train_dataset,
            "batch_size": batch_size,
        }
    )
    client_dataloaders = data_distributor.split_data()

    max_client_train_size = max(
        len(client_dataloaders[i].dataset) for i in range(nb_clients)
    )
    if nb_training_steps > max_client_train_size:
        raise ValueError(
            "Reduce the maximum amount of local steps, "
            "as client have not enough data for a complete training."
        )

    # Initialize Clients
    clients = [
        OnlineClient(
            {
                "model_name": params_manager.get_model_name(),
                "device": params_manager.get_device(),
                "optimizer_name": params_manager.get_optimizer_name(),
                "learning_rate": params_manager.get_learning_rate(),
                "learning_rate_decay": params_manager.get_learning_rate_decay(),
                "weight_decay": params_manager.get_weight_decay(),
                "loss_name": params_manager.get_loss_name(),
                "LabelFlipping": "LabelFlipping" == params_manager.get_attack_name(),
                "training_dataloader": client_dataloaders[i],
                "nb_labels": params_manager.get_nb_labels(),
                "store_per_client_metrics": (
                    params_manager.get_store_per_client_metrics()
                ),
            }
        )
        for i in range(nb_clients)
    ]

    # Server Setup, Use SGD Optimizer
    server = Server(
        {
            "model_name": params_manager.get_model_name(),
            "device": params_manager.get_device(),
            "validation_loader": val_loader,
            "test_loader": test_loader,
            "optimizer_name": params_manager.get_optimizer_name(),
            "learning_rate": params_manager.get_learning_rate(),
            "learning_rate_decay": params_manager.get_learning_rate_decay(),
            "weight_decay": params_manager.get_weight_decay(),
            "milestones": params_manager.get_milestones(),
            "aggregator_info": params_manager.get_aggregator_info(),
            "pre_agg_list": params_manager.get_preaggregators(),
        }
    )

    # Byzantine Client Setup

    attack_parameters = params_manager.get_attack_parameters()
    attack_parameters["aggregator_info"] = params_manager.get_aggregator_info()
    attack_parameters["pre_agg_list"] = params_manager.get_preaggregators()
    attack_parameters["f"] = nb_byz_clients

    attack_name = params_manager.get_attack_name()

    attack = {
        "name": attack_name,
        "f": nb_byz_clients,
        "parameters": attack_parameters,
    }
    byz_client = ByzantineClient(attack)

    set_random_seed(training_seed)

    evaluation_delta = params_manager.get_evaluation_delta()
    evaluate_on_test = params_manager.get_evaluate_on_test()

    store_models = params_manager.get_store_models()
    store_per_client_metrics = params_manager.get_store_per_client_metrics()

    val_accuracy_list = np.array([], dtype=float)
    test_accuracy_list = np.array([], dtype=float)
    evaluation_times = np.array([], dtype=int)

    start_time = time.time()

    training_algorithm_name = params_manager.get_training_algorithm_name()

    if training_algorithm_name not in ["RobustOnlineFL"]:
        raise ValueError(
            f"Training algorithm {training_algorithm_name} not supported,"
            "supported algorithm is 'RobustOnlineFL'"
        )

    if attack_name == "LabelFlipping":
        raise ValueError("RobustOnlineFL does not support Label Flipping attack.")

    training_algorithm_parameters = params_manager.get_training_algorithm_parameters()

    aggreg_freq_scale = training_algorithm_parameters["aggreg_freq_scale"]
    aggreg_mult_scale = training_algorithm_parameters["aggreg_mult_scale"]

    aggreg_times = np.asarray(
        k_schedule(
            nb_training_steps,
            aggreg_freq_scale,
            aggreg_mult_scale,
        ),
        dtype=int,
    )

    local_updates = np.diff(aggreg_times)
    nb_aggregation_rounds = len(local_updates)
    train_loss_list = np.zeros(nb_aggregation_rounds)
    byz_history = np.zeros(
        (nb_aggregation_rounds, nb_clients),
        dtype=bool,
    )

    if len(aggreg_times) < 2:
        raise ValueError(
            "Aggregation schedule must contain at least t_0 and one "
            "aggregation time."
        )

    if aggreg_times[0] != 0:
        raise ValueError("Aggregation schedule must start at t_0 = 0.")

    if aggreg_times[-1] != nb_training_steps:
        raise ValueError("The final aggregation time must equal nb_training_steps.")

    if np.any(local_updates <= 0):
        raise ValueError("Aggregation times must be strictly increasing.")

    # Training Loop
    for k, num_local_updates in enumerate(local_updates):
        t_start = int(aggreg_times[k])

        # Evaluate Global Model Every Evaluation Delta Aggregation Rounds
        if k % evaluation_delta == 0:
            evaluation_times = np.append(
                evaluation_times,
                t_start,
            )

            if val_loader is not None:
                val_acc = server.compute_validation_accuracy()

                val_accuracy_list = np.append(val_accuracy_list, val_acc)

            if evaluate_on_test:
                test_acc = server.compute_test_accuracy()
                test_accuracy_list = np.append(test_accuracy_list, test_acc)

            if store_models:
                file_manager.save_state_dict(
                    server.get_dict_parameters(), training_seed, dd_seed, t_start
                )

        new_model = server.get_dict_parameters()
        for client in clients:
            client.set_model_state(new_model)

        idx_selected_byz_clients = np.random.choice(
            nb_clients,
            size=nb_byz_clients,
            replace=False,
        )

        byz_idx = set(idx_selected_byz_clients)
        byz_history[k, idx_selected_byz_clients] = True

        train_loss_per_client = []
        honest_weights = []

        client_weights = [None] * nb_clients

        for i, client in enumerate(clients):
            if i in byz_idx:
                continue

            loss = client.compute_model_update(
                num_rounds=int(num_local_updates),
                start_time=t_start,
            )

            honest_weight = client.get_flat_parameters()

            train_loss_per_client.append(loss)
            honest_weights.append(honest_weight)
            client_weights[i] = honest_weight

        train_loss_list[k] = np.mean(train_loss_per_client)

        byz_weights = byz_client.apply_attack(honest_weights)

        if len(byz_weights) != nb_byz_clients:
            raise RuntimeError(
                "Byzantine attack returned "
                f"{len(byz_weights)} vectors, expected {nb_byz_clients}."
            )

        for client_idx, malicious_weight in zip(
            idx_selected_byz_clients,
            byz_weights,
        ):
            client_weights[client_idx] = malicious_weight

        if any(weight is None for weight in client_weights):
            raise RuntimeError("At least one client submission was not constructed.")

        server.update_model_with_weights(client_weights)

    end_time = time.time()

    file_manager.write_array_in_file(
        train_loss_list,
        "train_loss_tr_seed_"
        + str(training_seed)
        + "_dd_seed_"
        + str(dd_seed)
        + ".txt",
    )

    evaluation_times = np.append(
        evaluation_times,
        int(aggreg_times[-1]),
    )

    file_manager.write_array_in_file(
        evaluation_times,
        "evaluation_times.txt",
    )

    file_manager.write_array_in_file(
        aggreg_times,
        "aggregation_times.txt",
    )

    file_manager.write_array_in_file(
        byz_history.astype(int),
        "byz_history_tr_seed_"
        + str(training_seed)
        + "_dd_seed_"
        + str(dd_seed)
        + ".txt",
    )

    if val_loader is not None:
        val_acc = server.compute_validation_accuracy()
        val_accuracy_list = np.append(val_accuracy_list, val_acc)

        if len(evaluation_times) != len(val_accuracy_list):
            raise RuntimeError(
                "Number of validation accuracies does not match "
                "number of evaluation times."
            )

        file_manager.write_array_in_file(
            val_accuracy_list,
            "val_accuracy_tr_seed_"
            + str(training_seed)
            + "_dd_seed_"
            + str(dd_seed)
            + ".txt",
        )

    if evaluate_on_test:
        test_acc = server.compute_test_accuracy()
        test_accuracy_list = np.append(test_accuracy_list, test_acc)

        if len(evaluation_times) != len(test_accuracy_list):
            raise RuntimeError(
                "Number of test accuracies does not match "
                "number of evaluation times."
            )

        file_manager.write_array_in_file(
            test_accuracy_list,
            "test_accuracy_tr_seed_"
            + str(training_seed)
            + "_dd_seed_"
            + str(dd_seed)
            + ".txt",
        )

    if store_per_client_metrics:
        for client_id, client in enumerate(clients):
            client_loss = client.get_loss_list()
            acc = client.get_train_accuracy()

            file_manager.save_loss(client_loss, training_seed, dd_seed, client_id)

            file_manager.save_accuracy(acc, training_seed, dd_seed, client_id)

    if store_models:
        file_manager.save_state_dict(
            server.get_dict_parameters(), training_seed, dd_seed, int(aggreg_times[-1])
        )

    execution_time = end_time - start_time

    file_manager.write_array_in_file(
        np.array(execution_time),
        "train_time_tr_seed_"
        + str(training_seed)
        + "_dd_seed_"
        + str(dd_seed)
        + ".txt",
    )
