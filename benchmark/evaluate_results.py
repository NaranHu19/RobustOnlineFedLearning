import json
import math
import os
from typing import Any, TypeVar

import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt

T = TypeVar("T")


def custom_dict_to_str(dictionary: dict[str, Any]) -> str:
    """
    Convert a dictionary to a string.

    Return an empty string if the dictionary is empty.

    Parameters
    ----------
    dictionary : dict[str, Any]
        Dictionary to convert to a string.

    Returns
    -------
    str
        String representation of the dictionary, or an empty string if
        the dictionary is empty.
    """
    return "" if not dictionary else str(dictionary)


def ensure_list(value: T | list[T]) -> list[T]:
    """
    Ensure that a value is returned as a list.

    Wrap the value in a list if it is not already a list.

    Parameters
    ----------
    value : T or list[T]
        Value or list of values to convert to a list.

    Returns
    -------
    list[T]
        Original list if the input is already a list, otherwise a list
        containing the input value.
    """
    if not isinstance(value, list):
        value = [value]
    return value


def find_best_hyperparameters(path_to_results: str) -> None:
    """
    Find the best hyperparameters across different attacks.

    Find the learning rate, learning rate decay, and weight decay that
    maximize the minimum accuracy across different attacks. Read the
    configuration from ``config.json`` and save the best hyperparameters
    and the step at which the maximum accuracy is reached for each
    aggregator and attack.

    Parameters
    ----------
    path_to_results : str
        Path to the directory containing the benchmark configuration
        and result files.
    """
    try:
        with open(os.path.join(path_to_results, "config.json")) as file:
            data = json.load(file)
    except Exception as e:
        print(f"ERROR reading config.json: {e}")
        return

    path_hyperparameters = path_to_results + "/best_hyperparameters"

    # <-------------- Benchmark Config ------------->
    training_seed = data["benchmark_config"]["training_seed"]
    nb_training_seeds = data["benchmark_config"]["nb_training_seeds"]
    nb_clients = data["benchmark_config"]["nb_clients"]
    nb_byz = data["benchmark_config"]["f"]
    nb_declared = data["benchmark_config"].get("tolerated_f", None)
    data_distrib_seed = data["benchmark_config"]["data_distribution_seed"]
    nb_data_distribution_seeds = data["benchmark_config"]["nb_data_distribution_seeds"]
    data_distributions = data["benchmark_config"]["data_distribution"]
    nb_steps = data["benchmark_config"]["nb_steps"]

    # <-------------- Evaluation and Results ------------->
    evaluation_delta = data["evaluation_and_results"]["evaluation_delta"]

    # <-------------- Model Config ------------->
    model_name = data["model"]["name"]
    dataset_name = data["model"]["dataset_name"]
    lr_list = data["model"]["learning_rate"]
    lrd_list = data["model"]["learning_rate_decay"]
    wd_list = data["model"]["weight_decay"]

    # <-------------- Aggregators Config ------------->
    aggregators = data["aggregator"]

    if "pre_aggregators" not in data.keys():
        data["pre_aggregators"] = []

    for pre_agg in data["pre_aggregators"]:
        if "parameters" not in pre_agg.keys():
            pre_agg["parameters"] = {}
    pre_aggregators = data["pre_aggregators"]

    # <-------------- Attacks Config ------------->
    attacks = data["attack"]

    # <-------------- Training Algorithm Config ------------->
    af_list = data["benchmark_config"]["training_algorithm"]["parameters"][
        "aggreg_freq_scale"
    ]
    am_list = data["benchmark_config"]["training_algorithm"]["parameters"][
        "aggreg_mult_scale"
    ]

    # Ensure certain configurations are always lists
    nb_clients = ensure_list(nb_clients)
    nb_byz = ensure_list(nb_byz)
    nb_declared = ensure_list(nb_declared)
    data_distributions = ensure_list(data_distributions)
    aggregators = ensure_list(aggregators)

    # Pre-aggregators can be multiple or single dict; unify them
    if not pre_aggregators or isinstance(pre_aggregators[0], dict):
        pre_aggregators = [pre_aggregators]

    attacks = ensure_list(attacks)
    lr_list = ensure_list(lr_list)
    lrd_list = ensure_list(lrd_list)
    wd_list = ensure_list(wd_list)

    af_list = ensure_list(af_list)
    am_list = ensure_list(am_list)

    # Number of accuracy checkpoints
    nb_accuracies = 1 + math.ceil(nb_steps / evaluation_delta)

    # Main nested loops to explore configurations
    for nb_client in nb_clients:
        for nb_byzantine in nb_byz:
            if nb_declared[0] is None:
                nb_declared_list = [nb_byzantine]
            else:
                nb_declared_list = nb_declared.copy()
                nb_declared_list = [
                    item for item in nb_declared_list if item >= nb_byzantine
                ]

            for nb_decl in nb_declared_list:
                for data_dist in data_distributions:
                    distribution_parameter_list = ensure_list(
                        data_dist["distribution_parameter"]
                    )
                    for distrib_parameter in distribution_parameter_list:
                        for af in af_list:
                            for am in am_list:
                                for pre_agg in pre_aggregators:
                                    # Build a single name from all pre-aggregators
                                    pre_agg_names_list = [p["name"] for p in pre_agg]
                                    pre_agg_names = "_".join(pre_agg_names_list)

                                    real_hyper_parameters = np.zeros(
                                        (len(aggregators), 3)
                                    )
                                    real_steps = np.zeros(
                                        (len(aggregators), len(attacks))
                                    )

                                    for k, agg in enumerate(aggregators):
                                        num_combinations = (
                                            len(lr_list) * len(lrd_list) * len(wd_list)
                                        )
                                        max_acc_config = np.zeros(
                                            (num_combinations, len(attacks))
                                        )
                                        hyper_parameters = np.zeros(
                                            (num_combinations, 3)
                                        )
                                        steps_max_reached = np.zeros(
                                            (num_combinations, len(attacks))
                                        )

                                        index_combination = 0
                                        for lr in lr_list:
                                            for lrd in lrd_list:
                                                for wd in wd_list:
                                                    tab_acc = np.zeros(
                                                        (
                                                            len(attacks),
                                                            nb_data_distribution_seeds,
                                                            nb_training_seeds,
                                                            nb_accuracies,
                                                        )
                                                    )

                                                    for i, attack in enumerate(attacks):
                                                        for run_dd in range(
                                                            nb_data_distribution_seeds
                                                        ):
                                                            for run in range(
                                                                nb_training_seeds
                                                            ):
                                                                file_name = (
                                                                    f"{dataset_name}_"
                                                                    f"{model_name}_"
                                                                    f"n_{nb_client}_"
                                                                    f"f_{nb_byzantine}_"
                                                                    f"d_{nb_decl}_"
                                                                    f"{
                                                                     custom_dict_to_str(
                                                                     data_dist['name']
                                                                     )
                                                                    }_"
                                                                    f"{
                                                                     distrib_parameter
                                                                    }_"
                                                                    f"{
                                                                     custom_dict_to_str(
                                                                     agg['name']
                                                                     )
                                                                    }_"
                                                                    f"{pre_agg_names}_"
                                                                    f"{
                                                                     custom_dict_to_str(
                                                                     attack['name']
                                                                     )
                                                                    }_"
                                                                    f"lr_{lr}_"
                                                                    f"lrd_{lrd}_"
                                                                    f"wd_{wd}_"
                                                                    f"af_{af}_"
                                                                    f"am_{am}"
                                                                )
                                                                acc_path = os.path.join(
                                                                    path_to_results,
                                                                    file_name,
                                                                    f"val_accuracy_"
                                                                    f"tr_seed_"
                                                                    f"{
                                                                     run + training_seed
                                                                    }"
                                                                    f"_dd_seed_"
                                                                    f"{
                                                                     run_dd
                                                                     +
                                                                     data_distrib_seed
                                                                    }"
                                                                    f".txt",
                                                                )
                                                                tab_acc[
                                                                    i, run_dd, run
                                                                ] = genfromtxt(
                                                                    acc_path,
                                                                    delimiter=",",
                                                                )

                                                    tab_acc = tab_acc.reshape(
                                                        len(attacks),
                                                        nb_data_distribution_seeds
                                                        * nb_training_seeds,
                                                        nb_accuracies,
                                                    )

                                                    for i in range(len(attacks)):
                                                        avg_accuracy = np.mean(
                                                            tab_acc[i], axis=0
                                                        )
                                                        idx_max = np.argmax(
                                                            avg_accuracy
                                                        )
                                                        max_acc_config[
                                                            index_combination, i
                                                        ] = avg_accuracy[idx_max]
                                                        steps_max_reached[
                                                            index_combination, i
                                                        ] = idx_max * evaluation_delta

                                                    hyper_parameters[
                                                        index_combination
                                                    ] = [lr, lrd, wd]
                                                    index_combination += 1

                                        # Create path if needed
                                        if not os.path.exists(path_hyperparameters):
                                            try:
                                                os.makedirs(path_hyperparameters)
                                            except OSError as error:
                                                print(
                                                    f"Error creating directory: {error}"
                                                )

                                        max_minimum_idx = -1
                                        max_minimum_val = -1
                                        for i in range(num_combinations):
                                            current_min = np.min(max_acc_config[i])
                                            if current_min > max_minimum_val:
                                                max_minimum_idx = i
                                                max_minimum_val = current_min

                                        real_hyper_parameters[k] = hyper_parameters[
                                            max_minimum_idx
                                        ]
                                        real_steps[k] = steps_max_reached[
                                            max_minimum_idx
                                        ]

                                    # Save results to folder
                                    hyper_parameters_folder = os.path.join(
                                        path_hyperparameters, "hyperparameters"
                                    )
                                    steps_folder = os.path.join(
                                        path_hyperparameters, "better_step"
                                    )

                                    os.makedirs(hyper_parameters_folder, exist_ok=True)
                                    os.makedirs(steps_folder, exist_ok=True)

                                    for i, agg in enumerate(aggregators):
                                        # Save best hyperparameters
                                        file_name_hparams = (
                                            f"{dataset_name}_"
                                            f"{model_name}_"
                                            f"n_{nb_client}_"
                                            f"f_{nb_byzantine}_"
                                            f"d_{nb_decl}_"
                                            f"{custom_dict_to_str(
                                                    data_dist['name']
                                                )
                                            }_{distrib_parameter}_"
                                            f"{pre_agg_names}_{agg['name']}_"
                                            f"af_{af}_"
                                            f"am_{am}"
                                            f".txt"
                                        )
                                        np.savetxt(
                                            os.path.join(
                                                hyper_parameters_folder,
                                                file_name_hparams,
                                            ),
                                            real_hyper_parameters[i],
                                        )

                                        for j, attack in enumerate(attacks):
                                            file_name_steps = (
                                                f"{dataset_name}_"
                                                f"{model_name}_"
                                                f"n_{nb_client}_"
                                                f"f_{nb_byzantine}_"
                                                f"d_{nb_decl}_"
                                                f"{custom_dict_to_str(
                                                        data_dist['name']
                                                    )
                                                }_{distrib_parameter}_"
                                                f"{pre_agg_names}_{agg['name']}_"
                                                f"{custom_dict_to_str(attack['name'])}_"
                                                f"af_{af}_"
                                                f"am_{am}"
                                                f".txt"
                                            )
                                            step_val = np.array([real_steps[i, j]])
                                            np.savetxt(
                                                os.path.join(
                                                    steps_folder, file_name_steps
                                                ),
                                                step_val,
                                            )


colors = [
    (0, 0.4470, 0.7410),
    (0.8500, 0.3250, 0.0980),
    (0.4660, 0.6740, 0.1880),
    (120 / 255, 120 / 255, 120 / 255),
    (0.7, 0.2, 0.5),
]
tab_sign = ["-", "--", "-.", ":", "solid"]
markers = ["^", "s", "<", "o", "*"]


def test_accuracy_curve(
    path_to_results: str,
    path_to_plot: str,
    colors: list[tuple[float, float, float]] = colors,
    tab_sign: list[str] = tab_sign,
    markers: list[str] = markers,
) -> None:
    """
    Plot test accuracy curves for benchmark configurations.

    Load test accuracies across training and data-distribution seeds and
    generate one curve per attack using the selected hyperparameters.

    Parameters
    ----------
    path_to_results : str
        Path containing benchmark results and configuration files.
    path_to_plot : str
        Directory in which generated plots are stored.
    colors : list[tuple[float, float, float]], optional
        Colors used for the plotted attack curves.
    tab_sign : list[str], optional
        Line styles used for the plotted attack curves.
    markers : list[str], optional
        Markers used for the plotted attack curves.
    """
    try:
        with open(os.path.join(path_to_results, "config.json")) as file:
            data = json.load(file)
    except Exception as e:
        print(f"ERROR reading config.json: {e}")
        return

    try:
        os.makedirs(path_to_plot, exist_ok=True)
    except OSError as error:
        print(f"Error creating directory: {error}")

    path_to_hyperparameters = path_to_results + "/best_hyperparameters"

    # <-------------- Benchmark Config ------------->
    training_seed = data["benchmark_config"]["training_seed"]
    nb_training_seeds = data["benchmark_config"]["nb_training_seeds"]
    nb_clients = data["benchmark_config"]["nb_clients"]
    nb_byz = data["benchmark_config"]["f"]
    nb_declared = data["benchmark_config"].get("tolerated_f", None)
    data_distrib_seed = data["benchmark_config"]["data_distribution_seed"]
    nb_data_distribution_seeds = data["benchmark_config"]["nb_data_distribution_seeds"]
    data_distributions = data["benchmark_config"]["data_distribution"]
    nb_steps = data["benchmark_config"]["nb_steps"]

    # <-------------- Evaluation and Results ------------->
    evaluation_delta = data["evaluation_and_results"]["evaluation_delta"]

    # <-------------- Model Config ------------->
    model_name = data["model"]["name"]
    dataset_name = data["model"]["dataset_name"]
    lr_list = data["model"]["learning_rate"]
    lrd_list = data["model"]["learning_rate_decay"]
    wd_list = data["model"]["weight_decay"]

    # <-------------- Aggregators Config ------------->
    aggregators = data["aggregator"]

    if "pre_aggregators" not in data.keys():
        data["pre_aggregators"] = []

    for pre_agg in data["pre_aggregators"]:
        if "parameters" not in pre_agg.keys():
            pre_agg["parameters"] = {}
    pre_aggregators = data["pre_aggregators"]

    # <-------------- Attacks Config ------------->
    attacks = data["attack"]

    # <-------------- Training Algorithm Config ------------->
    af_list = data["benchmark_config"]["training_algorithm"]["parameters"][
        "aggreg_freq_scale"
    ]
    am_list = data["benchmark_config"]["training_algorithm"]["parameters"][
        "aggreg_mult_scale"
    ]

    # Ensure certain configurations are always lists
    nb_clients = ensure_list(nb_clients)
    nb_byz = ensure_list(nb_byz)
    nb_declared = ensure_list(nb_declared)
    data_distributions = ensure_list(data_distributions)
    aggregators = ensure_list(aggregators)

    # Pre-aggregators can be multiple or single dict; unify them
    if not pre_aggregators or isinstance(pre_aggregators[0], dict):
        pre_aggregators = [pre_aggregators]

    attacks = ensure_list(attacks)
    lr_list = ensure_list(lr_list)
    lrd_list = ensure_list(lrd_list)
    wd_list = ensure_list(wd_list)

    af_list = ensure_list(af_list)
    am_list = ensure_list(am_list)

    nb_accuracies = int(1 + math.ceil(nb_steps / evaluation_delta))

    for nb_client in nb_clients:
        for nb_byzantine in nb_byz:
            if nb_declared[0] is None:
                nb_declared_list = [nb_byzantine]
            else:
                nb_declared_list = nb_declared.copy()
                nb_declared_list = [
                    item for item in nb_declared_list if item >= nb_byzantine
                ]

            for nb_decl in nb_declared_list:
                for data_dist in data_distributions:
                    dist_parameter_list = data_dist["distribution_parameter"]
                    dist_parameter_list = ensure_list(dist_parameter_list)
                    for dist_parameter in dist_parameter_list:
                        for pre_agg in pre_aggregators:
                            pre_agg_list_names = [
                                one_pre_agg["name"] for one_pre_agg in pre_agg
                            ]
                            pre_agg_names = "_".join(pre_agg_list_names)
                            for agg in aggregators:
                                for af in af_list:
                                    for am in am_list:
                                        hyper_file_name = (
                                            f"{dataset_name}_"
                                            f"{model_name}_"
                                            f"n_{nb_client}_"
                                            f"f_{nb_byzantine}_"
                                            f"d_{nb_decl}_"
                                            f"{custom_dict_to_str(data_dist['name'])}_"
                                            f"{dist_parameter}_"
                                            f"{pre_agg_names}_{agg['name']}_"
                                            f"af_{af}_"
                                            f"am_{am}"
                                            f".txt"
                                        )

                                        full_path = os.path.join(
                                            path_to_hyperparameters,
                                            "hyperparameters",
                                            hyper_file_name,
                                        )

                                        if os.path.exists(full_path):
                                            hyperparameters = np.loadtxt(full_path)
                                            lr = hyperparameters[0]
                                            lrd = hyperparameters[1]
                                            wd = hyperparameters[2]
                                        else:
                                            lr = lr_list[0]
                                            lrd = lrd_list[0]
                                            wd = wd_list[0]

                                        tab_acc = np.zeros(
                                            (
                                                len(attacks),
                                                nb_data_distribution_seeds,
                                                nb_training_seeds,
                                                nb_accuracies,
                                            )
                                        )

                                        for i, attack in enumerate(attacks):
                                            for run_dd in range(
                                                nb_data_distribution_seeds
                                            ):
                                                for run in range(nb_training_seeds):
                                                    file_name = (
                                                        f"{dataset_name}_{model_name}_"
                                                        f"n_{nb_client}_"
                                                        f"f_{nb_byzantine}_"
                                                        f"d_{nb_decl}_"
                                                        f"{custom_dict_to_str(
                                                                data_dist['name']
                                                            )
                                                        }_{dist_parameter}_"
                                                        f"{custom_dict_to_str(
                                                                agg['name']
                                                            )
                                                        }_{pre_agg_names}_"
                                                        f"{custom_dict_to_str(
                                                                attack['name']
                                                            )
                                                        }_"
                                                        f"lr_{lr}_"
                                                        f"lrd_{lrd}_"
                                                        f"wd_{wd}_"
                                                        f"af_{af}_"
                                                        f"am_{am}"
                                                    )
                                                    acc_path = os.path.join(
                                                        path_to_results,
                                                        file_name,
                                                        f"test_accuracy_"
                                                        f"tr_seed_{run + training_seed}"
                                                        f"_dd_seed_"
                                                        f"{run_dd + data_distrib_seed}"
                                                        f".txt",
                                                    )
                                                    tab_acc[i, run_dd, run] = (
                                                        genfromtxt(
                                                            acc_path, delimiter=","
                                                        )
                                                    )

                                        tab_acc = tab_acc.reshape(
                                            len(attacks),
                                            nb_data_distribution_seeds
                                            * nb_training_seeds,
                                            nb_accuracies,
                                        )

                                        err = np.zeros((len(attacks), nb_accuracies))
                                        for i in range(len(err)):
                                            err[i] = (
                                                1.96 * np.std(tab_acc[i], axis=0)
                                            ) / math.sqrt(
                                                nb_training_seeds
                                                * nb_data_distribution_seeds
                                            )

                                        plt.rcParams.update({"font.size": 12})

                                        for i, attack in enumerate(attacks):
                                            attack = attack["name"]
                                            plt.plot(
                                                np.arange(nb_accuracies)
                                                * evaluation_delta,
                                                np.mean(tab_acc[i], axis=0),
                                                label=attack,
                                                color=colors[i],
                                                linestyle=tab_sign[i],
                                                marker=markers[i],
                                                markevery=1,
                                            )
                                            plt.fill_between(
                                                np.arange(nb_accuracies)
                                                * evaluation_delta,
                                                np.mean(tab_acc[i], axis=0) - err[i],
                                                np.mean(tab_acc[i], axis=0) + err[i],
                                                alpha=0.25,
                                            )

                                        plt.xlabel("Round")
                                        plt.ylabel("Accuracy")
                                        plt.xlim(
                                            0, (nb_accuracies - 1) * evaluation_delta
                                        )
                                        plt.ylim(0, 1)
                                        plt.grid()
                                        plt.legend()

                                        plot_name = (
                                            f"{dataset_name}_"
                                            f"{model_name}_"
                                            f"n_{nb_client}_"
                                            f"f_{nb_byzantine}_"
                                            f"d_{nb_decl}_"
                                            f"{custom_dict_to_str(data_dist['name'])}_"
                                            f"{dist_parameter}_"
                                            f"{custom_dict_to_str(agg['name'])}_"
                                            f"{pre_agg_names}_"
                                            f"lr_{lr}_"
                                            f"lrd_{lrd}_"
                                            f"wd_{wd}_"
                                            f"af_{af}_"
                                            f"am_{am}"
                                        )

                                        plt.savefig(
                                            path_to_plot + "/" + plot_name + "_plot.pdf"
                                        )
                                        plt.close()
