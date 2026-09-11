import json
import math
from pathlib import Path
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


def load_acc_and_times(
    accuracy_path: str | Path,
    evaluation_times_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load accuracy values and their corresponding global time steps.

    Parameters
    ----------
    accuracy_path : str or Path
        Path to the accuracy file.
    evaluation_times_path : str or Path
        Path to the evaluation-time file.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Accuracy values and corresponding global local-update times.

    Raises
    ------
    ValueError
        If the number of accuracy values differs from the number of
        evaluation times.
    """
    accuracies = np.atleast_1d(
        genfromtxt(
            accuracy_path,
            delimiter=",",
        )
    )

    evaluation_times = np.atleast_1d(
        genfromtxt(
            evaluation_times_path,
            delimiter=",",
        )
    ).astype(int)

    if len(accuracies) != len(evaluation_times):
        raise ValueError(
            "Accuracy and evaluation-time files have different lengths: "
            f"{accuracy_path} contains {len(accuracies)} values while "
            f"{evaluation_times_path} contains {len(evaluation_times)}."
        )

    return accuracies, evaluation_times


def find_best_hyperparameters(path_to_results: str | Path) -> None:
    """
    Find the best hyperparameters across different attacks.

    Find the learning rate, learning rate decay, and weight decay that
    maximize the minimum accuracy across different attacks. Accuracy
    measurements are associated with the actual global local-update
    times at which evaluation was performed.

    Parameters
    ----------
    path_to_results : str or Path
        Path to the directory containing the benchmark configuration
        and result files.
    """
    path_to_results = Path(path_to_results)

    try:
        with (path_to_results / "config.json").open() as file:
            data = json.load(file)
    except Exception as e:
        print(f"ERROR reading config.json: {e}")
        return

    path_hyperparameters = path_to_results / "best_hyperparameters"

    # <-------------- Benchmark Config ------------->
    training_seed = data["benchmark_config"]["training_seed"]
    nb_training_seeds = data["benchmark_config"]["nb_training_seeds"]
    nb_clients = data["benchmark_config"]["nb_clients"]
    nb_byz = data["benchmark_config"]["f"]
    nb_declared = data["benchmark_config"].get("tolerated_f", None)
    data_distrib_seed = data["benchmark_config"]["data_distribution_seed"]
    nb_data_distribution_seeds = data["benchmark_config"]["nb_data_distribution_seeds"]
    data_distributions = data["benchmark_config"]["data_distribution"]

    # <-------------- Model Config ------------->
    model_name = data["model"]["name"]
    dataset_name = data["model"]["dataset_name"]
    lr_list = data["model"]["learning_rate"]
    lrd_list = data["model"]["learning_rate_decay"]
    wd_list = data["model"]["weight_decay"]

    # <-------------- Aggregators Config ------------->
    aggregators = data["aggregator"]

    if "pre_aggregators" not in data:
        data["pre_aggregators"] = []

    for pre_agg in data["pre_aggregators"]:
        if "parameters" not in pre_agg:
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

    if not pre_aggregators or isinstance(pre_aggregators[0], dict):
        pre_aggregators = [pre_aggregators]

    attacks = ensure_list(attacks)
    lr_list = ensure_list(lr_list)
    lrd_list = ensure_list(lrd_list)
    wd_list = ensure_list(wd_list)

    af_list = ensure_list(af_list)
    am_list = ensure_list(am_list)

    # Main nested loops to explore configurations
    for nb_client in nb_clients:
        for nb_byzantine in nb_byz:
            if nb_declared[0] is None:
                nb_declared_list = [nb_byzantine]
            else:
                nb_declared_list = [
                    item for item in nb_declared.copy() if item >= nb_byzantine
                ]

            for nb_decl in nb_declared_list:
                for data_dist in data_distributions:
                    dist_parameter_list = ensure_list(
                        data_dist["distribution_parameter"]
                    )

                    for dist_parameter in dist_parameter_list:
                        for af in af_list:
                            for am in am_list:
                                for pre_agg in pre_aggregators:
                                    pre_agg_names_list = [p["name"] for p in pre_agg]
                                    pre_agg_names = "_".join(pre_agg_names_list)

                                    real_hyper_parameters = np.zeros(
                                        (len(aggregators), 3)
                                    )
                                    real_steps = np.zeros(
                                        (
                                            len(aggregators),
                                            len(attacks),
                                        )
                                    )

                                    for k, agg in enumerate(aggregators):
                                        num_combinations = (
                                            len(lr_list) * len(lrd_list) * len(wd_list)
                                        )

                                        max_acc_config = np.zeros(
                                            (
                                                num_combinations,
                                                len(attacks),
                                            )
                                        )
                                        hyper_parameters = np.zeros(
                                            (num_combinations, 3)
                                        )
                                        steps_max_reached = np.zeros(
                                            (
                                                num_combinations,
                                                len(attacks),
                                            )
                                        )

                                        index_combination = 0

                                        for lr in lr_list:
                                            for lrd in lrd_list:
                                                for wd in wd_list:
                                                    for i, attack in enumerate(attacks):
                                                        run_accuracies = []
                                                        run_times = []

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
                                                                    )}_"
                                                                    f"{dist_parameter}_"
                                                                    f"{
                                                                    custom_dict_to_str(
                                                                    agg['name']
                                                                    )}_"
                                                                    f"{pre_agg_names}_"
                                                                    f"{
                                                                    custom_dict_to_str(
                                                                    attack['name']
                                                                    )}_"
                                                                    f"lr_{lr}_"
                                                                    f"lrd_{lrd}_"
                                                                    f"wd_{wd}_"
                                                                    f"af_{af}_"
                                                                    f"am_{am}"
                                                                )

                                                                curr_train_seed = (
                                                                    run + training_seed
                                                                )
                                                                curr_dd_seed = (
                                                                    run_dd
                                                                    + data_distrib_seed
                                                                )

                                                                acc_path = (
                                                                    path_to_results
                                                                    / file_name
                                                                    / (
                                                                        "val_accuracy_"
                                                                        "tr_seed_"
                                                                        f"{
                                                                        curr_train_seed
                                                                        }_"
                                                                        f"dd_seed_{
                                                                        curr_dd_seed
                                                                        }.txt"
                                                                    )
                                                                )

                                                                times_path = (
                                                                    path_to_results
                                                                    / file_name
                                                                    / "evaluation_times"
                                                                    ".txt"
                                                                )

                                                                (
                                                                    accuracies,
                                                                    evaluation_times,
                                                                ) = load_acc_and_times(
                                                                    acc_path,
                                                                    times_path,
                                                                )

                                                                run_accuracies.append(
                                                                    accuracies
                                                                )
                                                                run_times.append(
                                                                    evaluation_times
                                                                )

                                                        reference_times = run_times[0]

                                                        for times in run_times[1:]:
                                                            if not np.array_equal(
                                                                times,
                                                                reference_times,
                                                            ):
                                                                raise ValueError(
                                                                    "Evaluation times "
                                                                    "differ between "
                                                                    "runs of the same "
                                                                    "configuration."
                                                                )

                                                        accuracy_array = np.stack(
                                                            run_accuracies,
                                                            axis=0,
                                                        )

                                                        avg_accuracy = np.mean(
                                                            accuracy_array,
                                                            axis=0,
                                                        )

                                                        idx_max = int(
                                                            np.argmax(avg_accuracy)
                                                        )

                                                        max_acc_config[
                                                            index_combination,
                                                            i,
                                                        ] = avg_accuracy[idx_max]

                                                        steps_max_reached[
                                                            index_combination,
                                                            i,
                                                        ] = reference_times[idx_max]

                                                    hyper_parameters[
                                                        index_combination
                                                    ] = [
                                                        lr,
                                                        lrd,
                                                        wd,
                                                    ]

                                                    index_combination += 1

                                        path_hyperparameters.mkdir(
                                            parents=True,
                                            exist_ok=True,
                                        )

                                        max_minimum_idx = -1
                                        max_minimum_val = -1.0

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

                                    hyper_parameters_folder = (
                                        path_hyperparameters / "hyperparameters"
                                    )
                                    steps_folder = path_hyperparameters / "better_step"

                                    hyper_parameters_folder.mkdir(
                                        parents=True,
                                        exist_ok=True,
                                    )
                                    steps_folder.mkdir(
                                        parents=True,
                                        exist_ok=True,
                                    )

                                    for i, agg in enumerate(aggregators):
                                        file_name_hparams = (
                                            f"{dataset_name}_"
                                            f"{model_name}_"
                                            f"n_{nb_client}_"
                                            f"f_{nb_byzantine}_"
                                            f"d_{nb_decl}_"
                                            f"{custom_dict_to_str(data_dist['name'])}_"
                                            f"{dist_parameter}_"
                                            f"{pre_agg_names}_"
                                            f"{agg['name']}_"
                                            f"af_{af}_"
                                            f"am_{am}.txt"
                                        )

                                        np.savetxt(
                                            hyper_parameters_folder / file_name_hparams,
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
                                                )}_"
                                                f"{dist_parameter}_"
                                                f"{pre_agg_names}_"
                                                f"{agg['name']}_"
                                                f"{custom_dict_to_str(
                                                    attack['name']
                                                )}_"
                                                f"af_{af}_"
                                                f"am_{am}.txt"
                                            )

                                            step_val = np.array([real_steps[i, j]])

                                            np.savetxt(
                                                steps_folder / file_name_steps,
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
    path_to_results: str | Path,
    path_to_plot: str | Path,
    colors: list[tuple[float, float, float]] = colors,
    tab_sign: list[str] = tab_sign,
    markers: list[str] = markers,
) -> None:
    """
    Plot test accuracy curves for benchmark configurations.

    Generate one plot per attack. Each curve corresponds to a different
    aggregation rule and uses the best hyperparameters selected for that
    aggregator.

    Parameters
    ----------
    path_to_results : str or Path
        Path containing benchmark results and configuration files.
    path_to_plot : str or Path
        Directory in which generated plots are stored.
    colors : list[tuple[float, float, float]], optional
        Colors used for the plotted aggregator curves.
    tab_sign : list[str], optional
        Line styles used for the plotted aggregator curves.
    markers : list[str], optional
        Markers used for the plotted aggregator curves.
    """
    path_to_results = Path(path_to_results)
    path_to_plot = Path(path_to_plot)

    try:
        with (path_to_results / "config.json").open() as file:
            data = json.load(file)
    except Exception as e:
        print(f"ERROR reading config.json: {e}")
        return

    try:
        path_to_plot.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        print(f"Error creating directory: {error}")

    path_to_hyperparameters = path_to_results / "best_hyperparameters"

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

    # <-------------- Model Config ------------->
    model_name = data["model"]["name"]
    dataset_name = data["model"]["dataset_name"]
    lr_list = data["model"]["learning_rate"]
    lrd_list = data["model"]["learning_rate_decay"]
    wd_list = data["model"]["weight_decay"]

    # <-------------- Aggregators Config ------------->
    aggregators = data["aggregator"]

    if "pre_aggregators" not in data:
        data["pre_aggregators"] = []

    for pre_agg in data["pre_aggregators"]:
        if "parameters" not in pre_agg:
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

    if not pre_aggregators or isinstance(pre_aggregators[0], dict):
        pre_aggregators = [pre_aggregators]

    attacks = ensure_list(attacks)
    lr_list = ensure_list(lr_list)
    lrd_list = ensure_list(lrd_list)
    wd_list = ensure_list(wd_list)

    af_list = ensure_list(af_list)
    am_list = ensure_list(am_list)

    if len(aggregators) > len(colors):
        raise ValueError("Not enough colors for the number of aggregators.")

    if len(aggregators) > len(tab_sign):
        raise ValueError("Not enough line styles for the number of aggregators.")

    if len(aggregators) > len(markers):
        raise ValueError("Not enough markers for the number of aggregators.")

    for nb_client in nb_clients:
        for nb_byzantine in nb_byz:
            if nb_declared[0] is None:
                nb_declared_list = [nb_byzantine]
            else:
                nb_declared_list = [
                    item for item in nb_declared.copy() if item >= nb_byzantine
                ]

            for nb_decl in nb_declared_list:
                for data_dist in data_distributions:
                    dist_parameter_list = ensure_list(
                        data_dist["distribution_parameter"]
                    )

                    for dist_parameter in dist_parameter_list:
                        for pre_agg in pre_aggregators:
                            pre_agg_list_names = [
                                one_pre_agg["name"] for one_pre_agg in pre_agg
                            ]
                            pre_agg_names = "_".join(pre_agg_list_names)

                            for af in af_list:
                                for am in am_list:
                                    for attack in attacks:
                                        plt.rcParams.update({"font.size": 12})

                                        attack_name = attack["name"]

                                        for i, agg in enumerate(aggregators):
                                            hyper_file_name = (
                                                f"{dataset_name}_"
                                                f"{model_name}_"
                                                f"n_{nb_client}_"
                                                f"f_{nb_byzantine}_"
                                                f"d_{nb_decl}_"
                                                f"{custom_dict_to_str(
                                                    data_dist['name']
                                                )}_"
                                                f"{dist_parameter}_"
                                                f"{pre_agg_names}_"
                                                f"{agg['name']}_"
                                                f"af_{af}_"
                                                f"am_{am}.txt"
                                            )

                                            full_path = (
                                                path_to_hyperparameters
                                                / "hyperparameters"
                                                / hyper_file_name
                                            )

                                            if full_path.exists():
                                                hyperparameters = np.atleast_1d(
                                                    np.loadtxt(full_path)
                                                )
                                                lr = hyperparameters[0]
                                                lrd = hyperparameters[1]
                                                wd = hyperparameters[2]
                                            else:
                                                lr = lr_list[0]
                                                lrd = lrd_list[0]
                                                wd = wd_list[0]

                                            run_accuracies = []
                                            reference_times = None

                                            for run_dd in range(
                                                nb_data_distribution_seeds
                                            ):
                                                for run in range(nb_training_seeds):
                                                    file_name = (
                                                        f"{dataset_name}_"
                                                        f"{model_name}_"
                                                        f"n_{nb_client}_"
                                                        f"f_{nb_byzantine}_"
                                                        f"d_{nb_decl}_"
                                                        f"{custom_dict_to_str(
                                                            data_dist['name']
                                                        )}_"
                                                        f"{dist_parameter}_"
                                                        f"{custom_dict_to_str(
                                                            agg['name']
                                                        )}_"
                                                        f"{pre_agg_names}_"
                                                        f"{custom_dict_to_str(
                                                            attack_name
                                                        )}_"
                                                        f"lr_{lr}_"
                                                        f"lrd_{lrd}_"
                                                        f"wd_{wd}_"
                                                        f"af_{af}_"
                                                        f"am_{am}"
                                                    )

                                                    current_training_seed = (
                                                        run + training_seed
                                                    )
                                                    current_dd_seed = (
                                                        run_dd + data_distrib_seed
                                                    )

                                                    acc_path = (
                                                        path_to_results
                                                        / file_name
                                                        / (
                                                            "test_accuracy_"
                                                            f"tr_seed_{
                                                                current_training_seed
                                                            }_"
                                                            f"dd_seed_{
                                                                current_dd_seed
                                                            }.txt"
                                                        )
                                                    )

                                                    times_path = (
                                                        path_to_results
                                                        / file_name
                                                        / "evaluation_times.txt"
                                                    )

                                                    (
                                                        accuracies,
                                                        evaluation_times,
                                                    ) = load_acc_and_times(
                                                        acc_path,
                                                        times_path,
                                                    )

                                                    if reference_times is None:
                                                        reference_times = (
                                                            evaluation_times
                                                        )
                                                    elif not np.array_equal(
                                                        evaluation_times,
                                                        reference_times,
                                                    ):
                                                        raise ValueError(
                                                            "Evaluation times "
                                                            "differ between runs "
                                                            "of the same "
                                                            "configuration."
                                                        )

                                                    run_accuracies.append(accuracies)

                                            if reference_times is None:
                                                raise RuntimeError(
                                                    "No evaluation times "
                                                    "were loaded."
                                                )

                                            accuracy_array = np.stack(
                                                run_accuracies,
                                                axis=0,
                                            )

                                            mean_accuracy = np.mean(
                                                accuracy_array,
                                                axis=0,
                                            )

                                            err = (
                                                1.96
                                                * np.std(
                                                    accuracy_array,
                                                    axis=0,
                                                )
                                                / math.sqrt(len(run_accuracies))
                                            )

                                            aggregator_name = agg["name"]

                                            plt.plot(
                                                reference_times,
                                                mean_accuracy,
                                                label=aggregator_name,
                                                color=colors[i],
                                                linestyle=tab_sign[i],
                                                marker=markers[i],
                                                markevery=1,
                                            )

                                            plt.fill_between(
                                                reference_times,
                                                mean_accuracy - err,
                                                mean_accuracy + err,
                                                alpha=0.25,
                                            )

                                        plt.xlabel("Time step $t$")
                                        plt.ylabel("Accuracy")
                                        plt.xlim(0, nb_steps)
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
                                            f"{custom_dict_to_str(attack_name)}_"
                                            f"{pre_agg_names}_"
                                            f"af_{af}_"
                                            f"am_{am}"
                                        )

                                        plt.savefig(
                                            path_to_plot / f"{plot_name}_plot.pdf"
                                        )

                                        plt.close()
