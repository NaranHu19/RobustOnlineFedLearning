import datetime
import json
import os
from typing import Any, TypeVar, cast

import numpy as np
import numpy.typing as npt
import torch

T = TypeVar("T")


class FileManager:
    """
    Manage files and directories for benchmark results.

    Create and manage directories used to store experiment results,
    model checkpoints, configuration files, losses, and accuracies.
    """

    def __init__(self, params: dict[str, Any]) -> None:
        self.files_path = (
            f"{params['result_path']}/"
            f"{params['dataset_name']}_{params['model_name']}_"
            f"n_{params['nb_clients']}_"
            f"f_{params['nb_byz']}_"
            f"d_{params['declared_nb_byz']}_"
            f"{params['data_distribution_name']}_"
            f"{params['distribution_parameter']}_"
            f"{params['aggregation_name']}_"
            f"{'_'.join(params['pre_aggregation_names'])}_"
            f"{params['attack_name']}_"
            f"lr_{params['learning_rate']}_"
            f"lrd_{params['learning_rate_decay']}_"
            f"wd_{params['weight_decay']}_"
            f"af_{params['aggreg_freq_scale']}_"
            f"am_{params['aggreg_mult_scale']}/"
        )
        os.makedirs(self.files_path, exist_ok=True)

        with open(os.path.join(self.files_path, "day.txt"), "w") as file:
            file.write(datetime.date.today().strftime("%d_%m_%y"))

        self.models_path = (
            f"{params['model_path']}/"
            f"{params['dataset_name']}_{params['model_name']}_"
            f"n_{params['nb_clients']}_"
            f"f_{params['nb_byz']}_"
            f"d_{params['declared_nb_byz']}_"
            f"{params['data_distribution_name']}_"
            f"{params['distribution_parameter']}_"
            f"{params['aggregation_name']}_"
            f"{'_'.join(params['pre_aggregation_names'])}_"
            f"{params['attack_name']}_"
            f"lr_{params['learning_rate']}_"
            f"lrd_{params['learning_rate_decay']}_"
            f"wd_{params['weight_decay']}_"
            f"af_{params['aggreg_freq_scale']}_"
            f"am_{params['aggreg_mult_scale']}/"
        )
        os.makedirs(self.models_path, exist_ok=True)

        with open(os.path.join(self.models_path, "day.txt"), "w") as file:
            file.write(datetime.date.today().strftime("%d_%m_%y"))

    def set_experiment_path(self, path: str) -> None:
        """
        Set the base path for experiment files.

        Parameters
        ----------
        path : str
            Path to the directory containing the experiment files.
        """
        self.files_path = path

    def get_experiment_path(self) -> str:
        """
        Return the current experiment path.

        Returns
        -------
        str
            Path to the directory containing the experiment files.
        """
        return self.files_path

    def save_config_dict(self, dict_to_save: dict[str, Any]) -> None:
        """
        Save a configuration dictionary as a JSON file.

        Parameters
        ----------
        dict_to_save : dict[str, Any]
            Configuration dictionary to save.
        """
        config_path = os.path.join(self.files_path, "config.json")
        with open(config_path, "w") as json_file:
            json.dump(
                dict_to_save,
                json_file,
                indent=4,
                separators=(",", ": "),
            )

    def write_array_in_file(
        self,
        array: npt.NDArray[np.float64],
        file_name: str,
    ) -> None:
        """
        Write an array to a file.

        Parameters
        ----------
        array : numpy.ndarray
            Array containing the values to write.
        file_name : str
            Name of the output file.
        """
        file_path = os.path.join(self.files_path, file_name)
        np.savetxt(file_path, [array], fmt="%.4f", delimiter=",")

    def save_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        training_seed: int,
        data_dist_seed: int,
        step: int,
    ) -> None:
        """
        Save a model state dictionary.

        Store the model state in a directory identified by the training
        and data-distribution seeds.

        Parameters
        ----------
        state_dict : dict[str, torch.Tensor]
            State dictionary containing the model parameters.
        training_seed : int
            Seed used for training.
        data_dist_seed : int
            Seed used for the data distribution.
        step : int
            Training step associated with the saved model.
        """
        model_dir = os.path.join(
            self.models_path, f"models_tr_seed_{training_seed}_dd_seed_{data_dist_seed}"
        )
        os.makedirs(model_dir, exist_ok=True)

        file_path = os.path.join(model_dir, f"model_step_{step}.pth")
        torch.save(state_dict, file_path)

    def save_loss(
        self,
        loss_array: list[float],
        training_seed: int,
        data_dist_seed: int,
        client_id: int,
    ) -> None:
        """
        Save training losses for a client.

        Parameters
        ----------
        loss_array : list[float]
            Training losses to save.
        training_seed : int
            Seed used for training.
        data_dist_seed : int
            Seed used for the data distribution.
        client_id : int
            Identifier of the client.
        """
        loss_dir = os.path.join(
            self.files_path,
            f"train_loss_tr_seed_{training_seed}_dd_seed_{data_dist_seed}",
        )
        os.makedirs(loss_dir, exist_ok=True)

        file_path = os.path.join(loss_dir, f"loss_client_{client_id}.txt")
        np.savetxt(file_path, loss_array, fmt="%.6f", delimiter=",")

    def save_accuracy(
        self,
        acc_array: list[float],
        training_seed: int,
        data_dist_seed: int,
        client_id: int,
    ) -> None:
        """
        Save training accuracies for a client.

        Parameters
        ----------
        acc_array : list[float]
            Training accuracies to save.
        training_seed : int
            Seed used for training.
        data_dist_seed : int
            Seed used for the data distribution.
        client_id : int
            Identifier of the client.
        """
        acc_dir = os.path.join(
            self.files_path,
            f"train_accuracy_tr_seed_{training_seed}_dd_seed_{data_dist_seed}",
        )
        os.makedirs(acc_dir, exist_ok=True)

        file_path = os.path.join(acc_dir, f"accuracy_client_{client_id}.txt")
        np.savetxt(file_path, acc_array, fmt="%.4f", delimiter=",")


class ParamsManager:
    """
    Manage benchmark configuration parameters.

    Store parameters read from the JSON configuration and provide
    accessors that return configured values or suitable defaults.

    Parameters
    ----------
    params : dict[str, Any]
        Dictionary containing the benchmark configuration.
    """

    def __init__(self, params: dict[str, Any]) -> None:
        self.data = params

    def _parameter_to_use(
        self,
        default: T,
        read: Any,
    ) -> T:
        if read is None:
            return default
        else:
            return cast(T, read)

    def _read_object(self, path: list[str]) -> Any:
        """
        Read an object from the nested configuration dictionary.

        Traverse ``self.data`` using the sequence of keys provided in
        ``path``. Return ``None`` if any key does not exist.

        Parameters
        ----------
        path : list[str]
            Sequence of keys identifying the object to retrieve.

        Returns
        -------
        Any
            Retrieved configuration value, or ``None`` if the path does
            not exist.
        """
        obj = self.data
        for p in path:
            if isinstance(obj, dict) and p in obj.keys():
                obj = obj[p]
            else:
                return None
        return obj

    def get_data(self) -> dict[str, Any]:
        """
        Return the complete benchmark configuration.

        Construct the configuration dictionary using the values returned
        by the parameter accessors.

        Returns
        -------
        dict[str, Any]
            Complete benchmark configuration.
        """
        return {
            "benchmark_config": {
                "device": self.get_device(),
                "training_seed": self.get_training_seed(),
                "nb_training_seeds": self.get_nb_training_seeds(),
                "nb_clients": self.get_nb_clients(),
                "nb_honest_clients": self.get_nb_honest_clients(),
                "f": self.get_f(),
                "tolerated_f": self.get_tolerated_f(),
                "size_train_set": self.get_size_train_set(),
                "data_distribution_seed": self.get_data_distribution_seed(),
                "nb_data_distribution_seeds": self.get_nb_data_distribution_seeds(),
                "data_distribution": self.get_data_distribution(),
                "training_algorithm": self.get_training_algorithm(),
                "nb_steps": self.get_nb_steps(),
            },
            "model": {
                "name": self.get_model_name(),
                "dataset_name": self.get_dataset_name(),
                "nb_labels": self.get_nb_labels(),
                "loss": self.get_loss_name(),
                "learning_rate": self.get_learning_rate(),
                "learning_rate_decay": self.get_learning_rate_decay(),
                "milestones": self.get_milestones(),
                "weight_decay": self.get_weight_decay(),
            },
            "aggregator": self.get_aggregator_info(),
            "pre_aggregators": self.get_preaggregators(),
            "honest_clients": {"batch_size": self.get_honest_clients_batch_size()},
            "attack": self.get_attack_info(),
            "evaluation_and_results": {
                "evaluation_delta": self.get_evaluation_delta(),
                "batch_size_evaluation": self.get_batch_size_evaluation(),
                "evaluate_on_test": self.get_evaluate_on_test(),
                "store_per_client_metrics": self.get_store_per_client_metrics(),
                "store_models": self.get_store_models(),
                "data_folder": self.get_data_folder(),
                "results_directory": self.get_results_directory(),
                "models_directory": self.get_models_directory(),
            },
        }

    # ----------------------------------------------------------------------
    #  Benchmark Config
    # ----------------------------------------------------------------------

    def get_device(self) -> str:
        """
        Return the computation device.

        Returns
        -------
        str
            Computation device.
        """
        default = "cpu"
        path = ["benchmark_config", "device"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_training_seed(self) -> int:
        """
        Return the training seed.

        Returns
        -------
        int
            Training seed.
        """
        default = 0
        path = ["benchmark_config", "training_seed"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_nb_training_seeds(self) -> int:
        """
        Return the number of training seeds.

        Returns
        -------
        int
            Number of training seeds.
        """
        default = 1
        path = ["benchmark_config", "nb_training_seeds"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_nb_clients(self) -> int:
        """
        Return the number of clients.

        Returns
        -------
        int
            Number of clients.
        """
        default = 1
        path = ["benchmark_config", "nb_clients"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_nb_honest_clients(self) -> int:
        """
        Return the number of honest clients.

        Returns
        -------
        int
            Number of honest clients.
        """
        default = 0
        path = ["benchmark_config", "nb_honest_clients"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_f(self) -> int:
        """
        Return the number of Byzantine clients.

        Returns
        -------
        int
            Number of Byzantine clients.
        """
        default = 0
        path = ["benchmark_config", "f"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_tolerated_f(self) -> int:
        """
        Return the tolerated number of Byzantine clients.

        Returns
        -------
        int
            Tolerated number of Byzantine clients.
        """
        default = self.get_f()
        path = ["benchmark_config", "tolerated_f"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_size_train_set(self) -> float:
        """
        Return the training-set proportion.

        Returns
        -------
        float
            Proportion of data used for training.
        """
        default = 0.8
        path = ["benchmark_config", "size_train_set"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_data_distribution_seed(self) -> int:
        """
        Return the data-distribution seed.

        Returns
        -------
        int
            Data-distribution seed.
        """
        default = 0
        path = ["benchmark_config", "data_distribution_seed"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_nb_data_distribution_seeds(self) -> int:
        """
        Return the number of data-distribution seeds.

        Returns
        -------
        int
            Number of data-distribution seeds.
        """
        default = 1
        path = ["benchmark_config", "nb_data_distribution_seeds"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_data_distribution(self) -> dict[str, str | float]:
        """
        Return the data-distribution configuration.

        Returns
        -------
        dict[str, str | float]
            Data-distribution name and associated parameters.
        """
        default: dict[str, str | float] = {
            "name": "iid",
            "distribution_parameter": 1.0,
        }
        path = ["benchmark_config", "data_distribution"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_name_data_distribution(self) -> str:
        """
        Return the data-distribution name.

        Returns
        -------
        str
            Name of the data distribution.
        """
        default = "iid"
        path = ["benchmark_config", "data_distribution", "name"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_parameter_data_distribution(self) -> float:
        """
        Return the data-distribution parameter.

        Returns
        -------
        float
            Parameter controlling the data distribution.
        """
        default = 1.0
        path = ["benchmark_config", "data_distribution", "distribution_parameter"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_training_algorithm(
        self,
    ) -> dict[str, str | dict[str, float]]:
        """
        Return the training-algorithm configuration.

        Returns
        -------
        dict[str, str | dict[str, float]]
            Training-algorithm name and parameters.
        """
        default: dict[str, str | dict[str, float]] = {
            "name": "RobustOnlineFL",
            "parameters": {},
        }
        path = ["benchmark_config", "training_algorithm"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_training_algorithm_name(self) -> str:
        """
        Return the training-algorithm name.

        Returns
        -------
        str
            Name of the training algorithm.
        """
        default = "RobustOnlineFL"
        path = ["benchmark_config", "training_algorithm", "name"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_training_algorithm_parameters(self) -> dict[str, float]:
        """
        Return the training-algorithm parameters.

        Returns
        -------
        dict[str, float]
            Parameters of the training algorithm.
        """
        default: dict[str, float] = {}
        path = ["benchmark_config", "training_algorithm", "parameters"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_nb_steps(self) -> int:
        """
        Return the number of training steps.

        Returns
        -------
        int
            Number of training steps.
        """
        default = 1000
        path = ["benchmark_config", "nb_steps"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    # ----------------------------------------------------------------------
    #  Model
    # ----------------------------------------------------------------------
    def get_model_name(self) -> str:
        """
        Return the model name.

        Returns
        -------
        str
            Name of the model.
        """
        default = "cnn_mnist"
        path = ["model", "name"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_dataset_name(self) -> str:
        """
        Return the dataset name.

        Returns
        -------
        str
            Name of the dataset.
        """
        default = "mnist"
        path = ["model", "dataset_name"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_nb_labels(self) -> int:
        """
        Return the number of labels.

        Returns
        -------
        int
            Number of labels in the dataset.
        """
        default = 10
        path = ["model", "nb_labels"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_loss_name(self) -> str:
        """
        Return the loss-function name.

        Returns
        -------
        str
            Name of the loss function.
        """
        default = "NLLLoss"
        path = ["model", "loss"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_optimizer_name(self) -> str:
        """
        Return the optimizer name.

        Returns
        -------
        str
            Name of the optimizer.
        """
        default = "SGD"
        path = ["model", "optimizer_name"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_learning_rate(self) -> float:
        """
        Return the learning rate.

        Returns
        -------
        float
            Learning rate.
        """
        default = 0.1
        path = ["model", "learning_rate"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_learning_rate_decay(self) -> float:
        """
        Return the learning-rate decay.

        Returns
        -------
        float
            Learning-rate decay factor.
        """
        default = 1.0
        path = ["model", "learning_rate_decay"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_milestones(self) -> list[int]:
        """
        Return the learning-rate milestones.

        Returns
        -------
        list[int]
            Training steps at which the learning rate is adjusted.
        """
        default: list[int] = []
        path = ["model", "milestones"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_weight_decay(self) -> float:
        """
        Return the weight decay.

        Returns
        -------
        float
            Weight-decay coefficient.
        """
        default = 1e-4
        path = ["model", "weight_decay"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    # ----------------------------------------------------------------------
    #  Aggregator
    # ----------------------------------------------------------------------
    def get_aggregator_info(
        self,
    ) -> dict[str, str | dict[str, float]]:
        """
        Return the aggregator configuration.

        Returns
        -------
        dict[str, str | dict[str, float]]
            Aggregator name and parameters.
        """
        default: dict[str, str | dict[str, float]] = {
            "name": "Average",
            "parameters": {},
        }
        path = ["aggregator"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_aggregator_name(self) -> str:
        """
        Return the aggregator name.

        Returns
        -------
        str
            Name of the aggregator.
        """
        default = "average"
        path = ["aggregator", "name"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_aggregator_parameters(self) -> dict[str, float]:
        """
        Return the aggregator parameters.

        Returns
        -------
        dict[str, float]
            Parameters of the aggregator.
        """
        default: dict[str, float] = {}
        path = ["aggregator", "parameters"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    # ----------------------------------------------------------------------
    #  Pre-Aggregators
    # ----------------------------------------------------------------------
    def get_preaggregators(
        self,
    ) -> list[dict[str, dict[str, float]]]:
        """
        Return the pre-aggregator configurations.

        Returns
        -------
        list[dict[str, dict[str, float]]]
            Configurations of the pre-aggregators.
        """
        default: list[dict[str, dict[str, float]]] = []
        path = ["pre_aggregators"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    # ----------------------------------------------------------------------
    #  Honest Nodes
    # ----------------------------------------------------------------------
    def get_honest_clients_batch_size(self) -> int:
        """
        Return the honest-client batch size.

        Returns
        -------
        int
            Batch size used by honest clients.
        """
        default = 32
        path = ["honest_clients", "batch_size"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    # ----------------------------------------------------------------------
    #  Attack
    # ----------------------------------------------------------------------

    def get_attack_info(
        self,
    ) -> dict[str, str | dict[str, float]]:
        """
        Return the attack configuration.

        Returns
        -------
        dict[str, str | dict[str, float]]
            Attack name and parameters.
        """
        default: dict[str, str | dict[str, float]] = {
            "name": "NoAttack",
            "parameters": {},
        }
        path = ["attack"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_attack_name(self) -> str:
        """
        Return the attack name.

        Returns
        -------
        str
            Name of the attack.
        """
        default = "NoAttack"
        path = ["attack", "name"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_attack_parameters(self) -> dict[str, Any]:
        """
        Return the attack parameters.

        Returns
        -------
        dict[str, Any]
            Parameters of the attack.
        """
        default: dict[str, float] = {}
        path = ["attack", "parameters"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    # ----------------------------------------------------------------------
    #  Evaluation and Results Accessors
    # ----------------------------------------------------------------------
    def get_evaluation_delta(self) -> int:
        """
        Return the evaluation interval.

        Returns
        -------
        int
            Evaluation steps.
        """
        default = 50
        path = ["evaluation_and_results", "evaluation_delta"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_batch_size_evaluation(self) -> int:
        """
        Return the evaluation batch size.

        Returns
        -------
        int
            Batch size used during evaluation.
        """
        default = 128
        path = ["evaluation_and_results", "batch_size_evaluation"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_evaluate_on_test(self) -> bool:
        """
        Return whether evaluation uses the test set.

        Returns
        -------
        bool
            Whether to evaluate the model on the test set.
        """
        default = True
        path = ["evaluation_and_results", "evaluate_on_test"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_store_per_client_metrics(self) -> bool:
        """
        Return whether per-client metrics are stored.

        Returns
        -------
        bool
            Whether metrics are stored separately for each client.
        """
        default = True
        path = ["evaluation_and_results", "store_per_client_metrics"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_store_models(self) -> bool:
        """
        Return whether trained models are stored.

        Returns
        -------
        bool
            Whether model checkpoints are stored.
        """
        default = False
        path = ["evaluation_and_results", "store_models"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_data_folder(self) -> str:
        """
        Return the data directory.

        Returns
        -------
        str
            Path to the data directory.
        """
        default = "./data"
        path = ["evaluation_and_results", "data_folder"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_results_directory(self) -> str:
        """
        Return the results directory.

        Returns
        -------
        str
            Path to the results directory.
        """
        default = "./results"
        path = ["evaluation_and_results", "results_directory"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_models_directory(self) -> str:
        """
        Return the models directory.

        Returns
        -------
        str
            Path to the models directory.
        """
        default = "./models"
        path = ["evaluation_and_results", "models_directory"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
