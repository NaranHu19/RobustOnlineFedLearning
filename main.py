import argparse

from benchmark.benchmark import run_benchmark
from benchmark.evaluate_results import (
    test_accuracy_curve,
)


def run_learning(
    config_file: str,
    n_jobs: int,
    gpus: list[int] | None,
) -> None:
    """
    Run benchmark training experiments.

    Parameters
    ----------
    config_file : str
        Path to the benchmark configuration file.
    n_jobs : int
        Number of training jobs to run in parallel.
    gpus : list[int]
        GPU identifiers available for training jobs.
    """
    run_benchmark(config_file, n_jobs, gpus)


def run_plotting(dataset: str) -> None:
    """
    Generate evaluation plots for a dataset.

    Parameters
    ----------
    dataset : str
        Name of the dataset whose results should be plotted.
    """
    path_training_results = f"./results/{dataset}"
    path_to_plot = f"./plot/{dataset}"

    test_accuracy_curve(path_training_results, path_to_plot)


def build_parser() -> argparse.ArgumentParser:
    """
    Build the command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured command-line argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Run federated learning benchmarks or generate plots."
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    learning_parser = subparsers.add_parser(
        "learning",
        help="Run benchmark training experiments.",
    )
    learning_parser.add_argument(
        "--config",
        required=True,
        help="Path to the benchmark configuration file.",
    )
    learning_parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of training jobs to run in parallel.",
    )
    learning_parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=None,
        metavar="GPU",
        help="GPU IDs to use, for example: --gpus 0 1 2.",
    )

    plot_parser = subparsers.add_parser(
        "plot",
        help="Generate plots from benchmark results.",
    )
    plot_parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        help="Dataset whose results should be plotted.",
    )

    return parser


def main() -> None:
    """Run the command selected from the command line."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "learning":
        run_learning(
            config_file=args.config,
            n_jobs=args.n_jobs,
            gpus=args.gpus,
        )
    elif args.command == "plot":
        run_plotting(dataset=args.dataset)


if __name__ == "__main__":
    main()
