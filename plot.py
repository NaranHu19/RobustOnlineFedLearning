from benchmark.evaluate_results import (
    aggregated_test_heatmap,
    loss_heatmap,
    test_accuracy_curve,
    test_heatmap,
)


def plot(
    path_training_results: str = "./results",
    path_to_plot: str = "./plot",
) -> None:
    """
    Generate benchmark evaluation plots.

    Create the test-accuracy curve, loss heatmap, test heatmap, and
    aggregated test heatmap from the stored training results.

    Parameters
    ----------
    path_training_results : str, optional
        Path to the directory containing the training results.
    path_to_plot : str, optional
        Path to the directory in which the generated plots are stored.
    """
    test_accuracy_curve(path_training_results, path_to_plot)

    loss_heatmap(path_training_results, path_to_plot)

    test_heatmap(path_training_results, path_to_plot)

    aggregated_test_heatmap(path_training_results, path_to_plot)


if __name__ == "__main__":
    plot()
