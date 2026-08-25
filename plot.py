from benchmark.evaluate_results import test_accuracy_curve, loss_heatmap, test_heatmap, aggregated_test_heatmap

def plot(
    path_training_results: str = "./results",
    path_to_plot: str = "./plot",
) -> None:

    test_accuracy_curve(path_training_results, path_to_plot)

    loss_heatmap(path_training_results, path_to_plot)

    test_heatmap(path_training_results, path_to_plot)

    aggregated_test_heatmap(path_training_results, path_to_plot)

if __name__ == "__main__":
    plot()
