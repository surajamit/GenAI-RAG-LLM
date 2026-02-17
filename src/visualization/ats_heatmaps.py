import os
import numpy as np
import matplotlib.pyplot as plt


def plot_ats_performance_heatmap(metric_matrix,
                                 x_labels,
                                 y_labels,
                                 title="ATS Component Performance Heatmap",
                                 save_path="results/heatmaps/ats_heatmap.png"):
    """
    metric_matrix: 2D numpy array
        rows -> components/models
        cols -> metrics

    Example:
        rows: [ATS, GENAI, LLM]
        cols: [Accuracy, F1, MAE, User Rating]
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.imshow(metric_matrix, aspect="auto")
    plt.colorbar(label="Score")

    plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha="right")
    plt.yticks(range(len(y_labels)), y_labels)

    plt.title(title)

    # annotate values (paper quality)
    for i in range(metric_matrix.shape[0]):
        for j in range(metric_matrix.shape[1]):
            plt.text(j, i,
                     f"{metric_matrix[i, j]:.3f}",
                     ha="center",
                     va="center")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return save_path
