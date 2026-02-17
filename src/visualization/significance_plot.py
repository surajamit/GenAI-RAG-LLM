import matplotlib.pyplot as plt
import numpy as np


def plot_with_confidence(baseline, proposed, save_path="significance_plot.png"):
    plt.figure(figsize=(5, 4))

    means = [np.mean(baseline), np.mean(proposed)]
    stds = [np.std(baseline), np.std(proposed)]

    labels = ["Vector Only", "GraphRAG"]

    plt.bar(labels, means, yerr=stds, capsize=5)
    plt.ylabel("Exact Match Accuracy")
    plt.title("Statistical Comparison with Error Bars")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
