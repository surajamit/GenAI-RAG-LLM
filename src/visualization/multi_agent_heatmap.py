def plot_ablation_heatmap(ablation_results,
                          save_path="results/heatmaps/ats_ablation_heatmap.png"):
    """
    ablation_results: list of dicts from ATS ablation

    Required keys:
        agents_active, fir_correlation, mae, user_rating
    """

    import numpy as np

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    models = [f"{r['agents_active']} agents" for r in ablation_results]

    matrix = np.array([
        [
            r["fir_correlation"],
            r["mae"],
            r["user_rating"]
        ]
        for r in ablation_results
    ])

    x_labels = ["FIR Corr", "MAE", "User Rating"]

    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, aspect="auto")
    plt.colorbar(label="Score")

    plt.xticks(range(len(x_labels)), x_labels)
    plt.yticks(range(len(models)), models)

    plt.title("ATS Multi-Agent Ablation Heatmap")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, f"{matrix[i, j]:.3f}",
                     ha="center", va="center")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return save_path
