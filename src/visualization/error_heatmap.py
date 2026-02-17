import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def plot_error_heatmap(csv_path="table_failure_analysis.csv"):

    df = pd.read_csv(csv_path)

    plt.figure(figsize=(5, 4))

    sns.heatmap(
        df[["Percentage (%)"]],
        annot=True,
        yticklabels=df["Error Code"],
        cmap="Reds",
        cbar=False,
    )

    plt.title("Failure Distribution Heatmap")
    plt.tight_layout()
    plt.savefig("failure_heatmap.png", dpi=300)
    plt.close()
