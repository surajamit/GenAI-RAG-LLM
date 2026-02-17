import matplotlib.pyplot as plt
import pandas as pd


def plot_accuracy_latency_tradeoff(csv_path="table_model_comparison.csv"):

    df = pd.read_csv(csv_path)

    plt.figure(figsize=(5.5, 4.2))

    plt.scatter(df["Latency (s)"], df["Accuracy (%)"])

    for _, row in df.iterrows():
        plt.annotate(row["Model"],
                     (row["Latency (s)"], row["Accuracy (%)"]),
                     fontsize=8)

    plt.xlabel("Latency (s)")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy–Latency Trade-off")

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("accuracy_latency_tradeoff.png", dpi=300)
    plt.close()
