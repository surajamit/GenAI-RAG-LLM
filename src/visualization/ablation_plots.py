import matplotlib.pyplot as plt
import pandas as pd


def plot_agent_ablation(csv_path="ablation_ats_agents.csv"):

    df = pd.read_csv(csv_path)

    plt.figure(figsize=(6, 4))
    plt.barh(df["Configuration"], df["MAE"])
    plt.xlabel("MAE ↓")
    plt.title("ATS Multi-Agent Ablation Study")

    plt.tight_layout()
    plt.savefig("ats_agent_ablation.png", dpi=300)
    plt.close()
