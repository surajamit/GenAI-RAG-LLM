import pandas as pd


def build_model_comparison_table():

    data = [
        ("GenAI-LLM-175B", 175, 95.2, 1.8, 280),
        ("GenAI-LLM-70B", 70, 92.1, 0.7, 110),
        ("GenAI-LLM-13B", 13, 86.3, 0.2, 26),
        ("GPT-4 API", 1700, 93.8, 2.3, None),
        ("GPT-3.5 API", 175, 88.7, 0.9, None),
    ]

    df = pd.DataFrame(data, columns=[
        "Model",
        "Parameters (B)",
        "Accuracy (%)",
        "Latency (s)",
        "GPU Memory (GB)"
    ])

    # ============================================================
    # Derived reviewer metrics
    # ============================================================

    df["Accuracy per Billion Params"] = (
        df["Accuracy (%)"] / df["Parameters (B)"]
    ).round(4)

    df["Efficiency Score"] = (
        df["Accuracy (%)"] / df["Latency (s)"]
    ).round(3)

    df["Throughput (req/s)"] = (
        1 / df["Latency (s)"]
    ).round(3)

    return df


if __name__ == "__main__":
    print(build_model_comparison_table())
