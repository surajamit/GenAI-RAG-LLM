import pandas as pd
from experiments.statistical_benchmark import run_statistical_analysis


def build_statistical_table():
    stats_res = run_statistical_analysis()

    df = pd.DataFrame([
        {
            "Metric": "Exact Match Accuracy",
            "Baseline Mean": f"{stats_res['Baseline Mean']:.3f}",
            "Proposed Mean": f"{stats_res['Proposed Mean']:.3f}",
            "p-value": f"{stats_res['p-value']:.4f}",
            "Effect Size (d)": f"{stats_res['Cohen_d']:.3f}",
        }
    ])

    df.to_csv("statistical_results.csv", index=False)
    return df


if __name__ == "__main__":
    print(build_statistical_table())
