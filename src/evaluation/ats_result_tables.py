"""
Generates ATS tables.
"""

import pandas as pd


def build_ats_comparison_table(summary_dict):

    rows = []

    for model, metrics in summary_dict.items():

        rows.append({
            "Model": model,
            "MAE": metrics["MAE"],
            "Correlation": metrics["Correlation"],
            "Avg Latency (s)": metrics["Avg_Latency"],
            "P95 Latency (s)": metrics["P95_Latency"]
        })

    df = pd.DataFrame(rows)
    return df.sort_values("Correlation", ascending=False)
