import numpy as np
import pandas as pd


def graphrag_metrics(p_at_10, r_at_10, latency):
    """
    Core GraphRAG KPIs.
    """
    return {
        "Precision@10": p_at_10,
        "Recall@10": r_at_10,
        "Latency (s)": latency,
    }


def graphrag_ablation(variants):
    """
    variants:
        {method: (p10, r10, latency)}
    """

    rows = []

    for name, vals in variants.items():
        metrics = graphrag_metrics(*vals)
        metrics["Method"] = name
        rows.append(metrics)

    return pd.DataFrame(rows)
