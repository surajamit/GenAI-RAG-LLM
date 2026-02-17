import numpy as np
import pandas as pd


def marginal_contribution(full_score, removed_score):
    """
    Δ = Full − WithoutAgent
    """
    return full_score - removed_score


def ats_performance_impact(full_metrics, ablated_metrics):
    """
    full_metrics: dict(agent -> score_full)
    ablated_metrics: dict(agent -> score_without)
    """

    rows = []

    for agent in full_metrics:
        rows.append({
            "Agent": agent,
            "Marginal Contribution": marginal_contribution(
                full_metrics[agent],
                ablated_metrics.get(agent, full_metrics[agent]),
            ),
        })

    return pd.DataFrame(rows)
