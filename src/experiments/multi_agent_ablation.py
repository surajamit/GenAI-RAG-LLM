import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def compute_fir_correlation(pred, human):
    """
    FIR correlation (treated as Pearson).

    ρ = cov(x,y)/(σx σy)
    """
    corr, _ = pearsonr(pred, human)
    return corr


def compute_mae(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def multi_agent_ablation_table(config_results):
    """
    config_results: dict
        { "Agents Active": (pred_scores, human_scores, user_rating) }
    """

    rows = []

    for name, (pred, human, rating) in config_results.items():
        rows.append({
            "Agents Active": name,
            "FIR Correlation": compute_fir_correlation(pred, human),
            "MAE": compute_mae(human, pred),
            "User Rating": float(np.mean(rating)),
        })

    return pd.DataFrame(rows)
