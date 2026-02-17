"""
Human–AI agreement analysis for ATS.
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr


def human_alignment_stats(pred_scores, human_scores):

    pearson_corr, _ = pearsonr(pred_scores, human_scores)
    spearman_corr, _ = spearmanr(pred_scores, human_scores)

    return {
        "pearson": float(pearson_corr),
        "spearman": float(spearman_corr),
        "mean_diff": float(np.mean(np.abs(
            np.array(pred_scores) - np.array(human_scores)
        )))
    }
