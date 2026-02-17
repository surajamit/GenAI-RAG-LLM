"""
Statistical validation for ATS improvements.
"""

from scipy.stats import ttest_rel, wilcoxon
import numpy as np


def paired_significance_test(model_a, model_b):

    t_stat, t_p = ttest_rel(model_a, model_b)
    w_stat, w_p = wilcoxon(model_a, model_b)

    return {
        "paired_t_p": float(t_p),
        "wilcoxon_p": float(w_p),
        "significant_0.05": bool(t_p < 0.05)
    }


def effect_size_cohen_d(a, b):

    a = np.array(a)
    b = np.array(b)

    diff = a - b
    return float(diff.mean() / diff.std())
