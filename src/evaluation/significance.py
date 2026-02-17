import numpy as np
from scipy import stats


def paired_t_test(system_a, system_b):
    """
    Tests statistical significance.
    """
    t_stat, p_value = stats.ttest_rel(system_a, system_b)

    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
    }
