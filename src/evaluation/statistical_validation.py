import numpy as np
from scipy import stats


def compute_confidence_interval(values, confidence=0.95):
    """
    95% Confidence Interval

    CI = μ ± t_(α/2) * (σ / √n)
    """
    values = np.array(values)
    mean = np.mean(values)
    sem = stats.sem(values)
    margin = sem * stats.t.ppf((1 + confidence) / 2., len(values) - 1)

    return mean, (mean - margin, mean + margin)


def paired_t_test(baseline, proposed):
    """
    Paired t-test

    H0: No difference
    H1: Proposed is better
    """
    t_stat, p_value = stats.ttest_rel(proposed, baseline)
    return t_stat, p_value


def cohens_d(baseline, proposed):
    """
    Effect size

    d = (μ1 − μ2) / σ_pooled
    """
    baseline = np.array(baseline)
    proposed = np.array(proposed)

    diff = proposed - baseline
    return np.mean(diff) / np.std(diff, ddof=1)
