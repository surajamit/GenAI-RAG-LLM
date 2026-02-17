import numpy as np
import pandas as pd

from src.evaluation.statistical_validation import (
    compute_confidence_interval,
    paired_t_test,
    cohens_d
)


def simulate_runs(n=10):
    """
    Replace with real experiment outputs.
    """

    # Baseline (vector only)
    baseline = np.random.normal(loc=0.62, scale=0.02, size=n)

    # Proposed GraphRAG
    proposed = np.random.normal(loc=0.85, scale=0.015, size=n)

    return baseline, proposed


def run_statistical_analysis():
    baseline, proposed = simulate_runs()

    mean_b, ci_b = compute_confidence_interval(baseline)
    mean_p, ci_p = compute_confidence_interval(proposed)

    t_stat, p_value = paired_t_test(baseline, proposed)
    effect = cohens_d(baseline, proposed)

    results = {
        "Baseline Mean": mean_b,
        "Proposed Mean": mean_p,
        "Baseline CI": ci_b,
        "Proposed CI": ci_p,
        "t-statistic": t_stat,
        "p-value": p_value,
        "Cohen_d": effect,
    }

    return results


if __name__ == "__main__":
    res = run_statistical_analysis()
    print(res)
