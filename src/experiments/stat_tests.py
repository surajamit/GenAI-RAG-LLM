import numpy as np
from scipy.stats import ttest_ind


def significance_test(a, b):
    """
    Two-sided t-test
    H0: means equal
    """
    stat, p = ttest_ind(a, b)
    return stat, p


def bootstrap_ci(data, n_boot=1000, alpha=0.05):

    means = []
    n = len(data)

    for _ in range(n_boot):
        sample = np.random.choice(data, size=n, replace=True)
        means.append(np.mean(sample))

    lower = np.percentile(means, 100 * (alpha / 2))
    upper = np.percentile(means, 100 * (1 - alpha / 2))

    return lower, upper
