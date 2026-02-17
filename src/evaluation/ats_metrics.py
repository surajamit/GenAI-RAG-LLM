import numpy as np


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))


def pearson_correlation(x, y):
    return np.corrcoef(x, y)[0, 1]
