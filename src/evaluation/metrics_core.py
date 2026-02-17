import numpy as np
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr


# ===============================
# Retrieval Metrics
# ===============================

def precision_at_k(relevant, retrieved, k=10):
    """
    Precision@K = (# relevant in top-k) / k
    """
    retrieved_k = retrieved[:k]
    rel_set = set(relevant)
    hits = sum([1 for r in retrieved_k if r in rel_set])
    return hits / k


def recall_at_k(relevant, retrieved, k=10):
    """
    Recall@K = (# relevant in top-k) / (# relevant total)
    """
    retrieved_k = retrieved[:k]
    rel_set = set(relevant)
    hits = sum([1 for r in retrieved_k if r in rel_set])
    return hits / len(rel_set) if len(rel_set) > 0 else 0


def mean_reciprocal_rank(ranked_lists):
    """
    MRR = mean(1 / rank_i)
    """
    rr_scores = []
    for ranked in ranked_lists:
        rank = 0
        for i, val in enumerate(ranked):
            if val == 1:
                rank = i + 1
                break
        rr_scores.append(1 / rank if rank > 0 else 0)
    return np.mean(rr_scores)


def hit_rate_at_k(ranked_lists, k=10):
    """
    Hit@K
    """
    hits = [1 if 1 in ranked[:k] else 0 for ranked in ranked_lists]
    return np.mean(hits)


# ===============================
# ATS Metrics
# ===============================

def ats_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def ats_correlation(y_true, y_pred):
    corr, _ = pearsonr(y_true, y_pred)
    return corr


# ===============================
# Text-to-SQL Metrics
# ===============================

def execution_accuracy(correct_flags):
    """
    Accuracy = correct / total
    """
    return np.mean(correct_flags)


def syntax_error_rate(error_flags):
    """
    Error rate = errors / total
    """
    return np.mean(error_flags)
