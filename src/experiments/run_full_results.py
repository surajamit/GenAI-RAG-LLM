import numpy as np
import pandas as pd

from src.evaluation.metrics_core import (
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    hit_rate_at_k,
    ats_mae,
    ats_correlation,
    execution_accuracy,
    syntax_error_rate,
)


np.random.seed(42)


# ===============================
# Simulated Retrieval Evaluation
# ===============================

def simulate_retrieval():
    relevant = list(range(5))

    retrieved_vector = np.random.permutation(20)
    retrieved_graph = np.random.permutation(20)

    p_vec = precision_at_k(relevant, retrieved_vector)
    p_graph = precision_at_k(relevant, retrieved_graph)

    r_vec = recall_at_k(relevant, retrieved_vector)
    r_graph = recall_at_k(relevant, retrieved_graph)

    ranked_lists_vec = [np.random.randint(0, 2, 10) for _ in range(20)]
    ranked_lists_graph = [np.random.randint(0, 2, 10) for _ in range(20)]

    mrr_vec = mean_reciprocal_rank(ranked_lists_vec)
    mrr_graph = mean_reciprocal_rank(ranked_lists_graph)

    hit_vec = hit_rate_at_k(ranked_lists_vec)
    hit_graph = hit_rate_at_k(ranked_lists_graph)

    return {
        "vector": (p_vec, r_vec, mrr_vec, hit_vec),
        "graph": (p_graph, r_graph, mrr_graph, hit_graph),
    }


# ===============================
# ATS Evaluation
# ===============================

def simulate_ats():
    y_true = np.random.uniform(0, 100, 500)
    y_pred = y_true + np.random.normal(0, 8, 500)

    mae = ats_mae(y_true, y_pred)
    corr = ats_correlation(y_true, y_pred)

    return mae, corr


# ===============================
# Text-to-SQL Evaluation
# ===============================

def simulate_text2sql():
    correct_flags = np.random.binomial(1, 0.94, 200)
    error_flags = np.random.binomial(1, 0.06, 200)

    acc = execution_accuracy(correct_flags)
    err = syntax_error_rate(error_flags)

    return acc, err
