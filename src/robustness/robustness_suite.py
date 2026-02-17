import numpy as np
import pandas as pd
from src.evaluation.metrics_core import f1_score, exact_match

np.random.seed(42)


# ============================================================
# 1. Noise Robustness (Query Perturbation)
# ============================================================

def noise_robustness_test(y_true, y_pred_clean, noise_levels=(0.0, 0.05, 0.1, 0.2)):
    """
    Tests degradation under query noise.

    R_noise = Performance(noisy) / Performance(clean)
    """

    results = []

    for nl in noise_levels:
        noise_mask = np.random.rand(len(y_pred_clean)) < nl
        noisy_pred = y_pred_clean.copy()
        noisy_pred[noise_mask] = np.random.permutation(noisy_pred[noise_mask])

        f1 = f1_score(y_true, noisy_pred)
        em = exact_match(y_true, noisy_pred)

        results.append({
            "Noise Level": nl,
            "F1": f1,
            "Exact Match": em,
        })

    return pd.DataFrame(results)


# ============================================================
# 2. Retrieval Depth Sensitivity
# ============================================================

def topk_sensitivity_test(relevant_docs, retrieved_pool, ks=(5, 10, 20, 40)):
    """
    Tests stability vs retrieval depth.
    """

    from src.evaluation.metrics_core import precision_at_k, recall_at_k

    rows = []

    for k in ks:
        p = precision_at_k(relevant_docs, retrieved_pool[:k], k)
        r = recall_at_k(relevant_docs, retrieved_pool[:k], k)

        rows.append({
            "Top-K": k,
            "Precision": p,
            "Recall": r,
        })

    return pd.DataFrame(rows)
