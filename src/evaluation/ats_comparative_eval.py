"""
ATS comparative evaluation.

Compares:
- GenAI Multi-Agent
- Single LLM
- Keyword baseline
- Rule-based ATS
"""

import numpy as np
from sklearn.metrics import mean_absolute_error


def compute_user_rating_correlation(pred_scores, human_scores):
    return np.corrcoef(pred_scores, human_scores)[0, 1]


def evaluate_ats_models(results_dict):
    """
    results_dict format:
    {
        "genai_agentic": {...},
        "llm_single": {...},
        "keyword_baseline": {...},
        "rule_based": {...}
    }
    """

    summary = {}

    for model_name, payload in results_dict.items():

        pred = np.array(payload["pred_scores"])
        human = np.array(payload["human_scores"])
        latency = np.array(payload["latency"])

        summary[model_name] = {
            "MAE": float(mean_absolute_error(human, pred)),
            "Correlation": float(compute_user_rating_correlation(pred, human)),
            "Avg_Latency": float(latency.mean()),
            "P95_Latency": float(np.percentile(latency, 95))
        }

    return summary
