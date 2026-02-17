"""
Evaluates contribution of each ATS agent.
"""

import numpy as np


def agent_ablation_study(agent_outputs, human_scores):
    """
    agent_outputs:
        {
            "full_system": [...],
            "minus_keyword": [...],
            "minus_format": [...],
            ...
        }
    """

    base = np.array(agent_outputs["full_system"])

    results = {}

    for name, scores in agent_outputs.items():

        scores = np.array(scores)

        results[name] = {
            "MAE": float(np.mean(np.abs(scores - human_scores))),
            "FIR_Correlation": float(np.corrcoef(scores, human_scores)[0, 1]),
            "Marginal_Contribution": float(
                np.mean(np.abs(base - human_scores))
                - np.mean(np.abs(scores - human_scores))
            )
        }

    return results
