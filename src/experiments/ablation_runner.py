"""
Systematic component ablation.
"""

from graphrag.hybrid_retriever import hybrid_score


def run_ablation(configs, vec_score=0.8, graph_score=0.6):
    """
    Runs ablation across alpha values.
    """
    results = []

    for alpha in configs["alpha_values"]:
        score = hybrid_score(
            vec_score,
            graph_score,
            graph_score,
            alpha=alpha,
        )

        results.append({
            "alpha": alpha,
            "score": score,
        })

    return results


if __name__ == "__main__":
    configs = {
        "alpha_values": [0.0, 0.3, 0.5, 0.7, 1.0]
    }

    print(run_ablation(configs))
