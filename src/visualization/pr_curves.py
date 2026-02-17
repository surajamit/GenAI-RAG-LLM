import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score


def plot_pr_curve(y_true, y_scores, model_name, save_dir="results/pr_curves"):
    """
    Plot Precision-Recall curve for a single model.
    """

    os.makedirs(save_dir, exist_ok=True)

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap_score = average_precision_score(y_true, y_scores)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, linewidth=2,
             label=f"{model_name} (AP={ap_score:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall Curve: {model_name}")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(save_dir, f"pr_curve_{model_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "model": model_name,
        "average_precision": float(ap_score),
        "save_path": save_path
    }


def compare_models_pr(models_dict, y_true, save_dir="results/pr_curves"):
    """
    Overlay PR curves for multiple models.

    models_dict:
        {
            "ATS": y_scores1,
            "GENAI": y_scores2,
            "LLM": y_scores3
        }
    """

    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(7, 6))

    results = []

    for model_name, scores in models_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)

        plt.plot(recall, precision, linewidth=2,
                 label=f"{model_name} (AP={ap:.3f})")

        results.append({
            "model": model_name,
            "average_precision": float(ap)
        })

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve Comparison: ATS vs GENAI vs LLM")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(save_dir, "pr_curve_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return results, save_path
