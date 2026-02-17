import pandas as pd
import numpy as np


def pdf_chat_ablation(results_dict):
    """
    results_dict:
        {variant: (exact_match, f1, latency)}
    """

    rows = []

    for name, (em, f1, lat) in results_dict.items():
        rows.append({
            "Variant": name,
            "Exact Match": em,
            "F1 Score": f1,
            "Latency (s)": lat,
        })

    return pd.DataFrame(rows)
