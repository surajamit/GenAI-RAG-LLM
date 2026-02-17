import numpy as np
import pandas as pd


def component_accuracy(y_true, y_pred):
    """
    Accuracy = (TP + TN) / N
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float((y_true == y_pred).mean())


def ats_component_analysis(component_outputs):
    """
    component_outputs:
        {component: (y_true, y_pred, proc_time)}
    """

    rows = []

    for comp, (yt, yp, t) in component_outputs.items():
        rows.append({
            "Component": comp,
            "Individual Accuracy": component_accuracy(yt, yp),
            "Processing Time (s)": float(np.mean(t)),
        })

    return pd.DataFrame(rows)
