import numpy as np
import pandas as pd


def mine_hard_cases(samples, scores, k=20):
    """
    Select lowest-performing cases.

    HardCases = argsort(score)[:k]
    """

    idx = np.argsort(scores)[:k]

    hard_df = pd.DataFrame({
        "SampleID": idx,
        "Score": scores[idx],
    })

    return hard_df
