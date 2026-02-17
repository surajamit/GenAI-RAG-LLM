
"""
Dataset Splitting Module

Supports:
- stratified split
- fixed random seed
- multi-task datasets
- reproducibility logging
"""

import random
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split


GLOBAL_SEED = 42


def set_global_seed(seed: int = GLOBAL_SEED):
    random.seed(seed)


def split_dataset(
    data: List[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    stratify_key: str = None,
    seed: int = GLOBAL_SEED
) -> Dict[str, List[dict]]:
    """
    Standard dataset split.

    Returns:
        {train, val, test}
    """

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    set_global_seed(seed)

    stratify_labels = None
    if stratify_key:
        stratify_labels = [x[stratify_key] for x in data]

    train_data, temp_data = train_test_split(
        data,
        test_size=(1 - train_ratio),
        stratify=stratify_labels,
        random_state=seed
    )

    if stratify_key:
        temp_labels = [x[stratify_key] for x in temp_data]
    else:
        temp_labels = None

    val_size = val_ratio / (val_ratio + test_ratio)

    val_data, test_data = train_test_split(
        temp_data,
        test_size=(1 - val_size),
        stratify=temp_labels,
        random_state=seed
    )

    return {
        "train": train_data,
        "val": val_data,
        "test": test_data
    }
