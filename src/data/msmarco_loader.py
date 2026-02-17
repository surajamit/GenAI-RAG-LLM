#URL: https://microsoft.github.io/msmarco/

from datasets import load_dataset
from .dataset_splitter import split_dataset

def load_msmarco(split="train"):
    """
    Loads MS MARCO passage ranking dataset.
    """
    dataset = load_dataset("ms_marco", "v1.1", split=split)
    return dataset

# Example
msmarco = load_msmarco("train")
print(msmarco[0])


def prepare_msmarco_splits(dataset):

    splits = split_dataset(
        dataset,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        stratify_key=None
    )

    return splits
