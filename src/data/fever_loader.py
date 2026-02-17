#URL: https://fever.ai/

from datasets import load_dataset
from .dataset_splitter import split_dataset


def load_fever():
    """
    Loads FEVER fact verification dataset.
    """
    dataset = load_dataset("fever", "v1.0")
    return dataset

# Example
fever = load_fever()
print(fever["train"][0])


def prepare_fever_splits(dataset):

    return split_dataset(
        dataset,
        stratify_key="label"
    )
