# URL: https://www.clips.uantwerpen.be/conll2003/ner/

from datasets import load_dataset
from .dataset_splitter import split_dataset

def load_conll2003():
    """
    Loads CoNLL-2003 NER dataset.
    """
    dataset = load_dataset("conll2003")
    return dataset

# Example
conll = load_conll2003()
print(conll["train"][0])


def prepare_conll_splits(dataset):

    splits = split_dataset(
        dataset,
        stratify_key="ner_tags"  # important for NER balance
    )

    return splits