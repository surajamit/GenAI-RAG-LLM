# URL: https://yale-lily.github.io/spider

from datasets import load_dataset
from .dataset_splitter import split_dataset


def load_spider():
    """
    Loads Spider text-to-SQL dataset.
    """
    dataset = load_dataset("spider")
    return dataset

# Example
spider = load_spider()
print(spider["train"][0])


def prepare_spider_splits(dataset):

    # SQL datasets typically not stratified
    return split_dataset(dataset)
