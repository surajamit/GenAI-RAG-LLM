import numpy as np


def precision_at_k(relevant, retrieved, k=10):
    retrieved_k = retrieved[:k]
    hits = sum(1 for r in retrieved_k if r in relevant)
    return hits / k


def recall_at_k(relevant, retrieved, k=10):
    retrieved_k = retrieved[:k]
    hits = sum(1 for r in retrieved_k if r in relevant)
    return hits / len(relevant)
