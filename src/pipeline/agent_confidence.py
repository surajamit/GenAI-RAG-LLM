import numpy as np


def agent_confidence(logits):
    """
    Confidence = max softmax probability
    """

    probs = logits.softmax(-1)
    return probs.max().item()
