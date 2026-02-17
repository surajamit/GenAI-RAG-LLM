import numpy as np


class GenAIEmbedV3:
    """
    Placeholder embedding model aligned with manuscript.
    Replace with real encoder.
    """

    def __init__(self, dim: int = 768):
        self.dim = dim

    def encode(self, texts):
        rng = np.random.default_rng(42)
        return rng.normal(size=(len(texts), self.dim))
