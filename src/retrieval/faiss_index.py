import faiss
import numpy as np


class FaissIndexer:
    """
    Dense vector index for GraphRAG.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.doc_ids = []

    def add(self, embeddings: np.ndarray, ids):
        """
        Adds normalized vectors.
        """
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype("float32"))
        self.doc_ids.extend(ids)

    def search(self, query_vec: np.ndarray, k: int = 5):
        q = query_vec.astype("float32").reshape(1, -1)
        faiss.normalize_L2(q)

        scores, idxs = self.index.search(q, k)

        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append((self.doc_ids[idx], float(score)))

        return results
