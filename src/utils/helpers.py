import numpy as np


# ================================
# Cosine Similarity (Eq. RAG-1)
# ================================
def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Computes cosine similarity between two embedding vectors.

    Formula:
        sim = (a · b) / (||a|| ||b||)
    """
    numerator = np.dot(vec_a, vec_b)
    denominator = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    return float(numerator / (denominator + 1e-12))


# ================================
# Softmax Temperature Scaling
# ================================
def temperature_softmax(scores: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """
    Softmax with temperature (used in contrastive learning).

    Formula:
        p_i = exp(s_i / τ) / Σ_j exp(s_j / τ)
    """
    scaled = scores / tau
    exp_scores = np.exp(scaled - np.max(scaled))
    return exp_scores / np.sum(exp_scores)


# ================================
# Mean Reciprocal Rank (MRR)
# ================================
def mean_reciprocal_rank(ranks):
    """
    Computes Mean Reciprocal Rank.

    Formula:
        MRR = (1/N) Σ (1 / rank_i)
    """
    ranks = np.array(ranks)
    return np.mean(1.0 / (ranks + 1e-12))
