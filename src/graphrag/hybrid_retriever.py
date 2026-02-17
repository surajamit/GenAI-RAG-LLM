import numpy as np
from utils.helpers import cosine_similarity


def hybrid_score(vec_q, vec_d, graph_score, alpha=0.7):
    vector_score = cosine_similarity(vec_q, vec_d)
    return alpha * vector_score + (1 - alpha) * graph_score
