import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_embedding(text):
    return model.encode(text, normalize_embeddings=True)

def cosine_similarity(q, d):
    return np.dot(q, d) / (np.linalg.norm(q) * np.linalg.norm(d))