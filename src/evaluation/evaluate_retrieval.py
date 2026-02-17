from src.data.msmarco_loader import load_msmarco
from .metrics import precision_at_k, recall_at_k, mean_reciprocal_rank

def evaluate_retrieval(system):

    dataset = load_msmarco(split="validation")

    p10, r10, ranks = [], [], []

    for sample in dataset:

        retrieved = system.retrieve(sample["query"])
        relevant = sample["passages"]["passage_text"]

        p10.append(precision_at_k(relevant, retrieved, 10))
        r10.append(recall_at_k(relevant, retrieved, 10))
        ranks.append(system.rank_of_first_relevant(sample))

    return {
        "Precision@10": sum(p10)/len(p10),
        "Recall@10": sum(r10)/len(r10),
        "MRR": mean_reciprocal_rank(ranks)
    }
