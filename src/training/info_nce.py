import torch.nn.functional as F

def info_nce_loss(query, positive, negatives, temperature):
    """
    Contrastive loss with temperature tuning.
    """

    query = F.normalize(query, dim=-1)
    positive = F.normalize(positive, dim=-1)
    negatives = F.normalize(negatives, dim=-1)

    pos_sim = (query * positive).sum(dim=-1) / temperature
    neg_sim = query @ negatives.t() / temperature

    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
    labels = torch.zeros(query.size(0), dtype=torch.long)

    return F.cross_entropy(logits, labels)
