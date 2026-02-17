import torch
import torch.nn as nn
import torch.nn.functional as F


class GenAIEmbedV3(nn.Module):
    """
    Contrastive embedding model.

    Optimizes InfoNCE:
    L = -log exp(sim(q,d+)/tau) / Σ exp(sim(q,d)/tau)
    """

    def __init__(self, dim=768):
        super().__init__()
        self.encoder = nn.Linear(dim, dim)

    def forward(self, x):
        z = F.normalize(self.encoder(x), dim=-1)
        return z


def contrastive_loss(q, pos, negs, tau=0.07):

    sim_pos = (q * pos).sum(-1) / tau
    sim_negs = (q.unsqueeze(1) * negs).sum(-1) / tau

    logits = torch.cat([sim_pos.unsqueeze(1), sim_negs], dim=1)
    labels = torch.zeros(q.size(0), dtype=torch.long)

    return F.cross_entropy(logits, labels)
