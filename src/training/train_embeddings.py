import torch
import torch.nn.functional as F

def info_nce_loss(q, k, temperature=0.05):
    """
    InfoNCE contrastive loss
    """

    sim = torch.matmul(q, k.T) / temperature
    labels = torch.arange(q.size(0)).to(q.device)

    loss = F.cross_entropy(sim, labels)
    return loss


def train_embedding_model(model, dataloader, temperature=0.05):

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    model.train()

    for batch in dataloader:

        q_emb, k_emb = model(batch)

        loss = info_nce_loss(q_emb, k_emb, temperature)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return model
