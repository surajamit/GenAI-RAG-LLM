import torch
import torch.nn.functional as F


def autoregressive_loss(logits, labels):
    """
    Standard causal LM loss.

    L_LLM = - Σ log p_theta(y_t | y_<t, x, C)
    """

    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )

    return loss
