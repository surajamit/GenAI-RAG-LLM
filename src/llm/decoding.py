import torch
import torch.nn.functional as F


def sample_with_temperature(logits, temperature=0.7):
    """
    Temperature sampling.

    p_i = softmax(z_i / T)
    """

    probs = F.softmax(logits / temperature, dim=-1)
    token = torch.multinomial(probs, num_samples=1)

    return token
