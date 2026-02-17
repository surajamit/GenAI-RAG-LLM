import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation layer for transformer projections.
    Rank = 64 as per manuscript.
    """

    def __init__(self, in_features, out_features, rank=64, alpha=16):
        super().__init__()

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Frozen base weight
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features),
            requires_grad=False
        )

        # LoRA adapters
        self.A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x):
        base = torch.nn.functional.linear(x, self.weight)
        lora_update = (x @ self.A.t() @ self.B.t()) * self.scaling
        return base + lora_update
