class PatchedAttention(nn.Module):
    """
    Attention with LoRA applied to Q,K,V,O projections.
    """

    def __init__(self, hidden_size):
        super().__init__()

        self.q_proj = LoRALinear(hidden_size, hidden_size, rank=64)
        self.k_proj = LoRALinear(hidden_size, hidden_size, rank=64)
        self.v_proj = LoRALinear(hidden_size, hidden_size, rank=64)
        self.o_proj = LoRALinear(hidden_size, hidden_size, rank=64)

    def forward(self, x):
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        attn = torch.softmax(Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1)), dim=-1)
        out = attn @ V
        return self.o_proj(out)
