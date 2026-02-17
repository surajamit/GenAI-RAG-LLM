class DPOTrainer:
    """
    Direct Preference Optimization implementation.
    """

    def __init__(self, model, lr=5e-7):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def dpo_loss(self, chosen_logp, rejected_logp):
        """
        DPO objective.
        """

        return -torch.log(torch.sigmoid(chosen_logp - rejected_logp)).mean()

    def train_step(self, chosen_logp, rejected_logp):
        loss = self.dpo_loss(chosen_logp, rejected_logp)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
