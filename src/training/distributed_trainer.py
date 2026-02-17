"""
Production-grade distributed trainer
Supports:
- DDP
- AMP
- gradient accumulation
- large-model training (175B ready pattern)
"""

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

from .ddp_utils import setup_ddp, cleanup_ddp


class DistributedTrainer:

    def __init__(
        self,
        model,
        train_loader,
        lr=5e-5,
        grad_accum_steps=4,
        use_amp=True
    ):
        setup_ddp()

        self.local_rank = int(torch.cuda.current_device())
        self.device = torch.device(f"cuda:{self.local_rank}")

        self.model = model.to(self.device)
        self.model = DDP(self.model, device_ids=[self.local_rank])

        self.train_loader = train_loader
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        self.grad_accum_steps = grad_accum_steps
        self.use_amp = use_amp
        self.scaler = GradScaler(enabled=use_amp)

    def train_epoch(self):

        self.model.train()
        total_loss = 0.0

        for step, batch in enumerate(self.train_loader):

            batch = {k: v.to(self.device) for k, v in batch.items()}

            with autocast(enabled=self.use_amp):
                outputs = self.model(**batch)
                loss = outputs.loss / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.grad_accum_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item()

        return total_loss

    def finalize(self):
        cleanup_ddp()
