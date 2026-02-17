"""
Distributed utilities for multi-GPU training.
"""

import os
import torch
import torch.distributed as dist


def setup_ddp():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup_ddp():
    dist.destroy_process_group()


def is_main_process():
    return dist.get_rank() == 0


def ddp_barrier():
    dist.barrier()
