"""DDP setup and training utilities."""
import os
import math
import random
import numpy as np
import torch
import torch.distributed as dist


def setup_ddp():
    if "LOCAL_RANK" not in os.environ:
        return 0, 0, 1
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    n_gpu = torch.cuda.device_count()
    if n_gpu == 0:
        raise RuntimeError(
            "DDP requested but no CUDA devices found. "
            "Run without torchrun for CPU-only mode."
        )
    if local_rank >= n_gpu:
        raise RuntimeError(
            f"local_rank={local_rank} but only {n_gpu} GPU(s) available. "
            f"Re-launch with: torchrun --nproc_per_node={n_gpu} <script>.py"
        )

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return local_rank, rank, world_size


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def all_reduce_mean(value: float, world_size: int) -> float:
    if world_size <= 1 or not dist.is_initialized():
        return value
    t = torch.tensor([value], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / world_size).item()


def set_seed(seed: int, rank: int = 0):
    s = seed + rank
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def cosine_warmup_lr(epoch: int, total_epochs: int, warmup_epochs: int,
                     base_lr: float, min_lr: float) -> float:
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def set_lr(optimizer, lr: float):
    for g in optimizer.param_groups:
        g["lr"] = lr