"""
ddp_utils.py — Distributed Data Parallel setup utilities.

Usage:
    from st.utils.ddp_utils import setup_ddp, teardown_ddp, reduce_tensor

Launch with torchrun:
    torchrun --standalone --nproc_per_node=4 -m st.training.train_st \
        --config configs/experiment/stage3.yaml
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def setup_ddp() -> tuple[bool, int, int, int, str]:
    """Initialize DDP if launched via torchrun; otherwise single-GPU fallback.

    Returns:
        is_ddp:     True if running in DDP mode.
        rank:       Global rank of this process (0 for single-GPU).
        local_rank: Rank on this machine (= GPU index).
        world_size: Total number of processes.
        device:     Device string e.g. "cuda:0".
    """
    if int(os.environ.get("RANK", -1)) != -1:
        assert torch.cuda.is_available(), "DDP requires CUDA"
        dist.init_process_group(backend="nccl")
        rank       = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device     = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        return True, rank, local_rank, world_size, device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return False, 0, 0, 1, device


def teardown_ddp() -> None:
    """Clean up the process group. Call at the end of training."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def reduce_tensor(t: torch.Tensor, op=dist.ReduceOp.AVG) -> torch.Tensor:
    """All-reduce a scalar tensor across all ranks.

    No-op when not in DDP mode (dist not initialized).
    Returns the reduced tensor (in-place on the input).
    """
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=op)
    return t


def barrier() -> None:
    """Synchronize all ranks. No-op outside DDP."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()