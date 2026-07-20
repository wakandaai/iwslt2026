"""
Learning rate schedulers.

CosineAnnealingWarmupRestarts:
    Cosine decay with linear warmup and optional restarts.
    Paper: https://arxiv.org/pdf/1608.03983.pdf
"""

from __future__ import annotations

import math

import torch
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """Cosine annealing with linear warmup and optional restarts.

    Args:
        optimizer:         Wrapped optimizer.
        first_cycle_steps: Steps in the first cycle.
        cycle_mult:        Multiplier for cycle length after each restart.
        max_lr:            Peak learning rate in the first cycle.
        min_lr:            Minimum learning rate.
        warmup_steps:      Linear warmup steps at the start of each cycle.
        gamma:             Decay factor applied to max_lr after each cycle.
        last_epoch:        Index of last epoch (for resume).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 0.1,
        min_lr: float = 1e-6,
        warmup_steps: int = 0,
        gamma: float = 1.0,
        last_epoch: int = -1,
    ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult        = cycle_mult
        self.base_max_lr       = max_lr
        self.max_lr            = max_lr
        self.min_lr            = min_lr
        self.warmup_steps      = warmup_steps
        self.gamma             = gamma

        self.cur_cycle_steps = first_cycle_steps
        self.cycle           = 0
        self.step_in_cycle   = last_epoch

        super().__init__(optimizer, last_epoch)
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [
                (self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr + (self.max_lr - base_lr)
                * (1 + math.cos(
                    math.pi
                    * (self.step_in_cycle - self.warmup_steps)
                    / (self.cur_cycle_steps - self.warmup_steps)
                )) / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle += 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle           += 1
                self.step_in_cycle   -= self.cur_cycle_steps
                self.cur_cycle_steps  = (
                    int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult)
                    + self.warmup_steps
                )
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle         = epoch // self.first_cycle_steps
                else:
                    n = int(math.log(
                        (epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1),
                        self.cycle_mult,
                    ))
                    self.cycle         = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps
                        * (self.cycle_mult ** n - 1)
                        / (self.cycle_mult - 1)
                    )
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** n
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle   = epoch

        self.max_lr    = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


def build_scheduler(
    name: str,
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    **kwargs,
) -> _LRScheduler | None:
    """Build a scheduler by name.

    Args:
        name:        "cosine_warmup_restarts" | "linear_warmup" | "none"
        optimizer:   The optimizer to wrap.
        total_steps: Total training steps.
        **kwargs:    Forwarded to the scheduler constructor.

    Returns:
        Scheduler instance, or None if name is "none".
    """
    if name in ("none", None):
        return None

    elif name == "cosine_warmup_restarts":
        return CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=kwargs.get("first_cycle_steps", total_steps),
            cycle_mult=kwargs.get("cycle_mult", 1.0),
            max_lr=kwargs.get("max_lr", optimizer.param_groups[0]["lr"]),
            min_lr=kwargs.get("min_lr", 1e-6),
            warmup_steps=kwargs.get("warmup_steps", 0),
            gamma=kwargs.get("gamma", 1.0),
        )

    elif name == "linear_warmup":
        from transformers import get_linear_schedule_with_warmup
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=kwargs.get("warmup_steps", 0),
            num_training_steps=total_steps,
        )

    else:
        raise ValueError(
            f"Unknown scheduler '{name}'. Choose: cosine_warmup_restarts, linear_warmup, none"
        )
