"""
Duration-based bucket sampler with synchronized batches for DDP.

Groups samples by duration into buckets, then builds batches that cap total
audio seconds per batch (not instance count). Prevents OOM from variable-
length sequences while keeping padding overhead low.

Synchronized batches across ranks (Zelasko et al. 2025, ASRU):
  - All ranks use the SAME shared RNG to shuffle samples within buckets
    and to shuffle bucket order → all ranks build an IDENTICAL global
    batch list.
  - Each rank then takes batches[rank::world_size] as its slice.

Mid-epoch resume:
  - Save `epoch` and `batches_consumed_this_epoch` in the training checkpoint.
  - On resume, call `set_epoch(epoch)` to rebuild the identical batch list,
    then call `skip(batches_consumed_this_epoch)` so the next __iter__ skips
    those batches. The first epoch after resume is partial; subsequent
    epochs are full.
"""

from __future__ import annotations

import random
from typing import Iterator

from torch.utils.data import Sampler


class DurationBucketSampler(Sampler):
    """Batch sampler that caps total audio duration per batch.

    Args:
        dataset:              Dataset with a `.durations` attribute (list[float]
                              or np.ndarray).
        target_duration:      Max total audio seconds per batch.
        max_batch_size:       Hard cap on instances per batch.
        bucket_width:         Multiplier — samples within a bucket have durations
                              spanning at most [min, min * bucket_width].
        shuffle:              Shuffle samples within each bucket before batching.
        shuffle_buckets:      Shuffle bucket order each epoch.
        drop_last:            Drop the final incomplete batch per bucket.
        rank:                 DDP rank (0 for single GPU).
        world_size:           DDP world size (1 for single GPU).
        seed:                 Shared seed — must be identical on all ranks.
    """

    def __init__(
        self,
        dataset,
        target_duration: float = 120.0,
        max_batch_size: int = 64,
        bucket_width: float = 1.5,
        shuffle: bool = True,
        shuffle_buckets: bool = True,
        drop_last: bool = False,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
    ):
        self.dataset          = dataset
        self.target_duration  = target_duration
        self.max_batch_size   = max_batch_size
        self.bucket_width     = bucket_width
        self.shuffle          = shuffle
        self.shuffle_buckets  = shuffle_buckets
        self.drop_last        = drop_last
        self.rank             = rank
        self.world_size       = world_size
        self.seed             = seed

        # Accept both list and np.ndarray
        durations = dataset.durations
        if hasattr(durations, "tolist"):
            self.durations: list[float] = durations.tolist()
        else:
            self.durations = list(durations)

        # Single shared RNG — same seed on all ranks → all ranks build
        # the same global batch list, then slice by rank.
        self._rng = random.Random(seed)
        self._current_epoch = 0

        # Mid-epoch resume: number of per-rank batches to skip in the next
        # __iter__. Reset to 0 when the epoch completes naturally.
        self._skip_batches = 0

        self._buckets     = self._make_buckets()
        self._all_batches = self._make_batches(self._buckets)

    # ------------------------------------------------------------------

    def _make_buckets(self) -> list[list[int]]:
        """Sort indices by duration and group into width-bounded buckets."""
        indexed = sorted(enumerate(self.durations), key=lambda x: x[1])
        buckets: list[list[int]] = []
        current: list[int]       = []
        bucket_min: float | None = None

        for idx, dur in indexed:
            if bucket_min is None:
                current    = [idx]
                bucket_min = dur
            elif dur <= bucket_min * self.bucket_width:
                current.append(idx)
            else:
                buckets.append(current)
                current    = [idx]
                bucket_min = dur

        if current:
            buckets.append(current)

        return buckets

    def _make_batches(self, buckets: list[list[int]]) -> list[list[int]]:
        """Build batches from buckets respecting target_duration and max_batch_size."""
        all_batches: list[list[int]] = []

        for bucket in buckets:
            batch: list[int]  = []
            batch_dur: float  = 0.0

            for idx in bucket:
                dur = self.durations[idx]
                if batch and (
                    batch_dur + dur > self.target_duration
                    or len(batch) >= self.max_batch_size
                ):
                    all_batches.append(batch)
                    batch, batch_dur = [], 0.0

                batch.append(idx)
                batch_dur += dur

            if batch and not self.drop_last:
                all_batches.append(batch)

        return all_batches

    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[list[int]]:
        # Step 1: Shuffle SAMPLES within each bucket using the SHARED RNG.
        buckets = [b.copy() for b in self._buckets]
        if self.shuffle:
            for b in buckets:
                self._rng.shuffle(b)

        # Step 2: Build batches.
        batches = self._make_batches(buckets)

        # Step 3: Shuffle BATCH ORDER using the SHARED RNG.
        if self.shuffle_buckets:
            self._rng.shuffle(batches)

        # Step 4: Slice for this rank.
        batches = batches[self.rank::self.world_size]

        # Step 5: Mid-epoch resume — skip already-consumed batches, then clear
        # the counter so the NEXT epoch starts fresh.
        if self._skip_batches > 0:
            batches = batches[self._skip_batches:]
            self._skip_batches = 0

        yield from batches

    def __len__(self) -> int:
        # Per-rank batch count. Use ceil-style division so rank 0 (which gets
        # any remainder) doesn't under-report.
        total = len(self._all_batches)
        return (total + self.world_size - 1 - self.rank) // self.world_size

    def set_epoch(self, epoch: int) -> None:
        """Call at the start of each epoch.

        Re-seeds the shared RNG deterministically so each epoch shuffles
        differently but all ranks stay synchronized. Idempotent: calling
        set_epoch(N) twice produces the same batch list.
        """
        self._current_epoch = epoch
        self._rng = random.Random(self.seed + epoch)

    def skip(self, n_batches: int) -> None:
        """Skip the first n_batches of the next __iter__ call (mid-epoch resume).

        Must be called AFTER set_epoch(). Cleared automatically once __iter__
        consumes it, so subsequent epochs are full.
        """
        self._skip_batches = max(0, int(n_batches))

    @property
    def current_epoch(self) -> int:
        return self._current_epoch