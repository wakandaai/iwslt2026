"""
Duration-based bucket sampler with synchronized batches for DDP.

Groups audio samples by duration into buckets, then builds batches that cap
total audio seconds per batch (not instance count). Prevents OOM from
variable-length sequences while keeping padding overhead low.

Synchronized batches across ranks (Zelasko et al. 2025, ASRU), adapted from
facebookresearch-style wakandaai/speechaura src/core/sampler.py:
  - All ranks use the SAME shared RNG to shuffle samples within buckets
    and to shuffle bucket order -> all ranks build an IDENTICAL global
    batch list.
  - Each rank then takes batches[rank::world_size] as its slice.

Requires the dataset to expose a .durations list (pre-extracted from CSV).
"""

from __future__ import annotations

import random
from typing import Iterator

from torch.utils.data import Sampler


class DurationBucketSampler(Sampler):
    """Batch sampler that caps total audio duration per batch.

    Args:
        dataset:              Dataset with a `.durations` attribute (list[float]).
        target_duration:      Max total audio seconds per batch.
        max_batch_size:       Hard cap on instances per batch (prevents huge
                              batches from many short utterances).
        bucket_width:         Multiplier — samples within a bucket have durations
                              spanning at most [min, min * bucket_width].
        shuffle:              Shuffle within each bucket before batching.
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

        self.durations: list[float] = list(dataset.durations)

        # Single shared RNG — same seed on all ranks → all ranks build the
        # same global batch list, then slice by rank.
        self._rng = random.Random(seed)
        self._current_epoch = 0
        self._skip_batches = 0  # mid-epoch resume: batches to skip in next __iter__

        self._buckets = self._make_buckets()
        self._epoch_batches = self._build_epoch_batches()

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
                # Flush if either cap would be exceeded
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

    def _build_epoch_batches(self) -> list[list[int]]:
        """Build this rank's batch list for the current RNG state.

        Called once per set_epoch — NOT from __iter__, so that iterating twice
        without an intervening set_epoch replays the same batches rather than
        advancing the shared RNG and silently desyncing the ranks.
        """
        buckets = [b.copy() for b in self._buckets]
        if self.shuffle:
            for b in buckets:
                self._rng.shuffle(b)

        batches = self._make_batches(buckets)

        if self.shuffle_buckets:
            self._rng.shuffle(batches)

        # Slice for this rank, truncated to the count every rank is guaranteed
        # to have (batches[rank::world_size] would hand low ranks an extra
        # batch whenever the total isn't divisible by world_size, and a rank
        # running ahead alone would block forever on the next collective).
        per_rank = len(batches) // self.world_size
        return batches[self.rank::self.world_size][:per_rank]

    def __iter__(self) -> Iterator[list[int]]:
        batches = self._epoch_batches

        if self._skip_batches > 0:
            batches = batches[self._skip_batches:]
            self._skip_batches = 0

        yield from batches

    def __len__(self) -> int:
        return len(self._epoch_batches)

    def set_epoch(self, epoch: int) -> None:
        """Call at the start of each epoch. Re-seeds the shared RNG
        deterministically so each epoch shuffles differently but all ranks
        stay synchronized. Idempotent: calling set_epoch(N) twice produces
        the same batch list."""
        self._current_epoch = epoch
        self._rng = random.Random(self.seed + epoch)
        self._epoch_batches = self._build_epoch_batches()

    def skip(self, n_batches: int) -> None:
        """Skip the first n_batches of the next __iter__ call (mid-epoch
        resume). Must be called AFTER set_epoch(). Cleared automatically
        once __iter__ consumes it."""
        self._skip_batches = max(0, int(n_batches))

    @property
    def current_epoch(self) -> int:
        return self._current_epoch


class _PartitionCyclingSampler(Sampler):
    """Shared machinery for weighted-partition batch samplers, DDP-safe.

    Duration-buckets + caps-into-batches each partition independently, then
    draws `num_batches` batches (global count, before rank-slicing) by
    repeatedly choosing a partition according to precomputed weights and
    cycling through that partition's own batch queue (reshuffling on
    exhaustion) — so small/underweighted-then-boosted partitions get
    revisited many times over a long fixed-step run, proportional to their
    weight rather than their raw size.

    The FULL global sequence of `num_batches` draws is built once,
    deterministically, from the shared seed — every rank runs the identical
    construction (same partitions, same weights, same RNG calls in the same
    order), so slicing `[rank::world_size]` afterward is DDP-safe with no
    inter-rank communication needed, matching DurationBucketSampler's pattern.

    Subclasses build `self.partition_indices` (dict[key, list[int]]) and
    `self.partition_weight` (dict[key, float], summing to ~1) in their own
    `__init__`, using whatever weighting scheme they need, then call
    `self._finalize()`. `key` can be any hashable partition identifier
    (e.g. a language string, or a (source, language) tuple).
    """

    def _finalize(self) -> None:
        self.partitions: list = list(self.partition_indices.keys())
        assert self.partitions, f"{type(self).__name__}: no partitions found"
        self._weights_list = [self.partition_weight[k] for k in self.partitions]
        self._partition_queue: dict = {
            key: self._make_batches(idxs) for key, idxs in self.partition_indices.items()
        }
        self._partition_cursor: dict = {key: 0 for key in self.partitions}

        # Precompute the full, deterministic draw sequence once. Every rank
        # calls this identically (same seed, same partitions => same RNG call
        # order), so the result is guaranteed identical across ranks.
        self._all_batches: list[list[int]] = [
            self._draw_next() for _ in range(self.num_batches)
        ]
        per_rank = len(self._all_batches) // self.world_size
        self._rank_batches = self._all_batches[self.rank::self.world_size][:per_rank]

    @staticmethod
    def _normalized_power_weights(hours_by_key: dict, beta: float) -> dict:
        """weight(key) = (hours[key] / total_hours) ** beta, renormalized to sum to 1."""
        total = sum(hours_by_key.values())
        raw = {k: (h / total) ** beta for k, h in hours_by_key.items()}
        raw_sum = sum(raw.values())
        return {k: w / raw_sum for k, w in raw.items()}

    def _make_batches(self, idxs: list[int]) -> list[list[int]]:
        """Duration-bucket + cap-batch a single partition's indices (same
        algorithm as DurationBucketSampler, scoped to one partition)."""
        indexed = sorted(idxs, key=lambda i: self.durations[i])
        buckets: list[list[int]] = []
        current: list[int]       = []
        bucket_min: float | None = None

        for i in indexed:
            dur = self.durations[i]
            if bucket_min is None:
                current, bucket_min = [i], dur
            elif dur <= bucket_min * self.bucket_width:
                current.append(i)
            else:
                buckets.append(current)
                current, bucket_min = [i], dur
        if current:
            buckets.append(current)

        batches: list[list[int]] = []
        for bucket in buckets:
            batch: list[int] = []
            batch_dur = 0.0
            for i in bucket:
                dur = self.durations[i]
                if batch and (
                    batch_dur + dur > self.target_duration
                    or len(batch) >= self.max_batch_size
                ):
                    batches.append(batch)
                    batch, batch_dur = [], 0.0
                batch.append(i)
                batch_dur += dur
            if batch:
                batches.append(batch)

        self._rng.shuffle(batches)
        return batches

    def _next_batch(self, key) -> list[int]:
        queue  = self._partition_queue[key]
        cursor = self._partition_cursor[key]
        if cursor >= len(queue):
            self._rng.shuffle(queue)
            cursor = 0
        batch = queue[cursor]
        self._partition_cursor[key] = cursor + 1
        return batch

    def _draw_next(self) -> list[int]:
        key = self._rng.choices(self.partitions, weights=self._weights_list, k=1)[0]
        return self._next_batch(key)

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._rank_batches

    def __len__(self) -> int:
        return len(self._rank_batches)


class WeightedPartitionSampler(_PartitionCyclingSampler):
    """Batch sampler with two-level (corpus-then-language) temperature-weighted
    partition sampling, matching Meta's omnilingual_asr mixture_parquet_storage
    algorithm (facebookresearch/omnilingual-asr,
    src/omnilingual_asr/datasets/storage/mixture_parquet_storage.py):

        partition       := (source, language) pair (their "corpus" == our "source")
        corpus_weight   := normalize( (corpus_hours / total_hours) ** beta_corpus )
        language_weight := normalize( (lang_hours_in_corpus / corpus_hours) ** beta_language )
                           — computed *within* each corpus separately
        partition_weight = corpus_weight[corpus] * language_weight[corpus, lang]

    beta=1.0 reproduces plain proportional-to-data-size sampling (no reweighting).
    beta=0.0 is fully uniform regardless of size.

    CAVEAT (found empirically on our data): this two-level hierarchy can
    misbehave when a target language is concentrated in one dominant corpus
    AND happens to be the *largest* language within that corpus — within-corpus
    flattening then pulls its share *down*, not up, even though it's globally
    low-resource. E.g. Amharic/Tigrinya are ~entirely WAXAL-sourced and are
    WAXAL's two largest languages, so beta_language<1 penalized them rather
    than boosting them. See WeightedLanguageSampler for a single-level
    alternative that avoids this by reweighting directly against global share.

    Args:
        dataset:         Dataset with `.entries` (list[dict] with "source" and
                         "language"/"src_language" keys) and `.durations`.
        beta_corpus:     Temperature exponent for corpus-level weighting.
        beta_language:   Temperature exponent for language-level weighting
                         (applied within each corpus).
        target_duration: Max total audio seconds per batch.
        max_batch_size:  Hard cap on instances per batch.
        bucket_width:    Same semantics as DurationBucketSampler.
        num_batches:     Total GLOBAL batches (before rank-slicing) this
                         sampler draws — sized well above
                         max_steps * grad_accum * world_size so a real run
                         never needs to cycle to a second "epoch" of this
                         sampler. Per-rank count is num_batches // world_size.
        rank:            DDP rank (0 for single GPU).
        world_size:      DDP world size (1 for single GPU).
        seed:            Shared seed — must be identical on all ranks.
    """

    def __init__(
        self,
        dataset,
        beta_corpus: float,
        beta_language: float,
        target_duration: float = 40.0,
        max_batch_size: int = 4,
        bucket_width: float = 1.5,
        num_batches: int = 200_000,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
    ):
        self.dataset          = dataset
        self.durations        = dataset.durations
        self.target_duration  = target_duration
        self.max_batch_size   = max_batch_size
        self.bucket_width     = bucket_width
        self.num_batches      = num_batches
        self.rank             = rank
        self.world_size       = world_size
        self._rng             = random.Random(seed)

        # --- 1. Partition indices by (source, language) ---
        partition_indices: dict[tuple[str, str], list[int]] = {}
        for idx, entry in enumerate(dataset.entries):
            source = entry.get("source", "")
            lang   = entry.get("language") or entry.get("src_language") or ""
            partition_indices.setdefault((source, lang), []).append(idx)

        assert partition_indices, "WeightedPartitionSampler: dataset has no entries"

        # --- 2. Hours per partition, per corpus ---
        partition_hours: dict[tuple[str, str], float] = {
            key: sum(self.durations[i] for i in idxs) / 3600.0
            for key, idxs in partition_indices.items()
        }
        corpus_hours: dict[str, float] = {}
        for (source, _lang), hours in partition_hours.items():
            corpus_hours[source] = corpus_hours.get(source, 0.0) + hours
        total_hours = sum(corpus_hours.values())
        assert total_hours > 0, "WeightedPartitionSampler: total hours is zero"

        # --- 3. Corpus-level weights ---
        corpus_weight = self._normalized_power_weights(corpus_hours, beta_corpus)

        # --- 4. Language-level weights, computed within each corpus ---
        lang_weight: dict[tuple[str, str], float] = {}
        for source in corpus_hours:
            langs_in_corpus = {
                lang: hours
                for (s, lang), hours in partition_hours.items()
                if s == source
            }
            normalized = self._normalized_power_weights(langs_in_corpus, beta_language)
            for lang, w in normalized.items():
                lang_weight[(source, lang)] = w

        # --- 5. Final partition weight = corpus_weight * language_weight ---
        # (Sums to 1 automatically: sum_c corpus_weight[c] * sum_{l in c} lang_weight[c,l]
        #  = sum_c corpus_weight[c] * 1 = 1 — same property as the reference implementation.)
        self.partition_indices = partition_indices
        self.partition_weight: dict[tuple[str, str], float] = {
            key: corpus_weight[key[0]] * lang_weight[key] for key in partition_indices
        }
        self._finalize()


class WeightedLanguageSampler(_PartitionCyclingSampler):
    """Single-level language-temperature weighted batch sampler — reweights
    each language against its GLOBAL share of the dataset directly, with no
    corpus/source dimension at all:

        weight(lang) = normalize( (lang_hours / total_hours) ** beta_language )

    Use this instead of WeightedPartitionSampler when target languages are
    concentrated in one dominant corpus — see the caveat on that class for
    why the two-level hierarchy can otherwise penalize exactly the languages
    you're trying to boost.

    Args:
        dataset:         Dataset with `.entries` (list[dict] with "language"/
                         "src_language" keys) and `.durations`.
        beta_language:   Temperature exponent. 1.0 = proportional to actual
                         data size (no reweighting). 0.0 = fully uniform
                         across languages regardless of size. 0.5-0.8 is the
                         typical middle ground.
        target_duration: Max total audio seconds per batch.
        max_batch_size:  Hard cap on instances per batch.
        bucket_width:    Same semantics as DurationBucketSampler.
        num_batches:     Total GLOBAL batches (before rank-slicing).
        rank:            DDP rank (0 for single GPU).
        world_size:      DDP world size (1 for single GPU).
        seed:            Shared seed — must be identical on all ranks.
    """

    def __init__(
        self,
        dataset,
        beta_language: float,
        target_duration: float = 40.0,
        max_batch_size: int = 4,
        bucket_width: float = 1.5,
        num_batches: int = 200_000,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
    ):
        self.dataset          = dataset
        self.durations        = dataset.durations
        self.target_duration  = target_duration
        self.max_batch_size   = max_batch_size
        self.bucket_width     = bucket_width
        self.num_batches      = num_batches
        self.rank             = rank
        self.world_size       = world_size
        self._rng             = random.Random(seed)

        partition_indices: dict[str, list[int]] = {}
        for idx, entry in enumerate(dataset.entries):
            lang = entry.get("language") or entry.get("src_language") or ""
            partition_indices.setdefault(lang, []).append(idx)

        assert partition_indices, "WeightedLanguageSampler: dataset has no entries"

        lang_hours: dict[str, float] = {
            lang: sum(self.durations[i] for i in idxs) / 3600.0
            for lang, idxs in partition_indices.items()
        }
        total_hours = sum(lang_hours.values())
        assert total_hours > 0, "WeightedLanguageSampler: total hours is zero"

        self.partition_indices = partition_indices
        self.partition_weight: dict[str, float] = self._normalized_power_weights(
            lang_hours, beta_language
        )
        self._finalize()
