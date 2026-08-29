"""Group-aware train/validation/test splitting."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def _distribution_score(labels: np.ndarray, indices: np.ndarray, target: np.ndarray) -> float:
    counts = np.bincount(labels[indices], minlength=len(target)).astype(np.float64)
    distribution = counts / max(counts.sum(), 1.0)
    return float(np.abs(distribution - target).mean())


def _best_group_split(
    indices: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    test_size: float,
    seed: int,
    candidates: int,
) -> Tuple[np.ndarray, np.ndarray]:
    target = np.bincount(labels[indices], minlength=int(labels.max()) + 1).astype(np.float64)
    target /= target.sum()
    best_score = float("inf")
    best_split = None

    for offset in range(candidates):
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed + offset)
        relative_train, relative_test = next(
            splitter.split(indices, labels[indices], groups[indices])
        )
        train_indices = indices[relative_train]
        test_indices = indices[relative_test]
        score = _distribution_score(labels, train_indices, target) + _distribution_score(
            labels, test_indices, target
        )
        if score < best_score:
            best_score = score
            best_split = (train_indices, test_indices)

    if best_split is None:
        raise RuntimeError("Unable to create a group-aware split")
    return best_split


def create_grouped_splits(
    labels: np.ndarray,
    groups: np.ndarray,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
    candidates: int = 64,
) -> Dict[str, np.ndarray]:
    total = train_fraction + val_fraction + test_fraction
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split fractions must sum to 1.0, got {total}")
    if len(labels) != len(groups):
        raise ValueError("labels and groups must have the same length")
    if len(np.unique(groups)) < 3:
        raise ValueError("At least three unique groups are required")

    all_indices = np.arange(len(labels), dtype=np.int64)
    train_indices, remainder = _best_group_split(
        all_indices,
        labels,
        groups,
        test_size=val_fraction + test_fraction,
        seed=seed,
        candidates=candidates,
    )
    remainder_test_fraction = test_fraction / (val_fraction + test_fraction)
    val_indices, test_indices = _best_group_split(
        remainder,
        labels,
        groups,
        test_size=remainder_test_fraction,
        seed=seed + candidates,
        candidates=candidates,
    )
    return {
        "train": np.sort(train_indices),
        "val": np.sort(val_indices),
        "test": np.sort(test_indices),
    }
