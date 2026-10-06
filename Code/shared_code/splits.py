"""Deterministic train and test splits, shared by every stage.

The same dataset sample always yields the same splits, so Stage 1 probes and later
fusion runs can be evaluated on identical rows.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def grouped_folds(groups, n_splits=5, seed=42):
    """Return (train, test) row indices with each group in exactly one held-out fold.

    A group's fold is the SHA-256 hash of the seed and its string, modulo n_splits, so the
    assignment is the same on every machine and library version, identical sequences
    always share a fold, and a sequence keeps its fold in any sample that contains it.
    These are the folds probe_scores uses when it receives groups.
    """
    fold_of_row = np.array([
        int(hashlib.sha256(f"{seed}:{g}".encode()).hexdigest(), 16) % n_splits
        for g in groups
    ])
    folds = [
        (np.flatnonzero(fold_of_row != k), np.flatnonzero(fold_of_row == k))
        for k in range(n_splits)
    ]
    if any(len(test) == 0 for _, test in folds):
        raise ValueError("A fold received no rows; use more groups or fewer folds.")
    return folds


def grouped_holdout_split(groups, test_fraction=0.3, seed=42):
    """Return (train, test) row indices with every copy of a group on the same side.

    A group goes to the test side when the SHA-256 hash of the seed and its string,
    modulo 1,000, falls below test_fraction * 1,000. As with grouped_folds, the split is
    the same on every machine, identical sequences never straddle it, and the test share
    is close to, not exactly, test_fraction. The CCA and retrieval analyses use it.
    """
    cutoff = round(test_fraction * 1000)
    is_test = np.array([
        int(hashlib.sha256(f"{seed}:holdout:{g}".encode()).hexdigest(), 16) % 1000 < cutoff
        for g in groups
    ])
    return np.flatnonzero(~is_test), np.flatnonzero(is_test)


def official_split(dataset):
    """Return {split name: row indices} from the dataset's published split column.

    Use these for comparisons with published results. In mRNA_Stability.csv thousands
    of test sequences also occur in training, so they overstate generalization.
    """
    column = dataset.spec.split_column
    if column is None:
        raise ValueError(f"{dataset.spec.key} has no published split column")
    values = dataset.rows[column].to_numpy()
    return {name: np.flatnonzero(values == name) for name in pd.unique(values)}
