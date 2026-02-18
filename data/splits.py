from __future__ import annotations

import os
from typing import Dict

import numpy as np


def ensure_splits(
    data_out_dir: str,
    n_samples: int,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    overwrite: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Deterministic train/val/test split indices saved to data_out/splits.npz.
    Use this SAME splits file for ERP vs TFR vs hybrid comparisons.
    """
    path = os.path.join(data_out_dir, "splits.npz")

    if os.path.exists(path) and not overwrite:
        d = np.load(path)
        return {
            "train_idx": d["train_idx"],
            "val_idx": d["val_idx"],
            "test_idx": d["test_idx"],
        }

    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    n = int(n_samples)
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)

    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    n_test = n - n_train - n_val

    train_idx = np.sort(perm[:n_train])
    val_idx = np.sort(perm[n_train:n_train + n_val])
    test_idx = np.sort(perm[n_train + n_val:])

    assert len(test_idx) == n_test

    np.savez(
        path,
        train_idx=train_idx.astype(np.int32),
        val_idx=val_idx.astype(np.int32),
        test_idx=test_idx.astype(np.int32),
        seed=np.array(int(seed), dtype=np.int32),
        train_frac=np.array(train_frac, dtype=np.float32),
        val_frac=np.array(val_frac, dtype=np.float32),
        test_frac=np.array(test_frac, dtype=np.float32),
        n_samples=np.array(n, dtype=np.int32),
    )

    return {"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx}
