from __future__ import annotations

import argparse
import os
import numpy as np

from data.splits import ensure_splits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-out", type=str, default="data_out")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    params_path = os.path.join(args.data_out, "params.npy")
    if not os.path.exists(params_path):
        raise SystemExit(f"ERROR: {params_path} not found. Run data.prepare_training_data first.")

    y = np.load(params_path, mmap_mode="r")
    n = int(y.shape[0])

    splits = ensure_splits(
        data_out_dir=args.data_out,
        n_samples=n,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        overwrite=args.overwrite,
    )

    print("[make_splits] DONE")
    print(f"  n_samples={n}")
    print(f"  train={len(splits['train_idx'])} val={len(splits['val_idx'])} test={len(splits['test_idx'])}")
    print(f"  wrote: {os.path.join(args.data_out, 'splits.npz')}")


if __name__ == "__main__":
    main()
