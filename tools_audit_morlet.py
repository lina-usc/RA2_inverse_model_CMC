from __future__ import annotations
import glob
import os
import numpy as np
import h5py

def show(name, arr):
    arr = np.asarray(arr)
    print(f"\n[{name}]")
    print("shape:", arr.shape, "dtype:", arr.dtype)
    if arr.size == 0:
        print("EMPTY")
        return
    if np.issubdtype(arr.dtype, np.number):
        print("min/max:", float(np.nanmin(arr)), float(np.nanmax(arr)))
        if arr.ndim >= 2:
            print("std per col:", arr.std(axis=0))
            print("first 5 rows:")
            print(arr[:5])
        else:
            print("std:", float(np.nanstd(arr)))
            print("head:", arr[:10])
    else:
        print(arr[:10])

print("=== H5 THETA AUDIT ===")
h5_path = "data/synthetic_cmc_dataset.h5"
print("exists:", os.path.isfile(h5_path), h5_path)
if os.path.isfile(h5_path):
    with h5py.File(h5_path, "r") as f:
        print("keys:", list(f.keys()))
        if "theta" in f:
            th = np.asarray(f["theta"][:2048], dtype=np.float32)
            show("H5 theta probe", th)
        if "param_names" in f:
            print("\n[param_names]")
            print(f["param_names"][:])

print("\n=== PARAMS.NPY AUDIT ===")
params_path = "data_out_morlet/params.npy"
print("exists:", os.path.isfile(params_path), params_path)
if os.path.isfile(params_path):
    y = np.load(params_path, mmap_mode="r")
    show("params.npy probe", np.asarray(y[:2048], dtype=np.float32))

print("\n=== SPLIT AUDIT ===")
cand = []
cand += ["data_out_morlet/splits.npz"]
cand += sorted(glob.glob("models_out_morlet/*/split_indices_used.npz"))[:6]

for p in cand:
    print("\nfile:", p, "exists:", os.path.isfile(p))
    if not os.path.isfile(p):
        continue
    z = np.load(p)
    print("keys:", list(z.keys()))
    for k in ("train_idx", "idx_train", "val_idx", "idx_val", "test_idx", "idx_test"):
        if k in z:
            a = np.asarray(z[k], dtype=np.int64)
            print(
                k,
                "shape=", a.shape,
                "min=", int(a.min()) if a.size else None,
                "max=", int(a.max()) if a.size else None,
                "n_unique=", int(np.unique(a).size),
            )
            print("head:", a[:10])

print("\n=== DIRECT ZERO-ROW CHECK ===")
if os.path.isfile(h5_path):
    with h5py.File(h5_path, "r") as f:
        th = f["theta"]
        zero_rows = np.where(np.all(np.asarray(th[:5000], dtype=np.float32) == 0.0, axis=1))[0]
        print("zero rows in first 5000 H5 rows:", zero_rows[:20], "count=", int(len(zero_rows)))
if os.path.isfile(params_path):
    y = np.load(params_path, mmap_mode="r")
    zero_rows = np.where(np.all(np.asarray(y[:5000], dtype=np.float32) == 0.0, axis=1))[0]
    print("zero rows in first 5000 params.npy rows:", zero_rows[:20], "count=", int(len(zero_rows)))
