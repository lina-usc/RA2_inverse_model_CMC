#!/usr/bin/env python3
"""
NeurIPS checklist upgrades: SNPE (flow posterior) baselines + mismatch eval,
built to work with YOUR existing repo structure.

Key design:
  - No configs/prior_bounds.json needed.
  - No CMC_SIMULATOR needed.
  - Reads theta + eeg from your existing H5 dataset (data/synthetic_cmc_dataset.h5).
  - Uses your canonical tokenizer: data/feature_tokens.py (TokenConfig + ERP/TFR).
  - Uses your existing split indices from data_out (seed) if found; else makes deterministic splits.

Outputs (default under results/neurips_sbi):
  results/neurips_sbi/
    snpe_erp_t25_f15/
    snpe_tfr_t25_f15/
    snpe_hybrid_t25_f15/
      posterior_snpe.pt
      train_meta.json
      eval_clean/
      eval_mismatch/

Run:
  source .venv_sbi/bin/activate
  python scripts/neurips_upgrades.py all --h5 data/synthetic_cmc_dataset.h5 --data-out data_out --split-seed 42 --outdir results/neurips_sbi --device cpu

Device notes:
  - On Mac: use --device mps (if available) or cpu.
  - On CUDA machine: use --device cuda.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Ensure repo root is on sys.path so `import data.feature_tokens` works no matter how you run this script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PARAM_NAMES = ["tau_e", "tau_i", "g", "p0", "stim_amp", "w_ei", "w_ie", "w_ff", "w_fb"]


# -------------------------
# Small utilities
# -------------------------
def mkdirp(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def write_json(path: str | Path, obj: Any) -> None:
    Path(path).write_text(json.dumps(obj, indent=2))

def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())

def _device_resolve(device: str) -> str:
    import torch
    d = device.lower().strip()
    if d == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        print("[WARN] --device cuda requested but torch.cuda.is_available() is False. Falling back to cpu.")
        return "cpu"
    if d == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        print("[WARN] --device mps requested but not available. Falling back to cpu.")
        return "cpu"
    return "cpu"

def _require_sbi():
    try:
        import torch  # noqa
        import sbi  # noqa
        import h5py  # noqa
        return
    except Exception as e:
        raise RuntimeError(
            "Missing deps for SBI baseline. Activate .venv_sbi and install:\n"
            "  pip install torch sbi nflows pyro-ppl h5py scipy numpy matplotlib\n"
            f"Original error: {repr(e)}"
        )


# -------------------------
# H5 loading (auto-detect eeg/theta datasets)
# -------------------------
def _iter_h5_datasets(g, prefix=""):
    import h5py
    for k, v in g.items():
        path = f"{prefix}/{k}" if prefix else k
        if isinstance(v, h5py.Dataset):
            yield path, v
        elif isinstance(v, h5py.Group):
            yield from _iter_h5_datasets(v, path)

def _pick_best(paths: List[str], prefer_substrings: List[str]) -> str:
    # Prefer paths that contain certain substrings like "theta" or "eeg"
    lower = [p.lower() for p in paths]
    for sub in prefer_substrings:
        for i, p in enumerate(lower):
            if sub in p:
                return paths[i]
    return paths[0]

def detect_h5_keys(h5_path: Path) -> Tuple[str, str]:
    """
    Returns (theta_key, eeg_key) by searching datasets with shapes:
      theta: (N,9) or (9,N)
      eeg:   (N,16,500-ish) or (N,500-ish,16)
    """
    import h5py

    with h5py.File(h5_path, "r") as f:
        theta_candidates: List[str] = []
        eeg_candidates: List[str] = []

        for path, ds in _iter_h5_datasets(f):
            shape = ds.shape
            if len(shape) == 2 and (shape[1] == 9 or shape[0] == 9):
                theta_candidates.append(path)
            if len(shape) == 3:
                # common layouts:
                # (N,16,500), (N,500,16), allow >=500 and slice later
                if (shape[1] == 16 and shape[2] >= 500) or (shape[2] == 16 and shape[1] >= 500):
                    eeg_candidates.append(path)

        if not theta_candidates or not eeg_candidates:
            # Print available datasets for debugging
            all_ds = [(p, d.shape) for p, d in _iter_h5_datasets(f)]
            msg = "Could not auto-detect theta/eeg datasets in H5.\n\nAvailable datasets:\n"
            msg += "\n".join([f"  - {p}: shape={s}" for p, s in all_ds])
            msg += "\n\nFix: rerun with --theta-key and --eeg-key explicitly."
            raise RuntimeError(msg)

        theta_key = _pick_best(theta_candidates, ["theta", "param", "params", "parameter"])
        eeg_key = _pick_best(eeg_candidates, ["eeg", "x", "data", "signal"])
        return theta_key, eeg_key

def load_h5_arrays(
    h5_path: Path,
    theta_key: Optional[str],
    eeg_key: Optional[str],
    max_n: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      theta: (N,9) float32
      eeg:   (N,16,500) float32
    """
    import h5py

    if theta_key is None or eeg_key is None:
        auto_theta, auto_eeg = detect_h5_keys(h5_path)
        theta_key = theta_key or auto_theta
        eeg_key = eeg_key or auto_eeg
        print(f"[h5] auto-detected theta_key='{theta_key}', eeg_key='{eeg_key}'")

    with h5py.File(h5_path, "r") as f:
        th = f[theta_key]
        eg = f[eeg_key]

        # theta: (N,9) or (9,N)
        theta = np.array(th[:], dtype=np.float32)
        if theta.ndim != 2:
            raise RuntimeError(f"theta dataset {theta_key} has shape {theta.shape}, expected 2D.")
        if theta.shape[1] == 9:
            pass
        elif theta.shape[0] == 9:
            theta = theta.T
        else:
            raise RuntimeError(f"theta dataset {theta_key} has shape {theta.shape}, expected (N,9) or (9,N).")

        # eeg: (N,16,T) or (N,T,16)
        eeg = np.array(eg[:], dtype=np.float32)
        if eeg.ndim != 3:
            raise RuntimeError(f"eeg dataset {eeg_key} has shape {eeg.shape}, expected 3D.")
        # transpose if needed
        if eeg.shape[1] == 16:
            pass  # (N,16,T)
        elif eeg.shape[2] == 16:
            eeg = np.transpose(eeg, (0, 2, 1))  # (N,16,T)
        else:
            raise RuntimeError(f"eeg dataset {eeg_key} has shape {eeg.shape}, expected channel dim=16.")

        # ensure 500 samples
        if eeg.shape[2] < 500:
            raise RuntimeError(f"eeg time dimension is {eeg.shape[2]} < 500; expected 500 samples.")
        if eeg.shape[2] != 500:
            eeg = eeg[:, :, :500]

        if max_n is not None:
            theta = theta[:max_n]
            eeg = eeg[:max_n]

        if theta.shape[0] != eeg.shape[0]:
            raise RuntimeError(f"theta N={theta.shape[0]} and eeg N={eeg.shape[0]} mismatch.")

        return theta.astype(np.float32), eeg.astype(np.float32)


# -------------------------
# Splits: load from data_out if possible; else make deterministic
# -------------------------
def find_splits_file(data_out: Path, split_seed: int) -> Optional[Path]:
    candidates = [
        data_out / f"splits_seed{split_seed}.npz",
        data_out / f"splits_{split_seed}.npz",
        data_out / f"splits_seed{split_seed}.json",
        data_out / f"splits_{split_seed}.json",
        data_out / "splits.npz",
        data_out / "splits.json",
    ]
    for c in candidates:
        if c.exists():
            return c

    # fallback search: anything with "split" and the seed in filename
    hits = list(data_out.glob(f"*split*{split_seed}*"))
    if hits:
        return hits[0]
    hits = list(data_out.glob("*split*"))
    if hits:
        # better than nothing; but warn
        print(f"[WARN] Found splits-like file(s) but not seed-specific. Using: {hits[0]}")
        return hits[0]

    return None

def load_or_make_splits(data_out: Path, split_seed: int, N: int) -> Dict[str, np.ndarray]:
    f = find_splits_file(data_out, split_seed)
    if f is None:
        print("[WARN] No splits file found in data_out; making deterministic 80/10/10 split.")
        rng = np.random.default_rng(split_seed)
        perm = rng.permutation(N)
        n_train = int(0.8 * N)
        n_val = int(0.1 * N)
        train_idx = perm[:n_train]
        val_idx = perm[n_train:n_train+n_val]
        test_idx = perm[n_train+n_val:]
        return {"train": train_idx, "val": val_idx, "test": test_idx}

    if f.suffix == ".npz":
        z = np.load(f, allow_pickle=True)
        keys = set(z.files)
        # try common key conventions
        for trio in [("train_idx","val_idx","test_idx"), ("train","val","test"), ("idx_train","idx_val","idx_test")]:
            if all(k in keys for k in trio):
                train_idx = np.array(z[trio[0]], dtype=np.int64)
                val_idx = np.array(z[trio[1]], dtype=np.int64)
                test_idx = np.array(z[trio[2]], dtype=np.int64)
                print(f"[splits] loaded {f} keys={trio}")
                return {"train": train_idx, "val": val_idx, "test": test_idx}
        raise RuntimeError(f"Splits npz {f} has keys {z.files} but no recognized (train,val,test) keys.")

    if f.suffix == ".json":
        d = read_json(f)
        # try common keys
        for trio in [("train","val","test"), ("train_idx","val_idx","test_idx")]:
            if all(k in d for k in trio):
                train_idx = np.array(d[trio[0]], dtype=np.int64)
                val_idx = np.array(d[trio[1]], dtype=np.int64)
                test_idx = np.array(d[trio[2]], dtype=np.int64)
                print(f"[splits] loaded {f} keys={trio}")
                return {"train": train_idx, "val": val_idx, "test": test_idx}
        raise RuntimeError(f"Splits json {f} has keys {list(d.keys())} but no recognized (train,val,test) keys.")

    raise RuntimeError(f"Unsupported splits file type: {f}")


# -------------------------
# Feature computation (uses YOUR canonical code)
# -------------------------
def make_token_cfg(n_time_patches: int, n_freq_patches: int):
    from data.feature_tokens import TokenConfig
    return TokenConfig(n_time_patches=n_time_patches, n_freq_patches=n_freq_patches)

def featurize_one(eeg_16_500: np.ndarray, feature_set: str, token_cfg) -> np.ndarray:
    from data.feature_tokens import compute_erp_tokens, compute_tfr_tokens

    if feature_set == "erp":
        return compute_erp_tokens(eeg_16_500, token_cfg).astype(np.float32)
    if feature_set == "tfr":
        tok, _meta = compute_tfr_tokens(eeg_16_500, token_cfg)
        return tok.astype(np.float32)
    if feature_set == "hybrid":
        erp = compute_erp_tokens(eeg_16_500, token_cfg).astype(np.float32)
        tfr, _meta = compute_tfr_tokens(eeg_16_500, token_cfg)
        return np.concatenate([erp, tfr.astype(np.float32)], axis=0).astype(np.float32)
    raise ValueError(feature_set)

def featurize_split(eeg: np.ndarray, idx: np.ndarray, feature_set: str, token_cfg) -> np.ndarray:
    # Determine length from first sample
    tok0 = featurize_one(eeg[int(idx[0])], feature_set, token_cfg)
    L = tok0.shape[0]
    X = np.zeros((len(idx), L, 16), dtype=np.float32)
    for i, k in enumerate(idx):
        X[i] = featurize_one(eeg[int(k)], feature_set, token_cfg)
    return X

def flatten_tokens(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1).astype(np.float32)

def zscore_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = x.mean(axis=0).astype(np.float32)
    sd = x.std(axis=0).astype(np.float32)
    sd = np.maximum(sd, 1e-6)
    return mu, sd

def zscore_apply(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return ((x - mu[None, :]) / sd[None, :]).astype(np.float32)


# -------------------------
# Metrics + calibration plots
# -------------------------
def pearsonr_per_param(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    P = y_true.shape[1]
    r = np.zeros((P,), dtype=np.float32)
    for p in range(P):
        a = y_true[:, p] - y_true[:, p].mean()
        b = y_pred[:, p] - y_pred[:, p].mean()
        denom = (np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()) + 1e-12)
        r[p] = float((a * b).sum() / denom)
    return r

def rmse_norm_per_param(y_true: np.ndarray, y_pred: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0)).astype(np.float32)
    scale = (high - low).astype(np.float32)
    return (rmse / (scale + 1e-12)).astype(np.float32)

def coverage_from_samples(theta_true: np.ndarray, samples: np.ndarray, levels: np.ndarray) -> np.ndarray:
    N, S, P = samples.shape
    cov = np.zeros((P, len(levels)), dtype=np.float32)
    for ki, c in enumerate(levels):
        lo_q = 0.5 - 0.5 * c
        hi_q = 0.5 + 0.5 * c
        lo = np.quantile(samples, lo_q, axis=1)
        hi = np.quantile(samples, hi_q, axis=1)
        inside = (theta_true >= lo) & (theta_true <= hi)
        cov[:, ki] = inside.mean(axis=0).astype(np.float32)
    return cov

def sbc_hist(theta_true: np.ndarray, samples: np.ndarray) -> np.ndarray:
    N, S, P = samples.shape
    hist = np.zeros((P, S + 1), dtype=np.int32)
    for p in range(P):
        ranks = (samples[:, :, p] < theta_true[:, None, p]).sum(axis=1).astype(np.int32)
        hist[p] = np.bincount(ranks, minlength=S + 1)
    return hist

def plot_coverage(levels: np.ndarray, cov: np.ndarray, outpath: Path) -> None:
    plt.figure()
    for p in range(cov.shape[0]):
        plt.plot(levels, cov[p], label=PARAM_NAMES[p])
    plt.plot(levels, levels, linestyle="--", linewidth=1.5, label="ideal")
    plt.xlabel("Credible mass")
    plt.ylabel("Empirical coverage")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_sbc(hist: np.ndarray, outpath: Path, max_params: int = 9) -> None:
    P, B = hist.shape
    cols = 3
    rows = int(np.ceil(min(P, max_params) / cols))
    plt.figure(figsize=(10, 2.8 * rows))
    for p in range(min(P, max_params)):
        ax = plt.subplot(rows, cols, p + 1)
        ax.bar(np.arange(B), hist[p])
        ax.set_title(PARAM_NAMES[p], fontsize=10)
        ax.set_xlabel("rank")
        ax.set_ylabel("count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# -------------------------
# Mismatch transforms (leadfield / noise PSD / stimulus jitter)
# -------------------------
def _colored_noise(beta: float, shape: Tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    n_channels, T = shape
    freqs = np.fft.rfftfreq(T, d=1.0)
    if len(freqs) > 1:
        freqs[0] = freqs[1]
    mag = 1.0 / (freqs ** (beta / 2.0))
    out = np.zeros((n_channels, T), dtype=np.float32)
    for ch in range(n_channels):
        phase = rng.uniform(0, 2 * np.pi, size=freqs.shape)
        spec = mag * (np.cos(phase) + 1j * np.sin(phase))
        x = np.fft.irfft(spec, n=T).astype(np.float32)
        x -= x.mean()
        x /= (x.std() + 1e-8)
        out[ch] = x
    return out

def apply_leadfield_mismatch(
    eeg: np.ndarray,
    rng: np.random.Generator,
    mode: str = "mix",
    strength: float = 0.15,
    scale_lo: float = 0.8,
    scale_hi: float = 1.25,
) -> np.ndarray:
    y = eeg.astype(np.float32, copy=True)
    scales = rng.uniform(scale_lo, scale_hi, size=(y.shape[0],)).astype(np.float32)
    y = scales[:, None] * y
    if mode == "scale":
        return y.astype(np.float32)
    if mode == "mix":
        C = y.shape[0]
        A = rng.normal(size=(C, C)).astype(np.float32)
        A /= (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
        M = (1.0 - strength) * np.eye(C, dtype=np.float32) + strength * A
        return (M @ y).astype(np.float32)
    raise ValueError(mode)

def apply_noise_mismatch(
    eeg: np.ndarray,
    rng: np.random.Generator,
    beta: float = 1.0,
    snr_db: float = 15.0,
) -> np.ndarray:
    y = eeg.astype(np.float32, copy=True)
    noise = _colored_noise(beta, (y.shape[0], y.shape[1]), rng)
    sig_rms = float(np.sqrt(np.mean(y**2)) + 1e-8)
    noise_rms = float(np.sqrt(np.mean(noise**2)) + 1e-8)
    snr_lin = 10.0 ** (snr_db / 20.0)
    scale = sig_rms / (snr_lin * noise_rms)
    return (y + scale * noise).astype(np.float32)

def apply_stimulus_jitter(
    eeg: np.ndarray,
    rng: np.random.Generator,
    fs: int = 250,
    jitter_ms: float = 25.0,
) -> np.ndarray:
    y = eeg.astype(np.float32, copy=True)
    T = y.shape[1]
    t = np.arange(T, dtype=np.float32)
    max_shift = (jitter_ms / 1000.0) * fs
    shift = float(rng.uniform(-max_shift, +max_shift))
    t_src = np.clip(t - shift, 0.0, T - 1.0)
    out = np.zeros_like(y, dtype=np.float32)
    for ch in range(y.shape[0]):
        out[ch] = np.interp(t, t_src, y[ch]).astype(np.float32)
    return out

def apply_mismatch_pipeline(
    eeg: np.ndarray,
    rng: np.random.Generator,
    token_cfg,
    lead_mode: str,
    lead_strength: float,
    lead_scale_lo: float,
    lead_scale_hi: float,
    noise_beta: float,
    noise_snr_db: float,
    stim_jitter_ms: float,
) -> np.ndarray:
    y = apply_leadfield_mismatch(eeg, rng, mode=lead_mode, strength=lead_strength,
                                 scale_lo=lead_scale_lo, scale_hi=lead_scale_hi)
    y = apply_noise_mismatch(y, rng, beta=noise_beta, snr_db=noise_snr_db)
    y = apply_stimulus_jitter(y, rng, fs=int(token_cfg.fs), jitter_ms=stim_jitter_ms)
    return y


# -------------------------
# Cache builder (tokens + theta for splits)
# -------------------------
def build_cache_npz(
    cache_path: Path,
    theta: np.ndarray,
    eeg: np.ndarray,
    splits: Dict[str, np.ndarray],
    feature_set: str,
    n_time_patches: int,
    n_freq_patches: int,
) -> Path:
    mkdirp(cache_path.parent)
    token_cfg = make_token_cfg(n_time_patches, n_freq_patches)

    train_idx = splits["train"]
    val_idx = splits["val"]
    test_idx = splits["test"]

    print(f"[cache] featurizing feature_set={feature_set}  t={n_time_patches}  f={n_freq_patches}")
    X_train = featurize_split(eeg, train_idx, feature_set, token_cfg)
    X_val = featurize_split(eeg, val_idx, feature_set, token_cfg)
    X_test = featurize_split(eeg, test_idx, feature_set, token_cfg)

    theta_train = theta[train_idx].astype(np.float32)
    theta_val = theta[val_idx].astype(np.float32)
    theta_test = theta[test_idx].astype(np.float32)

    # Prior bounds for evaluation normalization (use empirical bounds of TRAIN theta)
    prior_low = theta_train.min(axis=0).astype(np.float32)
    prior_high = theta_train.max(axis=0).astype(np.float32)

    meta = {
        "param_names": PARAM_NAMES,
        "feature_set": feature_set,
        "n_time_patches": int(n_time_patches),
        "n_freq_patches": int(n_freq_patches),
        "token_cfg": {
            "fs": int(token_cfg.fs),
            "duration": float(token_cfg.duration),
            "n_channels": int(token_cfg.n_channels),
            "n_time_patches": int(token_cfg.n_time_patches),
            "n_freq_patches": int(token_cfg.n_freq_patches),
            "stim_onset": float(token_cfg.stim_onset),
            "f_min": float(token_cfg.f_min),
            "f_max": float(token_cfg.f_max),
            "nperseg": int(token_cfg.nperseg),
            "noverlap": int(token_cfg.noverlap),
            "nfft": int(token_cfg.nfft),
            "eps": float(token_cfg.eps),
        },
    }

    np.savez_compressed(
        cache_path,
        x_train=X_train.astype(np.float32),
        theta_train=theta_train,
        x_val=X_val.astype(np.float32),
        theta_val=theta_val,
        x_test=X_test.astype(np.float32),
        theta_test=theta_test,
        eeg_test=eeg[test_idx].astype(np.float32),  # needed for mismatch without rereading H5
        prior_low=prior_low,
        prior_high=prior_high,
        meta_json=np.array(json.dumps(meta), dtype=object),
    )
    print(f"[cache] wrote {cache_path}")
    return cache_path


# -------------------------
# SNPE train/eval
# -------------------------
def train_snpe_from_cache(
    cache_npz: Path,
    run_dir: Path,
    device: str,
    flow: str = "nsf",
    hidden_features: int = 128,
    num_transforms: int = 8,
    embedding_dim: int = 256,
    batch_size: int = 256,
    max_epochs: int = 200,
    lr: float = 2e-4,
    seed: int = 0,
) -> Path:
    _require_sbi()
    import torch
    from sbi.inference import SNPE
    from sbi.utils import BoxUniform
    # sbi changed module paths across versions; resolve posterior_nn dynamically
    import importlib
    def _resolve_posterior_nn():
        candidates = [
            "sbi.utils.get_nn_models",
            "sbi.neural_nets.net_builders",
            "sbi.neural_nets.factory",
            "sbi.neural_nets",
        ]
        for mod_name in candidates:
            try:
                mod = importlib.import_module(mod_name)
                fn = getattr(mod, "posterior_nn", None)
                if fn is not None:
                    return fn
            except Exception:
                continue
        raise ModuleNotFoundError(
            "Could not locate posterior_nn in sbi (tried: " + ", ".join(candidates) + ")"
        )
    posterior_nn = _resolve_posterior_nn()

    device = _device_resolve(device)
    mkdirp(run_dir)

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    z = np.load(cache_npz, allow_pickle=True)
    meta = json.loads(str(z["meta_json"]))
    x_train = flatten_tokens(z["x_train"].astype(np.float32))
    theta_train = z["theta_train"].astype(np.float32)
    x_val = flatten_tokens(z["x_val"].astype(np.float32))  # not strictly used, but used for z-score
    # z-score by train
    mu, sd = zscore_fit(x_train)
    x_train_z = zscore_apply(x_train, mu, sd)

    # Prior for SNPE: BoxUniform over empirical TRAIN bounds
    low = theta_train.min(axis=0).astype(np.float32)
    high = theta_train.max(axis=0).astype(np.float32)
    # Add tiny padding
    pad = 1e-6 * (high - low + 1e-6)
    low = low - pad
    high = high + pad

    prior = BoxUniform(
        low=torch.tensor(low, dtype=torch.float32, device=device),
        high=torch.tensor(high, dtype=torch.float32, device=device),
    )

    x_train_t = torch.tensor(x_train_z, dtype=torch.float32, device=device)
    theta_train_t = torch.tensor(theta_train, dtype=torch.float32, device=device)

    # embedding net
    emb = torch.nn.Sequential(
        torch.nn.Linear(x_train_t.shape[1], 1024),
        torch.nn.GELU(),
        torch.nn.Linear(1024, embedding_dim),
        torch.nn.GELU(),
    ).to(device)

    # Build density estimator (posterior_nn signature differs across sbi versions)
    import inspect
    nn_kwargs = dict(
        model=flow,
        hidden_features=hidden_features,
        num_transforms=num_transforms,
        embedding_net=emb,
    )
    sig = inspect.signature(posterior_nn)
    nn_kwargs = {k: v for k, v in nn_kwargs.items() if k in sig.parameters}
    de_fn = posterior_nn(**nn_kwargs)

    inference = SNPE(prior=prior, density_estimator=de_fn, device=device)
    inference = inference.append_simulations(theta_train_t, x_train_t)

    # Train (uses internal validation split for early stopping)
    density_estimator = inference.train(
        training_batch_size=batch_size,
        learning_rate=lr,
        max_num_epochs=max_epochs,
        validation_fraction=0.1,
        stop_after_epochs=20,
        show_train_summary=True,
    )
    posterior = inference.build_posterior(density_estimator)

    import torch as _torch
    posterior_path = run_dir / "posterior_snpe.pt"
    _torch.save(posterior, posterior_path)

    train_meta = {
        "method": "SNPE",
        "device": device,
        "seed": seed,
        "cache_npz": str(cache_npz),
        "feature_set": meta["feature_set"],
        "n_time_patches": meta["n_time_patches"],
        "n_freq_patches": meta["n_freq_patches"],
        "flow": flow,
        "hidden_features": hidden_features,
        "num_transforms": num_transforms,
        "embedding_dim": embedding_dim,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "lr": lr,
        "x_zscore_mu": mu.tolist(),
        "x_zscore_sd": sd.tolist(),
        "prior_low_emp": low.tolist(),
        "prior_high_emp": high.tolist(),
    }
    write_json(run_dir / "train_meta.json", train_meta)
    print(f"[snpe] wrote {posterior_path}")
    return posterior_path

def sample_posterior(posterior, x_flat_z: np.ndarray, n_samples: int, device: str, batch_x: int = 32) -> np.ndarray:
    import torch

    device = _device_resolve(device)
    N = x_flat_z.shape[0]
    out = np.zeros((N, n_samples, len(PARAM_NAMES)), dtype=np.float32)

    # Try batching; fallback to per-example
    try:
        for i in range(0, N, batch_x):
            xb = torch.tensor(x_flat_z[i:i+batch_x], dtype=torch.float32, device=device)
            s = posterior.sample((n_samples,), x=xb)  # expected (n_samples, B, 9) in many sbi versions
            s_np = s.detach().cpu().numpy()

            if s_np.ndim == 3 and s_np.shape[0] == n_samples:
                s_np = np.transpose(s_np, (1, 0, 2))  # (B,S,9)
            elif s_np.ndim == 2 and s_np.shape[0] == n_samples:
                s_np = s_np[None, :, :]
            else:
                raise RuntimeError(f"Unexpected posterior.sample output shape: {s_np.shape}")

            out[i:i+s_np.shape[0]] = s_np.astype(np.float32)
        return out
    except Exception as e:
        print(f"[WARN] Batched sampling failed ({e}); falling back to per-example sampling (slower).")
        for i in range(N):
            xb = torch.tensor(x_flat_z[i], dtype=torch.float32, device=device)
            s = posterior.sample((n_samples,), x=xb).detach().cpu().numpy().astype(np.float32)
            out[i] = s
        return out

def eval_snpe(
    run_dir: Path,
    split: str,
    device: str,
    n_post_samples: int,
    tag: str,
    mismatch: Optional[Dict[str, Any]] = None,
) -> None:
    _require_sbi()
    import torch

    run_dir = Path(run_dir)
    cache_npz = run_dir / "cache_data.npz"
    if not cache_npz.exists():
        raise RuntimeError(f"Missing cache file: {cache_npz}. Run train first or build cache.")

    posterior_path = run_dir / "posterior_snpe.pt"
    if not posterior_path.exists():
        raise RuntimeError(f"Missing posterior: {posterior_path}. Train first.")

    train_meta = read_json(run_dir / "train_meta.json")
    mu = np.array(train_meta["x_zscore_mu"], dtype=np.float32)
    sd = np.array(train_meta["x_zscore_sd"], dtype=np.float32)

    z = np.load(cache_npz, allow_pickle=True)
    meta = json.loads(str(z["meta_json"]))
    prior_low = z["prior_low"].astype(np.float32)
    prior_high = z["prior_high"].astype(np.float32)

    if split != "test":
        raise RuntimeError("This baseline evaluator currently supports split='test' only (matching NeurIPS diagnostics).")

    theta_true = z["theta_test"].astype(np.float32)

    token_cfg = make_token_cfg(int(meta["n_time_patches"]), int(meta["n_freq_patches"]))
    feature_set = str(meta["feature_set"])

    if mismatch is None:
        x = flatten_tokens(z["x_test"].astype(np.float32))
    else:
        # apply mismatch to raw test EEG, then re-featurize
        eeg_test = z["eeg_test"].astype(np.float32)
        rng = np.random.default_rng(int(mismatch.get("seed", 999)))
        lead_mode = mismatch["lead_mode"]
        lead_strength = float(mismatch["lead_strength"])
        lead_scale_lo = float(mismatch["lead_scale_lo"])
        lead_scale_hi = float(mismatch["lead_scale_hi"])
        noise_beta = float(mismatch["noise_beta"])
        noise_snr_db = float(mismatch["noise_snr_db"])
        stim_jitter_ms = float(mismatch["stim_jitter_ms"])

        # build mismatched tokens
        # determine L
        y0 = apply_mismatch_pipeline(
            eeg_test[0], rng, token_cfg,
            lead_mode=lead_mode, lead_strength=lead_strength,
            lead_scale_lo=lead_scale_lo, lead_scale_hi=lead_scale_hi,
            noise_beta=noise_beta, noise_snr_db=noise_snr_db,
            stim_jitter_ms=stim_jitter_ms,
        )
        tok0 = featurize_one(y0, feature_set, token_cfg)
        L = tok0.shape[0]
        X = np.zeros((eeg_test.shape[0], L, 16), dtype=np.float32)
        for i in range(eeg_test.shape[0]):
            yi = apply_mismatch_pipeline(
                eeg_test[i], rng, token_cfg,
                lead_mode=lead_mode, lead_strength=lead_strength,
                lead_scale_lo=lead_scale_lo, lead_scale_hi=lead_scale_hi,
                noise_beta=noise_beta, noise_snr_db=noise_snr_db,
                stim_jitter_ms=stim_jitter_ms,
            )
            X[i] = featurize_one(yi, feature_set, token_cfg)
        x = flatten_tokens(X)

    x_z = zscore_apply(x, mu, sd)

    device = _device_resolve(device)
    try:
        posterior = torch.load(posterior_path, map_location=device, weights_only=False)
    except TypeError:
        posterior = torch.load(posterior_path, map_location=device)

    samples = sample_posterior(posterior, x_z, n_samples=n_post_samples, device=device, batch_x=32)
    post_mean = samples.mean(axis=1)

    r = pearsonr_per_param(theta_true, post_mean)
    rmse_n = rmse_norm_per_param(theta_true, post_mean, prior_low, prior_high)

    levels = np.linspace(0.05, 0.95, 19, dtype=np.float32)
    cov = coverage_from_samples(theta_true, samples, levels)
    hist = sbc_hist(theta_true, samples)

    out = run_dir / ("eval_mismatch" if mismatch is not None else "eval_clean")
    mkdirp(out)

    metrics = {
        "tag": tag,
        "feature_set": meta["feature_set"],
        "n_time_patches": meta["n_time_patches"],
        "n_freq_patches": meta["n_freq_patches"],
        "n_test": int(theta_true.shape[0]),
        "n_post_samples": int(n_post_samples),
        "mean_pearson": float(np.mean(r)),
        "mean_rmse_norm": float(np.mean(rmse_n)),
        "pearson_per_param": {PARAM_NAMES[i]: float(r[i]) for i in range(len(PARAM_NAMES))},
        "rmse_norm_per_param": {PARAM_NAMES[i]: float(rmse_n[i]) for i in range(len(PARAM_NAMES))},
        "mismatch": mismatch,
    }
    write_json(out / f"metrics_{tag}.json", metrics)

    np.savez_compressed(
        out / f"eval_outputs_{tag}.npz",
        theta_true=theta_true.astype(np.float32),
        post_mean=post_mean.astype(np.float32),
        post_samples=samples.astype(np.float32),
        levels=levels.astype(np.float32),
        coverage=cov.astype(np.float32),
        sbc_hist=hist.astype(np.int32),
    )

    plot_coverage(levels, cov, out / f"coverage_{tag}.png")
    plot_sbc(hist, out / f"sbc_{tag}.png")

    print(f"[eval] wrote {out} tag={tag}")


# -------------------------
# Main runner
# -------------------------
def run_snpe_pipeline(
    h5: Path,
    data_out: Path,
    split_seed: int,
    feature_set: str,
    n_time_patches: int,
    n_freq_patches: int,
    outdir: Path,
    device: str,
    n_post_samples: int,
    theta_key: Optional[str],
    eeg_key: Optional[str],
    max_n: Optional[int],
    seed: int,
    mismatch_cfg: Optional[Dict[str, Any]] = None,
) -> None:
    # Load raw arrays
    theta, eeg = load_h5_arrays(h5, theta_key=theta_key, eeg_key=eeg_key, max_n=max_n)
    splits = load_or_make_splits(data_out, split_seed, N=theta.shape[0])

    run_dir = mkdirp(outdir / f"snpe_{feature_set}_t{n_time_patches}_f{n_freq_patches}")

    cache_npz = run_dir / "cache_data.npz"
    if not cache_npz.exists():
        build_cache_npz(
            cache_path=cache_npz,
            theta=theta,
            eeg=eeg,
            splits=splits,
            feature_set=feature_set,
            n_time_patches=n_time_patches,
            n_freq_patches=n_freq_patches,
        )

    # Train
    posterior_path = run_dir / "posterior_snpe.pt"
    if not posterior_path.exists():
        train_snpe_from_cache(
            cache_npz=cache_npz,
            run_dir=run_dir,
            device=device,
            seed=seed,
        )

    # Eval clean
    eval_snpe(
        run_dir=run_dir,
        split="test",
        device=device,
        n_post_samples=n_post_samples,
        tag=f"snpe_{feature_set}_clean",
        mismatch=None,
    )

    # Eval mismatch (optional)
    if mismatch_cfg is not None:
        eval_snpe(
            run_dir=run_dir,
            split="test",
            device=device,
            n_post_samples=n_post_samples,
            tag=f"snpe_{feature_set}_mismatch",
            mismatch=mismatch_cfg,
        )

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    # common args helper
    def add_common(sp):
        sp.add_argument("--h5", type=str, default="data/synthetic_cmc_dataset.h5")
        sp.add_argument("--data-out", type=str, default="data_out")
        sp.add_argument("--split-seed", type=int, default=42)
        sp.add_argument("--features", type=str, choices=["erp", "tfr", "hybrid"], default="hybrid")
        sp.add_argument("--n-time-patches", type=int, default=25)
        sp.add_argument("--n-freq-patches", type=int, default=15)
        sp.add_argument("--outdir", type=str, default="results/neurips_sbi")
        sp.add_argument("--device", type=str, default="cpu")
        sp.add_argument("--n-post-samples", type=int, default=200)
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--max-n", type=int, default=None)
        sp.add_argument("--theta-key", type=str, default=None)
        sp.add_argument("--eeg-key", type=str, default=None)

    p_one = sub.add_parser("snpe", help="Train+eval SNPE for one feature set + patch grid")
    add_common(p_one)

    p_all = sub.add_parser("all", help="Run SNPE baselines for ERP/TFR/Hybrid + hybrid mismatch + patch ablation")
    add_common(p_all)
    p_all.add_argument("--do-patch-ablation", action="store_true")
    # mismatch knobs
    p_all.add_argument("--lead-mode", type=str, choices=["scale", "mix"], default="mix")
    p_all.add_argument("--lead-strength", type=float, default=0.15)
    p_all.add_argument("--lead-scale-lo", type=float, default=0.8)
    p_all.add_argument("--lead-scale-hi", type=float, default=1.25)
    p_all.add_argument("--noise-beta", type=float, default=1.0)
    p_all.add_argument("--noise-snr-db", type=float, default=15.0)
    p_all.add_argument("--stim-jitter-ms", type=float, default=25.0)

    args = p.parse_args()

    h5 = Path(args.h5)
    if not h5.exists():
        raise FileNotFoundError(f"H5 not found: {h5}")

    data_out = Path(args.data_out)
    outdir = Path(args.outdir)

    if args.cmd == "snpe":
        run_snpe_pipeline(
            h5=h5,
            data_out=data_out,
            split_seed=args.split_seed,
            feature_set=args.features,
            n_time_patches=args.n_time_patches,
            n_freq_patches=args.n_freq_patches,
            outdir=outdir,
            device=args.device,
            n_post_samples=args.n_post_samples,
            theta_key=args.theta_key,
            eeg_key=args.eeg_key,
            max_n=args.max_n,
            seed=args.seed,
            mismatch_cfg=None,
        )
        return

    if args.cmd == "all":
        mismatch_cfg = {
            "seed": int(args.seed) + 999,
            "lead_mode": args.lead_mode,
            "lead_strength": float(args.lead_strength),
            "lead_scale_lo": float(args.lead_scale_lo),
            "lead_scale_hi": float(args.lead_scale_hi),
            "noise_beta": float(args.noise_beta),
            "noise_snr_db": float(args.noise_snr_db),
            "stim_jitter_ms": float(args.stim_jitter_ms),
        }

        # Canonical patch res
        for feat in ["erp", "tfr", "hybrid"]:
            run_snpe_pipeline(
                h5=h5,
                data_out=data_out,
                split_seed=args.split_seed,
                feature_set=feat,
                n_time_patches=25,
                n_freq_patches=15,
                outdir=outdir,
                device=args.device,
                n_post_samples=args.n_post_samples,
                theta_key=args.theta_key,
                eeg_key=args.eeg_key,
                max_n=args.max_n,
                seed=args.seed,
                mismatch_cfg=(mismatch_cfg if feat == "hybrid" else None),
            )

        # Optional patch ablation (hybrid only)
        if args.do_patch_ablation:
            for (tp, fp) in [(20, 12), (25, 15), (30, 18)]:
                run_snpe_pipeline(
                    h5=h5,
                    data_out=data_out,
                    split_seed=args.split_seed,
                    feature_set="hybrid",
                    n_time_patches=tp,
                    n_freq_patches=fp,
                    outdir=outdir / "patch_ablation",
                    device=args.device,
                    n_post_samples=args.n_post_samples,
                    theta_key=args.theta_key,
                    eeg_key=args.eeg_key,
                    max_n=args.max_n,
                    seed=args.seed,
                    mismatch_cfg=None,
                )
        print(f"\n[DONE] SNPE outputs in: {outdir}")
        return


if __name__ == "__main__":
    main()
