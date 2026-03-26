"""
models.train

Train an amortized inverse model q_phi(theta | X) for the synthetic CMC->EEG task.

Supports:
  - Feature conditions: ERP / TFR / Hybrid (padded to a common token grid)
  - Architectures: Transformer (with/without parameter tokens) and a BiLSTM baseline
  - Posterior heads: diagonal-covariance Gaussian or full-covariance Gaussian (Cholesky)

Key:
  - Input token shape stays fixed, so swapping STFT vs Morlet is purely a DATA change.
  - Run dirs include posterior type to prevent collisions/overwrites.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import pickle
import random
import sys
from dataclasses import asdict
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

# --------------------------
# Repo root + imports
# --------------------------

# Expected location: <repo_root>/models/train.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from data.splits import ensure_splits  # noqa: E402
from models.param_transforms import theta_to_z  # noqa: E402
from models.posterior_fullcov import mvn_tril_nll, raw_tril_size  # noqa: E402
from models.transformer_noparamtoken import build_noparamtoken_transformer  # noqa: E402
from models.transformer_paramtoken import build_paramtoken_transformer  # noqa: E402
from models.train_config import TrainConfig  # noqa: E402
from models.bilstm_baseline import build_bilstm_baseline  # noqa: E402


# --------------------------
# Optional diag head fallback (matches eval script: mu + logvar)
# --------------------------

def _diag_nll_fallback(n_params: int):
    """Diagonal Gaussian NLL with outputs [mu, logvar]."""
    log2pi = tf.constant(np.log(2.0 * np.pi), dtype=tf.float32)

    def loss(y_true, y_pred):
        mu = y_pred[..., :n_params]
        logvar = y_pred[..., n_params: 2 * n_params]
        inv_var = tf.exp(-logvar)
        return 0.5 * tf.reduce_sum(
            tf.square(y_true - mu) * inv_var + logvar + log2pi,
            axis=-1,
        )

    return loss


try:
    from models.posterior_diag import diag_nll as _diag_nll_import  # type: ignore  # noqa: E402

    def diag_nll(n_params: int):
        return _diag_nll_import(n_params)

except Exception:
    def diag_nll(n_params: int):
        return _diag_nll_fallback(n_params)


# --------------------------
# Helpers
# --------------------------

def _resolve_dir(path: str) -> str:
    """Resolve a user path relative to repo root unless it's already absolute."""
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


def _feature_path(data_out_dir: str, feature_set: str) -> str:
    if feature_set == "hybrid":
        return os.path.join(data_out_dir, "features.npy")
    if feature_set == "erp":
        return os.path.join(data_out_dir, "features_erp.npy")
    if feature_set == "tfr":
        return os.path.join(data_out_dir, "features_tfr.npy")
    raise ValueError(f"Unknown feature_set: {feature_set}")


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _load_param_meta(data_out_dir: str) -> Dict[str, np.ndarray]:
    meta = np.load(os.path.join(data_out_dir, "param_meta.npz"), allow_pickle=True)
    low = meta["low"] if "low" in meta.files else meta["prior_low"]
    high = meta["high"] if "high" in meta.files else meta["prior_high"]
    names = meta["param_names"] if "param_names" in meta.files else meta["names"]
    return {"param_names": names, "low": low, "high": high}


def _load_tfr_meta(data_out_dir: str) -> Dict[str, object]:
    path = os.path.join(data_out_dir, "tfr_meta.npz")
    if not os.path.exists(path):
        return {}
    try:
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    except Exception:
        return {}


def _load_erp_meta(data_out_dir: str) -> Dict[str, object]:
    path = os.path.join(data_out_dir, "erp_meta.npz")
    if not os.path.exists(path):
        return {}
    try:
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    except Exception:
        return {}


def _infer_token_counts(data_out_dir: str, n_tokens_total: int) -> Tuple[int, int]:
    """Return (n_tokens_erp, n_tokens_tfr) with robust fallbacks.

    Prefer metadata if present; otherwise assume ERP=25 and TFR=rest.
    """
    tfr_meta = _load_tfr_meta(data_out_dir)
    erp_meta = _load_erp_meta(data_out_dir)

    n_tokens_erp = None

    # Most reliable in your repo (eval script uses it)
    if "n_tokens_erp" in tfr_meta:
        try:
            n_tokens_erp = int(np.asarray(tfr_meta["n_tokens_erp"]).item())
        except Exception:
            pass

    # ERP meta fallback
    if n_tokens_erp is None:
        for k in ["n_tokens", "n_tokens_erp"]:
            if k in erp_meta:
                try:
                    n_tokens_erp = int(np.asarray(erp_meta[k]).item())
                    break
                except Exception:
                    pass

    # Hard fallback
    if n_tokens_erp is None:
        n_tokens_erp = 25

    n_tokens_erp = int(max(1, min(n_tokens_erp, n_tokens_total)))
    n_tokens_tfr = int(max(1, n_tokens_total - n_tokens_erp))
    return n_tokens_erp, n_tokens_tfr


def _infer_tfr_grid(data_out_dir: str, n_tokens_tfr: int) -> Tuple[int, int]:
    """Return (n_time_patches, n_freq_patches) for TFR tokens.

    Prefer tfr_meta.npz if available; else factorize n_tokens_tfr.
    """
    tfr_meta = _load_tfr_meta(data_out_dir)

    def _get_int(keys):
        for k in keys:
            if k in tfr_meta:
                try:
                    return int(np.asarray(tfr_meta[k]).item())
                except Exception:
                    pass
        return None

    n_t = _get_int(["n_time_patches", "n_t_patches", "n_time", "n_times"])
    n_f = _get_int(["n_freq_patches", "n_f_patches", "n_freq", "n_freqs"])

    if n_t is not None and n_f is not None and n_t * n_f == n_tokens_tfr:
        return int(n_t), int(n_f)

    if n_t is not None and n_tokens_tfr % int(n_t) == 0:
        return int(n_t), int(n_tokens_tfr // int(n_t))

    if n_f is not None and n_tokens_tfr % int(n_f) == 0:
        return int(n_tokens_tfr // int(n_f)), int(n_f)

    # Known default for your 400-token grid
    if n_tokens_tfr == 375:
        return 25, 15

    # Generic factorization: choose factors closest together, with time>=freq
    best_t, best_f = n_tokens_tfr, 1
    best_diff = abs(best_t - best_f)
    root = int(np.sqrt(n_tokens_tfr))
    for f in range(1, root + 1):
        if n_tokens_tfr % f == 0:
            t = n_tokens_tfr // f
            diff = abs(t - f)
            if diff < best_diff:
                best_t, best_f = t, f
                best_diff = diff
    if best_t < best_f:
        best_t, best_f = best_f, best_t
    return int(best_t), int(best_f)


def _ensure_splits_compat(*, data_out_dir: str, seed: int, overwrite: bool, n_samples: int) -> Dict[str, np.ndarray]:
    """Call ensure_splits with signature compatibility and normalize keys."""
    def _normalize(d: Dict[str, object]) -> Dict[str, np.ndarray]:
        train_keys = ["train", "train_idx", "train_indices", "train_ids"]
        val_keys = ["val", "val_idx", "valid", "valid_idx", "val_indices", "valid_indices", "val_ids", "valid_ids"]
        test_keys = ["test", "test_idx", "test_indices", "test_ids"]

        out: Dict[str, np.ndarray] = {}

        for k in train_keys:
            if k in d:
                out["train"] = np.asarray(d[k], dtype=np.int64)
                break
        for k in val_keys:
            if k in d:
                out["val"] = np.asarray(d[k], dtype=np.int64)
                break
        for k in test_keys:
            if k in d:
                out["test"] = np.asarray(d[k], dtype=np.int64)
                break

        if {"train", "val", "test"}.issubset(out.keys()):
            return out
        raise KeyError(f"Could not normalize split dict keys. Found keys={list(d.keys())}")

    split = None
    try:
        split = ensure_splits(n_samples=n_samples, data_out_dir=data_out_dir, seed=seed, overwrite=overwrite)
    except TypeError:
        try:
            split = ensure_splits(n_samples, data_out_dir=data_out_dir, seed=seed, overwrite=overwrite)
        except TypeError:
            split = ensure_splits(data_out_dir=data_out_dir, seed=seed, overwrite=overwrite)

    if split is None:
        candidates = [
            os.path.join(data_out_dir, "splits.npz"),
            os.path.join(data_out_dir, f"splits_seed{seed}.npz"),
            os.path.join(data_out_dir, "split_idx.npz"),
            os.path.join(data_out_dir, f"split_idx_seed{seed}.npz"),
        ]
        for p in candidates:
            if os.path.exists(p):
                arr = np.load(p, allow_pickle=True)
                return _normalize({k: arr[k] for k in arr.files})
        raise RuntimeError("ensure_splits returned None and no splits file was found in data_out_dir.")

    if isinstance(split, dict):
        return _normalize(split)

    if isinstance(split, (tuple, list)) and len(split) == 3:
        return {
            "train": np.asarray(split[0], dtype=np.int64),
            "val": np.asarray(split[1], dtype=np.int64),
            "test": np.asarray(split[2], dtype=np.int64),
        }

    try:
        if hasattr(split, "files"):
            return _normalize({k: split[k] for k in split.files})
    except Exception:
        pass

    raise TypeError(f"Unexpected ensure_splits return type: {type(split)}")


def _call_builder_compat(fn, **kwargs):
    """Call a model-builder with only the kwargs it actually supports."""
    sig = inspect.signature(fn)
    params = sig.parameters

    # If builder accepts **kwargs, just pass through.
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(**kwargs)

    out = {k: v for k, v in kwargs.items() if k in params}

    # (extra robust) if builder has dropout-like arg name but not 'dropout'
    if "dropout" in kwargs and "dropout" not in out:
        for k in params.keys():
            if "dropout" in k.lower() and k not in out:
                out[k] = kwargs["dropout"]
                break

    # (extra robust) posterior-like arg name
    if "posterior" in kwargs and "posterior" not in out:
        for k in params.keys():
            if "posterior" in k.lower() and k not in out:
                out[k] = kwargs["posterior"]
                break

    return fn(**out)


# --------------------------
# Main
# --------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--run-name", type=str, default="cmc")
    ap.add_argument("--features", type=str, choices=["erp", "tfr", "hybrid"], default="hybrid")
    ap.add_argument("--arch", type=str, choices=["paramtoken", "noparamtoken", "bilstm"], default="paramtoken")
    ap.add_argument("--posterior", type=str, choices=["diag", "fullcov"], default="fullcov")

    ap.add_argument("--train-seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=42)

    ap.add_argument("--data-out", type=str, default="data_out")
    ap.add_argument("--models-out", type=str, default="models_out")
    ap.add_argument("--overwrite-run", action="store_true")

    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--warmup-epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--learning-rate", type=float, default=None)
    ap.add_argument("--augment-std", type=float, default=None)
    ap.add_argument("--token-drop", type=float, default=0.0)
    ap.add_argument("--channel-drop", type=float, default=0.0)

    ap.add_argument("--lstm-units", type=int, default=None)
    ap.add_argument("--lstm-layers", type=int, default=None)
    ap.add_argument("--mlp-units", type=int, default=256)

    ap.add_argument("--d-model", type=int, default=None)
    ap.add_argument("--num-heads", type=int, default=None)
    ap.add_argument("--ff-dim", type=int, default=None)
    ap.add_argument("--num-layers", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)

    args = ap.parse_args()

    data_out_dir = _resolve_dir(args.data_out)
    models_out_dir = _resolve_dir(args.models_out)
    os.makedirs(models_out_dir, exist_ok=True)

    set_all_seeds(args.train_seed)

    meta = _load_param_meta(data_out_dir)
    param_names = [str(x) for x in meta["param_names"].tolist()]
    low = meta["low"].astype(np.float32)
    high = meta["high"].astype(np.float32)
    n_params = len(param_names)

    X_path = _feature_path(data_out_dir, args.features)
    X_mem = np.load(X_path, mmap_mode="r")
    n_samples, n_tokens, feature_dim = X_mem.shape

    # Infer ERP/TFR token structure + TFR grid (needed by transformer builders in YOUR repo)
    n_tokens_erp, n_tokens_tfr = _infer_token_counts(data_out_dir, n_tokens_total=n_tokens)
    n_time_patches, n_freq_patches = _infer_tfr_grid(data_out_dir, n_tokens_tfr=n_tokens_tfr)

    # Token mask
    token_mask_1d = np.zeros((n_tokens,), dtype=np.float32)
    if args.features == "hybrid":
        token_mask_1d[:] = 1.0
    elif args.features == "erp":
        token_mask_1d[:n_tokens_erp] = 1.0
    else:  # tfr
        token_mask_1d[n_tokens_erp:] = 1.0
    valid_idx = np.where(token_mask_1d > 0.5)[0]

    # Splits
    split = _ensure_splits_compat(
        data_out_dir=data_out_dir,
        seed=args.split_seed,
        overwrite=False,
        n_samples=int(n_samples),
    )
    train_idx = split["train"]
    val_idx = split["val"]
    test_idx = split["test"]

    theta_all = np.load(os.path.join(data_out_dir, "params.npy"), mmap_mode="r").astype(np.float32)
    z_all = theta_to_z(theta_all, low, high).astype(np.float32)
    z_train = z_all[train_idx]
    z_val = z_all[val_idx]

    X_train = np.array(X_mem[train_idx], dtype=np.float32)
    X_val = np.array(X_mem[val_idx], dtype=np.float32)

    # Standardize on valid tokens only
    scaler = StandardScaler()
    scaler.fit(X_train[:, valid_idx, :].reshape(-1, feature_dim))

    def _apply_scaler_and_mask(X: np.ndarray) -> np.ndarray:
        Xs = X.copy()
        Xs[:, valid_idx, :] = scaler.transform(Xs[:, valid_idx, :].reshape(-1, feature_dim)).reshape(
            Xs.shape[0], len(valid_idx), feature_dim
        )
        Xs[:, token_mask_1d < 0.5, :] = 0.0
        return Xs

    X_train = _apply_scaler_and_mask(X_train)
    X_val = _apply_scaler_and_mask(X_val)

    m_train = np.tile(token_mask_1d[None, :], (X_train.shape[0], 1)).astype(np.float32)
    m_val = np.tile(token_mask_1d[None, :], (X_val.shape[0], 1)).astype(np.float32)

    # Train config
    cfg = TrainConfig()
    if args.epochs is not None:
        cfg.epochs = int(args.epochs)
    if args.warmup_epochs is not None:
        cfg.warmup_epochs = int(args.warmup_epochs)
    if args.batch_size is not None:
        cfg.batch_size = int(args.batch_size)
    if args.learning_rate is not None:
        cfg.learning_rate = float(args.learning_rate)
    if args.augment_std is not None:
        cfg.augment_std = float(args.augment_std)
    if args.d_model is not None:
        cfg.d_model = int(args.d_model)
    if args.num_heads is not None:
        cfg.num_heads = int(args.num_heads)
    if args.ff_dim is not None:
        cfg.ff_dim = int(args.ff_dim)
    if args.num_layers is not None:
        cfg.num_layers = int(args.num_layers)
    if args.dropout is not None:
        cfg.dropout = float(args.dropout)

    # Run dir
    arch_suffix = "" if args.arch == "paramtoken" else f"_{args.arch}"
    run_dir = os.path.join(
        models_out_dir,
        f"{args.run_name}_{args.features}{arch_suffix}_{args.posterior}_seed{args.train_seed}",
    )

    if args.overwrite_run and os.path.isdir(run_dir):
        for root, dirs, files in os.walk(run_dir, topdown=False):
            for fn in files:
                os.remove(os.path.join(root, fn))
            for dn in dirs:
                os.rmdir(os.path.join(root, dn))
        os.rmdir(run_dir)
    os.makedirs(run_dir, exist_ok=True)

    # Save scaler stats (compatible with multiple eval scripts)
    with open(os.path.join(run_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    mu = scaler.mean_.astype(np.float32)
    sd = scaler.scale_.astype(np.float32)
    np.savez(
        os.path.join(run_dir, "scaler_stats.npz"),
        mu=mu,
        sd=sd,
        mean=mu,
        std=sd,
        scale=sd,
        token_mask=token_mask_1d,
        valid_idx=valid_idx.astype(np.int32),
    )

    np.savez(
        os.path.join(run_dir, "param_bounds.npz"),
        low=low,
        high=high,
        names=np.array(param_names, dtype=object),
    )

    np.savez(
        os.path.join(run_dir, "split_indices_used.npz"),
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        train=train_idx,
        val=val_idx,
        test=test_idx,
    )
    np.savez(
        os.path.join(run_dir, "split_idx.npz"),
        train=train_idx,
        val=val_idx,
        test=test_idx,
    )

    # Posterior output dim + loss
    if args.posterior == "fullcov":
        out_dim = int(n_params + raw_tril_size(n_params))

        def loss_fn(y_true, y_pred):
            mu_pred = y_pred[..., :n_params]
            raw_tril = y_pred[..., n_params:]
            return mvn_tril_nll(y_true, mu_pred, raw_tril)

    else:  # diag
        out_dim = int(2 * n_params)
        loss_fn = diag_nll(n_params)

    # Common kwargs: include MANY aliases so signature mismatches never break again
    builder_kwargs = dict(
        # token shapes
        n_tokens=n_tokens,
        n_tokens_total=n_tokens,
        total_tokens=n_tokens,
        # feature dims
        feature_dim=feature_dim,
        feat_dim=feature_dim,
        # params / outputs
        n_params=n_params,
        n_parameters=n_params,
        out_dim=out_dim,
        n_outputs=out_dim,
        # patch grid / ERP tokens (YOUR builder requires these)
        n_tokens_erp=n_tokens_erp,
        n_erp_tokens=n_tokens_erp,
        erp_tokens=n_tokens_erp,
        n_time_patches=n_time_patches,
        n_t_patches=n_time_patches,
        n_time=n_time_patches,
        n_times=n_time_patches,
        n_freq_patches=n_freq_patches,
        n_f_patches=n_freq_patches,
        n_freq=n_freq_patches,
        n_freqs=n_freq_patches,
        n_tokens_tfr=n_tokens_tfr,
        n_tfr_tokens=n_tokens_tfr,
        tfr_tokens=n_tokens_tfr,
        # transformer hparams
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        ff_dim=cfg.ff_dim,
        num_layers=cfg.num_layers,
        dropout=float(cfg.dropout),
        # posterior selector
        posterior=args.posterior,
        posterior_type=args.posterior,
        posterior_head=args.posterior,
    )

    # Build model
    if args.arch == "paramtoken":
        model = _call_builder_compat(build_paramtoken_transformer, **builder_kwargs)

    elif args.arch == "noparamtoken":
        model = _call_builder_compat(build_noparamtoken_transformer, **builder_kwargs)

    else:  # bilstm
        lstm_units = 128 if args.lstm_units is None else int(args.lstm_units)
        lstm_layers = 2 if args.lstm_layers is None else int(args.lstm_layers)
        model = build_bilstm_baseline(
            n_tokens=n_tokens,
            feat_dim=feature_dim,
            out_dim=out_dim,
            hidden_size=lstm_units,
            depth=lstm_layers,
            dropout=float(cfg.dropout),
            mlp_units=int(args.mlp_units),
            name="bilstm_baseline",
        )

    model.summary()

    # Dataset elements are ((x, m), z)
    @tf.function
    def _augment(inputs, z):
        x, m = inputs

        if float(cfg.augment_std) > 0.0:
            x = x + tf.random.normal(tf.shape(x), stddev=float(cfg.augment_std))

        if float(args.token_drop) > 0.0:
            keep = tf.cast(tf.random.uniform(tf.shape(m)) > float(args.token_drop), tf.float32)
            m = m * keep
            x = x * tf.expand_dims(keep, -1)

        if float(args.channel_drop) > 0.0:
            keep_shape = tf.stack([tf.shape(x)[0], 1, tf.shape(x)[2]])
            keep_ch = tf.cast(tf.random.uniform(keep_shape) > float(args.channel_drop), tf.float32)
            x = x * keep_ch

        return (x, m), z

    ds_train = (
        tf.data.Dataset.from_tensor_slices(((X_train, m_train), z_train))
        .shuffle(min(20000, X_train.shape[0]), seed=args.train_seed, reshuffle_each_iteration=True)
        .batch(int(cfg.batch_size))
    )

    if float(cfg.augment_std) > 0.0 or float(args.token_drop) > 0.0 or float(args.channel_drop) > 0.0:
        ds_train = ds_train.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_val = (
        tf.data.Dataset.from_tensor_slices(((X_val, m_val), z_val))
        .batch(int(cfg.batch_size))
        .prefetch(tf.data.AUTOTUNE)
    )

    # IMPORTANT: keep filenames stable for your bash script
    best_path = os.path.join(run_dir, "paramtoken_best.keras")
    final_path = os.path.join(run_dir, "paramtoken_final.keras")

    # Warmup (mean-only)
    warmup_opt = Adam(learning_rate=float(cfg.learning_rate))
    model.compile(
        optimizer=warmup_opt,
        loss=lambda y_true, y_pred: tf.reduce_mean(tf.square(y_pred[..., :n_params] - y_true)),
    )
    hist_warmup = model.fit(ds_train, validation_data=ds_val, epochs=int(cfg.warmup_epochs))

    # Main (NLL)
    opt = Adam(learning_rate=float(cfg.learning_rate))
    model.compile(optimizer=opt, loss=loss_fn)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=int(cfg.patience), restore_best_weights=True),
        ModelCheckpoint(best_path, monitor="val_loss", save_best_only=True, save_weights_only=False),
    ]
    hist_main = model.fit(ds_train, validation_data=ds_val, epochs=int(cfg.epochs), callbacks=callbacks)

    model.save(final_path)

    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump({"warmup": hist_warmup.history, "main": hist_main.history}, f, indent=2)

    cfg_dict = asdict(cfg)
    cfg_dict.update(
        {
            "run_name": args.run_name,
            "features": args.features,
            "arch": args.arch,
            "posterior": args.posterior,
            "train_seed": args.train_seed,
            "split_seed": args.split_seed,
            "token_drop": float(args.token_drop),
            "channel_drop": float(args.channel_drop),
            "data_out": os.path.relpath(data_out_dir, BASE_DIR),
            "models_out": os.path.relpath(models_out_dir, BASE_DIR),
            "n_params": int(n_params),
            "n_tokens": int(n_tokens),
            "feature_dim": int(feature_dim),
            "n_tokens_erp": int(n_tokens_erp),
            "n_tokens_tfr": int(n_tokens_tfr),
            "n_time_patches": int(n_time_patches),
            "n_freq_patches": int(n_freq_patches),
            "out_dim": int(out_dim),
        }
    )
    with open(os.path.join(run_dir, "train_config.json"), "w") as f:
        json.dump(cfg_dict, f, indent=2)

    np.savez(
        os.path.join(run_dir, "model_config.npz"),
        arch=args.arch,
        posterior=args.posterior,
        n_params=int(n_params),
        n_tokens=int(n_tokens),
        feature_dim=int(feature_dim),
        out_dim=int(out_dim),
    )

    print("\n=== DONE ===")
    print(f"run_dir:  {run_dir}")
    print(f"best:     {best_path}")
    print(f"final:    {final_path}")


if __name__ == "__main__":
    main()