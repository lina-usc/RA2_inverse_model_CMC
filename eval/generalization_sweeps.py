from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------
# Paths + imports
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Ensure custom Keras layers are registered before load_model()
try:
    import models.transformer_paramtoken  # noqa: F401
except Exception:
    pass
try:
    import models.bilstm_baseline  # noqa: F401
except Exception:
    pass


DATA_OUT = os.path.join(BASE_DIR, "data_out")


def _import_tensorflow_or_die():
    try:
        import tensorflow as tf  # type: ignore
        return tf
    except ModuleNotFoundError as e:
        raise SystemExit(
            "\nERROR: TensorFlow not found in this environment.\n"
            "Activate your venv first:\n"
            "  source .venv/bin/activate\n"
        ) from e


def _feature_path(features: str) -> str:
    if features == "hybrid":
        return os.path.join(DATA_OUT, "features.npy")
    if features == "erp":
        return os.path.join(DATA_OUT, "features_erp.npy")
    if features == "tfr":
        return os.path.join(DATA_OUT, "features_tfr.npy")
    raise ValueError("features must be one of: hybrid, erp, tfr")


def _parse_list(s: str, cast=float) -> List:
    s = (s or "").strip()
    if not s:
        return []
    return [cast(x.strip()) for x in s.split(",") if x.strip() != ""]


def _softplus(x: np.ndarray) -> np.ndarray:
    # stable softplus
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _raw_tril_to_chol_row_softplus(
    raw: np.ndarray,
    P: int,
    diag_eps: float = 1e-3,
    offdiag_scale: float = 1.0,
) -> np.ndarray:
    """
    Matches the mapping you inferred & validated:
      ordering=row, diag_transform=softplus, diag_eps=0.001, offdiag_scale=1.0
    raw: (B, P*(P+1)/2)
    returns L: (B, P, P) lower-triangular Cholesky
    """
    raw = np.asarray(raw, dtype=np.float32)
    B = raw.shape[0]
    L = np.zeros((B, P, P), dtype=np.float32)

    k = 0
    for i in range(P):
        for j in range(i + 1):
            v = raw[:, k]
            if i == j:
                L[:, i, j] = _softplus(v) + float(diag_eps)
            else:
                L[:, i, j] = float(offdiag_scale) * v
            k += 1
    return L


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum()) + 1e-12
    return float((x * y).sum() / denom)


def _compute_metrics(
    theta_true: np.ndarray,          # (B,P)
    theta_mean: np.ndarray,          # (B,P)
    theta_samples: np.ndarray,       # (B,S,P)
    low: np.ndarray,                 # (P,)
    high: np.ndarray,                # (P,)
    param_names: List[str],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    P = theta_true.shape[1]
    rng = (high - low).astype(np.float64)
    rng = np.where(rng < 1e-12, 1.0, rng)

    per_param: Dict[str, Dict[str, float]] = {}

    # quantiles for coverage
    q05 = np.quantile(theta_samples, 0.05, axis=1)  # (B,P)
    q95 = np.quantile(theta_samples, 0.95, axis=1)  # (B,P)
    post_std = theta_samples.std(axis=1)            # (B,P)

    pears = []
    rmsen = []
    cov90 = []
    stdm = []

    for j in range(P):
        name = param_names[j]
        err = theta_mean[:, j] - theta_true[:, j]
        rmse = float(np.sqrt(np.mean(err * err)))
        rmse_norm = float(rmse / rng[j])

        pr = _pearson(theta_true[:, j], theta_mean[:, j])

        c90 = float(np.mean((theta_true[:, j] >= q05[:, j]) & (theta_true[:, j] <= q95[:, j])))
        ms = float(np.mean(post_std[:, j]))

        per_param[name] = {
            "rmse": rmse,
            "rmse_norm": rmse_norm,
            "pearson": pr,
            "cov90": c90,
            "mean_post_std": ms,
        }

        pears.append(pr)
        rmsen.append(rmse_norm)
        cov90.append(c90)
        stdm.append(ms)

    summary = {
        "mean_pearson": float(np.mean(pears)),
        "mean_rmse_norm": float(np.mean(rmsen)),
        "mean_cov90": float(np.mean(cov90)),
        "mean_post_std": float(np.mean(stdm)),
    }
    return per_param, summary


def _load_eval_indices(first_model_dir: str, split: str) -> np.ndarray:
    p = os.path.join(first_model_dir, "split_indices_used.npz")
    if not os.path.exists(p):
        raise SystemExit(
            f"ERROR: {p} not found.\n"
            "Expected training run dirs to contain split_indices_used.npz."
        )
    d = np.load(p)
    key = f"{split}_idx"
    if key not in d.files:
        raise SystemExit(f"ERROR: {p} missing {key}. Has: {d.files}")
    return d[key].astype(np.int64)


def _load_scaler_and_mask(first_model_dir: str, features: str, n_tokens: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # scaler stats
    sc = os.path.join(first_model_dir, "scaler_stats.npz")
    if not os.path.exists(sc):
        raise SystemExit(f"ERROR: scaler stats not found: {sc}")
    dsc = np.load(sc)
    mu = dsc["mu"].astype(np.float32)
    sd = dsc["sd"].astype(np.float32)
    sd = np.where(sd < 1e-6, 1.0, sd).astype(np.float32)

    # token mask
    mc = os.path.join(first_model_dir, "model_config.npz")
    if os.path.exists(mc):
        dmc = np.load(mc, allow_pickle=True)
        if "token_mask_1d" in dmc.files:
            token_mask_1d = dmc["token_mask_1d"].astype(np.float32)
            if token_mask_1d.shape[0] != n_tokens:
                raise SystemExit(
                    f"ERROR: token_mask_1d has len={token_mask_1d.shape[0]} but n_tokens={n_tokens}.\n"
                    f"Check that you're evaluating with matching feature set."
                )
            return mu, sd, token_mask_1d

    # fallback: rebuild from meta (should almost never be needed now)
    meta = np.load(os.path.join(DATA_OUT, "tfr_meta.npz"))
    n_tokens_erp = int(meta["n_tokens_erp"])

    token_mask_1d = np.zeros((n_tokens,), dtype=np.float32)
    if features == "hybrid":
        token_mask_1d[:] = 1.0
    elif features == "erp":
        token_mask_1d[:n_tokens_erp] = 1.0
    elif features == "tfr":
        token_mask_1d[n_tokens_erp:] = 1.0
    else:
        raise ValueError("features must be one of: hybrid, erp, tfr")
    return mu, sd, token_mask_1d


def _load_models(tf, model_dirs: List[str]):
    models = []
    used_paths = []
    for d in model_dirs:
        p_best = os.path.join(d, "paramtoken_best.keras")
        p_final = os.path.join(d, "paramtoken_final.keras")
        path = p_best if os.path.exists(p_best) else p_final
        if not os.path.exists(path):
            raise SystemExit(f"ERROR: could not find model in {d} (expected best/final keras file)")
        m = tf.keras.models.load_model(path, compile=False)
        models.append(m)
        used_paths.append(path)
    return models, used_paths


def _z_to_theta(z: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    # Try to use repo transform if available; fallback to logistic.
    try:
        from models.param_transforms import z_to_theta as z_to_theta_repo  # type: ignore
        return z_to_theta_repo(z, low, high)
    except Exception:
        z = np.asarray(z, dtype=np.float32)
        z = np.clip(z, -20.0, 20.0)
        sig = 1.0 / (1.0 + np.exp(-z))
        # broadcast low/high onto z
        shape_prefix = (1,) * (z.ndim - 1)
        lo = low.reshape(shape_prefix + (-1,))
        hi = high.reshape(shape_prefix + (-1,))
        return lo + (hi - lo) * sig


def _predict_and_sample_mixture(
    tf,
    models,
    X: np.ndarray,           # (B,T,F)
    M: np.ndarray,           # (B,T)
    low: np.ndarray,         # (P,)
    high: np.ndarray,        # (P,)
    n_post_samples: int,
    seed: int,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      theta_mean (B,P)
      theta_std  (B,P)
      theta_samples (B,S,P) from uniform mixture across models
    """
    rng = np.random.default_rng(seed)

    B = X.shape[0]

    # per-model predictions
    mu_list = []
    L_list = []

    for m in models:
        y = m.predict([X, M], batch_size=batch_size, verbose=0).astype(np.float32)
        # infer P + n_tril from output length
        out_dim = y.shape[1]
        # Solve P by reading low/high
        P = int(low.shape[0])
        n_tril = P * (P + 1) // 2

        mu_z = y[:, :P]
        raw = y[:, P:P + n_tril]
        raw = np.clip(raw, -10.0, 10.0)

        L = _raw_tril_to_chol_row_softplus(raw, P, diag_eps=1e-3, offdiag_scale=1.0)

        mu_list.append(mu_z)
        L_list.append(L)

    mu_all = np.stack(mu_list, axis=0)  # (K,B,P)
    L_all = np.stack(L_list, axis=0)    # (K,B,P,P)
    K = mu_all.shape[0]
    P = mu_all.shape[2]

    # mixture sampling
    model_choice = rng.integers(0, K, size=(B, n_post_samples), dtype=np.int64)
    eps = rng.standard_normal(size=(B, n_post_samples, P)).astype(np.float32)

    z_samps = np.zeros((B, n_post_samples, P), dtype=np.float32)

    # compute for each component then pick via mask
    for k in range(K):
        mk = (model_choice == k)  # (B,S)
        if not np.any(mk):
            continue
        delta = np.einsum("bij,bsj->bsi", L_all[k], eps)  # (B,S,P)
        zk = mu_all[k][:, None, :] + delta
        z_samps[mk] = zk[mk]

    theta_samps = _z_to_theta(z_samps, low, high).astype(np.float32)  # (B,S,P)
    theta_mean = theta_samps.mean(axis=1).astype(np.float32)
    theta_std = theta_samps.std(axis=1).astype(np.float32)
    return theta_mean, theta_std, theta_samps


def _write_csv(path: str, rows: List[Dict[str, float]], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _plot_sweep(xs: List[float], ys: Dict[str, List[float]], out_png: str, title: str, xlabel: str) -> None:
    plt.figure(figsize=(8, 5))
    for k, v in ys.items():
        plt.plot(xs, v, marker="o", linewidth=2, label=k)
    plt.xlabel(xlabel)
    plt.ylabel("metric")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main() -> None:
    tf = _import_tensorflow_or_die()

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dirs", nargs="+", required=True)
    ap.add_argument("--features", choices=["erp", "tfr", "hybrid"], required=True)
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-post-samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-eval", type=int, default=0, help="If >0, subsample this many eval examples for faster sweeps.")

    # Sweeps (comma-separated lists)
    ap.add_argument("--noise-stds", type=str, default="0,0.01,0.02,0.05,0.1")
    ap.add_argument("--token-drop-probs", type=str, default="0,0.1,0.2,0.4")
    ap.add_argument("--channel-gain-stds", type=str, default="0,0.05,0.1,0.2")
    ap.add_argument("--channel-drop-probs", type=str, default="0,0.1,0.25,0.5")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # Load data
    X_mem = np.load(_feature_path(args.features), mmap_mode="r")  # (N,T,F)
    theta_mem = np.load(os.path.join(DATA_OUT, "params.npy"), mmap_mode="r")  # (N,P)

    N, T, F = X_mem.shape
    P = int(theta_mem.shape[1])

    # param meta
    pm = np.load(os.path.join(DATA_OUT, "param_meta.npz"))
    param_names = [x.decode("utf-8") for x in pm["param_names"]]
    low = pm["prior_low"].astype(np.float32)
    high = pm["prior_high"].astype(np.float32)

    # eval indices from first model dir
    eval_idx = _load_eval_indices(args.model_dirs[0], args.split)
    if args.n_eval and args.n_eval > 0 and args.n_eval < len(eval_idx):
        eval_idx = rng.choice(eval_idx, size=args.n_eval, replace=False).astype(np.int64)

    theta_true = np.asarray(theta_mem[eval_idx], dtype=np.float32)  # (B,P)

    # scaler + mask from first model dir
    mu, sd, token_mask_1d = _load_scaler_and_mask(args.model_dirs[0], args.features, n_tokens=T)

    # build base X + base mask
    X = np.asarray(X_mem[eval_idx], dtype=np.float32)  # (B,T,F)
    M = np.tile(token_mask_1d[None, :], (X.shape[0], 1)).astype(np.float32)  # (B,T)

    # scale + ablate invalid tokens
    X = (X - mu[None, None, :]) / sd[None, None, :]
    X *= M[:, :, None]

    # load models
    models, used_paths = _load_models(tf, args.model_dirs)

    # parse sweep levels
    noise_stds = _parse_list(args.noise_stds, float)
    token_drops = _parse_list(args.token_drop_probs, float)
    gain_stds = _parse_list(args.channel_gain_stds, float)
    chan_drops = _parse_list(args.channel_drop_probs, float)

    # helper: evaluate a perturbed (X,M)
    def eval_once(Xp: np.ndarray, Mp: np.ndarray, seed_offset: int) -> Dict[str, float]:
        th_mean, th_std, th_samps = _predict_and_sample_mixture(
            tf=tf,
            models=models,
            X=Xp,
            M=Mp,
            low=low,
            high=high,
            n_post_samples=int(args.n_post_samples),
            seed=int(args.seed + seed_offset),
            batch_size=int(args.batch_size),
        )
        _, summary = _compute_metrics(theta_true, th_mean, th_samps, low, high, param_names)
        return summary

    # -----------------------
    # Sweep 1: additive noise
    # -----------------------
    rows_noise = []
    for i, s in enumerate(noise_stds):
        Xp = X.copy()
        Mp = M.copy()
        if s > 0:
            n = rng.standard_normal(size=Xp.shape).astype(np.float32) * float(s)
            Xp = Xp + n * Mp[:, :, None]
        summary = eval_once(Xp, Mp, seed_offset=1000 + i)
        summary = {"level": float(s), **summary}
        rows_noise.append(summary)

    csv_noise = os.path.join(args.out_dir, "sweep_noise.csv")
    _write_csv(csv_noise, rows_noise, fieldnames=list(rows_noise[0].keys()))
    _plot_sweep(
        xs=[r["level"] for r in rows_noise],
        ys={
            "mean_pearson": [r["mean_pearson"] for r in rows_noise],
            "mean_rmse_norm": [r["mean_rmse_norm"] for r in rows_noise],
            "mean_cov90": [r["mean_cov90"] for r in rows_noise],
        },
        out_png=os.path.join(args.out_dir, "sweep_noise_mean_metrics.png"),
        title="Generalization sweep: additive feature noise",
        xlabel="noise_std (scaled feature space)",
    )

    # -----------------------
    # Sweep 2: token dropout
    # -----------------------
    rows_tok = []
    for i, p in enumerate(token_drops):
        Xp = X.copy()
        Mp = M.copy()
        if p > 0:
            keep = (rng.random(size=Mp.shape) >= float(p)).astype(np.float32)
            Mp = Mp * keep
            Xp = Xp * keep[:, :, None]
        summary = eval_once(Xp, Mp, seed_offset=2000 + i)
        summary = {"level": float(p), **summary}
        rows_tok.append(summary)

    csv_tok = os.path.join(args.out_dir, "sweep_token_drop.csv")
    _write_csv(csv_tok, rows_tok, fieldnames=list(rows_tok[0].keys()))
    _plot_sweep(
        xs=[r["level"] for r in rows_tok],
        ys={
            "mean_pearson": [r["mean_pearson"] for r in rows_tok],
            "mean_rmse_norm": [r["mean_rmse_norm"] for r in rows_tok],
            "mean_cov90": [r["mean_cov90"] for r in rows_tok],
        },
        out_png=os.path.join(args.out_dir, "sweep_token_drop_mean_metrics.png"),
        title="Generalization sweep: token dropout",
        xlabel="token_drop_prob",
    )

    # -----------------------
    # Sweep 3: channel gain
    # -----------------------
    rows_gain = []
    for i, gs in enumerate(gain_stds):
        Xp = X.copy()
        Mp = M.copy()
        if gs > 0:
            gains = (1.0 + rng.standard_normal(size=(Xp.shape[0], F)).astype(np.float32) * float(gs)).astype(np.float32)
            Xp = Xp * gains[:, None, :]
        summary = eval_once(Xp, Mp, seed_offset=3000 + i)
        summary = {"level": float(gs), **summary}
        rows_gain.append(summary)

    csv_gain = os.path.join(args.out_dir, "sweep_channel_gain.csv")
    _write_csv(csv_gain, rows_gain, fieldnames=list(rows_gain[0].keys()))
    _plot_sweep(
        xs=[r["level"] for r in rows_gain],
        ys={
            "mean_pearson": [r["mean_pearson"] for r in rows_gain],
            "mean_rmse_norm": [r["mean_rmse_norm"] for r in rows_gain],
            "mean_cov90": [r["mean_cov90"] for r in rows_gain],
        },
        out_png=os.path.join(args.out_dir, "sweep_channel_gain_mean_metrics.png"),
        title="Generalization sweep: per-channel gain jitter",
        xlabel="channel_gain_std",
    )

    # -----------------------
    # Sweep 4: channel dropout
    # -----------------------
    rows_cd = []
    for i, p in enumerate(chan_drops):
        Xp = X.copy()
        Mp = M.copy()
        if p > 0:
            keepc = (rng.random(size=(Xp.shape[0], F)) >= float(p)).astype(np.float32)
            Xp = Xp * keepc[:, None, :]
        summary = eval_once(Xp, Mp, seed_offset=4000 + i)
        summary = {"level": float(p), **summary}
        rows_cd.append(summary)

    csv_cd = os.path.join(args.out_dir, "sweep_channel_drop.csv")
    _write_csv(csv_cd, rows_cd, fieldnames=list(rows_cd[0].keys()))
    _plot_sweep(
        xs=[r["level"] for r in rows_cd],
        ys={
            "mean_pearson": [r["mean_pearson"] for r in rows_cd],
            "mean_rmse_norm": [r["mean_rmse_norm"] for r in rows_cd],
            "mean_cov90": [r["mean_cov90"] for r in rows_cd],
        },
        out_png=os.path.join(args.out_dir, "sweep_channel_drop_mean_metrics.png"),
        title="Generalization sweep: channel dropout",
        xlabel="channel_drop_prob",
    )

    # write a small manifest for provenance
    manifest = {
        "features": args.features,
        "split": args.split,
        "n_eval": int(len(eval_idx)),
        "n_post_samples": int(args.n_post_samples),
        "model_dirs": args.model_dirs,
        "loaded_model_paths": used_paths,
        "sweeps": {
            "noise_stds": noise_stds,
            "token_drop_probs": token_drops,
            "channel_gain_stds": gain_stds,
            "channel_drop_probs": chan_drops,
        },
        "outputs": {
            "csv_noise": os.path.basename(csv_noise),
            "csv_token_drop": os.path.basename(csv_tok),
            "csv_channel_gain": os.path.basename(csv_gain),
            "csv_channel_drop": os.path.basename(csv_cd),
        },
    }
    with open(os.path.join(args.out_dir, "generalization_sweeps_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("[generalization_sweeps] wrote:", args.out_dir)
    print("[generalization_sweeps] csv:", os.path.basename(csv_noise), os.path.basename(csv_tok),
          os.path.basename(csv_gain), os.path.basename(csv_cd))


if __name__ == "__main__":
    main()
