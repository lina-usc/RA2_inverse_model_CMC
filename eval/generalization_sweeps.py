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
import matplotlib.pyplot as plt


def _find_repo_root() -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    cands = [os.getcwd(), here, os.path.dirname(here), os.path.dirname(os.path.dirname(here))]
    seen = []
    for c in cands:
        c = os.path.abspath(c)
        if c not in seen:
            seen.append(c)
    for c in seen:
        if os.path.isdir(os.path.join(c, "models")) and os.path.isdir(os.path.join(c, "data")):
            return c
    return os.getcwd()


BASE_DIR = _find_repo_root()
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    import models.transformer_paramtoken  # noqa: F401
except Exception:
    pass
try:
    import models.bilstm_baseline  # noqa: F401
except Exception:
    pass


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


def _set_plot_defaults() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )


def _feature_path(data_out: str, features: str) -> str:
    if features == "hybrid":
        return os.path.join(data_out, "features.npy")
    if features == "erp":
        return os.path.join(data_out, "features_erp.npy")
    if features == "tfr":
        return os.path.join(data_out, "features_tfr.npy")
    raise ValueError("features must be one of: hybrid, erp, tfr")


def _parse_list(s: str, cast=float) -> List:
    s = (s or "").strip()
    if not s:
        return []
    return [cast(x.strip()) for x in s.split(",") if x.strip() != ""]


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _raw_tril_to_chol_row_softplus(raw: np.ndarray, P: int, diag_eps: float = 1e-3, offdiag_scale: float = 1.0) -> np.ndarray:
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


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum()) + 1e-12
    return float((x * y).sum() / denom)


def _load_eval_indices(first_model_dir: str, split: str) -> np.ndarray:
    p = os.path.join(first_model_dir, "split_indices_used.npz")
    if not os.path.exists(p):
        raise SystemExit(f"ERROR: {p} not found.")
    d = np.load(p)
    key = f"{split}_idx"
    if key not in d.files:
        raise SystemExit(f"ERROR: {p} missing {key}. Has: {d.files}")
    return d[key].astype(np.int64)


def _load_scaler_and_mask(first_model_dir: str, data_out: str, features: str, n_tokens: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sc = os.path.join(first_model_dir, "scaler_stats.npz")
    if not os.path.exists(sc):
        raise SystemExit(f"ERROR: scaler stats not found: {sc}")
    dsc = np.load(sc)
    mu = dsc["mu"].astype(np.float32)
    sd = np.where(dsc["sd"].astype(np.float32) < 1e-6, 1.0, dsc["sd"].astype(np.float32))

    mc = os.path.join(first_model_dir, "model_config.npz")
    if os.path.exists(mc):
        dmc = np.load(mc, allow_pickle=True)
        if "token_mask_1d" in dmc.files:
            token_mask_1d = dmc["token_mask_1d"].astype(np.float32)
            if token_mask_1d.shape[0] != n_tokens:
                raise SystemExit(f"ERROR: token_mask_1d len={token_mask_1d.shape[0]} but n_tokens={n_tokens}")
            return mu, sd, token_mask_1d

    meta = np.load(os.path.join(data_out, "tfr_meta.npz"))
    n_tokens_erp = int(meta["n_tokens_erp"])
    token_mask_1d = np.zeros((n_tokens,), dtype=np.float32)
    if features == "hybrid":
        token_mask_1d[:] = 1.0
    elif features == "erp":
        token_mask_1d[:n_tokens_erp] = 1.0
    elif features == "tfr":
        token_mask_1d[n_tokens_erp:] = 1.0
    return mu, sd, token_mask_1d


def _find_model_path(model_dir: str) -> str:
    for name in ["paramtoken_best.keras", "paramtoken_final.keras", "model_best.keras", "model_final.keras"]:
        p = os.path.join(model_dir, name)
        if os.path.exists(p):
            return p
    for p in sorted([os.path.join(model_dir, x) for x in os.listdir(model_dir) if x.endswith(".keras")]):
        return p
    raise SystemExit(f"ERROR: could not find keras model in {model_dir}")


def _load_models(tf, model_dirs: List[str]):
    models = []
    used_paths = []
    for d in model_dirs:
        path = _find_model_path(d)
        m = tf.keras.models.load_model(path, compile=False)
        models.append(m)
        used_paths.append(path)
    return models, used_paths


def _z_to_theta(z: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    try:
        from models.param_transforms import z_to_theta as z_to_theta_repo  # type: ignore
        return z_to_theta_repo(z, low, high)
    except Exception:
        z = np.asarray(z, dtype=np.float32)
        z = np.clip(z, -20.0, 20.0)
        sig = 1.0 / (1.0 + np.exp(-z))
        shape_prefix = (1,) * (z.ndim - 1)
        lo = low.reshape(shape_prefix + (-1,))
        hi = high.reshape(shape_prefix + (-1,))
        return lo + (hi - lo) * sig


def _predict_and_sample_mixture(tf, models, X: np.ndarray, M: np.ndarray, low: np.ndarray, high: np.ndarray, n_post_samples: int, seed: int, batch_size: int = 64) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    B = X.shape[0]
    mu_list = []
    L_list = []
    for m in models:
        y = m.predict([X, M], batch_size=batch_size, verbose=0).astype(np.float32)
        P = int(low.shape[0])
        n_tril = P * (P + 1) // 2
        mu_z = y[:, :P]
        raw = np.clip(y[:, P:P + n_tril], -10.0, 10.0)
        L = _raw_tril_to_chol_row_softplus(raw, P, diag_eps=1e-3, offdiag_scale=1.0)
        mu_list.append(mu_z)
        L_list.append(L)
    mu_all = np.stack(mu_list, axis=0)
    L_all = np.stack(L_list, axis=0)
    K = mu_all.shape[0]
    P = mu_all.shape[2]

    model_choice = rng.integers(0, K, size=(B, n_post_samples), dtype=np.int64)
    eps = rng.standard_normal(size=(B, n_post_samples, P)).astype(np.float32)
    z_samps = np.zeros((B, n_post_samples, P), dtype=np.float32)
    for k in range(K):
        mk = model_choice == k
        if not np.any(mk):
            continue
        delta = np.einsum("bij,bsj->bsi", L_all[k], eps)
        zk = mu_all[k][:, None, :] + delta
        z_samps[mk] = zk[mk]
    theta_samps = _z_to_theta(z_samps, low, high).astype(np.float32)
    theta_mean = theta_samps.mean(axis=1).astype(np.float32)
    theta_std = theta_samps.std(axis=1).astype(np.float32)
    return theta_mean, theta_std, theta_samps


def _bootstrap_summary(theta_true: np.ndarray, theta_mean: np.ndarray, theta_samples: np.ndarray, low: np.ndarray, high: np.ndarray, n_boot: int, seed: int) -> Dict[str, float]:
    N, P = theta_true.shape
    prange = np.where((high - low) < 1e-12, 1.0, (high - low)).astype(np.float64)
    q05 = np.quantile(theta_samples, 0.05, axis=1)
    q95 = np.quantile(theta_samples, 0.95, axis=1)

    mean_pearson = []
    mean_rmse_norm = []
    mean_cov90 = []
    for j in range(P):
        mean_pearson.append(_safe_pearson(theta_true[:, j], theta_mean[:, j]))
        mean_rmse_norm.append(float(np.sqrt(np.mean((theta_mean[:, j] - theta_true[:, j]) ** 2)) / prange[j]))
        mean_cov90.append(float(np.mean((theta_true[:, j] >= q05[:, j]) & (theta_true[:, j] <= q95[:, j]))))

    point = {
        "mean_pearson": float(np.mean(mean_pearson)),
        "mean_rmse_norm": float(np.mean(mean_rmse_norm)),
        "mean_cov90": float(np.mean(mean_cov90)),
    }

    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, N, size=(n_boot, N), dtype=np.int64)
    boot_p = np.zeros(n_boot, dtype=np.float64)
    boot_r = np.zeros(n_boot, dtype=np.float64)
    boot_c = np.zeros(n_boot, dtype=np.float64)

    for b in range(n_boot):
        idx = boot_idx[b]
        tp = theta_true[idx]
        tm = theta_mean[idx]
        lo90 = q05[idx]
        hi90 = q95[idx]
        pears = []
        rmses = []
        covs = []
        for j in range(P):
            pears.append(_safe_pearson(tp[:, j], tm[:, j]))
            rmses.append(float(np.sqrt(np.mean((tm[:, j] - tp[:, j]) ** 2)) / prange[j]))
            covs.append(float(np.mean((tp[:, j] >= lo90[:, j]) & (tp[:, j] <= hi90[:, j]))))
        boot_p[b] = np.mean(pears)
        boot_r[b] = np.mean(rmses)
        boot_c[b] = np.mean(covs)

    point.update(
        {
            "mean_pearson_lo": float(np.quantile(boot_p, 0.025)),
            "mean_pearson_hi": float(np.quantile(boot_p, 0.975)),
            "mean_rmse_norm_lo": float(np.quantile(boot_r, 0.025)),
            "mean_rmse_norm_hi": float(np.quantile(boot_r, 0.975)),
            "mean_cov90_lo": float(np.quantile(boot_c, 0.025)),
            "mean_cov90_hi": float(np.quantile(boot_c, 0.975)),
        }
    )
    return point


def _write_csv(path: str, rows: List[Dict[str, float]], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _plot_sweep(rows: List[Dict[str, float]], out_png: str, title: str, xlabel: str) -> None:
    xs = [r["level"] for r in rows]
    metrics = [
        ("mean_pearson", "Mean Pearson", (0.0, 1.0)),
        ("mean_rmse_norm", "Mean nRMSE", None),
        ("mean_cov90", "Mean 90% coverage", (0.0, 1.0)),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 9.0), sharex=True)
    for ax, (key, ylabel, ylim) in zip(axes, metrics):
        y = np.array([r[key] for r in rows], dtype=np.float64)
        lo = np.array([r[f"{key}_lo"] for r in rows], dtype=np.float64)
        hi = np.array([r[f"{key}_hi"] for r in rows], dtype=np.float64)
        ax.fill_between(xs, lo, hi, alpha=0.22)
        ax.plot(xs, y, marker="o", linewidth=1.8)
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel(xlabel)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_png.replace(".png", ".pdf"))
    plt.close(fig)


def main() -> None:
    tf = _import_tensorflow_or_die()
    _set_plot_defaults()

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dirs", nargs="+", required=True)
    ap.add_argument("--features", choices=["erp", "tfr", "hybrid"], required=True)
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-out", default=os.path.join(BASE_DIR, "data_out"))
    ap.add_argument("--n-post-samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-eval", type=int, default=0)
    ap.add_argument("--n-bootstrap", type=int, default=500)
    ap.add_argument("--noise-stds", type=str, default="0,0.01,0.02,0.05,0.1")
    ap.add_argument("--token-drop-probs", type=str, default="0,0.1,0.2,0.4")
    ap.add_argument("--channel-gain-stds", type=str, default="0,0.05,0.1,0.2")
    ap.add_argument("--channel-drop-probs", type=str, default="0,0.1,0.25,0.5")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    X_mem = np.load(_feature_path(args.data_out, args.features), mmap_mode="r")
    theta_mem = np.load(os.path.join(args.data_out, "params.npy"), mmap_mode="r")
    N, T, F = X_mem.shape

    pm = np.load(os.path.join(args.data_out, "param_meta.npz"))
    param_names = [x.decode("utf-8") for x in pm["param_names"]]
    low = pm["prior_low"].astype(np.float32)
    high = pm["prior_high"].astype(np.float32)

    eval_idx = _load_eval_indices(args.model_dirs[0], args.split)
    if args.n_eval and args.n_eval > 0 and args.n_eval < len(eval_idx):
        eval_idx = rng.choice(eval_idx, size=args.n_eval, replace=False).astype(np.int64)
    theta_true = np.asarray(theta_mem[eval_idx], dtype=np.float32)

    mu, sd, token_mask_1d = _load_scaler_and_mask(args.model_dirs[0], args.data_out, args.features, n_tokens=T)
    X = np.asarray(X_mem[eval_idx], dtype=np.float32)
    M = np.tile(token_mask_1d[None, :], (X.shape[0], 1)).astype(np.float32)
    X = (X - mu[None, None, :]) / sd[None, None, :]
    X *= M[:, :, None]

    models, used_paths = _load_models(tf, args.model_dirs)

    noise_stds = _parse_list(args.noise_stds, float)
    token_drops = _parse_list(args.token_drop_probs, float)
    gain_stds = _parse_list(args.channel_gain_stds, float)
    chan_drops = _parse_list(args.channel_drop_probs, float)

    def eval_once(Xp: np.ndarray, Mp: np.ndarray, seed_offset: int) -> Dict[str, float]:
        th_mean, _, th_samps = _predict_and_sample_mixture(
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
        return _bootstrap_summary(theta_true, th_mean, th_samps, low, high, n_boot=args.n_bootstrap, seed=int(args.seed + seed_offset + 50000))

    rows_noise = []
    for i, s in enumerate(noise_stds):
        Xp = X.copy()
        Mp = M.copy()
        if s > 0:
            Xp = Xp + rng.standard_normal(size=Xp.shape).astype(np.float32) * float(s) * Mp[:, :, None]
        rows_noise.append({"level": float(s), **eval_once(Xp, Mp, 1000 + i)})
    csv_noise = os.path.join(args.out_dir, "sweep_noise.csv")
    _write_csv(csv_noise, rows_noise, fieldnames=list(rows_noise[0].keys()))
    _plot_sweep(rows_noise, os.path.join(args.out_dir, "sweep_noise_mean_metrics.png"), "Generalization sweep: additive feature noise", "Noise std (scaled feature space)")

    rows_tok = []
    for i, p in enumerate(token_drops):
        Xp = X.copy()
        Mp = M.copy()
        if p > 0:
            keep = (rng.random(size=Mp.shape) >= float(p)).astype(np.float32)
            Mp = Mp * keep
            Xp = Xp * keep[:, :, None]
        rows_tok.append({"level": float(p), **eval_once(Xp, Mp, 2000 + i)})
    csv_tok = os.path.join(args.out_dir, "sweep_token_drop.csv")
    _write_csv(csv_tok, rows_tok, fieldnames=list(rows_tok[0].keys()))
    _plot_sweep(rows_tok, os.path.join(args.out_dir, "sweep_token_drop_mean_metrics.png"), "Generalization sweep: token dropout", "Token-drop probability")

    rows_gain = []
    for i, gs in enumerate(gain_stds):
        Xp = X.copy()
        Mp = M.copy()
        if gs > 0:
            gains = 1.0 + rng.standard_normal(size=(Xp.shape[0], F)).astype(np.float32) * float(gs)
            Xp = Xp * gains[:, None, :]
        rows_gain.append({"level": float(gs), **eval_once(Xp, Mp, 3000 + i)})
    csv_gain = os.path.join(args.out_dir, "sweep_channel_gain.csv")
    _write_csv(csv_gain, rows_gain, fieldnames=list(rows_gain[0].keys()))
    _plot_sweep(rows_gain, os.path.join(args.out_dir, "sweep_channel_gain_mean_metrics.png"), "Generalization sweep: per-channel gain jitter", "Channel-gain std")

    rows_cd = []
    for i, p in enumerate(chan_drops):
        Xp = X.copy()
        Mp = M.copy()
        if p > 0:
            keepc = (rng.random(size=(Xp.shape[0], F)) >= float(p)).astype(np.float32)
            Xp = Xp * keepc[:, None, :]
        rows_cd.append({"level": float(p), **eval_once(Xp, Mp, 4000 + i)})
    csv_cd = os.path.join(args.out_dir, "sweep_channel_drop.csv")
    _write_csv(csv_cd, rows_cd, fieldnames=list(rows_cd[0].keys()))
    _plot_sweep(rows_cd, os.path.join(args.out_dir, "sweep_channel_drop_mean_metrics.png"), "Generalization sweep: channel dropout", "Channel-drop probability")

    manifest = {
        "features": args.features,
        "split": args.split,
        "n_eval": int(len(eval_idx)),
        "n_post_samples": int(args.n_post_samples),
        "n_bootstrap": int(args.n_bootstrap),
        "param_names": param_names,
        "model_dirs": args.model_dirs,
        "loaded_model_paths": used_paths,
        "sweeps": {
            "noise_stds": noise_stds,
            "token_drop_probs": token_drops,
            "channel_gain_stds": gain_stds,
            "channel_drop_probs": chan_drops,
        },
    }
    with open(os.path.join(args.out_dir, "generalization_sweeps_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("[generalization_sweeps] wrote:", args.out_dir)


if __name__ == "__main__":
    main()
