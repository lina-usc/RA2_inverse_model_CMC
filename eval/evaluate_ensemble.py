
import argparse
import csv
import json
import os
from typing import Dict, List, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import spearmanr

import tensorflow as tf

# Keras 3 safe-mode blocks Lambda(lambdas) by default.
# We only load models that were trained locally in this repo, so this is safe here.
try:
    import keras
    try:
        keras.config.enable_unsafe_deserialization()
    except Exception:
        # some Keras builds expose this under keras.saving
        keras.saving.enable_unsafe_deserialization()
except Exception:
    pass


from models.param_transforms import theta_to_z, z_to_theta
from models.posterior_fullcov import mvn_tril_nll, raw_tril_size
from models.transformer_paramtoken import get_custom_objects


# ----------------------------- utilities -----------------------------

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _make_token_mask(features: str, n: int, n_tokens: int, n_tokens_erp: int) -> np.ndarray:
    m = np.ones((n, n_tokens), dtype=np.float32)
    if features == "hybrid":
        return m
    if features == "erp":
        m[:, n_tokens_erp:] = 0.0
        return m
    if features == "tfr":
        m[:, :n_tokens_erp] = 0.0
        return m
    raise ValueError("--features must be one of: hybrid, erp, tfr")


def _stable_softplus_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _tri_indices(p: int, ordering: str) -> Tuple[np.ndarray, np.ndarray]:
    rr, cc = np.tril_indices(p)
    if ordering == "row":
        pass  # already row-major: (0,0),(1,0),(1,1),(2,0),...
    elif ordering == "col":
        # column-major: (0,0),(1,0),(2,0),(1,1),(2,1),(2,2),...
        order = np.lexsort((rr, cc))  # primary key: cc ascending, then rr
        rr, cc = rr[order], cc[order]
    elif ordering == "tfp":  # bottom row first (TFP fill_triangular style)
        # (2,0),(2,1),(2,2),(1,0),(1,1),(0,0),...
        order = np.lexsort((cc, -rr))  # primary key: rr descending, then cc
        rr, cc = rr[order], cc[order]
    else:
        raise ValueError(f"Unknown ordering: {ordering}")
    return rr.astype(np.int32), cc.astype(np.int32)


def raw_tril_to_L_np(
    raw: np.ndarray,
    p: int,
    ordering: str,
    diag_transform: str,
    diag_eps: float,
    offdiag_scale: float,
) -> np.ndarray:
    """
    Build lower-tri L from raw_tril using a chosen packing + diag transform.
    raw: (N, n_tril)
    returns L: (N, p, p)
    """
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim == 1:
        raw = raw[None, :]
    n = raw.shape[0]
    n_tril = p * (p + 1) // 2
    if raw.shape[1] != n_tril:
        raise ValueError(f"raw_tril length {raw.shape[1]} != {n_tril} for P={p}")

    rr, cc = _tri_indices(p, ordering)
    L = np.zeros((n, p, p), dtype=np.float32)
    L[:, rr, cc] = raw

    # scale off-diagonal (strict lower) if requested
    if offdiag_scale != 1.0:
        mask = np.tril(np.ones((p, p), dtype=bool), k=-1)
        L[:, mask] *= float(offdiag_scale)

    # transform diagonal
    d = np.diagonal(L, axis1=1, axis2=2).copy()  # (N,P)
    if diag_transform == "softplus":
        d2 = _stable_softplus_np(d) + float(diag_eps)
    elif diag_transform == "exp":
        d2 = np.exp(d).astype(np.float32) + float(diag_eps)
    elif diag_transform == "abs":
        d2 = np.abs(d).astype(np.float32) + float(diag_eps)
    elif diag_transform == "none":
        d2 = d + float(diag_eps)
    else:
        raise ValueError(f"Unknown diag_transform: {diag_transform}")

    for i in range(p):
        L[:, i, i] = d2[:, i]
    return L


def nll_from_L_tf(
    z: tf.Tensor,  # (B,P)
    mu: tf.Tensor,  # (B,P)
    L: tf.Tensor,  # (B,P,P)
    logdet_mode: str,
    jitter: float = 1e-12,
) -> tf.Tensor:
    """
    Compute MVN NLL in z-space given L, without constant.
    This is only used to match the loss behavior for mapping inference.
    """
    diff = (z - mu)[:, :, None]  # (B,P,1)
    y = tf.linalg.triangular_solve(L, diff, lower=True)  # (B,P,1)
    quad = tf.reduce_sum(tf.square(y), axis=[1, 2])  # (B,)

    d = tf.linalg.diag_part(L)  # (B,P)
    if logdet_mode == "chol":
        logdet = 2.0 * tf.reduce_sum(tf.math.log(d + jitter), axis=1)
    elif logdet_mode == "abs":
        logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.abs(d) + jitter), axis=1)
    elif logdet_mode == "square":
        logdet = tf.reduce_sum(tf.math.log(tf.square(d) + jitter), axis=1)
    else:
        raise ValueError(f"Unknown logdet_mode: {logdet_mode}")

    return 0.5 * (quad + logdet)


def infer_tril_mapping(
    z_b: np.ndarray,
    mu_b: np.ndarray,
    raw_b: np.ndarray,
    nll_tf_ref: np.ndarray,
    p: int,
) -> Dict[str, object]:
    """
    Infer (ordering, diag_transform, diag_eps, offdiag_scale, logdet_mode)
    by minimizing max|nll_candidate - nll_tf_ref|.
    """
    z = tf.convert_to_tensor(z_b.astype(np.float32))
    mu = tf.convert_to_tensor(mu_b.astype(np.float32))

    orderings = ["row", "col", "tfp"]
    diag_transforms = ["none", "softplus", "exp", "abs"]
    diag_eps_list = [1e-6, 1e-5, 1e-4, 1e-3]
    offdiag_scales = [1.0, 0.1]
    logdet_modes = ["chol", "abs", "square"]

    best = None
    best_max = float("inf")
    best_mean = float("inf")

    for ordering in orderings:
        for diag_transform in diag_transforms:
            for diag_eps in diag_eps_list:
                for offdiag_scale in offdiag_scales:
                    for logdet_mode in logdet_modes:
                        L_np = raw_tril_to_L_np(
                            raw_b, p=p,
                            ordering=ordering,
                            diag_transform=diag_transform,
                            diag_eps=float(diag_eps),
                            offdiag_scale=float(offdiag_scale),
                        )
                        L = tf.convert_to_tensor(L_np)
                        nll_c = nll_from_L_tf(z, mu, L, logdet_mode=logdet_mode).numpy().astype(np.float32)
                        diff = np.abs(nll_c - nll_tf_ref)
                        mx = float(np.max(diff))
                        mn = float(np.mean(diff))
                        if (mx < best_max) or (mx == best_max and mn < best_mean):
                            best_max = mx
                            best_mean = mn
                            best = {
                                "ordering": ordering,
                                "diag_transform": diag_transform,
                                "diag_eps": float(diag_eps),
                                "offdiag_scale": float(offdiag_scale),
                                "logdet_mode": logdet_mode,
                                "max_abs_diff": best_max,
                                "mean_abs_diff": best_mean,
                            }

    if best is None:
        raise RuntimeError("Failed to infer mapping.")
    return best


def mixture_nll_tf(
    z: np.ndarray,
    mus: List[np.ndarray],
    raws: List[np.ndarray],
) -> np.ndarray:
    """
    Equal-weight mixture NLL in z-space using *your* mvn_tril_nll exactly.
    """
    z_tf = tf.convert_to_tensor(z.astype(np.float32))
    logps = []
    for mu, raw in zip(mus, raws):
        mu_tf = tf.convert_to_tensor(mu.astype(np.float32))
        raw_tf = tf.convert_to_tensor(raw.astype(np.float32))
        nll = mvn_tril_nll(z_tf, mu_tf, raw_tf, include_const=False)
        logps.append(-nll)
    logps_tf = tf.stack(logps, axis=0)  # (M,N)
    logp_mix = tf.reduce_logsumexp(logps_tf, axis=0) - tf.math.log(float(len(mus)))
    return (-logp_mix).numpy().astype(np.float32)


def sample_mixture_theta(
    rng: np.random.Generator,
    mus: List[np.ndarray],
    raws: List[np.ndarray],
    low: np.ndarray,
    high: np.ndarray,
    mapping: Dict[str, object],
    n_samples: int,
) -> np.ndarray:
    """
    Sample from equal-weight mixture posterior in z-space and transform to theta.

    Returns theta_samps: (N, n_samples, P)
    """
    M = len(mus)
    N, P = mus[0].shape

    ordering = str(mapping["ordering"])
    diag_transform = str(mapping["diag_transform"])
    diag_eps = float(mapping["diag_eps"])
    offdiag_scale = float(mapping["offdiag_scale"])

    # Build L for each component
    Ls = [
        raw_tril_to_L_np(raws[m], p=P, ordering=ordering, diag_transform=diag_transform, diag_eps=diag_eps, offdiag_scale=offdiag_scale)
        for m in range(M)
    ]

    eps = rng.standard_normal(size=(N, n_samples, P)).astype(np.float32)
    comp = rng.integers(0, M, size=(N, n_samples))

    theta_samps = np.empty((N, n_samples, P), dtype=np.float32)

    for m in range(M):
        L = Ls[m]  # (N,P,P)
        z_m = mus[m][:, None, :] + np.einsum("nij,nkj->nki", L, eps)  # (N,K,P)
        sel = (comp == m)
        if np.any(sel):
            z_sel = z_m[sel].reshape(-1, P)
            th_sel = z_to_theta(z_sel, low, high).astype(np.float32)
            theta_samps.reshape(-1, P)[sel.reshape(-1)] = th_sel

    return theta_samps


# ----------------------------- main -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dirs", nargs="+", required=True)
    ap.add_argument("--features", choices=["hybrid", "erp", "tfr"], required=True)
    ap.add_argument("--data-out", type=str, default="data_out")
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--n-post-samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    _ensure_dir(args.out_dir)

    # ---- load arrays ----
    X_mem = np.load(os.path.join(args.data_out, "features.npy"), mmap_mode="r")
    theta_mem = np.load(os.path.join(args.data_out, "params.npy"), mmap_mode="r")
    splits = np.load(os.path.join(args.data_out, "splits.npz"))
    idx = splits[f"{args.split}_idx"]

    meta = np.load(os.path.join(args.data_out, "tfr_meta.npz"))
    param_meta = np.load(os.path.join(args.data_out, "param_meta.npz"))

    param_names = [x.decode("utf-8") for x in param_meta["param_names"]]
    low = param_meta["prior_low"].astype(np.float32)
    high = param_meta["prior_high"].astype(np.float32)

    n_tokens = int(X_mem.shape[1])
    feature_dim = int(X_mem.shape[2])
    P = int(theta_mem.shape[1])
    n_tokens_erp = int(meta["n_tokens_erp"]) if "n_tokens_erp" in meta.files else int(meta["n_time_patches"])

    X = np.asarray(X_mem[idx], dtype=np.float32)
    theta_true = np.asarray(theta_mem[idx], dtype=np.float32)
    z_true = theta_to_z(theta_true, low, high).astype(np.float32)

    mask = _make_token_mask(args.features, n=X.shape[0], n_tokens=n_tokens, n_tokens_erp=n_tokens_erp)

    # ---- scaler from first model dir ----
    sc_path = os.path.join(args.model_dirs[0], "scaler_stats.npz")
    if not os.path.exists(sc_path):
        raise SystemExit(f"Missing scaler_stats.npz in {args.model_dirs[0]}")
    sc = np.load(sc_path)
    mu = sc["mu"].astype(np.float32)
    sd = sc["sd"].astype(np.float32)
    sd = np.where(sd < 1e-6, 1.0, sd).astype(np.float32)

    Xs = (X - mu[None, None, :]) / sd[None, None, :]

    # ---- load models & predict ----
    custom = get_custom_objects()
    mus: List[np.ndarray] = []
    raws: List[np.ndarray] = []

    expected_tril = raw_tril_size(P)

    for md in args.model_dirs:
        best = os.path.join(md, "paramtoken_best.keras")
        final = os.path.join(md, "paramtoken_final.keras")
        path = best if os.path.exists(best) else final
        if not os.path.exists(path):
            raise SystemExit(f"Missing model in {md}: expected paramtoken_best.keras or paramtoken_final.keras")

        # NOPARAMTOKEN_REBUILD_FOR_EVAL: noparamtoken .keras contains Lambda(lambdas) that may deserialize with broken tf globals.
        # Workaround: load the saved model ONLY to extract weights, rebuild the architecture from model_config.npz, then copy weights.
        cfg_path = os.path.join(md, "model_config.npz")
        arch = None
        cfg = None
        if os.path.exists(cfg_path):
            try:
                cfg = np.load(cfg_path, allow_pickle=False)
                a = cfg.get("arch", None)
                if a is not None:
                    a0 = a.item() if hasattr(a, "item") else a
                    arch = a0.decode("utf-8") if isinstance(a0, (bytes, np.bytes_)) else str(a0)
            except Exception:
                cfg = None
                arch = None

        if arch == "noparamtoken" and cfg is not None:
            from models.transformer_noparamtoken import build_noparamtoken_transformer
            # Load to get weights (do NOT run a forward pass with this loaded model).
            model_bad = tf.keras.models.load_model(path, custom_objects=custom, compile=False)

            post = cfg["posterior"]
            post0 = post.item() if hasattr(post, "item") else post
            posterior = post0.decode("utf-8") if isinstance(post0, (bytes, np.bytes_)) else str(post0)

            model = build_noparamtoken_transformer(
                n_tokens=int(cfg["n_tokens"]),
                feature_dim=int(cfg["feature_dim"]),
                n_params=int(cfg["n_params"]),
                n_time_patches=int(cfg["n_time_patches"]),
                n_freq_patches=int(cfg["n_freq_patches"]),
                n_tokens_erp=int(cfg["n_tokens_erp"]),
                d_model=int(cfg["d_model"]),
                num_layers=int(cfg["num_layers"]),
                num_heads=int(cfg["num_heads"]),
                ff_dim=int(cfg["ff_dim"]),
                dropout_rate=float(cfg["dropout_rate"]),
                posterior=posterior,
                return_attention=False,
            )
            model.set_weights(model_bad.get_weights())
            del model_bad
        else:
            model = tf.keras.models.load_model(path, custom_objects=custom, compile=False)
        # PATCH_LAMBDA_TF_GLOBALS: Keras Lambda deserialization may turn `tf` into a dict; restore module refs.
        try:
            import keras
            for _ly in model.layers:
                if _ly.__class__.__name__ == "Lambda":
                    _fn = getattr(_ly, "function", None)
                    if _fn is not None and hasattr(_fn, "__globals__"):
                        _g = _fn.__globals__
                        if isinstance(_g.get("tf", None), dict):
                            _g["tf"] = tf
                        if isinstance(_g.get("np", None), dict):
                            _g["np"] = np
        except Exception:
            pass
        pred = model.predict([Xs, mask], batch_size=args.batch_size, verbose=0)
        pred = np.asarray(pred, dtype=np.float32)

        mu_z = pred[:, :P]
        raw = pred[:, P:]
        # --- diag posterior support: (mu + logvar) -> diagonal raw_tril so full-cov eval code can run ---
        if raw.shape[1] == P:
            # raw is log-variance per dim. Convert to diagonal Cholesky raw_tril (row-order).
            logvar = np.clip(raw.astype(np.float64), -10.0, 10.0)
            sigma = np.exp(0.5 * logvar)  # std in z-space
            diag_eps = 1e-3
            y = np.maximum(sigma - diag_eps, 1e-6)
            # inv_softplus(y): stable for large y (softplus(x)≈x for x>20)
            raw_diag = np.where(y > 20.0, y, np.log(np.expm1(y)))
            raw_full = np.zeros((raw.shape[0], expected_tril), dtype=np.float32)
            diag_idx = [ii * (ii + 3) // 2 for ii in range(P)]  # row-order diag indices
            raw_full[:, diag_idx] = raw_diag.astype(np.float32)
            raw = raw_full
        elif raw.shape[1] != expected_tril:
            raise ValueError(f"{md}: raw_tril dim {raw.shape[1]} != expected {expected_tril}")

        mus.append(mu_z)
        raws.append(raw)

    # ---- infer mapping using first model on a small batch ----
    b = min(32, z_true.shape[0])
    z_b = z_true[:b]
    mu_b = mus[0][:b]
    raw_b = raws[0][:b]

    # reference NLL from the *training* implementation
    nll_ref = mvn_tril_nll(
        tf.convert_to_tensor(z_b),
        tf.convert_to_tensor(mu_b),
        tf.convert_to_tensor(raw_b),
        include_const=False,
    ).numpy().astype(np.float32)

    mapping = infer_tril_mapping(z_b, mu_b, raw_b, nll_ref, p=P)

    # hard guard: if we can't match closely, we stop and ask for posterior_fullcov.py
    if float(mapping["max_abs_diff"]) > 1e-3:
        raise SystemExit(
            "[evaluate_ensemble] ERROR: Could not infer raw_tril mapping tightly.\n"
            f"best_mapping={mapping}\n"
            "Paste models/posterior_fullcov.py so we match it exactly."
        )

    print("[evaluate_ensemble] inferred mapping:", json.dumps(mapping, indent=2))

    # ---- mixture NLL in z-space using your exact mvn_tril_nll ----
    nll_mix = mixture_nll_tf(z_true, mus, raws)

    # ---- posterior samples (mixture) in theta ----
    rng = np.random.default_rng(args.seed)
    theta_samps = sample_mixture_theta(
        rng=rng,
        mus=mus,
        raws=raws,
        low=low,
        high=high,
        mapping=mapping,
        n_samples=int(args.n_post_samples),
    )

    theta_mean = theta_samps.mean(axis=1).astype(np.float32)
    theta_med = np.median(theta_samps, axis=1).astype(np.float32)
    theta_std = theta_samps.std(axis=1).astype(np.float32)

    # ---- metrics per parameter ----
    rows = []
    for pi, name in enumerate(param_names):
        err = theta_mean[:, pi] - theta_true[:, pi]
        mse = float(np.mean(err * err))
        rmse = float(np.sqrt(mse))
        r_pear = safe_pearson(theta_true[:, pi], theta_mean[:, pi])
        r_spear = float(spearmanr(theta_true[:, pi], theta_mean[:, pi]).correlation)

        # coverage at 50% and 90%
        covs = {}
        for cov_level in [0.50, 0.90]:
            lo_q = (1.0 - cov_level) / 2.0
            hi_q = 1.0 - lo_q
            lo = np.quantile(theta_samps[:, :, pi], lo_q, axis=1)
            hi = np.quantile(theta_samps[:, :, pi], hi_q, axis=1)
            inside = (theta_true[:, pi] >= lo) & (theta_true[:, pi] <= hi)
            covs[cov_level] = float(np.mean(inside))

        rows.append(
            {
                "param": name,
                "rmse_mean": rmse,
                "mse_mean": mse,
                "pearson_mean": r_pear,
                "spearman_mean": r_spear,
                "cov50": covs[0.50],
                "cov90": covs[0.90],
                "mean_post_std": float(np.mean(theta_std[:, pi])),
            }
        )

    # ---- write metrics CSV ----
    metrics_csv = os.path.join(args.out_dir, f"metrics_{args.split}.csv")
    with open(metrics_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # ---- reliability curves ----
    levels = np.linspace(0.05, 0.95, 19)
    ncols = 3
    nrows = int(np.ceil(P / ncols))

    fig = plt.figure(figsize=(12, 10))
    for pi, name in enumerate(param_names):
        ax = fig.add_subplot(nrows, ncols, pi + 1)
        emp = []
        for cov_level in levels:
            lo_q = (1.0 - cov_level) / 2.0
            hi_q = 1.0 - lo_q
            lo = np.quantile(theta_samps[:, :, pi], lo_q, axis=1)
            hi = np.quantile(theta_samps[:, :, pi], hi_q, axis=1)
            inside = (theta_true[:, pi] >= lo) & (theta_true[:, pi] <= hi)
            emp.append(float(np.mean(inside)))
        ax.plot(levels, emp)
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        ax.set_title(name)
        ax.set_xlabel("nominal")
        ax.set_ylabel("empirical")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.suptitle(f"Reliability curves ({args.features}, {args.split})", y=0.98)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, f"reliability_{args.split}.png"), dpi=200)
    plt.close(fig)

    # ---- SBC rank histograms ----
    K = int(args.n_post_samples)
    sbc_bins = 20
    fig = plt.figure(figsize=(12, 10))
    for pi, name in enumerate(param_names):
        ax = fig.add_subplot(nrows, ncols, pi + 1)
        ranks = np.sum(theta_samps[:, :, pi] < theta_true[:, pi][:, None], axis=1).astype(np.int32)
        bins = np.linspace(0, K + 1, sbc_bins + 1)
        ax.hist(ranks, bins=bins, density=False)
        ax.set_title(name)
        ax.set_xlabel("rank")
        ax.set_ylabel("count")
    fig.suptitle(f"SBC rank histograms ({args.features}, {args.split})", y=0.98)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, f"sbc_rankhist_{args.split}.png"), dpi=200)
    plt.close(fig)

    # ---- scatter true vs posterior mean ----
    fig = plt.figure(figsize=(12, 10))
    for pi, name in enumerate(param_names):
        ax = fig.add_subplot(nrows, ncols, pi + 1)
        ax.scatter(theta_true[:, pi], theta_mean[:, pi], s=6, alpha=0.4)
        r = safe_pearson(theta_true[:, pi], theta_mean[:, pi])
        ax.set_title(f"{name} (r={r:.2f})")
        ax.set_xlabel("true")
        ax.set_ylabel("post mean")
    fig.suptitle(f"True vs posterior mean ({args.features}, {args.split})", y=0.98)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, f"scatter_true_vs_mean_{args.split}.png"), dpi=200)
    plt.close(fig)

    # ---- NLL histogram ----
    fig = plt.figure(figsize=(6, 4))
    plt.hist(nll_mix, bins=40)
    plt.title(f"Mixture NLL(z) no-const ({args.features}, {args.split})")
    plt.xlabel("nll")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"nllz_hist_{args.split}.png"), dpi=200)
    plt.close(fig)

    # ---- save eval outputs ----
    out_npz = os.path.join(args.out_dir, f"eval_{args.split}_outputs.npz")
    np.savez(
        out_npz,
        eval_idx=np.asarray(idx, dtype=np.int64),
        param_names=np.array(param_names, dtype="S"),
        theta_true=theta_true.astype(np.float32),
        theta_mean=theta_mean.astype(np.float32),
        theta_median=theta_med.astype(np.float32),
        theta_std=theta_std.astype(np.float32),
        theta_samples=np.asarray(theta_samps, dtype=np.float32),
        nll_z_noconst=nll_mix.astype(np.float32),
        n_post_samples=np.array(K, dtype=np.int32),
        split=np.array(args.split, dtype="S"),
        features=np.array(args.features, dtype="S"),
        mapping_json=np.array(json.dumps(mapping), dtype="S"),
    )

    summary = {
        "features": args.features,
        "split": args.split,
        "n_models": len(args.model_dirs),
        "n_eval": int(theta_true.shape[0]),
        "mean_nll_z_noconst": float(np.mean(nll_mix)),
        "median_nll_z_noconst": float(np.median(nll_mix)),
        "metrics_csv": os.path.basename(metrics_csv),
        "out_npz": os.path.basename(out_npz),
        "mapping": mapping,
    }
    with open(os.path.join(args.out_dir, f"summary_{args.split}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("[evaluate_ensemble] wrote:", args.out_dir)
    print("[evaluate_ensemble] summary:", json.dumps(summary, indent=2))
    print("[evaluate_ensemble] metrics csv:", metrics_csv)


if __name__ == "__main__":
    main()
