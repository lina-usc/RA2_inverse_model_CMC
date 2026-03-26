from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------
# helpers
# -------------------------

def _decode_str_array(x) -> List[str]:
    x = np.asarray(x)
    out: List[str] = []
    for v in x:
        if isinstance(v, (bytes, np.bytes_)):
            out.append(v.decode("utf-8"))
        else:
            out.append(str(v))
    return out


def _pick(npz, keys: List[str], required: bool = True):
    for k in keys:
        if k in npz.files:
            return npz[k]
    if required:
        raise KeyError(f"None of keys found: {keys}. Available: {list(npz.files)}")
    return None


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum()) + 1e-12
    return float((x * y).sum() / denom)


def _rmse(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return float(np.sqrt(np.mean((x - y) ** 2)))


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


def _load_eval_dir(eval_dir: str, split: str) -> Dict[str, np.ndarray]:
    path = os.path.join(eval_dir, f"eval_{split}_outputs.npz")
    if not os.path.isfile(path):
        path = os.path.join(eval_dir, "eval_test_outputs.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Could not find eval outputs in {eval_dir}")

    d = np.load(path, allow_pickle=True)
    theta_true = _pick(d, ["theta_true", "theta", "y_true_theta", "theta_test"])
    theta_mean = _pick(d, ["theta_post_mean", "theta_mean", "post_mean_theta", "theta_mu", "theta_pred_mean"])
    theta_samps = _pick(d, ["theta_post_samples", "theta_samples", "post_samples_theta", "samples_theta"], required=False)
    param_names = _pick(d, ["param_names", "param_names_bytes"], required=False)
    eval_idx = _pick(d, ["eval_idx"], required=False)

    if theta_true.shape != theta_mean.shape:
        raise ValueError(f"Shape mismatch in {path}: theta_true={theta_true.shape}, theta_mean={theta_mean.shape}")

    names = _decode_str_array(param_names) if param_names is not None else [f"param_{i}" for i in range(theta_true.shape[1])]

    out = {
        "path": np.array(path, dtype=object),
        "theta_true": np.asarray(theta_true, dtype=np.float32),
        "theta_mean": np.asarray(theta_mean, dtype=np.float32),
        "param_names": np.array(names, dtype=object),
    }
    if theta_samps is not None:
        out["theta_samples"] = np.asarray(theta_samps, dtype=np.float32)
    if eval_idx is not None:
        out["eval_idx"] = np.asarray(eval_idx, dtype=np.int64)
    return out


def _check_alignment(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray], label_a: str, label_b: str) -> None:
    if a["theta_true"].shape != b["theta_true"].shape:
        raise ValueError(f"{label_a} and {label_b} have different theta_true shapes")
    if not np.allclose(a["theta_true"], b["theta_true"], atol=1e-6):
        if "eval_idx" in a and "eval_idx" in b and np.array_equal(a["eval_idx"], b["eval_idx"]):
            raise ValueError(f"{label_a} and {label_b} eval_idx match but theta_true values differ")
        raise ValueError(f"{label_a} and {label_b} do not appear aligned; check eval ordering")
    if list(a["param_names"].tolist()) != list(b["param_names"].tolist()):
        raise ValueError(f"{label_a} and {label_b} param_names differ")


def _point_metrics(theta_true: np.ndarray, theta_mean: np.ndarray, prange: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    P = theta_true.shape[1]
    pear = np.zeros(P, dtype=np.float64)
    rmse_norm = np.zeros(P, dtype=np.float64)
    for j in range(P):
        pear[j] = _safe_pearson(theta_true[:, j], theta_mean[:, j])
        rmse_norm[j] = _rmse(theta_true[:, j], theta_mean[:, j]) / float(prange[j])
    return pear, rmse_norm


def _bootstrap_metrics(theta_true: np.ndarray, theta_mean: np.ndarray, prange: np.ndarray, n_boot: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    N, P = theta_true.shape
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, N, size=(n_boot, N), dtype=np.int64)

    pear_lo = np.zeros(P, dtype=np.float64)
    pear_hi = np.zeros(P, dtype=np.float64)
    rmse_lo = np.zeros(P, dtype=np.float64)
    rmse_hi = np.zeros(P, dtype=np.float64)

    for j in range(P):
        x = theta_true[:, j].astype(np.float64)
        y = theta_mean[:, j].astype(np.float64)
        xb = x[boot_idx]
        yb = y[boot_idx]

        xm = xb - xb.mean(axis=1, keepdims=True)
        ym = yb - yb.mean(axis=1, keepdims=True)
        denom = np.sqrt(np.sum(xm * xm, axis=1) * np.sum(ym * ym, axis=1)) + 1e-12
        r_boot = np.sum(xm * ym, axis=1) / denom

        rmse_boot = np.sqrt(np.mean((xb - yb) ** 2, axis=1)) / float(prange[j])

        pear_lo[j], pear_hi[j] = np.quantile(r_boot, [0.025, 0.975])
        rmse_lo[j], rmse_hi[j] = np.quantile(rmse_boot, [0.025, 0.975])

    return pear_lo, pear_hi, rmse_lo, rmse_hi


def _bootstrap_hybrid_gain(
    theta_true: np.ndarray,
    theta_mean_erp: np.ndarray,
    theta_mean_tfr: np.ndarray,
    theta_mean_hyb: np.ndarray,
    n_boot: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    N, P = theta_true.shape
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, N, size=(n_boot, N), dtype=np.int64)
    lo = np.zeros(P, dtype=np.float64)
    hi = np.zeros(P, dtype=np.float64)

    for j in range(P):
        x = theta_true[:, j].astype(np.float64)
        xe = theta_mean_erp[:, j].astype(np.float64)
        xt = theta_mean_tfr[:, j].astype(np.float64)
        xh = theta_mean_hyb[:, j].astype(np.float64)

        xb = x[boot_idx]
        eb = xe[boot_idx]
        tb = xt[boot_idx]
        hb = xh[boot_idx]

        def corr_boot(yb_: np.ndarray) -> np.ndarray:
            xm = xb - xb.mean(axis=1, keepdims=True)
            ym = yb_ - yb_.mean(axis=1, keepdims=True)
            denom = np.sqrt(np.sum(xm * xm, axis=1) * np.sum(ym * ym, axis=1)) + 1e-12
            return np.sum(xm * ym, axis=1) / denom

        re = corr_boot(eb)
        rt = corr_boot(tb)
        rh = corr_boot(hb)
        gain = rh - np.maximum(re, rt)
        lo[j], hi[j] = np.quantile(gain, [0.025, 0.975])

    return lo, hi


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare ERP/TFR/Hybrid recoverability with readable figures and bootstrap CIs.")
    ap.add_argument("--data-out", type=str, default="data_out")
    ap.add_argument("--erp-dir", type=str, required=True)
    ap.add_argument("--tfr-dir", type=str, required=True)
    ap.add_argument("--hybrid-dir", type=str, required=True)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--out-dir", type=str, default="plots/compare_features")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    _set_plot_defaults()
    os.makedirs(args.out_dir, exist_ok=True)

    param_meta = np.load(os.path.join(args.data_out, "param_meta.npz"), allow_pickle=True)
    param_names = _decode_str_array(param_meta["param_names"])
    low = param_meta["prior_low"].astype(np.float32)
    high = param_meta["prior_high"].astype(np.float32)
    prange = (high - low).astype(np.float32)
    prange = np.where(prange < 1e-9, 1.0, prange)

    erp = _load_eval_dir(args.erp_dir, args.split)
    tfr = _load_eval_dir(args.tfr_dir, args.split)
    hyb = _load_eval_dir(args.hybrid_dir, args.split)

    _check_alignment(erp, tfr, "ERP", "TFR")
    _check_alignment(erp, hyb, "ERP", "Hybrid")

    feats = ["ERP", "TFR", "Hybrid"]
    theta_true = erp["theta_true"]
    means = [erp["theta_mean"], tfr["theta_mean"], hyb["theta_mean"]]

    P = len(param_names)
    pear = np.zeros((P, 3), dtype=np.float64)
    rmse_norm = np.zeros((P, 3), dtype=np.float64)
    pear_lo = np.zeros((P, 3), dtype=np.float64)
    pear_hi = np.zeros((P, 3), dtype=np.float64)
    rmse_lo = np.zeros((P, 3), dtype=np.float64)
    rmse_hi = np.zeros((P, 3), dtype=np.float64)

    for j, theta_mean in enumerate(means):
        pear[:, j], rmse_norm[:, j] = _point_metrics(theta_true, theta_mean, prange)
        plo, phi, rlo, rhi = _bootstrap_metrics(theta_true, theta_mean, prange, args.n_bootstrap, args.seed + 101 * (j + 1))
        pear_lo[:, j], pear_hi[:, j] = plo, phi
        rmse_lo[:, j], rmse_hi[:, j] = rlo, rhi

    gain = pear[:, 2] - np.maximum(pear[:, 0], pear[:, 1])
    gain_lo, gain_hi = _bootstrap_hybrid_gain(
        theta_true,
        erp["theta_mean"],
        tfr["theta_mean"],
        hyb["theta_mean"],
        n_boot=args.n_bootstrap,
        seed=args.seed + 777,
    )

    table_csv = os.path.join(args.out_dir, f"recoverability_table_{args.split}.csv")
    with open(table_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "param",
            "pearson_erp", "pearson_erp_lo", "pearson_erp_hi",
            "pearson_tfr", "pearson_tfr_lo", "pearson_tfr_hi",
            "pearson_hybrid", "pearson_hybrid_lo", "pearson_hybrid_hi",
            "rmse_norm_erp", "rmse_norm_erp_lo", "rmse_norm_erp_hi",
            "rmse_norm_tfr", "rmse_norm_tfr_lo", "rmse_norm_tfr_hi",
            "rmse_norm_hybrid", "rmse_norm_hybrid_lo", "rmse_norm_hybrid_hi",
            "hybrid_gain_over_best_single", "hybrid_gain_lo", "hybrid_gain_hi",
            "best_feature_by_pearson",
        ])
        for i, nm in enumerate(param_names):
            best = feats[int(np.nanargmax(pear[i]))]
            w.writerow([
                nm,
                pear[i, 0], pear_lo[i, 0], pear_hi[i, 0],
                pear[i, 1], pear_lo[i, 1], pear_hi[i, 1],
                pear[i, 2], pear_lo[i, 2], pear_hi[i, 2],
                rmse_norm[i, 0], rmse_lo[i, 0], rmse_hi[i, 0],
                rmse_norm[i, 1], rmse_lo[i, 1], rmse_hi[i, 1],
                rmse_norm[i, 2], rmse_lo[i, 2], rmse_hi[i, 2],
                gain[i], gain_lo[i], gain_hi[i],
                best,
            ])

    fig_h = max(5.8, 0.62 * P + 1.6)
    annot_fs = 8.0

    fig, ax = plt.subplots(figsize=(9.4, fig_h))
    im = ax.imshow(pear, aspect="auto", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(3))
    ax.set_xticklabels(feats)
    ax.set_yticks(range(P))
    ax.set_yticklabels(param_names)
    ax.set_title(f"Recoverability (Pearson correlation) [{args.split}]", pad=10)
    for i in range(P):
        for j in range(3):
            ax.text(j, i, f"{pear[i, j]:.2f}", ha="center", va="center", fontsize=annot_fs)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Pearson correlation")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, f"recoverability_pearson_heatmap_{args.split}.png"), dpi=args.dpi)
    fig.savefig(os.path.join(args.out_dir, f"recoverability_pearson_heatmap_{args.split}.pdf"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.4, fig_h))
    vmax = max(0.2, float(np.nanmax(rmse_norm)) * 1.05)
    im = ax.imshow(rmse_norm, aspect="auto", vmin=0.0, vmax=vmax)
    ax.set_xticks(range(3))
    ax.set_xticklabels(feats)
    ax.set_yticks(range(P))
    ax.set_yticklabels(param_names)
    ax.set_title(f"Recoverability (nRMSE) [{args.split}]", pad=10)
    for i in range(P):
        for j in range(3):
            ax.text(j, i, f"{rmse_norm[i, j]:.3f}", ha="center", va="center", fontsize=annot_fs)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Normalized RMSE")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, f"recoverability_rmse_norm_heatmap_{args.split}.png"), dpi=args.dpi)
    fig.savefig(os.path.join(args.out_dir, f"recoverability_rmse_norm_heatmap_{args.split}.pdf"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, max(5.6, 0.52 * P + 1.6)))
    y = np.arange(P)
    xerr = np.vstack([gain - gain_lo, gain_hi - gain])
    ax.barh(y, gain, xerr=xerr, capsize=3)
    ax.axvline(0.0, linestyle="--", linewidth=1.2, color="black")
    ax.set_yticks(y)
    ax.set_yticklabels(param_names)
    ax.invert_yaxis()
    ax.set_xlabel("Hybrid Pearson gain over best single-view representation")
    ax.set_title(f"Hybrid gain with 95% bootstrap intervals [{args.split}]", pad=10)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, f"hybrid_gain_over_best_single_{args.split}.png"), dpi=args.dpi)
    fig.savefig(os.path.join(args.out_dir, f"hybrid_gain_over_best_single_{args.split}.pdf"))
    plt.close(fig)

    summary = {
        "split": args.split,
        "n_eval": int(theta_true.shape[0]),
        "n_bootstrap": int(args.n_bootstrap),
        "erp_eval": str(erp["path"].item()),
        "tfr_eval": str(tfr["path"].item()),
        "hybrid_eval": str(hyb["path"].item()),
        "plots": [
            f"recoverability_pearson_heatmap_{args.split}.png",
            f"recoverability_rmse_norm_heatmap_{args.split}.png",
            f"hybrid_gain_over_best_single_{args.split}.png",
        ],
    }
    with open(os.path.join(args.out_dir, f"summary_compare_{args.split}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("[compare_feature_sets] wrote:", args.out_dir)
    print("[compare_feature_sets] table:", table_csv)


if __name__ == "__main__":
    main()
