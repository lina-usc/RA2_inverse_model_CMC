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
    param_names = _pick(d, ["param_names", "param_names_bytes"], required=False)
    eval_idx = _pick(d, ["eval_idx"], required=False)
    names = _decode_str_array(param_names) if param_names is not None else [f"param_{i}" for i in range(theta_true.shape[1])]
    out = {
        "path": np.array(path, dtype=object),
        "theta_true": np.asarray(theta_true, dtype=np.float32),
        "theta_mean": np.asarray(theta_mean, dtype=np.float32),
        "param_names": np.array(names, dtype=object),
    }
    if eval_idx is not None:
        out["eval_idx"] = np.asarray(eval_idx, dtype=np.int64)
    return out


def _check_alignment(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray], label_a: str, label_b: str) -> None:
    if a["theta_true"].shape != b["theta_true"].shape:
        raise ValueError(f"{label_a} and {label_b} have different theta_true shapes")
    if not np.allclose(a["theta_true"], b["theta_true"], atol=1e-6):
        raise ValueError(f"{label_a} and {label_b} theta_true differ; check eval alignment")
    if list(a["param_names"].tolist()) != list(b["param_names"].tolist()):
        raise ValueError(f"{label_a} and {label_b} param_names differ")


def _load_param_range(data_out: str) -> np.ndarray:
    pm = np.load(os.path.join(data_out, "param_meta.npz"), allow_pickle=True)
    low = pm["prior_low"].astype(np.float32)
    high = pm["prior_high"].astype(np.float32)
    prange = (high - low).astype(np.float32)
    return np.where(prange < 1e-9, 1.0, prange)


def _point_metrics(theta_true: np.ndarray, theta_mean: np.ndarray, prange: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    P = theta_true.shape[1]
    pear = np.zeros(P, dtype=np.float64)
    rmse_norm = np.zeros(P, dtype=np.float64)
    for j in range(P):
        pear[j] = _safe_pearson(theta_true[:, j], theta_mean[:, j])
        rmse_norm[j] = _rmse(theta_true[:, j], theta_mean[:, j]) / float(prange[j])
    return pear, rmse_norm


def _bootstrap_deltas(theta_true: np.ndarray, mean_a: np.ndarray, mean_b: np.ndarray, prange: np.ndarray, n_boot: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    N, P = theta_true.shape
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, N, size=(n_boot, N), dtype=np.int64)
    dp_lo = np.zeros(P, dtype=np.float64)
    dp_hi = np.zeros(P, dtype=np.float64)
    dr_lo = np.zeros(P, dtype=np.float64)
    dr_hi = np.zeros(P, dtype=np.float64)
    for j in range(P):
        x = theta_true[:, j].astype(np.float64)
        ya = mean_a[:, j].astype(np.float64)
        yb = mean_b[:, j].astype(np.float64)
        xb = x[boot_idx]
        yab = ya[boot_idx]
        ybb = yb[boot_idx]

        xm = xb - xb.mean(axis=1, keepdims=True)
        yadm = yab - yab.mean(axis=1, keepdims=True)
        ybdm = ybb - ybb.mean(axis=1, keepdims=True)
        da = np.sum(xm * yadm, axis=1) / (np.sqrt(np.sum(xm * xm, axis=1) * np.sum(yadm * yadm, axis=1)) + 1e-12)
        db = np.sum(xm * ybdm, axis=1) / (np.sqrt(np.sum(xm * xm, axis=1) * np.sum(ybdm * ybdm, axis=1)) + 1e-12)
        dpear = da - db

        rma = np.sqrt(np.mean((xb - yab) ** 2, axis=1)) / float(prange[j])
        rmb = np.sqrt(np.mean((xb - ybb) ** 2, axis=1)) / float(prange[j])
        drmse = rma - rmb

        dp_lo[j], dp_hi[j] = np.quantile(dpear, [0.025, 0.975])
        dr_lo[j], dr_hi[j] = np.quantile(drmse, [0.025, 0.975])
    return dp_lo, dp_hi, dr_lo, dr_hi


def main() -> None:
    ap = argparse.ArgumentParser(description="Architecture comparison with readable plots and bootstrap intervals.")
    ap.add_argument("--data-out", type=str, default="data_out")
    ap.add_argument("--transformer-erp-dir", required=True)
    ap.add_argument("--transformer-tfr-dir", required=True)
    ap.add_argument("--transformer-hybrid-dir", required=True)
    ap.add_argument("--bilstm-erp-dir", required=True)
    ap.add_argument("--bilstm-tfr-dir", required=True)
    ap.add_argument("--bilstm-hybrid-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="test")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--label-a", default="Transformer")
    ap.add_argument("--label-b", default="BiLSTM")
    args = ap.parse_args()

    _set_plot_defaults()
    os.makedirs(args.out, exist_ok=True)
    prange = _load_param_range(args.data_out)

    feats = ["erp", "tfr", "hybrid"]
    a_dirs = [args.transformer_erp_dir, args.transformer_tfr_dir, args.transformer_hybrid_dir]
    b_dirs = [args.bilstm_erp_dir, args.bilstm_tfr_dir, args.bilstm_hybrid_dir]

    a_eval = [_load_eval_dir(d, args.split) for d in a_dirs]
    b_eval = [_load_eval_dir(d, args.split) for d in b_dirs]
    for i, feat in enumerate(feats):
        _check_alignment(a_eval[i], b_eval[i], f"{args.label_a}-{feat}", f"{args.label_b}-{feat}")
    for i in range(1, 3):
        _check_alignment(a_eval[0], a_eval[i], f"{args.label_a}-erp", f"{args.label_a}-{feats[i]}")
        _check_alignment(b_eval[0], b_eval[i], f"{args.label_b}-erp", f"{args.label_b}-{feats[i]}")

    param_names = list(a_eval[0]["param_names"].tolist())
    P = len(param_names)
    M_p = np.zeros((P, 3), dtype=np.float64)
    M_r = np.zeros((P, 3), dtype=np.float64)
    M_p_lo = np.zeros((P, 3), dtype=np.float64)
    M_p_hi = np.zeros((P, 3), dtype=np.float64)
    M_r_lo = np.zeros((P, 3), dtype=np.float64)
    M_r_hi = np.zeros((P, 3), dtype=np.float64)

    rows = []
    for j, feat in enumerate(feats):
        theta_true = a_eval[j]["theta_true"]
        pa, ra = _point_metrics(theta_true, a_eval[j]["theta_mean"], prange)
        pb, rb = _point_metrics(theta_true, b_eval[j]["theta_mean"], prange)
        M_p[:, j] = pa - pb
        M_r[:, j] = ra - rb
        plo, phi, rlo, rhi = _bootstrap_deltas(theta_true, a_eval[j]["theta_mean"], b_eval[j]["theta_mean"], prange, args.n_bootstrap, args.seed + 41 * (j + 1))
        M_p_lo[:, j], M_p_hi[:, j] = plo, phi
        M_r_lo[:, j], M_r_hi[:, j] = rlo, rhi

    delta_csv = os.path.join(args.out, f"delta_metrics_{args.tag}.csv")
    with open(delta_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "param",
            "d_pearson_erp", "d_pearson_erp_lo", "d_pearson_erp_hi",
            "d_pearson_tfr", "d_pearson_tfr_lo", "d_pearson_tfr_hi",
            "d_pearson_hybrid", "d_pearson_hybrid_lo", "d_pearson_hybrid_hi",
            "d_rmse_norm_erp", "d_rmse_norm_erp_lo", "d_rmse_norm_erp_hi",
            "d_rmse_norm_tfr", "d_rmse_norm_tfr_lo", "d_rmse_norm_tfr_hi",
            "d_rmse_norm_hybrid", "d_rmse_norm_hybrid_lo", "d_rmse_norm_hybrid_hi",
        ])
        for i, nm in enumerate(param_names):
            w.writerow([
                nm,
                M_p[i, 0], M_p_lo[i, 0], M_p_hi[i, 0],
                M_p[i, 1], M_p_lo[i, 1], M_p_hi[i, 1],
                M_p[i, 2], M_p_lo[i, 2], M_p_hi[i, 2],
                M_r[i, 0], M_r_lo[i, 0], M_r_hi[i, 0],
                M_r[i, 1], M_r_lo[i, 1], M_r_hi[i, 1],
                M_r[i, 2], M_r_lo[i, 2], M_r_hi[i, 2],
            ])

    feat_labels = ["ERP", "TFR", "Hybrid"]
    annot_fs = 8.0

    def _heatmap(M: np.ndarray, title: str, fname: str, cbar_label: str) -> None:
        fig, ax = plt.subplots(figsize=(9.3, max(5.6, 0.62 * len(param_names) + 1.4)))
        vmax = float(np.max(np.abs(M)))
        vmax = max(vmax, 0.05)
        im = ax.imshow(M, aspect="auto", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(3))
        ax.set_xticklabels(feat_labels)
        ax.set_yticks(range(len(param_names)))
        ax.set_yticklabels(param_names)
        ax.set_title(title, pad=10)
        for i in range(len(param_names)):
            for j in range(3):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=annot_fs)
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label(cbar_label)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, fname), dpi=args.dpi)
        fig.savefig(os.path.join(args.out, fname.replace(".png", ".pdf")))
        plt.close(fig)

    _heatmap(M_p, f"ΔPearson ({args.label_a} − {args.label_b}) [{args.tag}]", f"delta_pearson_heatmap_{args.tag}.png", "ΔPearson")
    _heatmap(M_r, f"ΔnRMSE ({args.label_a} − {args.label_b}) [{args.tag}]", f"delta_rmse_norm_heatmap_{args.tag}.png", "ΔnRMSE")

    fig, ax = plt.subplots(figsize=(9.5, max(5.5, 0.52 * len(param_names) + 1.5)))
    y = np.arange(len(param_names))
    x = M_p[:, 2]
    xerr = np.vstack([x - M_p_lo[:, 2], M_p_hi[:, 2] - x])
    ax.barh(y, x, xerr=xerr, capsize=3)
    ax.axvline(0.0, linestyle="--", linewidth=1.2, color="black")
    ax.set_yticks(y)
    ax.set_yticklabels(param_names)
    ax.invert_yaxis()
    ax.set_xlabel(f"ΔPearson ({args.label_a} − {args.label_b})")
    ax.set_title(f"Hybrid ΔPearson with 95% bootstrap intervals [{args.tag}]", pad=10)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    out_bar = os.path.join(args.out, f"delta_pearson_hybrid_bar_{args.tag}.png")
    fig.savefig(out_bar, dpi=args.dpi)
    fig.savefig(out_bar.replace(".png", ".pdf"))
    plt.close(fig)

    summary = {
        "tag": args.tag,
        "n_bootstrap": int(args.n_bootstrap),
        "label_a": args.label_a,
        "label_b": args.label_b,
        "delta_csv": os.path.basename(delta_csv),
    }
    with open(os.path.join(args.out, f"plot_arch_compare_summary_{args.tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("[plot_arch_compare] wrote:", args.out)
    print("[plot_arch_compare] csv:", delta_csv)


if __name__ == "__main__":
    main()
