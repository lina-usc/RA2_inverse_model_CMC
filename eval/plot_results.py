from __future__ import annotations

import argparse
import os
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _decode_param_names(x) -> list[str]:
    x = np.asarray(x)
    if x.dtype.kind in ("S", "O"):
        out = []
        for v in x:
            if isinstance(v, (bytes, np.bytes_)):
                out.append(v.decode("utf-8"))
            else:
                out.append(str(v))
        return out
    return [str(v) for v in x.tolist()]


def _pick(npz, keys: list[str], required: bool = True):
    for k in keys:
        if k in npz.files:
            return npz[k]
    if required:
        raise KeyError(f"None of keys found: {keys}. Available: {npz.files}")
    return None


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum()) + 1e-12
    return float((x * y).sum() / denom)


def _set_plot_defaults() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-npz", required=True, help="Path to eval_test_outputs.npz")
    ap.add_argument("--out", required=True, help="Output directory (plots written here)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    _set_plot_defaults()
    _ensure_dir(args.out)
    d = np.load(args.eval_npz)

    param_names = _pick(d, ["param_names", "params", "param_names_bytes"], required=False)
    theta_true = _pick(d, ["theta_true", "theta", "y_true_theta", "theta_test"])
    theta_mean = _pick(d, ["theta_post_mean", "theta_mean", "post_mean_theta", "theta_mu", "theta_pred_mean"])

    if param_names is None:
        P = theta_true.shape[1]
        names = [f"param_{i}" for i in range(P)]
    else:
        names = _decode_param_names(param_names)

    if theta_true.shape != theta_mean.shape:
        raise ValueError(f"Shape mismatch: theta_true={theta_true.shape} theta_mean={theta_mean.shape}")

    N, P = theta_true.shape
    ncols = 3
    nrows = int(np.ceil(P / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.3 * nrows))
    axes = np.asarray(axes).reshape(-1)
    for j in range(P):
        ax = axes[j]
        x = theta_true[:, j]
        y = theta_mean[:, j]
        mn = float(np.min([x.min(), y.min()]))
        mx = float(np.max([x.max(), y.max()]))
        pad = 0.05 * (mx - mn + 1e-9)
        r = _safe_pearson(x, y)
        ax.scatter(x, y, s=10, alpha=0.35, rasterized=True)
        ax.plot([mn - pad, mx + pad], [mn - pad, mx + pad], linestyle="--", linewidth=1.2, color="black")
        ax.set_xlim(mn - pad, mx + pad)
        ax.set_ylim(mn - pad, mx + pad)
        ax.set_title(f"{names[j]}\nPearson={r:.2f}")
        ax.set_xlabel("True")
        ax.set_ylabel("Posterior mean")
        ax.grid(True, alpha=0.25)
    for k in range(P, len(axes)):
        axes[k].axis("off")
    if args.title:
        fig.suptitle(args.title)
    fig.tight_layout()
    out_scatter = os.path.join(args.out, "scatter_true_vs_postmean.png")
    fig.savefig(out_scatter, dpi=args.dpi)
    fig.savefig(out_scatter.replace(".png", ".pdf"))
    plt.close(fig)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.0 * nrows))
    axes = np.asarray(axes).reshape(-1)
    for j in range(P):
        ax = axes[j]
        r = theta_mean[:, j] - theta_true[:, j]
        ax.hist(r, bins=50)
        ax.axvline(0.0, linestyle="--", linewidth=1.0, color="black")
        ax.set_title(f"{names[j]} residuals")
        ax.set_xlabel("Posterior mean − true")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.25)
    for k in range(P, len(axes)):
        axes[k].axis("off")
    fig.tight_layout()
    out_resid = os.path.join(args.out, "residual_hist.png")
    fig.savefig(out_resid, dpi=args.dpi)
    fig.savefig(out_resid.replace(".png", ".pdf"))
    plt.close(fig)

    summary = {
        "eval_npz": args.eval_npz,
        "n_eval": int(N),
        "n_params": int(P),
        "wrote": {
            "scatter": os.path.basename(out_scatter),
            "residual_hist": os.path.basename(out_resid),
        },
    }
    with open(os.path.join(args.out, "plot_results_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("[plot_results] wrote:", out_scatter)
    print("[plot_results] wrote:", out_resid)


if __name__ == "__main__":
    main()
