from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _decode_param_names(x) -> list[str]:
    x = np.asarray(x)
    out = []
    for v in x:
        if isinstance(v, (bytes, np.bytes_)):
            out.append(v.decode("utf-8"))
        else:
            out.append(str(v))
    return out


def _pick(npz, keys: list[str], required: bool = True):
    for k in keys:
        if k in npz.files:
            return npz[k]
    if required:
        raise KeyError(f"None of keys found: {keys}. Available: {npz.files}")
    return None


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot empirical coverage curves with bootstrap confidence bands.")
    ap.add_argument("--eval-npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-levels", type=int, default=19, help="Number of nominal CI levels in (0,1)")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    _set_plot_defaults()
    _ensure_dir(args.out)
    d = np.load(args.eval_npz)

    param_names = _pick(d, ["param_names", "param_names_bytes"], required=False)
    theta_true = _pick(d, ["theta_true", "theta", "y_true_theta", "theta_test"])
    theta_samps = _pick(
        d,
        ["theta_post_samples", "theta_samples", "post_samples_theta", "samples_theta"],
        required=True,
    )

    if theta_samps.ndim != 3:
        raise ValueError(f"Expected theta_samples shape (N,S,P), got {theta_samps.shape}")

    N, S, P = theta_samps.shape
    if theta_true.shape != (N, P):
        raise ValueError(f"theta_true shape {theta_true.shape} not compatible with samples {theta_samps.shape}")

    names = _decode_param_names(param_names) if param_names is not None else [f"param_{i}" for i in range(P)]

    levels = np.linspace(0.05, 0.95, args.n_levels)
    cov = np.zeros((P, len(levels)), dtype=np.float64)
    cov_lo = np.zeros_like(cov)
    cov_hi = np.zeros_like(cov)
    rows = []

    rng = np.random.default_rng(args.seed)
    boot_idx = rng.integers(0, N, size=(args.n_bootstrap, N), dtype=np.int64)

    for li, a in enumerate(levels):
        lo_q = (1.0 - a) / 2.0
        hi_q = 1.0 - lo_q
        lo = np.quantile(theta_samps, lo_q, axis=1)
        hi = np.quantile(theta_samps, hi_q, axis=1)
        inside = ((theta_true >= lo) & (theta_true <= hi)).astype(np.float32)  # (N,P)
        cov[:, li] = inside.mean(axis=0)
        boot_cov = inside[boot_idx].mean(axis=1)  # (B,P)
        cov_lo[:, li], cov_hi[:, li] = np.quantile(boot_cov, [0.025, 0.975], axis=0)
        for j in range(P):
            rows.append([names[j], float(a), float(cov[j, li]), float(cov_lo[j, li]), float(cov_hi[j, li])])

    out_csv = os.path.join(args.out, "coverage_curve.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["param", "nominal", "empirical", "empirical_lo", "empirical_hi"])
        w.writerows(rows)

    coverage_summary_csv = os.path.join(args.out, "coverage_summary.csv")
    with open(coverage_summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["param", "cov50", "cov50_lo", "cov50_hi", "cov90", "cov90_lo", "cov90_hi"])
        i50 = int(np.argmin(np.abs(levels - 0.50)))
        i90 = int(np.argmin(np.abs(levels - 0.90)))
        for j in range(P):
            w.writerow([
                names[j],
                cov[j, i50], cov_lo[j, i50], cov_hi[j, i50],
                cov[j, i90], cov_lo[j, i90], cov_hi[j, i90],
            ])

    ncols = 3
    nrows = int(np.ceil(P / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.0 * nrows))
    axes = np.asarray(axes).reshape(-1)

    for j in range(P):
        ax = axes[j]
        ax.fill_between(levels, cov_lo[j], cov_hi[j], alpha=0.22)
        ax.plot(levels, cov[j], marker="o", markersize=3.5, linewidth=1.8)
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="black")
        ax.set_title(names[j])
        ax.set_xlabel("Nominal credible level")
        ax.set_ylabel("Empirical coverage")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25)

    for k in range(P, len(axes)):
        axes[k].axis("off")

    fig.tight_layout()
    out_png = os.path.join(args.out, "coverage_curves.png")
    fig.savefig(out_png, dpi=args.dpi)
    fig.savefig(out_png.replace(".png", ".pdf"))
    plt.close(fig)

    cal_err = float(np.mean(np.abs(cov - levels[None, :])))
    summary = {
        "eval_npz": args.eval_npz,
        "n_eval": int(N),
        "n_post_samples": int(S),
        "n_bootstrap": int(args.n_bootstrap),
        "mean_abs_calibration_error": cal_err,
        "wrote": {
            "csv": os.path.basename(out_csv),
            "summary_csv": os.path.basename(coverage_summary_csv),
            "png": os.path.basename(out_png),
        },
    }
    with open(os.path.join(args.out, "plot_reliability_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("[plot_reliability] wrote:", out_png)
    print("[plot_reliability] wrote:", out_csv)
    print("[plot_reliability] wrote:", coverage_summary_csv)


if __name__ == "__main__":
    main()
