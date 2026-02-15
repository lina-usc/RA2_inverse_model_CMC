
import argparse
import os
import csv
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-levels", type=int, default=19, help="Number of nominal CI levels in (0,1)")
    args = ap.parse_args()

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

    # Coverage curve levels
    levels = np.linspace(0.05, 0.95, args.n_levels)
    rows = []

    # Compute coverage per param and level
    cov = np.zeros((P, len(levels)), dtype=np.float64)
    for li, a in enumerate(levels):
        lo_q = (1.0 - a) / 2.0
        hi_q = 1.0 - lo_q
        lo = np.quantile(theta_samps, lo_q, axis=1)  # (N,P)
        hi = np.quantile(theta_samps, hi_q, axis=1)
        inside = (theta_true >= lo) & (theta_true <= hi)
        cov[:, li] = inside.mean(axis=0)
        for j in range(P):
            rows.append([names[j], float(a), float(cov[j, li])])

    out_csv = os.path.join(args.out, "coverage_curve.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["param", "nominal", "empirical"])
        w.writerows(rows)

    # Plot grid (3x3 default for P=9)
    ncols = 3
    nrows = int(np.ceil(P / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 3.8*nrows))
    axes = np.asarray(axes).reshape(-1)

    for j in range(P):
        ax = axes[j]
        ax.plot(levels, cov[j], marker="o", markersize=3)
        ax.plot([0, 1], [0, 1])
        ax.set_title(names[j])
        ax.set_xlabel("nominal CI level")
        ax.set_ylabel("empirical coverage")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25)

    for k in range(P, len(axes)):
        axes[k].axis("off")

    fig.tight_layout()
    out_png = os.path.join(args.out, "coverage_curves.png")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    # Summary JSON
    cal_err = float(np.mean(np.abs(cov - levels[None, :])))
    summary = {
        "eval_npz": args.eval_npz,
        "n_eval": int(N),
        "n_post_samples": int(S),
        "mean_abs_calibration_error": cal_err,
        "wrote": {
            "csv": os.path.basename(out_csv),
            "png": os.path.basename(out_png),
        },
    }
    with open(os.path.join(args.out, "plot_reliability_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("[plot_reliability] wrote:", out_png)
    print("[plot_reliability] wrote:", out_csv)


if __name__ == "__main__":
    main()
