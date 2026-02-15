
import argparse
import os
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-npz", required=True, help="Path to eval_test_outputs.npz")
    ap.add_argument("--out", required=True, help="Output directory (plots written here)")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    _ensure_dir(args.out)
    d = np.load(args.eval_npz)

    # Robust key picking (handles small naming differences)
    param_names = _pick(d, ["param_names", "params", "param_names_bytes"], required=False)
    if param_names is None:
        # fallback: infer P from arrays, create generic names
        theta_true = _pick(d, ["theta_true", "theta", "y_true_theta", "theta_test"])
        P = theta_true.shape[1]
        names = [f"param_{i}" for i in range(P)]
    else:
        names = _decode_param_names(param_names)

    theta_true = _pick(d, ["theta_true", "theta", "y_true_theta", "theta_test"])
    theta_mean = _pick(d, ["theta_post_mean", "theta_mean", "post_mean_theta", "theta_mu", "theta_pred_mean"])

    if theta_true.shape != theta_mean.shape:
        raise ValueError(f"Shape mismatch: theta_true={theta_true.shape} theta_mean={theta_mean.shape}")

    N, P = theta_true.shape

    # Scatter grid
    ncols = 3
    nrows = int(np.ceil(P / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 4.0*nrows))
    axes = np.asarray(axes).reshape(-1)

    for j in range(P):
        ax = axes[j]
        x = theta_true[:, j]
        y = theta_mean[:, j]
        ax.scatter(x, y, s=6, alpha=0.35)
        mn = np.min([x.min(), y.min()])
        mx = np.max([x.max(), y.max()])
        ax.plot([mn, mx], [mn, mx])
        ax.set_title(names[j])
        ax.set_xlabel("true")
        ax.set_ylabel("posterior mean")
        ax.grid(True, alpha=0.25)

    # hide unused axes
    for k in range(P, len(axes)):
        axes[k].axis("off")

    if args.title:
        fig.suptitle(args.title)

    fig.tight_layout()
    out_scatter = os.path.join(args.out, "scatter_true_vs_postmean.png")
    fig.savefig(out_scatter, dpi=200)
    plt.close(fig)

    # Residual hist grid
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 3.8*nrows))
    axes = np.asarray(axes).reshape(-1)
    for j in range(P):
        ax = axes[j]
        r = theta_mean[:, j] - theta_true[:, j]
        ax.hist(r, bins=60)
        ax.set_title(f"{names[j]} residuals")
        ax.set_xlabel("mean - true")
        ax.grid(True, alpha=0.25)

    for k in range(P, len(axes)):
        axes[k].axis("off")
    fig.tight_layout()
    out_resid = os.path.join(args.out, "residual_hist.png")
    fig.savefig(out_resid, dpi=200)
    plt.close(fig)

    # Small JSON summary
    summary = {
        "eval_npz": args.eval_npz,
        "n_eval": int(N),
        "n_params": int(P),
        "wrote": {
            "scatter": os.path.basename(out_scatter),
            "residual_hist": os.path.basename(out_resid),
        },
        "available_keys": d.files,
    }
    with open(os.path.join(args.out, "plot_results_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("[plot_results] wrote:", out_scatter)
    print("[plot_results] wrote:", out_resid)


if __name__ == "__main__":
    main()
