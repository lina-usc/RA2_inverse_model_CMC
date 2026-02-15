
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
    ap.add_argument("--bins", type=int, default=20)
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

    # ranks in [0, S]
    # rank = number of posterior samples < true value
    ranks = np.sum(theta_samps < theta_true[:, None, :], axis=1).astype(np.int32)  # (N,P)

    # Save ranks for paper provenance
    out_npz = os.path.join(args.out, "sbc_ranks.npz")
    np.savez(out_npz, ranks=ranks, n_post_samples=np.array(S), param_names=np.array(names, dtype="S"))

    # Plot histograms
    ncols = 3
    nrows = int(np.ceil(P / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 3.8*nrows))
    axes = np.asarray(axes).reshape(-1)

    # bins over 0..S
    bin_edges = np.linspace(0, S, args.bins + 1)

    # simple chi-square vs uniform (diagnostic only)
    chi2 = []
    for j in range(P):
        counts, _ = np.histogram(ranks[:, j], bins=bin_edges)
        expected = counts.sum() / len(counts)
        stat = float(np.sum((counts - expected) ** 2 / (expected + 1e-9)))
        chi2.append(stat)

        ax = axes[j]
        ax.hist(ranks[:, j], bins=bin_edges)
        ax.set_title(f"{names[j]} (chi2={stat:.1f})")
        ax.set_xlabel("rank")
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.25)

    for k in range(P, len(axes)):
        axes[k].axis("off")

    fig.tight_layout()
    out_png = os.path.join(args.out, "sbc_rank_hist.png")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    summary = {
        "eval_npz": args.eval_npz,
        "n_eval": int(N),
        "n_post_samples": int(S),
        "bins": int(args.bins),
        "chi2_by_param": {names[j]: chi2[j] for j in range(P)},
        "wrote": {
            "png": os.path.basename(out_png),
            "ranks_npz": os.path.basename(out_npz),
        },
    }
    with open(os.path.join(args.out, "plot_sbc_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("[plot_sbc] wrote:", out_png)
    print("[plot_sbc] wrote:", out_npz)


if __name__ == "__main__":
    main()
