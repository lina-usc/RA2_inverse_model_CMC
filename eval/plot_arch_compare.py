
import argparse
import csv
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="test")
    args = ap.parse_args()

    _ensure_dir(args.out)

    rows = []
    with open(args.delta_csv, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    params = [r["param"] for r in rows]
    feats = ["erp", "tfr", "hybrid"]

    def mat(prefix: str) -> np.ndarray:
        M = np.zeros((len(params), len(feats)), dtype=np.float32)
        for i, r in enumerate(rows):
            for j, feat in enumerate(feats):
                M[i, j] = float(r[f"{prefix}_{feat}"])
        return M

    M_p = mat("d_pearson")
    M_r = mat("d_rmse_norm")

    def heatmap(M, title, fname):
        fig, ax = plt.subplots(figsize=(8.0, max(4.0, 0.35*len(params))))
        im = ax.imshow(M, aspect="auto")
        ax.set_xticks(range(len(feats)))
        ax.set_xticklabels(feats)
        ax.set_yticks(range(len(params)))
        ax.set_yticklabels(params)
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        fig.tight_layout()
        out = os.path.join(args.out, fname)
        fig.savefig(out, dpi=220)
        plt.close(fig)
        print("[plot_arch_compare] wrote:", out)

    heatmap(M_p, f"ΔPearson (Transformer − BiLSTM) [{args.tag}]", f"delta_pearson_heatmap_{args.tag}.png")
    heatmap(M_r, f"ΔRMSE_norm (Transformer − BiLSTM) [{args.tag}]", f"delta_rmse_norm_heatmap_{args.tag}.png")

    # Simple hybrid-only bar
    fig, ax = plt.subplots(figsize=(10.0, 0.35*len(params) + 2.0))
    ax.barh(params, M_p[:, 2])
    ax.axvline(0.0)
    ax.set_title(f"ΔPearson Hybrid (Transformer − BiLSTM) [{args.tag}]")
    ax.set_xlabel("delta pearson")
    fig.tight_layout()
    out = os.path.join(args.out, f"delta_pearson_hybrid_bar_{args.tag}.png")
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print("[plot_arch_compare] wrote:", out)


if __name__ == "__main__":
    main()
