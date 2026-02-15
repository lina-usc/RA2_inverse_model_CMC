
import argparse
import csv
import json
import os
from typing import Dict, List

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_metrics_csv(path: str) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            nm = row["param"]
            out[nm] = {k: float(row[k]) if k != "param" else row[k] for k in row.keys() if k != "param"}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-out", type=str, default="data_out")
    ap.add_argument("--erp-dir", type=str, required=True)
    ap.add_argument("--tfr-dir", type=str, required=True)
    ap.add_argument("--hybrid-dir", type=str, required=True)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--out-dir", type=str, default="plots/compare_features")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    param_meta = np.load(os.path.join(args.data_out, "param_meta.npz"))
    param_names = [x.decode("utf-8") for x in param_meta["param_names"]]
    low = param_meta["prior_low"].astype(np.float32)
    high = param_meta["prior_high"].astype(np.float32)
    prange = (high - low).astype(np.float32)
    prange = np.where(prange < 1e-9, 1.0, prange)

    m_erp = read_metrics_csv(os.path.join(args.erp_dir, f"metrics_{args.split}.csv"))
    m_tfr = read_metrics_csv(os.path.join(args.tfr_dir, f"metrics_{args.split}.csv"))
    m_hyb = read_metrics_csv(os.path.join(args.hybrid_dir, f"metrics_{args.split}.csv"))

    feats = ["ERP", "TFR", "Hybrid"]
    mats = [m_erp, m_tfr, m_hyb]

    P = len(param_names)
    F = 3

    pear = np.zeros((P, F), dtype=np.float32)
    rmse = np.zeros((P, F), dtype=np.float32)
    cov90 = np.zeros((P, F), dtype=np.float32)

    for i, nm in enumerate(param_names):
        for j in range(F):
            d = mats[j][nm]
            pear[i, j] = float(d["pearson_mean"])
            rmse[i, j] = float(d["rmse_mean"])
            cov90[i, j] = float(d["cov90"])

    rmse_norm = rmse / prange[:, None]

    # ---- save a table for the paper ----
    table_csv = os.path.join(args.out_dir, f"recoverability_table_{args.split}.csv")
    with open(table_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["param",
                    "pearson_erp","pearson_tfr","pearson_hybrid",
                    "rmse_erp","rmse_tfr","rmse_hybrid",
                    "rmse_norm_erp","rmse_norm_tfr","rmse_norm_hybrid",
                    "cov90_erp","cov90_tfr","cov90_hybrid",
                    "best_feature_by_pearson"])
        for i, nm in enumerate(param_names):
            best = feats[int(np.nanargmax(pear[i]))]
            w.writerow([
                nm,
                pear[i,0], pear[i,1], pear[i,2],
                rmse[i,0], rmse[i,1], rmse[i,2],
                rmse_norm[i,0], rmse_norm[i,1], rmse_norm[i,2],
                cov90[i,0], cov90[i,1], cov90[i,2],
                best
            ])

    # ---- Heatmap: Pearson correlation ----
    fig = plt.figure(figsize=(8, 0.55 * P + 2.0))
    ax = fig.add_subplot(111)
    im = ax.imshow(pear, aspect="auto", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(F))
    ax.set_xticklabels(feats)
    ax.set_yticks(range(P))
    ax.set_yticklabels(param_names)
    ax.set_title(f"Recoverability (Pearson r) — split={args.split}")
    for i in range(P):
        for j in range(F):
            ax.text(j, i, f"{pear[i,j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, f"recoverability_pearson_heatmap_{args.split}.png"), dpi=220)
    fig.savefig(os.path.join(args.out_dir, f"recoverability_pearson_heatmap_{args.split}.pdf"))
    plt.close(fig)

    # ---- Heatmap: normalized RMSE ----
    fig = plt.figure(figsize=(8, 0.55 * P + 2.0))
    ax = fig.add_subplot(111)
    im = ax.imshow(rmse_norm, aspect="auto")
    ax.set_xticks(range(F))
    ax.set_xticklabels(feats)
    ax.set_yticks(range(P))
    ax.set_yticklabels(param_names)
    ax.set_title(f"Recoverability (RMSE / prior range) — split={args.split}")
    for i in range(P):
        for j in range(F):
            ax.text(j, i, f"{rmse_norm[i,j]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, f"recoverability_rmse_norm_heatmap_{args.split}.png"), dpi=220)
    fig.savefig(os.path.join(args.out_dir, f"recoverability_rmse_norm_heatmap_{args.split}.pdf"))
    plt.close(fig)

    # ---- Hybrid gain over best single (by Pearson) ----
    best_single = np.maximum(pear[:,0], pear[:,1])
    gain = pear[:,2] - best_single

    fig = plt.figure(figsize=(8, 0.45 * P + 2.0))
    ax = fig.add_subplot(111)
    y = np.arange(P)
    ax.barh(y, gain)
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(param_names)
    ax.set_xlabel("Hybrid Pearson gain over best single-feature (ERP or TFR)")
    ax.set_title(f"Does Hybrid add information? — split={args.split}")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, f"hybrid_gain_over_best_single_{args.split}.png"), dpi=220)
    fig.savefig(os.path.join(args.out_dir, f"hybrid_gain_over_best_single_{args.split}.pdf"))
    plt.close(fig)

    summary = {
        "split": args.split,
        "features": feats,
        "table_csv": os.path.basename(table_csv),
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
