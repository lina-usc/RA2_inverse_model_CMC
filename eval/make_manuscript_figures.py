"""Generate *all* manuscript figures from saved evaluation outputs.

This is intended to replace a pile of one-off plotting scripts and to enforce
readable, paper-friendly font sizes.

Expected directory layout (matches reproduce_morlet_all.sh):
  PLOTS_OUT/
    eval_erp_ens/
      metrics_test.csv
      eval_test_outputs.npz
    eval_tfr_ens/
    eval_hybrid_ens/
    eval_erp_bilstm_ens/
    eval_tfr_bilstm_ens/
    eval_hybrid_bilstm_ens/
    eval_hybrid_diag_ens/
    eval_hybrid_noparamtoken_ens/

Outputs (written to FIG_OUT):
  recoverability_pearson_heatmap_<split>.png
  recoverability_rmse_norm_heatmap_<split>.png
  hybrid_gain_over_best_single_<split>.png
  delta_pearson_heatmap_<split>.png
  delta_rmse_norm_heatmap_<split>.png
  delta_pearson_hybrid_bar_<split>.png
  scatter_true_vs_postmean.png
  residual_hist.png
  coverage_curves.png
  sbc_rank_hist.png
  nllz_hist_<split>.png
  info_gain_bits_<split>.csv
  info_gain_bits_<split>.png
  compare_feature_sets_summary_<split>.csv
  delta_metrics_<split>.csv
  ablation_summary_<split>.csv (if ablation eval dirs exist)
  ablation_mean_pearson_<split>.png (if ablation eval dirs exist)

Optional (if --run-ppc and --run-sweeps):
  ppc_*.png
  sweep_*_mean_metrics.png

Run example:
  python -m eval.make_manuscript_figures \
    --data-out data_out_morlet \
    --plots-out plots_morlet \
    --fig-out figures_morlet \
    --split test
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _decode(x) -> str:
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8")
    return str(x)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _read_metrics(dir_path: str, split: str) -> pd.DataFrame:
    csv_path = os.path.join(dir_path, f"metrics_{split}.csv")
    df = pd.read_csv(csv_path)
    df = df[df["param"] != "MEAN"].copy()
    return df.set_index("param")


def _read_eval_npz(dir_path: str, split: str) -> Dict[str, np.ndarray]:
    npz_path = os.path.join(dir_path, f"eval_{split}_outputs.npz")
    d = np.load(npz_path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _set_paper_rcparams() -> None:
    # Minimum ~8pt tick labels; titles/labels ~10pt.
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 10,
        }
    )


def _heatmap(
    mat: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    out_png: str,
    *,
    vmin=None,
    vmax=None,
    cmap=None,
    fmt: str = ".2f",
) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    im = ax.imshow(mat, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=0)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)

    # annotate
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            txt = "nan" if not np.isfinite(val) else format(val, fmt)
            ax.text(j, i, txt, ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close()


def _barplot(values: np.ndarray, labels: List[str], title: str, ylabel: str, out_png: str) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close()


def make_recoverability_figures(plots_out: str, fig_out: str, split: str) -> None:
    # Transformer ensembles
    dirs = {
        "ERP": os.path.join(plots_out, "eval_erp_ens"),
        "TFR": os.path.join(plots_out, "eval_tfr_ens"),
        "Hybrid": os.path.join(plots_out, "eval_hybrid_ens"),
    }

    dfs = {k: _read_metrics(v, split) for k, v in dirs.items()}
    param_names = list(dfs["Hybrid"].index)

    pearson = np.stack([dfs[k]["pearson_mean"].values for k in ["ERP", "TFR", "Hybrid"]], axis=1)
    rmse_norm = np.stack([dfs[k]["rmse_norm_mean"].values for k in ["ERP", "TFR", "Hybrid"]], axis=1)

    _heatmap(
        pearson,
        param_names,
        ["ERP", "TFR", "Hybrid"],
        title=f"Recoverability (Pearson r) — {split}",
        out_png=os.path.join(fig_out, f"recoverability_pearson_heatmap_{split}.png"),
        vmin=-1.0,
        vmax=1.0,
        cmap="coolwarm",
    )

    _heatmap(
        rmse_norm,
        param_names,
        ["ERP", "TFR", "Hybrid"],
        title=f"Recoverability (Normalized RMSE) — {split}",
        out_png=os.path.join(fig_out, f"recoverability_rmse_norm_heatmap_{split}.png"),
        vmin=0.0,
        vmax=float(np.nanpercentile(rmse_norm, 95)),
        cmap="viridis",
        fmt=".3f",
    )

    # Hybrid gain over best single modality (Pearson)
    best_single = np.maximum(pearson[:, 0], pearson[:, 1])
    gain = pearson[:, 2] - best_single
    _barplot(
        gain,
        param_names,
        title=f"Hybrid gain over best single modality (Pearson r) — {split}",
        ylabel="Δ Pearson r",
        out_png=os.path.join(fig_out, f"hybrid_gain_over_best_single_{split}.png"),
    )

    # Write summary CSV (useful for paper tables)
    out_rows = []
    for p in param_names:
        row = {"param": p}
        for feat in ["ERP", "TFR", "Hybrid"]:
            row[f"pearson_{feat.lower()}"] = float(dfs[feat].loc[p, "pearson_mean"])
            row[f"rmse_{feat.lower()}"] = float(dfs[feat].loc[p, "rmse_mean"])
            row[f"rmse_norm_{feat.lower()}"] = float(dfs[feat].loc[p, "rmse_norm_mean"])
            row[f"cov90_{feat.lower()}"] = float(dfs[feat].loc[p, "cov90"])
        row["hybrid_gain_over_best_single"] = float(
            dfs["Hybrid"].loc[p, "pearson_mean"]
            - max(dfs["ERP"].loc[p, "pearson_mean"], dfs["TFR"].loc[p, "pearson_mean"])
        )
        out_rows.append(row)

    pd.DataFrame(out_rows).to_csv(os.path.join(fig_out, f"compare_feature_sets_summary_{split}.csv"), index=False)


def make_architecture_compare_figures(plots_out: str, fig_out: str, split: str) -> None:
    # Transformer vs BiLSTM deltas
    pairs = {
        "ERP": (os.path.join(plots_out, "eval_erp_ens"), os.path.join(plots_out, "eval_erp_bilstm_ens")),
        "TFR": (os.path.join(plots_out, "eval_tfr_ens"), os.path.join(plots_out, "eval_tfr_bilstm_ens")),
        "Hybrid": (os.path.join(plots_out, "eval_hybrid_ens"), os.path.join(plots_out, "eval_hybrid_bilstm_ens")),
    }

    # param order from hybrid transformer
    tr_hybrid = _read_metrics(pairs["Hybrid"][0], split)
    param_names = list(tr_hybrid.index)

    d_pearson_cols = []
    d_rmse_cols = []

    for feat in ["ERP", "TFR", "Hybrid"]:
        tr = _read_metrics(pairs[feat][0], split).loc[param_names]
        bl = _read_metrics(pairs[feat][1], split).loc[param_names]

        d_pearson_cols.append((tr["pearson_mean"] - bl["pearson_mean"]).values)
        d_rmse_cols.append((tr["rmse_norm_mean"] - bl["rmse_norm_mean"]).values)

    d_pearson = np.stack(d_pearson_cols, axis=1)
    d_rmse = np.stack(d_rmse_cols, axis=1)

    _heatmap(
        d_pearson,
        param_names,
        ["ERP", "TFR", "Hybrid"],
        title=f"Transformer − BiLSTM (Pearson r) — {split}",
        out_png=os.path.join(fig_out, f"delta_pearson_heatmap_{split}.png"),
        vmin=-1.0,
        vmax=1.0,
        cmap="coolwarm",
    )

    # For RMSE_norm, negative means Transformer is better.
    lim = float(np.nanpercentile(np.abs(d_rmse), 95))
    _heatmap(
        d_rmse,
        param_names,
        ["ERP", "TFR", "Hybrid"],
        title=f"Transformer − BiLSTM (Normalized RMSE) — {split}",
        out_png=os.path.join(fig_out, f"delta_rmse_norm_heatmap_{split}.png"),
        vmin=-lim,
        vmax=lim,
        cmap="coolwarm",
        fmt=".3f",
    )

    # Hybrid-only bar for Pearson deltas
    _barplot(
        d_pearson[:, 2],
        param_names,
        title=f"Hybrid: Transformer − BiLSTM (Pearson r) — {split}",
        ylabel="Δ Pearson r",
        out_png=os.path.join(fig_out, f"delta_pearson_hybrid_bar_{split}.png"),
    )

    # Write delta CSV (useful for paper tables and sanity checks)
    delta_df = pd.DataFrame(
        {
            "param": param_names,
            "d_pearson_erp": d_pearson[:, 0],
            "d_pearson_tfr": d_pearson[:, 1],
            "d_pearson_hybrid": d_pearson[:, 2],
            "d_rmse_norm_erp": d_rmse[:, 0],
            "d_rmse_norm_tfr": d_rmse[:, 1],
            "d_rmse_norm_hybrid": d_rmse[:, 2],
        }
    )
    delta_df.to_csv(os.path.join(fig_out, f"delta_metrics_{split}.csv"), index=False)


def make_scatter_and_residual_figures(plots_out: str, fig_out: str, split: str) -> None:
    d = _read_eval_npz(os.path.join(plots_out, "eval_hybrid_ens"), split)

    param_names = [_decode(x) for x in d["param_names"]]
    theta_true = d["theta_true"].astype(float)
    theta_mean = d["theta_mean"].astype(float)

    P = theta_true.shape[1]

    # scatter
    ncols = 3
    nrows = int(np.ceil(P / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12.0, 10.5))
    axes = np.asarray(axes).reshape(-1)

    for i in range(P):
        ax = axes[i]
        ax.scatter(theta_true[:, i], theta_mean[:, i], s=6, alpha=0.5)
        lo = min(theta_true[:, i].min(), theta_mean[:, i].min())
        hi = max(theta_true[:, i].max(), theta_mean[:, i].max())
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
        ax.set_title(param_names[i])
        ax.set_xlabel("true")
        ax.set_ylabel("posterior mean")
        ax.grid(True, alpha=0.2)

    for j in range(P, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(fig_out, "scatter_true_vs_postmean.png"), dpi=250)
    plt.close()

    # residuals
    resid = theta_mean - theta_true

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12.0, 10.5))
    axes = np.asarray(axes).reshape(-1)

    for i in range(P):
        ax = axes[i]
        ax.hist(resid[:, i], bins=35, alpha=0.85)
        ax.axvline(0.0, linestyle="--", linewidth=1)
        ax.set_title(param_names[i])
        ax.set_xlabel("residual (mean − true)")
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.2)

    for j in range(P, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(fig_out, "residual_hist.png"), dpi=250)
    plt.close()


def make_coverage_curve(plots_out: str, fig_out: str, split: str) -> None:
    d = _read_eval_npz(os.path.join(plots_out, "eval_hybrid_ens"), split)

    theta_true = d["theta_true"].astype(float)
    theta_samps = d["theta_samples"].astype(float)

    P = theta_true.shape[1]

    levels = np.linspace(0.05, 0.95, 19)  # nominal mass
    cover = np.zeros((P, len(levels)), dtype=float)

    for li, mass in enumerate(levels):
        lo_q = (1.0 - mass) / 2.0
        hi_q = 1.0 - lo_q
        lo = np.quantile(theta_samps, lo_q, axis=1)
        hi = np.quantile(theta_samps, hi_q, axis=1)
        inside = (theta_true >= lo) & (theta_true <= hi)
        cover[:, li] = inside.mean(axis=0)

    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    for i in range(P):
        ax.plot(levels, cover[i], linewidth=1, alpha=0.5)

    ax.plot(levels, cover.mean(axis=0), linewidth=2, label="mean")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="ideal")

    ax.set_xlabel("nominal credible mass")
    ax.set_ylabel("empirical coverage")
    ax.set_title("Coverage curve (Hybrid ensemble)")
    ax.grid(True, alpha=0.25)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(fig_out, "coverage_curves.png"), dpi=250)
    plt.close()


def make_sbc_rank_hist(plots_out: str, fig_out: str, split: str, *, bins: int = 10) -> None:
    d = _read_eval_npz(os.path.join(plots_out, "eval_hybrid_ens"), split)

    param_names = [_decode(x) for x in d["param_names"]]
    theta_true = d["theta_true"].astype(float)
    theta_samps = d["theta_samples"].astype(float)

    N, S, P = theta_samps.shape

    # Rank of the true value among posterior samples (0..S)
    ranks = np.zeros((N, P), dtype=int)
    rng = np.random.default_rng(0)

    for i in range(P):
        # tie-breaking: add tiny noise
        eps = rng.uniform(-1e-9, 1e-9, size=(N, S))
        samp = theta_samps[:, :, i] + eps
        t = theta_true[:, i]
        ranks[:, i] = np.sum(samp < t[:, None], axis=1)

    # Save ranks
    np.savez(
        os.path.join(fig_out, f"sbc_ranks_{split}.npz"),
        ranks=ranks,
        param_names=np.array(param_names, dtype="S"),
    )

    # Plot (grid)
    ncols = 3
    nrows = int(np.ceil(P / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12.0, 10.0))
    axes = np.asarray(axes).reshape(-1)

    edges = np.linspace(0, S, bins + 1)

    for i in range(P):
        ax = axes[i]
        ax.hist(ranks[:, i], bins=edges, density=True, alpha=0.85)
        ax.set_title(param_names[i])
        ax.set_xlabel("rank")
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.2)

    for j in range(P, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(fig_out, "sbc_rank_hist.png"), dpi=250)
    plt.close()


def copy_nll_hist(plots_out: str, fig_out: str, split: str) -> None:
    src = os.path.join(plots_out, "eval_hybrid_ens", f"nllz_hist_{split}.png")
    dst = os.path.join(fig_out, f"nllz_hist_{split}.png")
    if os.path.exists(src):
        shutil.copy(src, dst)
    else:
        print("WARNING: nll histogram not found at", src)


def info_gain_bits(plots_out: str, data_out: str, fig_out: str, split: str, *, bins: int = 60) -> None:
    d = _read_eval_npz(os.path.join(plots_out, "eval_hybrid_ens"), split)
    meta = np.load(os.path.join(data_out, "param_meta.npz"), allow_pickle=True)

    param_names = [_decode(x) for x in meta["param_names"]]
    low = meta["prior_low"].astype(float)
    high = meta["prior_high"].astype(float)

    theta_true = d["theta_true"].astype(float)
    post = d["theta_samples"].astype(float)  # (N,S,P)

    P = theta_true.shape[1]

    def hist_prob(x, edges):
        h, _ = np.histogram(x, bins=edges)
        h = h.astype(float) + 1e-12
        return h / h.sum()

    ig_bits = []
    for i in range(P):
        edges = np.linspace(low[i], high[i], bins + 1)
        p = hist_prob(theta_true[:, i], edges)
        kls = []
        for n in range(theta_true.shape[0]):
            q = hist_prob(post[n, :, i], edges)
            kls.append(np.sum(q * (np.log(q) - np.log(p))))
        ig_bits.append(float(np.mean(kls) / np.log(2.0)))

    out = pd.DataFrame({"param": param_names, "info_gain_bits": ig_bits})
    csv_path = os.path.join(fig_out, f"info_gain_bits_{split}.csv")
    out.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    ax.bar(out["param"], out["info_gain_bits"])
    ax.set_ylabel("Information gain (bits)")
    ax.set_title(f"Marginal information gain — {split}")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_out, f"info_gain_bits_{split}.png"), dpi=250)
    plt.close()


def make_ablation_summary(plots_out: str, fig_out: str, split: str) -> None:
    """Create a small CSV summarizing key ablations (if present)."""
    candidates = {
        "Hybrid + param token + fullcov": os.path.join(plots_out, "eval_hybrid_ens"),
        "Hybrid + param token + diag": os.path.join(plots_out, "eval_hybrid_diag_ens"),
        "Hybrid − param token + fullcov": os.path.join(plots_out, "eval_hybrid_noparamtoken_ens"),
    }

    rows = []
    for label, d in candidates.items():
        csv_path = os.path.join(d, f"metrics_{split}.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        df = df[df["param"] != "MEAN"]
        rows.append(
            {
                "setting": label,
                "mean_pearson": float(np.nanmean(df["pearson_mean"])),
                "mean_rmse_norm": float(np.mean(df["rmse_norm_mean"])),
                "mean_cov90": float(np.mean(df["cov90"])),
            }
        )

    if not rows:
        return

    out_csv = os.path.join(fig_out, f"ablation_summary_{split}.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    ax.bar([r["setting"] for r in rows], [r["mean_pearson"] for r in rows])
    ax.set_ylabel("Mean Pearson r")
    ax.set_title(f"Ablations (mean recoverability) — {split}")
    ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_out, f"ablation_mean_pearson_{split}.png"), dpi=250)
    plt.close()


def run_ppc(plots_out: str, fig_out: str, split: str, *, n_examples: int, n_ppc_sims: int, seed: int) -> None:
    eval_npz = os.path.join(plots_out, "eval_hybrid_ens", f"eval_{split}_outputs.npz")

    # ppc.py lives in repo root in your project; call it via the current python
    cmd = [
        sys.executable,
        "ppc.py",
        "--eval-npz",
        eval_npz,
        "--out",
        fig_out,
        "--n-examples",
        str(n_examples),
        "--n-ppc-sims",
        str(n_ppc_sims),
        "--seed",
        str(seed),
    ]

    print("[PPC]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-out", required=True)
    ap.add_argument("--plots-out", required=True)
    ap.add_argument("--fig-out", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])

    ap.add_argument("--run-ppc", action="store_true")
    ap.add_argument("--ppc-n-examples", type=int, default=6)
    ap.add_argument("--ppc-n-sims", type=int, default=50)

    ap.add_argument("--run-sweeps", action="store_true")
    ap.add_argument(
        "--sweeps-model-dirs",
        type=str,
        nargs="+",
        default=None,
        help="Model run dirs (3 seeds) to use for sweeps; only used if --run-sweeps.",
    )
    ap.add_argument("--sweeps-n-eval", type=int, default=1500)
    ap.add_argument("--sweeps-n-post", type=int, default=200)

    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    _ensure_dir(args.fig_out)
    _set_paper_rcparams()

    make_recoverability_figures(args.plots_out, args.fig_out, args.split)
    make_architecture_compare_figures(args.plots_out, args.fig_out, args.split)

    make_scatter_and_residual_figures(args.plots_out, args.fig_out, args.split)
    make_coverage_curve(args.plots_out, args.fig_out, args.split)
    make_sbc_rank_hist(args.plots_out, args.fig_out, args.split)

    copy_nll_hist(args.plots_out, args.fig_out, args.split)
    info_gain_bits(args.plots_out, args.data_out, args.fig_out, args.split)
    make_ablation_summary(args.plots_out, args.fig_out, args.split)

    if args.run_ppc:
        run_ppc(
            args.plots_out,
            args.fig_out,
            args.split,
            n_examples=args.ppc_n_examples,
            n_ppc_sims=args.ppc_n_sims,
            seed=args.seed,
        )

    if args.run_sweeps:
        if not args.sweeps_model_dirs:
            raise SystemExit(
                "--run-sweeps was set but --sweeps-model-dirs was not provided. "
                "Pass the same three dirs you use for eval_hybrid_ens (seed0/1/2)."
            )

        # Determine if sweeps supports --data-out; otherwise use symlink ./data_out
        help_txt = subprocess.run(
            [sys.executable, "-m", "eval.generalization_sweeps", "--help"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        supports_data_out = "--data-out" in help_txt

        cmd = [sys.executable, "-m", "eval.generalization_sweeps"]
        if supports_data_out:
            cmd += ["--data-out", args.data_out]
        else:
            if os.path.exists("data_out") and not os.path.islink("data_out"):
                raise SystemExit("./data_out exists and is not a symlink. Rename/delete it, then rerun.")
            if os.path.islink("data_out"):
                os.unlink("data_out")
            os.symlink(os.path.abspath(args.data_out), "data_out")

        cmd += [
            "--model-dirs",
            *args.sweeps_model_dirs,
            "--features",
            "hybrid",
            "--split",
            args.split,
            "--n-eval",
            str(args.sweeps_n_eval),
            "--n-post-samples",
            str(args.sweeps_n_post),
            "--seed",
            str(args.seed),
            "--out-dir",
            args.fig_out,
        ]

        print("[SWEEPS]", " ".join(cmd))
        subprocess.run(cmd, check=True)

    print("All figures written to:", args.fig_out)


if __name__ == "__main__":
    main()