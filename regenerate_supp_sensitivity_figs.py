from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _rank_desc(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(-np.nan_to_num(values, nan=-np.inf))
    ranks = np.empty_like(order, dtype=int)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def _find_existing(paths: List[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError("None of the candidate paths exist:\n" + "\n".join(str(p) for p in paths))


def plot_acceptance_combined(results_root: Path, out_base: Path) -> None:
    morris_csv = _find_existing([
        results_root / "morris" / "morris_acceptance_stats.csv",
        results_root / "morris_acceptance_stats.csv",
    ])
    sobol_npz = _find_existing([
        results_root / "sobol" / "sobol_acceptance_stats.npz",
        results_root / "sobol_acceptance_stats.npz",
    ])

    morris = pd.read_csv(morris_csv)
    sobol = np.load(sobol_npz, allow_pickle=True)

    params = morris["param"].tolist()
    mu_star = morris["mu_star"].to_numpy(dtype=float)
    sigma = morris["sigma"].to_numpy(dtype=float)

    S1 = np.asarray(sobol["S1"], dtype=float).reshape(-1)
    ST = np.asarray(sobol["ST"], dtype=float).reshape(-1)

    x = np.arange(len(params))
    width = 0.38

    fig, axes = plt.subplots(
        2, 1, figsize=(8.2, 8.4), constrained_layout=True
    )

    ax = axes[0]
    ax.bar(x - width / 2, mu_star, width=width, label=r"$\mu^\ast$")
    ax.bar(x + width / 2, sigma, width=width, label=r"$\sigma$")
    ax.set_title("Morris sensitivity of acceptance indicator", pad=10)
    ax.set_ylabel("Morris sensitivity")
    ax.set_xticks(x)
    ax.set_xticklabels(params, rotation=30, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)

    ax = axes[1]
    ax.bar(x - width / 2, S1, width=width, label="S1")
    ax.bar(x + width / 2, ST, width=width, label="ST")
    ax.set_title("Sobol S1/ST for acceptance indicator", pad=10)
    ax.set_ylabel("Sobol index")
    ax.set_xticks(x)
    ax.set_xticklabels(params, rotation=30, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)

    for ext in [".png", ".pdf"]:
        fig.savefig(str(out_base) + ext, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_rank_comparison_tall(results_root: Path, out_base: Path) -> None:
    table_csv = _find_existing([
        results_root / "comparisons" / "sensitivity_recoverability_table.csv",
        results_root / "sensitivity_recoverability_table.csv",
    ])
    df = pd.read_csv(table_csv)

    families = ["erp", "tfr", "hybrid"]
    fig, axes = plt.subplots(
        3, 1, figsize=(7.6, 10.8), constrained_layout=True
    )

    for ax, family in zip(axes, families):
        sens = df[f"morris_{family}_mean_mu_star"].to_numpy(dtype=float)
        rec = df[f"recoverability_{family}_pearson"].to_numpy(dtype=float)
        params = df["param"].tolist()

        sens_rank = _rank_desc(sens)
        rec_rank = _rank_desc(rec)

        ax.plot([1, len(params)], [1, len(params)], linestyle="--", linewidth=1.5)
        ax.scatter(sens_rank, rec_rank, s=90)

        for x, y, label in zip(sens_rank, rec_rank, params):
            ax.annotate(
                label,
                (x, y),
                xytext=(5, 3),
                textcoords="offset points",
                fontsize=10,
            )

        ax.set_title(family.upper(), pad=8)
        ax.set_xlim(0.5, len(params) + 0.5)
        ax.set_ylim(len(params) + 0.5, 0.5)
        ax.set_xticks(np.arange(1, len(params) + 1))
        ax.set_yticks(np.arange(1, len(params) + 1))
        ax.set_xlabel("Sensitivity rank (1 = highest)")
        ax.set_ylabel("Recoverability rank (1 = highest)")
        ax.grid(True, alpha=0.25)

    for ext in [".png", ".pdf"]:
        fig.savefig(str(out_base) + ext, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=str, default="results_sensitivity_morlet_aligned_v2")
    ap.add_argument("--fig-dir", type=str, default=None)
    args = ap.parse_args()

    results_root = Path(args.results_root)
    fig_dir = Path(args.fig_dir) if args.fig_dir else (results_root / "figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_acceptance_combined(
        results_root=results_root,
        out_base=fig_dir / "acceptance_indicator_combined",
    )
    plot_rank_comparison_tall(
        results_root=results_root,
        out_base=fig_dir / "sensitivity_ranking_vs_recoverability_tall",
    )

    print("Wrote:")
    print(fig_dir / "acceptance_indicator_combined.png")
    print(fig_dir / "acceptance_indicator_combined.pdf")
    print(fig_dir / "sensitivity_ranking_vs_recoverability_tall.png")
    print(fig_dir / "sensitivity_ranking_vs_recoverability_tall.pdf")


if __name__ == "__main__":
    main()
