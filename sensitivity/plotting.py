from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _savefig_formats(fig: plt.Figure, out_base: Path, formats: Sequence[str]) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(out_base.with_suffix(f".{fmt}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _maybe_rotation(n: int) -> int:
    return 30 if n <= 12 else 60


def plot_morris_family_summary(
    mu_star_by_family: Mapping[str, np.ndarray],
    sigma_by_family: Mapping[str, np.ndarray],
    param_names: Sequence[str],
    out_base: Path,
    formats: Sequence[str] = ("png",),
    title: str = "",
) -> None:
    families = [k for k in ["scalar", "erp", "tfr", "hybrid"] if k in mu_star_by_family]
    x = np.arange(len(param_names), dtype=np.float64)
    width = 0.8 / max(1, len(families))

    fig, axes = plt.subplots(2, 1, figsize=(max(10, 1.1 * len(param_names)), 8), sharex=True)
    for i, fam in enumerate(families):
        dx = (i - 0.5 * (len(families) - 1)) * width
        axes[0].bar(x + dx, np.asarray(mu_star_by_family[fam], dtype=np.float64), width=width, label=fam.upper(), alpha=0.9)
        axes[1].bar(x + dx, np.asarray(sigma_by_family.get(fam, np.full_like(mu_star_by_family[fam], np.nan)), dtype=np.float64), width=width, label=fam.upper(), alpha=0.9)

    axes[0].set_ylabel("Mean Morris $\\mu^*$")
    axes[1].set_ylabel("Mean Morris $\\sigma$")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(param_names, rotation=_maybe_rotation(len(param_names)), ha="right")
    axes[0].legend(ncol=max(1, min(4, len(families))), frameon=False)
    if title:
        axes[0].set_title(title)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
    _savefig_formats(fig, out_base, formats)



def plot_erp_heatmaps(
    mu_star_time_channel: np.ndarray,
    param_names: Sequence[str],
    channel_names: Sequence[str],
    time_edges: np.ndarray,
    out_base: Path,
    formats: Sequence[str] = ("png",),
) -> None:
    arr = np.asarray(mu_star_time_channel, dtype=np.float64)
    P, T, C = arr.shape
    ncols = 3
    nrows = int(np.ceil(P / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.2 * nrows), squeeze=False)
    vmax = float(np.nanmax(arr)) if np.isfinite(arr).any() else 1.0

    for p in range(nrows * ncols):
        ax = axes.flat[p]
        if p >= P:
            ax.axis("off")
            continue
        im = ax.imshow(arr[p].T, origin="lower", aspect="auto", vmin=0.0, vmax=vmax)
        ax.set_title(str(param_names[p]))
        ax.set_xlabel("Time patch")
        ax.set_ylabel("Channel")
        ax.set_xticks(np.arange(T))
        ax.set_xticklabels([f"{0.5 * (time_edges[i] + time_edges[i + 1]):.1f}" for i in range(T)], rotation=90, fontsize=7)
        ax.set_yticks(np.arange(C))
        ax.set_yticklabels(channel_names, fontsize=7)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    cbar.set_label("Morris $\\mu^*$")
    fig.suptitle("ERP Morris sensitivity heatmaps", y=1.02)
    _savefig_formats(fig, out_base, formats)



def plot_tfr_heatmaps(
    mu_star_time_freq: np.ndarray,
    param_names: Sequence[str],
    time_edges: np.ndarray,
    freq_edges: np.ndarray,
    out_base: Path,
    formats: Sequence[str] = ("png",),
) -> None:
    arr = np.asarray(mu_star_time_freq, dtype=np.float64)
    P, T, F = arr.shape
    ncols = 3
    nrows = int(np.ceil(P / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.2 * nrows), squeeze=False)
    vmax = float(np.nanmax(arr)) if np.isfinite(arr).any() else 1.0

    for p in range(nrows * ncols):
        ax = axes.flat[p]
        if p >= P:
            ax.axis("off")
            continue
        im = ax.imshow(arr[p].T, origin="lower", aspect="auto", vmin=0.0, vmax=vmax)
        ax.set_title(str(param_names[p]))
        ax.set_xlabel("Time patch")
        ax.set_ylabel("Freq patch")
        ax.set_xticks(np.arange(T))
        ax.set_xticklabels([f"{0.5 * (time_edges[i] + time_edges[i + 1]):.1f}" for i in range(T)], rotation=90, fontsize=7)
        ax.set_yticks(np.arange(F))
        ax.set_yticklabels([f"{np.sqrt(freq_edges[i] * freq_edges[i + 1]):.1f}" for i in range(F)], fontsize=7)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    cbar.set_label("Mean channel-collapsed Morris $\\mu^*$")
    fig.suptitle("TFR Morris sensitivity heatmaps", y=1.02)
    _savefig_formats(fig, out_base, formats)



def plot_acceptance_morris(
    mu_star: np.ndarray,
    sigma: np.ndarray,
    param_names: Sequence[str],
    out_base: Path,
    formats: Sequence[str] = ("png",),
) -> None:
    x = np.arange(len(param_names), dtype=np.float64)
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(9, 1.1 * len(param_names)), 4.5))
    ax.bar(x - width / 2.0, np.asarray(mu_star, dtype=np.float64), width=width, label="$\\mu^*$", alpha=0.9)
    ax.bar(x + width / 2.0, np.asarray(sigma, dtype=np.float64), width=width, label="$\\sigma$", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(param_names, rotation=_maybe_rotation(len(param_names)), ha="right")
    ax.set_ylabel("Acceptance-indicator sensitivity")
    ax.set_title("Morris sensitivity of acceptance indicator")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    _savefig_formats(fig, out_base, formats)



def plot_sobol_bars(
    S1: np.ndarray,
    ST: np.ndarray,
    param_names: Sequence[str],
    output_names: Sequence[str],
    out_base: Path,
    family_name: str,
    formats: Sequence[str] = ("png",),
) -> None:
    S1 = np.asarray(S1, dtype=np.float64)
    ST = np.asarray(ST, dtype=np.float64)
    if S1.ndim != 2 or ST.shape != S1.shape:
        raise ValueError("S1 and ST must both have shape (n_outputs, n_params)")

    n_outputs, n_params = S1.shape
    ncols = 2 if n_outputs > 1 else 1
    nrows = int(np.ceil(n_outputs / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(max(10, 5.2 * ncols), max(3.8, 3.2 * nrows)), squeeze=False)
    x = np.arange(n_params, dtype=np.float64)
    width = 0.38

    for i in range(nrows * ncols):
        ax = axes.flat[i]
        if i >= n_outputs:
            ax.axis("off")
            continue
        ax.bar(x - width / 2.0, S1[i], width=width, label="S1", alpha=0.9)
        ax.bar(x + width / 2.0, ST[i], width=width, label="ST", alpha=0.9)
        ax.set_title(str(output_names[i]))
        ax.set_xticks(x)
        ax.set_xticklabels(param_names, rotation=45, ha="right")
        ax.set_ylim(bottom=min(-0.05, np.nanmin(S1[i]) - 0.05 if np.isfinite(S1[i]).any() else -0.05), top=max(1.0, np.nanmax(ST[i]) + 0.05 if np.isfinite(ST[i]).any() else 1.0))
        ax.grid(axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes.flat[0].legend(frameon=False)
    fig.suptitle(f"Sobol S1/ST for {family_name} outputs", y=1.02)
    _savefig_formats(fig, out_base, formats)



def plot_rank_comparison_grid(
    rank_grid: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    param_names: Sequence[str],
    out_base: Path,
    formats: Sequence[str] = ("png",),
) -> None:
    labels = list(rank_grid.keys())
    ncols = len(labels)
    fig, axes = plt.subplots(1, ncols, figsize=(5.0 * ncols, 4.4), squeeze=False)
    for ax, label in zip(axes.flat, labels):
        sens_rank, rec_rank = rank_grid[label]
        sens_rank = np.asarray(sens_rank, dtype=np.float64)
        rec_rank = np.asarray(rec_rank, dtype=np.float64)
        mask = np.isfinite(sens_rank) & np.isfinite(rec_rank)
        if np.any(mask):
            ax.scatter(sens_rank[mask], rec_rank[mask], s=40)
            for x, y, nm in zip(sens_rank[mask], rec_rank[mask], np.asarray(param_names)[mask]):
                ax.text(float(x) + 0.05, float(y) + 0.05, str(nm), fontsize=8)
            max_rank = int(max(np.nanmax(sens_rank[mask]), np.nanmax(rec_rank[mask])))
            ax.plot([1, max_rank], [1, max_rank], linestyle="--", linewidth=1.0)
            ax.set_xlim(0.5, max_rank + 0.5)
            ax.set_ylim(max_rank + 0.5, 0.5)
        ax.set_title(label)
        ax.set_xlabel("Sensitivity rank (1 = highest)")
        ax.set_ylabel("Recoverability rank (1 = highest)")
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Sensitivity ranking vs recoverability ranking", y=1.02)
    _savefig_formats(fig, out_base, formats)



def plot_value_scatter(
    x_values: np.ndarray,
    y_values: np.ndarray,
    param_names: Sequence[str],
    x_label: str,
    y_label: str,
    title: str,
    out_base: Path,
    formats: Sequence[str] = ("png",),
) -> None:
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    if np.any(mask):
        ax.scatter(x[mask], y[mask], s=45)
        for xv, yv, nm in zip(x[mask], y[mask], np.asarray(param_names)[mask]):
            ax.text(float(xv) + 0.01, float(yv) + 0.01, str(nm), fontsize=8)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _savefig_formats(fig, out_base, formats)
