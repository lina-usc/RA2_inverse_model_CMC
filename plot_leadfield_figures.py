#!/usr/bin/env python3
"""Plot geometry and leadfield topomaps for the synthetic CMC→EEG forward model.

This script is designed to produce publication-friendly versions of:
  - Figure 1: sensor & source geometry WITH a head mesh (sphere)
  - Figure 2: three scalp topomaps (one per source) instead of a 16×3 heatmap

It does **not** depend on importing MNE (which can be fragile in some envs).
Instead it reads the standard_1020 electrode coordinates directly from
MNE's bundled `standard_1020.elc` file (if available).

Usage
-----
python plot_leadfield_figures.py --out plots --seed 0

Optional:
  --elc PATH   Manually provide a path to standard_1020.elc
  --n-sources  Number of sources (default 3)
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from matplotlib.patches import Circle  # noqa: E402
from matplotlib.tri import Triangulation, LinearTriInterpolator  # noqa: E402


DEFAULT_16_CH_NAMES: Tuple[str, ...] = (
    "Fp1", "Fp2",
    "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "P4", "P8",
)


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _find_mne_standard_1020_elc() -> Optional[str]:
    """Locate MNE's standard_1020.elc without importing mne."""
    spec = importlib.util.find_spec("mne")
    if spec is None or not spec.submodule_search_locations:
        return None
    mne_root = list(spec.submodule_search_locations)[0]
    cand = os.path.join(mne_root, "channels", "data", "montages", "standard_1020.elc")
    return cand if os.path.isfile(cand) else None


def _read_elc(path: str) -> Tuple[List[str], np.ndarray]:
    """Parse an ELC file (ASA electrode file). Returns (labels, positions_m)."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f.readlines()]

    # Find unit
    unit = "mm"
    for ln in lines:
        if ln.lower().startswith("unitposition"):
            # e.g., "UnitPosition\tmm"
            parts = ln.replace("\t", " ").split()
            if len(parts) >= 2:
                unit = parts[1].lower()
            break

    # Find number of positions
    n_pos = None
    for ln in lines:
        if ln.lower().startswith("numberpositions"):
            # e.g., "NumberPositions=\t97"
            ln2 = ln.replace("\t", " ")
            rhs = ln2.split("=")[-1].strip()
            try:
                n_pos = int(rhs)
            except Exception:
                pass
            break
    if n_pos is None:
        raise ValueError(f"Could not parse NumberPositions from {path}")

    # Find Positions block
    try:
        i_pos = next(i for i, ln in enumerate(lines) if ln.lower() == "positions")
    except StopIteration as e:
        raise ValueError(f"Could not find 'Positions' block in {path}") from e

    pos_lines = lines[i_pos + 1 : i_pos + 1 + n_pos]
    if len(pos_lines) != n_pos:
        raise ValueError(f"Expected {n_pos} position lines, got {len(pos_lines)}")

    pos = []
    for ln in pos_lines:
        parts = ln.split()
        if len(parts) < 3:
            raise ValueError(f"Bad position line: '{ln}'")
        pos.append([float(parts[0]), float(parts[1]), float(parts[2])])
    pos = np.asarray(pos, dtype=np.float64)

    # Find Labels block
    try:
        i_lab = next(i for i, ln in enumerate(lines) if ln.lower() == "labels")
    except StopIteration as e:
        raise ValueError(f"Could not find 'Labels' block in {path}") from e

    labels = [ln.strip() for ln in lines[i_lab + 1 : i_lab + 1 + n_pos]]
    if len(labels) != n_pos:
        raise ValueError(f"Expected {n_pos} labels, got {len(labels)}")

    # Units → meters
    if unit in ("mm", "millimeter", "millimeters"):
        pos_m = pos / 1000.0
    elif unit in ("m", "meter", "meters"):
        pos_m = pos
    else:
        raise ValueError(f"Unsupported UnitPosition '{unit}' in {path}")

    return labels, pos_m


def get_channel_positions_m(
    ch_names: List[str],
    elc_path: Optional[str] = None,
) -> np.ndarray:
    """Return (n_channels, 3) electrode coordinates in meters."""
    if elc_path is None:
        elc_path = _find_mne_standard_1020_elc()

    if elc_path is None:
        raise RuntimeError(
            "Could not locate MNE's 'standard_1020.elc'.\n"
            "Options:\n"
            "  (1) Install MNE (pip install mne)\n"
            "  (2) Provide --elc /path/to/standard_1020.elc\n"
        )

    labels, pos_m = _read_elc(elc_path)
    mapping: Dict[str, np.ndarray] = {lab: pos_m[i] for i, lab in enumerate(labels)}

    missing = [ch for ch in ch_names if ch not in mapping]
    if missing:
        raise KeyError(f"Channels not found in {elc_path}: {missing}")

    return np.stack([mapping[ch] for ch in ch_names], axis=0).astype(np.float64)


def sample_sources(n_sources: int, head_radius_m: float, seed: int, radius_frac: float = 0.6) -> Tuple[np.ndarray, np.ndarray]:
    """Same sampling as leadfield_mne.py: radial positions + radial orientations."""
    rng = np.random.default_rng(int(seed))
    v = rng.normal(size=(int(n_sources), 3)).astype(np.float64)
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    src_ori = v.copy()
    src_pos = float(radius_frac * head_radius_m) * v
    return src_pos, src_ori


def analytic_leadfield(ch_pos: np.ndarray, src_pos: np.ndarray, src_ori: np.ndarray) -> np.ndarray:
    """Same formula as leadfield_mne.py (scalar, orientation-collapsed)."""
    n_ch = ch_pos.shape[0]
    n_src = src_pos.shape[0]
    L = np.zeros((n_ch, n_src), dtype=np.float64)

    for j in range(n_src):
        r = ch_pos - src_pos[j][None, :]
        dist = np.linalg.norm(r, axis=1) + 1e-12
        L[:, j] = (r @ src_ori[j]) / (dist ** 3)

    # average reference
    L = L - L.mean(axis=0, keepdims=True)

    # column normalize
    coln = np.linalg.norm(L, axis=0, keepdims=True) + 1e-12
    L = L / coln

    return L.astype(np.float32)


def project_sphere_to_2d(ch_pos: np.ndarray) -> np.ndarray:
    """Simple azimuthal projection for topomaps.

    Assumes positions are roughly on a sphere centered at origin.
    Maps the north pole (z=+1) to center and the equator (z=0) to radius 1.
    """
    p = np.asarray(ch_pos, dtype=np.float64)
    p = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-12)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]

    colat = np.arccos(np.clip(z, -1.0, 1.0))  # 0..pi
    r2 = colat / (0.5 * np.pi)               # 0 at top, 1 at equator

    xy_norm = np.sqrt(x * x + y * y)
    x2 = r2 * x / (xy_norm + 1e-12)
    y2 = r2 * y / (xy_norm + 1e-12)
    return np.stack([x2, y2], axis=1)


def plot_topomap(
    ax: plt.Axes,
    xy: np.ndarray,
    values: np.ndarray,
    ch_names: Optional[List[str]] = None,
    vlim: Optional[float] = None,
    n_grid: int = 250,
    n_levels: int = 40,
):
    """Contour topomap using only matplotlib + triangulation."""
    xy = np.asarray(xy, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    tri = Triangulation(xy[:, 0], xy[:, 1])
    interp = LinearTriInterpolator(tri, values)

    xi = np.linspace(-1.05, 1.05, int(n_grid))
    yi = np.linspace(-1.05, 1.05, int(n_grid))
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = interp(Xi, Yi)

    mask = (Xi * Xi + Yi * Yi) > 1.0
    Zi = np.ma.array(Zi, mask=mask)

    if vlim is None:
        vlim = float(np.nanmax(np.abs(values)))
        vlim = max(vlim, 1e-6)

    levels = np.linspace(-vlim, vlim, int(n_levels))
    cs = ax.contourf(Xi, Yi, Zi, levels=levels)

    # head outline
    ax.add_patch(Circle((0.0, 0.0), 1.0, fill=False, linewidth=1.2))

    # sensors
    ax.scatter(xy[:, 0], xy[:, 1], s=18, marker="o", edgecolors="k", facecolors="none", linewidths=0.8)

    if ch_names is not None:
        for (x, y), name in zip(xy, ch_names):
            ax.text(x, y, name, fontsize=7, ha="center", va="center")

    ax.set_aspect("equal")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xticks([])
    ax.set_yticks([])

    return cs


def plot_geometry_3d(
    ch_pos: np.ndarray,
    src_pos: np.ndarray,
    head_radius_m: float,
    ch_names: Optional[List[str]],
    out_png: str,
):
    fig = plt.figure(figsize=(7.0, 6.0))
    ax = fig.add_subplot(111, projection="3d")

    # head mesh (sphere wireframe)
    u = np.linspace(0.0, 2.0 * np.pi, 40)
    v = np.linspace(0.0, np.pi, 20)
    xs = head_radius_m * np.outer(np.cos(u), np.sin(v))
    ys = head_radius_m * np.outer(np.sin(u), np.sin(v))
    zs = head_radius_m * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, rstride=2, cstride=2, linewidth=0.5, alpha=0.25)

    # points
    ax.scatter(ch_pos[:, 0], ch_pos[:, 1], ch_pos[:, 2], s=35, label="EEG sensors")
    ax.scatter(src_pos[:, 0], src_pos[:, 1], src_pos[:, 2], s=80, marker="^", label="Sources")

    if ch_names is not None:
        for p, name in zip(ch_pos, ch_names):
            ax.text(p[0], p[1], p[2], name, fontsize=7)

    # formatting
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("Sensor and source geometry (with head mesh)")

    # equal aspect (manual for mpl < 3.3 compatibility)
    mins = np.min(np.vstack([ch_pos, src_pos]), axis=0)
    maxs = np.max(np.vstack([ch_pos, src_pos]), axis=0)
    center = 0.5 * (mins + maxs)
    span = np.max(maxs - mins)
    ax.set_xlim(center[0] - 0.6 * span, center[0] + 0.6 * span)
    ax.set_ylim(center[1] - 0.6 * span, center[1] + 0.6 * span)
    ax.set_zlim(center[2] - 0.6 * span, center[2] + 0.6 * span)

    ax.view_init(elev=18, azim=35)
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--seed", type=int, default=0, help="Seed for deterministic source sampling")
    ap.add_argument("--n-sources", type=int, default=3)
    ap.add_argument("--elc", default=None, help="Path to standard_1020.elc (optional)")
    ap.add_argument("--label-channels", action="store_true", help="Write channel labels on plots")
    args = ap.parse_args()

    _ensure_dir(args.out)

    ch_names = list(DEFAULT_16_CH_NAMES)
    ch_pos = get_channel_positions_m(ch_names, elc_path=args.elc)

    head_radius_m = float(np.median(np.linalg.norm(ch_pos, axis=1)))
    src_pos, src_ori = sample_sources(n_sources=args.n_sources, head_radius_m=head_radius_m, seed=args.seed, radius_frac=0.6)

    L = analytic_leadfield(ch_pos, src_pos, src_ori)

    # ---- Figure 1: geometry ----
    out_geom = os.path.join(args.out, "figure1_geometry.png")
    plot_geometry_3d(
        ch_pos=ch_pos,
        src_pos=src_pos,
        head_radius_m=head_radius_m,
        ch_names=(ch_names if args.label_channels else None),
        out_png=out_geom,
    )

    # ---- Figure 2: topomaps (one per source) ----
    xy = project_sphere_to_2d(ch_pos)

    vlim = float(np.max(np.abs(L)))
    vlim = max(vlim, 1e-6)

    # Leave room on the right for a dedicated colorbar axis
    fig, axes = plt.subplots(1, args.n_sources, figsize=(4.2 * args.n_sources + 1.0, 4.0))
    if args.n_sources == 1:
        axes = [axes]

    last_cs = None
    for k in range(args.n_sources):
        ax = axes[k]
        last_cs = plot_topomap(
            ax=ax,
            xy=xy,
            values=L[:, k],
            ch_names=(ch_names if args.label_channels else None),
            vlim=vlim,
        )
        ax.set_title(f"Source {k+1}")

    if last_cs is not None:
        # Add a stable colorbar that never overlaps subplots
        cax = fig.add_axes([0.93, 0.18, 0.015, 0.64])  # [left, bottom, width, height]
        fig.colorbar(last_cs, cax=cax, label="Leadfield weight (a.u.)")

    fig.suptitle("Leadfield scalp projections (topomaps)")
    fig.subplots_adjust(left=0.03, right=0.90, top=0.82, bottom=0.05, wspace=0.25)
    out_topo = os.path.join(args.out, "figure2_leadfield_topomaps.png")
    fig.savefig(out_topo, dpi=300)
    plt.close(fig)

    # Also save the actual leadfield + geometry for reproducibility
    out_npz = os.path.join(args.out, "leadfield_geometry_seeded.npz")
    np.savez(
        out_npz,
        ch_names=np.asarray(ch_names, dtype=object),
        ch_pos_m=ch_pos.astype(np.float32),
        src_pos_m=src_pos.astype(np.float32),
        src_ori=src_ori.astype(np.float32),
        leadfield=L.astype(np.float32),
        head_radius_m=np.float32(head_radius_m),
        seed=np.int32(args.seed),
    )

    print("[plot_leadfield_figures] wrote:")
    print(" ", out_geom)
    print(" ", out_topo)
    print(" ", out_npz)


if __name__ == "__main__":
    main()
