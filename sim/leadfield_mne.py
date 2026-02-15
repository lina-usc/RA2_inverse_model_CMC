from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


DEFAULT_16_CH_NAMES: Tuple[str, ...] = (
    "Fp1", "Fp2",
    "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "P4", "P8",
)


@dataclass(frozen=True)
class LeadfieldMeta:
    ch_names: Tuple[str, ...]
    montage_name: str
    head_radius_m: float
    source_radius_m: float
    n_sources: int
    seed: int


_CACHE: Dict[Tuple[Tuple[str, ...], int, int, int, str], Tuple[np.ndarray, object, LeadfieldMeta, np.ndarray, np.ndarray, np.ndarray]] = {}


def _import_mne_or_die():
    try:
        import mne  # type: ignore
        return mne
    except Exception as e:
        raise SystemExit(
            "ERROR: mne is required for Phase-1 leadfield.\n"
            "Install it:\n"
            "  python -m pip install mne\n"
        ) from e


def make_info(ch_names: List[str], fs: int, montage_name: str = "standard_1020"):
    """
    Deterministic EEG Info with MNE montage positions.
    """
    mne = _import_mne_or_die()
    info = mne.create_info(ch_names=ch_names, sfreq=float(fs), ch_types=["eeg"] * len(ch_names))
    montage = mne.channels.make_standard_montage(montage_name)
    info.set_montage(montage, match_case=False, on_missing="raise")
    return info


def _extract_ch_pos(info, ch_names: List[str]) -> np.ndarray:
    """
    Return electrode positions (n_channels,3) in meters.
    """
    montage = info.get_montage()
    pos = montage.get_positions()["ch_pos"]
    ch_pos = np.stack([pos[ch] for ch in ch_names], axis=0).astype(np.float64)
    return ch_pos


def _sample_sources(n_sources: int, head_radius: float, seed: int, radius_frac: float = 0.6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deterministically sample dipole source positions and radial orientations.

    Returns:
      src_pos: (n_sources,3)
      src_ori: (n_sources,3) unit vectors
    """
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n_sources, 3)).astype(np.float64)
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    src_ori = v.copy()
    src_pos = (radius_frac * head_radius) * v
    return src_pos, src_ori


def _analytic_leadfield(ch_pos: np.ndarray, src_pos: np.ndarray, src_ori: np.ndarray) -> np.ndarray:
    """
    Analytic dipole potential-like leadfield:
      L_ij = p_j · (r_i - r_sj) / ||r_i - r_sj||^3

    Then average-reference (columnwise zero-mean) and column-normalize.
    """
    n_ch = ch_pos.shape[0]
    n_src = src_pos.shape[0]
    L = np.zeros((n_ch, n_src), dtype=np.float64)

    for j in range(n_src):
        r = ch_pos - src_pos[j][None, :]
        dist = np.linalg.norm(r, axis=1) + 1e-12
        L[:, j] = (r @ src_ori[j]) / (dist ** 3)

    # average reference: remove column mean
    L = L - L.mean(axis=0, keepdims=True)

    # normalize columns
    coln = np.linalg.norm(L, axis=0, keepdims=True) + 1e-12
    L = L / coln
    return L.astype(np.float32)


def make_leadfield(
    fs: int = 250,
    n_sources: int = 3,
    ch_names: Optional[List[str]] = None,
    montage_name: str = "standard_1020",
    seed: int = 0,
) -> Tuple[np.ndarray, object, LeadfieldMeta, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      leadfield: (n_channels, n_sources) float32
      info: MNE Info (for plotting/reporting)
      meta: LeadfieldMeta
      ch_pos: (n_channels,3)
      src_pos: (n_sources,3)
      src_ori: (n_sources,3)
    """
    if ch_names is None:
        ch_names = list(DEFAULT_16_CH_NAMES)
    key = (tuple(ch_names), int(fs), int(n_sources), int(seed), montage_name)

    if key in _CACHE:
        return _CACHE[key]

    info = make_info(ch_names, fs=fs, montage_name=montage_name)
    ch_pos = _extract_ch_pos(info, ch_names)

    head_radius = float(np.median(np.linalg.norm(ch_pos, axis=1)))
    src_pos, src_ori = _sample_sources(n_sources=int(n_sources), head_radius=head_radius, seed=int(seed), radius_frac=0.6)

    L = _analytic_leadfield(ch_pos, src_pos, src_ori)

    meta = LeadfieldMeta(
        ch_names=tuple(ch_names),
        montage_name=montage_name,
        head_radius_m=head_radius,
        source_radius_m=float(0.6 * head_radius),
        n_sources=int(n_sources),
        seed=int(seed),
    )

    out = (L, info, meta, ch_pos.astype(np.float32), src_pos.astype(np.float32), src_ori.astype(np.float32))
    _CACHE[key] = out
    return out
