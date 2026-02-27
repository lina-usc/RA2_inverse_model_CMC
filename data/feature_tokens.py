#data/feature_tokens.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np
from scipy.signal import stft


@dataclass(frozen=True)
class TokenConfig:
    # signals
    fs: int = 250
    duration: float = 2.0
    n_channels: int = 16

    # token grid
    n_time_patches: int = 25
    n_freq_patches: int = 15  # TFR only

    # stimulus timing inside analysis window (seconds)
    stim_onset: float = 0.5

    # TFR definition (explicit + fast)
    f_min: float = 2.0
    f_max: float = 40.0
    nperseg: int = 128
    noverlap: int = 112
    nfft: int = 256

    eps: float = 1e-8


def compute_erp_tokens(eeg: np.ndarray, cfg: TokenConfig) -> np.ndarray:
    """
    ERP tokens: split time axis into n_time_patches.
    Each token = per-channel mean in the window.

    Output: (n_time_patches, n_channels) = (25, 16)
    """
    eeg = np.asarray(eeg, dtype=np.float32)
    n_ch, n_t = eeg.shape
    if n_ch != cfg.n_channels:
        raise ValueError(f"Expected n_channels={cfg.n_channels}, got {n_ch}")

    expected = int(round(cfg.duration * cfg.fs))
    if n_t != expected:
        raise ValueError(f"ERP: eeg length {n_t} != duration*fs {expected}")

    if n_t % cfg.n_time_patches != 0:
        raise ValueError("duration*fs must be divisible by n_time_patches")

    w = n_t // cfg.n_time_patches
    tokens = eeg.reshape(n_ch, cfg.n_time_patches, w).mean(axis=2).T.astype(np.float32)
    return tokens


def _bin_mask_or_nearest(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    m = (x >= lo) & (x < hi)
    if np.any(m):
        return m
    mid = 0.5 * (lo + hi)
    j = int(np.argmin(np.abs(x - mid)))
    m2 = np.zeros_like(x, dtype=bool)
    m2[j] = True
    return m2


def compute_tfr_tokens(eeg: np.ndarray, cfg: TokenConfig) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    TFR tokens: STFT power per channel, baseline-normalized in dB relative to t < stim_onset,
    then patch-averaged into (n_time_patches, n_freq_patches).
    Time-major then frequency ordering.

    Returns:
      tokens_tfr: (n_time_patches*n_freq_patches, n_channels) = (375,16)
      meta: f/t edges etc
    """
    eeg = np.asarray(eeg, dtype=np.float32)

    f, t, Z = stft(
        eeg,
        fs=cfg.fs,
        nperseg=cfg.nperseg,
        noverlap=cfg.noverlap,
        nfft=cfg.nfft,
        boundary="zeros",
        padded=True,
        axis=-1,
    )
    # Z: (ch, f, t)
    power = (np.abs(Z) ** 2).astype(np.float32)

    # restrict freq range
    fmask = (f >= cfg.f_min) & (f <= cfg.f_max)
    f_sel = f[fmask].astype(np.float32)
    power = power[:, fmask, :]

    t = t.astype(np.float32)
    bl_mask = t < cfg.stim_onset
    if not np.any(bl_mask):
        bl_mask = np.zeros_like(t, dtype=bool)
        bl_mask[:2] = True

    bl = power[:, :, bl_mask].mean(axis=2, keepdims=True) + cfg.eps
    power_db = 10.0 * np.log10((power + cfg.eps) / bl)

    time_edges = np.linspace(0.0, cfg.duration, cfg.n_time_patches + 1, dtype=np.float32)
    freq_edges = np.geomspace(cfg.f_min, cfg.f_max, cfg.n_freq_patches + 1).astype(np.float32)

    tfr_patch = np.zeros((cfg.n_time_patches, cfg.n_freq_patches, cfg.n_channels), dtype=np.float32)

    for ti in range(cfg.n_time_patches):
        t0, t1 = float(time_edges[ti]), float(time_edges[ti + 1])
        tm = _bin_mask_or_nearest(t, t0, t1)

        for fi in range(cfg.n_freq_patches):
            f0, f1 = float(freq_edges[fi]), float(freq_edges[fi + 1])
            fm = _bin_mask_or_nearest(f_sel, f0, f1)

            patch = power_db[:, fm, :][:, :, tm].mean(axis=(1, 2))
            tfr_patch[ti, fi, :] = patch.astype(np.float32)

    tokens = tfr_patch.reshape(cfg.n_time_patches * cfg.n_freq_patches, cfg.n_channels)
    meta = {
        "stft_f": f_sel,
        "stft_t": t,
        "time_edges": time_edges,
        "freq_edges": freq_edges,
        "tfr_patch_shape": np.array(tfr_patch.shape, dtype=np.int32),
    }
    return tokens, meta


from sim.regime_filter import RegimeFilterConfig, regime_reject as _regime_reject

def regime_reject(eeg: np.ndarray, cfg: TokenConfig):
    rf_cfg = RegimeFilterConfig(fs=cfg.fs, duration=cfg.duration, stim_onset=cfg.stim_onset)
    return _regime_reject(eeg, rf_cfg)
