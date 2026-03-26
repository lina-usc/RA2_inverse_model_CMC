# data/feature_tokens.py

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

    # TFR band
    f_min: float = 2.0
    f_max: float = 40.0

    # --- STFT params (current default backend) ---
    nperseg: int = 128
    noverlap: int = 112
    nfft: int = 256

    # --- NEW: choose TFR backend ---
    # "stft" keeps your current behavior.
    # "morlet" uses MNE Morlet wavelets, then we patch-average to the SAME 25x15 grid.
    tfr_method: str = "stft"  # {"stft", "morlet"}

    # --- NEW: Morlet params (only used if tfr_method == "morlet") ---
    morlet_n_freqs: int = 48            # number of wavelet freqs computed before binning to 15 patches
    morlet_n_cycles_low: float = 3.0    # cycles at f_min
    morlet_n_cycles_high: float = 10.0  # cycles at f_max
    morlet_decim: int = 1               # decimation factor for speed (1 keeps all time points)
    morlet_n_jobs: int = 1              # parallelism inside MNE

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
    tokens = np.zeros((cfg.n_time_patches, cfg.n_channels), dtype=np.float32)
    for i in range(cfg.n_time_patches):
        sl = slice(i * w, (i + 1) * w)
        tokens[i] = eeg[:, sl].mean(axis=1)
    return tokens


def _bin_mask_or_nearest(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Return boolean mask for [lo,hi); if empty, return mask of nearest bin."""
    m = (x >= lo) & (x < hi)
    if np.any(m):
        return m
    mid = 0.5 * (lo + hi)
    j = int(np.argmin(np.abs(x - mid)))
    m2 = np.zeros_like(x, dtype=bool)
    m2[j] = True
    return m2


def _tfr_power_stft(eeg: np.ndarray, cfg: TokenConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Return (freqs, times, power[ch,f,t], backend_meta)."""
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
    power = (np.abs(Z) ** 2).astype(np.float32)  # (ch, f, t)

    # restrict freq range
    fmask = (f >= cfg.f_min) & (f <= cfg.f_max)
    f_sel = f[fmask].astype(np.float32)
    power = power[:, fmask, :]

    return f_sel, t.astype(np.float32), power, {
        "backend": "stft",
        "nperseg": cfg.nperseg,
        "noverlap": cfg.noverlap,
        "nfft": cfg.nfft,
    }


def _tfr_power_morlet(eeg: np.ndarray, cfg: TokenConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Return (freqs, times, power[ch,f,t], backend_meta) using MNE Morlet."""
    try:
        from mne.time_frequency import tfr_array_morlet  # type: ignore
    except Exception as e:
        raise ImportError(
            "Morlet requested (tfr_method='morlet') but MNE is not available. "
            "Install mne, or switch back to tfr_method='stft'."
        ) from e

    # Compute wavelet power on a log-spaced frequency grid, then we bin to 15 patches later.
    freqs = np.geomspace(cfg.f_min, cfg.f_max, int(cfg.morlet_n_freqs)).astype(np.float64)

    # Vary cycles from low->high freq (keeps high-freq estimates from being too noisy).
    n_cycles = np.linspace(cfg.morlet_n_cycles_low, cfg.morlet_n_cycles_high, freqs.size).astype(np.float64)

    # MNE expects (n_epochs, n_channels, n_times)
    data = np.asarray(eeg, dtype=np.float64)[None, :, :]

    power = tfr_array_morlet(
        data,
        sfreq=float(cfg.fs),
        freqs=freqs,
        n_cycles=n_cycles,
        output="power",
        decim=int(cfg.morlet_decim),
        n_jobs=int(cfg.morlet_n_jobs),
        zero_mean=True,
    )[0]  # -> (ch, f, t)

    power = power.astype(np.float32)
    times = (np.arange(power.shape[-1], dtype=np.float32) * float(cfg.morlet_decim) / float(cfg.fs)).astype(np.float32)

    return freqs.astype(np.float32), times, power, {
        "backend": "morlet",
        "morlet_n_freqs": int(cfg.morlet_n_freqs),
        "morlet_n_cycles_low": float(cfg.morlet_n_cycles_low),
        "morlet_n_cycles_high": float(cfg.morlet_n_cycles_high),
        "morlet_decim": int(cfg.morlet_decim),
    }


def compute_tfr_tokens(eeg: np.ndarray, cfg: TokenConfig) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    TFR tokens: time-frequency power per channel, baseline-normalized in dB relative to t < stim_onset,
    then patch-averaged into (n_time_patches, n_freq_patches).
    Time-major then frequency ordering.

    Returns:
      tokens_tfr: (n_time_patches*n_freq_patches, n_channels) = (375,16)
      meta: f/t edges etc
    """
    eeg = np.asarray(eeg, dtype=np.float32)

    method = str(cfg.tfr_method).lower().strip()
    if method == "stft":
        f_sel, t, power, backend_meta = _tfr_power_stft(eeg, cfg)
    elif method == "morlet":
        f_sel, t, power, backend_meta = _tfr_power_morlet(eeg, cfg)
    else:
        raise ValueError("tfr_method must be one of: 'stft', 'morlet'")

    # Baseline mask (pre-stim). If empty, fall back to earliest samples.
    bl_mask = t < cfg.stim_onset
    if not np.any(bl_mask):
        bl_mask = np.zeros_like(t, dtype=bool)
        bl_mask[: max(1, min(2, t.size))] = True

    bl = power[:, :, bl_mask].mean(axis=2, keepdims=True) + cfg.eps
    power_db = 10.0 * np.log10((power + cfg.eps) / bl)

    # Patch edges
    time_edges = np.linspace(0.0, cfg.duration, cfg.n_time_patches + 1, dtype=np.float32)
    freq_edges = np.geomspace(cfg.f_min, cfg.f_max, cfg.n_freq_patches + 1).astype(np.float32)

    # Patch-average into (time, freq, ch)
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

    # Keep the original meta keys ("stft_f", "stft_t") for compatibility with your existing pipeline,
    # even when the backend is Morlet.
    meta: Dict[str, Any] = {
        "tfr_backend": np.array(method, dtype="S"),
        "stft_f": f_sel,
        "stft_t": t,
        "time_edges": time_edges,
        "freq_edges": freq_edges,
        "tfr_patch_shape": np.array(tfr_patch.shape, dtype=np.int32),
    }
    meta.update({f"backend_{k}": v for k, v in backend_meta.items()})
    return tokens, meta


from sim.regime_filter import RegimeFilterConfig, regime_reject as _regime_reject


def regime_reject(eeg: np.ndarray, cfg: TokenConfig):
    rf_cfg = RegimeFilterConfig(fs=cfg.fs, duration=cfg.duration, stim_onset=cfg.stim_onset)
    return _regime_reject(eeg, rf_cfg)