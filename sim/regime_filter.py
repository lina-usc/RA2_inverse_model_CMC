from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class RegimeFilterConfig:
    """
    Paper-explicit regime filter configuration.

    The filter defines which simulated EEG samples are considered "ERP-like"
    and therefore part of the effective generative model.
    """
    fs: int
    duration: float
    stim_onset: float

    # amplitude sanity
    abs99_min_uV: float = 1.0
    abs99_max_uV: float = 1500.0

    # evoked response requirement (GFP peak z-score)
    post_window_sec: float = 0.60
    gfp_peak_z_min: float = 5.0

    # sustained activity rejection
    late_window_sec: float = 0.50
    late_rms_factor: float = 4.0
    late_peak_factor: float = 0.80


def regime_reject(eeg: np.ndarray, cfg: RegimeFilterConfig) -> Tuple[bool, str]:
    """
    Reject pathological / non-ERP regimes.

    Returns
    -------
    ok : bool
    reason : str
      one of:
        - "ok"
        - "nan_or_inf"
        - "too_flat"
        - "too_large"
        - "no_evoked_peak"
        - "sustained_activity"
    """
    eeg = np.asarray(eeg, dtype=np.float32)

    if eeg.ndim != 2:
        return False, "bad_shape"
    if not np.isfinite(eeg).all():
        return False, "nan_or_inf"

    abs99 = float(np.percentile(np.abs(eeg), 99.0))
    if abs99 < float(cfg.abs99_min_uV):
        return False, "too_flat"
    if abs99 > float(cfg.abs99_max_uV):
        return False, "too_large"

    n_t = eeg.shape[1]
    stim_idx = int(round(float(cfg.stim_onset) * float(cfg.fs)))
    stim_idx = max(1, min(stim_idx, n_t - 2))

    gfp = np.std(eeg, axis=0).astype(np.float32)

    bl = gfp[:stim_idx]
    post_end = min(n_t, stim_idx + int(round(float(cfg.post_window_sec) * float(cfg.fs))))
    post = gfp[stim_idx:post_end]
    late = gfp[max(0, n_t - int(round(float(cfg.late_window_sec) * float(cfg.fs)))) :]

    bl_mean = float(bl.mean())
    bl_sd = float(bl.std() + 1e-6)
    bl_rms = float(np.sqrt(np.mean(bl**2)) + 1e-6)

    post_peak = float(post.max()) if post.size > 0 else float("nan")
    post_z = (post_peak - bl_mean) / bl_sd if np.isfinite(post_peak) else -np.inf

    if post_z < float(cfg.gfp_peak_z_min):
        return False, "no_evoked_peak"

    late_rms = float(np.sqrt(np.mean(late**2)) + 1e-6)
    if (late_rms > float(cfg.late_rms_factor) * bl_rms) and (late_rms > float(cfg.late_peak_factor) * post_peak):
        return False, "sustained_activity"

    return True, "ok"
