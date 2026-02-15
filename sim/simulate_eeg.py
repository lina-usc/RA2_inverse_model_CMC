from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
from scipy.signal import butter, filtfilt, resample_poly

from sim.cmc_simulator import simulate_sources_batch
from sim.leadfield_mne import make_leadfield


def simulate_eeg(
    params: Dict[str, float],
    fs: int = 250,
    duration: float = 2.0,
    n_channels: int = 16,
    seed: Optional[int] = None,
    bandpass: Optional[Tuple[float, float]] = (0.5, 40.0),
    stim_onset: float = 0.5,
    stim_sigma: float = 0.05,
    n_sources: int = 3,
    leadfield: Optional[np.ndarray] = None,
    sensor_noise_std: float = 2.0,
    n_trials: int = 10,
    input_noise_std: float = 0.2,
    internal_fs: int = 1000,
    baseline_correct: bool = True,
    baseline_window: Optional[Tuple[float, float]] = None,
    warmup_sec: float = 3.0,
    downsample_method: str = "slice",
    uV_scale: float = 100.0,
    return_trials: bool = False,
    return_sources: bool = False,
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    Paper-explicit CMC EEG forward wrapper (FAST batched simulation).

    Pipeline:
      (1) simulate_sources_batch for (n_trials*n_sources) realizations at internal_fs
      (2) reshape to (n_trials, n_sources, time)
      (3) trim warmup, downsample to fs
      (4) leadfield mixing to EEG + sensor noise (per trial)
      (5) average across trials
      (6) baseline correction (default [0, stim_onset))
      (7) bandpass filter
    """
    if warmup_sec <= 0:
        raise ValueError("warmup_sec must be > 0 for steady-state handling (paper requirement).")

    fs = int(fs)
    internal_fs = int(internal_fs)
    if internal_fs % fs != 0:
        raise ValueError("internal_fs must be an integer multiple of fs.")
    ds = internal_fs // fs

    n_out = int(np.round(duration * fs))
    n_warm_out = int(np.round(warmup_sec * fs))
    total_out = n_warm_out + n_out

    total_duration = total_out / float(fs)
    stim_onset_total = stim_onset + warmup_sec

    n_int_out = n_out * ds
    n_warm_int = n_warm_out * ds

    rng = np.random.default_rng(seed)

    # deterministic MNE-based leadfield if not provided
    if leadfield is None:
        L, _info, _meta, _ch_pos, _src_pos, _src_ori = make_leadfield(fs=fs, n_sources=n_sources, seed=0)
        leadfield = L

    leadfield = np.asarray(leadfield, dtype=np.float32)
    if leadfield.shape != (n_channels, n_sources):
        raise ValueError(f"leadfield must have shape {(n_channels, n_sources)}, got {leadfield.shape}")

    # ---- batched source sim for all trials*sources (one time loop) ----
    n_sims = int(n_trials) * int(n_sources)
    batch_seed = int(rng.integers(0, np.iinfo(np.int32).max))

    src_all = simulate_sources_batch(
        params=params,
        internal_fs=internal_fs,
        duration=total_duration,
        stim_onset=stim_onset_total,
        stim_sigma=stim_sigma,
        input_noise_std=input_noise_std,
        n_sims=n_sims,
        seed=batch_seed,
    )  # (n_sims, n_int_total)

    # reshape to (trials, sources, time)
    n_int_total = src_all.shape[1]
    src_all = src_all.reshape(int(n_trials), int(n_sources), n_int_total)

    # trim warmup
    src_use = src_all[:, :, n_warm_int : n_warm_int + n_int_out]  # (trials,sources,n_int_out)

    # downsample
    if downsample_method == "slice":
        src_ds = src_use[:, :, ::ds]
    elif downsample_method == "poly":
        src_ds = resample_poly(src_use, up=1, down=ds, axis=-1)
    else:
        raise ValueError("downsample_method must be 'slice' or 'poly'")

    src_ds = src_ds[:, :, :n_out]  # (trials,sources,n_out)

    # leadfield mixing for all trials
    # leadfield: (C,S), src_ds: (T,S,N) -> eeg_trials: (T,C,N)
    eeg_trials = np.einsum("cs,tsn->tcn", leadfield.astype(np.float64), src_ds.astype(np.float64))
    eeg_trials *= float(uV_scale)

    # add sensor noise per trial
    eeg_trials += rng.normal(0.0, float(sensor_noise_std), size=eeg_trials.shape)

    # average across trials -> (C,N)
    eeg_avg = eeg_trials.mean(axis=0).astype(np.float64)

    # baseline correction
    if baseline_correct:
        if baseline_window is None:
            end = int(max(1, np.round(stim_onset * fs)))
            bl = slice(0, min(end, n_out))
        else:
            t0, t1 = baseline_window
            start = int(np.round((stim_onset + t0) * fs))
            end = int(np.round((stim_onset + t1) * fs))
            start = max(0, min(start, n_out))
            end = max(0, min(end, n_out))
            bl = slice(start, max(start + 1, end))
        eeg_avg -= eeg_avg[:, bl].mean(axis=1, keepdims=True)

    # bandpass
    if bandpass is not None:
        lo, hi = float(bandpass[0]), float(bandpass[1])
        nyq = 0.5 * fs
        lo_n = max(1e-6, lo / nyq)
        hi_n = min(0.999, hi / nyq)
        if lo_n < hi_n:
            b, a = butter(4, [lo_n, hi_n], btype="band")
            eeg_avg = filtfilt(b, a, eeg_avg, axis=1)

    eeg_avg = eeg_avg.astype(np.float32)

    if not return_trials and not return_sources:
        return eeg_avg

    out: Dict[str, np.ndarray] = {"eeg": eeg_avg}
    if return_trials:
        out["eeg_trials"] = eeg_trials.astype(np.float32)  # (trials, channels, time)
    if return_sources:
        out["sources"] = src_ds.astype(np.float32)         # (trials, sources, time)
    return out
