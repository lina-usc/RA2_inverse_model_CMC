from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

from sim.stimulus import gaussian_bump

LOGGER = logging.getLogger(__name__)


def _sigmoid(x: np.ndarray, gain: float = 1.0, bias: float = 0.0) -> np.ndarray:
    """Smooth firing-rate nonlinearity."""
    x = np.asarray(x, dtype=np.float64)
    z = gain * (x - bias)
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def _build_connectivity(params: Dict[str, float]) -> np.ndarray:
    """
    Build 6x6 connectivity matrix W for the canonical microcircuit motifs.
    """
    W = np.zeros((6, 6), dtype=np.float64)

    # Within-layer motifs (E->I, I->E, I->I, E->E)
    w_ee = float(params.get("w_ee", 1.2))
    w_ei = float(params.get("w_ei", 1.0))    # E->I (positive)
    w_ie = float(params.get("w_ie", -1.4))   # I->E (negative)
    w_ii = float(params.get("w_ii", -0.6))   # I->I (negative)

    for (e, i) in [(0, 1), (2, 3), (4, 5)]:
        W[e, e] += w_ee
        W[i, e] += w_ei
        W[e, i] += w_ie
        W[i, i] += w_ii

    # Inter-layer canonical couplings (feedforward + feedback)
    w_ff = float(params.get("w_ff", 0.8))   # granular -> superficial
    w_fb = float(params.get("w_fb", 0.6))   # deep -> superficial
    w_sd = float(params.get("w_sd", 0.5))   # superficial -> deep

    # Feedforward: E_gran -> E_sup and I_sup
    W[0, 2] += w_ff
    W[1, 2] += 0.5 * w_ff

    # Feedback: E_deep -> E_sup and I_sup
    W[0, 4] += w_fb
    W[1, 4] += 0.3 * w_fb

    # Superficial -> deep
    W[4, 0] += w_sd
    W[5, 0] += 0.3 * w_sd

    return W


def simulate_sources_batch(
    params: Dict[str, float],
    internal_fs: int,
    duration: float,
    stim_onset: float,
    stim_sigma: float,
    input_noise_std: float,
    n_sims: int,
    seed: Optional[int],
    stim_causal: bool = False,
) -> np.ndarray:
    """
    Vectorized CMC simulation for many independent realizations in one time loop.

    Each realization has:
      - independent initial state
      - independent input noise

    Returns
    -------
    out : (n_sims, n_int) float64
        LFP-like source signal for each realization at internal_fs.
    """
    rng = np.random.default_rng(seed)

    internal_fs = int(internal_fs)
    dt = 1.0 / float(internal_fs)
    n_int = int(np.round(duration * internal_fs))
    t = np.arange(n_int, dtype=np.float64) * dt

    # External drive: baseline + explicit Gaussian bump + independent noise
    p0 = float(params.get("p0", 0.5))
    stim_amp = float(params.get("stim_amp", 1.0))
    stim = gaussian_bump(t, onset=stim_onset, sigma=stim_sigma, amp=stim_amp, causal=stim_causal)  # (n_int,)

    u = p0 + stim[None, :] + rng.normal(0.0, float(input_noise_std), size=(int(n_sims), n_int))

    # Time constants
    tau_e = float(params.get("tau_e", 0.02))  # seconds
    tau_i = float(params.get("tau_i", 0.01))

    # Global gain scaling
    g = float(params.get("g", 1.0))

    # Connectivity
    W = _build_connectivity(params)  # (6,6)

    # Input targets granular excit primarily
    inp_to = np.array([0.0, 0.0, 1.0, 0.2, 0.1, 0.0], dtype=np.float64)  # (6,)

    # State: membrane-like potentials, batched
    v = (0.01 * rng.normal(size=(int(n_sims), 6))).astype(np.float64)

    out = np.zeros((int(n_sims), n_int), dtype=np.float64)

    # One time loop only (fast); all realizations are vectorized
    for k in range(n_int):
        r = _sigmoid(v, gain=2.0, bias=0.0)  # (n_sims,6)

        # W @ r for each sim => (n_sims,6)
        Wr = r @ W.T

        # input drive to each population
        drive = inp_to[None, :] * u[:, k][:, None]  # (n_sims,6)

        dv = np.zeros_like(v)

        # E pops: 0,2,4
        dv[:, 0] = (-v[:, 0] + g * Wr[:, 0] + drive[:, 0]) / tau_e
        dv[:, 2] = (-v[:, 2] + g * Wr[:, 2] + drive[:, 2]) / tau_e
        dv[:, 4] = (-v[:, 4] + g * Wr[:, 4] + drive[:, 4]) / tau_e

        # I pops: 1,3,5
        dv[:, 1] = (-v[:, 1] + g * Wr[:, 1] + drive[:, 1]) / tau_i
        dv[:, 3] = (-v[:, 3] + g * Wr[:, 3] + drive[:, 3]) / tau_i
        dv[:, 5] = (-v[:, 5] + g * Wr[:, 5] + drive[:, 5]) / tau_i

        v += dt * dv

        # LFP-like readout per sim
        out[:, k] = (v[:, 0] + 0.8 * v[:, 4] + 0.5 * v[:, 2]) - (0.6 * v[:, 1] + 0.4 * v[:, 5] + 0.3 * v[:, 3])

    return out


def simulate_sources(
    params: Dict[str, float],
    internal_fs: int,
    duration: float,
    stim_onset: float,
    stim_sigma: float,
    input_noise_std: float,
    n_sources: int,
    seed: Optional[int],
) -> np.ndarray:
    """
    Backward-compatible wrapper: simulate n_sources independent sources.
    """
    return simulate_sources_batch(
        params=params,
        internal_fs=internal_fs,
        duration=duration,
        stim_onset=stim_onset,
        stim_sigma=stim_sigma,
        input_noise_std=input_noise_std,
        n_sims=int(n_sources),
        seed=seed,
        stim_causal=stim_causal,
    )
