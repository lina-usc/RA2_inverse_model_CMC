from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def gaussian_bump(
    t: np.ndarray,
    onset: float = 0.5,
    sigma: float = 0.05,
    amp: float = 1.0,
    *,
    causal: bool = False,
    t0: Optional[float] = None,
) -> np.ndarray:
    """Gaussian (optionally causal) stimulus bump.

    This keeps backward compatibility with the original codebase, which calls:

        gaussian_bump(t, onset=..., sigma=..., amp=...)

    while also allowing an alias keyword ``t0`` (used in some edits/refactors).

    Parameters
    ----------
    t : array-like
        Time axis in seconds.
    onset : float
        Stimulus onset time in seconds (center of the Gaussian).
    sigma : float
        Standard deviation of the Gaussian in seconds.
    amp : float
        Amplitude (arbitrary units; interpreted as simulator input-drive scale).
    causal : bool
        If True, truncate the Gaussian so that stim(t)=0 for t < onset.
        This is important when using a pre-stimulus baseline window [0, onset).
    t0 : float or None
        Alias for onset (if provided, overrides onset).

    Returns
    -------
    stim : np.ndarray
        Stimulus waveform, same shape as ``t``.
    """
    t = np.asarray(t, dtype=np.float64)

    if t0 is not None:
        onset = float(t0)

    sigma = float(sigma)
    onset = float(onset)
    amp = float(amp)

    stim = amp * np.exp(-0.5 * ((t - onset) / (sigma + 1e-12)) ** 2)

    if causal:
        stim = np.where(t >= onset, stim, 0.0)

    return stim


@dataclass(frozen=True)
class GaussianStimulus:
    """Paper-ready Gaussian stimulus spec."""
    onset: float = 0.5
    sigma: float = 0.05
    amp: float = 1.0
    causal: bool = False

    def __call__(self, t: np.ndarray) -> np.ndarray:
        return gaussian_bump(
            t,
            onset=self.onset,
            sigma=self.sigma,
            amp=self.amp,
            causal=self.causal,
        )
