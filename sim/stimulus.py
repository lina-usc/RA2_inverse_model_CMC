from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def gaussian_bump(t: np.ndarray, onset: float, sigma: float, amp: float) -> np.ndarray:
    """
    Explicit stimulus definition: Gaussian bump.

    Parameters
    ----------
    t : array, seconds
    onset : seconds
    sigma : seconds
    amp : amplitude (dimensionless)
    """
    t = np.asarray(t, dtype=np.float64)
    sigma = float(sigma)
    onset = float(onset)
    amp = float(amp)
    return amp * np.exp(-0.5 * ((t - onset) / (sigma + 1e-12)) ** 2)


@dataclass(frozen=True)
class GaussianStimulus:
    """
    Paper-ready stimulus spec.
    """
    onset: float
    sigma: float
    amp: float

    def __call__(self, t: np.ndarray) -> np.ndarray:
        return gaussian_bump(t, self.onset, self.sigma, self.amp)
