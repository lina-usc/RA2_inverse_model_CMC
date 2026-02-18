#data/priors.py

from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np


def trunc_gamma(rng: np.random.Generator, shape: float, scale: float, low: float, high: float) -> float:
    """
    Truncated Gamma(shape, scale) on [low, high].
    """
    for _ in range(50_000):
        x = float(rng.gamma(shape=shape, scale=scale))
        if low <= x <= high:
            return x
    raise RuntimeError("Failed to sample trunc_gamma within bounds.")


def trunc_normal(rng: np.random.Generator, mean: float, sd: float, low: float, high: float) -> float:
    """
    Truncated Normal(mean, sd) on [low, high].
    """
    for _ in range(50_000):
        x = float(rng.normal(loc=mean, scale=sd))
        if low <= x <= high:
            return x
    raise RuntimeError("Failed to sample trunc_normal within bounds.")


def build_prior_spec() -> Tuple[List[str], np.ndarray, np.ndarray, List[str], np.ndarray, str]:
    """
    Returns:
      names: list[str] length P
      low/high: (P,) float32 bounds (also used for z-transform later)
      dist: list[str] each in {"trunc_gamma","trunc_normal"}
      params: (P,2) float32; for gamma [shape, scale], for normal [mean, sd]
      spec_json: JSON string for paper tables
    """
    # Parameter vector θ (9 params, biologically meaningful & stable)
    names = ["tau_e", "tau_i", "g", "p0", "stim_amp", "w_ei", "w_ie", "w_ff", "w_fb"]

    # Hard bounds (also define the z-transform support)
    low = np.array([0.005, 0.003, 0.50, 0.05, 0.10, 0.20, -3.00, 0.10, 0.10], dtype=np.float32)
    high = np.array([0.050, 0.030, 2.00, 2.00, 4.00, 3.00, -0.20, 2.50, 2.00], dtype=np.float32)

    # Use Gamma for strictly-positive parameters; Normal for signed inhibitory coupling.
    dist = [
        "trunc_gamma",   # tau_e
        "trunc_gamma",   # tau_i
        "trunc_gamma",   # g
        "trunc_gamma",   # p0
        "trunc_gamma",   # stim_amp
        "trunc_gamma",   # w_ei
        "trunc_normal",  # w_ie (negative)
        "trunc_gamma",   # w_ff
        "trunc_gamma",   # w_fb
    ]

    # Gamma parameters: [shape, scale] => mean = shape*scale
    params = np.array([
        [8.0, 0.0025],    # tau_e mean=0.020
        [8.0, 0.00125],   # tau_i mean=0.010
        [9.0, 0.1111111], # g mean=1.0
        [4.0, 0.125],     # p0 mean=0.5
        [4.0, 0.25],      # stim_amp mean=1.0
        [4.0, 0.25],      # w_ei mean=1.0
        [-1.40, 0.35],    # w_ie normal(mean=-1.4, sd=0.35)
        [4.0, 0.20],      # w_ff mean=0.8
        [4.0, 0.15],      # w_fb mean=0.6
    ], dtype=np.float32)

    spec = []
    for i, nm in enumerate(names):
        spec.append({
            "name": nm,
            "dist": dist[i],
            "params": [float(params[i, 0]), float(params[i, 1])],
            "low": float(low[i]),
            "high": float(high[i]),
            "param_convention": ("[shape, scale]" if dist[i] == "trunc_gamma" else "[mean, sd]"),
        })
    spec_json = json.dumps(spec, indent=2)

    return names, low, high, dist, params, spec_json


def sample_theta(
    rng: np.random.Generator,
    names: List[str],
    low: np.ndarray,
    high: np.ndarray,
    dist: List[str],
    params: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Sample theta (P,) and return simulator params dict including fixed constants.
    """
    theta = np.zeros((len(names),), dtype=np.float32)
    p: Dict[str, float] = {}

    for i, nm in enumerate(names):
        lo = float(low[i])
        hi = float(high[i])

        if dist[i] == "trunc_gamma":
            shape = float(params[i, 0])
            scale = float(params[i, 1])
            x = trunc_gamma(rng, shape, scale, lo, hi)

        elif dist[i] == "trunc_normal":
            mean = float(params[i, 0])
            sd = float(params[i, 1])
            x = trunc_normal(rng, mean, sd, lo, hi)

        else:
            raise ValueError(f"Unknown dist: {dist[i]}")

        theta[i] = np.float32(x)
        p[nm] = float(x)

    # Fixed constants (paper-stable)
    p["w_ee"] = 1.2
    p["w_ii"] = -0.6
    p["w_sd"] = 0.5

    return theta, p


def theta_to_params(theta: np.ndarray, param_names: List[str]) -> Dict[str, float]:
    """
    Convert theta vector to simulator params dict, adding fixed constants.
    """
    theta = np.asarray(theta, dtype=np.float32)
    if theta.shape != (len(param_names),):
        raise ValueError("theta_to_params: wrong theta shape")

    p = {nm: float(theta[i]) for i, nm in enumerate(param_names)}
    p["w_ee"] = 1.2
    p["w_ii"] = -0.6
    p["w_sd"] = 0.5
    return p
