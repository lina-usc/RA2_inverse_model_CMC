from __future__ import annotations

import numpy as np


def theta_to_z(theta: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """
    Map bounded theta in [low, high] to unconstrained z via logit transform:
        x = (theta - low)/(high-low)
        z = log(x/(1-x))
    """
    theta = np.asarray(theta, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)

    denom = np.clip(high - low, 1e-6, np.inf)
    x = (theta - low) / denom
    x = np.clip(x, 1e-6, 1.0 - 1e-6)
    z = np.log(x) - np.log(1.0 - x)
    return z.astype(np.float32)


def z_to_theta(z: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """
    Inverse of theta_to_z:
        x = sigmoid(z)
        theta = low + x*(high-low)
    """
    z = np.asarray(z, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)

    x = 1.0 / (1.0 + np.exp(-z))
    theta = low + x * (high - low)
    return theta.astype(np.float32)
