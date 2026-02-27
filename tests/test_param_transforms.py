"""Tests for models/param_transforms.py — logit/sigmoid roundtrips."""
from __future__ import annotations

import numpy as np
import pytest

from models.param_transforms import theta_to_z, z_to_theta

LOW = np.array([0.005, 0.003, 0.50, 0.05, 0.10, 0.20, -3.00, 0.10, 0.10], dtype=np.float32)
HIGH = np.array([0.050, 0.030, 2.00, 2.00, 4.00, 3.00, -0.20, 2.50, 2.00], dtype=np.float32)


def _midpoints() -> np.ndarray:
    return ((LOW + HIGH) / 2).astype(np.float32)


def test_roundtrip_midpoint():
    theta = _midpoints()
    z = theta_to_z(theta, LOW, HIGH)
    theta_back = z_to_theta(z, LOW, HIGH)
    np.testing.assert_allclose(theta_back, theta, rtol=1e-4)


def test_roundtrip_batch():
    rng = np.random.default_rng(0)
    theta = np.column_stack([
        rng.uniform(float(LOW[i]), float(HIGH[i]), size=50)
        for i in range(len(LOW))
    ]).astype(np.float32)
    z = theta_to_z(theta, LOW, HIGH)
    theta_back = z_to_theta(z, LOW, HIGH)
    np.testing.assert_allclose(theta_back, theta, rtol=1e-4)


def test_z_to_theta_output_within_bounds():
    rng = np.random.default_rng(1)
    z = rng.normal(size=len(LOW)).astype(np.float32)
    theta = z_to_theta(z, LOW, HIGH)
    assert np.all(theta >= LOW)
    assert np.all(theta <= HIGH)


def test_midpoint_maps_to_zero():
    """theta at the midpoint of each interval should map to z ≈ 0."""
    theta = _midpoints()
    z = theta_to_z(theta, LOW, HIGH)
    np.testing.assert_allclose(z, 0.0, atol=1e-4)


def test_monotone():
    """theta_to_z must be strictly monotone (logit is monotone)."""
    theta_lo = (LOW + 0.01 * (HIGH - LOW)).astype(np.float32)
    theta_hi = (LOW + 0.99 * (HIGH - LOW)).astype(np.float32)
    z_lo = theta_to_z(theta_lo, LOW, HIGH)
    z_hi = theta_to_z(theta_hi, LOW, HIGH)
    assert np.all(z_hi > z_lo)


def test_output_dtype_float32():
    theta = _midpoints()
    z = theta_to_z(theta, LOW, HIGH)
    assert z.dtype == np.float32
    theta_back = z_to_theta(z, LOW, HIGH)
    assert theta_back.dtype == np.float32


def test_scalar_input():
    """Both functions should handle a single scalar (0-d broadcast)."""
    z = theta_to_z(np.float32(0.02), np.float32(0.005), np.float32(0.05))
    assert np.isfinite(z)
    theta = z_to_theta(np.float32(0.0), np.float32(0.005), np.float32(0.05))
    assert float(np.float32(0.005)) <= float(theta) <= float(np.float32(0.05))
