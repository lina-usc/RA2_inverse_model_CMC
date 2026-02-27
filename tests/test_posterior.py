"""Tests for models/posterior_fullcov.py — covariance parameterisation and NLL."""
from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from models.posterior_fullcov import _raw_to_tril, mvn_tril_nll, raw_tril_size  # noqa: E402


# ---------------------------------------------------------------------------
# raw_tril_size
# ---------------------------------------------------------------------------

class TestRawTrilSize:
    @pytest.mark.parametrize(
        "P, expected",
        [(1, 1), (2, 3), (3, 6), (5, 15), (9, 45)],
    )
    def test_formula(self, P, expected):
        assert raw_tril_size(P) == expected


# ---------------------------------------------------------------------------
# _raw_to_tril
# ---------------------------------------------------------------------------

class TestRawToTril:
    def test_output_shape(self):
        B, P = 4, 3
        raw = tf.zeros((B, raw_tril_size(P)), dtype=tf.float32)
        L = _raw_to_tril(raw, P)
        assert L.shape == (B, P, P)

    def test_upper_triangle_is_zero(self):
        B, P = 3, 5
        rng = np.random.default_rng(0)
        raw = tf.constant(rng.normal(size=(B, raw_tril_size(P))).astype(np.float32))
        L = _raw_to_tril(raw, P).numpy()
        for b in range(B):
            upper = np.triu(L[b], k=1)
            np.testing.assert_array_equal(upper, 0.0, err_msg=f"batch {b} has non-zero upper triangle")

    def test_diagonal_is_positive(self):
        B, P = 4, 4
        rng = np.random.default_rng(1)
        raw = tf.constant(rng.normal(size=(B, raw_tril_size(P))).astype(np.float32))
        L = _raw_to_tril(raw, P).numpy()
        for b in range(B):
            assert np.all(np.diag(L[b]) > 0), f"Non-positive diagonal in batch {b}"

    def test_diagonal_above_min_diag(self):
        """Diagonal should always exceed min_diag (softplus + min_diag)."""
        min_diag = 1e-3
        B, P = 2, 9
        # Use very negative raw values so softplus ≈ 0
        raw = tf.constant(np.full((B, raw_tril_size(P)), -50.0, dtype=np.float32))
        L = _raw_to_tril(raw, P, min_diag=min_diag).numpy()
        for b in range(B):
            assert np.all(np.diag(L[b]) > min_diag * 0.99)

    def test_output_dtype(self):
        B, P = 2, 3
        raw = tf.zeros((B, raw_tril_size(P)), dtype=tf.float32)
        L = _raw_to_tril(raw, P)
        assert L.dtype == tf.float32


# ---------------------------------------------------------------------------
# mvn_tril_nll
# ---------------------------------------------------------------------------

class TestMvnTrilNll:
    def _random_inputs(self, B: int, P: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        y = tf.constant(rng.normal(size=(B, P)).astype(np.float32))
        mu = tf.constant(rng.normal(size=(B, P)).astype(np.float32))
        raw = tf.constant(rng.normal(size=(B, raw_tril_size(P))).astype(np.float32))
        return y, mu, raw

    def test_output_shape(self):
        B, P = 8, 9
        y, mu, raw = self._random_inputs(B, P)
        nll = mvn_tril_nll(y, mu, raw)
        assert nll.shape == (B,)

    def test_output_is_finite(self):
        B, P = 8, 9
        y, mu, raw = self._random_inputs(B, P)
        nll = mvn_tril_nll(y, mu, raw)
        assert tf.reduce_all(tf.math.is_finite(nll))

    def test_nll_with_const_is_non_negative(self):
        """Full NLL (with log(2π) constant) must be non-negative."""
        B, P = 4, 3
        y, mu, raw = self._random_inputs(B, P)
        nll = mvn_tril_nll(y, mu, raw, include_const=True)
        assert np.all(nll.numpy() >= 0.0)

    def test_nll_increases_with_distance(self):
        """NLL should be larger when y is far from mu than when it is close."""
        B, P = 8, 9
        rng = np.random.default_rng(5)
        raw = tf.constant(rng.normal(size=(B, raw_tril_size(P))).astype(np.float32))
        mu = tf.zeros((B, P), dtype=tf.float32)

        y_close = tf.constant(rng.normal(0.0, 0.01, size=(B, P)).astype(np.float32))
        y_far = tf.constant(rng.normal(0.0, 100.0, size=(B, P)).astype(np.float32))

        nll_close = mvn_tril_nll(y_close, mu, raw).numpy().mean()
        nll_far = mvn_tril_nll(y_far, mu, raw).numpy().mean()
        assert nll_far > nll_close

    def test_nll_at_mode_lower_than_off_mode(self):
        """NLL should be lower when y == mu than when y != mu (same L)."""
        B, P = 4, 9
        rng = np.random.default_rng(2)
        mu = tf.constant(rng.normal(size=(B, P)).astype(np.float32))
        raw = tf.constant(rng.normal(size=(B, raw_tril_size(P))).astype(np.float32))

        nll_at_mode = mvn_tril_nll(mu, mu, raw).numpy()
        y_off = mu + tf.constant(rng.normal(0.0, 5.0, size=(B, P)).astype(np.float32))
        nll_off = mvn_tril_nll(y_off, mu, raw).numpy()

        assert np.all(nll_at_mode <= nll_off)
