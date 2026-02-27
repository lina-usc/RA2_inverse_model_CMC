"""Tests for data/priors.py — prior specification and sampling."""
from __future__ import annotations

import json

import numpy as np
import pytest

from data.priors import (
    build_prior_spec,
    sample_theta,
    theta_to_params,
    trunc_gamma,
    trunc_normal,
)

# ---------------------------------------------------------------------------
# trunc_gamma
# ---------------------------------------------------------------------------

class TestTruncGamma:
    def test_within_bounds(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            x = trunc_gamma(rng, shape=4.0, scale=0.25, low=0.1, high=1.0)
            assert 0.1 <= x <= 1.0

    def test_raises_on_impossible_bounds(self):
        rng = np.random.default_rng(0)
        with pytest.raises(RuntimeError, match="trunc_gamma"):
            trunc_gamma(rng, shape=1.0, scale=0.001, low=100.0, high=101.0)


# ---------------------------------------------------------------------------
# trunc_normal
# ---------------------------------------------------------------------------

class TestTruncNormal:
    def test_within_bounds(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            x = trunc_normal(rng, mean=-1.4, sd=0.35, low=-3.0, high=-0.2)
            assert -3.0 <= x <= -0.2

    def test_raises_on_impossible_bounds(self):
        rng = np.random.default_rng(0)
        with pytest.raises(RuntimeError, match="trunc_normal"):
            trunc_normal(rng, mean=0.0, sd=0.001, low=100.0, high=101.0)


# ---------------------------------------------------------------------------
# build_prior_spec
# ---------------------------------------------------------------------------

class TestBuildPriorSpec:
    def setup_method(self):
        self.names, self.low, self.high, self.dist, self.params, self.spec_json = (
            build_prior_spec()
        )

    def test_n_params(self):
        assert len(self.names) == 9

    def test_bounds_shape(self):
        assert self.low.shape == (9,)
        assert self.high.shape == (9,)

    def test_bounds_ordering(self):
        assert np.all(self.low < self.high)

    def test_dist_types(self):
        for d in self.dist:
            assert d in {"trunc_gamma", "trunc_normal"}

    def test_params_shape(self):
        assert self.params.shape == (9, 2)

    def test_spec_json_valid(self):
        obj = json.loads(self.spec_json)
        assert len(obj) == 9
        for entry in obj:
            assert "name" in entry and "dist" in entry and "low" in entry and "high" in entry

    def test_w_ie_is_negative_range(self):
        """w_ie is inhibitory — both bounds should be negative."""
        idx = self.names.index("w_ie")
        assert self.high[idx] < 0.0

    def test_expected_param_names(self):
        expected = {"tau_e", "tau_i", "g", "p0", "stim_amp", "w_ei", "w_ie", "w_ff", "w_fb"}
        assert set(self.names) == expected


# ---------------------------------------------------------------------------
# sample_theta
# ---------------------------------------------------------------------------

class TestSampleTheta:
    def setup_method(self):
        spec = build_prior_spec()
        self.names, self.low, self.high, self.dist, self.params = spec[:5]

    def test_shape(self):
        rng = np.random.default_rng(42)
        theta, _ = sample_theta(rng, self.names, self.low, self.high, self.dist, self.params)
        assert theta.shape == (9,)

    def test_within_bounds(self):
        rng = np.random.default_rng(42)
        theta, _ = sample_theta(rng, self.names, self.low, self.high, self.dist, self.params)
        assert np.all(theta >= self.low)
        assert np.all(theta <= self.high)

    def test_dtype(self):
        rng = np.random.default_rng(42)
        theta, _ = sample_theta(rng, self.names, self.low, self.high, self.dist, self.params)
        assert theta.dtype == np.float32

    def test_fixed_constants_in_params_dict(self):
        rng = np.random.default_rng(42)
        _, p = sample_theta(rng, self.names, self.low, self.high, self.dist, self.params)
        assert p["w_ee"] == pytest.approx(1.2)
        assert p["w_ii"] == pytest.approx(-0.6)
        assert p["w_sd"] == pytest.approx(0.5)

    def test_params_dict_has_all_keys(self):
        rng = np.random.default_rng(42)
        _, p = sample_theta(rng, self.names, self.low, self.high, self.dist, self.params)
        # 9 sampled + 3 fixed constants
        assert len(p) == 12

    def test_different_seeds_produce_different_samples(self):
        args = (self.names, self.low, self.high, self.dist, self.params)
        theta1, _ = sample_theta(np.random.default_rng(0), *args)
        theta2, _ = sample_theta(np.random.default_rng(1), *args)
        assert not np.allclose(theta1, theta2)


# ---------------------------------------------------------------------------
# theta_to_params
# ---------------------------------------------------------------------------

class TestThetaToParams:
    def setup_method(self):
        self.names = build_prior_spec()[0]

    def test_basic_mapping(self):
        theta = np.ones(9, dtype=np.float32)
        p = theta_to_params(theta, self.names)
        for nm in self.names:
            assert nm in p

    def test_fixed_constants(self):
        theta = np.ones(9, dtype=np.float32)
        p = theta_to_params(theta, self.names)
        assert p["w_ee"] == pytest.approx(1.2)
        assert p["w_ii"] == pytest.approx(-0.6)
        assert p["w_sd"] == pytest.approx(0.5)

    def test_total_keys(self):
        theta = np.ones(9, dtype=np.float32)
        p = theta_to_params(theta, self.names)
        assert len(p) == 12  # 9 sampled + 3 fixed

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            theta_to_params(np.ones(5, dtype=np.float32), self.names)
