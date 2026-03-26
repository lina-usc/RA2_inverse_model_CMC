"""Tests for sim/cmc_simulator.py — ODE simulator internals."""
from __future__ import annotations

import numpy as np
import pytest

from sim.cmc_simulator import _build_connectivity, _sigmoid, simulate_sources_batch

_DEFAULT_PARAMS = {
    "tau_e": 0.02,
    "tau_i": 0.01,
    "g": 1.0,
    "p0": 0.5,
    "stim_amp": 1.0,
    "w_ei": 1.0,
    "w_ie": -1.4,
    "w_ee": 1.2,
    "w_ii": -0.6,
    "w_ff": 0.8,
    "w_fb": 0.6,
    "w_sd": 0.5,
}

_SIM_KWARGS = dict(
    params=_DEFAULT_PARAMS,
    internal_fs=1000,
    duration=2.0,
    stim_onset=0.5,
    stim_sigma=0.05,
    input_noise_std=0.2,
)


# ---------------------------------------------------------------------------
# _sigmoid
# ---------------------------------------------------------------------------

class TestSigmoid:
    def test_output_in_unit_interval(self):
        x = np.linspace(-500, 500, 1000)
        y = _sigmoid(x)
        assert np.all(y >= 0.0) and np.all(y <= 1.0)

    def test_zero_maps_to_half(self):
        assert _sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)

    def test_monotone_increasing(self):
        x = np.linspace(-10, 10, 200)
        y = _sigmoid(x)
        assert np.all(np.diff(y) >= 0)

    def test_clipping_prevents_overflow(self):
        """Very large / small inputs should not produce NaN."""
        x = np.array([-1e9, 1e9])
        y = _sigmoid(x)
        assert np.isfinite(y).all()


# ---------------------------------------------------------------------------
# _build_connectivity
# ---------------------------------------------------------------------------

class TestBuildConnectivity:
    def test_shape(self):
        W = _build_connectivity(_DEFAULT_PARAMS)
        assert W.shape == (6, 6)

    def test_dtype(self):
        W = _build_connectivity(_DEFAULT_PARAMS)
        assert W.dtype == np.float64

    def test_inhibitory_to_excitatory_negative(self):
        """I→E connections (W[E, I]) should be negative."""
        W = _build_connectivity(_DEFAULT_PARAMS)
        for e, i in [(0, 1), (2, 3), (4, 5)]:
            assert W[e, i] < 0.0, f"W[{e},{i}] should be negative (I→E)"

    def test_inhibitory_to_inhibitory_negative(self):
        """I→I self-connections (W[I, I]) should be negative."""
        W = _build_connectivity(_DEFAULT_PARAMS)
        for _, i in [(0, 1), (2, 3), (4, 5)]:
            assert W[i, i] < 0.0, f"W[{i},{i}] should be negative (I→I)"

    def test_excitatory_to_excitatory_positive(self):
        """E→E connections should carry positive recurrent weight."""
        W = _build_connectivity(_DEFAULT_PARAMS)
        for e, _ in [(0, 1), (2, 3), (4, 5)]:
            assert W[e, e] > 0.0, f"W[{e},{e}] should be positive (E→E)"

    def test_default_params_used_when_key_missing(self):
        """Missing params should fall back to hard-coded defaults without error."""
        W = _build_connectivity({})
        assert W.shape == (6, 6)
        assert np.isfinite(W).all()


# ---------------------------------------------------------------------------
# simulate_sources_batch
# ---------------------------------------------------------------------------

class TestSimulateSourcesBatch:
    def test_output_shape(self):
        n_sims, dur, fs = 4, 2.0, 1000
        out = simulate_sources_batch(**_SIM_KWARGS, n_sims=n_sims, seed=0)
        assert out.shape == (n_sims, int(dur * fs))

    def test_output_is_finite(self):
        out = simulate_sources_batch(**_SIM_KWARGS, n_sims=3, seed=0)
        assert np.isfinite(out).all()

    def test_reproducible_with_same_seed(self):
        out1 = simulate_sources_batch(**_SIM_KWARGS, n_sims=2, seed=42)
        out2 = simulate_sources_batch(**_SIM_KWARGS, n_sims=2, seed=42)
        np.testing.assert_array_equal(out1, out2)

    def test_different_seeds_differ(self):
        out1 = simulate_sources_batch(**_SIM_KWARGS, n_sims=2, seed=0)
        out2 = simulate_sources_batch(**_SIM_KWARGS, n_sims=2, seed=1)
        assert not np.array_equal(out1, out2)

    def test_multiple_sims_differ_from_each_other(self):
        """Independent realizations in the same batch should not be identical."""
        out = simulate_sources_batch(**_SIM_KWARGS, n_sims=4, seed=7)
        for i in range(out.shape[0]):
            for j in range(i + 1, out.shape[0]):
                assert not np.array_equal(out[i], out[j])

    def test_no_seed_does_not_crash(self):
        out = simulate_sources_batch(**_SIM_KWARGS, n_sims=2, seed=None)
        assert out.shape[0] == 2
        assert np.isfinite(out).all()

    def test_stimulus_effect(self):
        """Simulation with stimulus should differ from one with zero amplitude."""
        params_no_stim = {**_DEFAULT_PARAMS, "stim_amp": 0.0}
        out_stim = simulate_sources_batch(**_SIM_KWARGS, n_sims=1, seed=0)
        out_flat = simulate_sources_batch(
            **{**_SIM_KWARGS, "params": params_no_stim}, n_sims=1, seed=0
        )
        assert not np.allclose(out_stim, out_flat, atol=1e-3)
