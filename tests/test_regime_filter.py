"""Tests for sim/regime_filter.py — ERP regime quality control."""
from __future__ import annotations

import numpy as np
import pytest

from sim.regime_filter import RegimeFilterConfig, regime_reject

FS = 250
DUR = 2.0
STIM_ONSET = 0.5
N_CH = 16
N_T = int(FS * DUR)


def _make_cfg(**kwargs) -> RegimeFilterConfig:
    return RegimeFilterConfig(fs=FS, duration=DUR, stim_onset=STIM_ONSET, **kwargs)


def _make_erp_signal(seed: int = 0) -> np.ndarray:
    """Synthetic ERP-like signal that should pass the filter."""
    rng = np.random.default_rng(seed)
    eeg = rng.normal(0.0, 5.0, size=(N_CH, N_T)).astype(np.float32)
    stim_idx = int(STIM_ONSET * FS)
    # Sharp post-stim peak well above baseline GFP — passes evoked-peak check
    eeg[:, stim_idx : stim_idx + 30] += 80.0
    return eeg


# ---------------------------------------------------------------------------
# Passing case
# ---------------------------------------------------------------------------

class TestRegimeRejectPasses:
    def test_erp_like_signal_accepted(self):
        eeg = _make_erp_signal()
        ok, reason = regime_reject(eeg, _make_cfg())
        assert ok, f"Expected 'ok', got reason='{reason}'"
        assert reason == "ok"


# ---------------------------------------------------------------------------
# Rejection cases
# ---------------------------------------------------------------------------

class TestRegimeRejectRejections:
    def test_nan_rejected(self):
        eeg = _make_erp_signal()
        eeg[0, 10] = np.nan
        ok, reason = regime_reject(eeg, _make_cfg())
        assert not ok
        assert reason == "nan_or_inf"

    def test_inf_rejected(self):
        eeg = _make_erp_signal()
        eeg[0, 10] = np.inf
        ok, reason = regime_reject(eeg, _make_cfg())
        assert not ok
        assert reason == "nan_or_inf"

    def test_too_flat_rejected(self):
        eeg = np.zeros((N_CH, N_T), dtype=np.float32)
        ok, reason = regime_reject(eeg, _make_cfg())
        assert not ok
        assert reason == "too_flat"

    def test_too_large_rejected(self):
        eeg = np.full((N_CH, N_T), 2000.0, dtype=np.float32)
        ok, reason = regime_reject(eeg, _make_cfg())
        assert not ok
        assert reason == "too_large"

    def test_no_evoked_peak_rejected(self):
        """Random low-amplitude noise with no clear post-stim peak is rejected."""
        rng = np.random.default_rng(7)
        # Very small amplitude: 99th-pct > abs99_min but GFP peak z-score < 5
        eeg = rng.normal(0.0, 3.0, size=(N_CH, N_T)).astype(np.float32)
        ok, reason = regime_reject(eeg, _make_cfg())
        assert not ok
        assert reason == "no_evoked_peak"

    def test_bad_shape_rejected(self):
        """1-D input should be rejected with 'bad_shape'."""
        eeg = np.zeros(N_T, dtype=np.float32)
        ok, reason = regime_reject(eeg, _make_cfg())
        assert not ok
        assert reason == "bad_shape"

    def test_sustained_activity_rejected(self):
        """Signal with sustained late-window activity is rejected."""
        rng = np.random.default_rng(3)
        eeg = rng.normal(0.0, 5.0, size=(N_CH, N_T)).astype(np.float32)
        stim_idx = int(STIM_ONSET * FS)
        # Peak at stim onset — passes evoked-peak check
        eeg[:, stim_idx : stim_idx + 20] += 80.0
        # Large sustained signal in late window — should trigger sustained-activity rejection
        eeg[:, N_T // 2 :] += 200.0
        ok, reason = regime_reject(eeg, _make_cfg())
        assert not ok
        assert reason == "sustained_activity"


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

class TestRegimeFilterConfig:
    def test_default_thresholds(self):
        cfg = RegimeFilterConfig(fs=250, duration=2.0, stim_onset=0.5)
        assert cfg.abs99_min_uV == pytest.approx(1.0)
        assert cfg.abs99_max_uV == pytest.approx(1500.0)
        assert cfg.gfp_peak_z_min == pytest.approx(5.0)

    def test_custom_thresholds(self):
        cfg = RegimeFilterConfig(
            fs=250, duration=2.0, stim_onset=0.5, abs99_min_uV=2.0, abs99_max_uV=500.0
        )
        assert cfg.abs99_min_uV == pytest.approx(2.0)
        assert cfg.abs99_max_uV == pytest.approx(500.0)
