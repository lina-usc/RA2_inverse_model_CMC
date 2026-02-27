"""Tests for data/feature_tokens.py — ERP/TFR token extraction."""
from __future__ import annotations

import numpy as np
import pytest

from data.feature_tokens import (
    TokenConfig,
    _bin_mask_or_nearest,
    compute_erp_tokens,
    compute_tfr_tokens,
)


@pytest.fixture
def cfg() -> TokenConfig:
    return TokenConfig()


def _make_eeg(cfg: TokenConfig, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_t = int(cfg.duration * cfg.fs)
    return rng.normal(size=(cfg.n_channels, n_t)).astype(np.float32)


# ---------------------------------------------------------------------------
# compute_erp_tokens
# ---------------------------------------------------------------------------

class TestComputeErpTokens:
    def test_output_shape(self, cfg):
        tokens = compute_erp_tokens(_make_eeg(cfg), cfg)
        assert tokens.shape == (cfg.n_time_patches, cfg.n_channels)

    def test_output_dtype(self, cfg):
        tokens = compute_erp_tokens(_make_eeg(cfg), cfg)
        assert tokens.dtype == np.float32

    def test_wrong_channels_raises(self, cfg):
        eeg = np.zeros((8, int(cfg.duration * cfg.fs)), dtype=np.float32)
        with pytest.raises(ValueError, match="n_channels"):
            compute_erp_tokens(eeg, cfg)

    def test_wrong_time_raises(self, cfg):
        eeg = np.zeros((cfg.n_channels, 100), dtype=np.float32)
        with pytest.raises(ValueError, match="duration"):
            compute_erp_tokens(eeg, cfg)

    def test_constant_signal_gives_constant_tokens(self, cfg):
        eeg = np.full(
            (cfg.n_channels, int(cfg.duration * cfg.fs)), 3.14, dtype=np.float32
        )
        tokens = compute_erp_tokens(eeg, cfg)
        np.testing.assert_allclose(tokens, 3.14, atol=1e-5)

    def test_first_token_matches_manual_mean(self, cfg):
        eeg = _make_eeg(cfg)
        tokens = compute_erp_tokens(eeg, cfg)
        w = int(cfg.duration * cfg.fs) // cfg.n_time_patches
        expected = eeg[:, :w].mean(axis=1)
        np.testing.assert_allclose(tokens[0], expected, rtol=1e-5)

    def test_no_nan(self, cfg):
        tokens = compute_erp_tokens(_make_eeg(cfg), cfg)
        assert np.isfinite(tokens).all()


# ---------------------------------------------------------------------------
# compute_tfr_tokens
# ---------------------------------------------------------------------------

class TestComputeTfrTokens:
    def test_output_shape(self, cfg):
        tokens, _ = compute_tfr_tokens(_make_eeg(cfg), cfg)
        assert tokens.shape == (cfg.n_time_patches * cfg.n_freq_patches, cfg.n_channels)

    def test_output_dtype(self, cfg):
        tokens, _ = compute_tfr_tokens(_make_eeg(cfg), cfg)
        assert tokens.dtype == np.float32

    def test_no_nan(self, cfg):
        tokens, _ = compute_tfr_tokens(_make_eeg(cfg), cfg)
        assert np.isfinite(tokens).all()

    def test_meta_keys_present(self, cfg):
        _, meta = compute_tfr_tokens(_make_eeg(cfg), cfg)
        for key in ("stft_f", "stft_t", "time_edges", "freq_edges", "tfr_patch_shape"):
            assert key in meta, f"Missing meta key: {key}"

    def test_meta_patch_shape(self, cfg):
        _, meta = compute_tfr_tokens(_make_eeg(cfg), cfg)
        expected = np.array(
            [cfg.n_time_patches, cfg.n_freq_patches, cfg.n_channels], dtype=np.int32
        )
        np.testing.assert_array_equal(meta["tfr_patch_shape"], expected)

    def test_freq_edges_within_range(self, cfg):
        _, meta = compute_tfr_tokens(_make_eeg(cfg), cfg)
        edges = meta["freq_edges"]
        assert float(edges[0]) >= cfg.f_min * 0.99
        assert float(edges[-1]) <= cfg.f_max * 1.01

    def test_time_edges_span_duration(self, cfg):
        _, meta = compute_tfr_tokens(_make_eeg(cfg), cfg)
        edges = meta["time_edges"]
        assert float(edges[0]) == pytest.approx(0.0, abs=1e-4)
        assert float(edges[-1]) == pytest.approx(cfg.duration, abs=1e-4)

    def test_zero_signal_no_crash(self, cfg):
        eeg = np.zeros((cfg.n_channels, int(cfg.duration * cfg.fs)), dtype=np.float32)
        tokens, _ = compute_tfr_tokens(eeg, cfg)
        assert tokens.shape == (cfg.n_time_patches * cfg.n_freq_patches, cfg.n_channels)


# ---------------------------------------------------------------------------
# _bin_mask_or_nearest
# ---------------------------------------------------------------------------

class TestBinMaskOrNearest:
    def test_in_range_selects_correct_elements(self):
        x = np.array([0.5, 1.5, 2.5, 3.5])
        m = _bin_mask_or_nearest(x, 1.0, 3.0)
        assert m[1] and m[2]
        assert not m[0] and not m[3]

    def test_empty_bin_falls_back_to_nearest(self):
        x = np.array([0.0, 1.0, 2.0])
        # bin [10, 11] contains no elements; midpoint 10.5, nearest is x[2]=2.0
        m = _bin_mask_or_nearest(x, 10.0, 11.0)
        assert m.sum() == 1
        assert m[2]

    def test_returns_bool_array(self):
        x = np.array([1.0, 2.0, 3.0])
        m = _bin_mask_or_nearest(x, 1.5, 2.5)
        assert m.dtype == bool

    def test_exactly_one_element_in_range(self):
        x = np.linspace(0, 1, 10)
        m = _bin_mask_or_nearest(x, 0.45, 0.55)
        assert m.sum() >= 1
