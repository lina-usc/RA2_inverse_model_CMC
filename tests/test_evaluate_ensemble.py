"""Tests for eval/evaluate_ensemble.py — tri-index ordering correctness."""
from __future__ import annotations

import numpy as np
import pytest

from eval.evaluate_ensemble import _tri_indices


def _lower_tri_pairs(p: int) -> list[tuple[int, int]]:
    """Reference implementation: all (row, col) pairs with row >= col."""
    return [(i, j) for i in range(p) for j in range(i + 1)]


class TestTriIndicesRow:
    """Row-major ordering: left-to-right within each row, top-to-bottom."""

    @pytest.mark.parametrize("p", [1, 2, 3, 4, 9])
    def test_length(self, p):
        rr, cc = _tri_indices(p, "row")
        assert len(rr) == p * (p + 1) // 2

    @pytest.mark.parametrize("p", [1, 2, 3, 4])
    def test_all_lower_triangular(self, p):
        rr, cc = _tri_indices(p, "row")
        assert np.all(rr >= cc), "row ordering: rr must always >= cc"

    def test_row_major_order_p3(self):
        rr, cc = _tri_indices(3, "row")
        expected_rr = np.array([0, 1, 1, 2, 2, 2], dtype=np.int32)
        expected_cc = np.array([0, 0, 1, 0, 1, 2], dtype=np.int32)
        np.testing.assert_array_equal(rr, expected_rr)
        np.testing.assert_array_equal(cc, expected_cc)

    @pytest.mark.parametrize("p", [1, 2, 3, 4, 9])
    def test_dtype(self, p):
        rr, cc = _tri_indices(p, "row")
        assert rr.dtype == np.int32
        assert cc.dtype == np.int32


class TestTriIndicesCol:
    """Column-major ordering: top-to-bottom within each column, left-to-right."""

    @pytest.mark.parametrize("p", [1, 2, 3, 4, 9])
    def test_length(self, p):
        rr, cc = _tri_indices(p, "col")
        assert len(rr) == p * (p + 1) // 2

    @pytest.mark.parametrize("p", [1, 2, 3, 4])
    def test_all_lower_triangular(self, p):
        rr, cc = _tri_indices(p, "col")
        assert np.all(rr >= cc)

    def test_col_major_order_p3(self):
        rr, cc = _tri_indices(3, "col")
        # col 0: (0,0),(1,0),(2,0); col 1: (1,1),(2,1); col 2: (2,2)
        expected_rr = np.array([0, 1, 2, 1, 2, 2], dtype=np.int32)
        expected_cc = np.array([0, 0, 0, 1, 1, 2], dtype=np.int32)
        np.testing.assert_array_equal(rr, expected_rr)
        np.testing.assert_array_equal(cc, expected_cc)

    def test_same_elements_as_row_different_order(self):
        p = 5
        rr_row, cc_row = _tri_indices(p, "row")
        rr_col, cc_col = _tri_indices(p, "col")
        # Same set of pairs, different order
        pairs_row = set(zip(rr_row.tolist(), cc_row.tolist()))
        pairs_col = set(zip(rr_col.tolist(), cc_col.tolist()))
        assert pairs_row == pairs_col


class TestTriIndicesTfp:
    """TFP fill_triangular style: bottom row first, left-to-right within each row."""

    @pytest.mark.parametrize("p", [1, 2, 3, 4, 9])
    def test_length(self, p):
        rr, cc = _tri_indices(p, "tfp")
        assert len(rr) == p * (p + 1) // 2

    @pytest.mark.parametrize("p", [1, 2, 3, 4])
    def test_all_lower_triangular(self, p):
        rr, cc = _tri_indices(p, "tfp")
        assert np.all(rr >= cc)

    def test_tfp_order_p3(self):
        rr, cc = _tri_indices(3, "tfp")
        # bottom row first: (2,0),(2,1),(2,2); then (1,0),(1,1); then (0,0)
        expected_rr = np.array([2, 2, 2, 1, 1, 0], dtype=np.int32)
        expected_cc = np.array([0, 1, 2, 0, 1, 0], dtype=np.int32)
        np.testing.assert_array_equal(rr, expected_rr)
        np.testing.assert_array_equal(cc, expected_cc)

    def test_first_group_is_bottom_row(self):
        p = 4
        rr, cc = _tri_indices(p, "tfp")
        # First p elements should all come from the last row
        assert np.all(rr[:p] == p - 1)

    def test_same_elements_as_row(self):
        p = 5
        rr_row, cc_row = _tri_indices(p, "row")
        rr_tfp, cc_tfp = _tri_indices(p, "tfp")
        pairs_row = set(zip(rr_row.tolist(), cc_row.tolist()))
        pairs_tfp = set(zip(rr_tfp.tolist(), cc_tfp.tolist()))
        assert pairs_row == pairs_tfp


class TestTriIndicesInvalid:
    def test_invalid_ordering_raises(self):
        with pytest.raises(ValueError, match="Unknown ordering"):
            _tri_indices(3, "zigzag")
