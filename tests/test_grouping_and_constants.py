"""Tests for grouping utilities and physical constants."""

from __future__ import annotations

import numpy as np

import pytest

from asymmetry.core.data.dataset import Histogram
from asymmetry.core.transform.grouping import (
    apply_grouping,
    apply_grouping_aligned,
    common_t0_for_groups,
    group_forward_backward,
)
from asymmetry.core.utils.constants import (
    GAUSS_TO_TESLA,
    MUON_GYROMAGNETIC_RATIO_MHZ_PER_T,
    MUON_LIFETIME_US,
)


def test_apply_grouping_sums_and_truncates_to_shortest() -> None:
    h0 = Histogram(counts=np.array([1, 2, 3, 4], dtype=float), bin_width=0.01)
    h1 = Histogram(counts=np.array([10, 20, 30], dtype=float), bin_width=0.01)
    h2 = Histogram(counts=np.array([100, 200, 300, 400, 500], dtype=float), bin_width=0.01)

    grouped = apply_grouping([h0, h1, h2], [0, 1, 2])
    np.testing.assert_allclose(grouped, [111.0, 222.0, 333.0])


def test_grouping_helpers_ignore_out_of_range_detectors() -> None:
    # A grouping (e.g. a HAL-9500 preset naming the backward ring) can reference
    # detectors a run does not contain; the summing helpers must not IndexError.
    hists = [
        Histogram(counts=np.array([1, 2, 3], dtype=float), bin_width=0.01, t0_bin=0),
        Histogram(counts=np.array([10, 20, 30], dtype=float), bin_width=0.01, t0_bin=0),
    ]
    # Indices 5/6 don't exist; only index 1 is present.
    np.testing.assert_allclose(apply_grouping(hists, [1, 5, 6]), [10.0, 20.0, 30.0])
    np.testing.assert_allclose(apply_grouping_aligned(hists, [1, 5, 6]), [10.0, 20.0, 30.0])
    # A group with no present detectors yields an empty array rather than crashing.
    assert apply_grouping(hists, [5, 6]).size == 0
    # common_t0 ignores absent indices instead of indexing past the list.
    assert common_t0_for_groups(hists, [0, 9]) == 0


def test_group_forward_backward_errors_when_group_absent_from_run() -> None:
    # Forward/backward groups that name detectors missing from this run should
    # raise a clear ValueError, not crash with IndexError (HAL preset applied to
    # a forward-only run).
    hists = [
        Histogram(counts=np.array([100, 90, 80], dtype=float), bin_width=0.01, t0_bin=0),
        Histogram(counts=np.array([50, 60, 70], dtype=float), bin_width=0.01, t0_bin=0),
    ]
    grouping = {
        "groups": {1: [1], 2: [2], 3: [9]},  # group 3 -> detector index 8 (absent)
        "forward_group": 1,
        "backward_group": 3,
    }
    with pytest.raises(ValueError, match="present in this run"):
        group_forward_backward(hists, grouping)


def test_physical_constants_reasonable_ranges() -> None:
    assert MUON_GYROMAGNETIC_RATIO_MHZ_PER_T > 100
    assert MUON_LIFETIME_US > 2.0
    assert GAUSS_TO_TESLA == 1.0e-4
