"""Tests for grouping utilities and physical constants."""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram
from asymmetry.core.transform.grouping import (
    apply_grouping,
    apply_grouping_aligned,
    common_t0_for_groups,
    good_event_count,
    good_frames,
    group_forward_backward,
)
from asymmetry.core.utils.constants import (
    GAUSS_TO_TESLA,
    MUON_GYROMAGNETIC_RATIO_MHZ_PER_T,
    MUON_LIFETIME_US,
)


def test_good_frames_accessor() -> None:
    assert good_frames({"good_frames": 1000.0}) == 1000.0
    # Missing / non-positive / unparseable collapse to the default.
    assert good_frames({}) == 1.0
    assert good_frames({"good_frames": 0.0}) == 1.0
    assert good_frames({"good_frames": -5.0}) == 1.0
    assert good_frames({"good_frames": "x"}) == 1.0
    assert good_frames(None) == 1.0
    # default=0.0 lets callers treat a falsy result as "unknown".
    assert good_frames({}, default=0.0) == 0.0
    assert (good_frames({}, default=0.0) or None) is None
    assert good_frames({"good_frames": 250.0}, default=0.0) == 250.0


def test_good_event_count_sums_good_range_over_fb_groups() -> None:
    # Four detectors; forward = {1,2}, backward = {3,4} (1-based ids).
    hists = [
        Histogram(counts=np.arange(0, 10, dtype=float), bin_width=0.01),  # det 1
        Histogram(counts=np.arange(10, 20, dtype=float), bin_width=0.01),  # det 2
        Histogram(counts=np.arange(20, 30, dtype=float), bin_width=0.01),  # det 3
        Histogram(counts=np.arange(30, 40, dtype=float), bin_width=0.01),  # det 4
    ]
    grouping = {
        "groups": {1: [1, 2], 2: [3, 4]},
        "forward_group": 1,
        "backward_group": 2,
        "first_good_bin": 2,
        "last_good_bin": 5,
    }
    # Sum bins [2..5] inclusive over all four detectors.
    expected = sum(float(np.sum(h.counts[2:6])) for h in hists)
    assert good_event_count(hists, grouping) == pytest.approx(expected)


def test_good_event_count_clamps_range_and_ignores_missing_detectors() -> None:
    hists = [
        Histogram(counts=np.ones(5, dtype=float), bin_width=0.01),  # det 1
        Histogram(counts=np.full(5, 2.0, dtype=float), bin_width=0.01),  # det 2
    ]
    grouping = {
        "groups": {1: [1], 2: [2, 99]},  # det 99 absent from the run
        "forward_group": 1,
        "backward_group": 2,
        "first_good_bin": 3,
        "last_good_bin": 100,  # past the end -> clamped to last bin
    }
    # det1 bins[3..4] = 1+1 = 2; det2 bins[3..4] = 2+2 = 4; det99 skipped.
    assert good_event_count(hists, grouping) == pytest.approx(6.0)


def test_good_event_count_returns_none_when_undetermined() -> None:
    h = [Histogram(counts=np.ones(5, dtype=float), bin_width=0.01)]
    # No grouping / no good-bin range / no named groups -> None (caller falls back).
    assert good_event_count(h, None) is None
    assert good_event_count(h, {}) is None
    assert good_event_count(h, {"first_good_bin": 0, "last_good_bin": 4}) is None
    assert good_event_count([], {"groups": {1: [1]}}) is None


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
