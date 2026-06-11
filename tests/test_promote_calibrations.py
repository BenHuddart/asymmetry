"""Unit tests for the count-fit calibration promote family (F7, F5, N3).

Each promote mirrors the deadtime Send-to-Group pattern: suggest-only (the
grouping is written only when the function is called), a before/after dict for
the GUI, and reference-run provenance keys.
"""

from __future__ import annotations

import pytest

from asymmetry.core.transform.promote import (
    promote_alpha_to_grouping,
    promote_background_to_grouping,
    promote_t0_to_grouping,
)

# --- F7: alpha --------------------------------------------------------------


def test_promote_alpha_writes_keys_and_before_after():
    grouping = {"alpha": 1.0}
    out = promote_alpha_to_grouping(grouping, 1.034, alpha_error=0.012, reference_run=3039)
    assert out["before"] == {"alpha": 1.0}
    assert out["after"] == {"alpha": 1.034}
    assert grouping["alpha"] == pytest.approx(1.034)
    assert grouping["alpha_error"] == pytest.approx(0.012)
    assert grouping["alpha_method"] == "count_fit"
    assert grouping["alpha_reference_run"] == 3039


def test_promote_alpha_defaults_before_to_unity():
    grouping = {}
    out = promote_alpha_to_grouping(grouping, 0.95)
    assert out["before"] == {"alpha": 1.0}
    assert "alpha_error" not in grouping  # error optional
    assert "alpha_reference_run" not in grouping


# --- F5: t0 -----------------------------------------------------------------


def test_promote_t0_rounds_to_nearest_bin_and_discloses_residual():
    # bin width 0.05 µs; a +0.16 µs offset → +3 bins (0.15 µs), residual +0.01 µs.
    grouping = {"t0_bin": 100}
    out = promote_t0_to_grouping(grouping, 0.16, bin_width_us=0.05, group_id=1, reference_run=42)
    assert out["before"] == {"t0_bin": 100}
    assert out["after"] == {"t0_bin": 103}
    assert out["residual_us"] == pytest.approx(0.16 - 3 * 0.05)
    assert out["group_id"] == 1
    assert grouping["t0_bin"] == 103
    assert grouping["t0_method"] == "count_fit"
    assert grouping["t0_reference_run"] == 42


def test_promote_t0_clamps_negative_bin_to_zero():
    grouping = {"t0_bin": 2}
    out = promote_t0_to_grouping(grouping, -1.0, bin_width_us=0.05)
    assert out["after"]["t0_bin"] == 0
    # The residual reflects the delta ACTUALLY applied after the ≥0 clamp
    # (−2 bins = −0.1 µs), not the rounded −20 bins — so the clamp's lost shift
    # is disclosed, not hidden as a zero residual.
    assert out["residual_us"] == pytest.approx(-1.0 - (-2) * 0.05)


def test_promote_t0_rejects_nonpositive_bin_width():
    with pytest.raises(ValueError, match="bin width"):
        promote_t0_to_grouping({"t0_bin": 0}, 0.1, bin_width_us=0.0)


# --- N3: background ---------------------------------------------------------


def test_promote_background_writes_fixed_pair_and_provenance():
    grouping = {}
    out = promote_background_to_grouping(grouping, forward=12.5, backward=11.0, reference_run=3039)
    assert out["before"] == {"forward": 0.0, "backward": 0.0}
    assert out["after"] == {"forward": 12.5, "backward": 11.0}
    assert grouping["background_fixed_values"] == [12.5, 11.0]
    assert grouping["background_mode"] == "fixed"
    # Self-enables the correction so the reduction actually applies it.
    assert grouping["background_correction"] is True
    assert grouping["background_method"] == "count_fit"
    assert grouping["background_reference_run"] == 3039


def test_promote_background_single_side_keeps_existing_other_side():
    grouping = {"background_fixed_values": [10.0, 9.0]}
    out = promote_background_to_grouping(grouping, forward=12.0)
    assert out["before"] == {"forward": 10.0, "backward": 9.0}
    assert grouping["background_fixed_values"] == [12.0, 9.0]
