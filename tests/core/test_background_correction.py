"""Tests for grouped raw-histogram background subtraction."""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.transform.background import (
    apply_grouped_background_correction,
    resolve_facility,
    supports_background_correction,
)


def test_estimated_background_subtracts_mean_range() -> None:
    forward = np.array([10.0, 14.0, 100.0, 104.0])
    backward = np.array([20.0, 22.0, 80.0, 82.0])

    result = apply_grouped_background_correction(
        forward,
        backward,
        grouping={"background_range": [0, 1]},
        t0_bin=2,
        bin_width_us=0.01,
    )

    assert result.applied is True
    assert result.method == "estimated"
    assert result.values == pytest.approx((12.0, 21.0))
    assert result.ranges == ((0, 1), (0, 1))
    np.testing.assert_allclose(result.forward, [-2.0, 2.0, 88.0, 92.0])
    np.testing.assert_allclose(result.backward, [-1.0, 1.0, 59.0, 61.0])
    assert result.forward_error is not None
    assert result.backward_error is not None
    np.testing.assert_allclose(result.forward_error, np.sqrt([16.0, 20.0, 106.0, 110.0]))
    np.testing.assert_allclose(result.backward_error, np.sqrt([30.5, 32.5, 90.5, 92.5]))


def test_fixed_background_subtracts_forward_and_backward_values() -> None:
    result = apply_grouped_background_correction(
        np.array([10.0, 12.0]),
        np.array([8.0, 9.0]),
        grouping={"background_fixed_values": [1.5, 2.0]},
        t0_bin=0,
        bin_width_us=0.01,
    )

    assert result.applied is True
    assert result.method == "fixed"
    assert result.values == pytest.approx((1.5, 2.0))
    np.testing.assert_allclose(result.forward, [8.5, 10.5])
    np.testing.assert_allclose(result.backward, [6.0, 7.0])
    assert result.forward_error is not None
    assert result.backward_error is not None
    np.testing.assert_allclose(result.forward_error, np.sqrt([10.0, 12.0]))
    np.testing.assert_allclose(result.backward_error, np.sqrt([8.0, 9.0]))


def test_default_range_follows_musrfit_t0_fraction() -> None:
    forward = np.arange(20.0)
    backward = np.arange(20.0) + 10.0

    result = apply_grouped_background_correction(
        forward,
        backward,
        grouping={},
        t0_bin=10,
        bin_width_us=0.01,
    )

    assert result.applied is True
    assert result.ranges == ((1, 6), (1, 6))
    assert result.values == pytest.approx((3.5, 13.5))


def test_invalid_background_range_leaves_counts_unchanged() -> None:
    forward = np.array([10.0, 12.0])
    backward = np.array([8.0, 9.0])

    result = apply_grouped_background_correction(
        forward,
        backward,
        grouping={"background_range": [0, 5]},
        t0_bin=0,
        bin_width_us=0.01,
    )

    assert result.applied is False
    assert result.method == "invalid_range"
    assert result.forward_error is None
    assert result.backward_error is None
    np.testing.assert_allclose(result.forward, forward)
    np.testing.assert_allclose(result.backward, backward)


# --- facility resolution and the accelerator-period trimming -----------------
#
# A ``range`` window is trimmed to whole accelerator periods only for facilities
# with a known beam period. That used to require the caller to pass ``facility``
# by hand, so every core-API caller who did not know to (while the GUI did) lost
# the trimming with no sign of it.


def _range_window(**kwargs):
    """The window a 200-bin ``range`` correction actually averaged over."""
    counts = np.arange(200.0) + 50.0
    result = apply_grouped_background_correction(
        counts,
        counts,
        grouping=kwargs.pop("grouping", {"background_range": [0, 100]}),
        t0_bin=150,
        # 0.005 µs bins: the PSI 0.01975 µs period spans just under four bins, so
        # a 100-bin window trims visibly.
        bin_width_us=0.005,
        **kwargs,
    )
    assert result.applied
    return result.ranges[0]


def test_facility_resolution_prefers_metadata_then_instrument_then_grouping() -> None:
    assert resolve_facility(metadata={"facility": "PSI"}) == "PSI"
    assert resolve_facility(metadata={"facility": " PSI "}) == "PSI"
    assert resolve_facility(metadata={"psi_format": "bin"}) == "PSI"
    assert resolve_facility(metadata={"facility": "", "instrument": "LEM"}) == "LEM"
    assert resolve_facility(grouping={"instrument": "GPS"}) == "GPS"
    # metadata wins over the grouping's canonical instrument identity.
    assert resolve_facility(metadata={"facility": "TRIUMF"}, grouping={"instrument": "GPS"}) == (
        "TRIUMF"
    )
    assert resolve_facility() == ""
    assert resolve_facility(metadata=None, grouping=None) == ""


def test_range_window_is_trimmed_to_whole_beam_periods_by_default() -> None:
    """The regression: this trimming needed an explicit ``facility=`` argument."""
    untrimmed = _range_window(facility="")
    derived = _range_window(metadata={"facility": "PSI"})
    explicit = _range_window(facility="PSI")
    assert derived == explicit
    assert derived != untrimmed
    assert derived[1] < untrimmed[1]


def test_explicit_empty_facility_still_opts_out_of_trimming() -> None:
    """``facility=""`` remains the "no facility, no trimming" opt-out, and must
    not be overridden by metadata that says otherwise."""
    assert _range_window(facility="", metadata={"facility": "PSI"}) == _range_window(facility="")


def test_explicit_facility_overrides_the_metadata() -> None:
    assert _range_window(facility="PSI", metadata={"facility": "ISIS"}) == _range_window(
        facility="PSI"
    )


def test_unknown_facility_leaves_the_window_alone() -> None:
    """ISIS and RAL have a zero period recorded, and an unknown label no period at
    all: both mean "do not trim", not "guess"."""
    untrimmed = _range_window(facility="")
    for metadata in ({"facility": "ISIS"}, {"facility": "RAL"}, {"facility": "Nowhere"}):
        assert _range_window(metadata=metadata) == untrimmed


def test_background_support_is_limited_to_psi_style_formats() -> None:
    assert supports_background_correction(metadata={"facility": "PSI"}) is True
    assert supports_background_correction(metadata={"instrument": "LEM"}) is True
    assert supports_background_correction(metadata={}, source_file="run.bin") is True
    assert (
        supports_background_correction(metadata={"facility": "ISIS"}, source_file="run.nxs")
        is False
    )
