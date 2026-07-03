"""Tests for grouped raw-histogram background subtraction."""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.transform.background import (
    apply_grouped_background_correction,
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


def test_background_support_is_limited_to_psi_style_formats() -> None:
    assert supports_background_correction(metadata={"facility": "PSI"}) is True
    assert supports_background_correction(metadata={"instrument": "LEM"}) is True
    assert supports_background_correction(metadata={}, source_file="run.bin") is True
    assert (
        supports_background_correction(metadata={"facility": "ISIS"}, source_file="run.nxs")
        is False
    )
