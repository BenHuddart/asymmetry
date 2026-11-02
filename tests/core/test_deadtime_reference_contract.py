"""Recorded synthetic Mantid 6.16.1 count references and the reduction gap."""

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram
from asymmetry.core.transform.deadtime import (
    apply_deadtime_correction,
    prepare_histograms_with_deadtime,
)
from asymmetry.core.transform.fb_counts import GroupCounts, linear_combination_variance


@pytest.mark.parametrize(
    "coefficient, expected",
    [
        (-0.2, [83.33333333333334, 23.80952380952381]),
        (0.0, [100.0, 25.0]),
        (0.2, [125.0, 26.315789473684212]),
    ],
)
def test_mantid_6_16_1_signed_count_reference(coefficient, expected):
    # ApplyDeadTimeCorr v1: 1 µs bins, 100 frames, supplied errors [10, 5].
    # Mantid preserves those errors; these independent values pin its means.
    result = apply_deadtime_correction([100.0, 25.0], coefficient, 1.0, num_good_frames=100.0)
    np.testing.assert_allclose(result, expected, rtol=1e-12)


@pytest.mark.parametrize(
    "coefficient, expected_count",
    [
        (-0.2, 149.56581113517456),
        (0.0, 170.0),
        (0.05, 176.28017446517896),
        (0.2, 199.2738623720619),
        (0.5, 281.89795136545234),
    ],
)
def test_mantid_6_16_1_grouped_binned_reference(coefficient, expected_count):
    # Two independent spectra, 1 µs bins and 100 frames, grouped then rebinned
    # into 2 µs bins. Mantid keeps the measured variance sum 100+25+36+9=170.
    raw = np.array([[100.0, 25.0, 100.0, 25.0], [36.0, 9.0, 36.0, 9.0]])
    corrected = np.array(
        [apply_deadtime_correction(row, coefficient, 1.0, num_good_frames=100.0) for row in raw]
    )
    grouped = GroupCounts(corrected.sum(axis=0), raw.sum(axis=0))
    np.testing.assert_allclose(grouped.counts.reshape(2, 2).sum(axis=1), expected_count, rtol=1e-12)
    sigma = np.sqrt(linear_combination_variance(grouped, [0, 2, 4]))
    np.testing.assert_allclose(sigma, [13.038404810405298] * 2, rtol=1e-12)


@pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason="#390: deadtime preparation has no counting-variance representation",
)
def test_deadtime_preparation_preserves_measured_count_variance():
    histogram = Histogram(np.array([100.0]), bin_width=1.0)
    prepared, _ = prepare_histograms_with_deadtime(
        [histogram],
        {"dead_time_us": [0.2], "good_frames": 100.0},
        True,
        mode="manual",
    )
    np.testing.assert_array_equal(prepared[0].counts, [125.0])
    np.testing.assert_array_equal(prepared[0].count_variance, [100.0])
    np.testing.assert_array_equal(histogram.counts, [100.0])
