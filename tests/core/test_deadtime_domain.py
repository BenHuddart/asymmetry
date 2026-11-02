"""Signed calibration coefficients and the inverse correction's numerical domain."""

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram
from asymmetry.core.transform import deadtime
from asymmetry.core.transform.rebin import binning_spec


@pytest.mark.parametrize(
    "coefficient,expected",
    [(-0.1, [100 / 21, 50 / 3]), (0.0, [5.0, 20.0]), (0.1, [100 / 19, 25.0])],
)
def test_signed_correction_matches_mantid_6_16_1(coefficient, expected):
    """Native ApplyDeadTimeCorr v1: counts [5,20], 1 μs bins, 10 good frames.

    Expected values also follow analytically from the published non-paralysable
    inverse. The negative coefficient reduces counts; it is not clamped to zero.
    """
    result = deadtime.apply_deadtime_correction([5, 20], coefficient, 1.0, num_good_frames=10)
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize("counts", [[10.0], [5.0, 20.0]])
def test_saturation_rejects_the_whole_detector(counts):
    with pytest.raises(ValueError, match="requires < 1") as error:
        deadtime.apply_deadtime_correction(counts, 0.1, 1.0, num_good_frames=1)
    assert error.value.bin_index == int(np.argmax(counts))
    assert error.value.occupancy == pytest.approx(max(counts) * 0.1)


@pytest.mark.parametrize(
    "counts,coefficient,width,frames",
    [
        ([1], np.nan, 1, 1),
        ([1], np.inf, 1, 1),
        ([1], -np.inf, 1, 1),
        ([1], 0.1, 0, 1),
        ([1], 0.1, -1, 1),
        ([1], 0.1, np.nan, 1),
        ([1], 0.1, 1, 0),
        ([1], 0.1, 1, -1),
        ([1], -0.1, 1, np.nan),
        ([-1], 0.1, 1, 1),
        ([np.nan], 0.1, 1, 1),
        ([np.inf], 0.1, 1, 1),
    ],
)
def test_invalid_correction_inputs_raise(counts, coefficient, width, frames):
    with pytest.raises(ValueError, match="Deadtime correction rejected"):
        deadtime.apply_deadtime_correction(counts, coefficient, width, num_good_frames=frames)


@pytest.mark.parametrize("coefficient", [-0.1, 0.1])
def test_nonzero_coefficients_require_declared_frames(coefficient):
    with pytest.raises(ValueError, match="good frames"):
        deadtime.prepare_histograms_with_deadtime(
            [Histogram(np.ones(2), bin_width=1)], {"dead_time_us": [coefficient]}, True
        )


def test_zero_table_needs_no_frame_normalisation():
    histogram = Histogram(np.array([1.0, 2.0]), bin_width=1)
    histograms, applied = deadtime.prepare_histograms_with_deadtime(
        [histogram], {"dead_time_us": [0.0]}, True, mode="manual"
    )
    assert applied.status == "applied"
    np.testing.assert_array_equal(histograms[0].counts, histogram.counts)


def test_later_detector_error_identifies_its_source():
    with pytest.raises(ValueError, match="detector 2") as error:
        deadtime.prepare_histograms_with_deadtime(
            [Histogram(np.array([5.0]), bin_width=1), Histogram(np.array([10.0]), bin_width=1)],
            {"dead_time_us": [0.1, 0.1], "good_frames": 1.0},
            True,
        )
    assert error.value.detector == 1


@pytest.mark.parametrize("text", ["2\n1 -0.1\n2 0.2", "-0.1\n0.2"])
def test_signed_calibration_text(text):
    assert deadtime.parse_deadtime_calibration_text(text, n_histograms=2) == [-0.1, 0.2]


@pytest.mark.parametrize("text", ["1\n1 nan", "1\n1 inf", "-inf"])
def test_nonfinite_calibration_text_is_rejected(text):
    with pytest.raises(deadtime.DeadtimeCorrectionError, match="coefficient"):
        deadtime.parse_deadtime_calibration_text(text)


def test_calibration_preserves_sign_and_rejects_nonfinite_results_atomically():
    grouping = {"dead_time_us": [0.1, 0.2]}
    calibration = deadtime.DeadtimeCalibration((-0.1, 0.1), (None, None), "count_fit", 7)
    deadtime.promote_deadtime_to_grouping(grouping, calibration)
    assert grouping["dead_time_us"] == pytest.approx([-0.1, 0.1])
    before = dict(grouping)
    with pytest.raises(deadtime.DeadtimeCorrectionError):
        deadtime.DeadtimeCalibration((np.inf,), (None,), "count_fit", 7)
    assert grouping == before


@pytest.mark.parametrize("value", [np.nan, "bad", np.inf])
def test_invalid_resolved_table_is_not_treated_as_absent(value):
    grouping = {"dead_time_us": [value], "good_frames": 10}
    with pytest.raises(deadtime.DeadtimeCorrectionError, match="detector 1"):
        deadtime.prepare_histograms_with_deadtime(
            [Histogram(np.ones(2), bin_width=1)], grouping, True
        )
    assert "deadtime_method" not in grouping


def test_negative_integer_coefficient_is_not_a_histogram_count_header():
    assert deadtime.parse_deadtime_calibration_text("-1\n0.2", n_histograms=2) == [-1, 0.2]


def test_invalid_indexed_coefficient_cannot_become_a_zero_entry():
    with pytest.raises(deadtime.DeadtimeCorrectionError, match="detector 1"):
        deadtime.parse_deadtime_calibration_text("2\n1 invalid\n2 0.1", n_histograms=2)


def test_reduction_rejects_a_saturated_unused_detector_outside_the_good_window():
    from asymmetry.core.transform import reduce_grouped_asymmetry

    histograms = [
        Histogram(np.array(counts), bin_width=1, good_bin_end=1)
        for counts in ([5, 5], [4, 4], [10, 0])
    ]
    grouping = {
        "good_frames": 1,
        "dead_time_us": [0.1] * 3,
        "first_good_bin": 1,
        "last_good_bin": 1,
    }
    with pytest.raises(deadtime.DeadtimeCorrectionError, match="detector 3") as error:
        reduce_grouped_asymmetry(
            histograms=histograms,
            binning=binning_spec(grouping),
            grouping=grouping,
            forward_idx=[0],
            backward_idx=[1],
            alpha=1,
            use_deadtime=True,
            deadtime_mode="file",
            use_background=False,
        )
    assert error.value.bin_index == 0


def test_increment_overflow_is_rejected_before_promotion():
    from asymmetry.core.transform.corrections import CorrectionOutcome, CorrectionStatus

    grouping = {"dead_time_us": [1e308, 0.0]}
    base = CorrectionOutcome(
        "deadtime",
        "manual",
        CorrectionStatus.APPLIED,
        effective={"tau_us": [1e308, 0.0]},
        uncertainty="unknown",
    )
    with pytest.raises(deadtime.DeadtimeCorrectionError):
        deadtime.DeadtimeCalibration.from_increment(
            base, 1e308, n_histograms=2, detector_indices=[0], error_us=None, source_run=7
        )
    assert grouping == {"dead_time_us": [1e308, 0.0]}
