"""Selected acquisition bins determine counts, coordinates and timing residuals."""

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram
from asymmetry.core.transform.grouping import group_forward_backward
from asymmetry.core.transform.reduce import corrected_grouped_counts


def tagged_histograms():
    return [
        Histogram(
            np.arange(10.0, 15.0), 0.1, 1, count_variance=np.arange(30.0, 35.0), t0_time_us=0.12
        ),
        Histogram(
            np.arange(20.0, 26.0), 0.1, 3, count_variance=np.arange(50.0, 56.0), t0_time_us=0.37
        ),
    ]


def test_selected_observed_support_counts_variance_and_exact_reference():
    histograms = tagged_histograms()
    grouping = {"groups": {1: [1], 2: [2]}}
    pair = group_forward_backward(histograms, grouping).counts
    np.testing.assert_array_equal(pair.forward.counts, [10, 11, 12, 13])
    np.testing.assert_array_equal(pair.backward.counts, [22, 23, 24, 25])
    np.testing.assert_array_equal(pair.forward.variance, [30, 31, 32, 33])
    np.testing.assert_array_equal(pair.backward.variance, [52, 53, 54, 55])
    assert pair.common_t0_bin == 1
    assert pair.t0_time_us == pytest.approx(0.17)
    assert pair.alignment.first_bin == 2
    assert pair.alignment.detector_offsets == ((0, 0), (1, -2))
    np.testing.assert_allclose(
        pair.alignment.timing_residuals_us, ((0, -0.05), (1, 0.0)), atol=1e-15
    )
    assert pair.alignment.timing_spread_us == pytest.approx(0.05)
    corrected = corrected_grouped_counts(
        histograms=histograms,
        grouping=grouping,
        forward_idx=[0],
        backward_idx=[1],
        use_deadtime=False,
        deadtime_mode="off",
        use_background=False,
    )
    np.testing.assert_array_equal(corrected.forward.counts, pair.forward.counts)
    assert corrected.t0_time_us == pair.t0_time_us
    np.testing.assert_array_equal(histograms[0].counts, np.arange(10.0, 15.0))


def test_spectator_grid_and_stored_scalar_cannot_change_selected_result():
    histograms = tagged_histograms()
    grouping = {"groups": {1: [1], 2: [2]}, "t0_time_us": 8.0}
    expected = group_forward_backward(histograms, grouping).counts
    histograms.insert(0, Histogram(np.ones(2), 9.0, 3, t0_time_us=100.0))
    actual = group_forward_backward(histograms, dict(grouping, groups={1: [2], 2: [3]})).counts
    np.testing.assert_array_equal(actual.forward.counts, expected.forward.counts)
    assert actual.t0_time_us == expected.t0_time_us
    assert actual.bin_width_us == expected.bin_width_us


@pytest.mark.parametrize("override", [[1], [1, 2.5], [True, 3], [None, 3], "1,3", [1.0, 3], None])
def test_malformed_explicit_overrides_are_unavailable_and_repairable(override):
    histograms = tagged_histograms()
    grouping = {"groups": {1: [1], 2: [2]}, "effective_detector_t0_bins": override}
    with pytest.raises(ValueError, match="t0"):
        group_forward_backward(histograms, grouping)
    assert grouping["effective_detector_t0_bins"] == override


def test_incompatible_selected_grid_is_rejected_at_grouping():
    histograms = tagged_histograms()
    histograms[1].bin_width = 0.11
    with pytest.raises(ValueError, match="grid"):
        group_forward_backward(histograms, {"groups": {1: [1], 2: [2]}})


@pytest.mark.parametrize("length, valid", [(0, False), (2, False), (3, True)])
def test_empty_and_one_bin_observed_intersections(length, valid):
    histograms = [Histogram(np.ones(4), 0.1, 0), Histogram(np.ones(length), 0.1, 2)]
    grouping = {"groups": {1: [1], 2: [2]}}
    if valid:
        pair = group_forward_backward(histograms, grouping).counts
        assert pair.forward.counts.size == 1
        np.testing.assert_array_equal(pair.forward.counts, [1.0])
        np.testing.assert_array_equal(pair.backward.counts, [1.0])
    else:
        with pytest.raises(ValueError, match="support"):
            group_forward_backward(histograms, grouping)


@pytest.mark.parametrize("width", [0.0, -1.0, np.inf, np.nan, True, 0.100000000001])
def test_selected_grid_requires_positive_finite_compatible_widths(width):
    histograms = tagged_histograms()
    histograms[1].bin_width = width
    with pytest.raises(ValueError, match="grid|width"):
        group_forward_backward(histograms, {"groups": {1: [1], 2: [2]}})


def test_tied_references_exclusions_and_changed_pairs_follow_acquired_ordinals():
    histograms = [
        Histogram(np.full(8, i + 1.0), 0.1, 3, t0_time_us=exact)
        for i, exact in enumerate((0.32, 0.39, 0.34))
    ]
    grouping = {"groups": {1: [2, 1], 2: [3]}, "excluded_detectors": []}
    pair = group_forward_backward(histograms, grouping).counts
    assert pair.t0_time_us == 0.32
    assert pair.alignment.timing_spread_us == pytest.approx(0.07)
    excluded = group_forward_backward(histograms, dict(grouping, excluded_detectors=[1])).counts
    assert excluded.t0_time_us == 0.39
    changed = group_forward_backward(histograms, dict(grouping, groups={1: [3], 2: [1]})).counts
    assert changed.t0_time_us == 0.32
    # IDs are acquired ordinals. Reordering the storage requires explicit membership remapping;
    # equal-anchor ties deliberately follow the new ordinal, not a physical detector registry.
    reordered = group_forward_backward(
        [histograms[1], histograms[0], histograms[2]], grouping
    ).counts
    assert reordered.t0_time_us == 0.39


def test_equal_shifted_exact_origins_have_zero_residuals():
    histograms = tagged_histograms()
    histograms[1].t0_time_us = 0.32
    counts = group_forward_backward(histograms, {"groups": {1: [1], 2: [2]}}).counts
    assert counts.t0_time_us == pytest.approx(0.12)
    np.testing.assert_allclose(
        [v for _, v in counts.alignment.timing_residuals_us], 0.0, atol=1e-16
    )


def test_whole_bin_policy_offset_preserves_fractional_residuals_and_raw_arrays():
    histograms = tagged_histograms()
    grouping = {"groups": {1: [1], 2: [2]}}
    base = group_forward_backward(histograms, grouping).counts
    shifted = group_forward_backward(
        histograms, dict(grouping, effective_detector_t0_bins=[3, 5])
    ).counts
    np.testing.assert_array_equal(shifted.forward.counts, base.forward.counts)
    assert shifted.t0_time_us == pytest.approx(base.t0_time_us + 0.2)
    assert shifted.alignment.timing_residuals_us == base.alignment.timing_residuals_us
    assert [hist.t0_bin for hist in histograms] == [1, 3]


def test_disjoint_vector_projections_resolve_their_own_exact_references():
    from asymmetry.core.data.dataset import Run
    from asymmetry.core.transform.projections import reduce_run_projections

    histograms = tagged_histograms() + [
        Histogram(np.full(8, level), 0.1, 2, t0_time_us=exact)
        for level, exact in ((40.0, 0.21), (50.0, 0.29))
    ]
    grouping = {
        "groups": {1: [1], 2: [2], 3: [3], 4: [4]},
        "projections": [
            {"label": "P_x", "forward_group": 1, "backward_group": 2},
            {"label": "P_y", "forward_group": 3, "backward_group": 4},
            {"label": "P_z", "forward_group": 1, "backward_group": 4},
        ],
    }
    run = Run(histograms=histograms, grouping=grouping)
    results = reduce_run_projections(run, 0)
    assert results["P_x"].time[0] == pytest.approx(-0.12)
    assert results["P_y"].time[0] == pytest.approx(-0.16)
    assert results["P_z"].time[0] == pytest.approx(-0.14)
    assert results["P_x"].metadata["grouped_alignment"]["timing_spread_us"] == pytest.approx(0.05)


def test_archived_musrfit_integer_grouped_count_anchor():
    """musrfit 1.12.1 (708f777b), ROOT 6.40.04: type 0, forward 1 2,
    t0 2 4, data 1 9, packing 1, width .1 μs; no corrections.
    Recorded native counts are exact; timestamp tolerance is 1e-12 μs.
    """
    from asymmetry.core.transform.grouping import group_counts_aligned

    histograms = [
        Histogram(100.0 + np.arange(100), 0.1, 2),
        Histogram(200.0 + 10 * np.arange(100), 0.1, 4),
    ]
    grouped = group_counts_aligned(histograms, [0, 1])
    # Original aligned bins 3…10 are local bins 1…8 after removing two padded bins.
    np.testing.assert_array_equal(grouped.counts[1:9], [331, 342, 353, 364, 375, 386, 397, 408])
    np.testing.assert_allclose(
        (np.arange(1, 9) - 2) * 0.1, np.arange(-1, 7) * 0.1, atol=1e-12, rtol=0
    )


def test_count_fit_background_rejects_unobserved_cropped_range():
    from asymmetry.core.fitting.count_corrections import count_background_domain

    histograms = [Histogram(np.arange(8.0) + 10.0, 0.1, n) for n in (1, 3)]
    grouping = {
        "groups": {1: [1], 2: [2]},
        "background_correction": True,
        "background_mode": "range",
        "background_range": [1, 2],
    }
    with pytest.raises(ValueError, match="background range"):
        count_background_domain(histograms, grouping, [(1, [0]), (2, [1])], {"facility": "PSI"})


def test_single_group_fourier_is_invariant_to_pair_and_excluded_spectators():
    from asymmetry.core.data.dataset import Run
    from asymmetry.core.fourier.grouped import build_group_signal_dataset
    from asymmetry.core.fourier.spectrum import fourier_grouping_digest

    histograms = [
        Histogram(np.arange(10.0) + level, 0.1, n) for level, n in ((10, 0), (20, 1), (30, 4))
    ]
    grouping = {
        "groups": {1: [1], 2: [2], 3: [3]},
        "forward_group": 1,
        "backward_group": 2,
        "first_good_bin": 4,
        "last_good_bin": 6,
    }
    run = Run(histograms=histograms, grouping=grouping)
    base = build_group_signal_dataset(run, 1, center_signal=False, apply_lifetime_correction=False)
    digest = fourier_grouping_digest(run)
    run.grouping = dict(grouping, backward_group=3)
    changed = build_group_signal_dataset(
        run, 1, center_signal=False, apply_lifetime_correction=False
    )
    assert fourier_grouping_digest(run) == digest
    np.testing.assert_array_equal(changed.asymmetry, base.asymmetry)
    np.testing.assert_array_equal(changed.time, base.time)
    run.grouping = dict(grouping, excluded_detectors=[2, 3])
    excluded = build_group_signal_dataset(
        run, 1, center_signal=False, apply_lifetime_correction=False
    )
    np.testing.assert_array_equal(excluded.asymmetry, base.asymmetry)
    np.testing.assert_array_equal(excluded.time, base.time)


@pytest.mark.parametrize("bounds,n_bins", [(([0, 7], [3, 9]), 10), (([0, 0], [9, 9]), 0)])
def test_disjoint_good_windows_and_zero_length_support_are_not_fabricated(bounds, n_bins):
    from asymmetry.core.transform.t0 import good_window_for_groups

    with pytest.raises(ValueError, match="support"):
        good_window_for_groups([0, 1], [0, 0], *bounds, common_t0_bin=0, n_bins=n_bins)


def test_integral_uses_observed_counts_variances_and_exact_time_window():
    from asymmetry.core.data.dataset import Run
    from asymmetry.core.transform.integral import integrate_run

    run = Run(histograms=tagged_histograms(), grouping={"groups": {1: [1], 2: [2]}})
    value, error = integrate_run(run, t_min=0.0, t_max=0.2)
    # Exact stamps .08 and .18 μs select raw F bins 2,3 and B bins 4,5.
    assert value == pytest.approx((25.0 - 49.0) / 74.0)
    assert error == pytest.approx(np.sqrt(98.0**2 * 65.0 + 50.0**2 * 109.0) / 74.0**2)
    value, _ = integrate_run(run)
    assert value == pytest.approx((46.0 - 94.0) / 140.0)


def test_final_period_provenance_replaces_inherited_loader_alignment():
    from asymmetry.core.transform.rebin import binning_spec
    from asymmetry.core.transform.reduce import reduce_period_asymmetry
    from asymmetry.core.utils.constants import PeriodMode

    histograms = tagged_histograms()
    grouping = {"groups": {1: [1], 2: [2]}, "period_histograms": [histograms, histograms]}
    result = reduce_period_asymmetry(
        histograms,
        grouping,
        str(PeriodMode.GREEN_PLUS_RED),
        binning=binning_spec(grouping),
        forward_idx=[0],
        backward_idx=[1],
        alpha=1.0,
        use_deadtime=False,
        deadtime_mode="off",
        use_background=False,
    )
    inherited = {
        "grouped_alignment": {"stale": True},
        "period_alignments": {"stale": True},
        "title": "run",
    }
    metadata = result.metadata_with_alignment(inherited)
    assert "period_alignments" not in metadata
    assert metadata["grouped_alignment"] == result.alignments["green_plus_red"].to_dict()
    assert metadata["title"] == "run"
    assert inherited["grouped_alignment"] == {"stale": True}


@pytest.mark.parametrize("first,last", [(0, 0), (6, 4), (20, 21), (True, 5), (2.5, 5)])
def test_fourier_rejects_invalid_or_unobserved_authored_windows(first, last):
    from asymmetry.core.data.dataset import Run
    from asymmetry.core.fourier.grouped import build_group_signal_dataset

    run = Run(
        histograms=tagged_histograms(),
        grouping={
            "groups": {1: [1, 2]},
            "t0_bin": 3,
            "first_good_bin": first,
            "last_good_bin": last,
        },
    )
    with pytest.raises(ValueError, match="good-bin window"):
        build_group_signal_dataset(run, 1, center_signal=False, apply_lifetime_correction=False)


def test_fourier_window_is_translated_once_after_leading_crop():
    from asymmetry.core.data.dataset import Run
    from asymmetry.core.fourier.grouped import build_group_signal_dataset

    run = Run(
        histograms=tagged_histograms(),
        grouping={"groups": {1: [1, 2]}, "t0_bin": 3, "first_good_bin": 3, "last_good_bin": 5},
    )
    result = build_group_signal_dataset(
        run, 1, center_signal=False, apply_lifetime_correction=False
    )
    np.testing.assert_array_equal(result.asymmetry, [34.0, 36.0, 38.0])
    np.testing.assert_allclose(result.time, [-0.02, 0.08, 0.18], atol=1e-15)


def test_fourier_absent_and_none_declared_t0_use_selected_anchor():
    from asymmetry.core.data.dataset import Run
    from asymmetry.core.fourier.grouped import build_group_signal_dataset

    run = Run(histograms=tagged_histograms(), grouping={"groups": {1: [1, 2]}})
    absent = build_group_signal_dataset(
        run, 1, center_signal=False, apply_lifetime_correction=False
    )
    run.grouping["t0_bin"] = None
    null = build_group_signal_dataset(run, 1, center_signal=False, apply_lifetime_correction=False)
    np.testing.assert_array_equal(null.asymmetry, absent.asymmetry)
    np.testing.assert_array_equal(null.time, absent.time)
