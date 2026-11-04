"""Window policies select acquired observations before forming final asymmetry."""

from dataclasses import replace

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.project.profiles import (
    GoodWindowPolicy,
    GroupingProfile,
    ProfileFingerprint,
    T0Policy,
    resolve_effective_grouping,
)
from asymmetry.core.transform.grouping import InvalidGroupingError, group_forward_backward
from asymmetry.core.transform.rebin import binning_spec
from asymmetry.core.transform.reduce import reduce_grouped_asymmetry


def acquired_run(anchors=(10, 10), bounds=((20, 80), (20, 80))):
    histograms = [
        Histogram(
            (i + 1) * 100 + np.arange(100, dtype=float),
            0.01,
            t0_bin=anchor,
            t0_time_us=(anchor + 0.24) * 0.01,
            good_bin_start=first,
            good_bin_end=last,
        )
        for i, (anchor, (first, last)) in enumerate(zip(anchors, bounds))
    ]
    return Run(
        run_number=1,
        histograms=histograms,
        grouping={
            "t0_bin": max(anchors),
            "first_good_bin": 20,
            "last_good_bin": 80,
            "detector_t0_bins": list(anchors),
            "detector_first_good_bins": [b[0] for b in bounds],
            "detector_last_good_bins": [b[1] for b in bounds],
        },
    )


def request(offset=0, first=None, last=None, *, manual=False):
    return GroupingProfile(
        name="window",
        fingerprint=ProfileFingerprint("", 2),
        groups={1: [1], 2: [2]},
        t0_policy=T0Policy(mode="manual", offset_bins=offset),
        good_window_policy=GoodWindowPolicy(
            mode="manual" if manual else "from_file",
            first_offset_bins=first,
            last_offset_bins=last,
        ),
    )


@pytest.mark.parametrize("offset", [-5, 0, 5])
@pytest.mark.parametrize("anchors", [(10, 10), (10, 15)])
def test_from_file_final_reduction_preserves_raw_membership(offset, anchors):
    run = acquired_run(anchors)
    effective = resolve_effective_grouping(request(offset), run)
    grouped = group_forward_backward(run.histograms, effective).counts
    # Independent intersection: raw k_i = r - (max(t0) - t0_i).
    shifts = np.array([max(anchors) - a for a in anchors])
    raw = np.arange(20 + max(shifts), 81 + min(shifts))
    first, last = raw[0], raw[-1]
    expected = [h.counts[raw - s] for h, s in zip(run.histograms, shifts)]
    for value, oracle in zip((grouped.forward, grouped.backward), expected):
        np.testing.assert_array_equal(value.counts[grouped.window.slice], oracle)
        np.testing.assert_array_equal(value.independent_variance[grouped.window.slice], oracle)
    result = reduce_grouped_asymmetry(
        histograms=run.histograms,
        grouping=effective,
        binning=binning_spec(effective),
        forward_idx=[0],
        backward_idx=[1],
        alpha=1,
        use_deadtime=False,
        deadtime_mode="off",
        use_background=False,
    )
    np.testing.assert_allclose(result.asymmetry, 100 * (expected[0] - expected[1]) / sum(expected))
    np.testing.assert_allclose(result.time, (raw + 0.5 - max(anchors) - offset - 0.24) * 0.01)
    assert (effective["first_good_bin"], effective["last_good_bin"]) == (first, last)
    assert [(h.good_bin_start, h.good_bin_end) for h in run.histograms] == [(20, 80)] * 2


@pytest.mark.parametrize(
    "first,last,expected",
    [(None, 70, (20, 85)), (10, None, (25, 80)), (-10, 75, (5, 90)), (5, 5, (20, 20))],
)
def test_manual_partial_signed_and_advisory_override_reaches_counts(first, last, expected):
    run = acquired_run()
    grouping = resolve_effective_grouping(request(5, first, last, manual=True), run)
    counts = group_forward_backward(run.histograms, grouping).counts
    lo, hi = expected
    np.testing.assert_array_equal(
        counts.forward.counts[counts.window.slice], run.histograms[0].counts[lo : hi + 1]
    )
    assert grouping["t_good_offset"] == lo - 15


@pytest.mark.parametrize("first,last", [(30, 10), (5000, 6000), (-20, 10), (0, 100)])
def test_invalid_authored_request_is_repairable_without_an_observation(first, last):
    run = acquired_run()
    profile = request(0, first, last, manual=True)
    original = run.histograms[0].counts.copy()
    with pytest.raises(InvalidGroupingError, match="[Ww]indow|support"):
        grouping = resolve_effective_grouping(profile, run)
        group_forward_backward(run.histograms, grouping)
    assert profile.good_window_policy.first_offset_bins == first
    np.testing.assert_array_equal(run.histograms[0].counts, original)
    repaired = replace(profile, good_window_policy=GoodWindowPolicy())
    counts = group_forward_backward(
        run.histograms, resolve_effective_grouping(repaired, run)
    ).counts
    assert counts.window.last - counts.window.first + 1 == 61


def test_manual_override_can_repair_an_empty_advisory_file_intersection():
    run = acquired_run(bounds=((20, 30), (40, 50)))
    profile = request(0, -5, 60, manual=True)
    counts = group_forward_backward(run.histograms, resolve_effective_grouping(profile, run)).counts
    np.testing.assert_array_equal(
        counts.forward.counts[counts.window.slice], run.histograms[0].counts[5:71]
    )


def test_released_project_migrates_display_without_inventing_file_facts():
    from asymmetry.core.project.schema import migrate_to_current

    recorded = {"t0_bin": 15, "first_good_bin": 5, "last_good_bin": 80, "t_good_offset": 0}
    old = {"schema_version": 30, "datasets": [{"grouping_overrides": recorded}]}
    migrated = migrate_to_current(old)
    grouping = migrated["datasets"][0]["grouping_overrides"]
    assert grouping == recorded | {"t_good_offset": -10}
    assert recorded["t_good_offset"] == 0
    assert migrate_to_current(migrated) == migrated
    assert not any(key.startswith("detector_") for key in grouping)


@pytest.mark.parametrize("offset,manual,first,last", [(5, False, 20, 80), (5, True, 5, 90)])
def test_finite_window_alpha_uses_the_final_selected_observations(offset, manual, first, last):
    from asymmetry.core.project.profiles import AlphaPolicy

    run = acquired_run(anchors=(10, 15))
    policy = request(offset, first - 20, last - 20, manual=manual)
    profile = replace(policy, alpha_policy=AlphaPolicy(mode="per_run_estimate"))
    grouping = resolve_effective_grouping(profile, run)
    raw = np.arange(first if manual else 25, last + 1)
    expected = run.histograms[0].counts[raw - 5].sum() / run.histograms[1].counts[raw].sum()
    assert grouping["alpha"] == pytest.approx(expected, rel=1e-15)


@pytest.mark.parametrize("manual", [False, True])
def test_independent_projections_resolve_their_own_file_limits_and_effective_t0(manual):
    from asymmetry.core.transform.projections import reduce_run_projections

    run = acquired_run()
    other = acquired_run(anchors=(30, 30), bounds=((40, 60), (40, 60)))
    run.histograms.extend(other.histograms)
    run.grouping.update(
        detector_t0_bins=[10, 10, 30, 30],
        detector_first_good_bins=[20, 20, 40, 40],
        detector_last_good_bins=[80, 80, 60, 60],
    )
    profile = replace(
        request(5, 0, 10, manual=manual),
        fingerprint=ProfileFingerprint("", 4),
        groups={i: [i] for i in range(1, 5)},
        projections=[
            {"label": "P_z", "forward_group": 1, "backward_group": 2},
            {"label": "P_x", "forward_group": 3, "backward_group": 4},
        ],
    )
    run.grouping = resolve_effective_grouping(profile, run)
    result = reduce_run_projections(run, 0)
    assert result["P_z"].n_points == (11 if manual else 61)
    assert result["P_x"].n_points == (11 if manual else 21)
    expected_first = (35 if manual else 40) + 0.5 - 35.24
    assert result["P_x"].time[0] == pytest.approx(expected_first * 0.01)


def test_period_file_windows_keep_the_identical_grid_requirement():
    from asymmetry.core.transform.reduce import reduce_period_asymmetry

    run = acquired_run()
    green = acquired_run(bounds=((40, 60), (40, 60)))
    run.grouping["period_histograms"] = [run.histograms, green.histograms]
    profile = replace(request(), period_mode="green_minus_red")
    grouping = resolve_effective_grouping(profile, run)
    with pytest.raises(ValueError, match="[Ii]dentical|[Ss]ame|[Ee]dges|[Gg]rid"):
        reduce_period_asymmetry(
            run.histograms,
            grouping,
            "green_minus_red",
            binning=binning_spec(grouping),
            forward_idx=[0],
            backward_idx=[1],
            alpha=1,
            use_deadtime=False,
            deadtime_mode="off",
            use_background=False,
        )


def test_held_period_timing_offsets_follow_its_acquired_anchors():
    from asymmetry.core.io.periods import select_period_histograms

    run = acquired_run()
    green = acquired_run(anchors=(20, 25))
    run.grouping["period_histograms"] = [run.histograms, green.histograms]
    grouping = resolve_effective_grouping(request(5, -5, 10, manual=True), run)
    histograms, selected = select_period_histograms(run.histograms, grouping, 1)
    counts = group_forward_backward(histograms, selected).counts
    assert selected["effective_detector_t0_bins"] == [25, 30]
    assert counts.alignment.common_t0_bin + counts.alignment.first_bin == 30
    np.testing.assert_array_equal(
        counts.forward.counts[counts.window.slice], green.histograms[0].counts[20:36]
    )


def test_signed_workflow_request_matches_the_profile_reduction():
    from asymmetry.core.workflow.reduction import ReductionSettings, reduce_run

    run = acquired_run()
    run.grouping.update(groups={1: [1], 2: [2]}, forward_group=1, backward_group=2)
    actual = reduce_run(run, ReductionSettings(t0_offset_bins=5, t_good_offset_bins=-10))
    grouping = resolve_effective_grouping(request(5, -10, None, manual=True), run)
    from asymmetry.core.simulate import reduce_run_to_dataset

    expected = reduce_run_to_dataset(replace(run, grouping=grouping))
    np.testing.assert_allclose(actual.time, expected.time)
    np.testing.assert_allclose(actual.asymmetry, expected.asymmetry)
    np.testing.assert_allclose(actual.error, expected.error)


@pytest.mark.parametrize("value", [True, 1.9, "2", float("nan")])
def test_manual_offsets_are_integer_bin_requests(value):
    with pytest.raises(ValueError, match="signed integer"):
        GoodWindowPolicy(mode="manual", first_offset_bins=value)


@pytest.mark.parametrize("first,last,expected", [(-5, None, (5, 80)), (None, 85, (90, 95))])
def test_partial_manual_request_repairs_an_unavailable_legacy_scalar_window(first, last, expected):
    run = acquired_run()
    run.grouping = {"t0_bin": 10, "first_good_bin": 90, "last_good_bin": 80}
    grouping = resolve_effective_grouping(request(0, first, last, manual=True), run)
    counts = group_forward_backward(run.histograms, grouping).counts
    lo, hi = expected
    np.testing.assert_array_equal(
        counts.forward.counts[counts.window.slice], run.histograms[0].counts[lo : hi + 1]
    )
