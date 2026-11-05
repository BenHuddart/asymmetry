"""Timing facts follow the selected acquisition and exact reference detector."""

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.io.periods import select_period_histograms
from asymmetry.core.project.profiles import (
    T0Policy,
    profile_fingerprint_for_run,
    profile_from_payload,
    resolve_effective_grouping,
)
from asymmetry.core.transform.grouping import group_forward_backward
from asymmetry.core.transform.rebin import binning_spec
from asymmetry.core.transform.reduce import reduce_period_asymmetry


def period_run():
    red = []
    green = []
    for i, (peak, anchor) in enumerate(((10, 20), (12, 25))):
        counts = np.full(80, 10.0)
        counts[peak] = 1000
        red.append(Histogram(counts, bin_width=0.1, t0_bin=0))
        green.append(Histogram(100 + 20 * i + np.arange(80), bin_width=0.1, t0_bin=anchor))
    return Run(
        run_number=1,
        histograms=red,
        metadata={"facility": "PSI"},
        grouping={
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "t0_source": "missing",
            "detector_t0_bins": [0, 0],
            "period_t0_sources": ["missing", "file"],
            "period_histograms": [red, green],
        },
    )


def resolve(run, mode="manual", offset=2):
    profile = profile_from_payload(run.grouping, "timing", profile_fingerprint_for_run(run))
    profile.t0_policy = T0Policy(mode=mode, offset_bins=offset if mode == "manual" else None)
    return resolve_effective_grouping(profile, run)


def test_manual_fallback_delay_is_not_transferred_to_file_calibrated_period():
    run = period_run()
    grouping = resolve(run)
    assert grouping["effective_detector_t0_bins"] == [12, 14]
    selected, selected_grouping = select_period_histograms(run.histograms, grouping, 1)
    assert "effective_detector_t0_bins" not in selected_grouping
    assert selected_grouping["t0_policy"] == {"mode": "manual", "offset_bins": 2}
    selected_counts = group_forward_backward(selected, selected_grouping).counts
    assert selected_counts.alignment.common_t0_bin + selected_counts.alignment.first_bin == 27
    result = reduce_period_asymmetry(
        run.histograms,
        grouping,
        "green",
        binning=binning_spec(grouping),
        forward_idx=[0],
        backward_idx=[1],
        alpha=1,
        use_deadtime=False,
        deadtime_mode="off",
        use_background=False,
        metadata=run.metadata,
    )
    assert result.alignments["green"].common_t0_bin + result.alignments["green"].first_bin == 27
    np.testing.assert_allclose(result.time, np.arange(75) * 0.1 - 2.2, atol=1e-15)
    assert run.histograms[0].t0_bin == 0
    assert selected[0].t0_bin == 20


def test_absent_period_source_table_does_not_inherit_another_period_file_fact():
    run = period_run()
    run.grouping.update(
        t0_source="file",
        period_t0_sources=["file", "missing"],
        detector_t0_sources=["file", "file"],
        period_detector_t0_sources=[["file", "file"], None],
    )
    run.histograms[0].counts[:] = 0
    run.grouping["period_histograms"][1][0].counts[:] = 0
    _, selected = select_period_histograms(run.histograms, run.grouping, 1)
    assert "detector_t0_sources" not in selected
    profile = profile_from_payload(run.grouping, "timing", profile_fingerprint_for_run(run))
    profile.period_mode = "green"
    with pytest.raises(ValueError, match="selected channels 1"):
        resolve_effective_grouping(profile, run)


@pytest.mark.parametrize("offset", [0, 2])
def test_changed_pair_uses_its_own_fractional_reference_origin(offset):
    histograms = [
        Histogram(100 + i * 20 + np.arange(8), bin_width=0.1, t0_bin=3, t0_time_us=origin)
        for i, origin in enumerate((0.32, 0.39, 0.34))
    ]
    run = Run(
        run_number=1,
        histograms=histograms,
        grouping={
            "groups": {1: [1], 2: [2], 3: [3]},
            "forward_group": 1,
            "backward_group": 2,
            "t0_bin": 3,
            "t0_time_us": 0.32,
            "detector_t0_bins": [3, 3, 3],
            "t0_source": "file",
        },
    )
    profile = profile_from_payload(run.grouping, "timing", profile_fingerprint_for_run(run))
    profile.forward_group, profile.backward_group = 2, 3
    profile.t0_policy = T0Policy(mode="manual", offset_bins=offset)
    grouping = resolve_effective_grouping(profile, run)
    expected = 0.39 + offset * 0.1
    assert grouping["t0_time_us"] == pytest.approx(expected)
    counts = group_forward_backward(histograms, grouping).counts
    assert counts.t0_time_us + counts.alignment.first_bin * 0.1 == pytest.approx(expected)
    np.testing.assert_allclose(
        (np.arange(8) + 0.5) * 0.1 - counts.t0_time_us, (np.arange(8) + 0.5) * 0.1 - expected
    )


def test_repairing_profile_does_not_resolve_superseded_manual_membership():
    run = period_run()
    run.grouping.update(
        groups={1: [1], 2: [1]},
        t0_source="file",
        period_t0_sources=["file", "file"],
        t0_policy={"mode": "manual", "offset_bins": 2},
    )
    profile = profile_from_payload(run.grouping, "repair", profile_fingerprint_for_run(run))
    profile.groups, profile.period_mode = {1: [1], 2: [2]}, "green"
    profile.t0_policy = T0Policy(mode="from_file")
    assert resolve_effective_grouping(profile, run)["t0_bin"] == 25


def test_unused_missing_timing_cannot_suppress_selected_manual_period_offset():
    run = period_run()
    for period in run.grouping["period_histograms"]:
        period.append(Histogram(np.ones(80), bin_width=0.1, t0_bin=0))
    run.grouping.update(
        t0_source="file",
        period_t0_sources=["file", "file"],
        detector_t0_sources=["file", "file", "missing"],
        period_detector_t0_sources=[["file", "file", "missing"]] * 2,
    )
    grouping = resolve(run)
    histograms, selected = select_period_histograms(run.histograms, grouping, 1)
    counts = group_forward_backward(histograms, selected).counts
    assert counts.alignment.common_t0_bin + counts.alignment.first_bin == 27
    assert counts.alignment.timing_sources == ((0, "manual"), (1, "manual"))


def test_deferred_manual_offset_uses_explicit_resolved_pair():
    histograms = [Histogram(np.ones(80), bin_width=0.1, t0_bin=i + 1) for i in range(3)]
    grouping = {
        "groups": {1: [1], 2: [2]},
        "detector_t0_sources": ["missing", "file", "file"],
        "t0_policy": {"mode": "manual", "offset_bins": 2},
    }
    counts = group_forward_backward(histograms, grouping, resolved_indices=([1], [2])).counts
    assert counts.alignment.common_t0_bin + counts.alignment.first_bin == 5
    assert counts.alignment.timing_sources == ((1, "manual"), (2, "manual"))
