"""Selected timing is a required input, independently of run-wide search success."""

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.project.profiles import (
    T0Policy,
    profile_fingerprint_for_run,
    profile_from_payload,
    resolve_effective_grouping,
)
from asymmetry.core.transform.grouping import InvalidGroupingError, group_forward_backward
from asymmetry.core.workflow.reduction import ReductionSettings, reduce_run


def timing_run(*, source="PSI", missing=False):
    histograms = []
    for anchor, peak in zip([4, 8, 6, 8], [10, None, 12, 14], strict=True):
        counts = np.zeros(40)
        if peak is not None:
            counts[peak] = 100
            counts[peak + 1 :] = 10
        histograms.append(Histogram(counts, bin_width=0.1, t0_bin=anchor))
    return Run(
        run_number=1,
        histograms=histograms,
        metadata={"facility": source},
        grouping={
            "groups": {1: [1], 2: [2], 3: [3], 4: [4]},
            "forward_group": 1,
            "backward_group": 2,
            "t0_source": "missing" if missing else "file",
            "detector_t0_bins": [h.t0_bin for h in histograms],
        },
    )


def resolve(run, mode="auto_detect", *, pair=(1, 2), **policy):
    profile = profile_from_payload(run.grouping, "timing", profile_fingerprint_for_run(run))
    profile.forward_group, profile.backward_group = pair
    profile.t0_policy = T0Policy(mode=mode, **policy)
    return resolve_effective_grouping(profile, run)


@pytest.mark.parametrize("source", ["PSI", ""])
def test_explicit_auto_requires_each_selected_estimate(source):
    run = timing_run(source=source)
    with pytest.raises(InvalidGroupingError, match="t0|timing"):
        resolve(run)


def test_unknown_strategy_does_not_use_saved_result_as_an_override():
    run = timing_run(source="")
    with pytest.raises(InvalidGroupingError, match="source|strategy"):
        resolve(run, pair=(1, 3), strategy="prompt_peak")


@pytest.mark.parametrize("mode", ["from_file", "manual"])
def test_unresolved_fallback_never_reduces_placeholder_timing(mode):
    run = timing_run(missing=True)
    with pytest.raises(InvalidGroupingError, match="2"):
        resolve(run, mode, **({"offset_bins": 3} if mode == "manual" else {}))
    with pytest.raises(ValueError, match="t0|timing"):
        reduce_run(run, ReductionSettings())


def test_unused_failure_does_not_block_selected_own_anchors():
    run = timing_run()
    grouping = resolve(run, pair=(1, 3))
    assert grouping["effective_detector_t0_bins"][0] == 10
    assert grouping["effective_detector_t0_bins"][2] == 12
    assert grouping["effective_detector_t0_bins"][1] == 8
    assert grouping["t0_source"] == "file"
    assert grouping["effective_t0_sources"] == [
        "estimated",
        "unresolved",
        "estimated",
        "unresolved",
    ]
    grouped = group_forward_backward(run.histograms, grouping)
    assert grouped.common_t0 + grouped.counts.alignment.first_bin == 12


def test_equal_valued_estimate_keeps_effective_provenance():
    run = timing_run()
    run.histograms[0].t0_bin = 10
    run.histograms[2].t0_bin = 12
    grouping = resolve(run, pair=(1, 3))
    assert grouping["effective_detector_t0_bins"][0] == 10
    assert grouping["effective_t0_sources"][0] == "estimated"
    assert run.grouping["t0_source"] == "file"


def test_unknown_advisory_search_does_not_block_valid_file_or_manual():
    run = timing_run(source="")
    assert resolve(run, "from_file")["t0_bin"] == 8
    assert resolve(run, "manual", offset_bins=2)["t0_bin"] == 10


@pytest.mark.parametrize("mode", ["from_file", "manual"])
def test_mixed_file_and_missing_timing_uses_only_required_fallback(mode):
    run = timing_run()
    run.grouping["detector_t0_sources"] = ["file", "missing", "missing", "file"]
    grouping = resolve(run, mode, pair=(1, 3), **({"offset_bins": 2} if mode == "manual" else {}))
    expected = [6, 14] if mode == "manual" else [4, 12]
    assert [grouping["effective_detector_t0_bins"][i] for i in (0, 2)] == expected
    assert grouping["detector_t0_sources"] == ["file", "missing", "missing", "file"]
    assert grouping["effective_t0_sources"][0] == ("manual" if mode == "manual" else "file")
    assert grouping["effective_t0_sources"][2] == ("manual" if mode == "manual" else "estimated")


def test_explicit_strategy_resolves_unknown_source_at_public_api_boundary():
    from asymmetry.core.transform.t0 import resolve_selected_timing

    run = timing_run(source="", missing=True)
    grouping = dict(run.grouping, backward_group=3)
    resolve_selected_timing(run.histograms, grouping, run.metadata, pulsed=False)
    assert grouping["effective_detector_t0_bins"] == [10, 8, 12, 8]
    assert grouping["t0_search_strategy"] == "prompt_peak"
    assert grouping["t0_source"] == "missing"


@pytest.mark.parametrize("source", ["PSI", ""])
def test_legitimate_zero_and_integer_only_timing_need_no_search(source):
    run = timing_run(source=source)
    for hist in run.histograms:
        hist.t0_bin = 0
    grouping = resolve(run, "from_file")
    assert grouping["t0_bin"] == 0
    assert "t0_search_strategy" not in grouping
    assert grouping["effective_t0_sources"] == ["file"] * 4


def test_excluded_failure_and_spectator_anchor_do_not_change_selected_counts():
    run = timing_run()
    run.grouping["groups"] = {1: [1, 2], 2: [3], 3: [4]}
    run.grouping["excluded_detectors"] = [2]
    run.histograms[3].t0_bin = 1000
    grouping = resolve(run)
    grouped = group_forward_backward(run.histograms, grouping)
    np.testing.assert_array_equal(grouped.forward, run.histograms[0].counts[:38])
    np.testing.assert_array_equal(grouped.backward, run.histograms[2].counts[2:])
    assert grouped.counts.alignment.timing_sources == ((0, "estimated"), (2, "estimated"))
    assert grouped.counts.alignment.timing_strategy == "prompt_peak"


def test_independent_projection_resolves_its_selected_channels():
    from asymmetry.core.transform.projections import reduce_run_projections

    run = timing_run()
    run.grouping.update(
        groups={1: [1], 2: [3], 3: [4]},
        projections=[
            {"label": "first", "forward_group": 1, "backward_group": 2},
            {"label": "second", "forward_group": 2, "backward_group": 3},
        ],
        t0_policy={"mode": "auto_detect"},
    )
    results = reduce_run_projections(run, 0)
    for label, expected in [
        ("first", [(0, "estimated"), (2, "estimated")]),
        ("second", [(2, "estimated"), (3, "estimated")]),
    ]:
        assert results[label].metadata["grouped_alignment"]["timing_sources"] == tuple(expected)


def test_explicit_absolute_manual_table_can_supply_missing_calibration():
    run = timing_run(source="", missing=True)
    grouping = dict(run.grouping, effective_detector_t0_bins=[4, 8, 6, 8])
    grouped = group_forward_backward(run.histograms, grouping)
    assert grouped.counts.alignment.timing_sources == ((0, "manual"), (1, "manual"))


def test_manual_offset_cannot_borrow_a_base_from_another_period():
    from asymmetry.core.io.periods import select_period_histograms

    run = timing_run()
    green = timing_run(missing=True)
    run.grouping.update(
        period_histograms=[run.histograms, green.histograms], period_t0_sources=["file", "missing"]
    )
    grouping = resolve(run, "manual", offset_bins=2)
    histograms, selected = select_period_histograms(run.histograms, grouping, 1)
    with pytest.raises(ValueError, match="t0"):
        group_forward_backward(histograms, selected)


def test_green_auto_uses_its_own_raw_features_and_required_failures():
    run = timing_run()
    green = timing_run()
    run.grouping.update(
        period_histograms=[run.histograms, green.histograms], period_t0_sources=["file", "file"]
    )
    profile = profile_from_payload(run.grouping, "timing", profile_fingerprint_for_run(run))
    profile.t0_policy = T0Policy(mode="auto_detect")
    profile.backward_group = 3
    profile.period_mode = "green"
    green.histograms[0].counts[:] = 0
    with pytest.raises(ValueError, match="selected channels 1"):
        resolve_effective_grouping(profile, run)
    profile.period_mode = "red"
    assert resolve_effective_grouping(profile, run)["t0_bin"] == 12


def test_raw_search_cancels_between_detectors_without_mutating_counts():
    from asymmetry.core.transform.t0 import find_t0_for_run

    run = timing_run()
    original = [h.counts.copy() for h in run.histograms]
    calls = 0

    def cancel():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise InterruptedError("cancelled")

    with pytest.raises(InterruptedError, match="cancelled"):
        find_t0_for_run(run.histograms, run.metadata, cancel_check=cancel)
    assert calls == 2
    for hist, counts in zip(run.histograms, original, strict=True):
        np.testing.assert_array_equal(hist.counts, counts)


def test_returning_to_file_timing_clears_prior_effective_override_and_strategy():
    from asymmetry.core.transform.t0 import effective_detector_t0_bins, resolve_selected_timing

    run = timing_run()
    grouping = dict(run.grouping, backward_group=3)
    resolve_selected_timing(run.histograms, grouping, run.metadata, mode="auto_detect")
    assert effective_detector_t0_bins(run.histograms, grouping)[0] == 10
    resolve_selected_timing(run.histograms, grouping, run.metadata, mode="from_file")
    assert effective_detector_t0_bins(run.histograms, grouping)[0] == 4
    assert "effective_detector_t0_bins" not in grouping
    assert "t0_search_strategy" not in grouping
    assert grouping["effective_t0_sources"] == ["file"] * 4


@pytest.mark.parametrize("source", ["General laboratory", "Problem detector"])
def test_unrelated_metadata_words_cannot_establish_a_strategy(source):
    run = timing_run(source=source)
    with pytest.raises(ValueError, match="source|strategy"):
        resolve(run, pair=(1, 3))
