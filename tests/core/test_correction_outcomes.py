"""Requested corrections must have an explicit usable outcome before analysis."""

from copy import deepcopy

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram
from asymmetry.core.transform.reduce import corrected_grouped_counts


@pytest.fixture
def histograms():
    return [Histogram(np.full(100, value), 0.016, t0_bin=20) for value in (100.0, 80.0)]


@pytest.mark.parametrize(
    "grouping,dt_mode,background,reason",
    [
        ({}, "from_file", False, "table"),
        ({"dead_time_us": [0.01]}, "manual", False, "table"),
        ({"dead_time_us": [0.0, 0.0]}, "from_file", False, "non-zero"),
        ({"dead_time_us": [0.01, 0.01], "deadtime_method": "manual"}, "from_file", False, "file"),
        ({"background_mode": "reference_run"}, "off", True, "reference"),
        ({"background_mode": "fixed"}, "off", True, "fixed"),
        ({"background_mode": "tail_fit"}, "off", True, "counts"),
    ],
)
def test_unavailable_correction_blocks_final_but_labels_preview(
    histograms, grouping, dt_mode, background, reason
):
    if grouping.get("background_mode") == "tail_fit":
        histograms = [Histogram(np.zeros(100), 0.016, t0_bin=20) for _ in range(2)]
    options = dict(
        histograms=histograms,
        grouping=grouping,
        forward_idx=[0],
        backward_idx=[1],
        use_deadtime=dt_mode != "off",
        deadtime_mode=dt_mode,
        use_background=background,
    )
    with pytest.raises(ValueError, match=reason):
        corrected_grouped_counts(**options)
    preview = corrected_grouped_counts(**options, allow_incomplete=True)
    assert not preview.corrections.complete
    outcome = preview.corrections.background if background else preview.corrections.deadtime
    assert outcome.status in {"unavailable", "failed"}
    assert reason in outcome.reason.lower()


@pytest.mark.parametrize(
    "mode,table,expected",
    [
        ("off", None, "off"),
        ("manual", [0.0, 0.0], "applied"),
        ("estimate", [0.0, 0.0], "applied"),
        ("from_file", [0.01, 0.02], "applied"),
    ],
)
def test_prepare_and_reduction_agree_without_mutation(histograms, mode, table, expected):
    from asymmetry.core.transform.deadtime import prepare_histograms_with_deadtime

    grouping = {"good_frames": 1e6, "deadtime_mode": mode}
    if table is not None:
        grouping["dead_time_us"] = table
    before = deepcopy(grouping)
    prepared, outcome = prepare_histograms_with_deadtime(
        histograms, grouping, mode != "off", mode=mode
    )
    reduced = corrected_grouped_counts(
        histograms=histograms,
        grouping=grouping,
        forward_idx=[0],
        backward_idx=[1],
        use_deadtime=mode != "off",
        deadtime_mode=mode,
        use_background=False,
    )
    assert outcome == reduced.corrections.deadtime
    assert outcome.status == expected
    assert reduced.corrections.complete
    np.testing.assert_array_equal(reduced.forward.counts, prepared[0].counts)
    assert grouping == before


@pytest.mark.parametrize(
    "requested,status,reason,effective,uncertainty",
    [
        ("off", "applied", None, {}, "known"),
        ("manual", "off", None, None, None),
        ("manual", "unavailable", None, None, None),
        ("manual", "applied", "failed", {}, "known"),
        ("manual", "applied", None, None, "known"),
        ("manual", "applied", None, {}, None),
        ("manual", "failed", "missing", {}, "known"),
        ("manual", "applied", None, {"tau_us": [float("nan")]}, "known"),
    ],
)
def test_outcome_invariants_reject_inconsistent_records(
    requested, status, reason, effective, uncertainty
):
    from asymmetry.core.transform.corrections import CorrectionOutcome

    with pytest.raises(ValueError):
        CorrectionOutcome("deadtime", requested, status, reason, effective, uncertainty)


def test_outcomes_round_trip_and_own_nested_provenance():
    import json

    from asymmetry.core.transform.corrections import CorrectionOutcome, CorrectionOutcomes

    effective = {"tau_us": [0.01, 0.02], "good_frames": 1e6, "source": "file"}
    outcomes = CorrectionOutcomes(
        deadtime=CorrectionOutcome(
            "deadtime",
            "from_file",
            "applied",
            effective=effective,
            uncertainty="Poisson approximation",
        )
    )
    effective["tau_us"][0] = 999
    assert outcomes.deadtime.effective["tau_us"][0] == 0.01
    saved = json.loads(json.dumps(outcomes.to_dict()))
    assert CorrectionOutcomes.from_dict(saved) == outcomes
    saved["deadtime"]["effective"]["tau_us"][0] = 2
    assert outcomes.deadtime.effective["tau_us"][0] == 0.01


def test_reference_provenance_records_scale_and_public_identity(histograms):
    from asymmetry.core.transform.corrections import BackgroundReference

    reference = BackgroundReference(histograms, 2.0, 42, "reference.nxs", 1000.0, 500.0)
    result = corrected_grouped_counts(
        histograms=histograms,
        grouping={"background_mode": "reference_run"},
        forward_idx=[0],
        backward_idx=[1],
        use_deadtime=False,
        deadtime_mode="off",
        use_background=True,
        reference_resolver=lambda _: reference,
    )
    assert result.corrections.background.effective["reference"] == {
        "run_number": 42,
        "file_name": "reference.nxs",
        "scale": 2.0,
        "scale_source": "explicit_scale",
        "good_frames_sample": 1000.0,
        "good_frames_reference": 500.0,
    }


def test_count_fit_refuses_an_unresolved_requested_background(histograms):
    from asymmetry.core.data.dataset import MuonDataset, Run
    from asymmetry.core.fitting.count_domain import fit_single_histogram
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    run = Run(
        histograms=histograms,
        grouping={
            "groups": {1: [1], 2: [2]},
            "background_correction": True,
            "background_mode": "reference_run",
        },
    )
    dataset = MuonDataset(np.array([]), np.array([]), np.array([]), run=run)
    with pytest.raises(ValueError, match="reference run is not loaded"):
        fit_single_histogram(
            dataset,
            1,
            lambda t: np.zeros_like(t),
            ParameterSet([Parameter("N0", 100), Parameter("background", 0, fixed=True)]),
        )


def test_reference_deadtime_uses_its_own_frame_exposure():
    from asymmetry.core.transform.corrections import BackgroundReference

    sample = [Histogram(np.array([50.0]), 1.0) for _ in range(2)]
    reference = BackgroundReference(
        [Histogram(np.array([20 / 1.4]), 1.0) for _ in range(2)],
        2.0,
        good_frames_sample=10.0,
        good_frames_reference=5.0,
    )
    result = corrected_grouped_counts(
        histograms=sample,
        grouping={
            "dead_time_us": [0.1, 0.1],
            "good_frames": 10.0,
            "background_mode": "reference_run",
        },
        forward_idx=[0],
        backward_idx=[1],
        use_deadtime=True,
        deadtime_mode="manual",
        use_background=True,
        reference_resolver=lambda _: reference,
    )
    np.testing.assert_allclose(result.forward.counts, [60.0], rtol=1e-12)
    assert result.corrections.complete


def test_disabled_request_ignores_unused_invalid_deadtime_settings(histograms):
    from asymmetry.core.transform.deadtime import prepare_histograms_with_deadtime

    prepared, outcome = prepare_histograms_with_deadtime(
        histograms, {"deadtime_mode": "unused invalid mode", "dead_time_us": [float("nan")]}, False
    )
    assert outcome.status == "off"
    np.testing.assert_array_equal(prepared[0].counts, histograms[0].counts)


def test_file_source_label_is_normalised_without_mutating_the_request():
    from asymmetry.core.transform.deadtime import resolve_deadtime

    grouping = {"dead_time_us": [0.01], "good_frames": 1e6, "deadtime_method": " File "}
    assert resolve_deadtime(grouping, 1, True, "file").status == "applied"
    assert grouping["deadtime_method"] == " File "
