"""Requested corrections must have an explicit usable outcome before analysis."""

from copy import deepcopy

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram
from asymmetry.core.transform.reduce import corrected_grouped_counts
from asymmetry.core.transform.reference_scale import ReferenceExposure, ReferenceScaleRequest


@pytest.fixture
def histograms():
    return [Histogram(np.full(100, value), 0.016, t0_bin=20) for value in (100.0, 80.0)]


@pytest.mark.parametrize(
    "grouping,dt_mode,background,reason",
    [
        ({"dead_time_us": [0.01]}, "manual", False, "table"),
        ({"background_mode": "reference_run"}, "off", True, "reference"),
        ({"background_mode": "fixed"}, "off", True, "fixed"),
    ],
)
def test_unavailable_correction_blocks_final_but_labels_preview(
    histograms, grouping, dt_mode, background, reason
):
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
    actual = reduced.corrections.deadtime.to_dict()
    if actual["effective"] is not None:
        assert actual["effective"].pop("scope")["raw_bin_ranges"] == {
            "0": [[0, 99]],
            "1": [[0, 99]],
        }
    assert outcome.to_dict() == actual
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

    reference = BackgroundReference(
        deepcopy(histograms),
        ReferenceScaleRequest("manual", 2.0).resolve(
            ReferenceExposure(1000, "user"), ReferenceExposure(500, "user")
        ),
        42,
        "reference.nxs",
        500.0,
    )
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
        "content_identity": None,
        "run_number": 42,
        "file_name": "reference.nxs",
        "source_file": "reference.nxs",
        "period_number": 1,
        "scale": 2.0,
        "scale_source": "manual",
        "scaling": reference.scaling.provenance(),
        "scale_summary": "Manual: 2",
        "deadtime_normaliser": 500.0,
        "timing": {"t0_policy": {"mode": "from_file"}},
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
    with pytest.raises(ValueError, match="reference-run subtraction is unsupported"):
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
        ReferenceScaleRequest("manual", 2.0).resolve(),
        deadtime_normaliser=5.0,
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


@pytest.mark.parametrize("use_background", [False, True])
def test_count_fit_accepts_skipped_file_deadtime_and_keeps_background(histograms, use_background):
    from asymmetry.core.data.dataset import MuonDataset, Run
    from asymmetry.core.fitting.grouped_time_domain import build_count_group

    run = Run(
        histograms=histograms,
        grouping={
            "groups": {1: [1], 2: [2]},
            "deadtime_correction": True,
            "deadtime_mode": "file",
            "background_correction": use_background,
            "background_mode": "fixed",
            "background_fixed_values": [1.0, 2.0],
        },
    )
    dataset = MuonDataset(np.array([]), np.array([]), np.array([]), run=run)
    with pytest.warns(UserWarning, match="File deadtime correction not applied"):
        group = build_count_group(dataset, 1)
    assert np.all(np.isfinite(group.counts))


@pytest.mark.parametrize("table", [[], [0.01], [0.01, None], [0.01, float("nan")]])
def test_file_fallback_retains_variance_and_enabled_background(table):
    import json
    import pickle

    from asymmetry.core.transform.corrections import CorrectionOutcomes

    histograms = [
        Histogram(np.array([20.0, 30.0, 40.0]), 0.1, count_variance=np.array([7.0, 8.0, 9.0])),
        Histogram(np.array([10.0, 15.0, 20.0]), 0.1, count_variance=np.array([3.0, 4.0, 5.0])),
    ]
    grouping = {
        "dead_time_us": table,
        "background_mode": "fixed",
        "background_fixed_values": [1.0, 2.0],
    }
    options = dict(
        histograms=histograms,
        grouping=grouping,
        forward_idx=[0],
        backward_idx=[1],
        deadtime_mode="file",
        use_background=True,
    )
    with pytest.warns(UserWarning, match="Continuing uncorrected"):
        actual = corrected_grouped_counts(**options, use_deadtime=True)
    expected = corrected_grouped_counts(**options, use_deadtime=False)
    assert actual.corrections.deadtime.status == "skipped"
    assert actual.corrections.background == expected.corrections.background
    assert actual.corrections.background.status == "applied"
    for a, e in [(actual.forward, expected.forward), (actual.backward, expected.backward)]:
        np.testing.assert_array_equal(a.counts, e.counts)
        np.testing.assert_array_equal(a.independent_variance, e.independent_variance)
    assert actual.alignment == expected.alignment
    assert (
        CorrectionOutcomes.from_dict(json.loads(json.dumps(actual.corrections.to_dict())))
        == actual.corrections
    )
    assert pickle.loads(pickle.dumps(actual.corrections)) == actual.corrections


@pytest.mark.parametrize("requested", ["manual", "estimate"])
def test_authored_coefficients_cannot_be_relabeled_file(histograms, requested):
    from asymmetry.core.data.dataset import Run
    from asymmetry.core.project.profiles import (
        DeadtimePolicy,
        ProfileFingerprint,
        profile_from_payload,
        resolve_effective_grouping,
    )

    grouping = {
        "groups": {1: [1], 2: [2]},
        "deadtime_mode": requested,
        "deadtime_method": requested,
        "dead_time_us": [0.0, 0.0],
        "deadtime_correction": True,
    }
    profile = profile_from_payload(grouping, "test", ProfileFingerprint("generic_layout", 2))
    profile.deadtime_policy = DeadtimePolicy(mode="from_file")
    effective = resolve_effective_grouping(profile, Run(histograms=histograms, grouping=grouping))
    assert effective["dead_time_us"] == []
    with pytest.warns(UserWarning, match="File deadtime correction not applied"):
        result = corrected_grouped_counts(
            histograms=histograms,
            grouping=effective,
            forward_idx=[0],
            backward_idx=[1],
            use_deadtime=True,
            deadtime_mode="file",
            use_background=False,
        )
    assert result.corrections.deadtime.status == "skipped"
    assert grouping["dead_time_us"] == [0.0, 0.0]


def test_skipped_file_does_not_erase_independent_failures(histograms):
    options = dict(
        histograms=histograms,
        grouping={"background_mode": "reference_run"},
        forward_idx=[0],
        backward_idx=[1],
        use_deadtime=True,
        deadtime_mode="file",
        use_background=True,
    )
    with pytest.warns(UserWarning, match="File deadtime correction not applied"):
        with pytest.raises(ValueError, match="reference"):
            corrected_grouped_counts(**options)
    options.update(forward_idx=[0], backward_idx=[0], use_background=False)
    with pytest.raises(ValueError, match="shared acquired"):
        corrected_grouped_counts(**options)


@pytest.mark.parametrize("missing", [0, 1])
@pytest.mark.parametrize("period_mode", ["green_minus_red", "green_plus_red"])
def test_missing_period_skips_the_entire_combined_deadtime_operation(missing, period_mode):
    from asymmetry.core.transform.rebin import BinningSpec
    from asymmetry.core.transform.reduce import reduce_period_asymmetry

    histograms = [
        Histogram(np.array([100000.0, 90000.0, 80000.0]), 0.1),
        Histogram(np.array([25000.0, 23000.0, 21000.0]), 0.1),
    ]
    tables = [[0.02, 0.02], [0.02, 0.02]]
    tables[missing] = []
    grouping = {
        "groups": {1: [1], 2: [2]},
        "period_histograms": [histograms, histograms],
        "period_dead_time_us": tables,
        "period_good_frames": [100000.0, 100000.0],
    }
    options = dict(
        histograms=histograms,
        grouping=grouping,
        period_mode=period_mode,
        binning=BinningSpec(),
        forward_idx=[0],
        backward_idx=[1],
        alpha=1.0,
        deadtime_mode="file",
        use_background=False,
    )
    expected = reduce_period_asymmetry(**options, use_deadtime=False)
    with pytest.warns(UserWarning, match="File deadtime correction not applied"):
        actual = reduce_period_asymmetry(**options, use_deadtime=True)
    for field in ("time", "asymmetry", "error"):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
    assert all(record.deadtime.status == "skipped" for record in actual.corrections.values())
    assert tables[1 - missing] == [0.02, 0.02]


@pytest.mark.parametrize("method", [None, "", "file"])
def test_manual_mode_cannot_supply_file_coefficients(method):
    from asymmetry.core.transform.deadtime import resolve_deadtime

    grouping = {"deadtime_mode": "manual", "dead_time_us": [0.02, 0.02], "good_frames": 100000.0}
    if method is not None:
        grouping["deadtime_method"] = method
    assert resolve_deadtime(grouping, 2, True, "file").status == "skipped"


def test_missing_reference_file_table_skips_sample_and_reference():
    from asymmetry.core.transform.corrections import BackgroundReference

    sample = [Histogram(np.array([100.0, 80.0, 60.0]), 1.0) for _ in range(2)]
    reference = BackgroundReference(
        [Histogram(np.array([20.0, 10.0, 5.0]), 1.0) for _ in range(2)],
        ReferenceScaleRequest("manual", 1.0).resolve(),
        deadtime_normaliser=100.0,
    )
    options = dict(
        histograms=sample,
        grouping={
            "dead_time_us": [0.1, 0.1],
            "good_frames": 100.0,
            "background_mode": "reference_run",
        },
        forward_idx=[0],
        backward_idx=[1],
        deadtime_mode="file",
        use_background=True,
        reference_resolver=lambda _: reference,
    )
    expected = corrected_grouped_counts(**options, use_deadtime=False)
    with pytest.warns(UserWarning, match="File deadtime correction not applied"):
        actual = corrected_grouped_counts(**options, use_deadtime=True)
    np.testing.assert_array_equal(actual.forward.counts, expected.forward.counts)
    np.testing.assert_array_equal(
        actual.forward.independent_variance, expected.forward.independent_variance
    )
    assert actual.corrections.deadtime.status == "skipped"
    assert actual.corrections.background.status == "applied"


def test_complete_period_file_tables_keep_their_own_coefficients_and_exposure():
    from asymmetry.core.transform.rebin import BinningSpec
    from asymmetry.core.transform.reduce import reduce_period_asymmetry

    raw_f, raw_b = np.array([100000.0, 90000.0, 80000.0]), np.array([25000.0, 23000.0, 21000.0])
    histograms = [Histogram(raw_f, 0.1), Histogram(raw_b, 0.1)]
    grouping = {
        "period_histograms": [histograms, histograms],
        "period_dead_time_us": [[0.02, 0.02], [-0.01, 0.03]],
        "period_good_frames": [100000.0, 200000.0],
    }
    result = reduce_period_asymmetry(
        histograms,
        grouping,
        "green_minus_red",
        binning=BinningSpec(),
        forward_idx=[0],
        backward_idx=[1],
        alpha=1.0,
        use_deadtime=True,
        deadtime_mode="file",
        use_background=False,
    )
    expected, variances = [], []
    for (tau_f, tau_b), frames in zip(
        grouping["period_dead_time_us"], grouping["period_good_frames"], strict=True
    ):
        f = raw_f / (1 - raw_f * tau_f / (0.1 * frames))
        b = raw_b / (1 - raw_b * tau_b / (0.1 * frames))
        expected.append(100 * (f - b) / (f + b))
        variances.append(
            100**2 * ((2 * b / (f + b) ** 2) ** 2 * raw_f + (2 * f / (f + b) ** 2) ** 2 * raw_b)
        )
    np.testing.assert_allclose(result.asymmetry, expected[1] - expected[0])
    np.testing.assert_allclose(result.error, np.sqrt(variances[0] + variances[1]))
    assert all(record.deadtime.status == "applied" for record in result.corrections.values())


@pytest.mark.parametrize("failure", ["exposure", "background"])
def test_combined_file_fallback_preserves_independent_fatal_outcomes(failure):
    from asymmetry.core.transform.rebin import BinningSpec
    from asymmetry.core.transform.reduce import reduce_period_asymmetry

    histograms = [Histogram(np.full(3, 20.0), 1.0) for _ in range(2)]
    grouping = {
        "period_histograms": [histograms, histograms],
        "period_dead_time_us": [[0.02, 0.02], []],
        "period_good_frames": [None if failure == "exposure" else 1000.0, 1000.0],
        "background_mode": "reference_run",
    }
    with pytest.raises(ValueError, match="good frames" if failure == "exposure" else "reference"):
        reduce_period_asymmetry(
            histograms,
            grouping,
            "green_minus_red",
            binning=BinningSpec(),
            forward_idx=[0],
            backward_idx=[1],
            alpha=1.0,
            use_deadtime=True,
            deadtime_mode="file",
            use_background=failure == "background",
        )


@pytest.mark.parametrize("missing", [0, 1])
def test_combined_missing_sample_table_also_skips_complete_reference(missing):
    from asymmetry.core.transform.corrections import BackgroundReference
    from asymmetry.core.transform.rebin import BinningSpec
    from asymmetry.core.transform.reduce import reduce_period_asymmetry

    histograms = [
        Histogram(np.array([100.0, 80.0, 60.0]), 1.0),
        Histogram(np.array([70.0, 60.0, 50.0]), 1.0),
    ]
    reference = BackgroundReference(
        [Histogram(np.array([20.0, 10.0, 5.0]), 1.0) for _ in range(2)],
        ReferenceScaleRequest("manual", 1.0).resolve(),
        deadtime_normaliser=100.0,
        file_dead_time_us=(0.1, 0.1),
    )
    tables = [[0.1, 0.1], [0.1, 0.1]]
    tables[missing] = []
    grouping = {
        "period_histograms": [histograms, histograms],
        "period_dead_time_us": tables,
        "period_good_frames": [100.0, 100.0],
        "background_mode": "reference_run",
    }
    options = dict(
        histograms=histograms,
        grouping=grouping,
        period_mode="green_plus_red",
        binning=BinningSpec(),
        forward_idx=[0],
        backward_idx=[1],
        alpha=1.0,
        deadtime_mode="file",
        use_background=True,
        reference_resolver=lambda _: reference,
    )
    expected = reduce_period_asymmetry(**options, use_deadtime=False)
    with pytest.warns(UserWarning, match="File deadtime correction not applied"):
        actual = reduce_period_asymmetry(**options, use_deadtime=True)
    for field in ("time", "asymmetry", "error"):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
    for outcome in actual.corrections.values():
        assert outcome.deadtime.status == "skipped"
        assert outcome.background.effective["reference"]["deadtime"]["status"] == "skipped"
