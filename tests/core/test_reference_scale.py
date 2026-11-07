"""Authored coefficients are independent of acquired exposure and DT units."""

import pytest

from asymmetry.core.io import (
    background_reference_for_grouping,
    resolve_background_reference,
)
from asymmetry.core.io.periods import select_period_grouping
from asymmetry.core.transform.reduce import corrected_grouped_counts
from asymmetry.core.transform.reference_scale import ReferenceExposure, ReferenceScaleRequest
from tests.io.test_background_reference_resolution import _dataset, _run


@pytest.mark.parametrize("sample,reference", [(200.0, 100.0), (1.0, 1.0)])
def test_manual_coefficient_overrides_frame_metadata(sample, reference):
    payload = {"scale_request": {"mode": "manual", "coefficient": 7.0}}
    outcome = ReferenceScaleRequest.from_payload(payload).resolve(
        ReferenceExposure(sample, "user"), ReferenceExposure(reference, "user")
    )
    assert outcome.coefficient == 7.0


@pytest.mark.parametrize(
    "sample,reference,expected,source",
    [
        (ReferenceExposure(200, "user"), ReferenceExposure(100, "user"), 2, "live_exposure"),
        (
            ReferenceExposure(1, "NeXus:good_frames"),
            ReferenceExposure(1, "NeXus:good_frames"),
            1,
            "live_exposure",
        ),
        (ReferenceExposure(), ReferenceExposure(100, "user"), 1, "exposure_unknown"),
        (
            ReferenceExposure(200, "user"),
            ReferenceExposure(origin="whole_run_deadtime"),
            1,
            "exposure_unknown",
        ),
        (
            ReferenceExposure(200, "user", "pulses"),
            ReferenceExposure(100, "user", "seconds"),
            1,
            "exposure_unknown",
        ),
    ],
)
def test_automatic_requires_compatible_evidenced_exposure(sample, reference, expected, source):
    result = ReferenceScaleRequest().resolve(sample, reference)
    assert result.coefficient == expected
    assert result.source == source
    assert (
        result.provenance()["uncertainty"] == "fixed scale; independent sample/reference variances"
    )
    if source == "exposure_unknown":
        assert result.summary() == "Exposure unknown; scale 1 assumed"


@pytest.mark.parametrize(
    "coefficient", [True, False, 0, -1, float("inf"), float("nan"), 10**310, "invalid", None]
)
def test_manual_request_rejects_invalid_authored_coefficients(coefficient):
    with pytest.raises(ValueError, match="coefficient must be finite and positive"):
        ReferenceScaleRequest.from_payload(
            {"scale_request": {"mode": "manual", "coefficient": coefficient}}
        )


def test_historical_snapshots_are_never_mixed_with_a_live_side():
    payload = {
        "historical_scale": {
            "good_frames_sample": 800,
            "good_frames_reference": 400,
            "exposure_origin": "unverified",
        }
    }
    for period in (1, 2):
        outcome = ReferenceScaleRequest.from_payload(payload).resolve(
            ReferenceExposure(period_number=period), ReferenceExposure(1000, "user")
        )
        assert outcome.coefficient == 1
        assert outcome.source == "exposure_unknown"


def test_manual_ignores_malformed_saved_metadata():
    payload = {
        "scale_request": {"mode": "manual", "coefficient": 7},
        "historical_scale": {"scale": "invalid", "good_frames_reference": "invalid"},
    }
    assert ReferenceScaleRequest.from_payload(payload).resolve().coefficient == 7


@pytest.mark.parametrize("mode,expected", [("automatic", 6), ("manual", 7)])
def test_selected_sample_period_adaptation_retains_authored_intent(mode, expected):
    reference = _run(42, good_frames=100, level=3)
    request = {"mode": mode} | ({"coefficient": 7} if mode == "manual" else {})
    payload = {"run_number": 42, "scale_request": request}
    resolved = resolve_background_reference(
        payload, sample_exposure=ReferenceExposure(200, "user"), datasets=[_dataset(reference)]
    )
    grouping = {
        "background_run": payload,
        "period_reference_exposure": [
            ReferenceExposure(200, "user").to_dict(),
            ReferenceExposure(600, "user", period_number=2).to_dict(),
        ],
    }
    adapted = background_reference_for_grouping(resolved, select_period_grouping(grouping, 1))
    assert adapted.scale == expected
    assert payload["scale_request"] == request


def test_unknown_exposure_is_an_applied_visible_assumption_and_keeps_reference_dt():
    reference = _run(42, good_frames=1, level=3)
    reference.grouping["reference_exposure"] = ReferenceExposure(
        origin="whole_run_deadtime"
    ).to_dict()
    resolved = resolve_background_reference({"run_number": 42}, datasets=[_dataset(reference)])
    assert resolved.deadtime_grouping({})["good_frames"] == 1
    sample = _run(41, good_frames=1, level=20)
    with pytest.warns(UserWarning, match="Exposure unknown; scale 1 assumed"):
        result = corrected_grouped_counts(
            histograms=sample.histograms,
            grouping={"background_mode": "reference_run"},
            forward_idx=[0],
            backward_idx=[1],
            use_deadtime=False,
            deadtime_mode="off",
            use_background=True,
            reference_resolver=lambda _: resolved,
        )
    assert result.corrections.background.status.value == "applied"
    assert result.corrections.background.warning == "Exposure unknown; scale 1 assumed"
    assert result.corrections.complete


def test_manual_negative_counts_retain_independent_supplied_variances():
    reference = _run(42, good_frames=100, level=3)
    sample = _run(41, good_frames=200, level=20)
    resolved = resolve_background_reference(
        {"run_number": 42, "scale_request": {"mode": "manual", "coefficient": 7}},
        sample_exposure=ReferenceExposure(200, "user"),
        datasets=[_dataset(reference)],
    )
    result = corrected_grouped_counts(
        histograms=sample.histograms,
        grouping={"background_mode": "reference_run"},
        forward_idx=[0],
        backward_idx=[1],
        use_deadtime=False,
        deadtime_mode="off",
        use_background=True,
        reference_resolver=lambda _: resolved,
    )
    assert (result.forward.counts == -1).all()
    assert (result.forward.variance == 167).all()  # 20 + 7² × 3, independent fixed-scale counts.


def test_combined_counts_cannot_inherit_first_contributor_exposure():
    from asymmetry.core.data.combine import combine_runs

    runs = [_run(number, good_frames=100, level=20) for number in (41, 42)]
    for run in runs:
        run.grouping.update(groups={1: [1], 2: [2]}, forward_group=1, backward_group=2)
        run.grouping["period_reference_exposure"] = [ReferenceExposure(100, "user").to_dict()]
    combined = combine_runs(runs, run_number=43)
    assert ReferenceExposure.from_grouping(combined.grouping).value is None
    assert "period_reference_exposure" not in combined.grouping
    assert ReferenceExposure.from_grouping(runs[0].grouping).value == 100


@pytest.mark.parametrize("operation", ["simulate", "degrade"])
def test_resampled_counts_do_not_inherit_measured_template_exposure(operation):
    from asymmetry.core.simulate import degrade_run, simulate_run
    from tests.io.test_simulate import _template, _zero_model

    template = _template()
    template.grouping["reference_exposure"] = ReferenceExposure(100, "user").to_dict()
    template.grouping["period_reference_exposure"] = [ReferenceExposure(100, "user").to_dict()]
    result = (
        simulate_run(template, _zero_model, total_events=1000, seed=7)
        if operation == "simulate"
        else degrade_run(template, 0.5, seed=7)
    )
    assert ReferenceExposure.from_grouping(result.grouping).value is None
    assert "period_reference_exposure" not in result.grouping
    assert ReferenceExposure.from_grouping(template.grouping).value == 100


@pytest.mark.parametrize(
    "obsolete",
    [{"scale": 7}, {"scale": "invalid"}, {"good_frames_sample": 200, "good_frames_reference": 100}],
)
def test_obsolete_api_scale_settings_require_explicit_intent(obsolete):
    with pytest.raises(ValueError, match="Use scale_request"):
        ReferenceScaleRequest.from_payload(obsolete)


@pytest.mark.parametrize("value", [True, False])
def test_boolean_exposure_is_unknown_and_cannot_supply_an_automatic_ratio(value):
    exposure = ReferenceExposure.from_dict({"value": value, "origin": "user"})
    assert exposure.value is None
    for sample, reference in (
        (exposure, ReferenceExposure(100, "user")),
        (ReferenceExposure(200, "user"), exposure),
    ):
        outcome = ReferenceScaleRequest().resolve(sample, reference)
        assert outcome.source == "exposure_unknown"
        assert outcome.summary() == "Exposure unknown; scale 1 assumed"


@pytest.mark.parametrize("value", [True, False])
def test_typed_scale_inputs_reject_boolean_numbers(value):
    with pytest.raises(ValueError, match="finite positive value"):
        ReferenceExposure(value, "user")
    with pytest.raises(ValueError, match="coefficient must be finite and positive"):
        ReferenceScaleRequest("manual", value)
