"""Explicit workflow calibration uses the numerical and record owners."""

from dataclasses import replace

import numpy as np
import pytest

from asymmetry.core.fitting.tf_calibration import (
    TFCalibrationRequest,
    TFEnvelope,
    fit_tf_calibration,
)
from asymmetry.core.workflow.calibration import CalibrationOptions, calibrate_run
from asymmetry.core.workflow.reduction import (
    ReductionSettings,
    reduce_run,
    resolve_reduction_grouping,
)
from tests.core.test_tf_calibration_fit import _raw_tf


@pytest.mark.parametrize("fit_beta", [False, True])
def test_explicit_exponential_matches_direct_core_and_applies_both(fit_beta):
    data = _raw_tf(beta=1.13 if fit_beta else 1.0)
    settings = ReductionSettings()
    outcome = calibrate_run(
        data, settings, CalibrationOptions("tf_calibration_fit", fit_beta, TFEnvelope.EXPONENTIAL)
    )
    direct = fit_tf_calibration(data, TFCalibrationRequest(1, 2, fit_beta=fit_beta))
    assert outcome.record.result == direct
    applied = replace(settings, alpha=outcome.record)
    grouping = resolve_reduction_grouping(data.run, applied)
    assert grouping["alpha"] == direct.alpha
    assert grouping.get("beta", 1.0) == direct.beta
    replay = ReductionSettings.from_dict(applied.to_dict())
    assert replay.calibration == outcome.record
    assert np.array_equal(
        reduce_run(data.run, replay).asymmetry, reduce_run(data.run, applied).asymmetry
    )


def test_quick_free_beta_is_rejected_at_input():
    with pytest.raises(ValueError, match="beta"):
        CalibrationOptions("diamagnetic", True)


@pytest.mark.parametrize("envelope", list(TFEnvelope))
@pytest.mark.parametrize("fit_beta", [False, True])
def test_envelopes_are_actually_constrained_and_truth_retained(envelope, fit_beta):
    from asymmetry.core.fitting.count_results import CountFitParameter as P
    from tests.core.test_tf_calibration_selection import _control

    data = _control(envelope.value, beta=0.8 if fit_beta else 1)
    options = CalibrationOptions(
        "tf_calibration_fit", fit_beta, envelope, parameters=(P("frequency", 1.3, fixed=True),)
    )
    outcome = calibrate_run(data, ReductionSettings(), options)
    result = outcome.record.result
    direct = fit_tf_calibration(data, result.request, envelope=envelope)
    assert result == direct
    assert result.ok, result.message
    assert result.envelope is envelope
    assert result.alpha == pytest.approx(1.25, abs=2e-4)
    assert result.beta == pytest.approx(0.8 if fit_beta else 1, abs=2e-4)
    names = {p.name for p in result.fit.parameters}
    assert ("stretch" in names) == (envelope is TFEnvelope.STRETCHED)
    assert ("sigma" in names) == (envelope is TFEnvelope.GAUSSIAN)
    assert result.beta_fixed == (not fit_beta)
    if not fit_beta:
        assert result.beta_error is None
        assert result.alpha_beta_correlation is None


@pytest.mark.parametrize("method", ["ratio", "diamagnetic", "general"])
def test_quick_uses_shared_result_and_source_timing(method):
    from asymmetry.core.transform.asymmetry import estimate_alpha_detailed
    from asymmetry.core.transform.fb_counts import native_count_time

    data = _raw_tf(beta=1)
    outcome = calibrate_run(
        data, ReductionSettings(t0_offset_bins=2, t_good_offset_bins=5), CalibrationOptions(method)
    )
    from asymmetry.core.transform.rebin import BinningSpec
    from asymmetry.core.transform.reduce import corrected_period_counts

    settings = ReductionSettings(t0_offset_bins=2, t_good_offset_bins=5)
    grouping = resolve_reduction_grouping(data.run, settings)
    counts = next(
        iter(
            corrected_period_counts(
                data.run.histograms,
                grouping,
                "red",
                binning=BinningSpec(),
                forward_idx=[0],
                backward_idx=[1],
                use_deadtime=False,
                deadtime_mode="off",
                use_background=False,
            ).values()
        )
    )
    expected = estimate_alpha_detailed(
        counts.forward,
        counts.backward,
        method=method,
        time_us=native_count_time(
            counts.forward.counts.size, counts.bin_width_us, counts.common_t0_bin, counts.t0_time_us
        ),
        first_good_bin=counts.window.first,
        last_good_bin=counts.window.last,
    )
    assert outcome.result.estimate == expected
    assert outcome.record is None
    assert (expected.alpha_error is not None) == (method == "ratio")
    assert outcome.to_dict()["beta_error"] is None


def test_selection_matches_core_without_comparing_independently_captured_ids():
    from asymmetry.core.fitting.tf_calibration import select_tf_calibration

    data = _raw_tf(beta=0.8)
    options = CalibrationOptions("tf_calibration_fit", True, frequency_mhz=1.3)
    outcome = calibrate_run(data, ReductionSettings(), options)
    direct = select_tf_calibration(
        data, TFCalibrationRequest(1, 2, fit_beta=True, frequency_mhz=1.3)
    )
    assert outcome.record.result == direct
    assert outcome.to_dict()["model"] == direct.selected.model


@pytest.mark.parametrize("deadtime", ["off", "from_file"])
@pytest.mark.parametrize("background", ["none", "range", "tail_fit"])
def test_source_corrections_and_native_window_match_direct(deadtime, background):
    from asymmetry.core.project.profiles import (
        AlphaPolicy,
        BackgroundPolicy,
        DeadtimePolicy,
        GoodWindowPolicy,
        T0Policy,
        profile_fingerprint_for_run,
        profile_from_payload,
        resolve_effective_grouping,
    )
    from tests.core.test_tf_deadtime_calibration import _deadtime_tf

    data = _deadtime_tf(fit_beta=True, mode="from_file")
    settings = ReductionSettings(
        deadtime=deadtime,
        background=background,
        background_range=(510, 550) if background == "range" else None,
        t0_offset_bins=1,
        t_good_offset_bins=4,
        rebin=3,
        t_min=0.5,
        t_max=5,
    )
    options = CalibrationOptions(
        "tf_calibration_fit", True, TFEnvelope.EXPONENTIAL, frequency_mhz=1.3
    )
    profile = profile_from_payload(
        data.run.grouping, "direct", profile_fingerprint_for_run(data.run)
    )
    profile.alpha_policy = AlphaPolicy(value=1)
    profile.deadtime_policy = DeadtimePolicy(mode=deadtime)
    profile.background_policy = BackgroundPolicy(
        mode=background, details={"background_range": [510, 550]} if background == "range" else {}
    )
    profile.t0_policy = T0Policy(mode="manual", offset_bins=1)
    profile.good_window_policy = GoodWindowPolicy(mode="manual", first_offset_bins=4)
    prepared = replace(
        data,
        run=replace(
            data.run,
            grouping=dict(data.run.grouping) | resolve_effective_grouping(profile, data.run),
        ),
    )
    outcome = calibrate_run(data, settings, options)
    direct = fit_tf_calibration(
        prepared, TFCalibrationRequest(1, 2, fit_beta=True, frequency_mhz=1.3, t_min=0.5, t_max=5)
    )
    assert outcome.record.result == direct
    assert direct.fit.domains[0].deadtime.status == (
        "applied" if deadtime == "from_file" else "off"
    )


def test_requested_missing_deadtime_is_skipped_and_retained():
    data = _raw_tf(beta=1)
    with pytest.warns(UserWarning, match="not applied"):
        outcome = calibrate_run(
            data,
            ReductionSettings(deadtime="from_file"),
            CalibrationOptions("tf_calibration_fit", envelope=TFEnvelope.EXPONENTIAL),
        )
    assert outcome.record.result.fit.domains[0].deadtime.status == "skipped"


def test_cancellation_reaches_numerical_work(monkeypatch):
    from asymmetry.core.fitting import tf_calibration as tf
    from asymmetry.core.fitting.engine import FitCancelledError

    original = tf._fit_fb_count_groups
    entered = False

    def cancel():
        return entered

    def work(*args, **kwargs):
        nonlocal entered
        entered = True
        return original(*args, **kwargs)

    monkeypatch.setattr(tf, "_fit_fb_count_groups", work)
    with pytest.raises(FitCancelledError):
        calibrate_run(
            _raw_tf(beta=1),
            ReductionSettings(),
            CalibrationOptions("tf_calibration_fit"),
            cancel_callback=cancel,
        )
    assert entered


def test_captured_record_target_reduction_matches_direct_profile():
    from asymmetry.core.project.calibration import apply_calibration
    from asymmetry.core.project.profiles import (
        profile_fingerprint_for_run,
        profile_from_payload,
        resolve_effective_grouping,
    )
    from asymmetry.core.transform.rebin import BinningSpec
    from asymmetry.core.transform.reduce import reduce_grouped_asymmetry

    data = _raw_tf(beta=0.8)
    outcome = calibrate_run(
        data,
        ReductionSettings(),
        CalibrationOptions("tf_calibration_fit", True, TFEnvelope.EXPONENTIAL, frequency_mhz=1.3),
    )
    target = _raw_tf(beta=0.8, seed=51).run
    profile = profile_from_payload(target.grouping, "direct", profile_fingerprint_for_run(target))
    apply_calibration(profile, outcome.record)
    grouping = resolve_effective_grouping(profile, target)
    direct = reduce_grouped_asymmetry(
        histograms=target.histograms,
        grouping=grouping,
        forward_idx=[0],
        backward_idx=[1],
        alpha=profile.alpha_policy.value,
        beta=profile.beta,
        binning=BinningSpec(),
        use_deadtime=False,
        deadtime_mode="off",
        use_background=False,
    )
    routed = reduce_run(target, ReductionSettings(alpha=outcome.record))
    np.testing.assert_array_equal(routed.time, direct.time)
    np.testing.assert_array_equal(routed.asymmetry, direct.asymmetry)
    np.testing.assert_array_equal(routed.error, direct.error)
    wrong = reduce_run(target, ReductionSettings(alpha=outcome.alpha))
    assert not np.allclose(routed.asymmetry, wrong.asymmetry)


@pytest.mark.parametrize("captured", [False, True])
def test_historical_target_beta_cannot_override_fixed_one(captured):
    source = _raw_tf(beta=1)
    settings = ReductionSettings(alpha=1.25, alpha_source="user")
    if captured:
        record = calibrate_run(
            source,
            ReductionSettings(),
            CalibrationOptions(
                "tf_calibration_fit", envelope=TFEnvelope.EXPONENTIAL, frequency_mhz=1.3
            ),
        ).record
        settings = replace(settings, alpha=record)
    target = _raw_tf(beta=1).run
    baseline = reduce_run(target, settings)
    target.grouping.update(
        beta=0.7, beta_method="count_fit", beta_error=0.02, beta_reference_run=42
    )
    actual = reduce_run(target, settings)
    np.testing.assert_array_equal(actual.asymmetry, baseline.asymmetry)
    np.testing.assert_array_equal(actual.error, baseline.error)


def test_omitted_explicit_options_select_tf_but_none_keeps_automatic_ratio():
    data = _raw_tf(beta=1)
    fresh = calibrate_run(data, ReductionSettings())
    assert fresh.method == "tf_calibration_fit"
    assert fresh.record is not None
    automatic = calibrate_run(data, ReductionSettings(), CalibrationOptions())
    assert automatic.method == "per_run_estimate"


def test_resolved_source_grouping_is_not_replaced_by_cli_defaults():
    from asymmetry.core.workflow.calibration import calibrate_source

    data = _raw_tf(beta=1)
    grouping = {
        **data.run.grouping,
        "background_mode": "fixed",
        "background_correction": True,
        "background_fixed_values": [2.0, 3.0],
    }
    source = replace(data, run=replace(data.run, grouping=grouping))
    outcome = calibrate_source(source, CalibrationOptions("ratio"))
    assert outcome.source.corrections.background.requested == "fixed"


def test_zero_forward_ratio_is_not_a_usable_calibration():
    data = _raw_tf(beta=1)
    data.run.histograms[0].counts[:] = 0
    outcome = calibrate_run(data, ReductionSettings(), CalibrationOptions("ratio"))
    assert outcome.alpha is None
    assert not outcome.to_dict()["ok"]
    assert outcome.to_dict()["message"]


@pytest.mark.parametrize("entry", ["source", "workflow"])
def test_joint_beta_rejects_projection_layout_at_shared_boundary(entry):
    from asymmetry.core.workflow.calibration import calibrate_source
    from docs.screenshots.data.calibration import calibration_control

    data = calibration_control(vector=True)
    options = CalibrationOptions("tf_calibration_fit", True, TFEnvelope.EXPONENTIAL)
    with pytest.raises(ValueError, match="single.*pair"):
        if entry == "source":
            calibrate_source(data, options)
        else:
            calibrate_run(data, ReductionSettings(), options)
