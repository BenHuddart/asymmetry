"""Applied TF records survive the real project file boundary."""

from dataclasses import replace

import pytest

from asymmetry.core.fitting.count_results import CountFitParameter, JointCountFit
from asymmetry.core.fitting.likelihood import Objective
from asymmetry.core.fitting.tf_calibration import (
    TFCalibrationRequest,
    TFCalibrationResult,
    TFCalibrationSelection,
    TFEnvelope,
)
from asymmetry.core.project.calibration import CalibrationRecord, apply_calibration
from asymmetry.core.project.profiles import AlphaPolicy, GroupingProfile, ProfileFingerprint
from asymmetry.core.project.schema import CURRENT_SCHEMA_VERSION, load_project, save_project


def candidate(envelope=TFEnvelope.EXPONENTIAL, *, fit_beta=True):
    parameters = (
        CountFitParameter("alpha", 1.2, min=0.01),
        CountFitParameter("beta", 1.1 if fit_beta else 1.0, fixed=not fit_beta),
    )
    order = ("alpha", "beta") if fit_beta else ("alpha",)
    covariance = ((0.04, 0.01), (0.01, 0.09)) if fit_beta else ((0.04,),)
    fit = JointCountFit(
        parameters,
        order,
        covariance,
        True,
        Objective.POISSON,
        23.0,
        12,
        True,
        "converged",
        minos_intervals=(("alpha", -0.18, 0.24),),
    )
    return TFCalibrationResult(
        TFCalibrationRequest(1, 2, fit_beta=fit_beta),
        fit,
        123,
        2,
        "absent-calibrant.nxs",
        True,
        "usable",
        envelope,
    )


def round_trip(tmp_path, profile):
    path = tmp_path / "calibration.asymp"
    state = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "datasets": [],
        "grouping_profiles": [profile.to_dict()],
    }
    save_project(state, path)
    loaded = load_project(path)
    restored = GroupingProfile.from_dict(loaded["grouping_profiles"][0])
    save_project({**loaded, "grouping_profiles": [restored.to_dict()]}, path)
    assert load_project(path) == loaded
    return restored


@pytest.mark.parametrize("fit_beta", [False, True])
def test_applied_fit_round_trip(tmp_path, fit_beta):
    result = candidate(fit_beta=fit_beta)
    record = CalibrationRecord("cal-1", result)
    profile = GroupingProfile("TF", ProfileFingerprint("test", 2))
    apply_calibration(profile, record)
    restored = round_trip(tmp_path, profile)
    assert restored.alpha_policy.value == result.alpha
    assert restored.alpha_policy.error == result.alpha_error
    assert restored.beta_policy.value == result.beta
    assert restored.alpha_policy.record is restored.beta_policy.record
    assert restored.calibration_records[0] is restored.alpha_policy.record
    assert restored.calibration_records[0].result == result


def test_selection_and_manual_edit_keep_history(tmp_path):
    results = tuple(
        replace(candidate(e), fit=replace(candidate(e).fit, statistic=i + 1.0))
        for i, e in enumerate(TFEnvelope)
    )
    record = CalibrationRecord("selection", TFCalibrationSelection(results))
    profile = GroupingProfile("TF", ProfileFingerprint("test", 2))
    apply_calibration(profile, record)
    profile.alpha_policy = AlphaPolicy(value=1.7)
    restored = round_trip(tmp_path, profile)
    assert restored.alpha_policy.value == 1.7
    assert restored.beta_policy.record is restored.calibration_records[0]
    assert restored.beta_policy.record.result.candidates == results


@pytest.mark.parametrize("status", ["available", "unavailable", "unreliable"])
@pytest.mark.parametrize("envelope", list(TFEnvelope))
def test_uncertainty_states_and_constraints(tmp_path, status, envelope):
    result = candidate(envelope, fit_beta=False)
    params = (
        *result.fit.parameters,
        CountFitParameter("phase", 0.3, min=0.3, max=0.3, fixed=True),
        CountFitParameter("frequency", 1.3, fixed=True),
    )
    result = replace(
        result,
        request=replace(result.request, parameters=params[2:]),
        fit=replace(
            result.fit,
            parameters=params,
            covariance=None if status == "unavailable" else result.fit.covariance,
            covariance_accurate=status == "available",
        ),
    )
    profile = GroupingProfile("TF", ProfileFingerprint("test", 2))
    apply_calibration(profile, CalibrationRecord("state", result))
    restored = round_trip(tmp_path, profile)
    fitted = restored.alpha_policy.result
    assert fitted.fit.uncertainty_status == status
    assert fitted.alpha_error == (pytest.approx(0.2) if status == "available" else None)
    assert fitted.beta_error is None
    assert fitted.alpha_beta_correlation is None
    assert fitted.request == result.request
    assert fitted.fit.parameters == params
    with pytest.raises(ValueError, match="resolvable source"):
        restored.alpha_policy.record.require_source()


def test_failed_selection_diagnostics_do_not_apply(tmp_path):
    results = tuple(
        replace(
            candidate(e),
            ok=False,
            message="unresolved precession",
            fit=replace(candidate(e).fit, success=False, message="minimizer failed"),
        )
        for e in TFEnvelope
    )
    record = CalibrationRecord("failed", TFCalibrationSelection(results))
    profile = GroupingProfile("TF", ProfileFingerprint("test", 2), calibration_records=(record,))
    with pytest.raises(ValueError, match="usable"):
        apply_calibration(profile, record)
    restored = round_trip(tmp_path, profile)
    assert restored.alpha_policy.value == 1.0
    assert restored.calibration_records[0].result.selected is None
    assert restored.calibration_records[0].result.candidates == results


@pytest.mark.parametrize("method", ["general", "diamagnetic", "count_fit", ""])
def test_released_scalar_provenance_is_not_joint(tmp_path, method):
    from asymmetry.core.project.profiles import BetaPolicy
    from asymmetry.core.project.schema import migrate_to_current

    profile = GroupingProfile(
        "old",
        ProfileFingerprint("test", 2),
        alpha_policy=AlphaPolicy("calibrated", 1.3, 0.07, method, 123),
        beta_policy=BetaPolicy("calibrated", 0.8, 0.04, "count_fit", 123),
    )
    state = {"schema_version": 34, "datasets": [], "grouping_profiles": [profile.to_dict()]}
    migrated = migrate_to_current(state)
    assert migrate_to_current(migrated) == migrated
    restored = round_trip(tmp_path, GroupingProfile.from_dict(migrated["grouping_profiles"][0]))
    assert restored.alpha_policy == profile.alpha_policy
    assert restored.beta_policy == profile.beta_policy
    assert restored.calibration_records == ()


@pytest.mark.parametrize("fit_beta", [False, True])
@pytest.mark.parametrize("corrected", [False, True])
def test_actual_selection_support_and_fit_round_trip(tmp_path, fit_beta, corrected):
    from asymmetry.core.fitting.tf_calibration import select_tf_calibration
    from tests.core.test_tf_calibration_selection import _control

    ds = _control("stretched", 0.8 if fit_beta else 1, corrected=corrected)
    ds.run.source_file = "missing-source.nxs"
    selection = select_tf_calibration(
        ds,
        TFCalibrationRequest(
            1,
            2,
            fit_beta=fit_beta,
            frequency_mhz=1.3,
            parameters=(CountFitParameter("frequency", 1.3, fixed=True),),
        ),
    )
    profile = GroupingProfile("actual", ProfileFingerprint("test", 2))
    apply_calibration(profile, CalibrationRecord("actual", selection))
    restored = round_trip(tmp_path, profile)
    frozen = restored.calibration_records[0].result
    assert frozen == selection
    for expected, actual in zip(selection.candidates, frozen.candidates, strict=True):
        for source_domain, restored_domain in zip(
            expected.fit.domains, actual.fit.domains, strict=True
        ):
            assert restored_domain.alignment.group_lengths == source_domain.alignment.group_lengths
            assert isinstance(restored_domain.alignment.group_lengths, tuple)
    assert all(c.fit.domains is frozen.candidates[0].fit.domains for c in frozen.candidates)
    assert restored.alpha_policy.result.envelope == selection.selected.envelope
    assert restored.alpha_policy.error == selection.selected.alpha_error
    assert restored.beta_policy.error == selection.selected.beta_error
    assert frozen.candidates[0].fit.objective is (
        Objective.GAUSSIAN if corrected else Objective.POISSON
    )
    payload = restored.to_dict()
    assert "domains" not in payload["calibration_records"][0]["candidates"][0]["fit"]


def test_project_rejects_corrupt_parameter_ownership_and_constraints(tmp_path):
    from copy import deepcopy

    profile = GroupingProfile("TF", ProfileFingerprint("test", 2))
    apply_calibration(profile, CalibrationRecord("canonical", candidate()))
    payload = profile.to_dict()
    for change in ("duplicate_order", "negative_alpha", "beta_fixed", "missing_identity"):
        data = deepcopy(payload)
        fit = data["calibration_records"][0]["candidates"][0]["fit"]
        if change == "duplicate_order":
            fit["covariance_parameters"] = ["alpha", "alpha"]
        elif change == "negative_alpha":
            fit["parameters"][0]["value"] = -1
        elif change == "beta_fixed":
            fit["parameters"][1]["fixed"] = True
        else:
            data["alpha_policy"]["calibration_id"] = "missing"
        with pytest.raises(ValueError):
            GroupingProfile.from_dict(data)


def test_shared_profile_resolves_two_runs_and_manual_beta(tmp_path):
    from asymmetry.core.project.profiles import BetaPolicy, resolve_effective_grouping
    from tests.core.test_tf_calibration_fit import _raw_tf

    profile = GroupingProfile("shared", ProfileFingerprint("test", 2), groups={1: [1], 2: [2]})
    apply_calibration(profile, CalibrationRecord("joint", candidate()))
    restored = round_trip(tmp_path, profile)
    for number in (7, 8):
        ds = _raw_tf()
        ds.run.run_number = number
        grouping = resolve_effective_grouping(restored, ds.run)
        assert grouping["alpha"] == 1.2
        assert grouping["beta"] == 1.1
    restored.beta_policy = BetaPolicy(value=0.9)
    manual = round_trip(tmp_path, restored)
    assert manual.beta_policy.value == 0.9
    assert manual.alpha_policy.record is manual.calibration_records[0]


def test_nonfinite_failure_and_strict_json(tmp_path):
    import json
    import math

    result = candidate(fit_beta=False)
    result = replace(
        result,
        ok=False,
        message="failed",
        fit=replace(result.fit, success=False, statistic=float("nan"), covariance=None),
    )
    profile = GroupingProfile(
        "diagnostics",
        ProfileFingerprint("test", 2),
        calibration_records=(CalibrationRecord("failed", result),),
    )
    path = tmp_path / "failed.asymp"
    save_project(
        {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "datasets": [],
            "grouping_profiles": [profile.to_dict()],
        },
        path,
    )

    def reject_constant(value):
        raise AssertionError(f"Bare nonfinite JSON constant: {value}")

    json.loads(path.read_text(), parse_constant=reject_constant)
    restored = GroupingProfile.from_dict(load_project(path)["grouping_profiles"][0])
    assert math.isnan(restored.calibration_records[0].result.fit.statistic)
    assert restored.alpha_policy.value == 1


def test_projection_history_retains_canonical_identity(tmp_path):
    from asymmetry.core.project.profiles import resolve_effective_grouping
    from asymmetry.core.transform.projections import projection_alphas
    from tests.core.test_tf_calibration_fit import _raw_tf

    profile = GroupingProfile("vector", ProfileFingerprint("test", 2), groups={1: [1], 2: [2]})
    apply_calibration(profile, CalibrationRecord("joint", candidate()))
    profile.projections = [{"label": "pair", "forward_group": 1, "backward_group": 2}]
    restored = round_trip(tmp_path, profile)
    assert restored.projections == profile.projections
    assert restored.alpha_policy.record is restored.beta_policy.record
    grouping = resolve_effective_grouping(restored, _raw_tf().run)
    assert projection_alphas(grouping)["pair"] == restored.alpha_policy.value


@pytest.mark.parametrize("seed", [0.0, -1.0])
def test_successful_fit_preserves_clipped_free_request_seed(tmp_path, seed):
    from asymmetry.core.fitting.tf_calibration import fit_tf_calibration
    from tests.core.test_tf_calibration_fit import _raw_tf

    request = TFCalibrationRequest(
        1,
        2,
        fit_beta=True,
        parameters=(
            CountFitParameter("beta", seed),
            CountFitParameter("frequency", 1.3, fixed=True),
        ),
    )
    result = fit_tf_calibration(_raw_tf(beta=1), request)
    assert result.ok, result.message
    profile = GroupingProfile("seed", ProfileFingerprint("test", 2))
    apply_calibration(profile, CalibrationRecord("seed", result))
    restored = round_trip(tmp_path, profile)
    assert restored.alpha_policy.result.request == request
    assert restored.beta_policy.value > 0
    assert restored.beta_policy.value == result.beta


@pytest.mark.parametrize("period", [None, 1, "sum"])
def test_recorded_period_and_requested_window_survive(tmp_path, period):
    result = candidate()
    request = replace(
        result.request, period=period, t_min=-0.02, t_max=6.0, exclude=(0.0, 0.04), minos=True
    )
    result = replace(result, request=request)
    profile = GroupingProfile("support", ProfileFingerprint("test", 2))
    apply_calibration(profile, CalibrationRecord("support", result))
    restored = round_trip(tmp_path, profile)
    assert restored.alpha_policy.result.request == request
    assert restored.alpha_policy.result.period_number == 2
    assert restored.alpha_policy.result.source_run_number == 123


@pytest.mark.parametrize("identity", ["", None, 7, ["invalid"]])
def test_invalid_identity_cannot_become_an_applied_record(identity):
    with pytest.raises(ValueError, match="nonempty identity"):
        CalibrationRecord(identity, candidate())


@pytest.mark.parametrize("identity", ["", None, 7, ["invalid"]])
def test_invalid_persisted_identity_is_rejected(identity):
    from asymmetry.core.project.calibration import calibration_payload, restore_calibration

    payload = calibration_payload(CalibrationRecord("valid", candidate()))
    payload["calibration_id"] = identity
    with pytest.raises(ValueError, match="nonempty identity"):
        restore_calibration(payload)


def test_schema35_records_upgrade_without_reinterpreting_coefficients():
    from copy import deepcopy

    from asymmetry.core.project.schema import migrate_to_current

    profile = GroupingProfile("saved", ProfileFingerprint("test", 2))
    apply_calibration(profile, CalibrationRecord("history", candidate()))
    profile.alpha_policy = AlphaPolicy(
        mode="calibrated", value=1.37, method="general", source_run=42
    )
    old = {"schema_version": 35, "datasets": [], "grouping_profiles": [profile.to_dict()]}
    original = deepcopy(old)
    migrated = migrate_to_current(old)
    assert CURRENT_SCHEMA_VERSION > 35
    assert migrated == {**original, "schema_version": CURRENT_SCHEMA_VERSION}
    assert old == original
    restored = GroupingProfile.from_dict(migrated["grouping_profiles"][0])
    assert restored.alpha_policy.to_dict() == profile.alpha_policy.to_dict()
    assert restored.calibration_records == profile.calibration_records


@pytest.mark.parametrize("invalid", ["zero", "beta"])
def test_applied_quick_policy_requires_a_usable_alpha(invalid):
    from asymmetry.core.workflow.calibration import CalibrationOptions, calibrate_run
    from asymmetry.core.workflow.reduction import ReductionSettings
    from tests.core.test_tf_calibration_fit import _raw_tf

    result = calibrate_run(
        _raw_tf(beta=1), ReductionSettings(), CalibrationOptions("ratio")
    ).result.to_dict()
    if invalid == "zero":
        result["alpha"] = 0.0
    payload = GroupingProfile("quick", ProfileFingerprint("test", 2)).to_dict()
    payload["beta_policy" if invalid == "beta" else "alpha_policy"] = {"quick_calibration": result}
    with pytest.raises(ValueError, match="quick|Quick"):
        GroupingProfile.from_dict(payload)


@pytest.mark.parametrize("fit_beta", [False, True])
def test_fixed_pulse_actual_record_replay_and_target_independence(tmp_path, fit_beta):
    import numpy as np

    from asymmetry.core.fitting.tf_calibration import fit_tf_calibration
    from asymmetry.core.fitting.time_response import TimeResponse
    from asymmetry.core.project.profiles import profile_fingerprint_for_run, profile_from_payload
    from tests.core.test_tf_pulse_calibration import _acquisition

    data = _acquisition(0.8 if fit_beta else 1)
    result = fit_tf_calibration(
        data,
        TFCalibrationRequest(1, 2, fit_beta=fit_beta, frequency_mhz=1.3, pulse_separation_us=0.324),
    )
    target = _acquisition(1).run
    target.grouping.update(
        t0_bin=2,
        first_good_bin=11,
        last_good_bin=400,
        time_response=TimeResponse("double_delta", 0.4).to_dict(),
    )
    profile = profile_from_payload(target.grouping, "target", profile_fingerprint_for_run(target))
    before = (profile.t0_policy, profile.good_window_policy, dict(profile.extra))
    arrays = [h.counts.copy() for h in target.histograms]
    apply_calibration(profile, CalibrationRecord("pulse", result))
    restored = round_trip(tmp_path, profile)
    saved = restored.calibration_records[0].selected
    assert saved == result
    replay = fit_tf_calibration(data, saved.request, envelope=saved.envelope)
    assert replay == result
    assert (restored.t0_policy, restored.good_window_policy, restored.extra) == before
    for actual, original in zip(target.histograms, arrays, strict=True):
        np.testing.assert_array_equal(actual.counts, original)


def test_restored_pulse_must_match_the_actual_fit():
    from asymmetry.core.project.calibration import calibration_payload, restore_calibration

    payload = calibration_payload(CalibrationRecord("invalid-pulse", candidate()))
    payload["candidates"][0]["request"]["pulse_separation_us"] = 0.324
    with pytest.raises(ValueError, match="pulse settings.*captured request"):
        restore_calibration(payload)
