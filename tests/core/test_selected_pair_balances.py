"""Selected-pair consumers derive balances without replacing the base fallback."""

from dataclasses import replace

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.fourier.fb_input import build_fb_signal_dataset
from asymmetry.core.fourier.spectrum import fourier_grouping_digest
from asymmetry.core.maxent import MaxEntConfig, build_maxent_input
from asymmetry.core.representation.time import TimeFBAsymmetry
from asymmetry.core.simulate import (
    PeriodSpec,
    degrade_run,
    reduce_run_to_dataset,
    simulate_double_pulse_run,
    simulate_run,
    simulate_two_period_run,
)
from asymmetry.core.transform.grouping import grouping_balances
from asymmetry.core.transform.integral import integrate_run


def pair_run(canonical=False):
    label = "P_x" if canonical else "FB"
    grouping = {
        "groups": {1: [1], 2: [2], 3: [3], 4: [4]},
        "projections": [
            {"label": label, "forward_group": 1, "backward_group": 2, "alpha": 0.75},
            {"label": "UD", "forward_group": 3, "backward_group": 4, "alpha": 1.1},
        ],
        "forward_group": 1,
        "backward_group": 2,
        "vector_axis": label,
        "alpha": 1.3,
        "beta": 1.4,
        "first_good_bin": 0,
        "last_good_bin": 63,
        "deadtime_correction": False,
        "background_correction": False,
    }
    if canonical:
        grouping["alpha_x"] = 0.75
    return Run(
        7110,
        [Histogram(np.full(64, value), 0.04) for value in (100, 80, 120, 60)],
        metadata={"field": 0.0},
        grouping=grouping,
    )


@pytest.mark.parametrize("canonical", [False, True])
@pytest.mark.parametrize(
    "consumer", ["balances", "reduce", "fourier", "degrade", "recipe", "integral"]
)
def test_active_pair_consumers_keep_the_analytic_balance(canonical, consumer):
    run = pair_run(canonical)
    if consumer == "balances":
        assert grouping_balances(run.grouping) == (0.75, 1.0)
    elif consumer == "integral":
        assert integrate_run(run)[0] == pytest.approx(0.25)
    else:
        data = (
            TimeFBAsymmetry().compute(run)[0]
            if consumer == "recipe"
            else build_fb_signal_dataset(run)
            if consumer == "fourier"
            else reduce_run_to_dataset(degrade_run(run, 1) if consumer == "degrade" else run)
        )
        np.testing.assert_allclose(data.asymmetry, 0.25 if consumer == "recipe" else 25.0)
    assert run.grouping["alpha"] == 1.3


def test_pair_identity_wins_over_stale_selection_text_and_keeps_missing_fallback():
    run = pair_run()
    run.grouping.update(forward_group=3, backward_group=4)
    assert grouping_balances(run.grouping) == (1.1, 1.0)
    run.grouping["projections"][1].pop("alpha")
    assert grouping_balances(run.grouping) == (1.3, 1.0)


def test_fb_fourier_digest_tracks_the_selected_projection_balance():
    run = pair_run()
    before = fourier_grouping_digest(run, "fb_asymmetry")
    run.grouping["projections"][0]["alpha"] = 0.8
    assert fourier_grouping_digest(run, "fb_asymmetry") != before


@pytest.mark.parametrize("selected,expected", [([1, 2], 0.75), ([3, 4], 1.1)])
def test_maxent_uses_the_explicitly_selected_pair_balance(selected, expected):
    result = build_maxent_input(
        pair_run(),
        MaxEntConfig(
            mode="zf_lf",
            selected_group_ids=selected,
            auto_window=False,
            n_spectrum_points=32,
            f_max_mhz=3,
        ),
    )
    assert result.zf_lf_alpha == expected


@pytest.mark.parametrize("kind", ["single", "double", "periods"])
def test_simulation_defaults_capture_the_selected_pair_alpha(kind):
    def zero(time):
        return np.zeros_like(time)

    run = pair_run()
    if kind == "single":
        simulated = simulate_run(run, zero, total_events=10000, seed=12)
    elif kind == "double":
        simulated = simulate_double_pulse_run(run, zero, total_events=10000, seed=12, dpsep_us=0)
    else:
        simulated = simulate_two_period_run(
            run, [PeriodSpec(zero), PeriodSpec(zero)], total_events=10000, seed=12
        )
    provenance = simulated.metadata["simulation"]
    assert (
        [p["alpha"] for p in provenance["periods"]] == [0.75, 0.75]
        if kind == "periods"
        else provenance["alpha"] == 0.75
    )


def test_selected_pair_does_not_change_scalar_legacy_balances():
    run = pair_run()
    run = replace(run, grouping={"alpha": 0.8, "beta": 1.2})
    assert grouping_balances(run.grouping) == (0.8, 1.2)


@pytest.mark.parametrize("canonical", [False, True])
def test_workflow_reduction_resolves_projection_instead_of_authored_fallback(canonical):
    from asymmetry.core.workflow.reduction import ReductionSettings, reduce_run

    run = pair_run(canonical)
    data = reduce_run(run, ReductionSettings(rebin=2))
    np.testing.assert_allclose(data.asymmetry, 25.0)
    assert run.grouping["alpha"] == 1.3


def test_projection_reduction_uses_each_alpha_and_fixed_beta():
    from asymmetry.core.transform.projections import reduce_run_projections

    run = pair_run()
    results = reduce_run_projections(run, 0)
    for label, forward, backward, alpha in (("FB", 100, 80, 0.75), ("UD", 120, 60, 1.1)):
        expected = 100 * (forward - alpha * backward) / (forward + alpha * backward)
        np.testing.assert_allclose(results[label].asymmetry, expected)
    assert run.grouping["beta"] == 1.4


def test_pair_outside_declared_projections_keeps_both_scalar_fallbacks():
    assert grouping_balances(pair_run().grouping, pair=(1, 4)) == (1.3, 1.4)


@pytest.mark.parametrize("canonical", [False, True])
@pytest.mark.parametrize(
    "operation", ["manual", "unity", "legacy_scalar", "automatic", "tf", "quick"]
)
def test_explicit_workflow_coefficient_overrides_only_the_selected_projection(canonical, operation):
    from asymmetry.core.project.calibration import CalibrationRecord
    from asymmetry.core.workflow.calibration import CalibrationOptions, calibrate_run
    from asymmetry.core.workflow.reduction import (
        ReductionSettings,
        reduce_run,
        resolve_reduction_grouping,
    )
    from tests.project.test_tf_calibration_records import candidate

    run = pair_run(canonical)
    settings = ReductionSettings(alpha=1.3, alpha_source="user")
    if operation == "unity":
        settings = ReductionSettings(alpha=1, alpha_source="user")
    elif operation == "legacy_scalar":
        settings = ReductionSettings(alpha=1.3)
    elif operation == "automatic":
        settings = ReductionSettings(alpha=1.25, alpha_source="estimated:7000")
    elif operation == "tf":
        settings = ReductionSettings(alpha=CalibrationRecord("explicit", candidate(fit_beta=False)))
    elif operation == "quick":
        result = calibrate_run(
            reduce_run_to_dataset(run), ReductionSettings(), CalibrationOptions("ratio")
        )
        settings = ReductionSettings(alpha=result.result)
    grouping = resolve_reduction_grouping(run, settings)
    assert grouping_balances(grouping) == (settings.effective_alpha, 1.0)
    assert grouping_balances(grouping, pair=(3, 4)) == (1.1, 1.0)
    expected = 100 * (100 - settings.effective_alpha * 80) / (100 + settings.effective_alpha * 80)
    np.testing.assert_allclose(reduce_run(run, settings).asymmetry, expected, atol=1e-12)
    replay = ReductionSettings.from_dict(settings.to_dict())
    np.testing.assert_array_equal(
        reduce_run(run, replay).asymmetry, reduce_run(run, settings).asymmetry
    )
    assert run.grouping["alpha"] == 1.3


def test_joint_record_applies_both_balances_to_an_unmatched_scalar_pair():
    from asymmetry.core.project.calibration import CalibrationRecord
    from asymmetry.core.workflow.reduction import (
        ReductionSettings,
        reduce_run,
        resolve_reduction_grouping,
    )
    from tests.project.test_tf_calibration_records import candidate

    run = pair_run()
    record = CalibrationRecord("outside", candidate(fit_beta=True))
    settings = ReductionSettings(alpha=record, pair=("1", "4"))
    alpha, beta = record.selected.alpha, record.selected.beta
    grouping = resolve_reduction_grouping(run, settings)
    assert grouping_balances(grouping) == (alpha, beta)
    expected = 100 * (100 - alpha * 60) / (beta * 100 + alpha * 60)
    np.testing.assert_allclose(reduce_run(run, settings).asymmetry, expected, atol=1e-12)


def test_joint_record_cannot_apply_beta_to_a_declared_projection():
    from asymmetry.core.project.calibration import CalibrationRecord
    from asymmetry.core.workflow.reduction import ReductionSettings, reduce_run
    from tests.project.test_tf_calibration_records import candidate

    with pytest.raises(ValueError, match="beta fixed"):
        reduce_run(
            pair_run(),
            ReductionSettings(alpha=CalibrationRecord("vector", candidate(fit_beta=True))),
        )


def test_projection_policy_transport_preserves_unmatched_scalar_beta():
    from asymmetry.core.project.calibration import (
        CalibrationRecord,
        apply_calibration_outcome,
        calibration_values,
    )
    from asymmetry.core.project.profiles import profile_fingerprint_for_run, profile_from_payload
    from asymmetry.core.workflow.calibration_results import CalibrationOutcome
    from tests.project.test_tf_calibration_records import candidate

    run = pair_run()
    profile = profile_from_payload(run.grouping, "paired", profile_fingerprint_for_run(run))
    apply_calibration_outcome(
        profile,
        CalibrationOutcome(CalibrationRecord("projection", candidate(fit_beta=False))),
        projection="FB",
    )
    payload = calibration_values(profile.to_dict())
    assert payload["beta"] == 1.4
    assert grouping_balances(payload, pair=(1, 4)) == (1.3, 1.4)
    assert grouping_balances(payload, pair=(1, 2)) == (1.2, 1.0)


@pytest.mark.parametrize("manual", [False, True])
def test_reduce_reports_the_selected_pair_coefficient_in_json_and_sidecar(
    workflow_folder, tmp_path, capsys, monkeypatch, manual
):
    import json

    from asymmetry import cli
    from asymmetry.core.simulate import reduce_run_to_dataset
    from asymmetry.core.workflow import reduction
    from tests.core.conftest import SCAN_RUNS

    run = pair_run()
    run.run_number = SCAN_RUNS[0]
    data = reduce_run_to_dataset(run)
    monkeypatch.setattr(reduction, "load", lambda path: data)
    args = [
        "reduce",
        str(workflow_folder),
        "--runs",
        str(SCAN_RUNS[0]),
        "--workdir",
        str(tmp_path / "state"),
        "--json",
    ]
    if manual:
        args += ["--alpha", "1"]
    cli.main(args)
    entry = json.loads(capsys.readouterr().out)["entries"][0]
    alpha = 1.0 if manual else 0.75
    assert entry["a0_percent"] == pytest.approx(100 * (100 - alpha * 80) / (100 + alpha * 80))
    assert entry["alpha"] == alpha
    saved = json.loads((tmp_path / "state" / "reduced" / f"{SCAN_RUNS[0]}.json").read_text())
    assert saved["alpha"] == alpha
    cli.main(args)
    assert json.loads(capsys.readouterr().out)["entries"][0]["alpha"] == alpha


@pytest.mark.parametrize("canonical", [False, True])
@pytest.mark.parametrize("operation", ["manual", "tf", "ratio", "general"])
def test_explicit_projection_override_replaces_only_its_derived_provenance(canonical, operation):
    from copy import deepcopy

    from asymmetry.core.project.calibration import CalibrationRecord
    from asymmetry.core.workflow.calibration import CalibrationOptions, calibrate_run
    from asymmetry.core.workflow.reduction import ReductionSettings, resolve_reduction_grouping
    from docs.screenshots.data.calibration import calibration_control
    from tests.project.test_tf_calibration_records import candidate

    run = pair_run(canonical)
    old = {"alpha_error": 0.03, "alpha_reference_run": 77, "alpha_method": "old"}
    run.grouping["projections"][0].update(old)
    if canonical:
        run.grouping.update(alpha_x_error=0.03, alpha_x_reference_run=77, alpha_x_method="old")
    run.grouping["projections"][1].update(old)
    before = deepcopy(run.grouping)
    settings = ReductionSettings(alpha=1.3, alpha_source="user")
    error, reference, method = None, None, None
    if operation == "tf":
        record = CalibrationRecord("new", candidate(fit_beta=False))
        settings = ReductionSettings(alpha=record)
        error, reference, method = (
            record.selected.alpha_error,
            record.selected.source_run_number,
            "tf_calibration_fit",
        )
    elif operation in ("ratio", "general"):
        source = calibration_control(relaxing=operation == "general")
        outcome = calibrate_run(source, ReductionSettings(), CalibrationOptions(operation))
        settings = ReductionSettings(alpha=outcome.result)
        error, reference, method = outcome.selected.alpha_error, source.run_number, operation
    resolved = resolve_reduction_grouping(run, settings)
    targets = [(resolved["projections"][0], "alpha")]
    if canonical:
        targets.append((resolved, "alpha_x"))
    for target, prefix in targets:
        assert target[prefix] == settings.effective_alpha
        for suffix, expected in (
            ("_error", error),
            ("_reference_run", reference),
            ("_method", method),
        ):
            key = prefix + suffix
            if expected is None:
                assert key not in target
            else:
                assert target[key] == expected
    for key, value in old.items():
        assert resolved["projections"][1][key] == value
    assert run.grouping == before


def test_saved_projection_policy_omits_derived_aliases_from_extra():
    from asymmetry.core.project.calibration import CalibrationRecord, apply_calibration_outcome
    from asymmetry.core.project.profiles import (
        GroupingProfile,
        profile_fingerprint_for_run,
        profile_from_payload,
    )
    from asymmetry.core.workflow.calibration_results import CalibrationOutcome
    from tests.project.test_tf_calibration_records import candidate

    run = pair_run(canonical=True)
    run.grouping.update(alpha_x_error=0.03, alpha_x_reference_run=77)
    profile = profile_from_payload(run.grouping, "projection", profile_fingerprint_for_run(run))
    outcome = CalibrationOutcome(CalibrationRecord("replacement", candidate(fit_beta=False)))
    apply_calibration_outcome(profile, outcome, projection="P_x")
    payload = profile.to_dict()
    for key in ("alpha_x", "alpha_x_error", "alpha_x_reference_run", "alpha_x_method"):
        assert key not in payload.get("extra", {})
        assert key not in payload
    restored = GroupingProfile.from_dict(payload)
    assert restored.projection_alpha_policies["P_x"].record == outcome.record
