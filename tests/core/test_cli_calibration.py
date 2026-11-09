"""Calibration transport through actual CLI JSON and saved workflow entries."""

import json
from dataclasses import replace

import numpy as np
import pytest

from asymmetry import cli
from asymmetry.core.fitting.tf_calibration import TFCalibrationSelection, TFEnvelope
from asymmetry.core.project.calibration import restore_calibration
from asymmetry.core.project.json_values import decode_non_finite
from asymmetry.core.workflow.calibration import CalibrationOptions, calibrate_run
from asymmetry.core.workflow.reduction import ReductionSettings, reduce_run
from asymmetry.core.workflow.workdir import WorkDir
from tests.core.conftest import CALIBRATION_RUN, SCAN_RUNS
from tests.core.test_tf_calibration_fit import _raw_tf


def strict_output(capsys):
    def reject(value):
        raise AssertionError(f"Non-standard JSON: {value}")

    return json.loads(capsys.readouterr().out, parse_constant=reject)


@pytest.mark.parametrize("fit_beta", [False, True])
@pytest.mark.parametrize("envelope", [None, *TFEnvelope])
def test_actual_alpha_json_matches_core(workflow_folder, capsys, monkeypatch, fit_beta, envelope):
    from asymmetry.core import io

    data = _raw_tf(beta=0.8 if fit_beta else 1)
    monkeypatch.setattr(io, "load", lambda path: data)
    args = [
        "alpha",
        str(workflow_folder),
        "--run",
        str(CALIBRATION_RUN),
        "--calibration-method",
        "tf_calibration_fit",
        "--calibration-frequency",
        "1.3",
        "--json",
    ]
    if fit_beta:
        args += ["--fit-beta"]
    if envelope is not None:
        args += ["--calibration-envelope", envelope.value]
    cli.main(args)
    summary = strict_output(capsys)["alpha"]
    captured = restore_calibration(decode_non_finite(summary["calibration"]))
    direct = calibrate_run(
        data,
        ReductionSettings(),
        CalibrationOptions("tf_calibration_fit", fit_beta, envelope, frequency_mhz=1.3),
    )
    assert captured.result == direct.record.result
    assert summary["alpha"] == (captured.selected.alpha if captured.selected.ok else None)
    assert summary["beta"] == captured.selected.beta
    assert summary["beta_fixed"] == (not fit_beta)
    if not fit_beta:
        assert summary["beta_error"] is None
        assert summary["alpha_beta_correlation"] is None
    assert isinstance(captured.result, TFCalibrationSelection) == (envelope is None)


@pytest.mark.parametrize("method", ["ratio", "general", "diamagnetic"])
def test_quick_json_uncertainty_is_honest(workflow_folder, capsys, method):
    cli.main(
        [
            "alpha",
            str(workflow_folder),
            "--run",
            str(CALIBRATION_RUN),
            "--calibration-method",
            method,
            "--json",
        ]
    )
    summary = strict_output(capsys)["alpha"]
    assert summary["method"] == method
    assert (summary["alpha_error"] is not None) == (method == "ratio")
    assert summary["beta_fixed"] and summary["beta_error"] is None
    assert "calibration" not in summary


@pytest.mark.parametrize(
    "options",
    [
        ["--fit-beta"],
        ["--calibration-method", "ratio", "--fit-beta"],
        ["--calibration-method", "general", "--calibration-envelope", "gaussian"],
        ["--calibration-envelope", "stretched"],
    ],
)
def test_invalid_combinations_are_user_errors(workflow_folder, capsys, options):
    with pytest.raises(SystemExit) as exc:
        cli.main(["alpha", str(workflow_folder), "--run", str(CALIBRATION_RUN), *options])
    assert exc.value.code == 1
    assert "require tf_calibration_fit" in capsys.readouterr().err


def test_joint_alpha_from_keeps_record_in_actual_saved_workdir(
    workflow_folder, tmp_path, capsys, monkeypatch
):
    from asymmetry.core import io
    from asymmetry.core.workflow import reduction

    source = _raw_tf(beta=0.8)
    target = _raw_tf(beta=0.8, seed=54)
    target.run.run_number = SCAN_RUNS[0]
    monkeypatch.setattr(io, "load", lambda path: source)
    monkeypatch.setattr(reduction, "load", lambda path: target)
    root = tmp_path / "wd"
    cli.main(
        [
            "reduce",
            str(workflow_folder),
            "--runs",
            str(SCAN_RUNS[0]),
            "--alpha-from",
            str(CALIBRATION_RUN),
            "--calibration-method",
            "tf_calibration_fit",
            "--fit-beta",
            "--calibration-envelope",
            "exponential",
            "--calibration-frequency",
            "1.3",
            "--workdir",
            str(root),
            "--json",
        ]
    )
    output = strict_output(capsys)
    emitted = ReductionSettings.from_dict(output["settings"])
    stored = WorkDir(root)
    settings = stored.entry(SCAN_RUNS[0]).settings
    assert settings == emitted
    assert (
        settings.calibration.calibration_id == output["settings"]["calibration"]["calibration_id"]
    )
    assert settings.calibration.selected.beta == pytest.approx(0.8, abs=2e-4)
    actual = stored.reduced(SCAN_RUNS[0])
    expected = reduce_run(target.run, settings)
    np.testing.assert_array_equal(actual.asymmetry, expected.asymmetry)
    np.testing.assert_array_equal(actual.error, expected.error)
    assert stored.alpha_calibration_runs() == {source.run.run_number}
    manifest = stored.read_manifest()
    assert ReductionSettings.from_dict(manifest["settings"]) == settings
    assert stored.entry(SCAN_RUNS[0]).settings.to_dict() == settings.to_dict()


@pytest.mark.parametrize("fit_beta", [False, True])
def test_strict_json_round_trip_preserves_endpoints_and_error_states(
    tmp_path, capsys, monkeypatch, workflow_folder, fit_beta
):
    from asymmetry.core import io
    from asymmetry.core.fitting.count_results import CountFitParameter as P
    from asymmetry.core.workflow import calibration
    from asymmetry.core.workflow.workdir import ReducedEntry

    data = _raw_tf(beta=0.8 if fit_beta else 1)
    result = calibrate_run(
        data,
        ReductionSettings(),
        CalibrationOptions(
            "tf_calibration_fit",
            fit_beta,
            TFEnvelope.EXPONENTIAL,
            frequency_mhz=1.3,
            parameters=(P("phase", 0.17, min=0.17, max=0.17, fixed=True),),
        ),
    ).record.result
    monkeypatch.setattr(io, "load", lambda path: data)
    for state in ("unreliable", "absent", "failed", "no_selection"):
        fit = replace(
            result.fit,
            covariance=None if state == "absent" else result.fit.covariance,
            covariance_accurate=False,
            statistic=float("inf") if state in {"failed", "no_selection"} else result.fit.statistic,
            success=state not in {"failed", "no_selection"},
        )
        altered = replace(result, fit=fit, ok=state not in {"failed", "no_selection"})
        selection = TFCalibrationSelection(tuple(replace(altered, envelope=e) for e in TFEnvelope))
        monkeypatch.setattr(calibration, "select_tf_calibration", lambda *a, **kw: selection)
        cli.main(
            [
                "alpha",
                str(workflow_folder),
                "--run",
                str(CALIBRATION_RUN),
                "--calibration-method",
                "tf_calibration_fit",
                *(["--fit-beta"] if fit_beta else []),
                "--json",
            ]
        )
        summary = strict_output(capsys)["alpha"]
        restored = restore_calibration(decode_non_finite(summary["calibration"]))
        assert restored.result == selection
        assert summary["alpha_error"] is None
        assert summary["beta_error"] is None
        assert summary["beta_fixed"] == (not fit_beta)
        if state in {"failed", "no_selection"}:
            assert summary["alpha"] is None
            with pytest.raises(ValueError, match="usable"):
                ReductionSettings(alpha=restored)
        else:
            settings = ReductionSettings(alpha=restored)
            root = WorkDir(tmp_path / state)
            reduced = reduce_run(data.run, settings)
            entry = ReducedEntry(
                data.run.run_number,
                "digest",
                "source.nxs",
                reduced.n_points,
                settings,
                {},
                result.alpha,
                "off",
                1,
                2,
                [],
            )
            root.write_reduced(reduced, entry)
            reread = root.entry(data.run.run_number).settings.calibration
            assert reread == restored
            params = {p.name: p for p in reread.result.selected.request.parameters}
            assert params["phase"].min == params["phase"].max == 0.17
            effective = {p.name: p for p in reread.result.selected.fit.parameters}
            assert effective["phase"].min == 0.17
            assert np.isinf(effective["N0"].max)
            if not fit_beta:
                assert np.isneginf(effective["beta"].min)
                assert np.isposinf(effective["beta"].max)


@pytest.mark.parametrize("method", ["diamagnetic", "general", "ratio"])
def test_explicit_quick_alpha_from_retains_result_context(
    workflow_folder, tmp_path, capsys, method
):
    root = tmp_path / method
    cli.main(
        [
            "reduce",
            str(workflow_folder),
            "--runs",
            str(SCAN_RUNS[0]),
            "--alpha-from",
            str(CALIBRATION_RUN),
            "--calibration-method",
            method,
            "--workdir",
            str(root),
            "--json",
        ]
    )
    output = strict_output(capsys)
    settings = ReductionSettings.from_dict(output["settings"])
    quick = settings.quick_calibration
    assert quick is not None
    assert quick.estimate.method == method
    assert (quick.estimate.alpha_error is not None) == (method == "ratio")
    assert settings.effective_alpha == quick.estimate.alpha
    assert settings.calibration is None
    stored = WorkDir(root)
    assert stored.entry(SCAN_RUNS[0]).settings == settings
    assert stored.alpha_calibration_runs() == {CALIBRATION_RUN}
