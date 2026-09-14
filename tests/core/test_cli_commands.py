"""Tests for the ``survey``/``alpha``/``reduce`` CLI subcommands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from asymmetry import __version__, cli
from asymmetry.cli._output import SCHEMA, UserError
from asymmetry.cli._runs import parse_run_spec, resolve_run, resolve_runs
from tests.core.conftest import (
    ALL_RUNS,
    CALIBRATION_ALPHA,
    CALIBRATION_RUN,
    DEADTIME_RUN,
    SCAN_RUNS,
)


def _json_output(capsys) -> dict:
    return json.loads(capsys.readouterr().out)


def _assert_stamped(payload: dict) -> None:
    assert payload["schema"] == SCHEMA
    assert payload["asymmetry_version"] == __version__


# -- run specs --------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("5", [5]),
        ("5,7", [5, 7]),
        ("5-8", [5, 6, 7, 8]),
        ("5-7,10,12-13", [5, 6, 7, 10, 12, 13]),
        (" 5 - 7 , 10 ", [5, 6, 7, 10]),
        ("7,5,6", [5, 6, 7]),
        ("5-7,6-8", [5, 6, 7, 8]),
    ],
)
def test_parse_run_spec(spec: str, expected: list[int]) -> None:
    assert parse_run_spec(spec) == expected


@pytest.mark.parametrize("spec", ["", "abc", "5-", "-5", "8-5", "5,,7", "5-7-9"])
def test_parse_run_spec_rejects_bad_input(spec: str) -> None:
    with pytest.raises(UserError):
        parse_run_spec(spec)


def test_resolve_runs_skips_gaps_but_needs_one_match(workflow_folder: Path) -> None:
    resolved = resolve_runs(workflow_folder, f"{SCAN_RUNS[0]}-{SCAN_RUNS[1]},900")
    assert [run for run, _prefix, _path in resolved] == [SCAN_RUNS[0], SCAN_RUNS[1]]
    assert {prefix for _run, prefix, _path in resolved} == {"SIM"}
    with pytest.raises(UserError):
        resolve_runs(workflow_folder, "900-910")


def test_resolve_run_reports_a_missing_run(workflow_folder: Path) -> None:
    assert resolve_run(workflow_folder, CALIBRATION_RUN).exists()
    with pytest.raises(UserError):
        resolve_run(workflow_folder, 900)


# -- survey -----------------------------------------------------------------


def test_survey_json_payload_and_written_file(
    workflow_folder: Path, tmp_path: Path, capsys
) -> None:
    workdir = tmp_path / "wd"
    cli.main(["survey", str(workflow_folder), "--json", "--workdir", str(workdir)])

    payload = _json_output(capsys)
    _assert_stamped(payload)
    survey = payload["survey"]
    assert [row["run_number"] for row in survey["runs"]] == list(ALL_RUNS)
    assert survey["best_calibration_run"] == CALIBRATION_RUN

    stored = json.loads((workdir / "survey.json").read_text(encoding="utf-8"))
    assert stored["schema"] == SCHEMA
    assert stored["best_calibration_run"] == CALIBRATION_RUN


def test_survey_human_output_names_the_calibration_run_and_the_scan(
    workflow_folder: Path, tmp_path: Path, capsys
) -> None:
    cli.main(["survey", str(workflow_folder), "--workdir", str(tmp_path / "wd")])
    out = capsys.readouterr().out
    assert f"run {CALIBRATION_RUN}" in out
    assert "temperature scan" in out
    assert "ZF" in out


def test_survey_defaults_its_workdir_into_the_data_folder(tmp_path: Path, capsys) -> None:
    folder = tmp_path / "empty"
    folder.mkdir()
    cli.main(["survey", str(folder)])
    assert (folder / ".asymmetry" / "survey.json").exists()


def test_survey_on_a_missing_folder_is_a_user_error(tmp_path: Path, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["survey", str(tmp_path / "nowhere")])
    assert exc.value.code == 1
    assert "not a directory" in capsys.readouterr().err


# -- alpha ------------------------------------------------------------------


def test_alpha_json_payload_recovers_the_simulated_balance(workflow_folder: Path, capsys) -> None:
    cli.main(["alpha", str(workflow_folder), "--run", str(CALIBRATION_RUN), "--json"])
    payload = _json_output(capsys)
    _assert_stamped(payload)
    assert payload["alpha"]["alpha"] == pytest.approx(CALIBRATION_ALPHA, rel=0.01)
    assert payload["alpha"]["method"] == "per_run_estimate"
    assert payload["is_calibration_candidate"] is True
    assert payload["warning"] is None


def test_alpha_warns_when_the_run_is_not_a_calibration_run(workflow_folder: Path, capsys) -> None:
    cli.main(["alpha", str(workflow_folder), "--run", str(SCAN_RUNS[0]), "--json"])
    payload = _json_output(capsys)
    assert payload["is_calibration_candidate"] is False
    assert "not a weak-transverse-field calibration run" in payload["warning"]


def test_alpha_on_an_absent_run_is_a_user_error(workflow_folder: Path, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["alpha", str(workflow_folder), "--run", "900"])
    assert exc.value.code == 1
    assert "Run 900 is not in" in capsys.readouterr().err


# -- reduce -----------------------------------------------------------------


def test_reduce_writes_entries_and_a_manifest(
    workflow_folder: Path, tmp_path: Path, capsys
) -> None:
    workdir = tmp_path / "wd"
    spec = f"{SCAN_RUNS[0]}-{SCAN_RUNS[2]}"
    cli.main(["reduce", str(workflow_folder), "--runs", spec, "--json", "--workdir", str(workdir)])

    payload = _json_output(capsys)
    _assert_stamped(payload)
    assert [entry["run_number"] for entry in payload["entries"]] == list(SCAN_RUNS[:3])
    assert all(entry["recomputed"] for entry in payload["entries"])
    assert payload["settings"]["alpha"] == pytest.approx(1.0)
    assert payload["settings"]["alpha_source"] == "assumed"

    first = payload["entries"][0]
    assert first["a0_percent"] == pytest.approx(20.0, abs=4.0)
    assert first["mean_error_percent"] > 0.0
    assert first["deadtime_mode"] == "off"

    for run_number in SCAN_RUNS[:3]:
        assert (workdir / "reduced" / f"{run_number}.npz").exists()
        assert (workdir / "reduced" / f"{run_number}.json").exists()

    manifest = json.loads((workdir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["runs"] == list(SCAN_RUNS[:3])
    assert manifest["asymmetry_version"] == __version__


def test_reduce_reuses_the_cache_and_recomputes_when_a_setting_changes(
    workflow_folder: Path, tmp_path: Path, capsys
) -> None:
    workdir = tmp_path / "wd"
    args = [
        "reduce",
        str(workflow_folder),
        "--runs",
        str(SCAN_RUNS[0]),
        "--json",
        "--workdir",
        str(workdir),
    ]

    cli.main(args)
    assert _json_output(capsys)["entries"][0]["recomputed"] is True

    cli.main(args)
    assert _json_output(capsys)["entries"][0]["recomputed"] is False

    cli.main([*args, "--rebin", "2"])
    second = _json_output(capsys)["entries"][0]
    assert second["recomputed"] is True
    assert second["settings"]["rebin"] == 2


def test_reduce_with_alpha_from_records_the_estimate_and_its_source(
    workflow_folder: Path, tmp_path: Path, capsys
) -> None:
    cli.main(
        [
            "reduce",
            str(workflow_folder),
            "--runs",
            str(SCAN_RUNS[0]),
            "--alpha-from",
            str(CALIBRATION_RUN),
            "--json",
            "--workdir",
            str(tmp_path / "wd"),
        ]
    )
    payload = _json_output(capsys)
    assert payload["settings"]["alpha"] == pytest.approx(CALIBRATION_ALPHA, rel=0.01)
    assert payload["settings"]["alpha_source"] == f"estimated:{CALIBRATION_RUN}"


def test_reduce_with_deadtime_from_file(workflow_folder: Path, tmp_path: Path, capsys) -> None:
    cli.main(
        [
            "reduce",
            str(workflow_folder),
            "--runs",
            str(DEADTIME_RUN),
            "--deadtime",
            "from_file",
            "--json",
            "--workdir",
            str(tmp_path / "wd"),
        ]
    )
    entry = _json_output(capsys)["entries"][0]
    assert entry["deadtime_mode"] == "file"
    assert entry["settings"]["deadtime"] == "from_file"


def test_reduce_rejects_alpha_and_alpha_from_together(
    workflow_folder: Path, tmp_path: Path, capsys
) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "reduce",
                str(workflow_folder),
                "--runs",
                str(SCAN_RUNS[0]),
                "--alpha",
                "1.1",
                "--alpha-from",
                str(CALIBRATION_RUN),
                "--workdir",
                str(tmp_path / "wd"),
            ]
        )
    assert exc.value.code == 1
    assert "not both" in capsys.readouterr().err


def test_reduce_rejects_a_bad_rebin(workflow_folder: Path, tmp_path: Path, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "reduce",
                str(workflow_folder),
                "--runs",
                str(SCAN_RUNS[0]),
                "--rebin",
                "0",
                "--workdir",
                str(tmp_path / "wd"),
            ]
        )
    assert exc.value.code == 1
    assert "Rebin factor" in capsys.readouterr().err


def test_reduce_human_table_lists_every_run(workflow_folder: Path, tmp_path: Path, capsys) -> None:
    cli.main(
        [
            "reduce",
            str(workflow_folder),
            "--runs",
            f"{SCAN_RUNS[0]}-{SCAN_RUNS[1]}",
            "--workdir",
            str(tmp_path / "wd"),
        ]
    )
    out = capsys.readouterr().out
    assert "A(0)/%" in out
    for run_number in SCAN_RUNS[:2]:
        assert str(run_number) in out


# -- wizard -----------------------------------------------------------------


def test_wizard_before_reduce_names_the_run_and_the_step_to_run(
    workflow_folder: Path, tmp_path: Path, capsys
) -> None:
    # Screening reads the reduced spectrum, not the file, so an un-reduced run
    # must say so rather than silently reloading and re-reducing it.
    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "wizard",
                str(workflow_folder),
                "--run",
                str(SCAN_RUNS[0]),
                "--workdir",
                str(tmp_path / "wd"),
            ]
        )
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert str(SCAN_RUNS[0]) in err
    assert "asymmetry reduce" in err


def test_wizard_rejects_an_unknown_scope_preset(
    workflow_folder: Path, tmp_path: Path, capsys
) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "wizard",
                str(workflow_folder),
                "--run",
                str(SCAN_RUNS[0]),
                "--scope",
                "nonsense",
                "--workdir",
                str(tmp_path / "wd"),
            ]
        )
    assert exc.value.code == 1
    assert "Unknown scope preset" in capsys.readouterr().err


# -- fit / fit-series / trend ------------------------------------------------


@pytest.fixture
def fitting_workdir(workflow_folder: Path, tmp_path: Path):
    """A work directory with the scan reduced and one recipe stored.

    The recipe is built straight from an expression rather than by running the
    wizard: these tests are about the ``fit``/``fit-series``/``trend``
    commands, and a screening run would cost seconds of fitting to produce a
    recipe they would not otherwise care about.
    """
    from asymmetry.core.workflow.recipe import FitRecipe
    from asymmetry.core.workflow.workdir import WorkDir

    workdir = tmp_path / "wd"
    cli.main(
        [
            "reduce",
            str(workflow_folder),
            "--runs",
            f"{SCAN_RUNS[0]}-{SCAN_RUNS[-1]}",
            "--workdir",
            str(workdir),
        ]
    )
    stored = WorkDir(workdir)
    stored.write_recipe(
        "relax",
        FitRecipe.from_expression("Exponential + Constant", dataset=stored.reduced(SCAN_RUNS[0])),
    )
    return workdir


def test_fit_reports_the_parameter_table_and_the_quality_verdict(
    workflow_folder: Path, fitting_workdir: Path, capsys
) -> None:
    cli.main(
        [
            "fit",
            str(workflow_folder),
            "--run",
            str(SCAN_RUNS[0]),
            "--recipe",
            "relax",
            "--workdir",
            str(fitting_workdir),
        ]
    )
    out = capsys.readouterr().out
    assert "Exponential + Constant" in out
    assert "Lambda" in out
    assert "chi2_red" in out


def test_fit_with_fix_pins_the_parameter_at_the_given_value(
    workflow_folder: Path, fitting_workdir: Path, capsys
) -> None:
    cli.main(
        [
            "fit",
            str(workflow_folder),
            "--run",
            str(SCAN_RUNS[0]),
            "--recipe",
            "relax",
            "--fix",
            "A_bg=0",
            "--json",
            "--workdir",
            str(fitting_workdir),
        ]
    )
    payload = _json_output(capsys)
    _assert_stamped(payload)
    assert payload["fit"]["parameters"]["A_bg"] == pytest.approx(0.0)
    assert payload["fit"]["free_params"] == ["A_1", "Lambda"]
    assert "A_bg" not in payload["fit"]["uncertainties"]
    # The stored recipe is unchanged — the override applies to this fit only.
    stored = json.loads((fitting_workdir / "recipes" / "relax.json").read_text(encoding="utf-8"))
    assert all(not entry["fixed"] for entry in stored["parameters"])


def test_fit_rejects_a_fix_that_names_no_parameter(
    workflow_folder: Path, fitting_workdir: Path, capsys
) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "fit",
                str(workflow_folder),
                "--run",
                str(SCAN_RUNS[0]),
                "--recipe",
                "relax",
                "--fix",
                "Nope=1",
                "--workdir",
                str(fitting_workdir),
            ]
        )
    assert exc.value.code == 1
    assert "Nope is not a parameter" in capsys.readouterr().err


def test_fit_rejects_a_recipe_name_the_workdir_does_not_hold(
    workflow_folder: Path, fitting_workdir: Path, capsys
) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "fit",
                str(workflow_folder),
                "--run",
                str(SCAN_RUNS[0]),
                "--recipe",
                "missing",
                "--workdir",
                str(fitting_workdir),
            ]
        )
    assert exc.value.code == 1
    assert "No recipe 'missing'" in capsys.readouterr().err


def test_fit_series_writes_a_stamped_series_file(
    workflow_folder: Path, fitting_workdir: Path, capsys
) -> None:
    cli.main(
        [
            "fit-series",
            str(workflow_folder),
            "--runs",
            f"{SCAN_RUNS[0]}-{SCAN_RUNS[-1]}",
            "--recipe",
            "relax",
            "--order",
            "temperature",
            "--name",
            "scan",
            "--json",
            "--workdir",
            str(fitting_workdir),
        ]
    )
    payload = _json_output(capsys)
    _assert_stamped(payload)
    assert [entry["run"] for entry in payload["series"]["results"]] == list(SCAN_RUNS)

    stored = json.loads((fitting_workdir / "series" / "scan.json").read_text(encoding="utf-8"))
    assert stored["schema"] == SCHEMA
    assert stored["asymmetry_version"] == __version__
    assert stored["order_key"] == "temperature"


def test_fit_series_defaults_its_name_from_the_recipe(
    workflow_folder: Path, fitting_workdir: Path, capsys
) -> None:
    cli.main(
        [
            "fit-series",
            str(workflow_folder),
            "--runs",
            f"{SCAN_RUNS[0]}-{SCAN_RUNS[1]}",
            "--recipe",
            "relax",
            "--order",
            "run",
            "--workdir",
            str(fitting_workdir),
        ]
    )
    assert (fitting_workdir / "series" / "series-relax.json").exists()


def test_fit_series_start_chains_outward_from_the_named_run(
    workflow_folder: Path, fitting_workdir: Path, capsys
) -> None:
    cli.main(
        [
            "fit-series",
            str(workflow_folder),
            "--runs",
            f"{SCAN_RUNS[0]}-{SCAN_RUNS[-1]}",
            "--recipe",
            "relax",
            "--order",
            "temperature",
            "--start",
            str(SCAN_RUNS[2]),
            "--name",
            "outward",
            "--json",
            "--workdir",
            str(fitting_workdir),
        ]
    )
    series = _json_output(capsys)["series"]
    assert series["start_run"] == SCAN_RUNS[2]
    assert [branch["direction"] for branch in series["branches"]] == [
        "descending",
        "ascending",
    ]
    assert series["branches"][0]["runs"] == list(reversed(SCAN_RUNS[:3]))
    assert series["branches"][1]["runs"] == list(SCAN_RUNS[2:])
    # Every run still has exactly one row, in scan order.
    assert [entry["run"] for entry in series["results"]] == list(SCAN_RUNS)


def test_fit_series_rejects_a_start_run_outside_the_series(
    workflow_folder: Path, fitting_workdir: Path, capsys
) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "fit-series",
                str(workflow_folder),
                "--runs",
                f"{SCAN_RUNS[0]}-{SCAN_RUNS[1]}",
                "--recipe",
                "relax",
                "--order",
                "temperature",
                "--start",
                "999",
                "--workdir",
                str(fitting_workdir),
            ]
        )
    assert exc.value.code == 1
    assert "--start 999 is not in the series" in capsys.readouterr().err


def test_fit_series_rejects_a_global_the_recipe_does_not_have(
    workflow_folder: Path, fitting_workdir: Path, capsys
) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "fit-series",
                str(workflow_folder),
                "--runs",
                f"{SCAN_RUNS[0]}-{SCAN_RUNS[1]}",
                "--recipe",
                "relax",
                "--order",
                "run",
                "--global",
                "Nope",
                "--workdir",
                str(fitting_workdir),
            ]
        )
    assert exc.value.code == 1
    assert "--global names Nope" in capsys.readouterr().err


def test_trend_csv_has_one_header_line_and_one_row_per_run(
    workflow_folder: Path, fitting_workdir: Path, tmp_path: Path, capsys
) -> None:
    cli.main(
        [
            "fit-series",
            str(workflow_folder),
            "--runs",
            f"{SCAN_RUNS[0]}-{SCAN_RUNS[-1]}",
            "--recipe",
            "relax",
            "--order",
            "temperature",
            "--name",
            "scan",
            "--workdir",
            str(fitting_workdir),
        ]
    )
    capsys.readouterr()

    csv_path = tmp_path / "out" / "trend.csv"
    cli.main(
        [
            "trend",
            str(workflow_folder),
            "--series",
            "scan",
            "--csv",
            str(csv_path),
            "--workdir",
            str(fitting_workdir),
        ]
    )
    out = capsys.readouterr().out
    assert "Lambda" in out

    lines = csv_path.read_text(encoding="utf-8").strip().split("\n")
    assert lines[0].split(",")[:2] == ["run", "x"]
    assert len(lines) == 1 + len(SCAN_RUNS)


def test_trend_names_the_series_the_workdir_does_hold(
    workflow_folder: Path, fitting_workdir: Path, capsys
) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "trend",
                str(workflow_folder),
                "--series",
                "nope",
                "--workdir",
                str(fitting_workdir),
            ]
        )
    assert exc.value.code == 1
    assert "No series 'nope'" in capsys.readouterr().err


# -- dispatcher -------------------------------------------------------------


def test_an_internal_error_exits_two_with_a_traceback(
    workflow_folder: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    def _boom(_folder):
        raise RuntimeError("kaboom")

    monkeypatch.setattr("asymmetry.core.workflow.survey.survey_folder", _boom)
    with pytest.raises(SystemExit) as exc:
        cli.main(["survey", str(workflow_folder)])
    assert exc.value.code == 2
    assert "kaboom" in capsys.readouterr().err
