"""Tests for the ``survey``/``alpha``/``reduce`` CLI subcommands."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pytest

from asymmetry import __version__, cli
from asymmetry.cli._output import SCHEMA, UserError
from asymmetry.cli._runs import parse_run_spec, resolve_run, resolve_runs, run_files
from asymmetry.core.workflow.workdir import SCHEMA as WORKDIR_SCHEMA
from tests.core.conftest import (
    ALL_RUNS,
    CALIBRATION_ALPHA,
    CALIBRATION_RUN,
    DEADTIME_RUN,
    DECOUPLING_RUN,
    SCAN_RUNS,
    SCAN_TEMPERATURES,
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


def _write_run_file(folder: Path, prefix: str, run_number: int) -> Path:
    """One small synthetic NeXus run in *folder*, named ``<prefix><run>.nxs``."""
    from asymmetry.core.io.nexus_writer import write_nexus_v1
    from asymmetry.core.simulate import simulate_run
    from tests.core.conftest import _relaxation_signal, _template_for

    title = f"Sample {prefix}{run_number}"
    run = simulate_run(
        _template_for(temperature=10.0, field=0.0, title=title),
        _relaxation_signal(0.2),
        total_events=2.0e4,
        seed=1,
        alpha=1.0,
        run_number=run_number,
        title=title,
    )
    path = folder / f"{prefix}{run_number:08d}.nxs"
    write_nexus_v1(run, path)
    return path


def test_one_run_number_under_two_prefixes_is_refused(tmp_path: Path) -> None:
    """The work directory is keyed on the run number, so a clash has no answer."""
    pytest.importorskip("h5py")
    _write_run_file(tmp_path, "SIM", 42)
    _write_run_file(tmp_path, "MUT", 42)

    with pytest.raises(UserError) as exc:
        run_files(tmp_path)

    message = str(exc.value)
    assert "42" in message
    assert "MUT00000042.nxs" in message
    assert "SIM00000042.nxs" in message
    assert "split the folder" in message


def test_a_duplicate_run_number_reaches_the_command_line(tmp_path: Path, capsys) -> None:
    pytest.importorskip("h5py")
    _write_run_file(tmp_path, "SIM", 42)
    _write_run_file(tmp_path, "MUT", 42)

    with pytest.raises(SystemExit) as exc:
        cli.main(["reduce", str(tmp_path), "--runs", "42"])

    assert exc.value.code == 1
    assert "split the folder" in capsys.readouterr().err


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
    assert all(row["total_events"] > 0 for row in survey["runs"])
    assert survey["best_calibration_run"] == CALIBRATION_RUN

    stored = json.loads((workdir / "survey.json").read_text(encoding="utf-8"))
    assert stored["schema"] == WORKDIR_SCHEMA
    assert stored["best_calibration_run"] == CALIBRATION_RUN


def test_survey_human_output_names_the_calibration_run_and_the_scan(
    workflow_folder: Path, tmp_path: Path, capsys
) -> None:
    cli.main(["survey", str(workflow_folder), "--workdir", str(tmp_path / "wd")])
    out = capsys.readouterr().out
    assert f"run {CALIBRATION_RUN}" in out
    assert "temperature scan" in out
    assert "ZF" in out


def test_survey_table_shows_the_precession_column_and_the_measured_geometry(
    workflow_folder: Path, tmp_path: Path, capsys
) -> None:
    cli.main(["survey", str(workflow_folder), "--workdir", str(tmp_path / "wd")])
    lines = capsys.readouterr().out.splitlines()
    header = next(line for line in lines if line.lstrip().startswith("run "))
    assert "prec" in header.split()
    assert "events" in header.split()

    calibration = next(line for line in lines if line.startswith(f"{CALIBRATION_RUN} "))
    # A measured geometry is starred so the column says at a glance that the
    # spectrum, not the file, decided it.
    assert "TF*" in calibration
    assert "larmor" in calibration

    # The decoupling run's file stamps TF; the spectrum refutes it, so the
    # geometry column must show nothing rather than repeat the claim.
    decoupling = next(line for line in lines if line.startswith(f"{DECOUPLING_RUN} "))
    assert "none" in decoupling
    assert "TF" not in decoupling


def test_survey_scans_block_names_the_instrument(
    workflow_folder: Path, tmp_path: Path, capsys
) -> None:
    cli.main(["survey", str(workflow_folder), "--workdir", str(tmp_path / "wd")])
    out = capsys.readouterr().out
    assert "temperature scan, SIM, ZF, B = 0 G" in out


def test_survey_candidate_block_names_the_source_of_each_candidate(
    workflow_folder: Path, tmp_path: Path, capsys
) -> None:
    cli.main(["survey", str(workflow_folder), "--workdir", str(tmp_path / "wd")])
    out = capsys.readouterr().out
    assert f"run {CALIBRATION_RUN} (best) [measured]" in out
    assert "Larmor frequency" in out
    assert f"run {DECOUPLING_RUN}" not in out.split("Alpha-calibration candidates:")[1]


def test_survey_defaults_its_workdir_into_the_current_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """The project the command is run from, not the folder the data sits in."""
    from asymmetry.core.workflow.workdir import WORKDIR_NAME

    folder = tmp_path / "empty"
    folder.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)

    cli.main(["survey", str(folder)])

    assert (project / WORKDIR_NAME / "survey.json").exists()
    assert list(folder.iterdir()) == []


def test_every_command_offers_the_same_default_work_directory() -> None:
    """The help text is built without importing the core, so the two are pinned here."""
    from asymmetry.cli._workdir import WORKDIR_NAME as CLI_NAME
    from asymmetry.core.workflow.workdir import WORKDIR_NAME

    assert CLI_NAME == WORKDIR_NAME

    parser = cli.build_parser()
    subparsers = next(
        action
        for action in parser._actions  # noqa: SLF001 — argparse exposes no public walk
        if isinstance(action, argparse._SubParsersAction)
    )
    with_workdir = {
        name
        for name, subparser in subparsers.choices.items()
        for action in subparser._actions  # noqa: SLF001
        if action.dest == "workdir"
        if f"default: ./{WORKDIR_NAME}" in (action.help or "")
    }
    assert with_workdir == {
        "survey",
        "reduce",
        "integral-scan",
        "wizard",
        "recipe",
        "fit",
        "fit-global",
        "fit-series",
        "trend",
        "fourier",
    }


def test_a_work_directory_holds_one_data_folder(
    workflow_folder: Path, tmp_path: Path, capsys
) -> None:
    """Run numbers key everything stored, so a second folder must not share it."""
    other = tmp_path / "other"
    other.mkdir()
    workdir = tmp_path / "wd"
    cli.main(["survey", str(workflow_folder), "--workdir", str(workdir)])
    capsys.readouterr()

    with pytest.raises(SystemExit) as exc:
        cli.main(["survey", str(other), "--workdir", str(workdir)])

    assert exc.value.code == 1
    error = capsys.readouterr().err
    assert str(workflow_folder.resolve()) in error
    assert str(other.resolve()) in error
    assert "--workdir" in error


def test_a_second_data_folder_gets_a_work_directory_of_its_own(
    workflow_folder: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """The way out of a mismatch: a named directory under the default one."""
    from asymmetry.core.workflow.workdir import WORKDIR_NAME

    other = tmp_path / "other"
    other.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    cli.main(["survey", str(workflow_folder)])

    cli.main(["survey", str(other), "--workdir", f"{WORKDIR_NAME}/other"])

    assert (project / WORKDIR_NAME / "other" / "survey.json").exists()


def test_the_same_folder_named_relatively_and_absolutely_is_one_session(
    workflow_folder: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    from asymmetry.core.workflow.workdir import WORKDIR_NAME

    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    cli.main(["survey", str(workflow_folder)])
    capsys.readouterr()

    cli.main(["survey", os.path.relpath(workflow_folder, project)])

    assert (project / WORKDIR_NAME / "survey.json").exists()


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
    assert payload["precession"]["state"] == "larmor"


def test_alpha_warns_when_the_run_is_not_a_calibration_run(workflow_folder: Path, capsys) -> None:
    cli.main(["alpha", str(workflow_folder), "--run", str(SCAN_RUNS[0]), "--json"])
    payload = _json_output(capsys)
    assert payload["is_calibration_candidate"] is False
    assert "not a weak-transverse-field calibration run" in payload["warning"]


def test_alpha_says_in_words_whether_the_run_precesses_at_the_larmor_frequency(
    workflow_folder: Path, capsys
) -> None:
    cli.main(["alpha", str(workflow_folder), "--run", str(CALIBRATION_RUN)])
    assert "precession      : at the Larmor frequency: yes" in capsys.readouterr().out

    # The decoupling run records a field and shows no precession in it — the
    # case an agent calibrating on a non-candidate needs told plainly.
    cli.main(["alpha", str(workflow_folder), "--run", str(DECOUPLING_RUN)])
    assert "precession      : at the Larmor frequency: no" in capsys.readouterr().out


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


def test_reduce_selects_and_records_a_multi_period_run(
    workflow_folder: Path, tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    from asymmetry.core.data.dataset import Histogram
    from asymmetry.core.io import load as real_load

    def _clone(histogram):
        return Histogram(
            counts=histogram.counts.copy(),
            bin_width=histogram.bin_width,
            t0_bin=histogram.t0_bin,
            good_bin_start=histogram.good_bin_start,
            good_bin_end=histogram.good_bin_end,
        )

    def _load_two_periods(path):
        dataset = real_load(path)
        red = [_clone(histogram) for histogram in dataset.run.histograms]
        green = [_clone(histogram) for histogram in dataset.run.histograms]
        dataset.run.grouping["period_histograms"] = [red, green]
        dataset.run.grouping["period_reduced"] = [
            (dataset.time.copy(), dataset.asymmetry.copy(), dataset.error.copy()),
            (dataset.time.copy(), dataset.asymmetry.copy(), dataset.error.copy()),
        ]
        dataset.run.metadata["period_count"] = 2
        return dataset

    monkeypatch.setattr("asymmetry.core.io.load", _load_two_periods)
    workdir = tmp_path / "period-wd"
    cli.main(
        [
            "reduce",
            str(workflow_folder),
            "--runs",
            str(SCAN_RUNS[0]),
            "--period",
            "green",
            "--json",
            "--workdir",
            str(workdir),
        ]
    )
    data = _json_output(capsys)
    assert data["settings"]["period"] == "green"
    assert data["entries"][0]["run"]["n_periods"] == 2
    assert (
        json.loads((workdir / "manifest.json").read_text(encoding="utf-8"))["settings"]["period"]
        == "green"
    )


def test_survey_reports_the_number_of_selectable_periods(
    workflow_folder: Path, tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    from asymmetry.core.io import load as real_load

    def _load_two_periods(path):
        dataset = real_load(path)
        reduced = (dataset.time.copy(), dataset.asymmetry.copy(), dataset.error.copy())
        dataset.run.grouping["period_reduced"] = [reduced, reduced]
        dataset.run.metadata["period_count"] = 2
        return dataset

    monkeypatch.setattr("asymmetry.core.io.load", _load_two_periods)
    cli.main(
        [
            "survey",
            str(workflow_folder),
            "--json",
            "--workdir",
            str(tmp_path / "survey-period-wd"),
        ]
    )
    data = _json_output(capsys)
    assert all(row["n_periods"] == 2 for row in data["survey"]["runs"])


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


def test_integral_scan_writes_points_for_the_named_runs(
    workflow_folder: Path, tmp_path: Path, capsys
) -> None:
    workdir = tmp_path / "wd"
    cli.main(
        [
            "integral-scan",
            str(workflow_folder),
            "--runs",
            f"{SCAN_RUNS[0]}-{SCAN_RUNS[2]}",
            "--order",
            "run",
            "--name",
            "integral",
            "--json",
            "--workdir",
            str(workdir),
        ]
    )
    data = _json_output(capsys)
    assert [point["run"] for point in data["scan"]["points"]] == list(SCAN_RUNS[:3])
    assert data["scan"]["units"] == "fraction"
    assert (workdir / "scans" / "integral.json").exists()


@pytest.mark.parametrize(
    "fit_only_args",
    [
        ["--initial", "B0=3000"],
        ["--fix", "Bwid=100"],
        ["--baseline", "Constant", "--baseline-regions", "0:1"],
    ],
)
def test_integral_scan_rejects_fit_options_without_a_model(
    workflow_folder: Path,
    tmp_path: Path,
    capsys,
    fit_only_args: list[str],
) -> None:
    with pytest.raises(SystemExit, match="1"):
        cli.main(
            [
                "integral-scan",
                str(workflow_folder),
                "--runs",
                str(SCAN_RUNS[0]),
                *fit_only_args,
                "--workdir",
                str(tmp_path / "wd"),
            ]
        )
    assert "require --model MODEL" in capsys.readouterr().err


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


def test_wizard_takes_the_geometry_the_survey_resolved(
    workflow_folder: Path, tmp_path: Path, capsys
) -> None:
    # Screening one run cannot see the folder's survey unless the command hands
    # it over; without that, a file recording no field state screens blind.
    from asymmetry.cli.commands.wizard import _survey_geometry
    from asymmetry.core.workflow.workdir import WorkDir

    workdir = WorkDir(tmp_path / "wd")
    assert _survey_geometry(workdir, CALIBRATION_RUN) is None

    cli.main(["survey", str(workflow_folder), "--workdir", str(workdir.root)])
    capsys.readouterr()
    assert _survey_geometry(workdir, CALIBRATION_RUN) == "TF"
    assert _survey_geometry(workdir, SCAN_RUNS[0]) == "ZF"
    assert _survey_geometry(workdir, 900) is None


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
    assert stored["schema"] == WORKDIR_SCHEMA
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


def test_fit_global_fits_a_shared_parameter_instead_of_pinning_it(
    workflow_folder: Path, fitting_workdir: Path, capsys
) -> None:
    cli.main(
        [
            "fit-global",
            str(workflow_folder),
            "--runs",
            f"{SCAN_RUNS[0]}-{SCAN_RUNS[1]}",
            "--recipe",
            "relax",
            "--shared",
            "A_bg",
            "--strategy",
            "least_squares",
            "--name",
            "joint",
            "--json",
            "--workdir",
            str(fitting_workdir),
        ]
    )
    data = _json_output(capsys)["global_fit"]
    assert data["shared_params"] == ["A_bg"]
    assert "A_bg" in data["shared"]
    assert len(data["results"]) == 2
    assert (fitting_workdir / "series" / "joint.json").exists()


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


def _fit_scan(workflow_folder: Path, fitting_workdir: Path, *extra: str) -> None:
    """Store the relaxation series ``scan`` over the whole temperature scan."""
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
            *extra,
            "--workdir",
            str(fitting_workdir),
        ]
    )


def test_trend_model_fits_a_trend_column_and_stores_the_fit(
    workflow_folder: Path, fitting_workdir: Path, capsys
) -> None:
    _fit_scan(workflow_folder, fitting_workdir)
    capsys.readouterr()
    cli.main(
        [
            "trend",
            str(workflow_folder),
            "--series",
            "scan",
            "--model",
            "Linear",
            "--param",
            "Lambda",
            "--plot",
            "--json",
            "--workdir",
            str(fitting_workdir),
        ]
    )
    fit = _json_output(capsys)["fit"]

    # The scan was simulated with Lambda = 0.10 + 0.004 T.
    assert fit["success"]
    assert fit["runs"] == list(SCAN_RUNS)
    assert abs(fit["parameters"]["m"] - 0.004) <= 3.0 * fit["uncertainties"]["m"]
    assert abs(fit["parameters"]["b"] - 0.10) <= 3.0 * fit["uncertainties"]["b"]
    stored = json.loads((fitting_workdir / "series" / "scan.json").read_text(encoding="utf-8"))
    assert stored["trend_fits"]["Lambda"] == fit
    assert (fitting_workdir / "plots" / "scan-trend-Lambda.png").exists()


def test_trend_model_prints_the_fit_and_what_it_left_out(
    workflow_folder: Path, fitting_workdir: Path, capsys
) -> None:
    _fit_scan(workflow_folder, fitting_workdir)
    capsys.readouterr()
    cli.main(
        [
            "trend",
            str(workflow_folder),
            "--series",
            "scan",
            "--model",
            "Linear",
            "--param",
            "Lambda",
            "--exclude",
            str(SCAN_RUNS[0]),
            "--xmax",
            "40",
            "--workdir",
            str(fitting_workdir),
        ]
    )
    out = capsys.readouterr().out
    assert (
        "Fit of Linear to Lambda against temperature, over the points' span "
        "20.0000 .. 40.0000: 3 point(s)"
    ) in out
    assert f"{SCAN_RUNS[0]} (excluded)" in out
    assert f"{SCAN_RUNS[-1]} (outside the x range)" in out


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--param", "Lambda"], "--param require --model EXPR"),
        (["--model", "Linear"], "--model needs --param NAME"),
        (["--model", "Linear", "--param", "Nope"], "The trend has no column 'Nope'"),
        (["--model", "Nope", "--param", "Lambda"], "Nope"),
    ],
)
def test_trend_model_refuses_a_malformed_request(
    workflow_folder: Path, fitting_workdir: Path, capsys, arguments: list[str], message: str
) -> None:
    _fit_scan(workflow_folder, fitting_workdir)
    capsys.readouterr()
    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "trend",
                str(workflow_folder),
                "--series",
                "scan",
                *arguments,
                "--workdir",
                str(fitting_workdir),
            ]
        )
    assert exc.value.code == 1
    assert message in capsys.readouterr().err


def test_trend_reads_a_fit_global_series(
    workflow_folder: Path, fitting_workdir: Path, capsys
) -> None:
    """A simultaneous fit's run-local parameters trend like a series' do."""
    cli.main(
        [
            "fit-global",
            str(workflow_folder),
            "--runs",
            f"{SCAN_RUNS[0]}-{SCAN_RUNS[-1]}",
            "--recipe",
            "relax",
            "--shared",
            "A_bg",
            "--order",
            "temperature",
            "--strategy",
            "least_squares",
            "--name",
            "joint",
            "--workdir",
            str(fitting_workdir),
        ]
    )
    out = capsys.readouterr().out
    assert "temperature" in out and "Lambda" in out

    cli.main(
        [
            "trend",
            str(workflow_folder),
            "--series",
            "joint",
            "--json",
            "--workdir",
            str(fitting_workdir),
        ]
    )
    trend = _json_output(capsys)["trend"]
    assert trend["order_key"] == "temperature"
    assert "A_bg" not in trend["columns"]
    assert [row["x"] for row in trend["rows"]] == list(SCAN_TEMPERATURES)


def test_fit_series_orders_along_values_the_analyst_supplies(
    workflow_folder: Path, fitting_workdir: Path, capsys
) -> None:
    runs = SCAN_RUNS[:3]
    values = ",".join(f"{run}={len(runs) - index}" for index, run in enumerate(runs))
    cli.main(
        [
            "fit-series",
            str(workflow_folder),
            "--runs",
            f"{runs[0]}-{runs[-1]}",
            "--recipe",
            "relax",
            "--order",
            "foils",
            "--x",
            values,
            "--json",
            "--workdir",
            str(fitting_workdir),
        ]
    )
    series = _json_output(capsys)["series"]
    assert series["order_key"] == "foils"
    assert [row["run"] for row in series["trend"]["rows"]] == list(reversed(runs))


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--order", "foils"], "'foils' is not recorded in the files"),
        (
            ["--order", "foils", "--x", f"{SCAN_RUNS[0]}=1"],
            f"No foils value for run(s) {SCAN_RUNS[1]}",
        ),
        (["--order", "foils", "--x", "banana"], "--x entry 'banana' is not RUN=VALUE"),
        (["--order", "field", "--x", f"{SCAN_RUNS[0]}=1"], "'field' is read from the files"),
        (["--order", "sample_temperature_logged"], "record no sample_temperature_logged"),
    ],
)
def test_fit_series_refuses_an_axis_it_cannot_build(
    workflow_folder: Path, fitting_workdir: Path, capsys, arguments: list[str], message: str
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
                *arguments,
                "--workdir",
                str(fitting_workdir),
            ]
        )
    assert exc.value.code == 1
    assert message in capsys.readouterr().err


def test_the_order_help_names_every_order_key() -> None:
    from asymmetry.cli._axis import ORDER_HELP
    from asymmetry.core.workflow.series import ORDER_KEYS

    for key in ORDER_KEYS:
        assert key in ORDER_HELP


def test_fourier_writes_arrays_and_quantitative_metadata(
    workflow_folder: Path, fitting_workdir: Path, capsys
) -> None:
    cli.main(
        [
            "fourier",
            str(workflow_folder),
            "--run",
            str(SCAN_RUNS[0]),
            "--name",
            "spectrum",
            "--fmax",
            "10",
            "--json",
            "--workdir",
            str(fitting_workdir),
        ]
    )
    data = _json_output(capsys)
    assert data["run"] == SCAN_RUNS[0]
    assert data["resolution_mhz"] > 0.0
    assert Path(data["array_path"]).exists()
    assert Path(data["metadata_path"]).exists()


@pytest.mark.parametrize("name", ["../escape", "a/b", ""])
def test_a_name_that_is_not_one_path_component_is_a_user_error(
    workflow_folder: Path, fitting_workdir: Path, capsys, name: str
) -> None:
    """Every flag whose value becomes a path under the work directory refuses it.

    Exit 1 with a message, not a traceback and not a file written somewhere
    outside ``recipes/``, ``series/`` or ``plots/``.
    """
    invocations = [
        ["trend", str(workflow_folder), "--series", name],
        ["fit", str(workflow_folder), "--run", str(SCAN_RUNS[0]), "--recipe", name],
        [
            "fit-series",
            str(workflow_folder),
            "--runs",
            f"{SCAN_RUNS[0]}-{SCAN_RUNS[1]}",
            "--recipe",
            "relax",
            "--order",
            "temperature",
            "--name",
            name,
        ],
    ]
    for invocation in invocations:
        with pytest.raises(SystemExit) as exc:
            cli.main([*invocation, "--workdir", str(fitting_workdir)])
        assert exc.value.code == 1, invocation[0]
        assert capsys.readouterr().err.startswith("asymmetry: "), invocation[0]

    # ``recipes/../escape.json`` and ``series/../escape.json`` both resolve here.
    assert not (Path(fitting_workdir) / "escape.json").exists()


# -- dispatcher -------------------------------------------------------------


def test_verbose_flag_is_recognised_by_the_parser() -> None:
    parser = cli.build_parser()
    assert parser.parse_args(["survey", "somefolder"]).verbose is False
    assert parser.parse_args(["--verbose", "survey", "somefolder"]).verbose is True


def test_a_repeated_warning_prints_once_without_verbose(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    import warnings as warnings_module

    class _Dummy:
        def summary(self) -> str:
            return ""

    def _warn_twice(_path: str) -> _Dummy:
        # `simplefilter("always")` bypasses Python's own per-location dedup
        # (the "default" action only shows a warning once per module+lineno),
        # so both calls reach `showwarning` — exactly the repeated-occurrence
        # flooding `--verbose`/its absence is about, reproduced deterministically.
        warnings_module.simplefilter("always")
        warnings_module.warn("boom", UserWarning)
        warnings_module.warn("boom", UserWarning)
        return _Dummy()

    monkeypatch.setattr("asymmetry.core.io.load", _warn_twice)
    cli.main(["info", "unused.nxs"])

    err = capsys.readouterr().err
    assert err.count("asymmetry: warning:") == 1
    assert "boom" in err


def test_verbose_leaves_pythons_own_warning_handling_in_place(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    # `main()` must not touch `warnings.showwarning` at all under `--verbose` —
    # proven here by the warning still reaching pytest's own recorder (which
    # only sees it if nothing upstream of it swallowed or reformatted it),
    # rather than by re-deriving what Python's literal default prints, which
    # pytest's own warnings-capture plugin intercepts before it reaches stderr.
    import warnings as warnings_module

    class _Dummy:
        def summary(self) -> str:
            return ""

    def _warn_once(_path: str) -> _Dummy:
        warnings_module.simplefilter("always")
        warnings_module.warn("boom", UserWarning)
        return _Dummy()

    monkeypatch.setattr("asymmetry.core.io.load", _warn_once)
    with pytest.warns(UserWarning, match="boom"):
        cli.main(["--verbose", "info", "unused.nxs"])

    assert "asymmetry: warning:" not in capsys.readouterr().err


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


def test_wizard_refuses_a_component_it_does_not_have(
    workflow_folder: Path, fitting_workdir: Path, capsys
) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "wizard",
                str(workflow_folder),
                "--run",
                str(SCAN_RUNS[0]),
                "--include",
                "Oscilatory",
                "--workdir",
                str(fitting_workdir),
            ]
        )
    assert exc.value.code == 1
    assert "Unknown component(s) Oscilatory" in capsys.readouterr().err


def test_recipe_writes_a_fittable_recipe_and_names_every_parameter(
    workflow_folder: Path, fitting_workdir: Path, capsys
) -> None:
    cli.main(
        [
            "recipe",
            str(workflow_folder),
            "--expression",
            "Exponential + Constant",
            "--name",
            "hand",
            "--run",
            str(SCAN_RUNS[0]),
            "--initial",
            "Lambda=0.2",
            "--fix",
            "A_bg=0",
            "--workdir",
            str(fitting_workdir),
        ]
    )
    out = capsys.readouterr().out
    for name in ("A_1", "Lambda", "A_bg"):
        assert name in out
    stored = json.loads((fitting_workdir / "recipes" / "hand.json").read_text(encoding="utf-8"))
    by_name = {p["name"]: p for p in stored["parameters"]}
    assert (by_name["Lambda"]["value"], by_name["Lambda"]["fixed"]) == (0.2, False)
    assert (by_name["A_bg"]["value"], by_name["A_bg"]["fixed"]) == (0.0, True)
    assert stored["pinned"] == ["A_bg"]

    cli.main(
        [
            "fit",
            str(workflow_folder),
            "--run",
            str(SCAN_RUNS[0]),
            "--recipe",
            "hand",
            "--workdir",
            str(fitting_workdir),
        ]
    )
    assert "chi2_red" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--expression", "Exponentail"], "Unknown component 'Exponentail'"),
        (["--expression", "Exponential", "--fix", "Lamda=1"], "Lamda is not a parameter"),
        (["--expression", "Exponential", "--initial", "Lambda"], "--initial 'Lambda' is not"),
    ],
)
def test_recipe_refuses_a_model_or_name_it_cannot_build(
    workflow_folder: Path, fitting_workdir: Path, capsys, arguments: list[str], message: str
) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "recipe",
                str(workflow_folder),
                *arguments,
                "--name",
                "bad",
                "--workdir",
                str(fitting_workdir),
            ]
        )
    assert exc.value.code == 1
    assert message in capsys.readouterr().err


def test_wizard_prints_its_spectral_evidence_and_keeps_a_screening_window(
    workflow_folder: Path, tmp_path: Path, capsys
) -> None:
    workdir = tmp_path / "wd"
    cli.main(
        [
            "reduce",
            str(workflow_folder),
            "--runs",
            str(CALIBRATION_RUN),
            "--workdir",
            str(workdir),
        ]
    )
    capsys.readouterr()
    cli.main(
        [
            "wizard",
            str(workflow_folder),
            "--run",
            str(CALIBRATION_RUN),
            "--geometry",
            "TF",
            "--tmax",
            "6",
            "--workdir",
            str(workdir),
        ]
    )
    out = capsys.readouterr().out
    # 100 G precesses at 1.355 MHz; the line and the fitted values are printed.
    assert "Spectral lines: 1.3" in out
    assert "Recommended fit: " in out
    stored = json.loads(
        (workdir / "recipes" / f"wizard-{CALIBRATION_RUN}.json").read_text(encoding="utf-8")
    )
    assert stored["t_max"] == 6.0


def test_fit_series_fits_inside_the_window_it_is_given(
    workflow_folder: Path, fitting_workdir: Path, capsys
) -> None:
    _fit_scan(workflow_folder, fitting_workdir, "--tmax", "5", "--json")
    capsys.readouterr()
    stored = json.loads((fitting_workdir / "series" / "scan.json").read_text(encoding="utf-8"))
    assert stored["recipe"]["t_max"] == 5.0


def test_a_trend_fit_report_scales_errors_and_warns_on_a_multi_component_parameter() -> None:
    from asymmetry.cli.commands.trend import _render_fit

    fit = {
        "param": "Lambda_1",
        "expression": "Redfield",
        "order_key": "field",
        "x_min": None,
        "x_max": None,
        "n_points": 12,
        "success": True,
        "message": "",
        "parameters": {"D": 30.0, "nu": 150.0, "m": 2.0},
        "uncertainties": {"D": 0.5, "nu": 10.0},
        "fixed": ["m"],
        "reduced_chi_squared": 4.0,
        "params_at_bound": [],
        "excluded": [],
        "flagged": [],
        "x_fitted": [1000.0, 38000.0],
        "units": {"D": "MHz", "nu": "MHz", "m": None},
        "turning_point": None,
    }
    text = "\n".join(_render_fit(fit, ["A_1", "Lambda_1", "A_2", "Lambda_2", "A_bg"]))
    # χ²ᵣ = 4 doubles the errors, and the scaled column leads.
    assert "error (x sqrt(chi2_red) = 2)" in text
    assert "unscaled error" in text
    d_row = next(line for line in text.splitlines() if line.startswith("D "))
    assert d_row.split()[2:4] == ["1.000000", "0.500000"]
    assert "NOTE: Lambda_1 is one of several Lambda components" in text
    assert "also Lambda_2" in text

    single = "\n".join(_render_fit(fit | {"reduced_chi_squared": 0.9}, ["A_1", "Lambda_1"]))
    assert "sqrt(chi2_red)" not in single
    assert "NOTE" not in single


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"success": False}, "the fit did not converge"),
        ({"params_at_bound": ["nu"]}, "nu is at a bound"),
        ({"uncertainties": {"D": 45.0, "nu": 10.0}}, "D's scaled error is missing, zero or as"),
        ({"uncertainties": {"D": 0.0, "nu": 10.0}}, "D's scaled error is missing, zero"),
    ],
)
def test_a_trend_law_that_did_not_fit_is_named_as_not_established(changes, reason) -> None:
    from asymmetry.cli.commands.trend import _render_fit

    fit = {
        "param": "Lambda",
        "expression": "Redfield",
        "order_key": "field",
        "x_min": None,
        "x_max": None,
        "n_points": 12,
        "success": True,
        "message": "did not converge",
        "parameters": {"D": 30.0, "nu": 150.0, "m": 2.0},
        "uncertainties": {"D": 0.5, "nu": 10.0},
        "fixed": ["m"],
        "reduced_chi_squared": 1.2,
        "params_at_bound": [],
        "excluded": [],
        "flagged": [],
        "x_fitted": [1000.0, 38000.0],
        "units": {"D": "MHz", "nu": "MHz", "m": None},
        "turning_point": None,
    }
    assert "LAW NOT ESTABLISHED" not in "\n".join(_render_fit(fit, ["Lambda"]))
    text = "\n".join(_render_fit(fit | changes, ["Lambda"]))
    assert "LAW NOT ESTABLISHED" in text
    assert reason in text


@pytest.mark.parametrize(
    ("order_key", "free_params", "expected"),
    [
        ("field", ["A_1", "Lambda", "A_bg"], ["Redfield --param Lambda"]),
        (
            "field",
            ["A_1", "Lambda_1", "Lambda_2"],
            ["Redfield --param Lambda_1", "splits the rate"],
        ),
        ("temperature", ["A_1", "frequency", "Lambda"], ["OrderParameter --param frequency"]),
        ("concentration", ["A_1", "Lambda_2"], ["Linear --param Lambda_2"]),
        ("run", ["A_1"], ["<law> --param <column>"]),
    ],
)
def test_trend_names_the_law_its_axis_and_parameters_call_for(
    order_key, free_params, expected
) -> None:
    from asymmetry.cli.commands.trend import _law_hints
    from asymmetry.core.workflow.series import TrendTable

    trend = TrendTable(order_key=order_key, columns=["run", "x", *free_params], rows=[])
    text = "\n".join(_law_hints("scan", trend, free_params))
    for fragment in expected:
        assert fragment in text


@pytest.mark.parametrize(
    ("frequencies", "expected"),
    [
        # A line falling to zero at the transition is an order parameter.
        ([15.2, 11.0, 2.8], "OrderParameter --param frequency"),
        # One held at the applied field's Larmor frequency is not.
        ([0.285, 0.280, 0.273], "frequency holds at 0.2800 MHz"),
    ],
)
def test_a_frequency_held_along_the_scan_is_not_called_an_order_parameter(
    frequencies, expected
) -> None:
    from asymmetry.cli.commands.trend import _law_hints
    from asymmetry.core.workflow.series import TrendTable

    rows = [
        {"run": run, "x": 50.0 * run, "frequency": value, "sigma": 0.3, "flags": []}
        for run, value in enumerate(frequencies, start=1)
    ]
    trend = TrendTable("temperature", ["run", "x", "frequency", "sigma", "flags"], rows)
    text = "\n".join(_law_hints("tf", trend, ["frequency", "sigma"]))
    assert expected in text


def test_a_trend_law_is_judged_on_its_physical_parameters_and_scaled_errors() -> None:
    from asymmetry.cli.commands.trend import _render_fit

    fit = {
        "param": "Lambda",
        "expression": "CriticalDivergence",
        "order_key": "temperature",
        "n_points": 12,
        "success": True,
        "message": "",
        "parameters": {"a": 12.3, "Tc": 84.44, "nu": 1.78, "c": 0.1},
        "uncertainties": {"a": 13.7, "Tc": 1.63, "nu": 0.76, "c": 0.5},
        "fixed": [],
        "reduced_chi_squared": 1.0,
        "params_at_bound": [],
        "excluded": [],
        "flagged": [],
        "x_fitted": [90.0, 290.0],
        "units": {"Tc": "K"},
        "turning_point": None,
    }
    # An undetermined prefactor or offset does not sink a well-determined Tc.
    text = "\n".join(_render_fit(fit, ["Lambda"]))
    assert "LAW NOT ESTABLISHED" not in text
    assert "Converged" in text
    # ... but scaled errors do: chi2_red 25 makes nu's error 3.8 > 1.78.
    text = "\n".join(_render_fit(fit | {"reduced_chi_squared": 25.0}, ["Lambda"]))
    assert "LAW NOT ESTABLISHED" in text
    assert "nu's scaled error" in text
    assert "Tc's scaled error" not in text
    # The determined parameter is named, with the way to report it.
    assert "Next: Tc is determined and nu not. Hold nu at a textbook value" in text


def test_a_failed_order_parameter_fit_is_pointed_at_its_shape_exponent() -> None:
    from asymmetry.cli.commands.trend import _render_fit

    fit = {
        "param": "frequency",
        "expression": "OrderParameter",
        "order_key": "temperature",
        "n_points": 7,
        "success": False,
        "message": "Fit failed",
        "parameters": {"y0": 27.0, "Tc": 357.7, "beta": 0.38, "alpha": 0.58},
        "uncertainties": {"y0": 0.12, "Tc": 0.009, "beta": 0.0004, "alpha": 0.008},
        "fixed": [],
        "reduced_chi_squared": 158.0,
        "params_at_bound": [],
        "excluded": [],
        "flagged": [],
        "x_fitted": [320.0, 356.0],
        "units": {"Tc": "K"},
        "turning_point": None,
    }
    text = "\n".join(_render_fit(fit, ["frequency"]))
    assert "LAW NOT ESTABLISHED" in text
    assert "refit with --fix alpha=1" in text
    # A free Redfield exponent below zero is no law at all.
    redfield = fit | {
        "expression": "Redfield",
        "success": True,
        "parameters": {"D": 25.9, "nu": 114.0, "m": -1.55},
        "uncertainties": {"D": 0.6, "nu": 17.0, "m": 0.1},
        "reduced_chi_squared": 1.0,
    }
    text = "\n".join(_render_fit(redfield, ["Lambda"]))
    assert "m is not positive" in text
    assert "refit with --fix m=2" in text
    # Once alpha is held, the hint has nothing left to say.
    held = fit | {"fixed": ["alpha"], "success": True}
    assert "--fix alpha" not in "\n".join(_render_fit(held, ["frequency"]))


def test_a_fit_on_a_windowed_reduction_says_so_and_plot_tmax_keeps_the_record(
    workflow_folder: Path, tmp_path: Path, capsys
) -> None:
    from asymmetry.core.workflow.recipe import FitRecipe
    from asymmetry.core.workflow.workdir import WorkDir

    workdir = tmp_path / "wd"
    run = SCAN_RUNS[0]
    base = ["reduce", str(workflow_folder), "--runs", str(run), "--workdir", str(workdir)]
    cli.main([*base, "--plot-tmax", "2", "--plot"])
    full = WorkDir(workdir).reduced(run).n_points
    assert WorkDir(workdir).entry(run).settings.t_max is None

    cli.main([*base, "--tmax", "2"])
    assert WorkDir(workdir).reduced(run).n_points < full
    WorkDir(workdir).write_recipe(
        "relax",
        FitRecipe.from_expression("Exponential + Constant", dataset=WorkDir(workdir).reduced(run)),
    )
    capsys.readouterr()
    cli.main(
        [
            "fit",
            str(workflow_folder),
            "--run",
            str(run),
            "--recipe",
            "relax",
            "--workdir",
            str(workdir),
        ]
    )
    assert (
        f"NOTE: run(s) {run} were reduced to a time window (start-2.0 µs)"
        in capsys.readouterr().out
    )
