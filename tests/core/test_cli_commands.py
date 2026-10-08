"""Tests for the ``survey``/``alpha``/``reduce`` CLI subcommands."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import pytest

from asymmetry import __version__, cli
from asymmetry.cli._output import SCHEMA, UserError
from asymmetry.cli._runs import parse_run_spec, resolve_run, resolve_runs, run_files
from asymmetry.core.data.dataset import Run
from asymmetry.core.workflow.workdir import SCHEMA as WORKDIR_SCHEMA
from asymmetry.core.workflow.workdir import RunSelection
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
    every_run = RunSelection(workflow_folder, None)
    resolved = resolve_runs(every_run, f"{SCAN_RUNS[0]}-{SCAN_RUNS[1]},900")
    assert [run for run, _prefix, _path in resolved] == [SCAN_RUNS[0], SCAN_RUNS[1]]
    assert {prefix for _run, prefix, _path in resolved} == {"SIM"}
    with pytest.raises(UserError):
        resolve_runs(every_run, "900-910")


def test_resolve_run_reports_a_missing_run(workflow_folder: Path) -> None:
    every_run = RunSelection(workflow_folder, None)
    assert resolve_run(every_run, CALIBRATION_RUN).exists()
    with pytest.raises(UserError):
        resolve_run(every_run, 900)


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


@pytest.fixture
def two_instruments(tmp_path: Path) -> Path:
    """A folder where ``EMU``/``emu`` (one instrument across eras) and ``MUSR`` share run 42."""
    pytest.importorskip("h5py")
    folder = tmp_path / "two"
    folder.mkdir()
    _write_run_file(folder, "EMU", 41)
    _write_run_file(folder, "emu", 42)
    _write_run_file(folder, "MUSR", 42)
    _write_run_file(folder, "MUSR", 43)
    return folder


def test_a_run_number_under_two_instruments_is_refused_naming_both(two_instruments: Path) -> None:
    """The work directory is keyed on the run number, so a clash needs an instrument."""
    with pytest.raises(UserError) as exc:
        run_files(RunSelection(two_instruments, None))

    message = str(exc.value)
    assert "EMU and MUSR" in message
    assert "MUSR00000042.nxs and emu00000042.nxs" in message
    assert "--instrument EMU or --instrument MUSR" in message


def test_an_instrument_selects_its_files_in_any_case(two_instruments: Path) -> None:
    emu = run_files(RunSelection(two_instruments, "EMU"))
    assert {run: path.name for run, (_prefix, path) in emu.items()} == {
        41: "EMU00000041.nxs",
        42: "emu00000042.nxs",
    }
    assert resolve_run(RunSelection(two_instruments, "MUSR"), 42).name == "MUSR00000042.nxs"
    with pytest.raises(UserError, match="holds no HIFI run files"):
        run_files(RunSelection(two_instruments, "HIFI"))


def test_one_run_number_twice_within_an_instrument_cannot_be_split_by_instrument(
    tmp_path: Path,
) -> None:
    # Two extensions, not two cases: macOS folders are case-insensitive by default.
    for name in ("EMU00000042.bin", "EMU00000042.nxs"):
        (tmp_path / name).touch()

    with pytest.raises(UserError) as exc:
        run_files(RunSelection(tmp_path, "EMU"))

    assert "EMU00000042.bin and EMU00000042.nxs" in str(exc.value)
    assert "move one of them out" in str(exc.value)


@pytest.mark.parametrize(
    "command",
    [
        ["reduce", "--runs", "42"],
        ["integral-scan", "--runs", "41-43"],
        ["alpha", "--run", "42"],
        ["reduce", "--runs", "41", "--alpha-from", "42"],
    ],
)
def test_every_command_reading_run_files_refuses_a_clash_without_an_instrument(
    two_instruments: Path, tmp_path: Path, command: list[str], capsys
) -> None:
    extra = [] if command[0] == "alpha" else ["--workdir", str(tmp_path / "wd")]
    with pytest.raises(SystemExit) as exc:
        cli.main([command[0], str(two_instruments), *command[1:], *extra])

    assert exc.value.code == 1
    error = capsys.readouterr().err
    assert "--instrument EMU or --instrument MUSR" in error


def test_reduce_and_fit_one_instrument_and_record_it(
    two_instruments: Path, tmp_path: Path, capsys
) -> None:
    workdir = tmp_path / "wd"
    cli.main(
        ["reduce", str(two_instruments), "--runs", "41-43", "--instrument", "emu"]
        + ["--json", "--workdir", str(workdir)]
    )

    entries = _json_output(capsys)["entries"]
    assert [(e["run_number"], Path(e["source_file"]).name) for e in entries] == [
        (41, "EMU00000041.nxs"),
        (42, "emu00000042.nxs"),
    ]
    manifest = json.loads((workdir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["instrument"] == "EMU"

    # Commands that only read the work directory take the instrument it holds.
    cli.main(
        ["recipe", str(two_instruments), "--expression", "Exponential + Constant"]
        + ["--run", "42", "--name", "relax", "--workdir", str(workdir)]
    )
    cli.main(
        ["fit", str(two_instruments), "--run", "42", "--recipe", "relax", "--json"]
        + ["--workdir", str(workdir)]
    )
    capsys.readouterr()


def test_a_second_instrument_cannot_share_the_work_directory(
    two_instruments: Path, tmp_path: Path, capsys
) -> None:
    workdir = tmp_path / "wd"
    cli.main(
        ["reduce", str(two_instruments), "--runs", "42", "--instrument", "EMU"]
        + ["--workdir", str(workdir)]
    )
    capsys.readouterr()

    with pytest.raises(SystemExit) as exc:
        cli.main(
            ["reduce", str(two_instruments), "--runs", "42", "--instrument", "MUSR"]
            + ["--workdir", str(workdir)]
        )

    assert exc.value.code == 1
    error = capsys.readouterr().err
    assert "(EMU)" in error
    assert "(MUSR)" in error
    assert "--workdir" in error


def test_survey_lists_both_instruments_and_measures_alpha_per_file(
    two_instruments: Path, tmp_path: Path, capsys
) -> None:
    cli.main(["survey", str(two_instruments), "--workdir", str(tmp_path / "wd")])
    out = capsys.readouterr().out

    assert "EMU 42" in out
    assert "MUSR 42" in out
    assert "RUN NUMBERS COLLIDE: EMU and MUSR" in out

    cli.main(
        ["survey", str(two_instruments), "--instrument", "MUSR", "--json"]
        + ["--workdir", str(tmp_path / "wd-musr")]
    )
    rows = _json_output(capsys)["survey"]["runs"]
    assert [row["file"] for row in rows] == ["MUSR00000042.nxs", "MUSR00000043.nxs"]


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
    assert survey["notes_scans"] == stored["notes_scans"] == []


def test_survey_measures_precession_on_the_named_pair(
    workflow_folder: Path, tmp_path: Path, capsys
) -> None:
    cli.main(["survey", str(workflow_folder), "--pair", "2/1", "--workdir", str(tmp_path / "wd")])
    out = capsys.readouterr().out
    # Swapping the pair negates the signal; the calibration line is still there.
    assert "precession measured on the 2/1 pair" in out
    assert f"run {CALIBRATION_RUN} (best)" in out

    with pytest.raises(SystemExit, match="1"):
        cli.main(
            ["survey", str(workflow_folder), "--pair", "Up/Down", "--workdir", str(tmp_path / "wd")]
        )
    assert "no group 'Up'" in capsys.readouterr().err


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


def test_survey_marks_a_coil_geometry_and_explains_a_psi_header_temperature(
    workflow_folder: Path, tmp_path: Path
) -> None:
    from dataclasses import replace

    from asymmetry.cli.commands.survey import _render
    from asymmetry.core.io.psi import PSI_HEADER_SAMPLE_SENSOR
    from asymmetry.core.workflow.survey import survey_folder

    survey = survey_folder(workflow_folder)
    assert "header sensor 1" not in _render(survey, tmp_path / "survey.json")

    row = replace(
        survey.row(DECOUPLING_RUN),
        geometry="LF",
        geometry_source="coils",
        sample_temperature_logged=52.76,
        sample_temperature_log_source=PSI_HEADER_SAMPLE_SENSOR,
    )
    text = _render(replace(survey, runs=[row]), tmp_path / "survey.json")
    line = next(line for line in text.splitlines() if line.startswith(f"{DECOUPLING_RUN} "))
    assert "LF+" in line
    assert "52.76" in line
    assert "geom+: geometry read from the run's logged field-coil readbacks" in text
    assert "T log (PSI): header sensor 1, an unlabelled sensor inferred" in text


def test_survey_scans_block_names_the_instrument(
    workflow_folder: Path, tmp_path: Path, capsys
) -> None:
    cli.main(["survey", str(workflow_folder), "--workdir", str(tmp_path / "wd")])
    out = capsys.readouterr().out
    assert "temperature scan, SIM, ZF, B = 0 G" in out


def test_survey_names_repeats_to_co_add_and_a_scan_the_files_do_not_record(
    workflow_folder: Path, tmp_path: Path
) -> None:
    from dataclasses import replace

    from asymmetry.cli.commands.survey import _render
    from asymmetry.core.workflow.survey import RepeatSet, survey_folder

    survey = survey_folder(workflow_folder)
    repeats = [
        RepeatSet("SIM", 300.0, 3000.0, "", [501, 502, 503, 504, 505], co_add=True),
        RepeatSet("SIM", 291.0, -100.0, "P scan", [601, 602], co_add=False),
    ]
    text = _render(replace(survey, repeats=repeats), tmp_path / "survey.json")
    assert (
        "REPEATS: runs 501-505 repeat one condition (3000 G, 300 K) — co-add them for "
        f"statistics before a spectrum: asymmetry reduce {workflow_folder} --runs 501-505 "
        "--coadd --workdir asymmetry-work-coadd"
    ) in text
    assert (
        'UNRECORDED SCAN: runs 601-602 record one condition (-100 G, 291 K, notes "P scan"), '
        "but their note names a scan"
    ) in text
    assert "REPEATS" not in _render(survey, tmp_path / "survey.json")


def test_survey_names_a_scan_written_in_the_notes_with_the_command_to_fit_it(
    workflow_folder: Path, tmp_path: Path
) -> None:
    from dataclasses import replace

    from asymmetry.cli.commands.survey import _render
    from asymmetry.core.workflow.survey import NotesScan, survey_folder

    survey = survey_folder(workflow_folder)
    scan = NotesScan(
        instrument="SIM",
        temperature=295.0,
        field=100.0,
        source="notes",
        template="Steering <x> A",
        runs=[803, 801, 802],
        values=[-0.5, 0.0, 0.5],
    )
    text = _render(replace(survey, notes_scans=[scan]), tmp_path / "survey.json")
    assert "NOTES SCANS: each set of runs below holds one temperature and field" in text
    assert "fit the series against that quantity, not against temperature" in text
    assert (
        '  runs 801-803 (295 K, 100 G), notes "Steering <x> A": steering from -0.5 to 0.5 on 3 runs'
    ) in text
    assert f"      asymmetry wizard {workflow_folder} --run 801\n" in text
    assert (
        f"      asymmetry fit-series {workflow_folder} --runs 801-803 --recipe wizard-801 "
        "--order steering --x 801=0,802=0.5,803=-0.5 --start 801\n"
    ) in text
    assert "NOTES SCANS" not in _render(survey, tmp_path / "survey.json")


def test_survey_points_a_red_green_field_scan_at_the_period_difference(
    workflow_folder: Path, tmp_path: Path
) -> None:
    from dataclasses import replace

    from asymmetry.cli.commands.survey import _render
    from asymmetry.core.workflow.survey import ScanGroup, survey_folder

    scan = ScanGroup(
        axis="field",
        instrument="SIM",
        geometry="LF",
        geometry_note="",
        temperature=300.0,
        field=None,
        runs=[701, 702, 703],
        values=[28500.0, 28600.0, 28700.0],
        n_periods=2,
    )
    survey = survey_folder(workflow_folder)
    text = _render(replace(survey, scans=[scan]), tmp_path / "survey.json")
    assert "T = 300 K, 2 periods (red/green): 3 runs" in text
    assert (
        f"asymmetry integral-scan {workflow_folder} --runs 701-703 --period green-red; with a "
        "field step between the periods (differential ALC) fit --model LorentzianLCRPair "
        "with its dB held at the value that command's Next line gives"
    ) in text
    # A single-period scan says neither.
    text = _render(replace(survey, scans=[replace(scan, n_periods=1)]), tmp_path / "s.json")
    assert "periods" not in text.split("Scans:")[1]
    assert "green-red" not in text


def test_survey_candidate_block_names_the_source_of_each_candidate(
    workflow_folder: Path, tmp_path: Path, capsys
) -> None:
    cli.main(["survey", str(workflow_folder), "--workdir", str(tmp_path / "wd")])
    out = capsys.readouterr().out
    assert f"run {CALIBRATION_RUN} (best) [measured]" in out
    assert "Larmor frequency" in out
    assert f"run {DECOUPLING_RUN}" not in out.split("Alpha-calibration candidates")[1]


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


def test_every_command_with_a_work_directory_or_run_files_takes_an_instrument() -> None:
    parser = cli.build_parser()
    subparsers = next(
        action
        for action in parser._actions  # noqa: SLF001 — argparse exposes no public walk
        if isinstance(action, argparse._SubParsersAction)
    )
    dests = {
        name: {action.dest for action in subparser._actions}  # noqa: SLF001
        for name, subparser in subparsers.choices.items()
    }
    with_instrument = {name for name, found in dests.items() if "instrument" in found}
    # `audit` reads work directories' output logs; it binds none to a folder.
    with_workdir = {name for name, found in dests.items() if "workdir" in found} - {"audit"}
    assert with_instrument == with_workdir | {"alpha"}


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


def test_survey_of_a_folder_with_only_subfolders_names_them_and_writes_nothing(
    tmp_path: Path, capsys
) -> None:
    data = tmp_path / "data"
    rg = data / "RG"
    rg.mkdir(parents=True)
    (rg / "SIM00000700.nxs").touch()
    (rg / "SIM00000701.nxs").touch()

    with pytest.raises(SystemExit) as exc:
        cli.main(["survey", str(data), "--workdir", str(tmp_path / "wd")])
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "RG (2 runs)" in err
    assert not (tmp_path / "wd").exists()


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
    # ``reduce`` loads through the reduction module, which binds the loader at import.
    monkeypatch.setattr("asymmetry.core.workflow.reduction.load", _load_two_periods)
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


def _two_identical_periods(
    monkeypatch: pytest.MonkeyPatch, single_period_run: int | None = None
) -> None:
    """Make every loaded run (bar *single_period_run*) a two-period run with red equal to green."""
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
        if dataset.run_number == single_period_run:
            return dataset
        dataset.run.grouping["period_histograms"] = [
            [_clone(histogram) for histogram in dataset.run.histograms] for _period in range(2)
        ]
        reduced = (dataset.time.copy(), dataset.asymmetry.copy(), dataset.error.copy())
        dataset.run.grouping["period_reduced"] = [reduced, reduced]
        dataset.run.metadata["period_count"] = 2
        return dataset

    monkeypatch.setattr("asymmetry.core.io.load", _load_two_periods)
    # ``reduce`` loads through the reduction module, which binds the loader at import.
    monkeypatch.setattr("asymmetry.core.workflow.reduction.load", _load_two_periods)


def test_reduce_green_red_is_the_difference_of_the_two_periods(
    workflow_folder: Path, tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    _two_identical_periods(monkeypatch)
    cli.main(
        [
            "reduce",
            str(workflow_folder),
            "--runs",
            str(SCAN_RUNS[0]),
            "--period",
            "green-red",
            "--json",
            "--workdir",
            str(tmp_path / "wd"),
        ]
    )
    data = _json_output(capsys)
    assert data["settings"]["period"] == "green_minus_red"
    assert data["entries"][0]["a0_percent"] == pytest.approx(0.0, abs=1e-12)


def test_a_single_period_calibration_run_calibrates_a_period_reduction(
    workflow_folder: Path, tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    # HiFi calibration runs are single-period beside a red/green scan; the
    # detector balance they measure holds in every period.
    _two_identical_periods(monkeypatch, single_period_run=CALIBRATION_RUN)
    cli.main(["alpha", str(workflow_folder), "--run", str(CALIBRATION_RUN), "--json"])
    alpha = _json_output(capsys)["alpha"]["alpha"]
    cli.main(
        [
            "reduce",
            str(workflow_folder),
            "--runs",
            str(SCAN_RUNS[0]),
            "--period",
            "red",
            "--alpha-from",
            str(CALIBRATION_RUN),
            "--json",
            "--workdir",
            str(tmp_path / "wd"),
        ]
    )
    assert _json_output(capsys)["settings"]["alpha"] == pytest.approx(alpha)


def test_integral_scan_green_red_needs_two_periods(
    workflow_folder: Path, tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = [
        "integral-scan",
        str(workflow_folder),
        "--runs",
        f"{SCAN_RUNS[0]}-{SCAN_RUNS[1]}",
        "--period",
        "green-red",
        "--order",
        "run",
        "--json",
        "--workdir",
        str(tmp_path / "wd"),
    ]
    with pytest.raises(SystemExit, match="1"):
        cli.main(args)
    assert "two-period" in capsys.readouterr().err

    _two_identical_periods(monkeypatch)
    cli.main(args)
    data = _json_output(capsys)
    assert [point["value"] for point in data["scan"]["points"]] == pytest.approx([0.0, 0.0])
    assert data["settings"]["period"] == "green_minus_red"


def test_integral_scan_green_red_suggests_holding_a_pair_at_the_period_field_offset(
    workflow_folder: Path, tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    from asymmetry.core import io

    _two_identical_periods(monkeypatch)
    two_period_load = io.load
    # A probe reading -1.25 per gauss about a 950-unit zero offset, at two fields:
    # the step converts by that slope, which one run's ratio of means would miss.
    logs = iter([(9000.0, 43.0), (10000.0, 45.0)])

    def _with_offset(path):
        dataset = two_period_load(path)
        main, step = next(logs)
        dataset.run.metadata["nexus_time_series"] = {
            "Field_Main": {"values": [main]},
            "Field_Hall_Z": {"values": [950.0 - 1.25 * main]},
        }
        dataset.run.metadata["period_hall_offset"] = step * 1.25
        return dataset

    monkeypatch.setattr("asymmetry.core.io.load", _with_offset)
    base = [
        "integral-scan",
        str(workflow_folder),
        "--runs",
        f"{SCAN_RUNS[0]}-{SCAN_RUNS[1]}",
        "--period",
        "green-red",
        "--order",
        "run",
        "--model",
        "LorentzianLCRPair",
        "--fix",
        "f=0",
        "--fix",
        "Bwid=1",
        "--fix",
        "B0=102.5",
        "--workdir",
        str(tmp_path / "wd"),
    ]
    cli.main(base)
    out = capsys.readouterr().out
    assert "period field offset (red - green): -44.00 G, mean of 2 run(s)" in out
    assert "--fix dB=44.00" in out
    # Each fitted parameter is printed with its error; a held one says so.
    bwid = next(line.split() for line in out.splitlines() if line.startswith("Bwid "))
    assert bwid[1:] == ["1.000000", "fixed"]

    logs = iter([(9000.0, 43.0), (10000.0, 45.0)])
    cli.main([*base, "--json"])
    data = _json_output(capsys)
    assert data["period_field_offset"] == {"gauss": pytest.approx(-44.0), "runs": 2}
    # The Next line survives --json.
    assert any("--fix dB=44.00" in note for note in data["notes"])


def test_integral_scan_of_two_period_runs_without_a_period_says_it_summed_them(
    workflow_folder: Path, tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    _two_identical_periods(monkeypatch)
    base = [
        "integral-scan",
        str(workflow_folder),
        "--runs",
        f"{SCAN_RUNS[0]}-{SCAN_RUNS[1]}",
        "--order",
        "run",
        "--workdir",
        str(tmp_path / "wd"),
    ]
    cli.main(base)
    assert (
        f"NOTE: runs {SCAN_RUNS[0]}-{SCAN_RUNS[1]} are two-period (red/green) runs, and "
        "without --period this scan summed both periods"
    ) in capsys.readouterr().out
    cli.main([*base, "--period", "green-red"])
    assert "summed both periods" not in capsys.readouterr().out


def test_integral_scan_single_period_reports_the_source_run_number(
    workflow_folder: Path, tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A period run's own number is encoded (run*1000+period); the scan reports
    the source run it was cut from, not that internal key.
    """
    _two_identical_periods(monkeypatch)
    cli.main(
        [
            "integral-scan",
            str(workflow_folder),
            "--runs",
            f"{SCAN_RUNS[0]}-{SCAN_RUNS[1]}",
            "--period",
            "red",
            "--order",
            "run",
            "--json",
            "--workdir",
            str(tmp_path / "wd"),
        ]
    )
    data = _json_output(capsys)
    assert [point["run"] for point in data["scan"]["points"]] == [SCAN_RUNS[0], SCAN_RUNS[1]]


def test_alpha_period_red_reports_the_source_run_number(
    workflow_folder: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    _two_identical_periods(monkeypatch)
    cli.main(
        [
            "alpha",
            str(workflow_folder),
            "--run",
            str(SCAN_RUNS[0]),
            "--period",
            "red",
            "--json",
        ]
    )
    data = _json_output(capsys)
    assert data["alpha"]["run_number"] == SCAN_RUNS[0]


def test_reduce_with_a_pair_and_offsets_records_them(
    workflow_folder: Path, tmp_path: Path, capsys
) -> None:
    cli.main(
        [
            "reduce",
            str(workflow_folder),
            "--runs",
            str(SCAN_RUNS[0]),
            "--pair",
            "2/1",
            "--t0-offset",
            "2",
            "--t-good-offset",
            "5",
            "--workdir",
            str(tmp_path / "wd"),
        ]
    )
    out = capsys.readouterr().out
    assert "pair 2/1, t0 offset 2, t_good offset 5" in out
    entry = json.loads((tmp_path / "wd" / "reduced" / f"{SCAN_RUNS[0]}.json").read_text())
    assert (entry["forward_group"], entry["backward_group"]) == (2, 1)
    assert entry["settings"]["pair"] == ["2", "1"]


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--pair", "Up/Down"], "no group 'Up'"),
        (["--pair", "Up"], "--pair takes FWD/BWD"),
        (["--background", "range"], "no pre-t0 region"),
        (["--background", "range:9"], "FIRST:LAST"),
        (["--background", "fixed"], "Unknown background mode 'fixed'"),
    ],
)
def test_reduce_rejects_reduction_options_the_run_cannot_take(
    workflow_folder: Path, tmp_path: Path, capsys, extra: list[str], message: str
) -> None:
    with pytest.raises(SystemExit, match="1"):
        cli.main(
            [
                "reduce",
                str(workflow_folder),
                "--runs",
                str(SCAN_RUNS[0]),
                *extra,
                "--workdir",
                str(tmp_path / "wd"),
            ]
        )
    assert message in capsys.readouterr().err


def test_alpha_is_estimated_on_the_named_pair(workflow_folder: Path, capsys) -> None:
    cli.main(
        ["alpha", str(workflow_folder), "--run", str(CALIBRATION_RUN), "--pair", "2/1", "--json"]
    )
    data = _json_output(capsys)
    assert data["alpha"]["forward_group"] == 2
    assert data["alpha"]["alpha"] == pytest.approx(1.0 / CALIBRATION_ALPHA, rel=0.01)


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


def test_integral_scan_subtracts_the_requested_background(
    workflow_folder: Path, tmp_path: Path, capsys
) -> None:
    def scan(*extra: str) -> dict:
        cli.main(
            [
                "integral-scan",
                str(workflow_folder),
                "--runs",
                f"{SCAN_RUNS[0]}-{SCAN_RUNS[2]}",
                *extra,
                "--json",
                "--workdir",
                str(tmp_path / "wd"),
            ]
        )
        return _json_output(capsys)

    plain = scan()
    subtracted = scan("--background", "tail_fit")
    assert subtracted["settings"]["background"] == "tail_fit"
    # The fitted level's correlated error only ever adds to the integral's.
    for before, after in zip(plain["scan"]["points"], subtracted["scan"]["points"], strict=True):
        assert after["error"] > before["error"]


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
    every_run = RunSelection(workflow_folder, None)
    assert _survey_geometry(workdir, every_run, CALIBRATION_RUN) is None

    cli.main(["survey", str(workflow_folder), "--workdir", str(workdir.root)])
    capsys.readouterr()
    assert _survey_geometry(workdir, every_run, CALIBRATION_RUN) == "TF"
    assert _survey_geometry(workdir, every_run, SCAN_RUNS[0]) == "ZF"
    assert _survey_geometry(workdir, every_run, 900) is None
    assert _survey_geometry(workdir, RunSelection(workflow_folder, "EMU"), SCAN_RUNS[0]) is None


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
    assert "Unknown scope" in capsys.readouterr().err


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
    assert lines[0].split(",")[:2] == ["key", "x"]
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
    assert fit["keys"] == [str(run) for run in SCAN_RUNS]
    assert abs(fit["parameters"]["m"] - 0.004) <= 3.0 * fit["uncertainties"]["m"]
    assert abs(fit["parameters"]["b"] - 0.10) <= 3.0 * fit["uncertainties"]["b"]
    stored = json.loads((fitting_workdir / "series" / "scan.json").read_text(encoding="utf-8"))
    assert stored["trend_fits"]["Lambda:Linear"] == fit
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
    # What the trend shows is still named beside the law fitted to one column.
    assert "NOTE: Lambda changes along the scan" in out


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
    assert [row["key"] for row in series["trend"]["rows"]] == [str(run) for run in reversed(runs)]


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


def _cli(workflow_folder: Path, fitting_workdir: Path, command: str, *arguments: str) -> None:
    cli.main([command, str(workflow_folder), *arguments, "--workdir", str(fitting_workdir)])


def _refused(workflow_folder: Path, fitting_workdir: Path, capsys, *arguments: str) -> str:
    """Run a command expected to exit 1; its stderr."""
    capsys.readouterr()
    with pytest.raises(SystemExit) as exc:
        _cli(workflow_folder, fitting_workdir, *arguments)
    assert exc.value.code == 1
    return capsys.readouterr().err


#: Two groups of two scan runs, at mean setpoints 15 K and 35 K.
_GROUPS = f"{SCAN_RUNS[0]},{SCAN_RUNS[1]};{SCAN_RUNS[2]},{SCAN_RUNS[3]}"


def _fit_batch(workflow_folder: Path, fitting_workdir: Path, *extra: str) -> None:
    """Store the batch ``batch`` of :data:`_GROUPS`, the relaxation rate shared per group."""
    _cli(
        workflow_folder,
        fitting_workdir,
        "fit-global",
        "--groups",
        _GROUPS,
        "--recipe",
        "relax",
        "--shared",
        "Lambda,A_bg",
        "--strategy",
        "least_squares",
        "--name",
        "batch",
        *extra,
    )


def test_fit_global_groups_stores_each_group_and_their_shared_trend(
    workflow_folder: Path, fitting_workdir: Path, capsys
) -> None:
    from tests.core.conftest import scan_rate

    _fit_batch(workflow_folder, fitting_workdir, "--json")
    batch = _json_output(capsys)["global_batch"]

    assert batch["kind"] == "global-batch"
    assert [member["series"] for member in batch["members"]] == ["batch-1", "batch-2"]
    assert [group["name"] for group in batch["groups"]] == ["batch-1", "batch-2"]
    trend = batch["trend"]
    assert trend["order_key"] == "temperature"
    assert trend["columns"] == ["key", "x", "Lambda", "Lambda_err", "A_bg", "A_bg_err", "flags"]
    assert [row["key"] for row in trend["rows"]] == ["batch-1", "batch-2"]
    assert [row["x"] for row in trend["rows"]] == [15.0, 35.0]
    # A rate shared by two runs of a linear law sits at the law's value at their mean.
    for row in trend["rows"]:
        assert row["Lambda"] == pytest.approx(scan_rate(row["x"]), abs=0.02)
    for member in ("batch-1", "batch-2"):
        stored = json.loads((fitting_workdir / "series" / f"{member}.json").read_text("utf-8"))
        assert stored["kind"] == "global"

    _cli(workflow_folder, fitting_workdir, "trend", "--series", "batch")
    assert "batch-2" in capsys.readouterr().out


def test_a_batch_orders_its_groups_along_values_the_analyst_supplies(
    workflow_folder: Path, fitting_workdir: Path, capsys
) -> None:
    _fit_batch(workflow_folder, fitting_workdir, "--group-order", "dose", "--group-x", "1=5,2=1")
    capsys.readouterr()
    stored = json.loads((fitting_workdir / "series" / "batch.json").read_text("utf-8"))
    assert stored["order_key"] == "dose"
    assert [(row["key"], row["x"]) for row in stored["trend"]["rows"]] == [
        ("batch-2", 1.0),
        ("batch-1", 5.0),
    ]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--group-order", "dose"], "'dose' is not recorded in the files"),
        (
            ["--group-order", "dose", "--group-x", "1=5,2=1,3=2"],
            "dose values given for group(s) batch-3, which are not in the series",
        ),
        (["--group-order", "dose", "--group-x", "one=5"], "--group-x entry 'one=5' is not GROUP"),
        (
            ["--groups", f"{SCAN_RUNS[0]},{SCAN_RUNS[1]};{SCAN_RUNS[1]},{SCAN_RUNS[2]}"],
            "in two groups",
        ),
        (["--groups", f"{SCAN_RUNS[0]},{SCAN_RUNS[1]};{SCAN_RUNS[2]}"], "too few in batch-2"),
        (["--runs", f"{SCAN_RUNS[0]},{SCAN_RUNS[1]}", "--group-x", "1=5"], "order a batch"),
    ],
)
def test_fit_global_refuses_a_batch_it_cannot_build(
    workflow_folder: Path, fitting_workdir: Path, capsys, arguments: list[str], message: str
) -> None:
    if "--groups" not in arguments and "--runs" not in arguments:
        arguments = ["--groups", _GROUPS, *arguments]
    err = _refused(
        workflow_folder,
        fitting_workdir,
        capsys,
        "fit-global",
        *arguments,
        "--recipe",
        "relax",
        "--shared",
        "A_bg",
        "--name",
        "batch",
    )
    assert message in err


def test_a_batch_whose_group_is_refitted_is_stale(
    workflow_folder: Path, fitting_workdir: Path, capsys
) -> None:
    _fit_batch(workflow_folder, fitting_workdir)
    _cli(
        workflow_folder,
        fitting_workdir,
        "fit-global",
        "--runs",
        f"{SCAN_RUNS[0]},{SCAN_RUNS[1]}",
        "--recipe",
        "relax",
        "--shared",
        "A_bg",
        "--strategy",
        "least_squares",
        "--name",
        "batch-1",
    )
    err = _refused(workflow_folder, fitting_workdir, capsys, "trend", "--series", "batch")
    assert "Series 'batch' is stale: its member(s) batch-1 were refitted" in err


#: Three overlapping stretches of the temperature scan, at mean setpoints 20, 30 and 40 K.
_STRETCHES = {"s1": SCAN_RUNS[0:3], "s2": SCAN_RUNS[1:4], "s3": SCAN_RUNS[2:5]}


def _fit_stretches(workflow_folder: Path, fitting_workdir: Path) -> None:
    """Store each stretch as a series, with Linear fitted to its relaxation rate."""
    for name, runs in _STRETCHES.items():
        _cli(
            workflow_folder,
            fitting_workdir,
            "fit-series",
            "--runs",
            ",".join(map(str, runs)),
            "--recipe",
            "relax",
            "--order",
            "temperature",
            "--name",
            name,
        )
        _cli(
            workflow_folder,
            fitting_workdir,
            "trend",
            "--series",
            name,
            "--model",
            "Linear",
            "--param",
            "Lambda",
        )


def test_trend_from_fits_tabulates_a_law_parameter_per_series_and_fits_it(
    workflow_folder: Path, fitting_workdir: Path, capsys
) -> None:
    _fit_stretches(workflow_folder, fitting_workdir)
    capsys.readouterr()
    _cli(
        workflow_folder,
        fitting_workdir,
        "trend",
        "--series",
        "slopes",
        "--from-fits",
        "s1,s2,s3",
        "--param",
        "m",
        "--model",
        "Linear",
        "--json",
    )
    data = _json_output(capsys)

    trend = data["trend"]
    assert trend["order_key"] == "temperature"
    assert [(row["key"], row["x"]) for row in trend["rows"]] == [
        ("s1", 20.0),
        ("s2", 30.0),
        ("s3", 40.0),
    ]
    # Every stretch of the scan has the slope it was simulated with, 0.004 per K.
    for row in trend["rows"]:
        assert row["m"] == pytest.approx(0.004, abs=3.0 * row["m_err"])
    fit = data["fit"]
    assert fit["keys"] == ["s1", "s2", "s3"]
    assert fit["parameters"]["b"] == pytest.approx(0.004, abs=3.0 * fit["uncertainties"]["b"])
    stored = json.loads((fitting_workdir / "series" / "slopes.json").read_text("utf-8"))
    assert stored["kind"] == "fit-trend"
    assert [member["fit"] for member in stored["members"]] == ["Lambda:Linear"] * 3
    assert stored["trend_fits"]["m:Linear"] == fit

    # A member series refitted leaves the trend built from it stale.
    _cli(
        workflow_folder,
        fitting_workdir,
        "fit-series",
        "--runs",
        ",".join(map(str, _STRETCHES["s2"])),
        "--recipe",
        "relax",
        "--order",
        "temperature",
        "--tmax",
        "6",
        "--name",
        "s2",
    )
    err = _refused(workflow_folder, fitting_workdir, capsys, "trend", "--series", "slopes")
    assert "its member(s) s2 were refitted" in err


def test_trend_from_fits_takes_supplied_values_and_excludes_by_series(
    workflow_folder: Path, fitting_workdir: Path, capsys
) -> None:
    _fit_stretches(workflow_folder, fitting_workdir)
    capsys.readouterr()
    _cli(
        workflow_folder,
        fitting_workdir,
        "trend",
        "--series",
        "slopes",
        "--from-fits",
        "s1,s2,s3",
        "--param",
        "m",
        "--fit",
        "Lambda",
        "--order",
        "pressure",
        "--x",
        "s1=3,s2=2,s3=1",
        "--json",
    )
    trend = _json_output(capsys)["trend"]
    assert trend["order_key"] == "pressure"
    assert [row["key"] for row in trend["rows"]] == ["s3", "s2", "s1"]

    err = _refused(
        workflow_folder,
        fitting_workdir,
        capsys,
        "trend",
        "--series",
        "slopes",
        "--model",
        "Linear",
        "--param",
        "m",
        "--exclude",
        "s1,nope",
    )
    assert "Not in the trend: nope" in err


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--from-fits", "s1"], "--from-fits needs --param NAME"),
        (["--fit", "Lambda"], "--fit, --order and --x build a trend --from-fits"),
        (["--from-fits", "s1,slopes", "--param", "m"], "cannot be a member too"),
        (["--from-fits", "s1", "--param", "m", "--fit", "A_1"], "none is --fit A_1"),
        (["--from-fits", "s1", "--param", "Ea"], "has no parameter 'Ea'"),
        (["--from-fits", "s1,nope", "--param", "m"], "No series 'nope'"),
    ],
)
def test_trend_from_fits_refuses_a_malformed_request(
    workflow_folder: Path, fitting_workdir: Path, capsys, arguments: list[str], message: str
) -> None:
    _cli(
        workflow_folder,
        fitting_workdir,
        "fit-series",
        "--runs",
        ",".join(map(str, _STRETCHES["s1"])),
        "--recipe",
        "relax",
        "--order",
        "temperature",
        "--name",
        "s1",
    )
    _cli(
        workflow_folder,
        fitting_workdir,
        "trend",
        "--series",
        "s1",
        "--model",
        "Linear",
        "--param",
        "Lambda",
    )
    err = _refused(
        workflow_folder, fitting_workdir, capsys, "trend", "--series", "slopes", *arguments
    )
    assert message in err


def test_two_laws_on_one_trend_column_are_both_kept(
    workflow_folder: Path, fitting_workdir: Path, capsys
) -> None:
    _fit_scan(workflow_folder, fitting_workdir)
    for law in ("Linear", "Quadratic"):
        _cli(
            workflow_folder,
            fitting_workdir,
            "trend",
            "--series",
            "scan",
            "--model",
            law,
            "--param",
            "Lambda",
        )
    stored = json.loads((fitting_workdir / "series" / "scan.json").read_text(encoding="utf-8"))
    assert sorted(stored["trend_fits"]) == ["Lambda:Linear", "Lambda:Quadratic"]
    capsys.readouterr()
    err = _refused(
        workflow_folder,
        fitting_workdir,
        capsys,
        "trend",
        "--series",
        "laws",
        "--from-fits",
        "scan",
        "--param",
        "b",
    )
    assert (
        "holds trend fits: Lambda:Linear, Lambda:Quadratic; name one with --fit PARAM:EXPR" in err
    )


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


#: The grouping `info` reads off whatever the (faked) loader returns.
_TWO_GROUP_RUN = Run(
    run_number=1,
    histograms=[],
    grouping={"groups": {1: [1], 2: [2]}, "forward_group": 1, "backward_group": 2},
)


def test_a_repeated_warning_prints_once_without_verbose(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    import warnings as warnings_module

    class _Dummy:
        run = _TWO_GROUP_RUN

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
        run = _TWO_GROUP_RUN

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
    def _boom(_folder, *, pair, instrument):
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
        ([0.285, 0.280, 0.273], "frequency stays near 0.2800 MHz"),
        # A held line that still moves by many errors is a shift to report.
        # ... with the shift and its error printed, so no one subtracts them by hand.
        (
            [5.3735, 5.385, 5.3977],
            "frequency moves from 5.37350 to 5.39770 MHz, a shift of 0.02420 ± 0.00141 MHz",
        ),
    ],
)
def test_a_frequency_held_along_the_scan_is_not_called_an_order_parameter(
    frequencies, expected
) -> None:
    from asymmetry.cli.commands.trend import _law_hints
    from asymmetry.core.workflow.series import TrendTable

    rows = [
        {
            "run": run,
            "x": 50.0 * run,
            "frequency": value,
            "frequency_err": 0.001 if value > 1.0 else 0.01,
            "sigma": 0.3,
            "sigma_err": 0.01,
            "flags": [],
        }
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
        "order_key": "field",
        "success": True,
        "parameters": {"D": 25.9, "nu": 114.0, "m": -1.55},
        "uncertainties": {"D": 0.6, "nu": 17.0, "m": 0.1},
        "reduced_chi_squared": 1.0,
    }
    text = "\n".join(_render_fit(redfield, ["Lambda"]))
    assert "m is not positive" in text
    assert "refit with --fix m=2" in text
    # Against the wrong axis the law's parameters mean nothing, and no refit is offered.
    text = "\n".join(_render_fit(redfield | {"order_key": "temperature"}, ["Lambda"]))
    assert "NOTE: Redfield is a law in field, and this trend is against temperature" in text
    assert "--fix m=2" not in text
    # Once alpha is held, the hint has nothing left to say.
    held = fit | {"fixed": ["alpha"], "success": True}
    assert "--fix alpha" not in "\n".join(_render_fit(held, ["frequency"]))


def _gap_trend():
    """A synthetic σ(T) scan through Tc = 7 K: an s-wave rise on a 0.1 μs⁻¹ normal-state width."""
    import numpy as np

    from asymmetry.core.fitting.parameter_models import ParameterCompositeModel
    from asymmetry.core.workflow.series import TrendTable

    x = np.arange(0.5, 12.5, 0.75)
    law = ParameterCompositeModel.from_expression("SC_SWave").function
    sigma = law(x, sigma_0=0.4, Tc=7.0, gap_ratio=1.764, sigma_bg=0.1)
    noise = np.random.default_rng(3).normal(0.0, 0.004, x.size)
    rows = [
        {"key": str(run), "x": float(t), "sigma": float(s), "sigma_err": 0.004, "flags": []}
        for run, (t, s) in enumerate(zip(x, sigma + noise, strict=True), start=1)
    ]
    return TrendTable("temperature", ["key", "x", "sigma", "sigma_err", "flags"], rows)


def test_a_gap_law_fit_asks_for_a_verdict_and_offers_the_nodal_rival() -> None:
    from asymmetry.cli.commands.trend import _gap_law_steps
    from asymmetry.core.workflow.trend_fit import fit_trend

    trend = _gap_trend()
    fit = fit_trend(trend, "sigma", "SC_SWave", initial={"Tc": 6.0}).to_dict()
    text = "\n".join(_gap_law_steps("tf", trend, fit, []))
    # The normal state is in the fit, so no coverage note: a verdict on the law ...
    assert "NOTE" not in text
    assert "Verdict due: say in the report whether SC_SWave describes sigma(T)" in text
    assert "the plot (rerun with --plot)" in text
    # ... and the ready command for the nodal rival on the same points.
    assert (
        "Fit SC_DWave to the same points and compare chi2_red: asymmetry trend <folder> "
        "--series tf --model SC_DWave --param sigma."
    ) in text
    assert "SC_TwoGap_SS" not in text
    # A nodal law gets the verdict prompt but no rival of its own.
    nodal = fit_trend(trend, "sigma", "SC_DWave", initial={"Tc": 6.0}).to_dict()
    text = "\n".join(_gap_law_steps("tf", trend, nodal, [Path("plots/tf-trend-sigma.png")]))
    assert "whether SC_DWave describes sigma(T)" in text
    assert "the curve on plots/tf-trend-sigma.png" in text
    assert "rival" not in text


def test_a_gap_law_cut_off_below_the_normal_state_is_sent_back_for_the_warm_points() -> None:
    from asymmetry.cli.commands.trend import _gap_law_steps, _render_fit
    from asymmetry.core.workflow.trend_fit import fit_trend

    trend = _gap_trend()
    for fixed in ({}, {"sigma_bg": 0.1}):
        fit = fit_trend(trend, "sigma", "SC_SWave", x_min=1.0, x_max=6.5, fixed=fixed).to_dict()
        (note,) = _gap_law_steps("tf", trend, fit, [])
        assert note.startswith("NOTE: 0 of the fitted points lie above the fitted Tc")
        assert "only normal-state points determine sigma_bg" in note
        # The refit keeps the cold-side range and drops --xmax and a held width.
        assert (
            "asymmetry trend <folder> --series tf --model SC_SWave --param sigma --xmin 1."
        ) in note
        assert "follow from that refit" in note
        # It follows the verdict and replaces the generic advice to hold the width.
        block = _render_fit(fit, ["sigma"], [note])
        assert block[block.index(note) - 1].startswith(("LAW NOT ESTABLISHED", "Converged"))
        assert "textbook value" not in "\n".join(block)
    # An order parameter is fitted below its transition: nothing to add.
    assert _gap_law_steps("tf", trend, fit | {"expression": "OrderParameter"}, []) == []


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


def test_integral_scan_fits_inside_the_window_and_reports_a_failed_fit(
    workflow_folder: Path, tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    from asymmetry.core.fitting.parameter_models import ParameterModelFitResult

    base = [
        "integral-scan",
        str(workflow_folder),
        "--runs",
        f"{SCAN_RUNS[0]}-{SCAN_RUNS[-1]}",
        "--order",
        "temperature",
        "--workdir",
        str(tmp_path / "wd"),
    ]
    cli.main([*base, "--model", "Linear", "--xmin", "15", "--xmax", "45", "--json"])
    data = _json_output(capsys)
    assert data["fit"]["n_points"] == 3
    assert (data["fit"]["x_min"], data["fit"]["x_max"]) == (15.0, 45.0)
    assert len(data["scan"]["points"]) == len(SCAN_RUNS)

    # No more points than free parameters is refused, not fitted.
    with pytest.raises(SystemExit, match="1"):
        cli.main([*base, "--model", "LorentzianLCR + Cubic", "--xmax", "25"])
    assert "2 point(s) to fit for 7 free parameter(s)" in capsys.readouterr().err

    # A fit that fails says so, but the scan is kept.
    monkeypatch.setattr(
        "asymmetry.core.workflow.integral_scan.fit_scan_model",
        lambda *args, **kwargs: ParameterModelFitResult(success=False, message="Fit failed"),
    )
    cli.main([*base, "--name", "failed", "--model", "Linear"])
    out = capsys.readouterr().out
    assert "FAILED (Fit failed)" in out
    # No resonance in the model, so no window to fit one in.
    assert "Next:" not in out
    assert (tmp_path / "wd" / "scans" / "failed.json").exists()


def _failed_resonance_fit(**changes) -> dict:
    return {
        "success": False,
        "message": "Fit failed: call limit reached, hesse failed",
        "expression": "LorentzianLCR + LorentzianLCR + Cubic",
        "parameters": {"B0_1": 1500.0, "B0_2": 1500.0},
        "uncertainties": {},
        "reduced_chi_squared": 9.0,
        "params_at_bound": ["B0_2"],
        "initial": {},
        "fixed": [],
        "resonance_windows": [
            {
                "parameter": "B0_1",
                "component": "LorentzianLCR",
                "centre": 1200.0,
                "x_min": 1000.0,
                "x_max": 1400.0,
            },
            {
                "parameter": "B0_2",
                "component": "LorentzianLCR",
                "centre": 1800.0,
                "x_min": 1600.0,
                "x_max": 2000.0,
            },
        ],
    } | changes


def test_a_failed_resonance_fit_says_why_and_names_a_window_per_dip(tmp_path: Path) -> None:
    from asymmetry.cli.commands.integral_scan import _notes, _render
    from asymmetry.core.workflow.reduction import ReductionSettings

    result = {
        "name": "scan",
        "scan": {"points": [], "order_key": "field"},
        "period_field_offset": None,
        "fit": _failed_resonance_fit(),
        "scan_path": str(tmp_path / "scan.json"),
        "plot": None,
    }
    text = _render(result, ReductionSettings(), _notes(result, [], [], [], []))
    assert "FAILED (Fit failed: call limit reached, hesse failed; at a bound: B0_2)" in text
    assert (
        "Next: the scan's own largest dips are at B0_1 1200, B0_2 1800. The fit already "
        "started each centre there, so the usual cause is the background"
    ) in text
    assert (
        "fit one resonance per window on its own local background: --model 'LorentzianLCR + "
        "Linear' --xmin 1000 --xmax 1400; --model 'LorentzianLCR + Linear' --xmin 1600 "
        "--xmax 2000."
    ) in text
    assert "--initial" not in text

    # A start inside its dip's window is where the fit already began ...
    result["fit"] = _failed_resonance_fit(initial={"B0_1": 1250.0})
    assert "--initial" not in _render(result, ReductionSettings(), _notes(result, [], [], [], []))
    # ... and one away from it is pointed back at the dips.
    result["fit"] = _failed_resonance_fit(initial={"B0_1": 1500.0})
    text = _render(result, ReductionSettings(), _notes(result, [], [], [], []))
    assert (
        "The fit started away from them: refit with --initial B0_1=1200 --initial "
        "B0_2=1800, or fit one resonance per window on its own local background"
    ) in text


def test_fit_series_skips_run_numbers_the_folder_does_not_hold(
    workflow_folder: Path, tmp_path: Path, capsys
) -> None:
    # A scan's range routinely has gaps (an aborted run); reduce skips them,
    # and so does the fit, rather than asking for a run that does not exist.
    workdir = str(tmp_path / "wd")
    folder = str(workflow_folder)
    spec = f"{SCAN_RUNS[0]}-{SCAN_RUNS[1]},9999"
    cli.main(["reduce", folder, "--runs", spec, "--workdir", workdir])
    cli.main(
        [
            "recipe",
            folder,
            "--expression",
            "Exponential + Constant",
            "--name",
            "relax",
            "--run",
            str(SCAN_RUNS[0]),
            "--workdir",
            workdir,
        ]
    )
    capsys.readouterr()
    cli.main(
        [
            "fit-series",
            folder,
            "--runs",
            spec,
            "--recipe",
            "relax",
            "--order",
            "temperature",
            "--json",
            "--workdir",
            workdir,
        ]
    )
    assert [entry["run"] for entry in _json_output(capsys)["series"]["results"]] == [
        SCAN_RUNS[0],
        SCAN_RUNS[1],
    ]


def test_a_small_step_in_a_width_is_named_with_where_it_happens() -> None:
    from asymmetry.cli.commands.trend import _rate_steps
    from asymmetry.core.workflow.series import TrendTable

    # A Kubo–Toyabe width that rises by a few percent below a transition near
    # 6 K: small against the value, large against the errors.
    rows = [
        {"key": str(run), "x": x, "Delta": delta, "Delta_err": 0.001, "flags": []}
        for run, (x, delta) in enumerate(
            [(0.3, 0.261), (2.0, 0.260), (4.0, 0.259), (6.8, 0.253), (8.0, 0.252), (10.0, 0.253)]
        )
    ]
    trend = TrendTable("temperature", ["key", "x", "Delta", "Delta_err", "flags"], rows)

    (note,) = _rate_steps(trend, ["Delta"])
    assert "the change lies between 4 and 6.8" in note
    # A flat width draws no note.
    flat = [row | {"Delta": 0.26} for row in rows]
    assert _rate_steps(TrendTable("temperature", trend.columns, flat), ["Delta"]) == []


def test_a_gradual_step_is_bracketed_by_where_the_width_leaves_each_level() -> None:
    from asymmetry.cli.commands.trend import _rate_steps
    from asymmetry.core.workflow.series import TrendTable

    # A width at one level below 5.6, another above 6.8, and intermediate
    # between: the best two-block split falls mid-rise, the onset does not.
    points = [
        *((x, 0.2597) for x in (0.3, 1.0, 2.0, 3.0, 4.0, 5.0, 5.6)),
        (6.0, 0.2585),
        (6.2, 0.2570),
        (6.4, 0.2555),
        (6.6, 0.2545),
        *((x, 0.2534) for x in (6.8, 7.5, 8.0, 9.0, 10.0)),
    ]
    rows = [
        {"key": str(run), "x": x, "Delta": delta, "Delta_err": 0.0002, "flags": []}
        for run, (x, delta) in enumerate(points)
    ]

    (note,) = _rate_steps(
        TrendTable("temperature", ["key", "x", "Delta", "Delta_err", "flags"], rows), ["Delta"]
    )

    assert "low-temperature level (0.2597 over 0.3–5.6) above 5.6" in note
    assert "high-temperature level (0.2534 over 6.8–10) below 6.8" in note
    assert "the change lies between 5.6 and 6.8" in note


@pytest.mark.parametrize(("delay_us", "direction"), [(0.02, "positive"), (-0.02, "negative")])
def test_a_phase_linear_in_field_is_named_a_t0_offset(delay_us: float, direction: str) -> None:
    from asymmetry.cli.commands.trend import _phase_drift
    from asymmetry.core.fitting.spectral import field_gauss_to_frequency_mhz
    from asymmetry.core.workflow.series import TrendTable

    # A signal arriving delay_us after t0 fits the phase 0.1 - 2π f Δt.
    fields = (100.0, 200.0, 400.0, 800.0, 1600.0)
    rows = [
        {
            "key": str(run),
            "x": field,
            "phase": 0.1
            - 2.0 * math.pi * field_gauss_to_frequency_mhz(field) * delay_us
            + (0.01 if run % 2 else -0.01),
            "phase_err": 0.01,
            "flags": [],
        }
        for run, field in enumerate(fields)
    ]
    trend = TrendTable("field", ["key", "x", "phase", "flags"], rows)

    (note,) = _phase_drift(trend, ["A_1", "phase"])
    assert f"Δt = {1e3 * delay_us:+.3g} ns" in note
    assert f"--t0-offset <bins> ({direction}" in note
    # A phase that holds, or a phase along another axis, draws no note.
    held = [row | {"phase": 0.1} for row in rows]
    assert _phase_drift(TrendTable("field", trend.columns, held), ["phase"]) == []
    assert _phase_drift(TrendTable("temperature", trend.columns, rows), ["phase"]) == []


def test_a_precession_amplitude_falling_with_frequency_is_the_instruments_response() -> None:
    from asymmetry.cli.commands.trend import _frequency_response
    from asymmetry.core.workflow.series import TrendTable

    fields = (100.0, 500.0, 1000.0, 1500.0, 2000.0, 3000.0)
    rows = [
        {
            "key": str(run),
            "x": field,
            "A_1": 20.0 * math.exp(-((field / 1500.0) ** 2)),
            "A_1_err": 0.2,
            "frequency": 0.013554 * field,
            "frequency_err": 0.001,
            "A_bg": 2.0,
            "A_bg_err": 0.1,
            "flags": [],
        }
        for run, field in enumerate(fields)
    ]
    trend = TrendTable("field", ["key", "x", "A_1", "frequency", "A_bg", "flags"], rows)
    series = {
        "expression": "Oscillatory * Gaussian + Constant",
        "free_params": ["A_1", "frequency", "phase", "sigma", "A_bg"],
    }

    (note,) = _frequency_response(series, trend)
    assert note.startswith("NOTE: A_1 falls from 19.91 to 0.3663 while frequency rises")
    assert "halving by frequency 20.33 (field 1500)" in note
    assert "frequency response" in note
    # An amplitude that holds draws no note.
    held = [row | {"A_1": 20.0} for row in rows]
    assert _frequency_response(series, TrendTable("field", trend.columns, held)) == []


def test_a_held_high_field_line_is_offered_a_two_line_fit() -> None:
    from asymmetry.cli.commands.trend import _doublet_hint
    from asymmetry.core.workflow.series import TrendTable

    rows = [
        {
            "key": str(run),
            "x": x,
            "frequency": 813.57,
            "frequency_err": 0.001,
            "Lambda": rate,
            "Lambda_err": 0.01,
            "flags": [],
        }
        for run, x, rate in ((686, 5.0, 0.2), (690, 30.0, 0.1), (693, 50.0, 0.1))
    ]
    trend = TrendTable("temperature", ["key", "x", "frequency", "Lambda", "flags"], rows)
    series = {
        "expression": "Oscillatory * Exponential + Constant",
        "free_params": ["A_1", "frequency", "Lambda", "A_bg"],
    }

    (hint,) = _doublet_hint(series, trend)
    assert "--run 686 --name two-line" in hint
    assert "'Oscillatory * Exponential + Oscillatory * Exponential + Constant'" in hint
    assert "--initial frequency_1=" in hint and "--initial frequency_3=" in hint
    # A low-field line with no exponential on the cold side draws no hint.
    low = [row | {"frequency": 1.36} for row in rows]
    assert _doublet_hint(series, TrendTable("temperature", trend.columns, low)) == []


def test_fourier_names_two_peaks_closer_than_two_resolution_elements() -> None:
    from asymmetry.cli.commands.fourier import _render

    result = {
        "run": 693,
        "field_gauss": 60000.0,
        "axis": "frequency",
        "n_points": 1000,
        "resolution_mhz": 0.105,
        "settings": {"window": "none"},
        "peak_analysis": {
            "peaks": [
                {"frequency_mhz": f, "amplitude": 1.0, "width_mhz": 0.1, "snr": 90.0}
                for f in (813.497, 813.596, 815.9)
            ]
        },
        "candidate_maxima": [],
        "frequency_min_mhz": 812.0,
        "frequency_max_mhz": 816.0,
        "full_band_mhz": [0.0, 900.0],
        "outside_band": [],
        "array_path": "a.npz",
        "metadata_path": "a.json",
        "plot": None,
    }
    text = _render(result)
    assert "band 812–816 of 0–900 MHz" in text
    assert "NOTE: 813.497 and 813.596 MHz lie within 2 resolution elements" in text
    assert "815.9" not in text.split("NOTE:")[1]
    # A lone tesla-field line may still hold two, and the transform says what it is not.
    assert "NOTE: the line at 815.9 MHz (width 0.1 MHz, 1.0 resolution elements)" in text
    assert "--initial frequency_1=815.95 --initial frequency_3=815.85" in text
    assert "maximum-entropy (MaxEnt) spectra and multi-group" in text
    # A line far from the applied field's Larmor frequency is a radical's, not a split one.
    assert "NOTE: the line at" not in _render(result | {"field_gauss": 3000.0})
    # Lines the transform detected outside the band are named, not hidden.
    hidden = _render(result | {"outside_band": [{"frequency_mhz": 208.7, "snr": 35.0}]})
    assert "NOTE: the transform also holds lines outside this band — 208.7 MHz (SNR 35)" in hidden


def test_a_muonium_phase_drift_is_timed_against_its_own_frequency() -> None:
    from asymmetry.cli.commands.trend import _phase_drift
    from asymmetry.core.workflow.series import TrendTable

    # Muonium precesses ~103 times faster than the bare muon, so the same
    # 40 ns timing offset turns its phase ~103 times faster per gauss.
    rows = [
        {
            "key": str(run),
            "x": field,
            "frequency": 1.394 * field,
            "frequency_err": 0.001,
            "phase": -2 * math.pi * 1.394 * field * 0.040,
            "phase_err": 0.01,
            "flags": [],
        }
        for run, field in enumerate((1.0, 2.0, 3.0, 4.0))
    ]
    columns = ["key", "x", "frequency", "phase", "flags"]
    (note,) = _phase_drift(TrendTable("field", columns, rows), ["frequency", "phase"])
    assert "Δt = +40 ns" in note


def test_a_converged_but_poor_resonance_fit_asks_for_more_dips() -> None:
    from asymmetry.cli.commands.integral_scan import _poor_fit_note

    # One resonance fitted at 19480 G; the seeder puts the next dip's window
    # at 19850–23000 G, and one around the fitted line is already covered.
    windows = [
        {"x_min": 18200.0, "x_max": 19850.0},
        {"x_min": 19850.0, "x_max": 23000.0},
    ]
    fit = {
        "parameters": {"f": -0.01, "B0": 19480.0, "Bwid": 150.0},
        "uncertainties": {"f": 0.001, "B0": 10.0, "Bwid": 20.0},
        "success": True,
        "params_at_bound": [],
        "reduced_chi_squared": 12.6,
        "x_range": [17000.0, 23000.0],
        "x_min": None,
        "x_max": None,
        "next_dip_windows": windows,
    }
    dip, poor = _poor_fit_note(fit)
    assert "another dip this fit does not include, in 19850–23000" in dip
    assert "--xmin 18200" not in dip
    assert "converged at chi2_red 12.600: over a long range the background" in poor
    assert poor.endswith("--model 'LorentzianLCR + Linear' --xmin 18880 --xmax 20080.")
    # With no further dip found it says the background may be why.
    (bare,) = _poor_fit_note(fit | {"next_dip_windows": windows[:1]})
    assert "a fit that cannot is not a result" in bare
    assert _poor_fit_note(fit | {"reduced_chi_squared": 1.2, "next_dip_windows": []}) == []
    # A line whose flank runs off the fitted range may be a step, not a dip.
    (edge,) = _poor_fit_note(
        fit
        | {
            "parameters": {"f": -0.02, "B0": 6918.0, "Bwid": 1082.0},
            "x_range": [5000.0, 11000.0],
            "x_min": 5000.0,
            "x_max": 11000.0,
            "reduced_chi_squared": 1.5,
            "next_dip_windows": [],
        }
    )
    assert "runs off the fitted range 5000–11000" in edge
    # A whole scan narrower than its line is not a window that cut it off.
    whole = {"x_min": None, "x_max": None}
    assert _poor_fit_note(fit | {"reduced_chi_squared": 1.5, "next_dip_windows": []} | whole) == []


def test_a_windowed_line_with_both_flanks_and_depth_is_called_a_resonance() -> None:
    from asymmetry.cli.commands.integral_scan import _poor_fit_note

    fit = {
        "parameters": {"f": -0.008, "B0": 19475.5, "Bwid": 232.5},
        "uncertainties": {"f": 0.0005, "B0": 6.5, "Bwid": 12.0},
        "success": True,
        "params_at_bound": [],
        "reduced_chi_squared": 3.143,
        "x_range": [18700.0, 20200.0],
        "x_min": 18700.0,
        "x_max": 20200.0,
        "next_dip_windows": [],
    }
    # Resolved even at a poor chi2_red, which then qualifies its errors instead of
    # sending the agent off to look for a background it does not need.
    (note,) = _poor_fit_note(fit)
    assert note.startswith("RESONANCE: the line at 19475.5 ± 6.5 (width 232.5)")
    assert "16.0 errors from zero" in note and "chi2_red of 3.143" in note
    # A shallow line is no resonance, and the poor fit's note returns.
    (poor,) = _poor_fit_note(fit | {"uncertainties": fit["uncertainties"] | {"f": 0.004}})
    assert poor.startswith("NOTE: the fit converged at chi2_red 3.143")
    # Nor is a deep "line" whose fit sits far above its errors: a background step.
    (step,) = _poor_fit_note(fit | {"reduced_chi_squared": 7.7})
    assert step.startswith("NOTE: the fit converged at chi2_red 7.700")


def test_a_mistyped_folder_is_named_as_missing_with_the_folder_the_session_holds(
    workflow_folder: Path, tmp_path: Path, capsys
) -> None:
    workdir = str(tmp_path / "wd")
    cli.main(["survey", str(workflow_folder), "--workdir", workdir])
    capsys.readouterr()
    typo = str(workflow_folder) + "-typo"
    with pytest.raises(SystemExit):
        cli.main(["reduce", typo, "--runs", str(SCAN_RUNS[0]), "--workdir", workdir])
    err = capsys.readouterr().err
    assert f"{typo} does not exist or is not a directory." in err
    assert f"This work directory holds {workflow_folder.resolve()}" in err


def test_readings_leave_out_unreliable_rows_and_small_frequency_drifts() -> None:
    from asymmetry.cli.commands.trend import _frequency_response, _rate_steps
    from asymmetry.core.workflow.series import TrendTable

    # A flat width with one bound-pinned row at a wild value draws no step.
    rows = [
        {"key": str(run), "x": 10.0 * run, "Lambda": 0.077, "Lambda_err": 0.001, "flags": []}
        for run in range(1, 7)
    ]
    rows[0] = rows[0] | {"Lambda": 1e-14, "flags": ["bound_pinned"]}
    steps = TrendTable("temperature", ["key", "x", "Lambda", "Lambda_err", "flags"], rows)
    assert _rate_steps(steps, ["Lambda"]) == []

    # An amplitude that sags while a held line moves by 0.15 % is not the
    # instrument's frequency response.
    held = [
        {
            "key": str(run),
            "x": x,
            "A_1": amplitude,
            "A_1_err": 0.05,
            "frequency": frequency,
            "frequency_err": 0.0001,
            "flags": [],
        }
        for run, (x, amplitude, frequency) in enumerate(
            [(5.0, 19.04, 2.071), (40.0, 18.9, 2.072), (80.0, 18.78, 2.074)]
        )
    ]
    columns = ["key", "x", "A_1", "frequency", "flags"]
    series = {
        "expression": "Oscillatory * Gaussian + Constant",
        "free_params": ["A_1", "frequency"],
    }
    assert _frequency_response(series, TrendTable("temperature", columns, held)) == []


def test_an_alpha_measured_on_corrected_counts_says_it_differs_from_the_survey() -> None:
    from asymmetry.cli._reduction import describe
    from asymmetry.core.workflow.reduction import ReductionSettings

    raw = ReductionSettings(alpha=1.232, alpha_source="estimated:7")
    corrected = ReductionSettings(alpha=1.2373, alpha_source="estimated:7", deadtime="from_file")
    assert describe(raw).startswith("alpha 1.2320 (estimated:7), deadtime off")
    assert "survey and `alpha` measure raw counts" in describe(corrected)


def test_a_result_command_ends_with_the_audit_step_and_a_reduction_does_not(
    workflow_folder: Path, fitting_workdir: Path, capsys
) -> None:
    from asymmetry.cli import AUDIT_STEP

    folder, workdir = str(workflow_folder), str(fitting_workdir)
    cli.main(["fourier", folder, "--run", str(SCAN_RUNS[0]), "--fmax", "10", "--workdir", workdir])
    assert capsys.readouterr().out.rstrip().endswith(AUDIT_STEP)
    cli.main(["reduce", folder, "--runs", str(SCAN_RUNS[0]), "--workdir", workdir])
    assert AUDIT_STEP not in capsys.readouterr().out


def test_one_line_resolved_in_two_scans_is_compared_with_directions() -> None:
    from asymmetry.cli.commands.integral_scan import _line_comparisons

    def fit(centre: float, width: float, chi2: float) -> dict:
        return {
            "parameters": {"f": -0.01, "B0": centre, "Bwid": width, "m": 0.0, "b": 0.1},
            "uncertainties": {"f": 0.0005, "B0": 15.0, "Bwid": 30.0, "m": 0.0, "b": 0.001},
            "success": True,
            "params_at_bound": [],
            "reduced_chi_squared": chi2,
            "x_range": [12000.0, 18000.0],
            "x_min": 12000.0,
            "x_max": 18000.0,
        }

    hot, cold, far = fit(14940.0, 680.0, 1.0), fit(15400.0, 1160.0, 1.0), fit(7080.0, 390.0, 1.0)
    (note,) = _line_comparisons(hot, {"cold": cold, "other": far | {"x_range": [5000.0, 9000.0]}})
    assert "and scan cold's at 15400" in note
    assert "this one is narrower and lower in field" in note
    # Within two combined errors there is no direction to report.
    (same,) = _line_comparisons(fit(15390.0, 1150.0, 1.0), {"cold": cold})
    assert "of a width these errors cannot tell apart and at a field these errors" in same


def test_a_wizard_component_at_twice_a_tesla_line_is_named_a_harmonic() -> None:
    from asymmetry.cli.commands.wizard import _harmonic_pair

    assert _harmonic_pair([813.6, 1627.1]) == (813.6, 1627.1)
    # Two lines near each other, or a low-field pair, are not a harmonic.
    assert _harmonic_pair([813.6, 813.5]) is None
    assert _harmonic_pair([1.36, 2.72]) is None


def test_relaxing_amplitude_moving_between_shapes_is_named() -> None:
    from asymmetry.cli.commands.trend import _weight_shift
    from asymmetry.core.workflow.series import TrendTable

    names = ["A_1", "Lambda", "A_2", "sigma"]
    columns = ["key", "x", *[c for n in names for c in (n, f"{n}_err")], "flags"]

    def row(t: float, a1: float, lam: float, a2: float, sig: float) -> dict:
        values = dict(zip(names, (a1, lam, a2, sig), strict=True))
        return (
            {"key": str(int(t)), "x": t, "flags": []} | values | {f"{n}_err": 0.05 for n in names}
        )

    # Cold: an exponential carries the relaxation; warm: its rate is zero and a
    # Gaussian relaxes — a change of shape, whatever the amplitudes alone say.
    rows = [row(t, 10.0, 3.0, 4.0, 0.5) for t in (10, 20, 30, 40)] + [
        row(t, 10.0, 0.0, 15.0, 0.2) for t in (50, 60, 70, 80)
    ]
    series = {"kind": "series", "expression": "Exponential + Gaussian + Constant"}
    (note,) = _weight_shift(series, TrendTable("temperature", columns, rows))
    assert note.startswith("NOTE: the relaxation changes shape along the scan: the Exponential")
    assert "71% of the relaxing amplitude over the first 2 runs" in note
    # One shape throughout says nothing.
    flat = [row(t, 10.0, 3.0, 4.0, 0.5) for t in range(10, 90, 10)]
    assert _weight_shift(series, TrendTable("temperature", columns, flat)) == []


def test_repeated_scan_points_are_named_with_whether_they_came_back() -> None:
    from asymmetry.cli.commands.integral_scan import _repeated_points

    points = [
        {"run": 1, "x": 100.0, "value": 0.0590, "error": 0.0004},
        {"run": 2, "x": 500.0, "value": 0.1109, "error": 0.0004},
        {"run": 3, "x": 100.0, "value": 0.0595, "error": 0.0004},
        {"run": 4, "x": 500.0, "value": 0.1131, "error": 0.0004},
    ]
    (low, low_runs, low_differ), (high, high_runs, high_differ) = _repeated_points(points)
    assert (low, [p["run"] for p in low_runs], low_differ) == (100.0, [1, 3], False)
    assert (high, high_differ) == (500.0, True)


def test_a_dip_another_scan_already_fitted_is_not_announced_again() -> None:
    from asymmetry.cli.commands.integral_scan import _poor_fit_note

    fit = {
        "parameters": {"f": -0.01, "B0": 21474.0, "Bwid": 255.0},
        "uncertainties": {"f": 0.001, "B0": 8.0, "Bwid": 20.0},
        "success": True,
        "params_at_bound": [],
        "reduced_chi_squared": 1.9,
        "x_range": [19950.0, 22950.0],
        "x_min": 19950.0,
        "x_max": 22950.0,
        "next_dip_windows": [{"x_min": 17950.0, "x_max": 19950.0}],
    }
    assert any("another dip" in note for note in _poor_fit_note(fit))
    assert not any("another dip" in note for note in _poor_fit_note(fit, [19481.4]))


def test_a_diamagnetic_shift_with_a_growing_width_offers_a_gap_law() -> None:
    from asymmetry.cli.commands.trend import _law_hints
    from asymmetry.core.workflow.series import TrendTable

    columns = ["key", "x", "frequency", "frequency_err", "sigma", "sigma_err", "flags"]
    temperatures = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    frequencies = [5.370, 5.372, 5.375, 5.380, 5.388, 5.396, 5.397, 5.397]
    widths = [0.45, 0.44, 0.42, 0.38, 0.30, 0.20, 0.16, 0.16]
    rows = [
        {
            "key": str(i),
            "x": t,
            "frequency": f,
            "frequency_err": 0.0005,
            "sigma": w,
            "sigma_err": 0.005,
            "flags": [],
        }
        for i, (t, f, w) in enumerate(zip(temperatures, frequencies, widths, strict=True))
    ]
    text = "\n".join(
        _law_hints("tf", TrendTable("temperature", columns, rows), ["frequency", "sigma"])
    )
    assert "if the sample is a superconductor" in text
    assert "--model SC_SWave --param sigma" in text
    # A width that narrows on cooling (no vortex lattice) gets no gap law.
    flipped = [row | {"sigma": w} for row, w in zip(rows, widths[::-1], strict=True)]
    text = "\n".join(
        _law_hints("tf", TrendTable("temperature", columns, flipped), ["frequency", "sigma"])
    )
    assert "SC_SWave" not in text


def test_a_reduction_leaving_off_the_files_deadtimes_says_so(
    workflow_folder: Path, tmp_path: Path, capsys
) -> None:
    folder, workdir = str(workflow_folder), str(tmp_path / "wd")
    cli.main(["reduce", folder, "--runs", str(DEADTIME_RUN), "--workdir", workdir])
    assert "carry per-detector deadtimes and this reduction leaves" in capsys.readouterr().out
    cli.main(
        [
            "reduce",
            folder,
            "--runs",
            str(DEADTIME_RUN),
            "--deadtime",
            "from_file",
            "--workdir",
            workdir,
        ]
    )
    assert "carry per-detector deadtimes" not in capsys.readouterr().out


def test_two_close_fourier_peaks_get_the_two_line_recipe() -> None:
    from asymmetry.cli.commands.fourier import _render

    result = {
        "run": 693,
        "field_gauss": None,
        "axis": "frequency",
        "n_points": 1000,
        "resolution_mhz": 0.105,
        "settings": {"window": "none"},
        "peak_analysis": {
            "peaks": [
                {"frequency_mhz": f, "amplitude": 1.0, "width_mhz": 0.1, "snr": 90.0}
                for f in (813.497, 813.595)
            ]
        },
        "candidate_maxima": [],
        "frequency_min_mhz": 812.0,
        "frequency_max_mhz": 816.0,
        "full_band_mhz": [0.0, 900.0],
        "outside_band": [],
        "array_path": "a.npz",
        "metadata_path": "a.json",
        "plot": None,
    }
    assert "--initial frequency_1=813.595 --initial frequency_3=813.497" in _render(result)


def test_two_fitted_frequencies_within_two_percent_are_a_pair() -> None:
    from asymmetry.cli.commands.fourier import close_pair

    assert close_pair({"frequency_1": 813.601, "frequency_3": 813.542}) == (813.542, 813.601)
    assert close_pair({"frequency_1": 813.6, "frequency_3": 1627.2}) is None
    assert close_pair({"frequency": 1.36, "Lambda": 1.37}) is None
