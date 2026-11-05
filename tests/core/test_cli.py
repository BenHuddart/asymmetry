"""Tests for command-line interface behavior."""

from __future__ import annotations

import json

import pytest

from asymmetry import __version__, cli
from asymmetry.cli._output import SCHEMA
from asymmetry.core.data.dataset import Run
from tests.core.conftest import CALIBRATION_RUN


class _FakeRun:
    run = Run(
        run_number=1,
        histograms=[],
        grouping={"groups": {1: [1], 2: [2]}, "forward_group": 1, "backward_group": 2},
    )

    def summary(self) -> str:
        return "fake run summary"


def test_info_command_loads_file_and_prints_summary(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    loaded: list[str] = []

    def _fake_load(path: str):
        loaded.append(path)
        return _FakeRun()

    monkeypatch.setattr("asymmetry.core.io.load", _fake_load)
    cli.main(["info", "sample.nxs"])

    out = capsys.readouterr().out
    assert loaded == ["sample.nxs"]
    assert "fake run summary" in out


def test_info_json_payload_carries_the_run_identity_and_its_metadata(
    workflow_folder, capsys
) -> None:
    """``--json`` is the CLI's contract on every command, ``info`` included."""
    path = workflow_folder / f"SIM{CALIBRATION_RUN:08d}.nxs"
    cli.main(["info", str(path), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == SCHEMA
    assert payload["asymmetry_version"] == __version__
    assert payload["file"] == str(path)
    assert payload["run_number"] == CALIBRATION_RUN
    assert payload["n_points"] > 0
    assert payload["summary"].startswith("MuonDataset")
    # The metadata the loader recovered, minus the thousands-of-nodes NeXus
    # tree — which would bury it.
    assert payload["metadata"]["temperature"] == pytest.approx(5.0)
    assert "nexus_fields" not in payload["metadata"]
    assert payload["groups"] == {"1": "Group 1", "2": "Group 2"}
    assert payload["default_pair"] == [1, 2]


def test_info_without_json_prints_the_summary_and_the_groups(workflow_folder, capsys) -> None:
    path = workflow_folder / f"SIM{CALIBRATION_RUN:08d}.nxs"
    cli.main(["info", str(path)])

    out = capsys.readouterr().out
    assert out.startswith("MuonDataset")
    assert "Groups      : 1 Group 1, 2 Group 2  (default pair Group 1/Group 2)" in out
    assert "schema" not in out


def test_main_without_command_prints_help(capsys) -> None:
    cli.main([])
    out = capsys.readouterr().out
    assert "Asymmetry" in out
    assert "info" in out


def test_version_flag_exits_with_version(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert f"asymmetry {cli.__version__}" in out


def test_reduce_warns_and_caches_requested_file_deadtime_without_a_table(
    workflow_folder, tmp_path, capsys
):
    import json
    import shutil

    from asymmetry.core.workflow.workdir import WorkDir

    h5py = pytest.importorskip("h5py")
    folder = tmp_path / "raw"
    folder.mkdir()
    source = folder / "SIM00000101.nxs"
    shutil.copy2(workflow_folder / source.name, source)
    with h5py.File(source, "r+") as handle:
        del handle["run/histogram_data_1/dead_time"]
    workdir = tmp_path / "workdir"
    args = [
        "reduce",
        str(folder),
        "--runs",
        "101",
        "--deadtime",
        "from_file",
        "--workdir",
        str(workdir),
        "--json",
    ]
    for recomputed in (True, False):
        cli.main(args)
        captured = capsys.readouterr()
        assert "File deadtime correction not applied" in captured.err
        result = json.loads(captured.out)
        row = result["entries"][0]
        assert row["recomputed"] is recomputed
        outcomes = row["run"]["correction_outcomes"]
        assert next(iter(outcomes.values()))["deadtime"]["status"] == "skipped"
    restored = WorkDir(workdir).reduced(101)
    assert "correction_error" not in restored.metadata
    assert restored.metadata["correction_outcomes"] == outcomes
