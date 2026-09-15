"""Tests for command-line interface behavior."""

from __future__ import annotations

import json

import pytest

from asymmetry import __version__, cli
from asymmetry.cli._output import SCHEMA
from tests.core.conftest import CALIBRATION_RUN


class _FakeRun:
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


def test_info_without_json_still_prints_only_the_summary(workflow_folder, capsys) -> None:
    path = workflow_folder / f"SIM{CALIBRATION_RUN:08d}.nxs"
    cli.main(["info", str(path)])

    out = capsys.readouterr().out
    assert out.startswith("MuonDataset")
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
