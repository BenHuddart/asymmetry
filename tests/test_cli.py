"""Tests for command-line interface behavior."""

from __future__ import annotations

import pytest

from asymmetry import cli


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
    cli.main(["info", "sample.wim"])

    out = capsys.readouterr().out
    assert loaded == ["sample.wim"]
    assert "fake run summary" in out


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
    assert "asymmetry 0.2.1" in out
