"""Tests for GUI export path persistence helpers."""

from __future__ import annotations

from pathlib import Path

import asymmetry.gui.export_paths as export_paths


class _FakeSettings:
    _store: dict[str, str] = {}

    def value(self, key: str, default: str = "", _type=str) -> str:
        value = self._store.get(key, default)
        return str(value)

    def setValue(self, key: str, value: str) -> None:
        self._store[key] = str(value)


def test_default_export_path_prefers_last_export_dir(monkeypatch) -> None:
    monkeypatch.setattr(export_paths, "QSettings", _FakeSettings)
    _FakeSettings._store = {
        "io/last_export_dir": "/tmp/exports",
        "io/last_open_dir": "/tmp/open",
    }

    path = export_paths.default_export_path("figure.gle")

    assert path == str(Path("/tmp/exports") / "figure.gle")


def test_default_export_path_falls_back_to_last_open_dir(monkeypatch) -> None:
    monkeypatch.setattr(export_paths, "QSettings", _FakeSettings)
    _FakeSettings._store = {
        "io/last_open_dir": "/tmp/open",
    }

    path = export_paths.default_export_path("table.csv")

    assert path == str(Path("/tmp/open") / "table.csv")


def test_default_export_path_falls_back_to_home(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(export_paths, "QSettings", _FakeSettings)
    _FakeSettings._store = {}
    monkeypatch.setattr(export_paths.Path, "home", staticmethod(lambda: tmp_path))

    path = export_paths.default_export_path("result.gle")

    assert path == str(tmp_path / "result.gle")


def test_remember_export_path_persists_parent_directory(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(export_paths, "QSettings", _FakeSettings)
    _FakeSettings._store = {}
    out_file = tmp_path / "nested" / "fit.gle"

    export_paths.remember_export_path(str(out_file))

    assert _FakeSettings._store["io/last_export_dir"] == str((tmp_path / "nested").resolve())
