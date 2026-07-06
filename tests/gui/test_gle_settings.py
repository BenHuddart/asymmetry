"""Tests for GLE executable settings + the Setup dialog's path validation.

The dialog must reject a configured path that cannot actually run — a saved
bad path only resurfaces later as an opaque export failure.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

pytestmark = [pytest.mark.gui]

pytest.importorskip("PySide6")
from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from asymmetry.gui.gle_settings import (  # noqa: E402
    _SETTINGS_KEY,
    GleSetupDialog,
    validate_gle_executable,
)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _fake_gle(tmp_path: Path, banner: str = "GLE version 4.3.10") -> Path:
    script = tmp_path / "gle"
    script.write_text(f'#!/bin/sh\necho "{banner}"\n', encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


# ---------------------------------------------------------------------------
# validate_gle_executable
# ---------------------------------------------------------------------------
def test_validate_missing_path(tmp_path: Path) -> None:
    problem = validate_gle_executable(str(tmp_path / "nope"))
    assert problem is not None and "No file" in problem


def test_validate_non_executable_file(tmp_path: Path) -> None:
    plain = tmp_path / "gle"
    plain.write_text("not a binary", encoding="utf-8")
    plain.chmod(0o644)
    if os.access(plain, os.X_OK):  # pragma: no cover - e.g. exotic ACLs
        pytest.skip("filesystem reports non-executable file as executable")
    problem = validate_gle_executable(str(plain))
    assert problem is not None and "not executable" in problem


def test_validate_executable_without_gle_banner(tmp_path: Path) -> None:
    impostor = _fake_gle(tmp_path, banner="something else entirely")
    problem = validate_gle_executable(str(impostor))
    assert problem is not None and "does not look like GLE" in problem


def test_validate_working_gle(tmp_path: Path) -> None:
    gle = _fake_gle(tmp_path)
    assert validate_gle_executable(str(gle)) is None


# ---------------------------------------------------------------------------
# GleSetupDialog accept-path validation
# ---------------------------------------------------------------------------
def test_dialog_rejects_unrunnable_path(qapp: QApplication, tmp_path: Path) -> None:
    bad = tmp_path / "gle"
    bad.write_text("plain file", encoding="utf-8")
    bad.chmod(0o644)

    dialog = GleSetupDialog()
    try:
        dialog._path_edit.setText(str(bad))
        dialog._on_accept()

        assert dialog.result() != dialog.DialogCode.Accepted
        assert "Cannot use this GLE path" in dialog._status_label.text()
        assert QSettings().value(_SETTINGS_KEY, "", str) == ""
    finally:
        dialog.close()


def test_dialog_accepts_working_gle(qapp: QApplication, tmp_path: Path) -> None:
    gle = _fake_gle(tmp_path)

    dialog = GleSetupDialog()
    try:
        dialog._path_edit.setText(str(gle))
        dialog._on_accept()

        assert QSettings().value(_SETTINGS_KEY, "", str) == str(gle)
    finally:
        dialog.close()


def test_dialog_accepts_empty_path_as_auto_detect(qapp: QApplication) -> None:
    """Clearing the field must stay allowed — it means 'use auto-detection'."""
    dialog = GleSetupDialog()
    try:
        dialog._path_edit.setText("")
        dialog._on_accept()

        assert QSettings().value(_SETTINGS_KEY, "", str) == ""
    finally:
        dialog.close()
