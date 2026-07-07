"""Focused tests for RunRangeDialog's folder-scan prefill (F4).

``scan_run_files`` bounds its directory walk (``DEFAULT_MAX_SCAN_ENTRIES``)
so a facility folder with tens of thousands of files does not stat+regex
every entry unbounded. These tests pin the dialog-side surface of that cap:
the truncation notice appears only when the scan hit the cap.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from asymmetry.core.io import run_range
from asymmetry.gui.windows.run_range_dialog import RunRangeDialog


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _touch(folder: Path, name: str) -> None:
    (folder / name).write_bytes(b"")


def test_prefill_hides_truncation_notice_within_cap(qapp: QApplication, tmp_path: Path) -> None:
    for run in range(5):
        _touch(tmp_path, f"MUSR{run:08d}.nxs")

    dialog = RunRangeDialog(initial_dir=str(tmp_path))

    assert dialog._scan_truncated_label.isHidden()
    assert dialog.first_run() == 0
    assert dialog.last_run() == 4
    dialog.close()


def test_prefill_shows_truncation_notice_when_folder_exceeds_cap(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Synthetic ~facility-scale folder: cap monkeypatched down (60 cheap
    empty files instead of a real 25k-file corpus, per the harness's cost
    guidance) so the scan is forced to truncate."""
    monkeypatch.setattr(run_range, "DEFAULT_MAX_SCAN_ENTRIES", 50)

    for run in range(60):
        _touch(tmp_path, f"MUSR{run:08d}.nxs")

    dialog = RunRangeDialog(initial_dir=str(tmp_path))

    assert not dialog._scan_truncated_label.isHidden()
    assert "first" in dialog._scan_truncated_label.text().lower()
    dialog.close()
