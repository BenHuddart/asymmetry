"""GUI: warn at project save when synthetic/degraded runs have no backing file.

Study decision 2 — synthetic runs persist via Save-as-NeXus, not in the
project file, so a project referencing one cannot reload it.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QMessageBox

import asymmetry.gui.mainwindow as mw_module
from asymmetry.core.simulate import build_builtin_template, reduce_run_to_dataset, simulate_run
from asymmetry.gui.mainwindow import MainWindow


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def win(qapp: QApplication) -> MainWindow:
    settings = QSettings()
    settings.setValue(mw_module._UI_SCALE_SETTINGS_KEY, 1.0)
    w = MainWindow()
    w.show()
    qapp.processEvents()
    return w


def _add_synthetic(win) -> None:
    run = simulate_run(
        build_builtin_template("ideal_pulsed_fb"),
        lambda t, a0=20.0, rate=0.4: a0 * np.exp(-rate * t),
        {"a0": 20.0, "rate": 0.4},
        total_events=20.0e6,
        seed=1,
        run_number=90001,
    )
    win._data_browser.add_dataset(reduce_run_to_dataset(run))


def test_unsaved_synthetic_run_is_flagged(win, qapp) -> None:
    assert win._unsaved_synthetic_run_labels() == []
    _add_synthetic(win)
    labels = win._unsaved_synthetic_run_labels()
    assert any("90001" in label for label in labels)


def test_run_with_backing_file_not_flagged(win, qapp) -> None:
    run = simulate_run(
        build_builtin_template("ideal_pulsed_fb"),
        lambda t, a0=20.0: a0 * np.ones_like(t),
        total_events=10.0e6,
        seed=1,
        run_number=90002,
    )
    run.source_file = "/data/SIM90002.nxs"  # pretend it was saved as NeXus
    win._data_browser.add_dataset(reduce_run_to_dataset(run))
    assert win._unsaved_synthetic_run_labels() == []


def test_cancel_aborts_the_save(win, qapp, tmp_path, monkeypatch) -> None:
    _add_synthetic(win)
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Cancel),
    )
    saved = {"called": False}
    monkeypatch.setattr(
        mw_module, "save_project", lambda *a, **k: saved.__setitem__("called", True)
    )
    win._write_project(str(tmp_path / "p.asymp"))
    assert saved["called"] is False


def test_save_anyway_proceeds(win, qapp, tmp_path, monkeypatch) -> None:
    _add_synthetic(win)
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Save),
    )
    saved = {"called": False}
    monkeypatch.setattr(
        mw_module, "save_project", lambda *a, **k: saved.__setitem__("called", True)
    )
    win._write_project(str(tmp_path / "p.asymp"))
    assert saved["called"] is True
