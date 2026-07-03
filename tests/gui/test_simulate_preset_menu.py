"""GUI: the File → Simulate Preset gallery adds badged synthetic runs."""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

import asymmetry.gui.mainwindow as mw_module
from asymmetry.core.simulate_presets import ARCHETYPE_PRESETS
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


def test_single_run_preset_adds_one_badged_dataset(win, qapp) -> None:
    before = len(win._data_browser.all_datasets())
    win._on_generate_preset("ag_zf_kt")
    qapp.processEvents()
    datasets = win._data_browser.all_datasets()
    assert len(datasets) == before + 1
    run = datasets[-1].run
    assert run.metadata["synthetic"] is True
    assert run.metadata["simulation"]["preset"] == "ag_zf_kt"


def test_scan_preset_adds_the_whole_family(win, qapp) -> None:
    before = len(win._data_browser.all_datasets())
    win._on_generate_preset("euo_tscan")
    qapp.processEvents()
    added = len(win._data_browser.all_datasets()) - before
    assert added == len(ARCHETYPE_PRESETS["euo_tscan"].specs) == 5


def test_preset_runs_get_distinct_run_numbers(win, qapp) -> None:
    win._on_generate_preset("ag_lf_decoupling")
    qapp.processEvents()
    numbers = [
        ds.run_number
        for ds in win._data_browser.all_datasets()
        if isinstance(ds.run, object) and ds.run is not None and ds.run.metadata.get("synthetic")
    ]
    assert len(numbers) == len(set(numbers))  # no collisions
