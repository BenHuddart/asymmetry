"""RED target for branch ``fix/logged-t-trend-propagation``.

Round-3 GUI finding (CdS, ``_findings/windows-gui/Round3_progress.md``):
``Options → Use temperature from log`` switches the **Data Browser** T column to the
logged sample temperature, but the batch FitSeries trend X-axis (and CSV export)
still use the parked **setpoint**. Root cause: ``MainWindow._dataset_trend_coords``
returns ``metadata['temperature']`` unconditionally, ignoring the logged-T toggle —
so a parked-setpoint series (e.g. CdS 20711–20721, all setpoint 1 K) collapses onto
one abscissa point and a T-trend / Arrhenius is impossible.

Contract: when the browser displays a logged temperature for a run, the trend
coordinate must use that same value (i.e. match ``_temperature_for_display``).
RED today (it returns the setpoint).
"""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

import numpy as np
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.gui.mainwindow import MainWindow
from asymmetry.gui.ui_manager import UI_SCALE_SETTINGS_KEY


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def mw(app):
    QSettings().setValue(UI_SCALE_SETTINGS_KEY, 1.0)
    return MainWindow()


def _dataset(run_number: int, setpoint_temp: float) -> MuonDataset:
    run = Run(
        run_number=run_number,
        histograms=[
            Histogram(np.array([10.0, 20.0, 30.0, 40.0]), 0.1, 0),
            Histogram(np.array([8.0, 16.0, 24.0, 32.0]), 0.1, 0),
        ],
        metadata={"field": 100.0, "temperature": setpoint_temp},
        grouping={
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 3,
        },
    )
    return MuonDataset(
        np.array([0.0, 0.1, 0.2, 0.3]),
        np.array([0.1, 0.1, 0.1, 0.1]),
        np.array([0.01, 0.01, 0.01, 0.01]),
        {"run_number": run_number, "field": 100.0, "temperature": setpoint_temp},
        run,
    )


def test_trend_coords_use_logged_temperature_when_enabled(mw, monkeypatch):
    # Parked setpoint 1 K, true logged sample T 268 K (the CdS situation).
    ds = _dataset(20711, setpoint_temp=1.0)
    mw._data_browser.add_dataset(ds)
    mw._data_browser.set_use_temperature_from_log(True)
    # The browser resolves the logged value from the run's selog; stub that source
    # so the test does not depend on the exact selog metadata shape.
    monkeypatch.setattr(
        mw._data_browser, "_temperature_from_log_for_display", lambda dataset: 268.0
    )

    # Sanity: the browser now DISPLAYS the logged temperature, not the setpoint.
    assert mw._data_browser._temperature_for_display(ds) == pytest.approx(268.0)

    coords = mw._dataset_trend_coords(20711)
    assert coords["temperature"] == pytest.approx(268.0), (
        "trend coordinate used the parked setpoint (1 K) instead of the logged "
        "temperature the browser displays (268 K)"
    )
