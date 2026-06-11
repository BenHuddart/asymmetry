"""GUI: the MaxEnt apply-deadtime handler writes the grouping via the chokepoint.

The first test pins the observable key writes of ``_on_maxent_apply_deadtime``
*before* it was rerouted through :func:`promote_deadtime_to_grouping` (F6), so
the reroute is provably behaviour-preserving. The later tests assert the gains
the reroute brings: a before/after display and the "Re-reduce" message.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

import asymmetry.gui.mainwindow as mw_module
from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
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


def _dataset(run_number: int = 7) -> MuonDataset:
    bin_width = 0.04
    n = 64
    time = np.arange(n, dtype=float) * bin_width
    histograms = [
        Histogram(counts=np.full(n, 100.0), bin_width=bin_width, t0_bin=0),
        Histogram(counts=np.full(n, 100.0), bin_width=bin_width, t0_bin=0),
    ]
    run = Run(
        run_number=run_number,
        histograms=histograms,
        metadata={},
        grouping={
            "groups": {1: [1], 2: [2]},
            "first_good_bin": 0,
            "last_good_bin": n - 1,
            "deadtime_correction": False,
        },
    )
    return MuonDataset(time=time, asymmetry=np.zeros(n), error=np.ones(n), run=run)


def test_maxent_apply_deadtime_writes_grouping_keys(win, qapp) -> None:
    """Pin the grouping keys written by the MaxEnt apply-deadtime handler."""
    ds = _dataset(run_number=7)
    win._current_dataset = ds
    win._maxent_fitted_deadtime = (7, [0.012, 0.018])

    win._on_maxent_apply_deadtime()

    grouping = ds.run.grouping
    assert grouping["dead_time_us"] == [0.012, 0.018]
    assert grouping["deadtime_method"] == "maxent_fit"
    assert grouping["deadtime_correction"] is True


def test_maxent_apply_deadtime_shows_before_after_and_reduce_message(win, qapp) -> None:
    """The rerouted handler gains a before/after display and the re-reduce hint."""
    ds = _dataset(run_number=7)
    ds.run.grouping["dead_time_us"] = [0.0, 0.0]
    win._current_dataset = ds
    win._maxent_fitted_deadtime = (7, [0.012, 0.018])

    captured: dict[str, object] = {}
    win._maxent_panel.set_deadtime_text = (  # type: ignore[method-assign]
        lambda text, can_apply=True: captured.update(text=text, can_apply=can_apply)
    )

    win._on_maxent_apply_deadtime()

    text = str(captured["text"])
    assert "→" in text  # before → after disclosed
    assert "Re-reduce" in text
    assert captured["can_apply"] is False


def test_maxent_apply_deadtime_summarises_across_detectors(win, qapp) -> None:
    """Multi-detector deadtimes are summarised, not read off detector 0 alone."""
    ds = _dataset(run_number=7)
    win._current_dataset = ds
    # Detector 0 fits ~0, detector 1 fits a real value — a det-0 readout would
    # wrongly say "0.00 → 0.00".
    win._maxent_fitted_deadtime = (7, [0.0, 0.018])

    captured: dict[str, object] = {}
    win._maxent_panel.set_deadtime_text = (  # type: ignore[method-assign]
        lambda text, can_apply=True: captured.update(text=text)
    )
    win._on_maxent_apply_deadtime()

    text = str(captured["text"])
    assert "0.00 → 9.00 ns" in text  # mean of [0, 18] ns
    assert "2 detector(s)" in text


def test_maxent_apply_deadtime_requires_matching_run(win, qapp) -> None:
    """A fitted deadtime for a different run must not touch the grouping."""
    ds = _dataset(run_number=7)
    win._current_dataset = ds
    win._maxent_fitted_deadtime = (9, [0.012, 0.018])

    win._on_maxent_apply_deadtime()

    assert "dead_time_us" not in ds.run.grouping
