"""MaxEnt promotes a validated total deadtime table with its source run."""

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
from asymmetry.core.transform.deadtime import DeadtimeCalibration
from asymmetry.gui.mainwindow import MainWindow


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
    win._data_browser.add_dataset(ds)
    win._current_dataset = ds
    win._maxent_fitted_deadtime = DeadtimeCalibration((0.012, 0.018), (None, None), "maxent_fit", 7)

    win._on_maxent_apply_deadtime()

    grouping = ds.run.grouping
    assert grouping["dead_time_us"] == [0.012, 0.018]
    assert grouping["deadtime_method"] == "maxent_fit"
    assert grouping["deadtime_correction"] is True
    assert ds.metadata["grouping_overrides"]
    saved = win.collect_project_state()["datasets"][0]["grouping_overrides"]
    assert saved["dead_time_us"] == [0.012, 0.018]
    assert saved["deadtime_method"] == "maxent_fit"
    assert saved["deadtime_errors_us"] == [None, None]


def test_maxent_apply_deadtime_shows_before_after_and_reduce_message(win, qapp) -> None:
    """The rerouted handler gains a before/after display and the re-reduce hint."""
    ds = _dataset(run_number=7)
    ds.run.grouping["dead_time_us"] = [0.0, 0.0]
    win._current_dataset = ds
    win._maxent_fitted_deadtime = DeadtimeCalibration((0.012, 0.018), (None, None), "maxent_fit", 7)

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
    win._maxent_fitted_deadtime = DeadtimeCalibration((0.0, 0.018), (None, None), "maxent_fit", 7)

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
    win._maxent_fitted_deadtime = DeadtimeCalibration((0.012, 0.018), (None, None), "maxent_fit", 9)

    win._on_maxent_apply_deadtime()

    assert "dead_time_us" not in ds.run.grouping


def test_obsolete_calibration_completions_do_not_change_the_panel(win, monkeypatch):
    win._current_dataset = _dataset(7)
    win._maxent_deadtime_generation = 2
    messages = []
    monkeypatch.setattr(
        win._maxent_panel, "set_deadtime_text", lambda *args, **kwargs: messages.append(args)
    )
    obsolete = DeadtimeCalibration((0.01, 0.02), (None, None), "maxent_fit", 7)
    win._on_maxent_deadtime_fitted(obsolete, generation=1, source_run=7)
    win._on_maxent_deadtime_error("obsolete", generation=1, source_run=7)
    win._on_maxent_deadtime_fitted(None, generation=2, source_run=9)
    win._on_maxent_deadtime_error("other run", generation=2, source_run=9)
    assert win._maxent_fitted_deadtime is None
    assert messages == []
