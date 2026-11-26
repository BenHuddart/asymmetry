"""Shared window and histogram-backed runs for toolbar bunching tests."""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.transform.preparation import histogram_observation
from asymmetry.core.transform.rebin import binned_fb_asymmetry, binning_spec
from asymmetry.gui.mainwindow import MainWindow
from asymmetry.gui.ui_manager import UI_SCALE_SETTINGS_KEY
from tests.counts import aligned_counts

BIN_WIDTH_US = 0.016


@pytest.fixture
def mainwindow(qapp: QApplication) -> MainWindow:
    """Create a mainwindow for testing."""
    settings = QSettings()
    settings.setValue(UI_SCALE_SETTINGS_KEY, 1.0)
    return MainWindow()


def histogram_run_dataset(
    counts_f: np.ndarray,
    counts_b: np.ndarray,
    *,
    run_number: int = 4242,
    base_bunch: int = 1,
    grouping_extra: dict | None = None,
) -> MuonDataset:
    """A run with two raw histograms and its recorded (counts-first) reduction."""
    n = counts_f.size
    grouping = {
        "groups": {1: [1], 2: [2]},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.0,
        "t0_bin": 0,
        "first_good_bin": 0,
        "last_good_bin": n - 1,
        "bunching_factor": base_bunch,
        "deadtime_correction": False,
        "background_correction": False,
    }
    if grouping_extra:
        grouping.update(grouping_extra)
    run = Run(
        run_number=run_number,
        histograms=[
            Histogram(counts=counts_f.copy(), bin_width=BIN_WIDTH_US, t0_bin=0),
            Histogram(counts=counts_b.copy(), bin_width=BIN_WIDTH_US, t0_bin=0),
        ],
        metadata={"run_number": run_number},
        grouping=grouping,
    )
    time, asym, err = binned_fb_asymmetry(
        aligned_counts(
            counts_f,
            counts_b,
            common_t0=0,
            bin_width_us=BIN_WIDTH_US,
            first_good_bin=0,
            last_good_bin=n - 1,
        ),
        binning=binning_spec(grouping),
        alpha=1.0,
    )
    return MuonDataset(
        time=time,
        asymmetry=asym * 100.0,
        error=err * 100.0,
        metadata={"run_number": run_number},
        run=run,
        histogram_observation=histogram_observation(run),
    )
