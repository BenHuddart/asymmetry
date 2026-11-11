"""Synthetic acquired-run pooling with unequal file clocks and correction facts."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import QLabel, QSplitter, QVBoxLayout, QWidget

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.simulate import reduce_run_to_dataset
from asymmetry.core.utils.constants import MUON_LIFETIME_US
from asymmetry.gui.panels.data_browser import DataBrowserPanel
from asymmetry.gui.panels.plot_panel import PlotPanel

from ._base import Scenario, register


class ArithmeticDemo(QWidget):
    """Actual browser combination and selection over isolated synthetic acquisitions."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Synthetic acquired-run co-add")
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel("Synthetic co-add · own file clocks and deadtime facts · common good support")
        )
        split = QSplitter(Qt.Orientation.Horizontal)
        self.browser = DataBrowserPanel()
        self.plot = PlotPanel()
        split.addWidget(self.browser)
        split.addWidget(self.plot)
        split.setStretchFactor(1, 2)
        layout.addWidget(split)
        for number, pair, anchor, frames, tau in (
            (101, (80000, 20000), 0, 100000, 0.05),
            (102, (10000, 90000), 3, 200000, 0.1),
        ):
            elapsed = (np.arange(16) - anchor) * 0.1
            histograms = [
                Histogram(
                    np.rint(level * np.exp(-elapsed / MUON_LIFETIME_US)),
                    0.1,
                    t0_bin=anchor,
                    good_bin_start=anchor,
                    good_bin_end=anchor + 8,
                )
                for level in pair
            ]
            grouping = {
                "groups": {1: [1], 2: [2]},
                "forward_group": 1,
                "backward_group": 2,
                "alpha": 1.0,
                "deadtime_correction": True,
                "deadtime_mode": "file",
                "dead_time_us": [tau] * 2,
                "deadtime_file_us": [tau] * 2,
                "good_frames": frames,
                "good_window_policy": {"mode": "from_file"},
                "t0_policy": {"mode": "from_file"},
                "background_correction": False,
            }
            run = Run(
                number,
                histograms=histograms,
                grouping=grouping,
                metadata={"title": "Synthetic acquired counts", "run_number": number},
            )
            self.browser.add_dataset(reduce_run_to_dataset(run))
        self.browser.select_runs([101, 102])
        self.browser._coadd_selected()
        combined = next(iter(self.browser._combined_datasets), None)
        if combined is None:
            raise ValueError("Synthetic unequal-fact co-add must pass the core preflight")
        self.browser.dataset_selected.connect(self.show_run)
        self.browser.select_runs([combined])
        self.show_run(combined)

    @Slot(int)
    def show_run(self, number):
        dataset = self.browser.get_dataset(number)
        if dataset is not None:
            self.plot.plot_dataset(dataset)


class RunArithmeticScenario(Scenario):
    name = "run_arithmetic"
    description = "Synthetic co-add retains different acquired clocks and correction facts."
    size = (1250, 660)

    def build(self):
        return ArithmeticDemo()


register(RunArithmeticScenario())
