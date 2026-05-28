"""Main window hero shot: EuO ZF temperature scan loaded with one run selected.

Shows the WiMDA-style layout (data browser left, plot centre, fit panel
right docked) populated with the six-temperature EuO ferromagnet scan
crossing Tc=69 K (Blundell PRB 81, 092407, 2010 — textbook Fig 6.6). The
selected run sits at T=65 K, just inside the ordered state, where the
spontaneous-field precession is at its slowest and the critical damping
is largest — making the time-domain signature of the order parameter
clearly visible.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from ..data import make_euo_tf_tscan
from ._base import Scenario, register


class MainWindowScenario(Scenario):
    name = "main_window"
    description = "Default layout with an EuO temperature scan loaded and one run selected."
    size = (1500, 920)

    def build(self) -> QWidget:
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        # Surface the fit dock so the full WiMDA-style layout is visible.
        window._on_fit()
        # Wider data-browser dock so Run/Title/T(K)/B(G) all fit.
        window.resizeDocks(
            [window._dock_data_browser], [380], Qt.Orientation.Horizontal
        )
        for dataset in make_euo_tf_tscan():
            window._data_browser.add_dataset(dataset)
        window._on_dataset_selected(3003)  # T=65 K, just below Tc
        return window


register(MainWindowScenario())
