"""Run-metadata logbook view backed by the EuO temperature scan.

The data browser doubles as the logbook in the Asymmetry GUI — there is
no separate logbook panel — so this scenario foregrounds the data browser
populated with the six-temperature EuO scan and sorted by T(K), making
the run/title/temperature/field metadata visible at a glance.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from ..data import make_euo_tf_tscan
from ._base import Scenario, register


class LogbookViewScenario(Scenario):
    name = "logbook_view"
    description = "Data-browser logbook view of an EuO temperature scan sorted by T(K)."
    size = (820, 640)

    def build(self) -> QWidget:
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        # Widen the data browser dock so all metadata columns are visible
        # while the central plot panel shrinks out of focus — this page is
        # about the logbook metadata, not the plot.
        window.resizeDocks(
            [window._dock_data_browser], [620], Qt.Orientation.Horizontal
        )
        for dataset in make_euo_tf_tscan():
            window._data_browser.add_dataset(dataset)
        # Sort by temperature column (index 2) so the scan reads top-to-bottom.
        browser = window._data_browser
        browser._current_sort_column = 2
        browser._current_sort_order = Qt.SortOrder.AscendingOrder
        browser._sort_table(rebuild=True)
        return window


register(LogbookViewScenario())
