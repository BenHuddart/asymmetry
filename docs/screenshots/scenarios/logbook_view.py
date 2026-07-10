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
from ._base import Scenario, _process_events_for, register


class LogbookViewScenario(Scenario):
    name = "logbook_view"
    description = "Data-browser logbook view of an EuO temperature scan sorted by T(K)."
    # Wide enough that a 620 px data-browser dock still leaves the inspector
    # deck its char-based minimum width (see settle()) instead of squeezing
    # it below what MainWindow enforces.
    size = (1180, 640)

    def build(self) -> QWidget:
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        # Hide the Fit/Parameters inspector deck and the log pane: this page
        # is about the logbook metadata, not fitting, and reclaiming their
        # space lets the data browser dominate the frame instead of a large
        # empty plot flanked by an unrelated fit form.
        window._dock_fit.hide()
        window._dock_fourier.hide()
        window._dock_fit_parameters.hide()
        window._dock_log.hide()
        for dataset in make_euo_tf_tscan():
            window._data_browser.add_dataset(dataset)
        # Sort by temperature column (index 2) so the scan reads top-to-bottom.
        browser = window._data_browser
        browser._current_sort_column = 2
        browser._current_sort_order = Qt.SortOrder.AscendingOrder
        browser._sort_table(rebuild=True)
        # Select the 65 K run through the real pathway so the central plot
        # shows actual data — a blank 0–1 axis reads as a broken session.
        # 3003 rather than the 30 K run: ν(30 K) ≈ 22 MHz aliases into a
        # braided moiré at the full 6.3 µs view, while the slower 65 K
        # precession renders as a legible decaying signal at this zoom.
        # select_runs performs a true browser-table selection (row highlight
        # + selection count); _on_dataset_selected is the same MainWindow
        # slot the browser's own signal drives, and binds the run to the plot.
        browser.select_runs([3003])
        window._on_dataset_selected(3003)
        return window

    def settle(self, widget: QWidget) -> None:
        # MainWindow re-applies its own default dock widths from a
        # QTimer.singleShot(0, ...) queued on its first showEvent (see
        # MainWindow._apply_default_dock_widths / showEvent) — a resize
        # issued in build(), before the window is shown, or even here before
        # that deferred call has run, is silently clobbered once it fires.
        # Pump the event loop first so MainWindow's own pass completes, then
        # override it with the wide browser dock this scenario needs.
        _process_events_for(milliseconds=50)
        widget.resizeDocks(
            [widget._dock_data_browser], [620], Qt.Orientation.Horizontal
        )
        super().settle(widget)


register(LogbookViewScenario())
