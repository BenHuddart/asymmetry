"""ALC mode: a synthetic field scan with baseline and peak fits.

A data-free companion to the corpus walkthrough
:doc:`/workflows/alc_scan_tcnq`: a synthetic longitudinal-field scan whose
integral asymmetry dips at an avoided-level-crossing resonance near 3100 G,
taken through the ALC workflow — Build Scan, fit a baseline over the
non-resonant edges, then fit a Gaussian peak to the resonance.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTableWidgetItem, QWidget

from ..data import make_alc_field_scan
from ._base import Scenario, _process_events_for, register


class AlcFieldScanScenario(Scenario):
    name = "alc_field_scan"
    description = "ALC integral-asymmetry field scan with baseline and Gaussian peak fits."
    size = (1500, 920)
    requires_fit = True

    def build(self) -> QWidget:
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window.resizeDocks(
            [window._dock_data_browser], [340], Qt.Orientation.Horizontal
        )

        datasets = make_alc_field_scan()
        for dataset in datasets:
            window._data_browser.add_dataset(dataset)
        window._data_browser.create_data_group(
            [int(ds.run_number) for ds in datasets], name="ALC scan — 2000–5000 G"
        )
        # Multi-select every run so the fit panel's batch (which the scan build
        # reads) is populated.
        window._data_browser._table.selectAll()
        _process_events_for(milliseconds=200)

        # Enter ALC mode (the integral-scan view) and build the scan — one
        # integral-asymmetry point per run. The build is synchronous.
        window._plot_workspace.set_active_view("integral_scan")
        _process_events_for(milliseconds=150)
        window._alc_fit_panel.build_requested.emit()
        _process_events_for(milliseconds=300)

        view = window._alc_scan_view

        # Baseline: a linear fit over the two non-resonant edges (the dip is
        # near 3100 G). Both ALC fits are synchronous.
        view._baseline_model_combo.setCurrentText("Linear")
        view._add_region()
        view._regions_table.setItem(0, 0, QTableWidgetItem("2000"))
        view._regions_table.setItem(0, 1, QTableWidgetItem("2700"))
        view._add_region()
        view._regions_table.setItem(1, 0, QTableWidgetItem("3500"))
        view._regions_table.setItem(1, 1, QTableWidgetItem("5000"))
        _process_events_for(milliseconds=120)
        view.baseline_fit_requested.emit()
        _process_events_for(milliseconds=250)

        # Peak: a Gaussian seeded a little off the resonance so the fit moves.
        view._add_peak("Gaussian")
        view._peaks_table.setItem(0, 1, QTableWidgetItem("3000"))
        view._peaks_table.setItem(0, 2, QTableWidgetItem("300"))
        view._peaks_table.setItem(0, 3, QTableWidgetItem("-5"))
        _process_events_for(milliseconds=120)
        view.peaks_fit_requested.emit()
        _process_events_for(milliseconds=300)
        return window


register(AlcFieldScanScenario())
