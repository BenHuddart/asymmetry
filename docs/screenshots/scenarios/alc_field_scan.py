"""ALC mode: a synthetic field scan with baseline and peak fits.

A data-free companion to the corpus walkthrough
:doc:`/workflows/alc_scan_tcnq`: a synthetic longitudinal-field scan whose
integral asymmetry dips at an avoided-level-crossing resonance near 3100 G,
taken through the ALC workflow — Build Scan, fit a baseline over the
non-resonant edges, then fit a Gaussian peak to the resonance.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

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
        window.resizeDocks([window._dock_data_browser], [340], Qt.Orientation.Horizontal)

        from asymmetry.core.simulate import reduce_run_to_dataset

        runs = [ds.run for ds in make_alc_field_scan()]
        # An explicitly supplied zero variance preserves the defined ratio but
        # removes its Gaussian weight; the remaining edge points fit the baseline.
        runs[0] = replace(
            runs[0],
            histograms=[replace(h, count_variance=np.zeros(h.n_bins)) for h in runs[0].histograms],
        )
        datasets = [reduce_run_to_dataset(run) for run in runs]
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
        # count-integral point per run through the cancellable worker.
        window._plot_workspace.set_active_view("integral_scan")
        _process_events_for(milliseconds=150)
        window._alc_fit_panel.build_requested.emit()
        _process_events_for(milliseconds=300)

        assert "Count integral" in window._alc_scan_view._provenance_label.text()
        assert "1 without Gaussian uncertainty" in window._alc_scan_view._provenance_label.text()
        view = window._alc_scan_view
        view._derivative_check.setChecked(True)
        assert "fit the original scan" in view._derivative_check.toolTip()
        assert window._alc_baseline is None
        assert not window._alc_rf_fitted
        view._derivative_check.setChecked(False)

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
        summary = view._peaks_results.text()
        assert summary.count("±") == 3
        assert "Uncertainties include the fitted baseline." in summary
        return window


register(AlcFieldScanScenario())


class AlcDerivativeScenario(AlcFieldScanScenario):
    name = "alc_derivative"
    description = "Adjacent midpoint derivatives in %/G, for display and export."

    def build(self) -> QWidget:
        window = super().build()
        window._alc_scan_view._derivative_check.setChecked(True)
        assert window._alc_baseline is None
        assert not window._alc_rf_fitted
        return window


register(AlcDerivativeScenario())


class AlcLegacyFitHistoryScenario(AlcFieldScanScenario):
    name = "alc_legacy_fit_history"
    description = "Restored settings-only scan with explicit missing-result guidance."
    size = (440, 560)

    def build(self) -> QWidget:
        from tempfile import TemporaryDirectory
        from pathlib import Path

        from asymmetry.core.project import save_project
        from asymmetry.gui.mainwindow import MainWindow

        original = super().build()
        state = original.collect_project_state()
        state["schema_version"] = 39
        extra = next(b["extra"] for b in state["batches"] if b["extra"].get("kind") == "alc_scan")
        extra.pop("fit_history")
        extra.update(baseline_fitted=True, peaks_fitted=True)
        original._clear_dirty()
        original.close()
        restored = MainWindow()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "owned-scan.asymp"
            save_project(state, path)
            restored._open_project_file(str(path))
        assert "no captured numerical result" in restored._alc_scan_view._fit_history_status.text()
        assert restored._alc_scan_view._fit_curve is None
        panel = restored._alc_scan_view.analysis_widget()
        panel.setParent(None)
        restored._clear_dirty()
        restored.close()
        restored.deleteLater()
        return panel


register(AlcLegacyFitHistoryScenario())
