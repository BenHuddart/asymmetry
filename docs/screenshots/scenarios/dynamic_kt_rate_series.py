"""Synthetic Gaussian strong-collision rate sweep at the Ag nuclear-field width."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from ..data import make_ag_dynamic_rate_scan
from ._base import Scenario, register


class DynamicKtRateSeriesScenario(Scenario):
    name = "dynamic_kt_rate_series"
    description = (
        "Gaussian strong-collision relaxation from the static to the fast-fluctuation regime."
    )
    size = (1500, 920)

    def build(self) -> QWidget:
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window.resizeDocks([window._dock_data_browser], [340], Qt.Orientation.Horizontal)
        datasets = make_ag_dynamic_rate_scan()
        for dataset in datasets:
            window._data_browser.add_dataset(dataset)
        run_numbers = [int(dataset.run_number) for dataset in datasets]
        window._data_browser.create_data_group(run_numbers, name="Dynamic KT — Ag")
        window._plot_panel.set_overlay_enabled(True, emit_signal=True)
        window._data_browser._table.selectAll()
        window._on_dataset_selected(run_numbers[0])
        label = window._plot_panel._label_field_combo.findData("comment")
        window._plot_panel._label_field_combo.setCurrentIndex(label)
        window._dock_fit.hide()
        return window


register(DynamicKtRateSeriesScenario())
