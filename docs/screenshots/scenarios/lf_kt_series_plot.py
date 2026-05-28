"""LF Kubo–Toyabe field-decoupling series overlay on Ag polycrystal.

Loads the five-field LF series (B_L = 0, 5, 10, 25, 50 G against Δ=0.39
μs⁻¹, spanning the textbook decoupling units γ_μB_L/Δ ∈ {0, 1, 2, 5, 10},
Fig 5.6 of Blundell et al.) and selects all runs so the central plot
overlays the decoupling progression. The data browser groups them as
"LF decoupling — Ag" to keep the run table tidy.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from ..data import make_ag_lf_decoupling
from ._base import Scenario, register


class LfKtSeriesPlotScenario(Scenario):
    name = "lf_kt_series_plot"
    description = "Time-domain overlay of an Ag LF Kubo–Toyabe field-decoupling series."
    size = (1500, 920)

    def build(self) -> QWidget:
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window.resizeDocks(
            [window._dock_data_browser], [320], Qt.Orientation.Horizontal
        )

        datasets = make_ag_lf_decoupling()
        for dataset in datasets:
            window._data_browser.add_dataset(dataset)

        run_numbers = [int(ds.run_number) for ds in datasets]
        window._data_browser.create_data_group(
            run_numbers, name="LF decoupling — Ag"
        )

        # Enable Overlay before selecting so all five LF runs render together.
        window._plot_panel.set_overlay_enabled(True, emit_signal=True)
        window._data_browser._table.selectAll()
        window._on_dataset_selected(run_numbers[0])
        return window


register(LfKtSeriesPlotScenario())
