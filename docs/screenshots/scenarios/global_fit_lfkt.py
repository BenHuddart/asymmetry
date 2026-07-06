"""Global fit setup for an Ag LF Kubo–Toyabe field-decoupling series.

Loads four longitudinal-field Ag datasets (B_L = 0, 15, 50, 100 G against
Δ=0.39 μs⁻¹, spanning the textbook decoupling units γ_μB_L/Δ ∈ {0, 3, 9, 19}),
groups them, switches the fit panel to the Global tab, and captures the main
window so the docs show the typical pre-fit setup users see before running a
global optimisation. The Δ parameter is shared across all four runs; LF
field B_L is the local parameter that scans (Hayano PRB 20, 850, 1979).
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from ..data import make_ag_lf_decoupling
from ._base import Scenario, register


class GlobalFitLfktScenario(Scenario):
    name = "global_fit_lfkt"
    description = "Global fit tab populated with an Ag LF-KT decoupling series."
    size = (1500, 920)

    def build(self) -> QWidget:
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window._on_fit()
        window.resizeDocks(
            [window._dock_data_browser], [340], Qt.Orientation.Horizontal
        )

        datasets = make_ag_lf_decoupling(fields_g=(0.0, 15.0, 50.0, 100.0))
        for dataset in datasets:
            window._data_browser.add_dataset(dataset)

        run_numbers = [int(ds.run_number) for ds in datasets]
        window._data_browser.create_data_group(
            run_numbers, name="LF decoupling — Ag"
        )

        # Surface the Global tab of the fit panel.
        window._fit_panel._tabs.setCurrentWidget(window._fit_panel._global_tab)
        window._fit_panel.set_datasets(datasets)

        # Select the first run so the central plot shows the series in context.
        window._on_dataset_selected(run_numbers[0])
        return window


register(GlobalFitLfktScenario())
