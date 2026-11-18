"""Synthetic spread of static widths at the Ag nuclear-field archetype scale."""

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.models import gaussian_broadened_kt
from asymmetry.core.simulate_presets import DELTA_AG_PER_US

from ..data.archetypes import make_ag_zf_gkt
from ._base import Scenario, register


class GaussianBroadenedKtSeriesScenario(Scenario):
    name = "gaussian_broadened_kt_series"
    description = "Synthetic Gaussian-width broadening and longitudinal-field decoupling."
    size = (1500, 920)

    def build(self) -> QWidget:
        from asymmetry.gui.mainwindow import MainWindow

        reference = make_ag_zf_gkt()
        rng = np.random.default_rng(72)
        window = MainWindow()
        window.resizeDocks([window._dock_data_browser], [340], Qt.Orientation.Horizontal)
        runs = []
        for index, (width, field) in enumerate(((0, 0), (0.3, 0), (1, 0), (1, 5))):
            clean = 24 * gaussian_broadened_kt(reference.time, DELTA_AG_PER_US, field, width) + 0.3
            run = 5801 + index
            label = f"w_rel={width:g}, B_L={field:g} G"
            window._data_browser.add_dataset(
                MuonDataset(
                    time=reference.time,
                    asymmetry=clean + rng.normal(0, reference.error),
                    error=reference.error,
                    metadata={
                        "run_number": run,
                        "title": f"Synthetic static widths: {label}",
                        "comment": label,
                        "field": field,
                        "temperature": 20.0,
                    },
                )
            )
            runs.append(run)
        window._data_browser.create_data_group(runs, name="Synthetic width distributions")
        window._plot_panel.set_overlay_enabled(True, emit_signal=True)
        window._data_browser._table.selectAll()
        window._on_dataset_selected(runs[0])
        label = window._plot_panel._label_field_combo.findData("comment")
        window._plot_panel._label_field_combo.setCurrentIndex(label)
        for dock in (window._dock_fit, window._dock_fit_parameters, window._dock_fourier):
            dock.hide()
        return window


register(GaussianBroadenedKtSeriesScenario())
