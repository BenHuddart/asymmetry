"""Vector polarization (3-axis) display on an EMU-style configuration.

Shows the three projection traces P_x, P_y, P_z from a synthetic vector-mode
measurement: the Z component carries the dominant longitudinal decay, X
carries a weak transverse oscillation, and Y is near zero with statistical
noise. The display demonstrates the GUI's overlay capability for
vector-polarimeter data (textbook Ch 6.3).
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from ..data import make_emu_vector
from ._base import Scenario, register


class VectorPolarizationEmuScenario(Scenario):
    name = "vector_polarization_emu"
    description = "Three EMU vector-polarization traces (P_x, P_y, P_z) overlaid."
    size = (1500, 920)

    def build(self) -> QWidget:
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window.resizeDocks(
            [window._dock_data_browser], [320], Qt.Orientation.Horizontal
        )

        datasets = make_emu_vector()
        for dataset in datasets:
            window._data_browser.add_dataset(dataset)

        run_numbers = [int(ds.run_number) for ds in datasets]
        window._data_browser.create_data_group(
            run_numbers, name="EMU vector projections"
        )
        window._plot_panel.set_overlay_enabled(True, emit_signal=True)
        window._data_browser._table.selectAll()
        window._on_dataset_selected(run_numbers[0])
        return window


register(VectorPolarizationEmuScenario())
