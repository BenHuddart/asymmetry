"""Muon-fluorine entanglement signal in PbF₂.

PbF₂ provides a clean F-μ-F demonstration: the heavy Pb host carries no
significant nuclear moment so the polarization is dominated by the
analytical F-μ-F dipolar pattern (Brewer et al. PRB 33, 7813, 1986;
textbook Ch 4.6). The scenario loads the dataset, switches the fit panel
to ``FmuF_Linear + Constant``, and selects the run so the characteristic
beat envelope is visible across a 20 μs time window. No fit is run —
the screenshot is about the model selection workflow.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from ..data import make_pbf2_fmuf
from ._base import Scenario, register, _process_events_for


class MuonFluorinePbf2Scenario(Scenario):
    name = "muon_fluorine_pbf2"
    description = "Main window with PbF₂ F-μ-F dataset and FmuF_Linear+Constant model selected."
    size = (1500, 920)

    def build(self) -> QWidget:
        from asymmetry.core.fitting.composite import CompositeModel
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window._on_fit()
        window.resizeDocks(
            [window._dock_data_browser], [320], Qt.Orientation.Horizontal
        )

        dataset = make_pbf2_fmuf()
        window._data_browser.add_dataset(dataset)
        window._on_dataset_selected(dataset.run_number)

        single_tab = window._fit_panel._single_tab
        single_tab._set_composite_model(
            CompositeModel(["FmuF_Linear", "Constant"], operators=["+"])
        )
        _process_events_for(milliseconds=80)
        return window


register(MuonFluorinePbf2Scenario())
