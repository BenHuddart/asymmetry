"""Conditional magnitude error bars on a synthetic scalar TF display."""

from __future__ import annotations

from PySide6.QtWidgets import QWidget

from asymmetry.gui.panels.plot_panel import PlotPanel

from ..data import make_euo_tf_tscan
from ._base import Scenario, register


class ScalarRRFMagnitudeScenario(Scenario):
    name = "scalar_rrf_magnitude"
    description = "Synthetic scalar TF Magnitude with shared-quadrature error bars."
    size = (1100, 650)

    def build(self) -> QWidget:
        panel = PlotPanel()
        panel.set_rrf_feature_enabled(True)
        controls = panel._rrf_controls
        controls._freq_spin.setValue(22.18)
        controls._phase_spin.setValue(37.0)
        controls._component_combo.setCurrentIndex(controls._component_combo.findData("magnitude"))
        controls._enable_check.setChecked(True)
        panel.plot_dataset(make_euo_tf_tscan()[0])
        return panel


register(ScalarRRFMagnitudeScenario())
