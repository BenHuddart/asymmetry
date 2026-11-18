"""Muonium intratriplet decay equation, angular-rate units and applicability."""

from asymmetry.core.fitting.composite import COMPONENTS
from asymmetry.gui.widgets.component_info_dialog import show_component_info_dialog

from ._base import Scenario, register


class MuoniumLFComponentInfoScenario(Scenario):
    name = "muonium_lf_component_info"
    description = "Muonium LF envelope help with equation, angular-rate units and observable limits."
    size = (900, 820)

    def build(self):
        return show_component_info_dialog(None, COMPONENTS["MuoniumLFRelax"])


register(MuoniumLFComponentInfoScenario())
