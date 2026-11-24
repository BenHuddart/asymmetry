"""Model help distinguishes angular averages and processed spectral widths."""

from dataclasses import dataclass

from asymmetry.core.fitting.composite import COMPONENTS
from asymmetry.gui.widgets.component_info_dialog import show_component_info_dialog

from ._base import Scenario, register


@dataclass
class ComponentHelpScenario(Scenario):
    component_name: str
    size = (900, 540)

    def build(self):
        return show_component_info_dialog(None, COMPONENTS[self.component_name])


class StaticKTComponentHelpScenario(ComponentHelpScenario):
    name = "static_kt_component_help"
    description = "Static Gaussian KT help with component width and orientation-average tail."


class SpectralPeakComponentHelpScenario(ComponentHelpScenario):
    name = "spectral_peak_component_help"
    description = "Gaussian peak help with processed FWHM and limits of physical interpretation."


for scenario_type, component in (
    (StaticKTComponentHelpScenario, "StaticGKT_ZF"),
    (SpectralPeakComponentHelpScenario, "GaussianPeak"),
):
    register(scenario_type(component))
