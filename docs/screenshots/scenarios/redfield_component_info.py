"""Cyclic-frequency Redfield scales and physical correlation-time mapping."""

from asymmetry.core.fitting.parameter_models import PARAMETER_MODEL_COMPONENTS
from asymmetry.gui.widgets.component_info_dialog import show_component_info_dialog

from ._base import Scenario, register


class RedfieldComponentInfoScenario(Scenario):
    name = "redfield_component_info"
    description = "Cyclic-frequency Redfield scales and physical correlation-time mapping."
    size = (820, 850)

    def build(self):
        return show_component_info_dialog(None, PARAMETER_MODEL_COMPONENTS["Redfield"])


register(RedfieldComponentInfoScenario())
