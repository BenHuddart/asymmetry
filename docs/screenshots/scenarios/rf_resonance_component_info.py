"""Selected radical RF branches and additive peak-parameter conventions."""

from asymmetry.core.fitting.parameter_models import PARAMETER_MODEL_COMPONENTS
from asymmetry.gui.widgets.component_info_dialog import show_component_info_dialog

from ._base import Scenario, register


class RFResonanceComponentInfoScenario(Scenario):
    name = "rf_resonance_component_info"
    description = "Selected radical RF branches and additive peak-parameter conventions."
    size = (820, 850)

    def build(self):
        return show_component_info_dialog(None, PARAMETER_MODEL_COMPONENTS["RFResonanceMuP"])


register(RFResonanceComponentInfoScenario())
