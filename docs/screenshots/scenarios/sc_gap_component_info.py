"""Declared gap-basis and thermal-response interpretation limits."""

from asymmetry.core.fitting.parameter_models import PARAMETER_MODEL_COMPONENTS
from asymmetry.gui.widgets.component_info_dialog import show_component_info_dialog

from ._base import Scenario, register


class SCGapComponentInfoScenario(Scenario):
    name = "sc_gap_component_info"
    description = "Declared gap-basis and thermal-response interpretation limits."
    size = (820, 850)

    def build(self):
        return show_component_info_dialog(None, PARAMETER_MODEL_COMPONENTS["SC_NonmonotonicD"])


register(SCGapComponentInfoScenario())
