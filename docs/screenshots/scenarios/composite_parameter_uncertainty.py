"""Composite parameter preview with a missing fitted uncertainty."""

from PySide6.QtWidgets import QWidget

from asymmetry.core.fitting.composite_parameters import CompositeParameterDefinition
from asymmetry.gui.panels.composite_parameter_dialog import CompositeParameterDialog

from ._base import Scenario, register


class CompositeParameterUncertaintyScenario(Scenario):
    name = "composite_parameter_uncertainty"
    description = "Composite frequency difference with an unavailable uncertainty."
    size = (800, 520)

    def build(self) -> QWidget:
        return CompositeParameterDialog(
            available_parameters=["frequency", "frequency_2"],
            existing_parameter_names=[],
            initial_definition=CompositeParameterDefinition(
                "frequency_difference", "frequency - frequency_2"
            ),
            preview_values={"frequency": 5.1, "frequency_2": 5.0},
            preview_uncertainties={"frequency": 0.02},
        )


register(CompositeParameterUncertaintyScenario())
