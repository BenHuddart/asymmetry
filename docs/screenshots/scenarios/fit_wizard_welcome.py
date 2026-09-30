"""Fit Wizard — Welcome page with the model family picker, on the Ag ZF dataset.

The single-spectrum Fit Wizard opens on its **Welcome** page, where the model
family picker sits open above **Analyze**. This scenario reaches that state via
``set_analysis_context`` alone — no fit runs. The synthetic run records no
field direction, so the scenario answers **Zero field** the way a user would
(the "Set by you" note then shows and the TF- and LF-only models drop out) and
points the details panel at the static Gaussian Kubo–Toyabe model, the
function this Ag spectrum follows (Blundell et al. Ch 5.2).
"""

from __future__ import annotations

from PySide6.QtWidgets import QWidget

from ..data import make_ag_zf_gkt
from ._base import Scenario, _process_events_for, register

#: The direction row's "Zero field" button (Zero field, Longitudinal, Transverse, Not recorded).
_ZERO_FIELD = 0


class FitWizardWelcomeScenario(Scenario):
    name = "fit_wizard_welcome"
    description = (
        "Fit Wizard Welcome page on the Ag ZF dataset: the model family picker "
        "with a zero-field answer, above Analyze."
    )
    size = (1180, 820)

    def build(self) -> QWidget:
        from asymmetry.gui.windows.fit_wizard_window import FitWizardWindow

        window = FitWizardWindow()
        window.set_analysis_context(make_ag_zf_gkt())
        picker = window._picker
        picker._direction_group.button(_ZERO_FIELD).click()
        picker._show_details("StaticGKT_ZF")
        _process_events_for(milliseconds=150)
        return window


register(FitWizardWelcomeScenario())
