"""Global Fit Wizard — Setup page on an Ag LF Kubo–Toyabe decoupling series.

The Global Fit Wizard is a three-state window (Setup → Running → Result). This
scenario drives it to the opening **Setup** state via ``set_analysis_context``
alone — no fit runs — and frames the **Scope** section: the model family
picker. The synthetic runs record no field direction, so the scenario answers
**Longitudinal** the way a user would (the "Set by you" note then shows and
the ZF-only models drop out), points the details panel at the dynamic Gaussian
Kubo–Toyabe model, and is tall enough to show the whole page down to the
*Run screening* button.

Uses the four-field Ag LF decoupling series (B_L = 0, 15, 50, 100 G against
Δ=0.39 μs⁻¹). No ``requires_fit`` marker: the Setup page shows before any
analysis, so this captures instantly in every environment.
"""

from __future__ import annotations

from PySide6.QtWidgets import QWidget

from ..data import make_ag_lf_decoupling
from ._base import Scenario, _process_events_for, register

#: The direction row's "Longitudinal" button (Zero field, Longitudinal, Transverse, Not recorded).
_LONGITUDINAL = 1


class GlobalFitWizardSetupScenario(Scenario):
    name = "global_fit_wizard_setup"
    description = (
        "Global Fit Wizard Setup page on the Ag LF-KT decoupling series "
        "(model family picker with an LF answer, run-screening CTA)."
    )
    size = (1180, 930)

    def build(self) -> QWidget:
        from asymmetry.gui.windows.global_fit_wizard_window import GlobalFitWizardWindow

        datasets = make_ag_lf_decoupling(fields_g=(0.0, 15.0, 50.0, 100.0))
        window = GlobalFitWizardWindow()
        window.set_analysis_context(datasets)
        picker = window._picker
        picker._direction_group.button(_LONGITUDINAL).click()
        picker._show_details("DynamicGaussianKT")
        _process_events_for(milliseconds=150)
        return window


register(GlobalFitWizardSetupScenario())
