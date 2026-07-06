"""Global Fit Wizard — Setup page on an Ag LF Kubo–Toyabe decoupling series.

The rebuilt Global Fit Wizard is a three-state window
(Setup → Running → Result). This scenario drives it to the opening **Setup**
state via ``set_analysis_context`` alone — no fit runs. That populates the
Series overview (one row per run, with Run / Field (G) / Temperature (K)
filled immediately, the classification columns still ``—`` until screening),
the Scope selector, the collapsed *Guide the search (optional)* expectations
editor, the search settings, and the primary *Run screening* button.

Uses the four-field Ag LF decoupling series (B_L = 0, 15, 50, 100 G against
Δ=0.39 μs⁻¹). No ``requires_fit`` marker: the Setup page shows before any
analysis, so this captures instantly in every environment.
"""

from __future__ import annotations

from PySide6.QtWidgets import QWidget

from ..data import make_ag_lf_decoupling
from ._base import Scenario, register, _process_events_for


class GlobalFitWizardSetupScenario(Scenario):
    name = "global_fit_wizard_setup"
    description = (
        "Global Fit Wizard Setup page on the Ag LF-KT decoupling series "
        "(series overview, scope, run-screening CTA)."
    )
    size = (1180, 880)

    def build(self) -> QWidget:
        from asymmetry.gui.windows.global_fit_wizard_window import GlobalFitWizardWindow

        datasets = make_ag_lf_decoupling(fields_g=(0.0, 15.0, 50.0, 100.0))
        window = GlobalFitWizardWindow()
        window.set_analysis_context(datasets)
        _process_events_for(milliseconds=150)
        return window


register(GlobalFitWizardSetupScenario())
