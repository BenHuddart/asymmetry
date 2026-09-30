"""Global Fit Wizard — the Apply step's review of the recommended LF-KT split.

Builds the Compare state of ``global_fit_wizard_result`` (a real coupled
optimisation of ``Longitudinal-field KT + Constant`` on the Ag decoupling series)
and presses **Continue with A →**. Apply then reviews what the global fit tab
will receive: the model, its Global/Local/Fixed roles, the starting values from
the first run's fit, and the collapsed "Why these roles?" rationale, above
**Apply to the global fit tab**.

Marked ``requires_fit = True`` because the coupled optimisation runs real fits.
"""

from __future__ import annotations

from PySide6.QtWidgets import QWidget

from ._base import _process_events_for, register
from .global_fit_wizard_result import GlobalFitWizardResultScenario


class GlobalFitWizardApplyScenario(GlobalFitWizardResultScenario):
    name = "global_fit_wizard_apply"
    description = (
        "Global Fit Wizard Apply step reviewing the LF-KT model, its parameter "
        "roles and starting values before they reach the global fit tab."
    )
    size = (1180, 560)

    def build(self) -> QWidget:
        window = super().build()
        window._compare_panel._continue.click()
        _process_events_for(milliseconds=150)
        return window


register(GlobalFitWizardApplyScenario())
