"""Global Fit Wizard — the Apply step's review of the recommended rung.

Builds the Compare state of ``global_fit_wizard_ladder`` (the sharing ladders of
the trend objective on the synthetic hopping scan) and presses **Continue with
A →**. Apply then reviews what the global fit tab will receive: the model, its
Global/Local/Fixed roles, the run the rung exempts and leaves out of the
coupled fit, the starting values from the first coupled run's fit, and the
collapsed "Why these roles?" rationale, above **Apply to the global fit tab**.

Marked ``requires_fit = True`` because the ladders are real coupled fits.
"""

from __future__ import annotations

from PySide6.QtWidgets import QWidget

from ._base import _process_events_for, register
from .global_fit_wizard_ladder import GlobalFitWizardLadderScenario


class GlobalFitWizardApplyScenario(GlobalFitWizardLadderScenario):
    name = "global_fit_wizard_apply"
    description = (
        "Global Fit Wizard Apply step reviewing the dynamic Kubo–Toyabe rung: its "
        "parameter roles, the exempt run left out, and the starting values."
    )
    size = (1180, 560)

    def build(self) -> QWidget:
        window = super().build()
        window._compare_panel._continue.click()
        _process_events_for(milliseconds=150)
        return window


register(GlobalFitWizardApplyScenario())
