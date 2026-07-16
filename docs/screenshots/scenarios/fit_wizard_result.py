"""Fit Wizard Result page on the Ag ZF GKT dataset.

The rebuilt single-spectrum Fit Wizard is a three-state window
(Welcome → Running → Result). This scenario drives it straight to the
**Result** state: ``build_fit_wizard_recommendation`` is called
synchronously (instead of via the GUI's background worker) and handed to
``set_cached_recommendation``, exactly as the fit-panel cache does when a
previously analysed run is reopened. That populates the answer card — a
plain-language verdict, a confidence grade, and the data-with-recommended-fit
overlay — above the six-step decision trail. The wizard typically recommends
``StaticGKT_ZF + Constant`` for an Ag ZF dataset: the canonical
nuclear-dipolar fingerprint (Blundell et al. Ch 5.2).

Marked ``requires_fit = True`` because the underlying ``iminuit``-based
fits trip on numpy ≥ 2.3 in dev environments; CI keeps numpy < 2.3.
"""

from __future__ import annotations

from PySide6.QtWidgets import QWidget

from ..data import make_ag_zf_gkt
from ._base import Scenario, _process_events_for, register


class FitWizardResultScenario(Scenario):
    name = "fit_wizard_result"
    description = (
        "Fit Wizard Result page (answer card + decision trail) on the Ag ZF GKT dataset."
    )
    size = (1180, 920)
    requires_fit = True

    def build(self) -> QWidget:
        from asymmetry.core.fitting.fit_wizard import (
            SelectionMetric,
            build_fit_wizard_recommendation,
        )
        from asymmetry.gui.windows.fit_wizard_window import FitWizardWindow

        dataset = make_ag_zf_gkt()
        window = FitWizardWindow()
        window.set_analysis_context(dataset)
        _process_events_for(milliseconds=60)

        # Run the analysis synchronously instead of via the background worker,
        # then hand the recommendation to the window the same way the fit-panel
        # cache does when a previously analysed run is reopened.
        # ``set_cached_recommendation`` populates the answer card and decision
        # trail and switches the stacked content to the Result page.
        recommendation = build_fit_wizard_recommendation(
            dataset, current_model=None, metric=SelectionMetric.AICC
        )
        window.set_cached_recommendation(recommendation)
        _process_events_for(milliseconds=150)
        return window


register(FitWizardResultScenario())
