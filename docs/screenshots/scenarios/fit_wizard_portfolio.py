"""Fit Wizard portfolio comparison page on the Ag ZF GKT dataset.

The wizard's analysis routine (``build_fit_wizard_recommendation``) is
called synchronously here instead of via the GUI's background worker, so
the captured frame shows the populated **Candidate Portfolio** tab with
the recommended model ranked first by the AICc metric. The wizard
typically picks ``StaticGKT_ZF + Constant`` for an Ag ZF dataset — the
canonical nuclear-dipolar fingerprint (Blundell et al. Ch 5.2).

Marked ``requires_fit = True`` because the underlying ``iminuit``-based
fits trip on numpy ≥ 2.3 in dev environments; CI keeps numpy < 2.3.
"""

from __future__ import annotations

from PySide6.QtWidgets import QWidget

from ..data import make_ag_zf_gkt
from ._base import Scenario, register, _process_events_for


class FitWizardPortfolioScenario(Scenario):
    name = "fit_wizard_portfolio"
    description = "Fit Wizard Candidate Portfolio page on the Ag ZF GKT dataset."
    size = (1280, 920)
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

        # Run the analysis synchronously instead of via the background worker
        # so the screenshot captures a populated state.
        recommendation = build_fit_wizard_recommendation(
            dataset, current_model=None, metric=SelectionMetric.AICC
        )
        window._on_analysis_finished(window._analysis_request_id, recommendation)
        _process_events_for(milliseconds=120)

        # Switch to the Candidate Portfolio tab.
        window._tabs.setCurrentWidget(window._portfolio_tab)
        _process_events_for(milliseconds=60)
        return window


register(FitWizardPortfolioScenario())
