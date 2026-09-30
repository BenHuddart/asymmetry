"""Fit Wizard Result page on the Ag ZF GKT dataset, with a candidate pinned as B.

The single-spectrum Fit Wizard is a three-state window (Welcome → Running →
Result). This scenario drives it straight to the **Result** state:
``build_fit_wizard_recommendation`` is called synchronously (instead of via the
GUI's background worker) and handed to ``set_cached_recommendation``, exactly
as the fit-panel cache does when a previously analysed run is reopened. That
populates the answer card — a plain-language verdict, a confidence grade, and
the data-with-recommended-fit overlay — and the Compare candidates section
beneath it, with the recommendation as candidate A. The wizard typically
recommends ``StaticGKT_ZF + Constant`` for an Ag ZF dataset: the canonical
nuclear-dipolar fingerprint (Blundell et al. Ch 5.2).

The candidates just behind it on this record are the same function reached
through a nested model (a rate or width fitted at zero), so their curves would
hide under A's. B is therefore the best candidate at least
``SHORTLIST_MAX_DELTA`` behind A — one with "essentially no support" — so the
dashed B and its structured grey residuals read at a glance. The build leaves
that row without dense curves, so pinning it exercises the on-demand curve
path; the scenario waits for the window's worker before the grab.

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
        "Fit Wizard Result page (answer card + Compare candidates with A and a pinned B) "
        "on the Ag ZF GKT dataset."
    )
    size = (1180, 1100)
    requires_fit = True

    def build(self) -> QWidget:
        from asymmetry.core.fitting.fit_wizard import (
            SelectionMetric,
            build_fit_wizard_recommendation,
        )
        from asymmetry.core.fitting.model_comparison import (
            SHORTLIST_MAX_DELTA,
            summarise_single_candidates,
        )
        from asymmetry.gui.windows.fit_wizard_window import FitWizardWindow

        dataset = make_ag_zf_gkt()
        window = FitWizardWindow()
        window.set_analysis_context(dataset)
        _process_events_for(milliseconds=60)

        # Run the analysis synchronously instead of via the background worker,
        # then hand the recommendation to the window the same way the fit-panel
        # cache does when a previously analysed run is reopened.
        recommendation = build_fit_wizard_recommendation(
            dataset, current_model=None, metric=SelectionMetric.AICC
        )
        window.set_cached_recommendation(recommendation)
        _process_events_for(milliseconds=150)

        summaries = summarise_single_candidates(recommendation, dataset, recommendation.metric)
        b_key = next(s.key for s in summaries if s.delta >= SHORTLIST_MAX_DELTA)
        window._model_compare.set_b(b_key)
        # B's dense curve is built on the window's TaskRunner; wait for it.
        for _ in range(100):
            if not window._tasks.active_count:
                break
            _process_events_for(milliseconds=50)
        _process_events_for(milliseconds=150)
        return window


register(FitWizardResultScenario())
