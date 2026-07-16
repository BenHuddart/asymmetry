"""Global Fit Wizard — Result answer card on the Ag LF-KT decoupling series.

Drives the rebuilt three-state Global Fit Wizard straight to its **Result**
state. A real recommendation is built synchronously by
``build_global_fit_wizard_recommendation`` and handed to the window via
``set_cached_recommendation`` — the same path the fit-panel cache uses when a
previously analysed series is reopened. That populates the series answer card
(verdict headline, overlaid data-and-fit traces colour-graded along the series
axis, and the local-parameter trend panel).

The window height crops the capture at the bottom of the answer card, just
above the screening shortlist: on the cached-recommendation path every
surviving candidate has been through coupled optimisation, so
``sorted_prescreen_assessments()`` is empty and the shortlist renders with no
rows — an empty table would read as a broken UI in the docs. The shortlist in
a populated state is a live-journey artefact (between screening and coupled
optimisation) and is described in prose instead.

The scope is restricted to the longitudinal-field Kubo–Toyabe family so the
screening portfolio stays small and the whole build completes in well under a
minute; the recommendation, its shortlist scores, and the fit overlays are all
genuinely computed, not fabricated. The wizard recommends
``Longitudinal-field KT + Constant`` with Δ shared globally and B_L local — the
textbook decoupling model (Hayano et al., Phys. Rev. B **20**, 850 (1979)).

Marked ``requires_fit = True`` because the coupled optimisation uses the
``iminuit``-based engine, which trips on numpy ≥ 2.3 in dev environments; CI
keeps numpy < 2.3.
"""

from __future__ import annotations

from PySide6.QtWidgets import QWidget

from ..data import make_ag_lf_decoupling
from ._base import Scenario, _process_events_for, register


class GlobalFitWizardResultScenario(Scenario):
    name = "global_fit_wizard_result"
    description = (
        "Global Fit Wizard Result answer card on the Ag LF-KT decoupling "
        "series (verdict + series overlay + local-parameter trend)."
    )
    size = (1180, 748)
    requires_fit = True

    def build(self) -> QWidget:
        from asymmetry.core.fitting.global_fit_wizard import (
            build_global_fit_wizard_recommendation,
        )
        from asymmetry.core.fitting.wizard_scope import WizardScope, WizardScopePreset
        from asymmetry.gui.windows.global_fit_wizard_window import GlobalFitWizardWindow

        datasets = make_ag_lf_decoupling(fields_g=(0.0, 15.0, 50.0, 100.0))

        # Keep the screening portfolio small so the build is fast: the
        # longitudinal-field dynamics preset with the competing relaxation
        # leaves excluded leaves the LF-KT family (plus the always-on baselines).
        exclude = frozenset(
            {
                "StaticGKT_ZF",
                "DynamicGaussianKT",
                "DynamicLorentzianKT",
                "GaussianBroadenedKT",
                "ExponentialRelaxation",
                "GaussianRelaxation",
                "StretchedExponential",
                "RischKehr",
                "MuoniumLF",
                "Oscillatory",
            }
        )
        scope = WizardScope(
            preset=WizardScopePreset.LF_DYNAMICS, exclude_components=exclude
        )

        # Only the LF-KT candidate goes through coupled optimisation, which keeps
        # the Global/Local role search tractable while still exercising it.
        recommendation = build_global_fit_wizard_recommendation(
            datasets,
            scope=scope,
            selected_template_keys=("lf_kt_constant",),
        )

        window = GlobalFitWizardWindow()
        window.set_analysis_context(datasets)
        _process_events_for(milliseconds=60)
        window.set_cached_recommendation(
            recommendation, signature={"scope": scope.to_payload()}
        )
        _process_events_for(milliseconds=200)
        return window


register(GlobalFitWizardResultScenario())
