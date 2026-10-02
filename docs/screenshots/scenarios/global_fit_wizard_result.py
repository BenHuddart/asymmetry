"""Global Fit Wizard — the Compare step on the Ag LF-KT series, A against B.

A real recommendation is built synchronously by
``build_global_fit_wizard_recommendation`` and handed to the window through
``set_cached_recommendation``, the path the fit panel's cache takes when an
analysed series is reopened. It lands on **Compare**, since the recommendation
holds optimised role splits.

The scope is restricted to the longitudinal-field Kubo–Toyabe family, so the
build completes in seconds. Only ``lf_kt_constant`` goes through the coupled
role search, and it yields eight role splits of the one template. A is the
recommendation: Δ shared, B_L local, the textbook decoupling model (Hayano
et al., Phys. Rev. B **20**, 850 (1979)). B is pinned to the runner-up split,
which frees Δ per run as well. It fits every run as well by eye, but scores
+34 AICc and fails the runs test at 100 G. The trend plot follows Δ: A's
shared value is one line across the series, while B's per-run Δ falls away at
100 G, where the decoupled signal no longer constrains it and Δ trades off
against B_L. The capture shows why sharing a parameter is the better answer
when the data allow it.

Marked ``requires_fit = True`` because the coupled optimisation runs real
fits.
"""

from __future__ import annotations

from PySide6.QtWidgets import QWidget

from ..data import make_ag_lf_decoupling
from ._base import Scenario, _process_events_for, register


class GlobalFitWizardResultScenario(Scenario):
    name = "global_fit_wizard_result"
    description = (
        "Global Fit Wizard Compare step on the Ag LF-KT decoupling series — "
        "the shared-Δ split as A against a per-run-Δ split pinned as B."
    )
    size = (1180, 748)
    requires_fit = True

    def build(self) -> QWidget:
        from asymmetry.core.fitting.component_tags import FieldGeometry, PhysicsClass
        from asymmetry.core.fitting.composite import COMPONENTS
        from asymmetry.core.fitting.global_fit_wizard import (
            build_global_fit_wizard_recommendation,
        )
        from asymmetry.core.fitting.global_search.trend_objective import SelectionObjective
        from asymmetry.core.fitting.wizard_scope import WizardScope
        from asymmetry.gui.windows.global_fit_wizard_window import GlobalFitWizardWindow

        # Default noise seed. At 100 G, with B_L free (field recorded, geometry
        # not), Delta and B_L trade off, so on some seeds a per-run Delta scores
        # better; on this one the shared Delta wins (AICc 1964 against 1998).
        datasets = make_ag_lf_decoupling(fields_g=(0.0, 15.0, 50.0, 100.0))

        # Keep the screening portfolio small so the build is fast: LF dynamics
        # and magnetism with the competing relaxation leaves excluded leaves the
        # LF-KT family (plus the always-on baselines). The runs record no field
        # direction, so the components that cannot apply in LF are excluded too.
        not_lf = {
            name
            for name, definition in COMPONENTS.items()
            if FieldGeometry.LF not in definition.field_geometries
        }
        exclude = frozenset(not_lf) | frozenset(
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
            physics=frozenset({PhysicsClass.DYNAMICS, PhysicsClass.MAGNETISM}),
            exclude_components=exclude,
        )

        # Only the LF-KT candidate goes through coupled optimisation, which keeps
        # the Global/Local role search tractable while still exercising it.
        # The statistical objective, which is what the page describes: the
        # Compare step's ladder view of the trend objective is not built yet.
        recommendation = build_global_fit_wizard_recommendation(
            datasets,
            scope=scope,
            selected_template_keys=("lf_kt_constant",),
            objective=SelectionObjective.STATISTICAL,
        )

        window = GlobalFitWizardWindow()
        window.set_analysis_context(datasets)
        _process_events_for(milliseconds=60)
        window.set_cached_recommendation(recommendation, signature={"scope": scope.to_payload()})
        panel = window._compare_panel
        # B: the same template with Δ free per run, the runner-up split.
        b_key = next(
            assessment.selection_key
            for assessment in recommendation.sorted_optimized_assessments()
            if set(assessment.local_param_names) == {"Delta", "B_L"}
        )
        panel.set_b(b_key)
        # Plot Δ along the series: A's shared value as a band, B's per-run values.
        panel._on_table_clicked(
            next(index for index, pair in enumerate(panel._pairs) if pair.name == "Delta"), 0
        )
        _process_events_for(milliseconds=200)
        return window


register(GlobalFitWizardResultScenario())
