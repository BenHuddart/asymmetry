"""Converged single Gaussian Kubo–Toyabe fit on ZF Ag polycrystal.

Ag is the canonical nuclear-dipolar reference sample at every μSR facility:
the ZF static Gaussian Kubo–Toyabe (Δ≈0.39 μs⁻¹) used here is the function
introduced by Kubo & Toyabe in 1966 and discussed in Blundell et al. Ch 5.2
(eq 5.13). The fit panel is configured with ``StaticGKT_ZF + Constant`` and
the fit is run so the screenshot captures the parameter table populated
with fitted values and uncertainties.

Marked as ``requires_fit = True`` because iminuit/numba breaks on
numpy ≥ 2.3 on dev environments; CI keeps numpy < 2.3 via constraints.txt.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from ..data import make_ag_zf_gkt
from ._base import Scenario, _process_events_for, register


class FitWizardGktScenario(Scenario):
    name = "fit_wizard_gkt"
    description = "Single-fit panel with a converged Gaussian Kubo–Toyabe fit on Ag."
    size = (1500, 920)
    requires_fit = True

    def build(self) -> QWidget:
        from asymmetry.core.fitting.composite import CompositeModel
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window._on_fit()
        window.resizeDocks(
            [window._dock_data_browser], [320], Qt.Orientation.Horizontal
        )

        dataset = make_ag_zf_gkt()
        window._data_browser.add_dataset(dataset)
        window._on_dataset_selected(dataset.run_number)

        # Configure the single-fit tab with a GKT + baseline composite. Defaults
        # for A0/Delta/baseline sit close enough to the synthetic truth that
        # the fit converges in one call without further parameter tweaking.
        single_tab = window._fit_panel._single_tab
        single_tab._set_composite_model(
            CompositeModel(["StaticGKT_ZF", "Constant"], operators=["+"])
        )
        _process_events_for(milliseconds=80)
        single_tab._run_fit()
        # The fit runs on a worker thread; block (with a live event loop) until
        # it lands so the screenshot captures the converged parameter table
        # rather than the transient "Fitting…" state. The wait is bounded, so a
        # stalled fit cannot wedge the capture indefinitely.
        single_tab.wait_for_fit()
        return window


register(FitWizardGktScenario())
