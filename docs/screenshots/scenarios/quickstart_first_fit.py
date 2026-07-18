"""Quickstart's first converged fit: ZF Ag Gaussian Kubo–Toyabe.

Companion screenshot to :doc:`/getting_started/quickstart`. The quickstart
walkthrough generates the **Ag — ZF Gaussian Kubo–Toyabe** simulate preset,
opens the Fit Wizard, and applies its recommendation — ``StaticGKT_ZF +
Constant`` — back into the single-fit tab, recovering the field width
:math:`\\Delta \\approx 0.39` μs⁻¹ the preset was built from. This scenario
mirrors that exact run and model so the image and the page's prose agree: it
loads the same Ag ZF static Gaussian Kubo–Toyabe archetype
(:func:`docs.screenshots.data.make_ag_zf_gkt`, Δ=0.39 μs⁻¹, Kubo & Toyabe
1966; Blundell et al. Ch 5.2, eq 5.13), configures the single-fit tab with the
recommended ``StaticGKT_ZF + Constant`` composite, and runs the fit so the
capture shows the converged parameter table beside the fit overlaid on the
data — the reassuring "data in, model fitted, parameter out" end state.

Marked ``requires_fit = True`` because the iminuit-based fit trips on
numpy ≥ 2.3 in dev environments; CI keeps numpy < 2.3 via constraints.txt.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from ..data import make_ag_zf_gkt
from ._base import Scenario, _process_events_for, register


class QuickstartFirstFitScenario(Scenario):
    name = "quickstart_first_fit"
    description = (
        "Quickstart's first fit: converged StaticGKT_ZF + Constant on the ZF Ag preset."
    )
    size = (1500, 920)
    requires_fit = True

    def build(self) -> QWidget:
        from asymmetry.core.fitting.composite import CompositeModel
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        # Surface the fit dock so the full WiMDA-style layout is visible.
        window._on_fit()
        window.resizeDocks(
            [window._dock_data_browser], [320], Qt.Orientation.Horizontal
        )

        dataset = make_ag_zf_gkt()
        window._data_browser.add_dataset(dataset)
        window._on_dataset_selected(dataset.run_number)

        # The Fit Wizard recommends and applies ``StaticGKT_ZF + Constant`` into
        # the single-fit tab; configuring the same composite here reproduces the
        # exact model the walkthrough ends on. Defaults for A0/Delta/baseline
        # sit close enough to the synthetic truth that the fit converges in one
        # call without further tweaking.
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


register(QuickstartFirstFitScenario())
