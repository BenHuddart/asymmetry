"""Grouped TF Knight-shift fitting on normal-state YBa₂Cu₃O₇₋δ.

Loads a YBCO TF run synthesised with four detector histograms + a
grouping payload that puts one detector per group. The scenario then:

1. Selects the run.
2. Switches the central plot workspace to the **Individual Groups**
   domain so the four per-group asymmetries are visible side by side.
3. Opens the Fit dock, which auto-engages the **MultiGroupFitWindow**
   (the dedicated count-domain grouped-fit surface).

The per-group N₀, amplitude, baseline, and relative phase fit as local
nuisance parameters while the Larmor frequency and damping are shared —
the canonical workflow for extracting the muon Knight shift in the
normal state of a superconductor (Sonier RMP 72, 769, 2000).
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from ..data import make_ybco_knight_grouped
from ._base import Scenario, register, _process_events_for


class GroupedFitYbcoKnightScenario(Scenario):
    name = "grouped_fit_ybco_knight"
    description = (
        "MultiGroupFitWindow on YBCO TF above Tc, Individual Groups domain, 4 detector groups."
    )
    size = (1500, 920)

    def build(self) -> QWidget:
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window.resizeDocks(
            [window._dock_data_browser], [340], Qt.Orientation.Horizontal
        )

        dataset = make_ybco_knight_grouped()
        window._data_browser.add_dataset(dataset)
        window._on_dataset_selected(dataset.run_number)
        _process_events_for(milliseconds=120)

        # _refresh_time_view_selector() decides whether the "groups" view is
        # available for the active dataset (requires a Run with grouping). It
        # runs automatically on selection but we call it explicitly here so
        # the assertions below catch any synthesis-side regression.
        window._refresh_time_view_selector()
        assert "groups" in window._plot_workspace.enabled_views(), (
            "Individual Groups view not enabled — check make_ybco_knight_grouped "
            "synthesises a Run with at least two detector groups."
        )

        # Switch to the Individual Groups domain so the four per-group
        # asymmetries are shown in the central plot. The fit dock then
        # auto-engages the MultiGroupFitWindow via _sync_fit_dock_mode().
        window._on_domain_button_clicked("groups")
        _process_events_for(milliseconds=120)
        window._on_fit()
        _process_events_for(milliseconds=160)
        return window


register(GroupedFitYbcoKnightScenario())
