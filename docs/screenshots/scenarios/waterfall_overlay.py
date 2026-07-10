"""Waterfall-stacked time-domain overlay of the Ag LF decoupling series.

Reuses the same five-field LF Kubo–Toyabe field-decoupling series as
``lf_kt_series_plot.py`` (B_L = 0, 5, 10, 25, 50 G against Δ=0.39 μs⁻¹), but
with **Waterfall** turned on alongside **Overlay** so each field's trace is
shifted vertically by ``i * Δ`` (automatic spacing) rather than piled on the
same baseline. The two scenarios are meant to be read side by side: the plain
overlay shows the decoupling progression collapsing onto a shared axis, the
waterfall view shows the same five traces cleanly resolved with their own
zero baselines.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from ..data import make_ag_lf_decoupling
from ._base import Scenario, register


class WaterfallOverlayScenario(Scenario):
    name = "waterfall_overlay"
    description = "Waterfall-stacked time-domain overlay of an Ag LF decoupling series."
    size = (1500, 920)

    def build(self) -> QWidget:
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window.resizeDocks(
            [window._dock_data_browser], [320], Qt.Orientation.Horizontal
        )

        datasets = make_ag_lf_decoupling()
        # Per-add table rebuilds are O(n²); batch_updates defers to one rebuild.
        with window._data_browser.batch_updates():
            for dataset in datasets:
                window._data_browser.add_dataset(dataset)

        run_numbers = [int(ds.run_number) for ds in datasets]
        window._data_browser.create_data_group(
            run_numbers, name="LF decoupling — Ag"
        )

        # Enable Overlay, then Waterfall (auto spacing) before selecting, so
        # all five LF runs render stacked rather than piled on one baseline.
        window._plot_panel.set_overlay_enabled(True, emit_signal=True)
        window._plot_panel.set_waterfall_enabled(True, emit_signal=True)
        window._data_browser._table.selectAll()
        window._on_dataset_selected(run_numbers[0])
        return window


register(WaterfallOverlayScenario())
