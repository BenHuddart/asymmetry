"""Converged single fit on the EuO T=30 K run.

Companion screenshot to
:doc:`/user_guide/workflows/temperature_scan_magnetism`. Selects the lowest-
temperature EuO run (T=30 K, well inside the ordered phase) and fits the
``Oscillatory * Exponential + Constant`` composite that the workflow text
recommends for the below-Tc regime. The screenshot captures the fit panel
with the converged parameters and uncertainties, plus the central plot
showing the fit overlaid on the data.

Marked ``requires_fit = True`` because the underlying iminuit-based fit
trips on numpy ≥ 2.3 in dev environments; CI keeps numpy < 2.3.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from ..data import make_euo_tf_tscan
from ._base import Scenario, register, _process_events_for


class EuoFitOscillatoryScenario(Scenario):
    name = "euo_fit_oscillatory"
    description = (
        "Converged Oscillatory*Exponential+Constant fit on the EuO T=30 K ZF run."
    )
    size = (1500, 920)
    requires_fit = True

    def build(self) -> QWidget:
        from asymmetry.core.fitting.composite import CompositeModel
        from asymmetry.gui.mainwindow import MainWindow
        from asymmetry.gui.panels.fit_panel import (
            _param_table_rows_by_name,
            _set_param_table_value,
        )

        window = MainWindow()
        window._on_fit()
        window.resizeDocks(
            [window._dock_data_browser], [340], Qt.Orientation.Horizontal
        )

        datasets = make_euo_tf_tscan()
        for dataset in datasets:
            window._data_browser.add_dataset(dataset)

        # T=30 K is the first run (run number 3001).
        target_run = datasets[0].run_number
        window._on_dataset_selected(target_run)

        single_tab = window._fit_panel._single_tab
        single_tab._set_composite_model(
            CompositeModel(
                ["Oscillatory", "Exponential", "Constant"],
                operators=["*", "+"],
            )
        )
        _process_events_for(milliseconds=80)

        # Seed initial values close to the expected ν(30 K) ≈ 22.3 MHz so the
        # fit converges in a single call without hitting a wrong local minimum.
        # The composite shares one amplitude A_1 across the multiplicative
        # Oscillatory*Exponential chain; A_bg is the Constant background.
        param_table = single_tab._param_table
        rows_by_name = _param_table_rows_by_name(param_table)
        seeds = {
            "A_1": 22.0,
            "frequency": 22.0,
            "phase": 0.0,
            "Lambda": 0.2,
            "A_bg": 0.4,
        }
        for name, value in seeds.items():
            if name in rows_by_name:
                _set_param_table_value(param_table, rows_by_name[name], value)
        _process_events_for(milliseconds=40)

        single_tab._run_fit()
        # The fit runs on a worker thread; block (with a live event loop) until
        # it lands so the screenshot captures the converged parameters rather
        # than the transient "Fitting…" state. The wait is bounded, so a stalled
        # fit cannot wedge the capture indefinitely.
        single_tab.wait_for_fit()
        return window


register(EuoFitOscillatoryScenario())
