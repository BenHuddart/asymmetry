"""Converged single fit with MINOS asymmetric errors shown in the table.

Companion screenshot to :doc:`/reference/fit_workflow_diagnostics` §
"Asymmetric (MINOS) errors". Reuses the converged
``Oscillatory * Exponential + Constant`` fit on the lowest-temperature EuO
run (T=30 K, well inside the ordered phase), but ticks the **Asymmetric
errors** checkbox before fitting so the backend walks the χ² profile of each
free parameter and reports the asymmetric ±1σ (MINOS) interval. With MINOS
present the parameter table renders each fitted value inline as
``value +σ₊ / −σ₋`` in place of ``± σ`` — exactly the display the page
describes.

Marked ``requires_fit = True`` because the underlying iminuit-based fit (and
the MINOS profile scan) trips on numpy ≥ 2.3 in dev environments; CI keeps
numpy < 2.3.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from ..data import make_euo_tf_tscan
from ._base import Scenario, _process_events_for, register


class FitAsymmetricErrorsScenario(Scenario):
    name = "fit_asymmetric_errors"
    description = (
        "Converged EuO T=30 K fit with the Asymmetric errors (MINOS) checkbox "
        "ticked and asymmetric ±1σ intervals shown inline in the parameter table."
    )
    size = (1820, 920)
    requires_fit = True

    def build(self) -> QWidget:
        from asymmetry.core.fitting.composite import CompositeModel
        from asymmetry.gui.mainwindow import MainWindow
        from asymmetry.gui.panels.fit.tab_base import (
            _param_table_rows_by_name,
            _set_param_table_value,
        )

        window = MainWindow()
        window._on_fit()

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

        # Tick "Asymmetric errors" so the fit runs MINOS after the minimisation
        # and the parameter table shows each value as "value +σ₊ / −σ₋".
        single_tab._minos_checkbox.setChecked(True)
        _process_events_for(milliseconds=20)

        single_tab._run_fit()
        # The fit (plus the MINOS profile scan) runs on a worker thread; block
        # with a live event loop until it lands so the screenshot captures the
        # converged asymmetric errors rather than the transient "Fitting…"
        # state. The wait is bounded, so a stalled fit cannot wedge the capture.
        single_tab.wait_for_fit()

        # The parameter table's Value column is sized (12 chars) for the compact
        # "value ± σ" form; the asymmetric "value +σ₊ / −σ₋" overlay the delegate
        # paints for MINOS results is wider, so widen the column and the fit dock
        # so the whole interval is legible — this page is about that display.
        single_tab._param_table.setColumnWidth(single_tab._param_table.COL_VALUE, 300)
        # A hard minimum-width floor on the fit dock is honoured even against the
        # plot's expanding central widget (a plain resizeDocks request is not, so
        # the dock otherwise stays at its ~368 px minimum and clips the interval).
        window._dock_fit.setMinimumWidth(560)
        window.resizeDocks([window._dock_data_browser], [300], Qt.Orientation.Horizontal)
        _process_events_for(milliseconds=40)

        # ν(30 K) ≈ 22.3 MHz gives a ~0.045 µs period; zoom to the first 0.5 µs
        # (~11 cycles) through the real X-range toolbar fields so individual
        # oscillations and the fit overlay are both resolved.
        _x_min, _x_max, y_min, y_max = window._plot_panel.get_view_limits()
        window._plot_panel.set_view_limits(0.0, 0.5, y_min, y_max)
        _process_events_for(milliseconds=80)
        return window


register(FitAsymmetricErrorsScenario())
