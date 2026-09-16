"""A run in two series, with both fits shown on the "Fits on this run" strip.

Forms a data group over four runs of the EuO ZF temperature scan and drives
two real global-fit completions over it through
``MainWindow._on_fit_group_requested`` and ``_on_global_fit_completed`` with a
synthetic per-run ``FitResult`` (as ``tests/gui/test_batch_series_editor.py``
does). The second completion narrows the fit range first, which is a recipe
change, so it records a *second* series over the same members rather than
replacing the first (D3) — every run in the group, including the one selected
here, now carries two fits. Both are switched on for that run with the public
``PlotPanel.set_shown_fits`` (the same store a pill click writes to), so the
"Fits on this run" strip shows two checked pills above the canvas — one is
the active series (drawn in the fit accent colour), the other a trace colour.
See *Reference > GUI usage > Fitting a group directly* and *Reference >
Parameter trending > Group-bound series and staleness*.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.parameters import Parameter, ParameterSet

from ..data import make_euo_tf_tscan
from ._base import Scenario, register


def _result(lambda_value: float) -> FitResult:
    return FitResult(
        success=True,
        chi_squared=48.0,
        reduced_chi_squared=0.97,
        parameters=ParameterSet([Parameter("A", 22.0), Parameter("Lambda", lambda_value)]),
        uncertainties={"A": 0.4, "Lambda": 0.02},
    )


_CURVE = ([0.0, 3.0, 6.0], [22.0, 8.0, 2.0])


class PlotFitsOnRunScenario(Scenario):
    name = "plot_fits_on_run"
    description = 'A run in two series, with both fits shown on the "Fits on this run" strip.'
    size = (1500, 920)

    def build(self) -> QWidget:
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window._dock_log.hide()
        window.resizeDocks([window._dock_data_browser], [300], Qt.Orientation.Horizontal)

        datasets = make_euo_tf_tscan()[:4]  # 30, 50, 65, 69 K
        for dataset in datasets:
            window._data_browser.add_dataset(dataset)
        run_numbers = [int(ds.run_number) for ds in datasets]

        gid = window._data_browser.create_data_group(run_numbers, name="T scan — EuO")
        window._on_fit_group_requested(gid)

        # First series: the draft's inherited (project-wide) fit range.
        window._on_global_fit_completed(
            {run: (_result(0.30), _CURVE, []) for run in run_numbers}, ParameterSet()
        )
        series_a_id = window._fit_panel.open_series_id()

        # A narrower window is a recipe change, so this run records a second
        # series over the same members (D3) rather than replacing the first.
        global_tab = window._fit_panel._global_tab
        global_tab._fit_range_min_spin.setValue(0.5)
        global_tab._fit_range_max_spin.setValue(5.0)
        window._on_global_fit_completed(
            {run: (_result(0.42), _CURVE, []) for run in run_numbers}, ParameterSet()
        )
        series_b_id = window._fit_panel.open_series_id()

        window._on_dataset_selected(run_numbers[0])
        window._plot_panel.set_shown_fits(run_numbers[0], [series_a_id, series_b_id])
        return window


register(PlotFitsOnRunScenario())
