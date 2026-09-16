"""The Batch tab after a second series is recorded on one data group.

The selector's own popup menu (``GlobalFitTab._show_series_menu``) is a
``QMenu.exec()`` call: a real top-level popup, not a child widget the capture
driver's ``widget.grab()`` would ever composite in, and it blocks the event
loop for a click nothing offscreen can supply. So instead of the open menu,
this scenario captures what choosing an entry from it leads to: two series
recorded on the same data group, with the second (the more recent) left open.
The selector then reads a *recorded* series rather than a draft, and the
Series section's status tag reads ``Fitted n/n · HH:MM`` instead of
``Draft``/``Edited …`` — the two states the existing
``batch_tab_group_binding`` scenario does not show.

Both series are recorded through the real handlers
(``MainWindow._on_fit_group_requested``, ``_on_global_fit_completed`` with a
synthetic per-run ``FitResult``), exactly as
``tests/gui/test_batch_series_editor.py`` drives them. The two runs share the
group's four members and the tab's default composite model; only the fit
range differs between them, which is enough of a recipe difference for the
second run to record a *new* series (D3) rather than replacing the first, so
the group ends up owning two. ``MainWindow._open_series_in_batch_tab`` — the
same call the menu's own entries make — then reopens the newer one, so the
results card replays its recorded outcome exactly as choosing it from the
menu would.
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


class BatchTabSeriesMenuScenario(Scenario):
    name = "batch_tab_series_menu"
    description = (
        "The Batch tab with a recorded series open, after a second was recorded on its group."
    )
    size = (1500, 1000)

    def build(self) -> QWidget:
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window._on_fit()
        window._dock_log.hide()
        window.resizeDocks([window._dock_data_browser], [300], Qt.Orientation.Horizontal)

        datasets = make_euo_tf_tscan()[:4]  # 30, 50, 65, 69 K
        for dataset in datasets:
            window._data_browser.add_dataset(dataset)
        run_numbers = [int(ds.run_number) for ds in datasets]

        gid = window._data_browser.create_data_group(run_numbers, name="T scan — EuO")
        window._on_fit_group_requested(gid)

        global_tab = window._fit_panel._global_tab
        window._fit_panel._tabs.setCurrentWidget(global_tab)

        # First series: the draft's inherited (project-wide) fit range.
        window._on_global_fit_completed(
            {run: (_result(0.30), _CURVE, []) for run in run_numbers}, ParameterSet()
        )

        # Narrowing the window is a recipe change, so this second run records
        # a *second* series on the same group rather than replacing the first
        # (D3) — the group now owns two, and this one is left open.
        global_tab._fit_range_min_spin.setValue(0.5)
        global_tab._fit_range_max_spin.setValue(5.0)
        window._on_global_fit_completed(
            {run: (_result(0.42), _CURVE, []) for run in run_numbers}, ParameterSet()
        )
        newer_batch_id = window._fit_panel.open_series_id()

        window._on_dataset_selected(run_numbers[0])
        # Recording a fit raises the Parameters dock (item 3) to show the
        # trend; reopen the newer series exactly as its menu entry would —
        # the real route to a *recorded* card, not a "results replaced"
        # notice — and raise the Fit dock's Batch tab back to show it.
        window._open_series_in_batch_tab(newer_batch_id)
        window._dock_fit.raise_()
        window._fit_panel._tabs.setCurrentWidget(global_tab)
        return window


register(BatchTabSeriesMenuScenario())
