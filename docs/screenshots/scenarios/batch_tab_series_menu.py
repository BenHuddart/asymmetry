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

Both series are recorded from two real, synchronous batch fits over the same
four-run EuO group (``GlobalFitTab._run_global_fit`` + ``wait_for_fit``, as
``grouped_fit_ybco_knight``/``fit_asymmetric_errors`` run theirs) rather than
a synthetic result handed to ``_on_global_fit_completed`` — a docs screenshot
of a fit must show what the real engine draws. A bare ``Exponential +
Constant`` cannot describe the precessing signal below Tc (it degenerates to
a near-instant decay for every run regardless of window), so the model is
``Oscillatory * Exponential + Constant`` instead — the composite
``fit_asymmetric_errors``/``grouped_fit_ybco_knight`` already use for this
same archetype — with each run's frequency and damping seeded from the known
generator (``make_euo_tf_tscan``'s own ν(T) and damping(T), written directly
into the "Per-run seeds…" store, ``GlobalFitTab._user_initial_values_by_run``,
rather than driven through the dialog) so all four converge tightly
(χ²ᵣ ≈ 1) instead of aliasing to the wrong harmonic from one generic guess.

The fit range is set explicitly before each run (``0.5–5 µs``, then
``0–6 µs``): both are recipe differences from each other, so the second run
records a *new* series (D3) rather than replacing the first, and both carry
a genuine bounded window rather than the unbounded "as fitted" default a
never-set range would leave. ``MainWindow._open_series_in_batch_tab`` — the
same call the menu's own entries make — then reopens the newer one, so the
results card replays its recorded outcome exactly as choosing it from the
menu would.

The Fit dock is widened past its layout-negotiated default (the same
``setMinimumWidth`` + ``resizeDocks`` pairing ``fit_asymmetric_errors`` uses)
so the parameter table's Type column reads in full rather than eliding to
"Loc…".
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from ..data import make_euo_tf_tscan
from ._base import Scenario, _process_events_for, register

#: Per-run seeds for Oscillatory * Exponential + Constant, computed from
#: make_euo_tf_tscan's own generator (nu_0=28 MHz, beta=0.4, Tc=69 K; damping
#: floor=0.10, peak=4.0, width=6 K) for the 30/50/65/69 K runs (3001-3004) —
#: close enough to the injected truth that all four batch members converge
#: cleanly rather than one aliasing to the wrong harmonic from a generic seed.
_SEEDS_BY_RUN = {
    3001: {"A_1": 22.0, "frequency": 22.28, "phase": 0.0, "Lambda": 0.10, "A_bg": 0.4},
    3002: {"A_1": 22.0, "frequency": 16.71, "phase": 0.0, "Lambda": 0.10, "A_bg": 0.4},
    3003: {"A_1": 22.0, "frequency": 8.96, "phase": 0.0, "Lambda": 2.66, "A_bg": 0.4},
    3004: {"A_1": 22.0, "frequency": 0.01, "phase": 0.0, "Lambda": 4.10, "A_bg": 0.4},
}


class BatchTabSeriesMenuScenario(Scenario):
    name = "batch_tab_series_menu"
    description = (
        "The Batch tab with a recorded series open, after a second was recorded on its group."
    )
    size = (1500, 1000)
    requires_fit = True

    def build(self) -> QWidget:
        from asymmetry.core.fitting.composite import CompositeModel
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window._on_fit()
        window._dock_log.hide()
        # A hard minimum-width floor on the fit dock is honoured even against
        # the plot's expanding central widget (a plain resizeDocks request is
        # not), so the Type column of the parameter table stays legible.
        window._dock_fit.setMinimumWidth(560)
        window.resizeDocks([window._dock_data_browser], [300], Qt.Orientation.Horizontal)

        datasets = make_euo_tf_tscan()[:4]  # 30, 50, 65, 69 K
        for dataset in datasets:
            window._data_browser.add_dataset(dataset)
        run_numbers = [int(ds.run_number) for ds in datasets]

        gid = window._data_browser.create_data_group(run_numbers, name="T scan — EuO")
        window._on_fit_group_requested(gid)

        global_tab = window._fit_panel._global_tab
        window._fit_panel._tabs.setCurrentWidget(global_tab)
        _process_events_for(milliseconds=80)

        global_tab._set_composite_model(
            CompositeModel(["Oscillatory", "Exponential", "Constant"], operators=["*", "+"])
        )
        _process_events_for(milliseconds=80)
        global_tab._user_initial_values_by_run = dict(_SEEDS_BY_RUN)
        global_tab.set_batch_seeding_mode("as_provided")
        _process_events_for(milliseconds=40)

        # First series: a real batch fit over 0.5-5 us.
        global_tab._fit_range_min_spin.setValue(0.5)
        global_tab._fit_range_max_spin.setValue(5.0)
        _process_events_for(milliseconds=40)
        global_tab._run_global_fit()
        global_tab.wait_for_fit()
        _process_events_for(milliseconds=80)
        first_batch_id = window._fit_panel.open_series_id()

        # Second series: widening back to the full 0-6 us window is a recipe
        # change from the first series (D3), so this real fit records a
        # *second* series on the same group rather than replacing the first.
        global_tab._fit_range_min_spin.setValue(0.0)
        global_tab._fit_range_max_spin.setValue(6.0)
        _process_events_for(milliseconds=40)
        global_tab._run_global_fit()
        global_tab.wait_for_fit()
        _process_events_for(milliseconds=80)
        newer_batch_id = window._fit_panel.open_series_id()

        for batch_id in (first_batch_id, newer_batch_id):
            series = window._project_model.batch(batch_id)
            fit_range = series.recipe["fit_range"]
            assert fit_range["min"] is not None and fit_range["max"] is not None, (
                f"series {batch_id} recorded an unbounded fit range: {fit_range}"
            )

        window._on_dataset_selected(run_numbers[0])
        # Recording a fit raises the Parameters dock (item 3) to show the
        # trend; reopen the newer series exactly as its menu entry would —
        # the real route to a *recorded* card, not a "results replaced"
        # notice — and raise the Fit dock's Batch tab back to show it.
        window._open_series_in_batch_tab(newer_batch_id)
        window._dock_fit.raise_()
        window._fit_panel._tabs.setCurrentWidget(global_tab)
        _process_events_for(milliseconds=80)
        return window


register(BatchTabSeriesMenuScenario())
