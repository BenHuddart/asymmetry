"""A run in two series, with both fits drawn and the toolbar reading "Fits · 2".

Forms a data group over four runs of the EuO ZF temperature scan and runs
two real, synchronous batch fits over it (``GlobalFitTab._run_global_fit`` +
``wait_for_fit``, as ``grouped_fit_ybco_knight``/``fit_asymmetric_errors`` run
theirs) rather than handing ``_on_global_fit_completed`` a synthetic result —
a docs screenshot of fits must show curves the real engine actually drew,
not a fixed array stamped onto every series. A bare ``Exponential +
Constant`` cannot describe the precessing signal below Tc, so the model is
``Oscillatory * Exponential + Constant`` instead (the composite
``fit_asymmetric_errors``/``grouped_fit_ybco_knight`` already use for this
same archetype), with each run's frequency and damping seeded from the
known generator (``make_euo_tf_tscan``'s own ν(T) and damping(T) — the
per-run "Per-run seeds…" store, ``GlobalFitTab._user_initial_values_by_run``,
set directly rather than driven through the dialog) so all four converge
tightly (χ²ᵣ ≈ 1) instead of one run failing outright from a generic guess.

The fit range is set explicitly before each run (``0.5–5 µs``, then
``0–6 µs``): a recipe change, so the second run records a *second* series
over the same members (D3) rather than replacing the first. Because a fitted
curve is drawn only over its own literal fit range (not extrapolated across
the whole axis), the two series' curves end up visibly distinct even where
the physics agrees closely: the narrower series' curve stops at t=0.5 and
t=5, the wider one spans the full 0–6 canvas. Both are switched on for the
displayed run with the public ``PlotPanel.set_shown_fits`` (the same store a
menu tick writes to), so the plot toolbar's Fits button reads "Fits · 2" in
its more-than-one-fit accent and the legend names both. The popup itself is
not captured: a QMenu is its own top-level window, which the offscreen
platform never renders into the grabbed window, so the shot shows the
button and both overlays instead.

The Parameters dock is widened past its layout-negotiated default (the same
``setMinimumWidth`` + ``resizeDocks`` pairing ``fit_asymmetric_errors`` uses
for the Fit dock) so both series' chips read on one row under the group's
section header rather than wrapping or eliding past recognition.
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


class PlotFitsOnRunScenario(Scenario):
    name = "plot_fits_on_run"
    description = 'A run in two series, with both fits drawn and the toolbar reading "Fits · 2".'
    size = (1500, 920)
    requires_fit = True

    def build(self) -> QWidget:
        from asymmetry.core.fitting.composite import CompositeModel
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window._dock_log.hide()
        # A hard minimum-width floor on the Parameters dock is honoured even
        # against the plot's expanding central widget (a plain resizeDocks
        # request is not), so both series' chips fit on one row.
        window._dock_fit_parameters.setMinimumWidth(560)
        window.resizeDocks([window._dock_data_browser], [300], Qt.Orientation.Horizontal)

        datasets = make_euo_tf_tscan()[:4]  # 30, 50, 65, 69 K
        for dataset in datasets:
            window._data_browser.add_dataset(dataset)
        run_numbers = [int(ds.run_number) for ds in datasets]

        gid = window._data_browser.create_data_group(run_numbers, name="T scan — EuO")
        window._on_fit_group_requested(gid)

        global_tab = window._fit_panel._global_tab
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
        series_a_id = window._fit_panel.open_series_id()

        # Second series: widening back to the full 0-6 us window is a recipe
        # change from the first series (D3), so this real fit records a
        # *second* series over the same members rather than replacing it.
        global_tab._fit_range_min_spin.setValue(0.0)
        global_tab._fit_range_max_spin.setValue(6.0)
        _process_events_for(milliseconds=40)
        global_tab._run_global_fit()
        global_tab.wait_for_fit()
        _process_events_for(milliseconds=80)
        series_b_id = window._fit_panel.open_series_id()

        for batch_id in (series_a_id, series_b_id):
            series = window._project_model.batch(batch_id)
            fit_range = series.recipe["fit_range"]
            assert fit_range["min"] is not None and fit_range["max"] is not None, (
                f"series {batch_id} recorded an unbounded fit range: {fit_range}"
            )

        window._on_dataset_selected(run_numbers[0])
        window._plot_panel.set_shown_fits(run_numbers[0], [series_a_id, series_b_id])
        _process_events_for(milliseconds=80)

        fits_button = window._plot_panel._fits_button
        assert fits_button.text() == "Fits · 2", (
            f"the toolbar's Fits button reads {fits_button.text()!r}, not the two fits this "
            "run carries"
        )

        # ν(30 K) ≈ 22.3 MHz gives a ~0.045 µs period; the default 6 µs view
        # compresses ~130 cycles into the canvas and renders both fitted
        # curves as a solid band. Zoom to ~0.8 µs (~18 cycles, real X-range
        # toolbar fields, as fit_asymmetric_errors does) so the oscillations
        # resolve and, crucially, so the gap before t=0.5 µs shows plainly:
        # only the wider (0-6 µs) series' curve is drawn there, since a
        # fitted curve is never extrapolated past its own recorded range.
        _x_min, _x_max, y_min, y_max = window._plot_panel.get_view_limits()
        window._plot_panel.set_view_limits(0.0, 0.8, y_min, y_max)
        _process_events_for(milliseconds=80)
        return window


register(PlotFitsOnRunScenario())
