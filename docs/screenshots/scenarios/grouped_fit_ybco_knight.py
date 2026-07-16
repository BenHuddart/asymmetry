"""Grouped TF Knight-shift fitting on normal-state YBa₂Cu₃O₇₋δ.

Loads a YBCO TF run synthesised with four detector histograms + a
grouping payload that puts one detector per group. The scenario then:

1. Selects the run.
2. Switches the central plot workspace to the **Individual Groups**
   domain so the four per-group asymmetries are visible side by side, and
   zooms the time axis to ~1.5 µs so the ~2.72 MHz Larmor oscillations
   resolve individually instead of rendering as a near-solid band.
3. Opens the Fit dock, which auto-engages the **MultiGroupFitWindow**
   (the dedicated count-domain grouped-fit surface), selects the
   ``OscillatoryField * Exponential`` composite matching the generator's
   exp(-0.08 t)-damped precession, and runs the real grouped time-domain
   fit synchronously (the per-group N₀/background/amplitude/phase
   nuisances and the shared field already auto-seed from the run's own
   counts and applied field, so a single fit call converges with
   average χ²ᵣ ≈ 1.0).

The per-group N₀, amplitude, baseline, and relative phase fit as local
nuisance parameters while the Larmor frequency and damping are shared —
the canonical workflow for extracting the muon Knight shift in the
normal state of a superconductor (Sonier RMP 72, 769, 2000).

Marked ``requires_fit = True`` because it runs the real grouped time-domain
fit (see :mod:`asymmetry.core.fitting.grouped_time_domain`), which trips on
numpy ≥ 2.3 in dev environments; CI keeps numpy < 2.3.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from ..data import make_ybco_knight_grouped
from ._base import Scenario, _process_events_for, register


class GroupedFitYbcoKnightScenario(Scenario):
    name = "grouped_fit_ybco_knight"
    description = (
        "MultiGroupFitWindow on YBCO TF above Tc, Individual Groups domain, 4 detector groups."
    )
    size = (1500, 920)
    requires_fit = True

    def build(self) -> QWidget:
        from asymmetry.core.fitting.composite import CompositeModel
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

        # The Larmor frequency is ~2.72 MHz (200 G x 1.005 Knight shift), a
        # ~0.37 µs period; the default 12 µs view compresses ~30 cycles into
        # each subplot and renders the oscillations as a near-solid band.
        # Zoom to ~1.5 µs (~4 cycles) through the real X-range toolbar fields.
        _x_min, _x_max, y_min, y_max = window._plot_panel.get_view_limits()
        window._plot_panel.set_view_limits(0.0, 1.5, y_min, y_max)
        _process_events_for(milliseconds=80)

        # The synthetic per-detector signals carry an exp(-0.08 t) damping
        # envelope (see make_ybco_knight_grouped), so the panel's default
        # bare OscillatoryField model would systematically undershoot the
        # early-time peaks — and with N0 ~ 1e6 counts per group even that
        # small envelope mismatch inflates chi^2 enormously. Select the
        # matching OscillatoryField * Exponential composite through the real
        # model pathway (the multiplicative chain shares one amplitude,
        # which the grouped surface hides in favour of the per-group
        # amplitude nuisance).
        single_fit_tab = window._multi_group_fit_window._single_fit_tab
        single_fit_tab._set_composite_model(
            CompositeModel(["OscillatoryField", "Exponential"], operators=["*"])
        )
        _process_events_for(milliseconds=80)

        # Run the real grouped time-domain fit synchronously (worker thread,
        # blocked on with a live event loop so the capture is deterministic —
        # mirrors euo_fit_oscillatory.py). The per-group nuisances and shared
        # field parameter already auto-seed from the run's own counts/applied
        # field when the table was built above, so a single fit call
        # converges and the parameter table reads as a completed workflow
        # instead of pre-fit defaults.
        single_fit_tab._run_global_fit()
        single_fit_tab.wait_for_fit()
        _process_events_for(milliseconds=80)
        return window


register(GroupedFitYbcoKnightScenario())
