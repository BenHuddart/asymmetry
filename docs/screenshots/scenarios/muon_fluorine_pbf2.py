"""Muon-fluorine entanglement signal in PbF₂.

PbF₂ provides a clean F-μ-F demonstration: the heavy Pb host carries no
significant nuclear moment so the polarization is dominated by the
analytical F-μ-F dipolar pattern (Brewer et al. PRB 33, 7813, 1986;
textbook Ch 4.6). The synthetic dataset carries a λ=0.3 μs⁻¹ damping
envelope on the beat pattern (literature-plausible for F-μ-F powders, see
``make_pbf2_fmuf``'s docstring) so the trace reads as physically realistic
rather than an undamped analytical curve. The scenario loads the dataset,
switches the fit panel to the wizard-recommended
``FmuF_Linear * Exponential + Constant`` composite, and runs the real fit
so the converged parameters and fit overlay are visible over the first
10 μs, where the beats (and the damping) are clearly resolved.

Marked ``requires_fit = True`` because the underlying iminuit-based fit
trips on numpy ≥ 2.3 in dev environments; CI keeps numpy < 2.3.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from ..data import make_pbf2_fmuf
from ._base import Scenario, _process_events_for, register


class MuonFluorinePbf2Scenario(Scenario):
    name = "muon_fluorine_pbf2"
    description = (
        "Converged FmuF_Linear*Exponential+Constant fit on a damped PbF₂ F-μ-F dataset."
    )
    size = (1500, 920)
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
        window.resizeDocks(
            [window._dock_data_browser], [320], Qt.Orientation.Horizontal
        )

        dataset = make_pbf2_fmuf(relaxation_lambda_per_us=0.3)
        window._data_browser.add_dataset(dataset)
        window._on_dataset_selected(dataset.run_number)

        single_tab = window._fit_panel._single_tab
        single_tab._set_composite_model(
            CompositeModel(["FmuF_Linear", "Exponential", "Constant"], operators=["*", "+"])
        )
        _process_events_for(milliseconds=80)

        # Seed initial values close to the generator's truth (A_1=22,
        # r_muF=1.17 Å, Lambda=0.3 us^-1, A_bg=0.2) so the fit converges in a
        # single call rather than risking a wrong local minimum in the
        # multi-line F-mu-F beat pattern.
        param_table = single_tab._param_table
        rows_by_name = _param_table_rows_by_name(param_table)
        seeds = {
            "A_1": 22.0,
            "r_muF": 1.17,
            "Lambda": 0.3,
            "A_bg": 0.2,
        }
        for name, value in seeds.items():
            if name in rows_by_name:
                _set_param_table_value(param_table, rows_by_name[name], value)
        _process_events_for(milliseconds=40)

        single_tab._run_fit()
        # The fit runs on a worker thread; block (with a live event loop)
        # until it lands so the screenshot captures the converged parameters
        # rather than the transient "Fitting…" state (mirrors
        # the corpus euo_ordering.py scenarios). The wait is bounded, so a stalled fit
        # cannot wedge the capture indefinitely.
        single_tab.wait_for_fit()

        # t_max=20 us was chosen so the full beat envelope is visible, but
        # with the added lambda=0.3 damping the signal has decayed into flat
        # noise past ~10 us; narrow the view through the real X-range
        # toolbar fields so the resolved beats and their decay fill the plot
        # instead of a long flat tail.
        _x_min, _x_max, y_min, y_max = window._plot_panel.get_view_limits()
        window._plot_panel.set_view_limits(0.0, 10.0, y_min, y_max)
        _process_events_for(milliseconds=80)
        return window


register(MuonFluorinePbf2Scenario())
