"""Member readout from a global Ag LF-KT fit, with shared Δ and its uncertainty.

The classification is authored as Global for amplitude, Δ, and background,
and File for each run's fixed applied field. The captured Results window
shows the actual fitted values and Hessian errors, rather than seed values.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from ..data import make_ag_lf_decoupling
from ._base import Scenario, _process_events_for, register


class LfKtGlobalResultsScenario(Scenario):
    name = "lf_kt_global_results"
    description = "Member results from a global Ag LF-KT fit, including shared Δ and Hessian error."
    size = (950, 290)
    requires_fit = True

    def build(self) -> QWidget:
        from asymmetry.core.fitting.composite import CompositeModel
        from asymmetry.core.fitting.engine import FitEngine
        from asymmetry.core.fitting.parameters import Parameter, ParameterSet
        from asymmetry.gui.mainwindow import MainWindow
        from asymmetry.gui.panels.fit.global_tab import FitLaunch

        window = MainWindow()
        window._on_fit()
        window.resizeDocks([window._dock_data_browser], [340], Qt.Orientation.Horizontal)

        datasets = make_ag_lf_decoupling(fields_g=(0.0, 15.0, 50.0, 100.0))
        for dataset in datasets:
            window._data_browser.add_dataset(dataset)

        run_numbers = [int(ds.run_number) for ds in datasets]
        window._data_browser.create_data_group(run_numbers, name="LF decoupling — Ag")

        # Configure the global-fit tab and set the model.
        fit_panel = window._fit_panel
        fit_panel._tabs.setCurrentWidget(fit_panel._global_tab)
        fit_panel.set_datasets(datasets)
        global_tab = fit_panel._global_tab
        model = CompositeModel(["LongitudinalFieldKT", "Constant"])
        global_tab._set_composite_model(
            model,
            seed_values={"A_1": "24", "Delta": "0.4", "B_L": "0", "A_bg": "0.3"},
        )
        for row, name in enumerate(model.param_names):
            global_tab._param_table.cellWidget(row, 2).setCurrentText(
                "File" if name == "B_L" else "Global"
            )
        _process_events_for(milliseconds=80)

        # Build the initial parameter sets directly and run the fit synchronously
        # to keep the screenshot deterministic. The Global tab's worker runs in a
        # QThread; for capture we bypass it and feed results into the same
        # success-render path the worker would have used.
        engine = FitEngine()
        initial_params: dict[int, ParameterSet] = {}
        for ds in datasets:
            initial_params[int(ds.run_number)] = ParameterSet(
                [
                    Parameter("A_1", value=24.0, min=0.0, max=40.0),
                    Parameter("Delta", value=0.4, min=0.0, max=2.0),
                    Parameter("B_L", value=float(ds.metadata.get("field", 0.0)), fixed=True),
                    Parameter("A_bg", value=0.3, min=-1.0, max=2.0),
                ]
            )

        global_params = ["A_1", "Delta", "A_bg"]
        local_params = ["B_L"]
        results_dict, fitted_global = engine.global_fit(
            datasets=datasets,
            model_fn=model.function,
            global_params=global_params,
            local_params=local_params,
            initial_params=initial_params,
        )
        global_tab._emit_global_fit_success(
            launch=FitLaunch(
                model=model,
                global_params=tuple(global_params),
                datasets=tuple(datasets),
            ),
            results_dict=results_dict,
            successful=results_dict,
            fitted_global=fitted_global,
            detail_html=global_tab._batch_detail_html(
                results_dict, results_dict, {}, global_params, ""
            ),
        )
        _process_events_for(milliseconds=120)

        # The Seed table is recipe input; the member readout owns fitted values/errors.
        window._on_dataset_selected(run_numbers[0])
        global_tab._show_member_results_window(run_numbers[0])
        _process_events_for(milliseconds=40)
        return global_tab._member_windows[run_numbers[0]]


register(LfKtGlobalResultsScenario())
