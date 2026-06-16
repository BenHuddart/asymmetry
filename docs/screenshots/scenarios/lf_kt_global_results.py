"""Global fit results for the Ag LF Kubo–Toyabe decoupling series.

.. note:: This scenario is **temporarily excluded** from the capture pipeline
   (not imported in ``docs/screenshots/capture._import_scenarios``).  Re-enable
   it — along with ``global_fit_lfkt`` — once the global fit wizard feature has
   been further developed and its CI run time is acceptable.  When re-enabling,
   also verify the 4-dataset global fit completes well within the
   ``_CAPTURE_TIMEOUT_S`` budget set in ``capture.py``.


Companion screenshot to
:doc:`/user_guide/workflows/lf_decoupling_dynamics`. The setup is the same
as ``global_fit_lfkt`` (four-field Ag LF series, ``LongitudinalFieldKT +
Constant`` model, Δ shared globally, B_L fixed per run), but here the fit
is actually run synchronously and the screenshot captures the converged
state: the parameter table shows the fitted global Δ with its Hessian
uncertainty, the result text reports the average reduced χ², and the
central plot overlays the per-run fit curves on the data.

Marked ``requires_fit = True`` because the iminuit fit trips on
numpy ≥ 2.3 in dev environments; CI keeps numpy < 2.3.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from ..data import make_ag_lf_decoupling
from ._base import Scenario, register, _process_events_for


class LfKtGlobalResultsScenario(Scenario):
    name = "lf_kt_global_results"
    description = (
        "Global fit completed on the Ag LF-KT decoupling series, "
        "showing the fitted global Δ with Hessian uncertainty."
    )
    size = (1500, 920)
    requires_fit = True

    def build(self) -> QWidget:
        from asymmetry.core.fitting.composite import CompositeModel
        from asymmetry.core.fitting.engine import FitEngine
        from asymmetry.core.fitting.parameters import Parameter, ParameterSet
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window._on_fit()
        window.resizeDocks(
            [window._dock_data_browser], [340], Qt.Orientation.Horizontal
        )

        datasets = make_ag_lf_decoupling(fields_g=(0.0, 15.0, 50.0, 100.0))
        for dataset in datasets:
            window._data_browser.add_dataset(dataset)

        run_numbers = [int(ds.run_number) for ds in datasets]
        window._data_browser.create_data_group(
            run_numbers, name="LF decoupling — Ag"
        )

        # Configure the global-fit tab and set the model.
        fit_panel = window._fit_panel
        fit_panel._tabs.setCurrentWidget(fit_panel._global_tab)
        fit_panel.set_datasets(datasets)
        global_tab = fit_panel._global_tab
        model = CompositeModel(["LongitudinalFieldKT", "Constant"])
        global_tab._set_composite_model(model)
        _process_events_for(milliseconds=80)

        # Build the initial parameter sets directly and run the fit synchronously
        # to keep the screenshot deterministic. The Global tab's worker runs in a
        # QThread; for capture we bypass it and feed results into the same
        # success-render path the worker would have used.
        engine = FitEngine()
        initial_params: dict[int, ParameterSet] = {}
        for ds in datasets:
            initial_params[int(ds.run_number)] = ParameterSet([
                Parameter("A_1", value=24.0, min=0.0, max=40.0),
                Parameter("Delta", value=0.4, min=0.0, max=2.0),
                Parameter(
                    "B_L", value=float(ds.metadata.get("field", 0.0)), fixed=True
                ),
                Parameter("A_bg", value=0.3, min=-1.0, max=2.0),
            ])

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
            model=model,
            results_dict=results_dict,
            fitted_global=fitted_global,
            global_param_names=global_params,
        )
        _process_events_for(milliseconds=120)

        # Select the lowest-field run so the central plot focuses on the most
        # depolarised trace and the fit overlay reads clearly.
        window._on_dataset_selected(run_numbers[0])
        _process_events_for(milliseconds=40)
        return window


register(LfKtGlobalResultsScenario())
