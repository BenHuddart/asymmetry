"""Capture the Al-LLZ calibration, three-field Keren fits, and Arrhenius trend."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.workflow.calibration import CalibrationOptions, calibrate_run
from asymmetry.core.workflow.reduction import (
    ReductionSettings,
    reduce_run,
    resolve_reduction_grouping,
)

from .._base import CaptureContext
from ._corpus import CorpusScenario, _process_events_for, load_corpus_datasets, register

EXAMPLE = "Nuclear magnetism and ionic motion/Ionic motion in a solid electrolyte"
_DATA = EXAMPLE + "/Data"
CALIBRATION_RUN = 51315
CALIBRATION_METHOD = "diamagnetic"
FIRST_TEMPERATURE = 160

# ZF run number that opens each temperature's triplet; triplet = (zf, zf+1, zf+2)
# = (0 G, 5 G, 10 G).  Setpoint temperatures from GROUND_TRUTH.md §3.
TRIPLETS: dict[int, int] = {
    160: 51341,
    180: 51344,
    200: 51347,
    220: 51350,
    240: 51353,
    264: 51356,
    284: 51359,
    304: 51362,
    324: 51365,
    344: 51368,
    364: 51371,
    384: 51374,
    404: 51377,
}

# Seed, limits, and Batch classification for each time-domain parameter.
FIT_PARAMETERS = {
    "A_1": (15.0, 0.0, 40.0, "Global"),
    "Delta": (0.3, 0.0, 2.0, "Global"),
    "nu": (0.2, 0.0, 8.0, "Global"),
    "B_L": (0.0, -float("inf"), float("inf"), "File"),
    "A_bg": (5.0, -10.0, 20.0, "Global"),
}
FIT_TMAX = 12.0  # µs
ACTIVATED_MIN_T = 264.0  # K, nominal setpoint


def _calibration_dataset() -> MuonDataset:
    return load_corpus_datasets([f"{_DATA}/emu000{CALIBRATION_RUN}.nxs"])[0]


def _llz_reduction() -> ReductionSettings:
    outcome = calibrate_run(
        _calibration_dataset(), ReductionSettings(), CalibrationOptions(CALIBRATION_METHOD)
    )
    return ReductionSettings(alpha=outcome.result)


def _load_triplet(temperature: int, settings: ReductionSettings) -> list[MuonDataset]:
    first_run = TRIPLETS[temperature]
    datasets = load_corpus_datasets(
        [f"{_DATA}/emu000{run}.nxs" for run in range(first_run, first_run + 3)]
    )
    reduced = []
    for dataset in datasets:
        run = dataset.run
        grouped = replace(run, grouping=resolve_reduction_grouping(run, settings))
        reduced.append(replace(reduce_run(run, settings), run=grouped))
    return reduced


def _keren_model():
    from asymmetry.core.fitting.composite import CompositeModel

    return CompositeModel(["Keren", "Constant"])


class LlzCalibrationScenario(CorpusScenario):
    """TF 20 G calibration run 51315 — estimate α before the science fits."""

    name = "corpus_llz_calibration"
    description = (
        "Alpha calibration on the Al-LLZ TF 20 G run 51315: the data-prep step "
        "that fixes the detector balance α before the LF triplet fits."
    )
    example = EXAMPLE
    size = (1220, 760)

    def capture(self, ctx: CaptureContext) -> Path:  # noqa: D401
        # The quick result updates the draft coefficient and its preview.
        from asymmetry.gui.windows.grouping.dialog import GroupingDialog

        dataset = _calibration_dataset()
        dialog = GroupingDialog([dataset], selected_run_number=int(dataset.run_number))
        dialog.resize(*self.size)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        dialog.show()
        section = dialog._alpha_section
        section._method_combo.setCurrentIndex(section._method_combo.findData(CALIBRATION_METHOD))
        section._on_estimate()
        preview = dialog._preview_pane
        _wait_until(
            lambda: section._tasks.active_count == 0 and preview.corrections_complete,
            "calibrated grouping preview",
        )
        dialog._corrections_scroll.ensureWidgetVisible(section)
        _process_events_for(milliseconds=100)

        pix = dialog.grab()
        out_path = ctx.output_dir / f"{self.name}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not pix.save(str(out_path), "PNG"):
            raise RuntimeError(f"Failed to save screenshot to {out_path}")
        # Release workers without entering a headless discard prompt.
        dialog._teardown_workers()
        dialog.deleteLater()
        _process_events_for(milliseconds=40)
        return out_path


class LlzLfTripletScenario(CorpusScenario):
    """Raw 0/5/10 G overlay at 160 K — the weak-LF-decoupling signature."""

    name = "corpus_llz_lf_triplet"
    description = (
        "Al-LLZ 160 K longitudinal-field triplet (0/5/10 G) overlaid: the raw "
        "signature of weak LF decoupling of the muon from Li nuclear fields."
    )
    example = EXAMPLE
    size = (1400, 880)

    def build(self) -> QWidget:
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window.resizeDocks([window._dock_data_browser], [300], Qt.Orientation.Horizontal)
        window._dock_fit.hide()
        window._dock_fit_parameters.hide()
        window._dock_log.hide()

        datasets = _load_triplet(FIRST_TEMPERATURE, _llz_reduction())
        self.add_to_browser(window, datasets)

        run_numbers = [int(ds.run_number) for ds in datasets]
        window._data_browser.create_data_group(run_numbers, name="Al-LLZ 160 K (0/5/10 G)")

        window._plot_panel.set_overlay_enabled(True, emit_signal=True)
        window._plot_panel.set_bunch_factor(6, emit_signal=True)
        window._data_browser._table.selectAll()
        window._on_dataset_selected(run_numbers[0])
        _process_events_for(milliseconds=120)
        # Clip to the 12 µs analysis window: past ~13 µs the F–B asymmetry
        # diverges as the counts vanish and swamps the decoupling signature.
        window._plot_panel.set_view_limits(0.0, FIT_TMAX, 0.0, 22.0)
        _process_events_for(milliseconds=120)
        return window


class LlzGlobalSetupScenario(CorpusScenario):
    """Batch/global-fit panel with the triplet loaded and parameters tied."""

    name = "corpus_llz_global_setup"
    description = (
        "Batch setup on the Al-LLZ 160 K triplet with Keren + Constant, "
        "Δ, ν and amplitudes set to Global, and the measured B_L set to File."
    )
    example = EXAMPLE
    size = (1680, 1000)

    def build(self) -> QWidget:
        return _triplet_fit_window(self)

    def settle(self, widget: QWidget) -> None:
        _settle_triplet_fit_window(widget)


class LlzGlobalResultScenario(CorpusScenario):
    """Converged simultaneous Keren fit of the triplet."""

    name = "corpus_llz_global_result"
    description = "Batch fit of the three fields at 160 K, showing the curve on the 0 G run."
    example = EXAMPLE
    size = (1680, 1000)
    requires_fit = True

    def build(self) -> QWidget:
        window = _triplet_fit_window(self)
        global_tab = window._fit_panel._global_tab
        fitted_runs = []
        global_tab.global_fit_completed.connect(lambda results, _: fitted_runs.extend(results))
        global_tab._fit_btn.click()
        _wait_until(lambda: global_tab._fit_worker is None, "160 K Batch fit")

        if len(fitted_runs) != 3:
            raise RuntimeError(f"Expected all three fitted fields at 160 K, got {fitted_runs!r}")
        window._dock_fit.raise_()
        return window

    def settle(self, widget: QWidget) -> None:
        _settle_triplet_fit_window(widget)


class LlzNuArrheniusScenario(CorpusScenario):
    """One fitted rate per triplet, with a linear fit to the activated branch."""

    name = "corpus_llz_nu_arrhenius"
    description = (
        "Fit Parameters trend of ln(ν − plateau) against reciprocal setpoint "
        "temperature, with a Linear fit to the eight setpoints from 264 to 404 K."
    )
    example = EXAMPLE
    size = (1240, 780)
    requires_fit = True

    def build(self) -> QWidget:
        import numpy as np

        from asymmetry.core.fitting.axis_transforms import AxisTransform
        from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel

        temps, nu, nu_err = _fit_nu_of_t()
        branch = temps >= ACTIVATED_MIN_T
        baseline = float(np.mean(nu[~branch]))
        y_transform = AxisTransform.custom(f"log(x - {baseline:.6g})")
        x_transform = AxisTransform.preset("reciprocal")
        fit = _build_arrhenius_line_fit(
            temps[branch], nu[branch], nu_err[branch], x_transform, y_transform
        )

        batch_id = "llz-nu-t"
        row_dicts = [
            {
                "run_number": TRIPLETS[int(t)],
                "run_label": f"{int(t)} K",
                "field": 0.0,
                "temperature": float(t),
                "values": {"nu": float(nu[i])},
                "errors": {"nu": float(nu_err[i])},
                "include_in_trend": bool(branch[i]),
            }
            for i, t in enumerate(temps)
        ]

        panel = FitParametersPanel()
        panel.load_representation_series(
            [(batch_id, "ν(T) — Al-LLZ", row_dicts)], select_id=batch_id
        )
        panel._x_combo.setCurrentText("𝑇 (K)")

        panel._set_y_transform("nu", y_transform)
        panel._set_x_transform(x_transform)
        panel._axis_transform_custom_memory["y:nu"] = y_transform.expression

        panel._model_fits["nu"] = fit
        panel._model_fit_transform_sig["nu"] = panel._transform_signature("nu")
        panel._sync_active_group_state()
        panel._refresh_model_fit_button_labels()
        _process_events_for(milliseconds=80)
        return panel

    def settle(self, widget: QWidget) -> None:
        widget._refresh_plot()
        _wait_until(
            lambda: widget._precomputed_trend_curves is not None,
            "Arrhenius trend curve",
        )


def _triplet_fit_window(scenario: CorpusScenario) -> QWidget:
    from asymmetry.gui.mainwindow import MainWindow

    window = MainWindow()
    window._on_fit()
    window.resizeDocks([window._dock_data_browser], [280], Qt.Orientation.Horizontal)
    datasets = _load_triplet(FIRST_TEMPERATURE, _llz_reduction())
    scenario.add_to_browser(window, datasets)
    run_numbers = [int(ds.run_number) for ds in datasets]
    window._data_browser.create_data_group(run_numbers, name="Al-LLZ 160 K (0/5/10 G)")
    window._data_browser._table.selectAll()
    window._on_dataset_selected(run_numbers[0])

    fit_panel = window._fit_panel
    global_tab = fit_panel._global_tab
    fit_panel._tabs.setCurrentWidget(global_tab)
    fit_panel.set_batch_fit_range(0.0, FIT_TMAX)
    fit_panel.set_datasets(window._selected_time_fit_datasets())
    global_tab.set_current_dataset(datasets[0])
    global_tab._set_composite_model(_keren_model())

    table = global_tab._param_table
    for row in range(table.rowCount()):
        name = table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        seed, lower, upper, role = FIT_PARAMETERS[name]
        if role != "File":
            table.item(row, 1).setText(f"{seed:g}")
            table.item(row, 3).setText(f"{lower:g}")
            table.item(row, 4).setText(f"{upper:g}")
        table.cellWidget(row, 2).setCurrentText(role)

    window._plot_panel.set_view_limits(0.0, FIT_TMAX, 0.0, 22.0)
    return window


def _settle_triplet_fit_window(window: QWidget) -> None:
    window._dock_log.hide()
    window.resizeDocks([window._dock_fit], [565], Qt.Orientation.Horizontal)
    _process_events_for(milliseconds=100)


def _fit_nu_of_t():
    """Fit each triplet, carrying the shared seeds up in temperature."""
    import numpy as np

    from asymmetry.core.fitting.engine import FitEngine
    from asymmetry.core.fitting.recipe_inputs import build_recipe_engine_inputs

    model = _keren_model()
    engine = FitEngine()
    reduction = _llz_reduction()
    seeds = {name: row[0] for name, row in FIT_PARAMETERS.items()}
    points = []
    for temperature, first_run in sorted(TRIPLETS.items()):
        datasets = _load_triplet(temperature, reduction)
        recipe = {
            "fit_range": {"min": 0.0, "max": FIT_TMAX},
            "parameters": [
                {
                    "name": name,
                    "value": seeds[name],
                    "bounds": f"{lower:g}, {upper:g}",
                    "type": role,
                }
                for name, (_, lower, upper, role) in FIT_PARAMETERS.items()
            ],
        }
        prepared = build_recipe_engine_inputs(recipe, model, datasets)
        results, shared = engine.global_fit(
            datasets,
            model.function,
            prepared.global_params,
            prepared.local_params,
            prepared.initial_params,
            t_min=prepared.t_min,
            t_max=prepared.t_max,
        )
        if len(results) != 3 or not all(result.success for result in results.values()):
            raise RuntimeError(f"Expected all three fitted fields at {temperature} K")
        seeds.update({parameter.name: parameter.value for parameter in shared})
        error = results[first_run].uncertainties.get("nu")
        if error is None or not np.isfinite(error) or error <= 0.0:
            raise RuntimeError(f"Expected a finite positive rate uncertainty at {temperature} K")
        points.append((temperature, shared["nu"].value, error))
    return np.asarray(points, dtype=float).T


def _build_arrhenius_line_fit(temperature, rate, rate_error, x_transform, y_transform):
    """Fit the selected temperatures in the same coordinates as their plot."""
    import numpy as np

    from asymmetry.core.fitting.parameter_models import (
        ModelFitRange,
        ParameterCompositeModel,
        ParameterModelFit,
        fit_parameter_model,
    )
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    x, _ = x_transform.apply(temperature)
    y, error = y_transform.apply(rate, rate_error)
    if not np.all(np.isfinite(y) & np.isfinite(error) & (error > 0.0)):
        raise RuntimeError("The selected Arrhenius branch requires finite log rates and errors")
    x_min, x_max = float(x.min()), float(x.max())
    model = ParameterCompositeModel(["Linear"])
    seed = ParameterSet(
        [
            Parameter(name="m", value=-2500.0, min=-1.0e6, max=1.0e6),  # −E_a/k_B  [K]
            Parameter(name="b", value=5.0, min=-50.0, max=50.0),
        ]
    )
    result = fit_parameter_model(x, y, error, model, seed, x_min=x_min, x_max=x_max)
    if not result.success:
        raise RuntimeError(f"Al-LLZ Arrhenius fit failed: {result.message}")

    fit_range = ModelFitRange(
        x_min=x_min, x_max=x_max, model=model, parameters=result.parameters, result=result
    )
    return ParameterModelFit(
        parameter_name="nu", x_key="temperature", ranges=[fit_range], active=True
    )


def _wait_until(predicate, description: str) -> None:
    for _ in range(200):
        if predicate():
            return
        _process_events_for(milliseconds=50)
    raise RuntimeError(f"Timed out waiting for {description}")


register(LlzCalibrationScenario())
register(LlzLfTripletScenario())
register(LlzGlobalSetupScenario())
register(LlzGlobalResultScenario())
register(LlzNuArrheniusScenario())
