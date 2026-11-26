"""Photo-μSR in silicon on the two-period HiFi runs from the Rigi school."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from docs.workflows.silicon_lifetime import (
    CALIBRATION_MODEL,
    CALIBRATION_RUNS,
    DECAY_MODEL,
    DELAY_RUNS,
    MAX_DELAY_US,
    REFERENCE_DENSITY_CM3,
    fit_calibration,
    fit_carrier_decay,
)

from ._corpus import CorpusScenario, _process_events_for, corpus_path, register

EXAMPLE = "Semiconductors/Photo-muSR in silicon"
_DATA = EXAMPLE + "/Data"
RATE_FIT_END_US = 1.0


def _rel(run: int) -> str:
    return str(corpus_path(f"{_DATA}/HIFI00{run}.nxs"))


def _fit_exponential(dataset, amplitude: float | None = None):
    """Fit the first microsecond with a fixed amplitude when one is supplied."""
    from asymmetry.core.fitting.composite import CompositeModel
    from asymmetry.core.fitting.engine import FitEngine
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    params = ParameterSet(
        [
            Parameter(
                "A_1",
                15.0 if amplitude is None else amplitude,
                min=0.0,
                fixed=amplitude is not None,
            ),
            Parameter("Lambda", 0.3, min=0.0),
        ]
    )
    model = CompositeModel(["Exponential"])
    result = FitEngine().fit(dataset, model.function, params, t_min=0.0, t_max=RATE_FIT_END_US)
    if not result.success:
        raise RuntimeError(f"Run {dataset.run_number} exponential fit failed ({result.message})")
    return result


def light_on_rate(run: int) -> tuple[float, float]:
    """Return the illuminated rate and its error conditional on the dark amplitude."""
    from asymmetry.core.io import load, select_period

    combined = load(_rel(run))
    dark = _fit_exponential(select_period(combined, "green"))
    light = _fit_exponential(select_period(combined, "red"), amplitude=dark.parameters["A_1"].value)
    if light.covariance is None:
        raise RuntimeError(f"Run {run} illuminated rate uncertainty is unavailable")
    return light.parameters["Lambda"].value, light.uncertainties["Lambda"]


def _register_derived_labels() -> None:
    """Use dimensionless calibration density and scaled physical decay density."""
    from asymmetry.core.fitting.parameters import register_derived_param_info

    register_derived_param_info(
        "DnRatio",
        plain="Dn/Dn0",
        unicode="Δn/Δn₀",
        latex=r"$\Delta n/\Delta n_0$",
        gle=r"\Delta n/\Delta n_0",
        unit="",
    )
    register_derived_param_info(
        "Dn",
        plain="Dn",
        unicode="Δn",
        latex=r"$\Delta n$",
        gle=r"\Delta n",
        unit="10¹³ cm⁻³",
    )
    register_derived_param_info(
        "dT",
        plain="dT",
        unicode="ΔT",
        latex=r"$\Delta T$",
        gle=r"\Delta T",
        unit="µs",
    )


class SiGroupingScenario(CorpusScenario):
    name = "corpus_si_grouping_rg"
    description = "The Grouping window's RG Mode choices on two-period silicon run 103277."
    example = EXAMPLE
    size = (1220, 760)

    def build(self) -> QWidget:
        from asymmetry.core.io import load
        from asymmetry.gui.windows.grouping.dialog import GroupingDialog

        run = load(_rel(103277))
        return GroupingDialog([run], selected_run_number=int(run.run_number))

    def settle(self, widget: QWidget) -> None:
        widget._grouping_scroll.ensureWidgetVisible(widget._period_row_widget)
        preview = widget._preview_pane
        _wait_until(
            lambda: preview._last_result is not None and preview.corrections_complete,
            "silicon grouping preview",
        )

    def teardown(self, widget: QWidget) -> None:
        widget._teardown_workers()
        widget.deleteLater()
        _process_events_for(milliseconds=40)


class SiOnOffOverlayScenario(CorpusScenario):
    name = "corpus_si_on_off_overlay"
    description = (
        "Light-ON (Red) vs light-OFF (Green) asymmetry of silicon photo-µSR "
        "run 103277 overlaid: the laser-induced extra muon-spin relaxation "
        "(carrier depolarisation) is plainly visible against the near-flat "
        "light-OFF spectrum."
    )
    example = EXAMPLE
    size = (1400, 860)

    def build(self) -> QWidget:
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window.resizeDocks([window._dock_data_browser], [320], Qt.Orientation.Horizontal)

        on_ds, off_ds = _labelled_on_off(window, 103277)
        run_numbers = [int(on_ds.run_number), int(off_ds.run_number)]
        window._data_browser.create_data_group(run_numbers, name="Si 103277 — laser ON / OFF")

        window._on_dataset_selected(run_numbers[0])
        window._plot_panel.set_overlay_enabled(True, emit_signal=True)
        window._data_browser._table.selectAll()
        _process_events_for(milliseconds=120)
        # Frame the first ~6 µs, where the light-ON depolarisation runs its
        # course; past there both spectra are flat and the counts thin out.
        window._plot_panel.set_view_limits(0.0, 6.0, -2.0, 18.0)
        _process_events_for(milliseconds=120)
        return window


# --------------------------------------------------------------------------- #
# 3. Single-exponential λ fit on the light-ON period of run 103277
# --------------------------------------------------------------------------- #
class SiLambdaFitScenario(CorpusScenario):
    name = "corpus_si_lambda_fit"
    description = (
        "Single-exponential fit on the light-ON (Red) period of the highest-Δn "
        "silicon run 103277 over the first 1 µs, amplitude A₀ fixed at the "
        "light-OFF value → λ ≈ 1.27 µs⁻¹."
    )
    example = EXAMPLE
    size = (1500, 920)
    requires_fit = True

    def build(self) -> QWidget:
        from dataclasses import replace

        from asymmetry.core.fitting.composite import CompositeModel
        from asymmetry.core.fitting.seeding import SeedContext, seed_parameters
        from asymmetry.core.io import load
        from asymmetry.core.io.periods import select_period
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window._on_fit()
        window.resizeDocks([window._dock_data_browser], [300], Qt.Orientation.Horizontal)

        combined = load(_rel(103277))
        # A₀ from the light-OFF (Green) period, per the guide workflow.
        dark = _fit_exponential(select_period(combined, "green"))
        a0 = dark.parameters["A_1"].value

        on_ds = select_period(combined, "red")
        on_ds.run.run_number = 1032770 + 1  # derived id (light-ON period)
        on_ds.metadata["run_label"] = "103277 laser ON"
        window._data_browser.add_dataset(on_ds)
        window._on_dataset_selected(on_ds.run_number)
        _process_events_for(milliseconds=80)

        # Restrict the fit to the first 1 µs (GROUND_TRUTH §4). The plot panel is
        # the canonical fit-range owner; set_fit_range slices the dataset handed
        # to the fit tab (a spinbox setValue would not commit the range).
        window._plot_panel.set_fit_range(0.0, RATE_FIT_END_US)
        _process_events_for(milliseconds=60)

        single_tab = window._fit_panel._single_tab
        model = CompositeModel(["Exponential"])
        single_tab._set_composite_model(model)
        _process_events_for(milliseconds=60)
        # Re-populate with A₀ fixed at the light-OFF amplitude and λ seeded near
        # the expected value; the guide holds A₀ constant for the light-ON fit.
        seeds = seed_parameters(model, SeedContext())
        single_tab._param_table.populate(
            model,
            seeds={
                "A_1": replace(seeds["A_1"], value=a0, fixed=True),
                "Lambda": replace(seeds["Lambda"], value=0.9),
            },
        )
        _process_events_for(milliseconds=40)

        single_tab._run_fit()
        single_tab.wait_for_fit()
        _process_events_for(milliseconds=80)

        # Show the decay and the first bit of the flat tail: 0–2.5 µs.
        window._plot_panel.set_view_limits(0.0, 2.5, -2.0, 18.0)
        _process_events_for(milliseconds=80)
        return window


class SiLambdaVsDnScenario(CorpusScenario):
    name = "corpus_si_lambda_vs_dn"
    description = "Weighted zero-offset calibration of light-on λ versus Δn/Δn₀."
    example = EXAMPLE
    size = (1240, 800)
    requires_fit = True

    def build(self) -> QWidget:
        density = np.array(list(CALIBRATION_RUNS.values()))
        rate, error = np.array([light_on_rate(run) for run in CALIBRATION_RUNS]).T
        calibration = fit_calibration(density, rate, error)
        a, n, c = (calibration.parameters[name].value for name in ("a", "n", "c"))
        density_ratio = density / REFERENCE_DENSITY_CM3
        return _trend_panel(
            "Calibration",
            CALIBRATION_RUNS,
            "DnRatio",
            density_ratio,
            "Lambda",
            rate,
            error,
            CALIBRATION_MODEL,
            calibration,
            (float(density_ratio.min()), float(density_ratio.max())),
            annotation=(0.14, 1.22, f"a = {a:.3f} µs⁻¹, n = {n:.3f}, c = {c:g}"),
            log_axes=True,
        )

    def settle(self, widget: QWidget) -> None:
        _settle_trend(widget)


class SiTauDecayScenario(CorpusScenario):
    name = "corpus_si_tau_decay"
    description = "Carrier densities from the displayed calibration and their early-delay fit."
    example = EXAMPLE
    size = (1240, 800)
    requires_fit = True

    def build(self) -> QWidget:
        density = np.array(list(CALIBRATION_RUNS.values()))
        cal_rate, cal_error = np.array([light_on_rate(run) for run in CALIBRATION_RUNS]).T
        calibration = fit_calibration(density, cal_rate, cal_error)
        delay = np.array(list(DELAY_RUNS.values()))
        rate, error = np.array([light_on_rate(run) for run in DELAY_RUNS]).T
        decay = fit_carrier_decay(calibration, delay, rate, error)
        return _trend_panel(
            "Carrier decay",
            DELAY_RUNS,
            "dT",
            delay,
            "Dn",
            decay.density,
            decay.error,
            DECAY_MODEL,
            decay.fit,
            (float(delay.min()), MAX_DELAY_US),
            annotation=(31.0, 1.15, "Extrapolated densities from 25 µs"),
        )

    def settle(self, widget: QWidget) -> None:
        _settle_trend(widget)


def _trend_panel(
    title, runs, x_name, x, y_name, y, error, model, result, limits, *, annotation, log_axes=False
):
    """Show the fitted trend and conditional point errors on their physical axes."""
    from asymmetry.core.fitting.parameter_models import ModelFitRange, ParameterModelFit
    from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel

    _register_derived_labels()
    rows = [
        {
            "run_number": run,
            "run_label": str(run),
            "field": -100.0,
            "temperature": 291.0,
            "values": {x_name: float(x_value), y_name: float(y_value)},
            "errors": {y_name: float(y_error)},
        }
        for run, x_value, y_value, y_error in zip(runs, x, y, error, strict=True)
    ]
    panel = FitParametersPanel()
    panel.load_representation_series([("silicon", title, rows)], select_id="silicon")
    _process_events_for(milliseconds=60)

    x_key = f"param:{x_name}"
    panel._x_combo.setCurrentIndex(panel._x_combo.findData(x_key))
    for name, chip in panel._y_chips.items():
        chip.setChecked(name == y_name)
    _process_events_for(milliseconds=30)

    model_range = ModelFitRange(
        x_min=limits[0],
        x_max=limits[1],
        model=model,
        parameters=result.parameters,
        result=result,
    )
    panel._model_fits[y_name] = ParameterModelFit(
        parameter_name=y_name, x_key=x_key, ranges=[model_range], active=True
    )
    if log_axes:
        panel._log_x_check.setChecked(True)
        panel._card_stack.card(y_name).log_check.setChecked(True)
    label_x, label_y, label_text = annotation
    panel._plot_annotations.append(
        {"x": label_x, "y": label_y, "text": label_text, "axis_tag": y_name, "artist": None}
    )
    panel._sync_active_group_state()
    panel._refresh_model_fit_button_labels()
    _process_events_for(milliseconds=80)
    return panel


def _labelled_on_off(window, run: int):
    """Give both periods display names and independent browser rows."""
    from asymmetry.core.io import load, select_period

    combined = load(_rel(run))
    on_ds = select_period(combined, "red")
    off_ds = select_period(combined, "green")
    for dataset, state in ((on_ds, "ON"), (off_ds, "OFF")):
        dataset.run.run_number = window._data_browser.next_derived_run_number()
        dataset.metadata["run_label"] = f"{run} laser {state}"
        window._data_browser.add_dataset(dataset)
    return on_ds, off_ds


def _settle_trend(widget: QWidget) -> None:
    _process_events_for(milliseconds=200)
    widget._refresh_plot()
    _wait_until(
        lambda: (
            not widget._trend_curve_compute_active and widget._precomputed_trend_curves is not None
        ),
        "silicon trend curve",
    )
    _process_events_for(milliseconds=200)


def _wait_until(predicate, description: str) -> None:
    for _ in range(200):
        if predicate():
            return
        _process_events_for(milliseconds=50)
    raise RuntimeError(f"Timed out waiting for {description}")


register(SiGroupingScenario())
register(SiOnOffOverlayScenario())
register(SiLambdaFitScenario())
register(SiLambdaVsDnScenario())
register(SiTauDecayScenario())
