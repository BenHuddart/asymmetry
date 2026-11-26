"""Corpus scenarios — A molecular antiferromagnet (Magnetism).

Drives the Asymmetry GUI through the WiMDA muon-school **molecular
antiferromagnet** example on the real **ISIS/MUSR NeXus-v1 (HDF4)** corpus
files (``MUSR00017094.nxs`` … ``MUSR00017104.nxs``, runs 17094–17104, April
2008). The teaching guide is the spec (``GROUND_TRUTH.md``); no reference paper
or logbook is present, so every number is graded against the **[M] measured**
values in GROUND_TRUTH §3a / §6.

The 20 G transverse-field run supplies the detector balance for the
zero-field runs. The ordered spectra resolve spontaneous precession from
1.2 to 6 K. Comparing runs 17099 and 17100 brackets the disappearance of
resolved precession between 6 and 7 K.

Scenarios registered:

* ``corpus_molafm_alpha``     — α / balance calibration on the 20 G TF run
  17104 (10 K, paramagnetic): the guide's "which data set for α?" data-prep
  step; the Estimate balances F/B and returns α ≈ 1.25.
* ``corpus_molafm_zf_fit``    — the money shot: a converged damped-oscillation
  fit (Oscillatory×Exponential+Constant) on the 1.2 K ZF run 17094,
  ν ≈ 1.55 MHz, zoomed so the slow precession resolves.
* ``corpus_molafm_nu_t``      — headline: real per-run ZF fits → ν(T) with the
  fitted OrderParameter curve and the 6–7 K transition bracket.
* ``corpus_molafm_zf_overlay`` — 6 K vs 7 K ZF spectra overlaid: the coherent
  oscillation present at 6 K (17099) is gone at 7 K (17100), fixing T_N ≈ 6–7 K
  directly from the data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from asymmetry.gui.styles import tokens

from .._base import _process_events_for
from ._corpus import CorpusScenario, load_corpus_datasets, register

EXAMPLE = "Magnetism/A molecular antiferromagnet"
_DATA = "Magnetism/A molecular antiferromagnet/Data/MUSR000%d.nxs"

_ZF_MODEL = (["Oscillatory", "Exponential", "Constant"], ["*", "+"])
_ZF_FIT_RANGE_US = (0.11, 8.0)

# The resolved precession branch spans 1.2–6 K.
_ZF_OP_RUNS: list[tuple[int, float]] = [
    (17094, 1.2),
    (17095, 2.0),
    (17096, 3.0),
    (17097, 4.0),
    (17098, 5.0),
    (17099, 6.0),
]

# The money-shot run: base temperature, cleanest low-frequency oscillation.
_ZF_FIT_RUN = 17094

# The α / balance run: 20 G transverse field at 10 K (paramagnetic, above T_N),
# where the full TF precession amplitude calibrates the F/B balance
# (GROUND_TRUTH §4/§5 Q1).
_ALPHA_RUN = 17104

# T_N bracket: the last oscillating run (6 K) vs the first paramagnetic run
# (7 K) — the direct-comparison overlay that fixes T_N ≈ 6–7 K.
_OVERLAY_RUNS = (17099, 17100)  # (6 K ordered, 7 K paramagnetic)


def _process(ms: int = 80) -> None:
    _process_events_for(milliseconds=ms)


def _molecular_balance():
    """Estimate the detector balance on the 20 G TF run, retaining its source."""
    from asymmetry.core.workflow.calibration import CalibrationOptions, calibrate_run
    from asymmetry.core.workflow.reduction import ReductionSettings

    dataset = load_corpus_datasets([_DATA % _ALPHA_RUN])[0]
    outcome = calibrate_run(dataset, ReductionSettings(), CalibrationOptions(method="diamagnetic"))
    if outcome.alpha is None:
        raise RuntimeError(f"Molecular-magnet TF calibration failed: {outcome.message}")
    return outcome.result


def _fit_zf_frequency(dataset, nu_seed: float):
    """Fit damped precession over the same time window as the Single tab."""
    from asymmetry.core.fitting.composite import CompositeModel
    from asymmetry.core.fitting.engine import FitEngine
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    model = CompositeModel(_ZF_MODEL[0], operators=_ZF_MODEL[1])
    t_min, t_max = _ZF_FIT_RANGE_US
    tail = (dataset.time > 3.0) & (dataset.time < t_max)
    params = ParameterSet(
        [
            Parameter("A_1", 4.0, min=0.0),
            Parameter("frequency", nu_seed, min=0.3, max=2.5),
            Parameter("phase", 0.0),
            Parameter("Lambda", 0.2, min=0.0),
            Parameter("A_bg", float(np.nanmedian(dataset.asymmetry[tail]))),
        ]
    )
    return FitEngine().fit(dataset, model.function, params, t_min=t_min, t_max=t_max)


def _wait_until(predicate, *, timeout_ms: int, poll_ms: int = 30) -> None:
    elapsed = 0
    while elapsed < timeout_ms:
        if predicate():
            return
        _process_events_for(milliseconds=poll_ms)
        elapsed += poll_ms


# ---------------------------------------------------------------------------
# 1. α / balance calibration on the 20 G TF run (data-prep)
# ---------------------------------------------------------------------------
class MolAfmAlphaScenario(CorpusScenario):
    name = "corpus_molafm_alpha"
    description = (
        "Alpha calibration on the 20 G transverse-field run MUSR00017104 "
        "(10 K, paramagnetic, above T_N): the guide's 'which data set for α?' "
        "step — Estimate balances the forward/backward asymmetry (α ≈ 1.25)."
    )
    example = EXAMPLE
    size = (1220, 760)

    def capture(self, ctx) -> Path:  # noqa: D401
        # The quick result updates the draft coefficient and its preview.
        from asymmetry.gui.windows.grouping.dialog import GroupingDialog

        dataset = load_corpus_datasets([_DATA % _ALPHA_RUN])[0]
        dialog = GroupingDialog([dataset], selected_run_number=int(dataset.run_number))
        dialog.resize(*self.size)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        dialog.show()
        _process(150)

        # Pump the event loop until the worker fills the calibrated draft.
        section = dialog._alpha_section
        section._method_combo.setCurrentIndex(section._method_combo.findData("diamagnetic"))
        section._on_estimate()
        _wait_until(lambda: section._tasks.active_count == 0, timeout_ms=15000)
        # Let the calibrated draft preview settle before grabbing.
        _process(500)
        dialog._corrections_scroll.ensureWidgetVisible(section)
        _process(80)

        out_path = ctx.output_dir / f"{self.name}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pix = dialog.grab()
        if not pix.save(str(out_path), "PNG"):
            raise RuntimeError(f"Failed to save screenshot to {out_path}")
        # Release workers without entering a headless discard prompt.
        dialog._teardown_workers()
        dialog.deleteLater()
        _process(40)
        return out_path


# ---------------------------------------------------------------------------
# 2. Money shot — converged low-frequency ZF oscillation fit at base T
# ---------------------------------------------------------------------------
class MolAfmZfFitScenario(CorpusScenario):
    name = "corpus_molafm_zf_fit"
    description = (
        "Converged Oscillatory×Exponential+Constant fit on the molecular-AFM "
        "1.2 K zero-field run MUSR00017094: a slow coherent internal-field "
        "precession (ν ≈ 1.55 MHz, period ~0.65 µs), zoomed so the "
        "low-frequency oscillation resolves."
    )
    example = EXAMPLE
    size = (1500, 920)
    requires_fit = True

    def build(self) -> QWidget:
        from asymmetry.core.fitting.composite import CompositeModel
        from asymmetry.core.workflow.reduction import ReductionSettings, resolve_reduction_grouping
        from asymmetry.gui.mainwindow import MainWindow
        from asymmetry.gui.panels.fit.tab_base import (
            _param_table_rows_by_name,
            _set_param_table_value,
        )

        window = MainWindow()
        window._on_fit()
        window.resizeDocks([window._dock_data_browser], [320], Qt.Orientation.Horizontal)

        dataset = load_corpus_datasets([_DATA % _ZF_FIT_RUN])[0]
        settings = ReductionSettings(alpha=_molecular_balance())
        payload = resolve_reduction_grouping(dataset.run, settings)
        applied, _ = window._apply_grouping_settings_to_dataset(dataset, payload)
        if not applied:
            raise RuntimeError(f"Could not apply TF balance to run {dataset.run_number}")
        self.add_to_browser(window, [dataset])
        window._on_dataset_selected(dataset.run_number)
        _process(80)

        single_tab = window._fit_panel._single_tab
        single_tab._set_composite_model(CompositeModel(_ZF_MODEL[0], operators=_ZF_MODEL[1]))
        _process(80)

        table = single_tab._param_table
        rows = _param_table_rows_by_name(table)
        tail = (dataset.time > 3.0) & (dataset.time < _ZF_FIT_RANGE_US[1])
        seeds = {
            "A_1": 4.0,
            "frequency": 1.6,
            "phase": 0.0,
            "Lambda": 0.2,
            "A_bg": float(np.nanmedian(dataset.asymmetry[tail])),
        }
        for name, value in seeds.items():
            _set_param_table_value(table, rows[name], value)
        for name in ("A_1", "Lambda"):
            table.item(rows[name], table.COL_MIN).setText("0.0")
        table.item(rows["frequency"], table.COL_MIN).setText("0.3")
        table.item(rows["frequency"], table.COL_MAX).setText("2.5")
        window._plot_panel.set_fit_range(*_ZF_FIT_RANGE_US)
        single_tab._run_fit()
        single_tab.wait_for_fit()
        _process(80)

        # ν ≈ 1.55 MHz ⇒ ~0.65 µs period; zoom to the first ~5 µs (~8 cycles) so
        # the slow, weakly-damped molecular-magnet oscillation resolves, and
        # frame Y to show the precession and slowly varying baseline.
        window._plot_panel.set_view_limits(0.0, 5.0, *window._plot_panel.get_view_limits()[2:])
        _process(60)
        self._frame_y_to_window(window, 0.0, 5.0)
        _process(80)
        return window

    @staticmethod
    def _frame_y_to_window(window, x_min: float, x_max: float) -> None:
        ds = window._plot_panel
        t = getattr(ds, "_last_plot_time", None)
        a = getattr(ds, "_last_plot_asymmetry", None)
        if t is None or a is None or not len(t):
            return
        t = np.asarray(t, dtype=float)
        a = np.asarray(a, dtype=float)
        m = (t >= x_min) & (t <= x_max)
        if not np.any(m):
            return
        lo, hi = float(np.nanmin(a[m])), float(np.nanmax(a[m]))
        pad = 0.12 * (hi - lo or 1.0)
        window._plot_panel.set_view_limits(x_min, x_max, lo - pad, hi + pad)


# ---------------------------------------------------------------------------
# 3. Headline — ν(T) order parameter with the fitted power law → T_N
# ---------------------------------------------------------------------------
class MolAfmNuTScenario(CorpusScenario):
    name = "corpus_molafm_nu_t"
    description = (
        "Molecular-AFM order parameter: spontaneous ZF precession frequency "
        "ν(T) from real per-run fits (1.2 → 6 K), with the fitted "
        "phenomenological OrderParameter curve."
    )
    example = EXAMPLE
    size = (1240, 760)
    requires_fit = True

    def build(self) -> QWidget:
        from asymmetry.core.fitting.parameter_models import (
            ModelFitRange,
            ParameterCompositeModel,
            ParameterModelFit,
            fit_parameter_model,
        )
        from asymmetry.core.fitting.parameters import Parameter, ParameterSet
        from asymmetry.core.workflow.reduction import ReductionSettings, reduce_run
        from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel

        settings = ReductionSettings(alpha=_molecular_balance())
        rows = []
        for run, temperature in _ZF_OP_RUNS:
            source = load_corpus_datasets([_DATA % run])[0]
            dataset = reduce_run(source.run, settings)
            # The observed band shifts from about 1.6 MHz at base temperature.
            nu_seed = 1.6 - 0.15 * (temperature - _ZF_OP_RUNS[0][1])
            result = _fit_zf_frequency(dataset, nu_seed)
            nu = result.parameters["frequency"].value
            error = result.uncertainties.get("frequency")
            if not result.success or error is None or not np.isfinite(error) or error <= 0:
                raise RuntimeError(
                    f"Molecular-magnet run {run} has no usable frequency uncertainty"
                )
            rows.append(
                {
                    "run_number": run,
                    "run_label": dataset.run_label,
                    "field": 0.0,
                    "temperature": temperature,
                    "values": {"frequency": nu},
                    "errors": {"frequency": error},
                    "model_name": "Oscillatory * Exponential + Constant",
                    "chi_squared": result.chi_squared,
                    "reduced_chi_squared": result.reduced_chi_squared,
                }
            )

        batch_id = "molafm-nu-t-corpus"
        panel = FitParametersPanel()
        panel.load_representation_series(
            [(batch_id, "ν(T) — molecular AFM ZF", rows)],
            select_id=batch_id,
        )

        panel._x_combo.setCurrentText("𝑇 (K)")
        selected_rows = panel._included_trend_rows("temperature")
        temperature = np.array([row.temperature for row in selected_rows])
        nu, nu_err = panel._series_y_arrays(selected_rows, "frequency")

        # The shape exponent is fixed to one for this illustrative power law.
        model = ParameterCompositeModel(["OrderParameter"])
        params = ParameterSet(
            [
                Parameter(name="y0", value=1.6, min=0.5),
                Parameter(name="Tc", value=6.7, min=6.0, max=9.0),
                Parameter(name="beta", value=0.4, min=0.1, max=0.9),
                Parameter(name="alpha", value=1.0, fixed=True),
            ]
        )
        result = fit_parameter_model(
            temperature,
            nu,
            nu_err,
            model,
            params,
            x_min=float(temperature.min()),
            x_max=float(temperature.max()),
        )
        if not result.success:
            raise RuntimeError("Molecular-AFM OrderParameter ν(T) fit did not converge")

        fit_range = ModelFitRange(
            x_min=float(temperature.min()),
            x_max=float(temperature.max()),
            model=model,
            parameters=result.parameters,
            result=result,
        )
        panel._model_fits["frequency"] = ParameterModelFit(
            parameter_name="frequency",
            x_key="temperature",
            ranges=[fit_range],
            active=True,
        )
        panel._sync_active_group_state()
        panel._refresh_model_fit_button_labels()
        _process(80)
        return panel

    def settle(self, widget: QWidget) -> None:
        _process(200)
        widget._refresh_plot()
        _wait_until(
            lambda: (
                not widget._trend_curve_compute_active
                and widget._precomputed_trend_curves is not None
            ),
            timeout_ms=20000,
        )
        _process(200)


# ---------------------------------------------------------------------------
# 4. T_N by direct comparison — 6 K (ordered) vs 7 K (paramagnetic) overlay
# ---------------------------------------------------------------------------
class MolAfmZfOverlayScenario(CorpusScenario):
    name = "corpus_molafm_zf_overlay"
    description = (
        "6 K vs 7 K zero-field spectra overlaid: the coherent internal-field "
        "oscillation present at 6 K (MUSR00017099) is gone at 7 K "
        "(MUSR00017100), fixing T_N ≈ 6–7 K directly from the data."
    )
    example = EXAMPLE
    size = (1120, 640)

    def build(self) -> QWidget:
        from asymmetry.core.workflow.reduction import ReductionSettings, reduce_run
        from asymmetry.gui.widgets.mpl_canvas import create_canvas

        settings = ReductionSettings(alpha=_molecular_balance(), rebin=8)
        figure, canvas = create_canvas(layout="tight")
        axes = figure.add_subplot(111)
        for run, colour, label in zip(
            _OVERLAY_RUNS, (tokens.ACCENT, tokens.TEXT_DIM), ("6 K", "7 K"), strict=True
        ):
            source = load_corpus_datasets([_DATA % run])[0]
            dataset = reduce_run(source.run, settings).time_range(0.11, 5.0)
            axes.errorbar(
                dataset.time,
                dataset.asymmetry,
                yerr=dataset.error,
                color=colour,
                linewidth=1,
                marker=".",
                label=f"{label} · run {run}",
            )

        axes.set_xlabel("Time (µs)")
        axes.set_ylabel("Asymmetry (%)")
        axes.set_xlim(0.1, 5.0)
        axes.legend(loc="upper right")
        self._canvas = canvas
        return canvas

    def settle(self, widget: QWidget) -> None:
        _process(200)
        widget.draw()
        _process(120)


register(MolAfmAlphaScenario())
register(MolAfmZfFitScenario())
register(MolAfmNuTScenario())
register(MolAfmZfOverlayScenario())
