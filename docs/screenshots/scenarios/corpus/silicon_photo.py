"""Corpus scenarios — Photo-µSR in silicon (Semiconductors).

The WiMDA muon-school **period-mode** worked example, on the real HiFi (ISIS)
HDF4 NeXus corpus (``HIFI00103277``–``103299``, RB1520457). Pulsed-laser
photoexcitation injects excess carriers into intrinsic Si; implanted µ⁺ form
Muonium whose relaxation rate λ is a yardstick for the excess carrier density
Δn. The laser fires every other muon pulse, so the DAE records **light-OFF and
light-ON spectra as two periods in one ``.nxs`` file** — this example is the
corpus's unique multi-period showcase. Convention (GROUND_TRUTH §1): period 1 =
**Red = light-ON**, period 2 = **Green = light-OFF**.

The paper is the spec (same data): K. Yokoyama, J. S. Lord, J. Miao,
P. Murahari, A. J. Drew, *"A New Method for Measuring Excess Carrier Lifetime in
Bulk Silicon: Photoexcited Muon Spin Spectroscopy,"* PRL **119**, 226601 (2017)
(open arXiv:1702.06846). Grade against the 291 K, LF 10 mT Fig. 1 numbers:
calibration exponent **α = 0.68(4)** (Fig 1d) and carrier recombination
lifetime **τ₀ = 11.1(9) µs** (Fig 1e). See GROUND_TRUTH §4/§5/§10a.

Workflow (GROUND_TRUTH §4): fit the light-OFF (Green) period with a single
exponential → amplitude A₀; refit each light-ON (Red) period over the **first
1 µs** with **A₀ fixed** → λ. Then (Trend 1) λ vs Δn on log–log fits the power
law λ = β(Δn/Δn₀)^α (calibration set 103277–103286), and (Trend 2) Δn vs ΔT
fits a single exponential → τ₀ (delay scan 103287–103298).

Scenarios registered:

* ``corpus_si_period_mapping`` — the period → Red/Green mapping dialog on real
  run 103277: period 1 → Red (light-ON), period 2 → Green (light-OFF). THE
  period-mode UI.
* ``corpus_si_on_off_overlay`` — light-ON vs light-OFF asymmetry of run 103277
  overlaid: the light-induced extra relaxation, visible by eye.
* ``corpus_si_lambda_fit`` — single-exponential fit on the light-ON period of
  the highest-Δn run 103277 over the first 1 µs (A₀ fixed at the light-OFF
  value) → λ ≈ 1.27 µs⁻¹ (digitised target ~1.29, GROUND_TRUTH §10a).
* ``corpus_si_lambda_vs_dn`` — Trend 1 calibration: λ vs Δn on log–log across
  the 10 injection runs, with the fitted power law (exponent ≈ α ≈ 0.68).
* ``corpus_si_tau_decay`` — headline Trend 2: Δn vs ΔT across the delay scan
  with the single-exponential carrier-decay fit → τ₀ ≈ 11 µs.

See ``NOTES_silicon.md`` for run selection, fitted values vs targets, and the
period-mode UX notes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtWidgets import QApplication, QWidget

from .._base import CaptureContext
from ._corpus import CorpusScenario, _process_events_for, corpus_path, register

EXAMPLE = "Semiconductors/Photo-muSR in silicon"
_DATA = EXAMPLE + "/Data"

# Injection / calibration set (ΔT = 0): run → injected excess density Δn (cm⁻³),
# GROUND_TRUTH §3 (exact, from the guide run table). Highest Δn first.
CAL_RUNS: list[tuple[int, float]] = [
    (103277, 8.9e13),
    (103278, 7.9e13),
    (103279, 7.1e13),
    (103280, 6.3e13),
    (103281, 5.6e13),
    (103282, 4.7e13),
    (103283, 3.7e13),
    (103284, 2.7e13),
    (103285, 1.8e13),
    (103286, 9.3e12),
]

# Delay scan (fixed injected Δn): run → laser delay ΔT (µs), GROUND_TRUTH §3.
DELAY_RUNS: list[tuple[int, float]] = [
    (103287, 0.1),
    (103288, 3.0),
    (103289, 5.0),
    (103290, 10.0),
    (103291, 15.0),
    (103292, 20.0),
    (103293, 25.0),
    (103294, 30.0),
    (103295, 40.0),
    (103296, 50.0),
    (103297, 60.0),
    (103298, 70.0),
]

REF_DENSITY = 8.9e13  # Δn₀ reference density (paper Fig 1d, GROUND_TRUTH §10)
LIGHT_ON_TMAX = 1.0  # light-ON fit window: first 1 µs only (GROUND_TRUTH §4)


def _rel(run: int) -> str:
    return str(corpus_path(f"{_DATA}/HIFI00{run}.nxs"))


# --------------------------------------------------------------------------- #
# Core λ extraction (GUI-free — used by the trend scenarios)
# --------------------------------------------------------------------------- #


def _fit_exponential(dataset, t_max: float, a0: float | None = None):
    """Fit a single exponential A₀·exp(−λt) to *dataset* over [0, t_max].

    When *a0* is given the amplitude is held fixed (the light-ON step of the
    guide workflow); otherwise A₀ is free (the light-OFF step). Returns
    ``(lambda, A0)``.
    """
    from asymmetry.core.fitting.composite import CompositeModel
    from asymmetry.core.fitting.engine import FitEngine
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    model = CompositeModel(["Exponential"])
    if a0 is None:
        params = ParameterSet([Parameter("A_1", 15.0, min=0.0), Parameter("Lambda", 0.3, min=0.0)])
    else:
        params = ParameterSet(
            [Parameter("A_1", a0, min=0.0, fixed=True), Parameter("Lambda", 0.5, min=0.0)]
        )
    result = FitEngine().fit(dataset, model.function, params, t_min=0.0, t_max=t_max)
    return (
        abs(result.parameters["Lambda"].value),
        float(result.parameters["A_1"].value),
    )


def _light_on_lambda(run: int) -> float:
    """Guide workflow for one run → light-ON λ.

    Fit the Green (light-OFF) period free to get A₀, then refit the Red
    (light-ON) period over the first 1 µs with A₀ fixed.
    """
    from asymmetry.core.io import load
    from asymmetry.core.io.periods import select_period

    combined = load(_rel(run))
    _, a0 = _fit_exponential(select_period(combined, "green"), 1.0)
    lam, _ = _fit_exponential(select_period(combined, "red"), LIGHT_ON_TMAX, a0=a0)
    return lam


def _register_derived_labels() -> None:
    """Give Δn / ΔT proper symbols + units on every label path (idempotent)."""
    from asymmetry.core.fitting.parameters import register_derived_param_info

    register_derived_param_info(
        "Dn",
        plain="Dn",
        unicode="Δn",
        latex=r"$\Delta n$",
        gle=r"\Delta n",
        unit="cm⁻³",
    )
    register_derived_param_info(
        "dT",
        plain="dT",
        unicode="ΔT",
        latex=r"$\Delta T$",
        gle=r"\Delta T",
        unit="µs",
    )


# --------------------------------------------------------------------------- #
# 1. Period → Red/Green mapping dialog (THE period-mode UI)
# --------------------------------------------------------------------------- #
class SiPeriodMappingScenario(CorpusScenario):
    name = "corpus_si_period_mapping"
    description = (
        "Period → Red/Green mapping dialog on the real two-period silicon "
        "photo-µSR run 103277: period 1 → Red (laser ON), period 2 → Green "
        "(laser OFF) — the multi-period DAE structure resolved into the two "
        "light spectra."
    )
    example = EXAMPLE
    size = (640, 190)

    def capture(self, ctx: CaptureContext) -> Path:  # noqa: D401
        from asymmetry.core.io import load
        from asymmetry.core.io.periods import select_period
        from asymmetry.gui.windows.period_mapping_dialog import PeriodMappingDialog

        combined = load(_rel(103277))
        periods = [select_period(combined, "red"), select_period(combined, "green")]
        _inject_real_good_frames(combined, periods)

        dialog = PeriodMappingDialog(periods)
        dialog.resize(*self.size)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        dialog.show()
        _pump(120)

        pix = dialog.grab()
        out_path = ctx.output_dir / f"{self.name}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not pix.save(str(out_path), "PNG"):
            raise RuntimeError(f"Failed to save screenshot to {out_path}")
        dialog.close()
        dialog.deleteLater()
        _pump(40)
        return out_path


# --------------------------------------------------------------------------- #
# 2. Light-ON vs light-OFF asymmetry overlay
# --------------------------------------------------------------------------- #
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
        for ds in (on_ds, off_ds):
            window._data_browser.add_dataset(ds)
        run_numbers = [int(on_ds.run_number), int(off_ds.run_number)]
        window._data_browser.create_data_group(run_numbers, name="Si 103277 — laser ON / OFF")

        window._plot_panel.set_overlay_enabled(True, emit_signal=True)
        window._data_browser._table.selectAll()
        window._on_dataset_selected(run_numbers[0])
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
        "light-OFF value → λ ≈ 1.27 µs⁻¹ (digitised target ~1.29)."
    )
    example = EXAMPLE
    size = (1500, 920)
    requires_fit = True

    def build(self) -> QWidget:
        from asymmetry.core.fitting.composite import CompositeModel
        from asymmetry.core.io import load
        from asymmetry.core.io.periods import select_period
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window._on_fit()
        window.resizeDocks([window._dock_data_browser], [300], Qt.Orientation.Horizontal)

        combined = load(_rel(103277))
        # A₀ from the light-OFF (Green) period, per the guide workflow.
        _, a0 = _fit_exponential(select_period(combined, "green"), 1.0)

        on_ds = select_period(combined, "red")
        on_ds.run.run_number = 1032770 + 1  # derived id (light-ON period)
        on_ds.metadata["run_label"] = "103277 laser ON"
        window._data_browser.add_dataset(on_ds)
        window._on_dataset_selected(on_ds.run_number)
        _process_events_for(milliseconds=80)

        # Restrict the fit to the first 1 µs (GROUND_TRUTH §4). The plot panel is
        # the canonical fit-range owner; set_fit_range slices the dataset handed
        # to the fit tab (a spinbox setValue would not commit the range).
        window._plot_panel.set_fit_range(0.0, LIGHT_ON_TMAX)
        _process_events_for(milliseconds=60)

        single_tab = window._fit_panel._single_tab
        model = CompositeModel(["Exponential"])
        single_tab._set_composite_model(model)
        _process_events_for(milliseconds=60)
        # Re-populate with A₀ fixed at the light-OFF amplitude and λ seeded near
        # the expected value; the guide holds A₀ constant for the light-ON fit.
        single_tab._param_table.populate(
            model,
            value_overrides={"A_1": a0, "Lambda": 0.9},
            fixed_names={"A_1", "shape_factor_a"},
        )
        _process_events_for(milliseconds=40)

        single_tab._run_fit()
        single_tab.wait_for_fit()
        _process_events_for(milliseconds=80)

        # Show the decay and the first bit of the flat tail: 0–2.5 µs.
        window._plot_panel.set_view_limits(0.0, 2.5, -2.0, 18.0)
        _process_events_for(milliseconds=80)
        return window


# --------------------------------------------------------------------------- #
# 4. Trend 1 — λ vs Δn calibration power law (log–log)
# --------------------------------------------------------------------------- #
class SiLambdaVsDnScenario(CorpusScenario):
    name = "corpus_si_lambda_vs_dn"
    description = (
        "Calibration trend: light-ON relaxation λ vs injected carrier density "
        "Δn across the 10 silicon injection runs (103277–103286) on log–log "
        "axes, with the fitted power law λ = β(Δn/Δn₀)^α (exponent ≈ α ≈ 0.68)."
    )
    example = EXAMPLE
    size = (1240, 800)
    requires_fit = True

    def build(self) -> QWidget:
        from asymmetry.core.fitting.parameter_models import (
            ModelFitRange,
            ParameterCompositeModel,
            ParameterModelFit,
            fit_parameter_model,
        )
        from asymmetry.core.fitting.parameters import Parameter, ParameterSet
        from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel

        _register_derived_labels()

        dn = np.array([d for _r, d in CAL_RUNS])
        lam = np.array([_light_on_lambda(r) for r, _d in CAL_RUNS])

        row_dicts = [
            {
                "run_number": run,
                "run_label": f"Δn={dn[i]:.1e}",
                "field": -100.0,
                "temperature": 291.0,
                "values": {"Lambda": float(lam[i]), "Dn": float(dn[i])},
                "errors": {"Lambda": float(0.05 * lam[i]), "Dn": 0.0},
            }
            for i, (run, _d) in enumerate(CAL_RUNS)
        ]

        panel = FitParametersPanel()
        panel.load_representation_series(
            [("si-cal", "λ vs Δn — Si photo-µSR", row_dicts)], select_id="si-cal"
        )
        _process_events_for(milliseconds=60)
        _select_axes(panel, y_param="Lambda", x_key="param:Dn")

        # Power law λ = a·|Δn|^n (+c); the exponent n is the calibration α.
        model = ParameterCompositeModel(["PowerLaw"])
        params = ParameterSet(
            [
                Parameter("a", 1e-9, min=0.0),
                Parameter("n", 0.68, min=0.0, max=3.0),
                Parameter("c", 0.0),
            ]
        )
        result = fit_parameter_model(
            dn,
            lam,
            np.array([0.05 * v for v in lam]),
            model,
            params,
            x_min=float(dn.min()),
            x_max=float(dn.max()),
        )
        if not result.success:
            raise RuntimeError("Si λ vs Δn calibration power-law fit did not converge")
        self._alpha = float(result.parameters["n"].value)

        fit_range = ModelFitRange(
            x_min=float(dn.min()),
            x_max=float(dn.max()),
            model=model,
            parameters=result.parameters,
            result=result,
        )
        panel._model_fits["Lambda"] = ParameterModelFit(
            parameter_name="Lambda", x_key="param:Dn", ranges=[fit_range], active=True
        )
        # log–log, as the guide prescribes for the power law.
        panel._log_x_check.setChecked(True)
        _set_y_log(panel, "Lambda", True)
        panel._sync_active_group_state()
        panel._refresh_model_fit_button_labels()
        _process_events_for(milliseconds=80)
        return panel

    def settle(self, widget: QWidget) -> None:
        _settle_trend(widget)


# --------------------------------------------------------------------------- #
# 5. Trend 2 — headline carrier decay Δn vs ΔT → τ₀
# --------------------------------------------------------------------------- #
class SiTauDecayScenario(CorpusScenario):
    name = "corpus_si_tau_decay"
    description = (
        "Headline result: excess carrier density Δn vs laser delay ΔT across "
        "the silicon delay scan (103287–103298), fitted with a single "
        "exponential Δn(ΔT)=Δn₀·exp(−ΔT/τ₀) → carrier recombination lifetime "
        "τ₀ ≈ 11 µs (paper 11.1(9) µs)."
    )
    example = EXAMPLE
    size = (1240, 800)
    requires_fit = True

    def build(self) -> QWidget:
        from asymmetry.core.fitting.parameter_models import (
            ModelFitRange,
            ParameterCompositeModel,
            ParameterModelFit,
            fit_parameter_model,
        )
        from asymmetry.core.fitting.parameters import Parameter, ParameterSet
        from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel

        _register_derived_labels()

        # First build the calibration power law from the injection set, then use
        # it to invert each delay-scan λ into Δn (GROUND_TRUTH §4 steps 5–7).
        dn_cal = np.array([d for _r, d in CAL_RUNS])
        lam_cal = np.array([_light_on_lambda(r) for r, _d in CAL_RUNS])
        beta, alpha = _fit_power_law(dn_cal, lam_cal)
        self._alpha = float(alpha)

        dt = np.array([t for _r, t in DELAY_RUNS])
        lam_d = np.array([_light_on_lambda(r) for r, _t in DELAY_RUNS])
        dn_d = REF_DENSITY * (lam_d / beta) ** (1.0 / alpha)

        row_dicts = [
            {
                "run_number": run,
                "run_label": f"ΔT={dt[i]:g} µs",
                "field": -100.0,
                "temperature": 291.0,
                "values": {"Dn": float(dn_d[i]), "dT": float(dt[i])},
                "errors": {"Dn": float(0.08 * dn_d[i]), "dT": 0.0},
            }
            for i, (run, _t) in enumerate(DELAY_RUNS)
        ]

        panel = FitParametersPanel()
        panel.load_representation_series(
            [("si-decay", "Δn vs ΔT — Si photo-µSR", row_dicts)], select_id="si-decay"
        )
        _process_events_for(milliseconds=60)
        _select_axes(panel, y_param="Dn", x_key="param:dT")

        model = ParameterCompositeModel(["ExponentialDecay"])
        params = ParameterSet(
            [
                Parameter("a", 9.4e13, min=0.0),
                Parameter("tau", 11.0, min=0.1, max=200.0),
                Parameter("c", 0.0),
            ]
        )
        result = fit_parameter_model(
            dt,
            dn_d,
            np.array([0.08 * v for v in dn_d]),
            model,
            params,
            x_min=float(dt.min()),
            x_max=float(dt.max()),
        )
        if not result.success:
            raise RuntimeError("Si Δn vs ΔT carrier-decay fit did not converge")
        self._tau0 = float(result.parameters["tau"].value)

        fit_range = ModelFitRange(
            x_min=float(dt.min()),
            x_max=float(dt.max()),
            model=model,
            parameters=result.parameters,
            result=result,
        )
        panel._model_fits["Dn"] = ParameterModelFit(
            parameter_name="Dn", x_key="param:dT", ranges=[fit_range], active=True
        )
        panel._sync_active_group_state()
        panel._refresh_model_fit_button_labels()
        _process_events_for(milliseconds=80)
        return panel

    def settle(self, widget: QWidget) -> None:
        _settle_trend(widget)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _fit_power_law(dn: np.ndarray, lam: np.ndarray) -> tuple[float, float]:
    """Fit λ = β·(Δn/Δn₀)^α; return (β, α)."""
    from scipy.optimize import curve_fit

    def model(x, beta, a):
        return beta * (x / REF_DENSITY) ** a

    popt, _ = curve_fit(model, dn, lam, p0=[1.4, 0.68], maxfev=10000)
    return float(popt[0]), float(popt[1])


def _labelled_on_off(window, run: int):
    """Return (light-ON, light-OFF) period datasets with distinct derived run
    numbers + friendly labels so both survive the run-number-keyed browser."""
    from asymmetry.core.io import load
    from asymmetry.core.io.periods import select_period

    combined = load(_rel(run))
    on_ds = select_period(combined, "red")
    off_ds = select_period(combined, "green")
    on_ds.run.run_number = window._data_browser.next_derived_run_number()
    on_ds.metadata["run_label"] = f"{run} laser ON"
    off_ds.run.run_number = window._data_browser.next_derived_run_number()
    off_ds.metadata["run_label"] = f"{run} laser OFF"
    return on_ds, off_ds


def _inject_real_good_frames(combined, periods) -> None:
    """Fill each period's good-frame count from the file's Beamlog.

    The loader leaves ``grouping['good_frames'] = 1.0`` for period datasets, so
    the mapping dialog would show a meaningless "1". The real per-period good
    frames live in the NeXus ``Beamlog_Good_Frames_Total`` log; split evenly
    across the two periods (they share beam exposure equally). Best-effort: on
    any structural surprise the datasets are left untouched.
    """
    try:
        nf = combined.metadata.get("nexus_fields", {})
        total = float(nf["Beamlog_Good_Frames_Total"]["values"]["value"][-1])
    except (KeyError, IndexError, TypeError, ValueError):
        return
    per = total / max(len(periods), 1)
    for ds in periods:
        if ds.run is not None and isinstance(ds.run.grouping, dict):
            ds.run.grouping["good_frames"] = per


def _select_axes(panel, *, y_param: str, x_key: str) -> None:
    """Select the y-parameter row and set the trend x-axis on a FitParametersPanel."""
    table = panel._y_selector_table
    for row in range(table.rowCount()):
        item = table.item(row, 0)
        if item is not None and item.data(Qt.ItemDataRole.UserRole) == y_param:
            table.selectRow(row)
            break
    _process_events_for(milliseconds=30)
    idx = panel._x_combo.findData(x_key)
    if idx >= 0:
        panel._x_combo.setCurrentIndex(idx)
    _process_events_for(milliseconds=30)


def _set_y_log(panel, y_param: str, enabled: bool) -> None:
    """Enable log scaling on one y-parameter's per-row control, if present."""
    controls = panel._y_controls.get(y_param)
    if controls is not None and getattr(controls, "log", None) is not None:
        controls.log.setChecked(enabled)
    else:
        panel._log_y_check.setChecked(enabled)


def _settle_trend(widget: QWidget) -> None:
    _process_events_for(milliseconds=200)
    widget._refresh_plot()
    elapsed = 0
    while elapsed < 20000:
        if not widget._trend_curve_compute_active and widget._precomputed_trend_curves is not None:
            break
        _process_events_for(milliseconds=30)
        elapsed += 30
    _process_events_for(milliseconds=200)


def _pump(milliseconds: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(int(milliseconds), loop.quit)
    loop.exec()
    QApplication.processEvents()


register(SiPeriodMappingScenario())
register(SiOnOffOverlayScenario())
register(SiLambdaFitScenario())
register(SiLambdaVsDnScenario())
register(SiTauDecayScenario())
