"""Corpus scenarios — Muonium reaction with maleic acid (Chemistry).

Drives the Asymmetry GUI through the WiMDA muon-school *chemical-kinetics*
example on the **real EMU ``.nxs`` corpus files** (runs 78251–78302, ISIS/RAL,
2018). See the example's ``GROUND_TRUTH.md``.

The chemistry: **muonium** (Mu = μ⁺e⁻, a light H-atom isotope) adds across the
C=C double bond of **maleic acid** in aqueous solution. The reaction is
pseudo-first-order in Mu, so the Mu **relaxation rate** obeys

    λ_Mu = λ₀ + k_Mu·[x]

with [x] the relative maleic-acid concentration and k_Mu the bimolecular rate
constant (the deliverable). A small transverse field of **2 G** makes the
triplet-muonium precession visible at ν_Mu ≈ 1.394 MHz/G × 2 G ≈ **2.79 MHz** —
a striking **103×** the diamagnetic (μ⁺) 2 G Larmor line at ≈ 0.027 MHz. Each
2 G run is fitted with a relaxing Mu oscillation over a slowly-relaxing
diamagnetic baseline; λ_Mu rises with concentration (headline) and with
temperature (Arrhenius).

Because the source files give concentration **relative only** (quarter : half :
full = 1 : 2 : 4 of one stock; GROUND_TRUTH §3, §9), the concentration is *not*
in the file metadata — the headline trend needs a **manual fit-table column**
(the key feature this example demonstrates), and k_Mu is delivered in
*relative-concentration units* (µs⁻¹ per rel-unit), not absolute M⁻¹s⁻¹.

Scenarios registered:

* ``corpus_maleic_mu_precession``   — converged Mu-precession fit on deox water
  (2 G, ν ≈ 2.79 MHz), the "muonium in water" render.
* ``corpus_maleic_concentration``   — water/half/full overlay: Mu relaxation
  visibly faster with [x].
* ``corpus_maleic_kmu_trend``       — headline: λ_Mu vs [x] with a manual
  concentration column and the fitted line whose slope is k_Mu.
* ``corpus_maleic_arrhenius``       — λ_Mu(T) full-series Arrhenius plot → E_a.

Workflow / model choices (GROUND_TRUTH §4). The Mu frequency is a physics
constant (2 G field, γ_Mu known), so it is **fixed** at 2.788 MHz; on this
noisy single-run data a floating frequency latches onto per-bin noise for the
fast-relaxing maleic runs. The Mu amplitude A_Mu (the muon *fraction* that forms
Mu) is set by the water and is common across the concentration series, so it and
the phase are **anchored** from the clean deox-water run — only λ_Mu (and the
diamagnetic baseline) float per run. The additive slow ``Exponential`` absorbs
the strongly-sloping diamagnetic baseline that a bare ``Constant`` cannot.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtWidgets import QApplication, QWidget

from asymmetry.gui.styles import tokens

from ._corpus import (
    CorpusScenario,
    _process_events_for,
    load_corpus_datasets,
    register,
)

EXAMPLE = "Chemistry/Muonium reaction with maleic acid"
_DATA = EXAMPLE + "/Data/emu000%d.nxs"

# Mu triplet precession frequency at the 2.00 G observation field:
# γ_Mu ≈ 1.394 MHz/G × 2.00 G ≈ 2.788 MHz (GROUND_TRUTH §4/§6, literature const).
FMU = 2.788
# Diamagnetic (μ⁺) Larmor at 2 G: γ_μ ≈ 0.01355 MHz/G × 2 G ≈ 0.0271 MHz.
FDIA_2G = 0.01355 * 2.0

# Room-temperature (~290 K) concentration series, 2 G Mu runs (GROUND_TRUTH §3).
# [x] in relative units: deox water 0, quarter 1, half 2, full 4 (ratio 1:2:4).
_CONC_SERIES: list[tuple[int, float, str]] = [
    (78251, 0.0, "deox. water"),
    (78279, 1.0, "quarter"),
    (78277, 2.0, "half"),
    (78257, 4.0, "full"),
]

# Full-concentration temperature scan, 2 G Mu runs, ascending T (GROUND_TRUTH §3).
_FULL_TSCAN: list[tuple[int, float]] = [
    (78259, 278.0),
    (78261, 288.0),
    (78257, 290.0),
    (78263, 298.0),
    (78265, 308.0),
    (78267, 318.0),
    (78269, 328.0),
    (78271, 338.0),
    (78273, 348.0),
    (78275, 358.0),
]

_ANCHOR_RUN = 78251  # deox water: the clean, slowly-relaxing Mu reference
_CONC_KEY = "custom:maleic_conc"


def _rel(run: int) -> str:
    return _DATA % run


def _pump(ms: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(int(ms), loop.quit)
    loop.exec()
    QApplication.processEvents()


def _pump_until(predicate, timeout_ms: int = 20_000) -> None:
    if predicate():
        return
    loop = QEventLoop()
    check = QTimer()
    check.timeout.connect(lambda: loop.quit() if predicate() else None)
    check.start(10)
    guard = QTimer()
    guard.setSingleShot(True)
    guard.timeout.connect(loop.quit)
    guard.start(int(timeout_ms))
    loop.exec()
    check.stop()
    guard.stop()


def _grab(widget: QWidget, name: str, output_dir: Path) -> Path:
    out = output_dir / f"{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    pix = widget.grab()
    if not pix.save(str(out), "PNG"):
        raise RuntimeError(f"Failed to save screenshot to {out}")
    return out


# ── the Mu-relaxation model & fit (GROUND_TRUTH §4) ────────────────────────
def _mu_model():
    from asymmetry.core.fitting.composite import CompositeModel

    # Osc·Exp (relaxing Mu precession) + Exp (sloping diamagnetic baseline).
    # Params: A_1, frequency, phase, Lambda_2 (=λ_Mu), A_3, Lambda_3.
    # A bare additive Constant on top of the slow Exp is degenerate at 2 G (the
    # smooth baseline ≈ const), which makes MINUIT flag an invalid minimum and
    # the GUI refuse to draw the fit; the single additive Exp captures the
    # sloping diamagnetic baseline on its own and converges cleanly.
    return CompositeModel(
        ["Oscillatory", "Exponential", "Exponential"],
        operators=["*", "+"],
    )


def _fit_mu(dataset, *, a_mu=None, phase=None, seed_lam=1.0, t_min=0.2, t_max=10.0):
    """Fit one 2 G run's Mu relaxation via the real core engine.

    Frequency is fixed at ``FMU``. When *a_mu*/*phase* are given (anchored from
    the deox-water run) they are held too, so only λ_Mu and the diamagnetic
    baseline float — the robust route for the fast-relaxing maleic runs.
    Returns ``(lambda_mu, lambda_err, result, params_dict)``.
    """
    from asymmetry.core.fitting.engine import FitEngine
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    freq = Parameter("frequency", FMU)
    freq.fixed = True
    p_amp = Parameter("A_1", a_mu if a_mu is not None else 2.0, min=0.0, max=20.0)
    if a_mu is not None:
        p_amp.fixed = True
    p_phase = Parameter("phase", phase if phase is not None else -0.3, min=-3.2, max=3.2)
    if phase is not None:
        p_phase.fixed = True
    params = ParameterSet(
        [
            p_amp,
            freq,
            p_phase,
            Parameter("Lambda_2", seed_lam, min=0.02, max=12.0),
            Parameter("A_3", 15.0, min=0.0, max=45.0),
            Parameter("Lambda_3", 0.1, min=0.0, max=3.0),
        ]
    )
    result = FitEngine().fit(dataset, _mu_model().function, params, t_min=t_min, t_max=t_max)
    p = result.parameters
    lam = abs(p["Lambda_2"].value)
    lam_err = float(result.uncertainties.get("Lambda_2", 0.0)) or 0.02
    return lam, lam_err, result, dict(result.uncertainties)


def _anchor():
    """Fit the deox-water reference to pin A_Mu and phase for the series."""
    ds = load_corpus_datasets([_rel(_ANCHOR_RUN)])[0]
    from asymmetry.core.fitting.engine import FitEngine
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    freq = Parameter("frequency", FMU)
    freq.fixed = True
    params = ParameterSet(
        [
            Parameter("A_1", 2.0, min=0.0, max=20.0),
            freq,
            Parameter("phase", -0.3, min=-3.2, max=3.2),
            Parameter("Lambda_2", 0.5, min=0.02, max=3.0),
            Parameter("A_3", 15.0, min=0.0, max=45.0),
            Parameter("Lambda_3", 0.1, min=0.0, max=3.0),
        ]
    )
    result = FitEngine().fit(ds, _mu_model().function, params, t_min=0.2, t_max=10.0)
    p = result.parameters
    return abs(p["A_1"].value), p["phase"].value, abs(p["Lambda_2"].value)


def _baseline_subtracted(dataset, result, rebin_factor=3):
    """Return (t, Mu-oscillation) with the fitted diamagnetic baseline removed."""
    ds = dataset.rebin(rebin_factor) if rebin_factor > 1 else dataset
    t = np.asarray(ds.time, dtype=float)
    a = np.asarray(ds.asymmetry, dtype=float)
    p = result.parameters
    a3 = abs(p["A_3"].value)
    l3 = abs(p["Lambda_3"].value)
    baseline = a3 * np.exp(-l3 * t)
    return t, a - baseline


# ══════════════════════════════════════════════════════════════════════════
# 1. Mu precession fit on deox water (the "muonium in water" render)
# ══════════════════════════════════════════════════════════════════════════
class MaleicMuPrecessionScenario(CorpusScenario):
    name = "corpus_maleic_mu_precession"
    description = (
        "Converged relaxing-Mu-oscillation fit on the deoxygenated-water 2 G run "
        "78251: triplet muonium precesses at ν ≈ 2.79 MHz (103× the diamagnetic "
        "2 G Larmor line), slowly relaxing at λ₀."
    )
    example = EXAMPLE
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
        window.resizeDocks([window._dock_data_browser], [320], Qt.Orientation.Horizontal)

        # Rebin ×3 (48 ns bins; Nyquist ≈ 10 MHz ≫ 2.79 MHz) so the ~0.36 µs Mu
        # oscillation reads clearly in the plot instead of a wall of per-bin noise.
        ds = load_corpus_datasets([_rel(_ANCHOR_RUN)])[0].rebin(3)
        window._data_browser.add_dataset(ds)
        window._on_dataset_selected(ds.run_number)

        single_tab = window._fit_panel._single_tab
        single_tab._set_composite_model(
            CompositeModel(
                ["Oscillatory", "Exponential", "Exponential"],
                operators=["*", "+"],
            )
        )
        _process_events_for(milliseconds=80)

        rows = _param_table_rows_by_name(single_tab._param_table)
        seeds = {
            "A_1": 2.0,
            "frequency": FMU,
            "phase": -0.3,
            "Lambda_2": 0.5,
            "A_3": 15.0,
            "Lambda_3": 0.1,
        }
        for name, value in seeds.items():
            if name in rows:
                _set_param_table_value(single_tab._param_table, rows[name], value)
        # Fix the Mu precession frequency (a physics constant: 2 G × γ_Mu). On
        # this noisy single-run data a floating frequency drives the fit to a
        # "call limit reached / invalid minimum" failure; fixing it — the same
        # constraint the standalone anchor uses — keeps the fit well-conditioned.
        self._fix_param(single_tab._param_table, rows.get("frequency"))
        _process_events_for(milliseconds=40)

        window._plot_panel.set_fit_range(0.2, 8.0)
        _process_events_for(milliseconds=40)
        single_tab._run_fit()
        single_tab.wait_for_fit()

        # Zoom to the first ~2.5 µs (~7 Mu cycles), framing Y to that window so
        # the damped precession sits large rather than as ripple on the baseline.
        window._plot_panel.set_view_limits(0.0, 2.5, *window._plot_panel.get_view_limits()[2:])
        _process_events_for(milliseconds=60)
        self._frame_y(window, 0.0, 2.5)
        _process_events_for(milliseconds=80)
        return window

    @staticmethod
    def _fix_param(param_table, row) -> None:
        from PySide6.QtWidgets import QCheckBox

        if row is None:
            return
        cell = param_table.cellWidget(row, 2)
        box = cell.findChild(QCheckBox) if cell else None
        if box is not None:
            box.setChecked(True)

    @staticmethod
    def _frame_y(window, x0, x1):
        pp = window._plot_panel
        t = getattr(pp, "_last_plot_time", None)
        a = getattr(pp, "_last_plot_asymmetry", None)
        if t is None or a is None or not len(t):
            return
        t = np.asarray(t, float)
        a = np.asarray(a, float)
        m = (t >= x0) & (t <= x1)
        if not np.any(m):
            return
        lo, hi = float(np.nanmin(a[m])), float(np.nanmax(a[m]))
        pad = 0.12 * (hi - lo or 1.0)
        pp.set_view_limits(x0, x1, lo - pad, hi + pad)


# ══════════════════════════════════════════════════════════════════════════
# 2. Concentration comparison — Mu relaxation faster with [x]
# ══════════════════════════════════════════════════════════════════════════
class MaleicConcentrationScenario(CorpusScenario):
    name = "corpus_maleic_concentration"
    description = (
        "Baseline-subtracted 2 G Mu oscillation for deox water, half and full "
        "maleic-acid concentration: the muonium precession damps visibly faster "
        "as [x] rises — the reaction consuming Mu."
    )
    example = EXAMPLE
    size = (1120, 640)
    requires_fit = True

    def capture(self, ctx) -> Path:
        from asymmetry.gui.widgets.mpl_canvas import create_canvas

        a_mu, phase, _ = _anchor()
        show = [
            (78251, 0.0, "deox. water  [x] = 0", tokens.ACCENT),
            (78277, 2.0, "half  [x] = 2", tokens.WARN),
            (78257, 4.0, "full  [x] = 4", tokens.ACCENT_RED),
        ]
        t_lo, t_hi = 0.2, 3.5
        grid = np.linspace(t_lo, t_hi, 800)
        figure, canvas = create_canvas(layout="tight")
        axes = figure.add_subplot(111)
        for run, _x, label, colour in show:
            ds = load_corpus_datasets([_rel(run)])[0]
            lam, lam_err, result, _ = _fit_mu(ds, a_mu=a_mu, phase=phase, seed_lam=1.0)
            # Faint binned baseline-subtracted data behind the smooth fit.
            t, osc = _baseline_subtracted(ds, result, rebin_factor=6)
            sel = (t >= t_lo) & (t <= t_hi)
            axes.plot(t[sel], osc[sel], color=colour, linewidth=0.7, alpha=0.32)
            # Bold fitted Mu component: same A_Mu/phase/frequency, λ_Mu the only
            # difference — so the three curves start together and damp apart.
            curve = a_mu * np.exp(-lam * grid) * np.cos(2 * np.pi * FMU * grid + phase)
            axes.plot(
                grid,
                curve,
                color=colour,
                linewidth=2.0,
                label=f"{label}   λ_Mu = {lam:.2f} µs⁻¹",
            )
        axes.axhline(0.0, color=tokens.TEXT_DIM, linewidth=0.7)
        axes.set_xlabel("Time (µs)")
        axes.set_ylabel("Mu asymmetry (baseline-subtracted, %)")
        axes.set_title(
            "Muonium precession at 2 G — faster relaxation with maleic-acid concentration"
        )
        axes.set_xlim(t_lo, t_hi)
        axes.legend(loc="upper right", fontsize="small", framealpha=0.92)

        canvas.resize(*self.size)
        canvas.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        canvas.show()
        _pump(200)
        canvas.draw()
        out = _grab(canvas, self.name, ctx.output_dir)
        canvas.close()
        canvas.deleteLater()
        _pump(40)
        return out


# ══════════════════════════════════════════════════════════════════════════
# 3. HEADLINE — λ_Mu vs [x] with a manual concentration column → slope = k_Mu
# ══════════════════════════════════════════════════════════════════════════
class MaleicKmuTrendScenario(CorpusScenario):
    name = "corpus_maleic_kmu_trend"
    description = (
        "Headline: per-run Mu relaxation rate λ_Mu vs relative maleic-acid "
        "concentration (a manual fit-table column — concentration is not in the "
        "file metadata), fitted with λ_Mu = λ₀ + k_Mu·[x]; the slope is k_Mu."
    )
    example = EXAMPLE
    size = (1180, 720)
    requires_fit = True

    def __init__(self) -> None:
        super().__init__()
        self._summary: dict[str, float] = {}

    def build(self) -> QWidget:
        from asymmetry.core.fitting.parameter_models import (
            ModelFitRange,
            ParameterCompositeModel,
            ParameterModelFit,
            fit_parameter_model,
        )
        from asymmetry.core.fitting.parameters import Parameter, ParameterSet
        from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel

        a_mu, phase, lam0 = _anchor()
        xs: list[float] = []
        lams: list[float] = []
        errs: list[float] = []
        rows = []
        for run, conc, label in _CONC_SERIES:
            if run == _ANCHOR_RUN:
                lam, lam_err = lam0, 0.02
            else:
                ds = load_corpus_datasets([_rel(run)])[0]
                lam, lam_err, _res, _ = _fit_mu(ds, a_mu=a_mu, phase=phase, seed_lam=1.0)
            xs.append(conc)
            lams.append(lam)
            errs.append(lam_err)
            rows.append(
                {
                    "run_number": run,
                    "run_label": f"{label}",
                    "field": 2.0,
                    "temperature": 290.0,
                    "values": {"Lambda": lam},
                    "errors": {"Lambda": lam_err},
                    "model_name": "Relaxing Mu oscillation (2 G)",
                    "custom_values": {_CONC_KEY: f"{conc:.2f}"},
                }
            )
        x = np.array(xs)
        y = np.array(lams)
        yerr = np.array(errs)

        # Straight-line trend λ_Mu = λ₀ + k_Mu·[x] via the real trend minimiser
        # (the panel's Model Fit "Linear": m = k_Mu, b = λ₀).
        model = ParameterCompositeModel(["Linear"])
        seed = ParameterSet([Parameter("m", 0.6), Parameter("b", float(y.min()))])
        result = fit_parameter_model(
            x, y, yerr, model, seed, x_min=float(x.min()), x_max=float(x.max())
        )
        if not result.success:
            raise RuntimeError("λ_Mu(conc) linear trend fit did not converge")
        self._summary = {
            "k_Mu": float(result.parameters["m"].value),
            "lambda0": float(result.parameters["b"].value),
        }

        batch_id = "maleic-kmu"
        panel = FitParametersPanel()
        panel.set_custom_x_fields([("Maleic conc (rel. units)", _CONC_KEY)])
        panel.load_representation_series(
            [(batch_id, "λ_Mu vs maleic concentration", rows)],
            select_id=batch_id,
        )
        idx = panel._x_combo.findData(_CONC_KEY)
        if idx >= 0:
            panel._x_combo.setCurrentIndex(idx)
        _process_events_for(milliseconds=80)

        fit_range = ModelFitRange(
            x_min=float(x.min()),
            x_max=float(x.max()),
            model=model,
            parameters=result.parameters,
            result=result,
        )
        panel._model_fits["Lambda"] = ParameterModelFit(
            parameter_name="Lambda",
            x_key=_CONC_KEY,
            ranges=[fit_range],
            active=True,
        )
        panel._sync_active_group_state()
        panel._refresh_model_fit_button_labels()
        _process_events_for(milliseconds=80)
        return panel

    def settle(self, widget: QWidget) -> None:
        _process_events_for(milliseconds=150)
        widget._refresh_plot()
        _pump_until(
            lambda: (
                not widget._trend_curve_compute_active
                and widget._precomputed_trend_curves is not None
            ),
            timeout_ms=20_000,
        )
        _process_events_for(milliseconds=200)


# ══════════════════════════════════════════════════════════════════════════
# 4. Arrhenius — λ_Mu(T) across the full-concentration temperature scan
# ══════════════════════════════════════════════════════════════════════════
R_GAS = 8.314  # J mol⁻¹ K⁻¹ — Arrhenius slope → E_a


def _build_maleic_arrhenius_fit(T, lam, lam_err):  # noqa: N803 (physics symbol)
    """Fit ``Linear`` to the *transformed* Arrhenius line ln λ_Mu vs 1/T.

    Reproduces the panel's Model-Fit dialog once the axes are set to Y→``ln x``
    and X→``reciprocal``: transform (T, λ) through the real :class:`AxisTransform`
    presets — propagating λ's error to σ(ln λ)=σ(λ)/λ — then fit ``Linear``. The
    slope is −E_a/R. Unlike the LLZ ν(T) case, no baseline subtraction is needed
    here: λ_Mu(T) rises monotonically, so the plain ``ln x`` preset linearises it
    directly. Returns ``(ParameterModelFit, summary)``.
    """
    from asymmetry.core.fitting.axis_transforms import AxisTransform
    from asymmetry.core.fitting.parameter_models import (
        ModelFitRange,
        ParameterCompositeModel,
        ParameterModelFit,
        fit_parameter_model,
    )
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    y_vals, y_err = AxisTransform.preset("log").apply(
        np.asarray(lam, float), np.asarray(lam_err, float)
    )
    x_vals, _ = AxisTransform.preset("reciprocal").apply(np.asarray(T, float))

    model = ParameterCompositeModel(["Linear"])
    seed = ParameterSet(
        [
            Parameter(name="m", value=-900.0, min=-1.0e6, max=1.0e6),  # −E_a/R  [K]
            Parameter(name="b", value=4.0, min=-50.0, max=50.0),
        ]
    )
    result = fit_parameter_model(
        x_vals, y_vals, y_err, model, seed, x_min=float(x_vals.min()), x_max=float(x_vals.max())
    )
    if not result.success:
        raise RuntimeError("Maleic λ_Mu(T) Arrhenius Linear fit did not converge")

    slope = float(result.parameters["m"].value)
    unc = result.uncertainties or {}
    ea_kj = -slope * R_GAS / 1000.0
    ea_err_kj = float(unc.get("m", 0.0)) * R_GAS / 1000.0
    summary = {
        "slope_K": slope,
        "Ea_kJ_mol": float(ea_kj),
        "Ea_err_kJ_mol": float(ea_err_kj),
        "chi2r": float(result.reduced_chi_squared or 0.0),
    }
    fit_range = ModelFitRange(
        x_min=float(x_vals.min()),
        x_max=float(x_vals.max()),
        model=model,
        parameters=result.parameters,
        result=result,
    )
    fit = ParameterModelFit(
        parameter_name="Lambda", x_key="temperature", ranges=[fit_range], active=True
    )
    return fit, summary


class MaleicArrheniusScenario(CorpusScenario):
    name = "corpus_maleic_arrhenius"
    description = (
        "Arrhenius plot of the full-concentration Mu relaxation rate λ_Mu(T) "
        "(278–358 K, 2 G) in the real trending panel: Y → ln λ_Mu, X → 1/T "
        "(axis transforms), a Linear model fit whose slope gives E_a ≈ 7.3 "
        "kJ/mol (a lower bound — see NOTES)."
    )
    example = EXAMPLE
    size = (1180, 720)
    requires_fit = True

    def __init__(self) -> None:
        super().__init__()
        self._summary: dict[str, float] = {}

    def build(self) -> QWidget:
        from asymmetry.core.fitting.axis_transforms import AxisTransform
        from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel

        a_mu, phase, _ = _anchor()
        temps: list[float] = []
        lams: list[float] = []
        errs: list[float] = []
        seed = 2.5
        for run, temp in _FULL_TSCAN:
            ds = load_corpus_datasets([_rel(run)])[0]
            lam, lam_err, _res, _ = _fit_mu(ds, a_mu=a_mu, phase=phase, seed_lam=max(seed, 0.3))
            temps.append(temp)
            lams.append(lam)
            errs.append(lam_err)
            seed = lam
        T = np.array(temps)  # noqa: N806 (physics symbol)
        lam = np.array(lams)
        lam_err = np.array(errs)

        fit, self._summary = _build_maleic_arrhenius_fit(T, lam, lam_err)

        batch_id = "maleic-arrhenius"
        row_dicts = [
            {
                "run_number": run,
                "run_label": f"{int(temp)} K",
                "field": 2.0,
                "temperature": float(temp),
                "values": {"Lambda": float(lams[i])},
                "errors": {"Lambda": float(errs[i])},
                "model_name": "Relaxing Mu oscillation (2 G)",
            }
            for i, (run, temp) in enumerate(_FULL_TSCAN)
        ]

        panel = FitParametersPanel()
        panel.load_representation_series(
            [(batch_id, "λ_Mu(T) — full conc.", row_dicts)], select_id=batch_id
        )
        idx = panel._x_combo.findData("temperature")
        if idx >= 0:
            panel._x_combo.setCurrentIndex(idx)

        panel._set_axis_transform("y", AxisTransform.preset("log"))
        panel._set_axis_transform("x", AxisTransform.preset("reciprocal"))

        panel._model_fits["Lambda"] = fit
        panel._model_fit_transform_sig["Lambda"] = panel._transform_signature()
        panel._sync_active_group_state()
        panel._refresh_model_fit_button_labels()
        _process_events_for(milliseconds=80)
        return panel

    def settle(self, widget: QWidget) -> None:
        _process_events_for(milliseconds=150)
        widget._refresh_plot()
        _pump_until(
            lambda: (
                not widget._trend_curve_compute_active
                and widget._precomputed_trend_curves is not None
            ),
            timeout_ms=20_000,
        )
        _process_events_for(milliseconds=200)


register(MaleicMuPrecessionScenario())
register(MaleicConcentrationScenario())
register(MaleicKmuTrendScenario())
register(MaleicArrheniusScenario())
