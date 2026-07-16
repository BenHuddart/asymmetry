"""Corpus scenarios — Dynamics in a magnetic plateau system (Ca₃Co₂O₆).

Drives the Asymmetry GUI through the WiMDA muon-school "magnetic decoupling"
example on the **real HiFi HDF4 NeXus ``.nxs`` corpus files** (runs 9023–9051):
a TF20 calibration (9023) plus a 15 K longitudinal-field decoupling scan from
zero field to 3.8 T (9031–9051). Ca₃Co₂O₆ is a frustrated Ising-chain magnet
showing a partial 1/3 magnetization plateau; LF-µSR decouples the muon from the
static vs **dynamic** internal fields inside that plateau.

The spec is the example's ``GROUND_TRUTH.md`` (Baker, Lord & Prabhakaran,
*J. Phys.: Condens. Matter* **23**, 306001 (2011), arXiv:1105.2200 — the same
data/campaign, author-matched in the NeXus headers, GT §11).

Workflow (GT §4): fit each run's spin polarization with a single exponential
**P_z(t) = A·exp(−λt) + bg** to extract the relaxation rate λ(B); build the
λ(µ₀H) trend (paper Fig. 2(a)); then linearise via Redfield's equation
**λ = 2γ_µ²Δ²τ / (1 + γ_µ²B²τ²)** so that **λ⁻¹ is exactly linear in µ₀²H²**
across the plateau (paper Fig. 2(b)). The slope τ/(2Δ²) and intercept
1/(2γ_µ²Δ²τ) of that line recover the **static field-distribution width
Δ = 40.6(3) mT** and the **fluctuation time τ = 880(30) ps** (GT §6).

Scenarios registered:

* ``corpus_plateau_lf_overlay`` — raw LF spectra at ZF / 0.5 / 1.5 / 3.5 T
  overlaid: the relaxation flattens (and the observed asymmetry recovers) as the
  field decouples the muon — the qualitative decoupling picture.
* ``corpus_plateau_exp_fit`` — converged single-exponential fit on one mid-field
  run (1.0 T, 9044): λ ≈ 1.33 µs⁻¹.
* ``corpus_plateau_lambda_field`` — λ(µ₀H) trend across the scan (paper
  Fig. 2(a)): the three-regime falloff from ~9 µs⁻¹ near ZF to ~0.3 µs⁻¹ at
  3.8 T.
* ``corpus_plateau_redfield`` — **headline**: λ⁻¹ vs µ₀²H² with the linear
  Redfield fit over the 0.5–3.6 T plateau, recovering Δ and τ (paper Fig. 2(b)).

See ``NOTES_plateau.md`` for run selection, the λ(B) table and the Δ/τ
comparison against the paper.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from ._corpus import CorpusScenario, _process_events_for, load_corpus_datasets, register

EXAMPLE = "Magnetism/Dynamics in a magnetic plateau system"
_DATA = EXAMPLE + "/Data/HIFI0000%d.nxs"

# Muon gyromagnetic ratio used by the paper (GT §6): γ_µ = 2π × 135.5 MHz/T,
# i.e. 2π × 135.5 in units of µs⁻¹ T⁻¹.
GAMMA_MU = 2.0 * np.pi * 135.5  # µs⁻¹ T⁻¹
GAMMA_MU_GAUSS = GAMMA_MU / 1.0e4  # µs⁻¹ G⁻¹ (field stored/plotted in gauss)

# Fit window for the per-run exponential (µs). The late-time bins get noisy as
# the counts vanish (F–B asymmetry diverges past ~16 µs), so cap there.
FIT_TMAX = 16.0

# 15 K LF scan run → applied longitudinal field in tesla (GT §3). ZF runs
# 9031–9034 are all 0 T; a single ZF representative (9031) is enough. Runs
# 9024–9030 are thermally unsettled cooldown points (GT §9) — excluded.
SCAN_FIELDS_T: dict[int, float] = {
    9031: 0.0,
    9035: 0.1,
    9036: 0.2,
    9037: 0.3,
    9038: 0.4,
    9039: 0.5,
    9040: 0.6,
    9041: 0.7,
    9042: 0.8,
    9043: 0.9,
    9044: 1.0,
    9045: 1.5,
    9046: 2.0,
    9047: 2.5,
    9048: 2.9,
    9049: 3.2,
    9050: 3.5,
    9051: 3.8,
}

# Runs shown in the raw-spectra overlay (ZF / 0.5 / 1.5 / 3.5 T) — the
# decoupling progression the job calls for.
_OVERLAY_RUNS = [9031, 9039, 9045, 9050]

# The plateau fit window is 0.5–3.6 T (GT §4/§6): runs 9039 (0.5 T) through
# 9050 (3.5 T). 9051 (3.8 T) is in the saturated phase beyond the plateau and is
# shown but excluded from the Redfield line fit.
_PLATEAU_RUNS = [9039, 9044, 9045, 9046, 9047, 9048, 9049, 9050]


def _rel(run: int) -> str:
    return _DATA % run


def _exp_model():
    from asymmetry.core.fitting.composite import CompositeModel

    return CompositeModel(["Exponential", "Constant"])  # A·exp(−λt) + A_bg


def _fit_lambda(dataset, lam_seed: float):
    """Fit ``Exponential + Constant`` to one LF run; return (λ, λ_err).

    ``P_z(t) = A·exp(−λt) + A_bg`` over 0–16 µs via the core ``FitEngine`` (the
    same engine the single-fit panel drives). The large additive baseline
    ``A_bg`` (which grows with field as decay positrons spiral, GT §4) is fitted
    from the late-time level; the amplitude is seeded on the percent scale the
    loader returns.
    """
    from asymmetry.core.fitting.engine import FitEngine
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    a = np.asarray(dataset.asymmetry, dtype=float)
    params = ParameterSet(
        [
            Parameter("A_1", 8.0, min=0.0),
            Parameter("Lambda", lam_seed, min=0.0),
            Parameter("A_bg", float(np.nanmean(a[-200:]))),
        ]
    )
    result = FitEngine().fit(dataset, _exp_model().function, params, t_min=0.0, t_max=FIT_TMAX)
    lam = abs(result.parameters["Lambda"].value)
    err = float((result.uncertainties or {}).get("Lambda", 0.02) or 0.02)
    return lam, err


def _lambda_of_field(runs):
    """Fit every run in *runs* (ascending field), warm-starting λ downward.

    Real LF data walks to spurious minima from a cold seed on the low-field runs
    (GT §9 pulse-width / asymmetry-suppression regime), so λ is carried forward
    from the previous, higher-relaxation run (README lesson: warm-start in field
    order). Returns arrays (B_tesla, λ, λ_err).
    """
    fields, lams, errs = [], [], []
    seed = 9.0
    for run in runs:
        ds = load_corpus_datasets([_rel(run)])[0]
        lam, err = _fit_lambda(ds, seed)
        fields.append(SCAN_FIELDS_T[run])
        lams.append(lam)
        errs.append(err)
        seed = max(lam * 0.9, 0.2)
    return np.asarray(fields), np.asarray(lams), np.asarray(errs)


def _redfield_from_line(slope: float, intercept: float):
    """Solve the linearised Redfield form for (Δ [G], τ [µs]).

    λ⁻¹ = [1 + γ_µ²(µ₀H)²τ²] / (2γ_µ²Δ²τ)  ⇒
      intercept = 1/(2γ_µ²Δ²τ),  slope = τ/(2Δ²)  (GT §4).
    slope/intercept = γ_µ²τ² ⇒ τ = √(slope/intercept)/γ_µ; Δ = √(τ/(2·slope)).

    Field is stored/plotted in **gauss** (the app's native ``B (G)`` axis), so
    the line is fitted in B² [G²]: the slope is in µs G⁻², and γ_µ is taken in
    µs⁻¹ G⁻¹. Δ comes out in gauss. The gauss and tesla forms give identical
    Δ (in mT) and τ (in µs) — the fit units cancel out of the physics.
    """
    tau = np.sqrt(slope / intercept) / GAMMA_MU_GAUSS  # µs
    delta = np.sqrt(tau / (2.0 * slope))  # G
    return delta, tau


def _build_redfield_linear_fit(fields_g, lam, lam_err):
    """Fit ``Linear`` to the *transformed* Redfield plateau (1/λ vs B², G units).

    Reproduces exactly what the panel's Model-Fit dialog does once the axes are
    set to Y→``reciprocal`` and X→``square``: transform the (field, λ) arrays
    through the real :class:`AxisTransform` presets — propagating λ's error to
    σ(1/λ)=σ(λ)/λ² — then fit ``Linear`` on those transformed coordinates with
    the core trend minimiser. Returns ``(ParameterModelFit, summary)``; the fit
    is tagged with ``x_key="field"`` so the panel samples its overlay in the
    same B² domain the scatter is drawn in.

    *fields_g* are in **gauss** — the panel's native field unit (#262 normalises
    NeXus field to gauss), so the X→square axis reads correctly as ``B² (G²)``.
    The slope therefore comes out in µs G⁻²; :func:`_redfield_from_line` uses the
    matching γ_µ in µs⁻¹ G⁻¹ so Δ/τ are unchanged from the old tesla path.
    """
    from asymmetry.core.fitting.axis_transforms import AxisTransform
    from asymmetry.core.fitting.parameter_models import (
        ModelFitRange,
        ParameterCompositeModel,
        ParameterModelFit,
        fit_parameter_model,
    )
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    x2, _ = AxisTransform.preset("square").apply(np.asarray(fields_g, dtype=float))
    inv_lam, inv_err = AxisTransform.preset("reciprocal").apply(
        np.asarray(lam, dtype=float), np.asarray(lam_err, dtype=float)
    )

    model = ParameterCompositeModel(["Linear"])
    seed = ParameterSet(
        [
            # slope τ/(2Δ²) [µs G⁻²] ≈ 0.27 µs T⁻² / 1e8; intercept 1/(2γ²Δ²τ) [µs].
            Parameter("m", value=2.7e-9, min=0.0, max=1.0e-5),
            Parameter("b", value=0.45, min=0.0, max=5.0),
        ]
    )
    result = fit_parameter_model(
        x2, inv_lam, inv_err, model, seed, x_min=float(x2.min()), x_max=float(x2.max())
    )
    if not result.success:
        raise RuntimeError("Redfield Linear trend fit did not converge for the screenshot")

    slope = float(result.parameters["m"].value)
    intercept = float(result.parameters["b"].value)
    unc = result.uncertainties or {}
    delta, tau = _redfield_from_line(slope, intercept)
    summary = {
        "slope": slope,
        "slope_err": float(unc.get("m", 0.0)),
        "intercept": intercept,
        "intercept_err": float(unc.get("b", 0.0)),
        "delta_mT": float(delta * 0.1),  # 1 G = 0.1 mT
        "tau_ps": float(tau * 1e6),
        "chi2r": float(result.reduced_chi_squared or 0.0),
    }
    fit_range = ModelFitRange(
        x_min=float(x2.min()),
        x_max=float(x2.max()),
        model=model,
        parameters=result.parameters,
        result=result,
    )
    fit = ParameterModelFit(parameter_name="Lambda", x_key="field", ranges=[fit_range], active=True)
    return fit, summary


def _wait_until(predicate, *, timeout_ms: int, poll_ms: int = 30) -> None:
    elapsed = 0
    while elapsed < timeout_ms:
        if predicate():
            return
        _process_events_for(milliseconds=poll_ms)
        elapsed += poll_ms


# ---------------------------------------------------------------------------
# 1. Raw LF spectra overlay — the decoupling picture
# ---------------------------------------------------------------------------
class PlateauLfOverlayScenario(CorpusScenario):
    name = "corpus_plateau_lf_overlay"
    description = (
        "Ca₃Co₂O₆ 15 K longitudinal-field spectra at ZF / 0.5 / 1.5 / 3.5 T "
        "overlaid: exponential relaxation flattens (and the observed asymmetry "
        "recovers) as the applied field decouples the muon — the decoupling "
        "picture."
    )
    example = EXAMPLE
    size = (1500, 920)

    def build(self) -> QWidget:
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window.resizeDocks([window._dock_data_browser], [320], Qt.Orientation.Horizontal)

        datasets = load_corpus_datasets([_rel(r) for r in _OVERLAY_RUNS])
        with window._data_browser.batch_updates():
            for dataset in datasets:
                window._data_browser.add_dataset(dataset)
        run_numbers = [int(ds.run_number) for ds in datasets]
        window._data_browser.create_data_group(run_numbers, name="Ca₃Co₂O₆ 15 K LF scan")

        window._plot_panel.set_overlay_enabled(True, emit_signal=True)
        window._plot_panel.set_bunch_factor(10, emit_signal=True)
        window._data_browser._table.selectAll()
        window._on_dataset_selected(run_numbers[0])
        _process_events_for(milliseconds=100)
        # Frame the 0–12 µs analysis window: the fast-relaxing low-field traces
        # decay into their baselines within a few µs while the high-field trace
        # stays flat and high; past ~14 µs the F–B asymmetry diverges as the
        # counts vanish. Y spans the recovered baselines (ZF ~12 % → 3.5 T ~40 %).
        window._plot_panel.set_view_limits(0.0, 12.0, 0.0, 48.0)
        _process_events_for(milliseconds=80)
        return window


# ---------------------------------------------------------------------------
# 2. Converged single-exponential fit on one mid-field run (1.0 T)
# ---------------------------------------------------------------------------
class PlateauExpFitScenario(CorpusScenario):
    name = "corpus_plateau_exp_fit"
    description = (
        "Converged Exponential + Constant fit on the Ca₃Co₂O₆ 1.0 T run (9044): "
        "P_z(t) = A·exp(−λt) + bg, λ ≈ 1.33 µs⁻¹ — one point of the λ(B) trend."
    )
    example = EXAMPLE
    size = (1500, 920)
    requires_fit = True

    def build(self) -> QWidget:
        from asymmetry.gui.mainwindow import MainWindow
        from asymmetry.gui.panels.fit.tab_base import (
            _param_table_rows_by_name,
            _set_param_table_value,
        )

        window = MainWindow()
        window._on_fit()
        window.resizeDocks([window._dock_data_browser], [320], Qt.Orientation.Horizontal)

        # Run 9044 — 1.0 T, mid-plateau, full asymmetry recovered.
        datasets = load_corpus_datasets([_rel(9044)])
        for dataset in datasets:
            window._data_browser.add_dataset(dataset)
        window._on_dataset_selected(datasets[0].run_number)

        single_tab = window._fit_panel._single_tab
        single_tab._set_composite_model(_exp_model())
        _process_events_for(milliseconds=80)

        a = np.asarray(datasets[0].asymmetry, dtype=float)
        param_table = single_tab._param_table
        rows = _param_table_rows_by_name(param_table)
        seeds = {
            "A_1": 8.0,
            "Lambda": 1.3,
            "A_bg": float(np.nanmean(a[-200:])),
        }
        for pname, value in seeds.items():
            if pname in rows:
                _set_param_table_value(param_table, rows[pname], value)
        _process_events_for(milliseconds=40)

        # Fit over the 0–16 µs window (README lesson: the single-tab fit-range
        # spinbox does not commit — set it on the plot panel). The fit itself
        # runs on the unbinned data; only the display is bunched afterwards.
        window._plot_panel.set_fit_range(0.0, FIT_TMAX)
        _process_events_for(milliseconds=40)
        single_tab._run_fit()
        single_tab.wait_for_fit()
        _process_events_for(milliseconds=80)

        # Bunch the display to tame the late-time F–B fan (the asymmetry
        # diverges past ~8 µs as the counts vanish), then frame the decay:
        # 0–10 µs in time, Y snug to the data in that window so the single
        # exponential and its fitted baseline read clearly.
        window._plot_panel.set_bunch_factor(5, emit_signal=True)
        _process_events_for(milliseconds=60)
        window._plot_panel.set_view_limits(0.0, 10.0, *window._plot_panel.get_view_limits()[2:])
        _process_events_for(milliseconds=60)
        self._frame_y_to_window(window, 0.0, 10.0)
        _process_events_for(milliseconds=80)
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
# 3. λ(µ₀H) trend across the scan (paper Fig. 2(a))
# ---------------------------------------------------------------------------
class PlateauLambdaFieldScenario(CorpusScenario):
    name = "corpus_plateau_lambda_field"
    description = (
        "Ca₃Co₂O₆ relaxation rate λ vs applied longitudinal field (paper "
        "Fig. 2(a)): three regimes — rapid drop below 0.5 T, slow decrease "
        "across the 0.5–3.6 T plateau, ~constant above 3.6 T."
    )
    example = EXAMPLE
    size = (1240, 780)
    requires_fit = True

    def build(self) -> QWidget:
        from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel

        # Fit every field point 0.2 → 3.8 T. Below 0.2 T (ZF 9031, 0.1 T 9035)
        # the asymmetry is suppressed and the fast relaxation is unresolved at
        # the ISIS pulse width, so the exponential is degenerate (λ collapses to
        # a spurious flat-line minimum) and not physical — GT §4/§9.
        runs = [r for r in SCAN_FIELDS_T if SCAN_FIELDS_T[r] >= 0.2]
        runs.sort(key=lambda r: SCAN_FIELDS_T[r])
        fields_t, lam, lam_err = _lambda_of_field(runs)

        batch_id = "plateau-lambda-b"
        row_dicts = [
            {
                "run_number": runs[i],
                "run_label": f"{fields_t[i]:.1f} T",
                # Field stored in gauss (the panel's native "B (G)" axis).
                "field": float(fields_t[i] * 1e4),
                "temperature": 15.0,
                "values": {"Lambda": float(lam[i])},
                "errors": {"Lambda": float(lam_err[i])},
            }
            for i in range(len(runs))
        ]

        panel = FitParametersPanel()
        panel.load_representation_series(
            [(batch_id, "λ(B) — Ca₃Co₂O₆ 15 K", row_dicts)], select_id=batch_id
        )
        panel._sync_active_group_state()
        _process_events_for(milliseconds=80)
        return panel

    def settle(self, widget: QWidget) -> None:
        _process_events_for(milliseconds=200)
        widget._refresh_plot()
        _process_events_for(milliseconds=200)


# ---------------------------------------------------------------------------
# 4. Headline — λ⁻¹ vs µ₀²H² Redfield line (paper Fig. 2(b))
# ---------------------------------------------------------------------------
# Sub-plateau (0.4 T) and saturated (3.8 T) points: shown on the plot but
# excluded from the Redfield trend fit (they lie off the plateau line).
_REDFIELD_EXCLUDED = [9038, 9051]


class PlateauRedfieldScenario(CorpusScenario):
    name = "corpus_plateau_redfield"
    description = (
        "Headline: Ca₃Co₂O₆ λ⁻¹ vs µ₀²H² (paper Fig. 2(b)) in the real trending "
        "panel — Y axis transformed to 1/λ (reciprocal), X axis to B² (square), a "
        "Linear model fit on the 0.5–3.6 T plateau. Slope/intercept give "
        "Δ ≈ 41 mT and τ ≈ 0.93 ns (paper Δ = 40.6 mT, τ = 880 ps); the 0.4 T "
        "and 3.8 T points sit off the line, excluded from the trend."
    )
    example = EXAMPLE
    size = (1240, 760)
    requires_fit = True  # real per-run exponential fits + the linear Redfield fit

    def __init__(self) -> None:
        super().__init__()
        self._fit_summary: dict[str, float] = {}

    def build(self) -> QWidget:
        from asymmetry.core.fitting.axis_transforms import AxisTransform
        from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel

        included = list(_PLATEAU_RUNS)  # 9039–9050 (0.5–3.5 T), the plateau
        all_runs = sorted(set(included) | set(_REDFIELD_EXCLUDED), key=lambda r: SCAN_FIELDS_T[r])
        # One per-run exponential fit pass for every plotted point (ascending
        # field, warm-started downward). Field is stored in **gauss** — the
        # panel's native "B (G)" axis (#262 normalises NeXus field to gauss) — so
        # the X→square transform reads correctly as ``B² (G²)`` and the Redfield
        # slope comes out in µs G⁻². Δ/τ are identical to the old tesla path
        # (the fit units cancel out of the physics).
        fields_t, lam, lam_err = _lambda_of_field(all_runs)
        fields_g = fields_t * 1.0e4
        incl_mask = np.array([r in included for r in all_runs])

        fit, self._fit_summary = _build_redfield_linear_fit(
            fields_g[incl_mask], lam[incl_mask], lam_err[incl_mask]
        )

        batch_id = "plateau-redfield"
        row_dicts = [
            {
                "run_number": run,
                "run_label": f"{SCAN_FIELDS_T[run]:.1f} T",
                "field": float(SCAN_FIELDS_T[run] * 1.0e4),  # gauss (native B axis)
                "temperature": 15.0,
                "values": {"Lambda": float(lam[i])},
                "errors": {"Lambda": float(lam_err[i])},
                "include_in_trend": bool(incl_mask[i]),
            }
            for i, run in enumerate(all_runs)
        ]

        panel = FitParametersPanel()
        panel.load_representation_series(
            [(batch_id, "λ(B) — Ca₃Co₂O₆ 15 K", row_dicts)], select_id=batch_id
        )
        # The Redfield linearisation: reciprocal Y (1/λ), square X (B²). The
        # transform drives the plotted points, the propagated error bars, AND the
        # trend fit's coordinate system — one lens over all three.
        panel._set_axis_transform("y", AxisTransform.preset("reciprocal"))
        panel._set_axis_transform("x", AxisTransform.preset("square"))

        # Inject the Linear fit computed on the transformed plateau, tagged with
        # the active transform so its overlay is drawn (not suppressed as stale).
        panel._model_fits["Lambda"] = fit
        panel._model_fit_transform_sig["Lambda"] = panel._transform_signature()
        panel._sync_active_group_state()
        panel._refresh_model_fit_button_labels()
        _process_events_for(milliseconds=80)
        return panel

    def settle(self, widget: QWidget) -> None:
        _process_events_for(milliseconds=200)
        widget._refresh_plot()
        _wait_until(
            lambda: (
                not widget._trend_curve_compute_active
                and widget._precomputed_trend_curves is not None
            ),
            timeout_ms=20000,
        )
        _process_events_for(milliseconds=200)


register(PlateauLfOverlayScenario())
register(PlateauExpFitScenario())
register(PlateauLambdaFieldScenario())
register(PlateauRedfieldScenario())
