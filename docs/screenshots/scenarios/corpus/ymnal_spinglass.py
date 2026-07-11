"""Corpus scenarios — Spin relaxation in the spin glass Y(Mn₀.₉₅Al₀.₀₅)₂.

Worked example ``Magnetism/Spin Glass YMnAl`` (WiMDA muon school corpus),
driving Asymmetry through the longitudinal-field µSR study of the topologically
frustrated cubic-Laves spin glass Y(Mn₀.₉₅Al₀.₀₅)₂. Data are 20 MuSR runs
(ISIS NeXus-v1 HDF4 ``.nxs``): a TF 20 G calibration (24563) plus the LF 110 G
temperature series 24573–24591. Same experiment as M. T. F. Telling *et al.*,
Phys. Rev. B **85**, 184416 (2012). See the example's ``GROUND_TRUTH.md``.

This is the corpus's **stretched-exponential / spin-glass showcase**. The LF
110 G relaxation is fitted per run with the guide's model (GROUND_TRUTH §4)

    G(t) = A · exp[−(λ t)^β] + A_bg

following the prescribed parameter-fixing protocol: fix the background A_bg
from the 90 K run, fix the full asymmetry A from the 280 K run (β = 1), then
batch-fit the series with β free. Approaching the spin-glass freezing
transition T_g ≈ 88 K the dynamic relaxation rate λ(T) diverges (critical
slowing-down) and the lineshape crosses over from simple-exponential
(β → 1 at 280 K) to the concentrated-spin-glass value β → 1/3 (GROUND_TRUTH
§6/§10, Fig 6c/d).

Scenarios registered:

* ``corpus_ymnal_spectra``       — LF 110 G spectra overlaid 280 K → 85 K:
  relaxation grows dramatically toward T_g.
* ``corpus_ymnal_stretched_fit`` — converged A·exp[−(λt)^β]+bg fit on a run
  near the transition (95 K), β clearly < 1 in the parameter table.
* ``corpus_ymnal_lambda_t``      — the headline: λ(T) with the critical rise
  toward T_g (log y) and the fitted CriticalDivergence trend.
* ``corpus_ymnal_beta_t``        — β(T) falling from 1 toward 1/3 as T → T_g.

``requires_fit = True`` on every scenario that runs real iminuit fits at
capture time (the three fit-bearing scenarios; the spectrum overlay is a plain
render).
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QWidget

from .._base import _process_events_for
from ._corpus import CorpusScenario, load_corpus_datasets, register

EXAMPLE = "Magnetism/Spin Glass YMnAl"
_DATA = "Magnetism/Spin Glass YMnAl/data/MUSR000%d.nxs"

# Series model (GROUND_TRUTH §4): stretched exponential + flat background,
#   G(t) = A·exp[−(λ t)^β] + A_bg.  The composite renames the amplitude A_1.
_SERIES_MODEL = (["StretchedExponential", "Constant"], ["+"])

# Fit window from the source thesis (GROUND_TRUTH §3/§10: 0.5 ≤ t ≤ 12 µs).
_T_MIN, _T_MAX = 0.5, 12.0

# LF 110 G paramagnetic-regime series, run → measured T (K) from the .nxs
# headers (GROUND_TRUTH §3). The guide's batch window is 24580–24590
# (100–280 K); we extend down to 90 K (24576) to approach T_g. Runs below
# ~90 K (24575/24574/24573 at 85/80/75 K) sit at/below the frozen transition,
# outside the guide's paramagnetic series, so they are excluded from the trend.
_SERIES: list[tuple[int, float]] = [
    (24576, 90.0),
    (24587, 91.0),
    (24578, 92.5),
    (24577, 95.0),
    (24588, 97.5),
    (24580, 100.0),
    (24581, 111.0),
    (24582, 120.0),
    (24583, 135.0),
    (24584, 150.0),
    (24585, 180.0),
    (24586, 221.0),
    (24590, 280.0),
]

# Background-fix run (90 K) and A-fix run (280 K) — GROUND_TRUTH §4 protocol.
_BG_RUN = 24576  # 90 K
_A_RUN = 24590  # 280 K

# Spectrum-evolution overlay: a clean descending-T spread through the series
# down to just below T_g, to show relaxation growing toward the transition.
_SPECTRA_RUNS = [24590, 24584, 24582, 24580, 24577, 24576, 24575]

# Single-fit showcase: a run near the transition where β is clearly < 1.
_FIT_RUN = 24577  # 95 K


# --------------------------------------------------------------------------- #
#  Core-engine fitting helpers (drive the same FitEngine the GUI uses).
# --------------------------------------------------------------------------- #
def _fit_run(run, seed, *, fixed=()):
    """Fit the stretched-exp + bg model to one run; return (values, uncertainties).

    ``seed`` maps every parameter name to a starting value; ``fixed`` names the
    parameters held constant. Positive physical parameters (A_1, Lambda, beta)
    are bounded ≥ 0 and beta ≤ 2, matching the GUI's default bounds.
    """
    from asymmetry.core.fitting.composite import CompositeModel
    from asymmetry.core.fitting.engine import FitEngine
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    model = CompositeModel(_SERIES_MODEL[0], operators=_SERIES_MODEL[1])
    params = []
    for name in model.param_names:
        lo = 0.0 if name in ("A_1", "Lambda", "beta") else None
        hi = 2.0 if name == "beta" else None
        p = Parameter(name, seed[name], min=lo, max=hi)
        if name in fixed:
            p.fixed = True
        params.append(p)
    result = FitEngine().fit(
        load_corpus_datasets([_DATA % run])[0],
        model.function,
        ParameterSet(params),
        t_min=_T_MIN,
        t_max=_T_MAX,
    )
    values = {p.name: p.value for p in result.parameters}
    return values, (result.uncertainties or {})


def _calibrate_amplitudes():
    """Run the GROUND_TRUTH §4 protocol → (A, A_bg) fixed for the batch fit.

    1. Fit 90 K (24576) free → take A_bg (background level).
    2. Fit 280 K (24590) with β = 1 and A_bg fixed → take A (full t=0 asymmetry).
    """
    bg_fit, _ = _fit_run(_BG_RUN, {"A_1": 22.0, "Lambda": 0.9, "beta": 0.6, "A_bg": 2.0})
    a_bg = bg_fit["A_bg"]
    a_fit, _ = _fit_run(
        _A_RUN,
        {"A_1": 22.0, "Lambda": 0.01, "beta": 1.0, "A_bg": a_bg},
        fixed=("beta", "A_bg"),
    )
    return a_fit["A_1"], a_bg


def _fit_series():
    """Batch-fit λ(T), β(T) over the series (A, A_bg fixed; β free).

    Runs are fitted in descending-temperature order, warm-starting λ and β from
    the previous (higher-T) fit so each lands in the correct minimum — the
    stretched exponential's λ/β anticorrelation makes cold seeds unreliable on
    real data (wave-1 lesson). Returns arrays sorted ascending in T.
    """
    a, a_bg = _calibrate_amplitudes()
    seed = {"A_1": a, "Lambda": 0.01, "beta": 1.0, "A_bg": a_bg}
    rows: list[tuple[float, float, float, float, float]] = []
    for run, temp in sorted(_SERIES, key=lambda rt: -rt[1]):
        values, unc = _fit_run(run, seed, fixed=("A_1", "A_bg"))
        lam = abs(values["Lambda"])
        beta = values["beta"]
        rows.append(
            (
                temp,
                lam,
                beta,
                float(unc.get("Lambda", np.nan)),
                float(unc.get("beta", np.nan)),
            )
        )
        seed = {"A_1": a, "Lambda": lam, "beta": beta, "A_bg": a_bg}
    rows.sort(key=lambda r: r[0])
    temps = np.array([r[0] for r in rows])
    lam = np.array([r[1] for r in rows])
    beta = np.array([r[2] for r in rows])
    lam_err = np.array([r[3] for r in rows])
    beta_err = np.array([r[4] for r in rows])
    return temps, lam, beta, lam_err, beta_err


def _trend_panel(temps, values, errors, param_name, title):
    """Build a FitParametersPanel showing one fitted parameter vs temperature."""
    from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel

    order = np.argsort(temps)
    row_dicts = [
        {
            "run_number": int(1000 + i),
            "run_label": f"{temps[j]:.1f} K",
            "field": 110.0,
            "temperature": float(temps[j]),
            "values": {param_name: float(values[j])},
            "errors": {param_name: float(errors[j]) if np.isfinite(errors[j]) else 0.0},
        }
        for i, j in enumerate(order)
    ]
    panel = FitParametersPanel()
    panel.load_representation_series(
        [(f"ymnal-{param_name}", title, row_dicts)],
        select_id=f"ymnal-{param_name}",
    )
    _process_events_for(milliseconds=80)
    return panel


def _wait_until(predicate, *, timeout_ms: int, poll_ms: int = 40) -> None:
    elapsed = 0
    while elapsed < timeout_ms:
        if predicate():
            return
        _process_events_for(milliseconds=poll_ms)
        elapsed += poll_ms


def _set_fixed(table, name: str) -> None:
    """Tick the Fix checkbox for parameter *name* in a single-fit param table."""
    from asymmetry.gui.panels.fit.tab_base import _param_table_rows_by_name

    rows = _param_table_rows_by_name(table)
    if name not in rows:
        return
    fix_widget = table.cellWidget(rows[name], table.COL_FIX)
    checkbox = fix_widget.findChild(QCheckBox) if fix_widget else None
    if checkbox is not None:
        checkbox.setChecked(True)


# --------------------------------------------------------------------------- #
#  1. Spectrum evolution — LF 110 G relaxation growing toward T_g.
# --------------------------------------------------------------------------- #
class YmnalSpectraScenario(CorpusScenario):
    name = "corpus_ymnal_spectra"
    description = (
        "LF 110 G Y(Mn,Al)₂ spectra overlaid from 280 K down to 85 K: the "
        "muon relaxation grows dramatically as the spin-glass transition "
        "T_g ≈ 88 K is approached (critical slowing-down)."
    )
    example = EXAMPLE
    size = (1500, 920)

    def build(self) -> QWidget:
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window.resizeDocks([window._dock_data_browser], [340], Qt.Orientation.Horizontal)

        datasets = load_corpus_datasets([_DATA % r for r in _SPECTRA_RUNS])
        with window._data_browser.batch_updates():
            for dataset in datasets:
                window._data_browser.add_dataset(dataset)
        run_numbers = [int(ds.run_number) for ds in datasets]
        window._data_browser.create_data_group(run_numbers, name="Y(Mn,Al)₂ LF 110 G — 280→85 K")

        window._plot_panel.set_overlay_enabled(True, emit_signal=True)
        window._data_browser._table.selectAll()
        window._on_dataset_selected(run_numbers[0])
        _process_events_for(milliseconds=120)

        # The relaxation develops over the first several µs; beyond ~10 µs the
        # LF asymmetry's late-time error fan swamps the contrast. Frame 0–10 µs
        # so the flat 280 K trace and the fast near-T_g decay both read.
        x_min, x_max, y_min, y_max = window._plot_panel.get_view_limits()
        window._plot_panel.set_view_limits(0.0, 10.0, y_min, y_max)
        _process_events_for(milliseconds=80)
        return window


# --------------------------------------------------------------------------- #
#  2. Stretched-exponential fit near the transition — β clearly < 1.
# --------------------------------------------------------------------------- #
class YmnalStretchedFitScenario(CorpusScenario):
    name = "corpus_ymnal_stretched_fit"
    description = (
        "Converged A·exp[−(λt)^β] + background fit on the Y(Mn,Al)₂ 95 K LF "
        "110 G run: near T_g the exponent β ≈ 0.6 sits well below 1 — the "
        "stretched (glassy) lineshape."
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

        # Calibrate A / A_bg via the guide's protocol so the single fit shows
        # the same well-determined β the batch fit produces.
        a, a_bg = _calibrate_amplitudes()

        window = MainWindow()
        window._on_fit()
        window.resizeDocks([window._dock_data_browser], [320], Qt.Orientation.Horizontal)

        datasets = load_corpus_datasets([_DATA % _FIT_RUN])
        self.add_to_browser(window, datasets)
        window._on_dataset_selected(datasets[0].run_number)
        _process_events_for(milliseconds=80)

        single_tab = window._fit_panel._single_tab
        single_tab._set_composite_model(
            CompositeModel(_SERIES_MODEL[0], operators=_SERIES_MODEL[1])
        )
        _process_events_for(milliseconds=80)

        table = single_tab._param_table
        rows = _param_table_rows_by_name(table)
        seeds = {"A_1": a, "Lambda": 0.1, "beta": 0.7, "A_bg": a_bg}
        for name, value in seeds.items():
            if name in rows:
                _set_param_table_value(table, rows[name], value)
        # Guide protocol: A and A_bg fixed; λ and β are the free glassy params.
        _set_fixed(table, "A_1")
        _set_fixed(table, "A_bg")
        _process_events_for(milliseconds=60)

        single_tab._run_fit()
        single_tab.wait_for_fit()
        _process_events_for(milliseconds=80)

        # Frame the fit window (0.5–12 µs) plus a little context so the decay
        # and the model overlay both read without the late-time error fan.
        x_min, x_max, y_min, y_max = window._plot_panel.get_view_limits()
        window._plot_panel.set_view_limits(0.0, 12.0, y_min, y_max)
        _process_events_for(milliseconds=80)
        return window


# --------------------------------------------------------------------------- #
#  3. λ(T) divergence — the headline, with the critical-divergence trend fit.
# --------------------------------------------------------------------------- #
class YmnalLambdaTScenario(CorpusScenario):
    name = "corpus_ymnal_lambda_t"
    description = (
        "Headline: dynamic muon relaxation rate λ(T) for Y(Mn,Al)₂ LF 110 G, "
        "diverging on a log axis as the spin-glass transition T_g ≈ 88 K is "
        "approached, with the fitted CriticalDivergence trend a·|T−T_c|^(−ν)."
    )
    example = EXAMPLE
    size = (1240, 760)
    requires_fit = True

    def __init__(self) -> None:
        super().__init__()
        self._fit_summary: dict[str, float] = {}

    def build(self) -> QWidget:
        from asymmetry.core.fitting.parameter_models import (
            ModelFitRange,
            ParameterCompositeModel,
            ParameterModelFit,
            fit_parameter_model,
        )
        from asymmetry.core.fitting.parameters import Parameter, ParameterSet

        temps, lam, _beta, lam_err, _beta_err = _fit_series()
        self._temps, self._lam = temps, lam

        panel = _trend_panel(temps, lam, lam_err, "Lambda", "λ(T) — Y(Mn,Al)₂ LF 110 G")

        # Critical-divergence trend fit λ(T) = a·|T−T_c|^(−ν) + c. The transition
        # sits below the lowest data point (90 K), so T_c is bounded < 90 K.
        # NOTE (GROUND_TRUTH §10): this generic form's exponent ν is NOT the
        # paper's Eq-7 γ — compare T_c, and compare ν only against the matching
        # functional form. We recover T_c a few K below the paper's 88.2(2) K
        # and ν ≈ 0.8 (close to the thesis γ = 0.80).
        model = ParameterCompositeModel(["CriticalDivergence"])
        seeds = {"a": 1.0, "Tc": 85.0, "nu": 0.9, "c": 0.005}
        params = []
        for name in model.param_names:
            p = Parameter(name=name, value=float(seeds[name]))
            if name == "Tc":
                p.min, p.max = 50.0, 89.5
            params.append(p)
        # Weight the trend fit uniformly. The per-point λ uncertainties from the
        # time-domain fits shrink steeply toward low T, over-weighting the
        # divergent tail and stalling the minimiser; equal weights (the way a
        # digitised λ(T) would be read) converge cleanly to T_c a few K below
        # the paper's 88.2 K. The *plotted* error bars still carry the real
        # per-point uncertainties (from the series row dicts).
        err = np.full(temps.shape, 0.003)
        result = fit_parameter_model(
            temps,
            lam,
            err,
            model,
            ParameterSet(params),
            x_min=float(temps.min()),
            x_max=float(temps.max()),
        )
        self._fit_summary = {n: float(result.parameters[n].value) for n in model.param_names}

        fit_range = ModelFitRange(
            x_min=float(temps.min()),
            x_max=float(temps.max()),
            model=model,
            parameters=result.parameters,
            result=result,
        )
        panel._model_fits["Lambda"] = ParameterModelFit(
            parameter_name="Lambda",
            x_key="temperature",
            ranges=[fit_range],
            active=True,
        )
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
            timeout_ms=25000,
        )
        _process_events_for(milliseconds=200)
        # Log-y makes the ~10× rise toward T_g read as a clean divergence.
        axes = list(widget._figure.axes)
        if axes:
            axes[0].set_yscale("log")
            widget._canvas.draw()
        _process_events_for(milliseconds=150)


# --------------------------------------------------------------------------- #
#  4. β(T) fall — stretched exponent dropping from 1 toward 1/3 at T_g.
# --------------------------------------------------------------------------- #
class YmnalBetaTScenario(CorpusScenario):
    name = "corpus_ymnal_beta_t"
    description = (
        "Stretching exponent β(T) for Y(Mn,Al)₂ LF 110 G: β = 1 (simple "
        "exponential) at 280 K falling toward the concentrated-spin-glass "
        "value β → 1/3 as T → T_g ≈ 88 K."
    )
    example = EXAMPLE
    size = (1240, 760)
    requires_fit = True

    def build(self) -> QWidget:
        temps, _lam, beta, _lam_err, beta_err = _fit_series()
        self._temps, self._beta = temps, beta
        return _trend_panel(temps, beta, beta_err, "beta", "β(T) — Y(Mn,Al)₂ LF 110 G")

    def settle(self, widget: QWidget) -> None:
        widget._refresh_plot()
        _process_events_for(milliseconds=200)
        # Frame β on 0–1.1 and mark the β = 1 (exponential) and β = 1/3
        # (spin-glass) reference levels the trend runs between.
        axes = list(widget._figure.axes)
        if axes:
            ax = axes[0]
            ax.set_ylim(0.0, 1.12)
            for y, label in (
                (1.0, r"$\beta=1$ (exponential)"),
                (1.0 / 3.0, r"$\beta=1/3$ (spin glass)"),
            ):
                ax.axhline(y, color="0.6", linestyle="--", linewidth=1.0, zorder=1)
                ax.text(
                    ax.get_xlim()[1],
                    y,
                    "  " + label,
                    color="0.4",
                    fontsize=9,
                    va="center",
                    ha="left",
                )
            widget._canvas.draw()
        _process_events_for(milliseconds=150)


register(YmnalSpectraScenario())
register(YmnalStretchedFitScenario())
register(YmnalLambdaTScenario())
register(YmnalBetaTScenario())
