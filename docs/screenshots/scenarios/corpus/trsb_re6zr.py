"""Corpus scenarios — TRSB in the noncentrosymmetric superconductor Re₆Zr.

Worked example ``Superconductivity/TRSB`` (WiMDA muon school corpus), driving
Asymmetry through the time-reversal-symmetry-breaking (TRSB) measurement of
Singh, Hillier *et al.*, PRL **112**, 107002 (2014). Data are the MuSR runs
38176–38275 (ISIS NeXus-v1 HDF4 ``.nxs``), resolved through the corpus root.

Five scenarios, matched to the ground truth (``GROUND_TRUTH.md``):

* ``corpus_trsb_zf_kt_fit`` — a single low-T zero-field run fitted with the
  Gaussian Kubo–Toyabe × exponential model (paper Eqs. 2–3): the canonical KT
  lineshape on real data. Fitted Δ ≈ 0.26 µs⁻¹ (GT §6b σ(T→0) ≈ 0.2625).
* ``corpus_trsb_sigma_t_step`` — the **headline**: a per-run ZF batch fit
  rendered as σ(T) on a tight 0.250–0.270 µs⁻¹ axis, resolving the small
  spontaneous rise of the Gaussian KT rate below T_c = 6.75 K (GT §6b Target 1,
  the TRSB signature).
* ``corpus_trsb_lf_decoupling`` — the static-origin proof: base-T ZF vs 10 mT
  (100 G) longitudinal-field spectra overlaid; the LF trace stays polarised
  while the ZF trace relaxes (paper Fig. 3, GT §4a.5).
* ``corpus_trsb_tf_vortex_fit`` — a 400 G (40 mT) transverse-field mixed-state
  run fitted with Oscillatory × Gaussian + Constant.
* ``corpus_trsb_sigma_sc_t`` — the fitted TF Gaussian linewidth σ_G(T) in µs⁻¹
  against temperature.

``requires_fit = True`` on every scenario that runs a real iminuit fit at
capture time (iminuit/numba trips on numpy ≥ 2.3 in dev environments; CI pins
numpy < 2.3).
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from .._base import _process_events_for
from ._corpus import CorpusScenario, load_corpus_datasets, register

EXAMPLE = "Superconductivity/TRSB"
_DATA = "Superconductivity/TRSB/data/MUSR000%d.nxs"

# ZF Gaussian Kubo–Toyabe × exponential + constant background (paper Eqs. 2–3):
#   G(t) = A₀·G^KT(t;Δ)·exp(−Λt) + A_bg.
_ZF_MODEL = (["StaticGKT_ZF", "Exponential", "Constant"], ["*", "+"])
# The TF fit uses A·exp(−(σ_G t)²)·cos(2πνt+φ) + A_bg.
_TF_MODEL = (["Oscillatory", "Gaussian", "Constant"], ["*", "+"])

# Contiguous ZF temperature scan (GT §3a); the TRSB σ(T) headline series.
_ZF_RUNS = list(range(38224, 38261))
# Dense TF 0.01–8 K scan (GT §3b); the shipped WiMDA batch-fit runs.
_TF_RUNS = list(range(38180, 38220))
# Base-temperature (0.3 K) pair for the LF-decoupling overlay (GT §3a/§3c).
_ZF_BASE_RUN = 38224  # F = 0
_LF_BASE_RUN = 38263  # F = 100 G = 10 mT
_TF_BASE_RUN = 38180  # F = 400 G = 40 mT, T = 0.01 K


def _fit_runs(runs, model, seeds, bounds):
    """Yield the dataset and its full-range fit for each run."""
    from asymmetry.core.fitting.engine import FitEngine
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    engine = FitEngine()

    for run in runs:
        dataset = load_corpus_datasets([_DATA % run])[0]
        params = ParameterSet(
            [
                Parameter(
                    name=name,
                    value=seeds[name],
                    min=bounds.get(name, (None, None))[0],
                    max=bounds.get(name, (None, None))[1],
                )
                for name in model.param_names
            ]
        )
        yield dataset, engine.fit(dataset, model.function, params)


def _load_trend_panel(rows, title):
    """Build a Fit Parameters panel from the fitted trend rows."""
    from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel

    panel = FitParametersPanel()
    panel.load_representation_series(
        [("trsb-sigma-t", title, rows)],
        select_id="trsb-sigma-t",
    )
    _process_events_for(milliseconds=80)
    return panel


def _frame_trend_y(widget, y_lo, y_hi, *, tc_line=None):
    """Redraw the trend plot and clamp the σ axis to (y_lo, y_hi).

    The FitParametersPanel autoscales σ, which for the TRSB step spreads a
    ~0.006 µs⁻¹ signal across the axis' rounding margins and hides it. Reframing
    to a tight window (the caption's 0.250–0.270 µs⁻¹) makes the spontaneous
    rise legible. An optional dashed T_c marker anchors the reader's eye.
    """
    widget._refresh_plot()
    _process_events_for(milliseconds=200)
    axes = list(widget._figure.axes)
    if axes:
        ax = axes[0]
        ax.set_ylim(y_lo, y_hi)
        if tc_line is not None:
            ax.axvline(tc_line, color="0.5", linestyle="--", linewidth=1.0, zorder=1)
            ax.text(
                tc_line,
                y_hi,
                r"  $T_\mathrm{c}$ = 6.75 K",
                color="0.35",
                fontsize=9,
                va="top",
                ha="left",
            )
        widget._canvas.draw()
    _process_events_for(milliseconds=120)


def _configure_single_fit(window, components, operators, seeds, positive):
    """Set up and run a single time-domain fit in the main window's fit panel.

    *positive* names parameters whose Min bound is pinned to 0 so the fit does
    not settle in the sign-degenerate mirror minimum (the Gaussian width enters
    squared, so an unbounded minimiser is free to return a negative rate).
    """
    from asymmetry.core.fitting.composite import CompositeModel
    from asymmetry.gui.panels.fit.tab_base import (
        _param_table_rows_by_name,
        _set_param_table_value,
    )

    single_tab = window._fit_panel._single_tab
    single_tab._set_composite_model(CompositeModel(components, operators=operators))
    _process_events_for(milliseconds=80)

    table = single_tab._param_table
    rows_by_name = _param_table_rows_by_name(table)
    for name, value in seeds.items():
        if name in rows_by_name:
            _set_param_table_value(table, rows_by_name[name], value)
    for name in positive:
        if name in rows_by_name:
            item = table.item(rows_by_name[name], table.COL_MIN)
            if item is not None:
                item.setText("0.0")
    _process_events_for(milliseconds=60)

    single_tab._run_fit()
    single_tab.wait_for_fit()
    _process_events_for(milliseconds=80)
    return single_tab


# --------------------------------------------------------------------------- #
#  1. ZF Gaussian Kubo–Toyabe fit — the canonical KT lineshape on real data.
# --------------------------------------------------------------------------- #
class TrsbZfKtFitScenario(CorpusScenario):
    name = "corpus_trsb_zf_kt_fit"
    description = (
        "Converged Gaussian Kubo–Toyabe × exp fit on the Re₆Zr base-T (0.3 K) zero-field run 38224."
    )
    example = EXAMPLE
    size = (1500, 920)
    requires_fit = True

    def build(self) -> QWidget:
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window._on_fit()
        window.resizeDocks([window._dock_data_browser], [320], Qt.Orientation.Horizontal)

        datasets = load_corpus_datasets([_DATA % _ZF_BASE_RUN])
        self.add_to_browser(window, datasets)
        window._on_dataset_selected(datasets[0].run_number)
        _process_events_for(milliseconds=80)

        _configure_single_fit(
            window,
            *_ZF_MODEL,
            seeds={"A_1": 19.0, "Delta": 0.26, "Lambda": 0.01, "A_bg": 14.5},
            positive=("A_1", "Delta", "A_bg"),
        )

        # Frame the first ~13 µs so the KT "dip and ⅓ recovery" and the fit
        # overlay both read, before the ZF asymmetry's late-time error fan
        # (the F−B denominator vanishes as counts decay) swamps the panel.
        x_min, x_max, y_min, y_max = window._plot_panel.get_view_limits()
        window._plot_panel.set_view_limits(0.0, 13.0, y_min, y_max)
        _process_events_for(milliseconds=80)
        return window


# --------------------------------------------------------------------------- #
#  2. σ(T) TRSB step — the headline, framed on a tight axis.
# --------------------------------------------------------------------------- #
class TrsbSigmaTStepScenario(CorpusScenario):
    name = "corpus_trsb_sigma_t_step"
    description = (
        "ZF Gaussian KT σ(T) trend for Re₆Zr, framed 0.250–0.270 µs⁻¹ to "
        "resolve the spontaneous TRSB step across T_c = 6.75 K."
    )
    example = EXAMPLE
    size = (1240, 760)
    requires_fit = True

    def build(self) -> QWidget:
        from asymmetry.core.fitting.composite import CompositeModel

        components, operators = _ZF_MODEL
        fits = _fit_runs(
            _ZF_RUNS,
            CompositeModel(components, operators=operators),
            seeds={"A_1": 19.0, "Delta": 0.26, "Lambda": 0.01, "A_bg": 14.5},
            bounds={
                "A_1": (0.0, 40.0),
                "Delta": (0.0, 1.0),
                "Lambda": (-0.05, 0.5),
                "A_bg": (0.0, 30.0),
            },
        )
        points = [
            (
                float(dataset.temperature),
                abs(float(result.parameters["Delta"].value)),
                float(result.uncertainties.get("Delta", np.nan)),
            )
            for dataset, result in fits
        ]
        rows = [
            {
                "run_number": 1000 + i,
                "run_label": f"{temperature:.2f} K",
                "field": 0.0,
                "temperature": temperature,
                "values": {"sigma": width},
                "errors": {"sigma": error},
            }
            for i, (temperature, width, error) in enumerate(sorted(points, key=lambda p: p[0]))
        ]
        return _load_trend_panel(rows, "σ(T) — Re₆Zr ZF (TRSB)")

    def settle(self, widget: QWidget) -> None:
        _frame_trend_y(widget, 0.250, 0.270, tc_line=6.75)


# --------------------------------------------------------------------------- #
#  3. LF decoupling — static-origin proof (ZF vs 10 mT LF at base T).
# --------------------------------------------------------------------------- #
class TrsbLfDecouplingScenario(CorpusScenario):
    name = "corpus_trsb_lf_decoupling"
    description = (
        "Base-T (0.3 K) overlay of the Re₆Zr zero-field spectrum against the "
        "10 mT (100 G) longitudinal-field run: the LF decouples the relaxation."
    )
    example = EXAMPLE
    size = (1500, 920)

    def build(self) -> QWidget:
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window.resizeDocks([window._dock_data_browser], [320], Qt.Orientation.Horizontal)

        datasets = load_corpus_datasets([_DATA % _ZF_BASE_RUN, _DATA % _LF_BASE_RUN])
        with window._data_browser.batch_updates():
            for dataset in datasets:
                window._data_browser.add_dataset(dataset)

        run_numbers = [int(ds.run_number) for ds in datasets]
        window._data_browser.create_data_group(run_numbers, name="ZF vs 10 mT LF — Re₆Zr (0.3 K)")

        window._plot_panel.set_overlay_enabled(True, emit_signal=True)
        window._data_browser._table.selectAll()
        window._on_dataset_selected(run_numbers[0])
        _process_events_for(milliseconds=120)

        # Frame the first ~10 µs: the ZF trace relaxes toward its ⅓ tail while
        # the LF trace stays polarised — the decoupling that proves the
        # spontaneous fields are static (paper Fig. 3). Beyond ~10 µs the ZF
        # asymmetry error fan overwhelms the contrast, so stop short of it.
        x_min, x_max, y_min, y_max = window._plot_panel.get_view_limits()
        window._plot_panel.set_view_limits(0.0, 10.0, y_min, y_max)
        _process_events_for(milliseconds=80)
        return window


# --------------------------------------------------------------------------- #
#  4. TF mixed-state fit — Gaussian-damped precession.
# --------------------------------------------------------------------------- #
class TrsbTfVortexFitScenario(CorpusScenario):
    name = "corpus_trsb_tf_vortex_fit"
    description = (
        "Oscillatory × Gaussian + Constant fit on the Re₆Zr 400 G (40 mT) TF "
        "mixed-state run 38180 (0.01 K), showing the fitted Gaussian linewidth σ_G."
    )
    example = EXAMPLE
    size = (1500, 920)
    requires_fit = True

    def build(self) -> QWidget:
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window._on_fit()
        window.resizeDocks([window._dock_data_browser], [320], Qt.Orientation.Horizontal)

        datasets = load_corpus_datasets([_DATA % _TF_BASE_RUN])
        self.add_to_browser(window, datasets)
        window._on_dataset_selected(datasets[0].run_number)
        _process_events_for(milliseconds=80)

        # 400 G ⇒ γ_μ·B ≈ 5.42 MHz; seed near it so the fit locks the phase.
        _configure_single_fit(
            window,
            *_TF_MODEL,
            seeds={
                "A_1": 8.0,
                "frequency": 5.36,
                "phase": 2.37,
                "sigma": 0.4,
                "A_bg": 0.0,
            },
            positive=("A_1", "sigma"),
        )

        # Zoom to the first ~1.4 µs so individual precession cycles and the
        # Gaussian relaxation envelope are both resolved.
        x_min, x_max, y_min, y_max = window._plot_panel.get_view_limits()
        window._plot_panel.set_view_limits(0.0, 1.4, y_min, y_max)
        _process_events_for(milliseconds=80)
        return window


# --------------------------------------------------------------------------- #
#  5. TF fitted Gaussian linewidth through T_c.
# --------------------------------------------------------------------------- #
class TrsbSigmaScTScenario(CorpusScenario):
    name = "corpus_trsb_sigma_sc_t"
    description = "Fitted Re₆Zr TF Gaussian linewidth σ_G(T) in µs⁻¹ at 40 mT."
    example = EXAMPLE
    size = (1240, 760)
    requires_fit = True

    def build(self) -> QWidget:
        from asymmetry.core.fitting.composite import CompositeModel
        from asymmetry.core.fitting.sc.trend_width import model_width_sources

        components, operators = _TF_MODEL
        model = CompositeModel(components, operators=operators)
        parameter_sources = model_width_sources(model)
        fits = _fit_runs(
            _TF_RUNS,
            model,
            seeds={
                "A_1": 8.0,
                "frequency": 5.36,
                "phase": 2.37,
                "sigma": 0.4,
                "A_bg": 0.0,
            },
            bounds={
                "A_1": (0.0, 40.0),
                "frequency": (5.0, 5.8),
                "phase": (None, None),
                "sigma": (0.0, 1.0),
                "A_bg": (-5.0, 5.0),
            },
        )
        rows = []
        for dataset, result in fits:
            if not result.success or "sigma" not in result.parameters:
                continue
            temperature = dataset.temperature
            width = result.parameters["sigma"].value
            error = result.uncertainties.get("sigma", np.nan)
            if (
                temperature is None
                or not np.isfinite([temperature, width, error]).all()
                or error <= 0
            ):
                continue
            rows.append(
                {
                    "run_number": dataset.run_number,
                    "run_label": dataset.run_label,
                    "field": dataset.field,
                    "temperature": temperature,
                    "values": {"sigma": width},
                    "errors": {"sigma": error},
                    "parameter_sources": parameter_sources,
                }
            )
        if len({row["values"]["sigma"] for row in rows}) < 2:
            raise RuntimeError("The Re₆Zr TF trend requires distinct successful linewidth fits.")
        rows.sort(key=lambda row: row["temperature"])
        panel = _load_trend_panel(rows, "σ_G(T) — Re₆Zr TF 40 mT")
        panel._x_combo.setCurrentText("𝑇 (K)")
        panel._overlay_button.setChecked(True)
        return panel

    def settle(self, widget: QWidget) -> None:
        _frame_trend_y(widget, 0.10, 0.50, tc_line=6.75)
        ax = widget._figure.axes[0]
        ax.set_title("Fitted TF Gaussian linewidth")
        ax.set_ylabel(r"$\sigma_G$ (µs⁻¹)")
        ax.legend(ax.containers, ["Per-run fit ± 1σ"], loc="best", fontsize="small")
        widget._canvas.draw()


register(TrsbZfKtFitScenario())
register(TrsbSigmaTStepScenario())
register(TrsbLfDecouplingScenario())
register(TrsbTfVortexFitScenario())
register(TrsbSigmaScTScenario())
