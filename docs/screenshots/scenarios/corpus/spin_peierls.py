"""Corpus scenarios — A spin-Peierls transition (Magnetism).

Worked example ``Magnetism/A spin-Peierls transition`` (WiMDA muon school
corpus), driving Asymmetry through the zero-field µSR study of the quasi-1D
S = ½ charge-transfer salt **KTCNQF₄**. The reduced TCNQF₄⁻ stacks carry an
S = ½ moment; near ≈150 K adjacent molecules dimerise and the chain undergoes a
**spin-Peierls transition** into a singlet ground state. The guide
(``GROUND_TRUTH.md`` §1) is titled "Fluctuations in a spin-Peierls compound":
the teaching point is that the **electronic spin dynamics move through the µSR
time window** as the sample is cooled, so the ZF line-shape *changes character*
across the transition.

Data are 26 EMU (ISIS) NeXus-v1 HDF4 ``.nxs`` runs 29919–29944 (GT §2). The
on-disk ZF temperature series is the contiguous block 29931–29944 (130–185 K),
plus three lower-T ZF runs 29921/29923/29924 (30/60/90 K) and a 300 K ZF run
29920; run 29919 is the TF 100 G / 300 K calibration. The 110–125 K runs
29926–29930 the guide would like as "ZF" are on disk at **100 G transverse**
(GT §2 file-header audit), so they precess and are unusable for a ZF relaxation
trend — see ``NOTES_spinpeierls.md``.

What the ZF fits show (deliverable, GT §4–6):

* **T ≳ 150 K (above the transition):** the paramagnetic S = ½ moments fluctuate
  fast (exchange-narrowed 1D chain); the muon is decoupled from them and senses
  only the **static nuclear-dipolar field** → a near-Gaussian / Kubo–Toyabe
  line-shape (stretched exponent β ≈ 1.9, static models fit, exponential fails).
* **cooling toward / below the transition:** the electronic correlation time
  slows into the µSR window → **dynamic** relaxation; β falls to ≈0.5 and the
  relaxation rate λ **peaks near ~60 K** (fluctuations at their slowest in the
  window).
* **deep in the singlet phase (~30 K):** the moments gap out; λ drops and β
  recovers toward the static (nuclear) form.

So the single observable — the ZF line-shape (β) and rate (λ) vs T — captures
the static→dynamic→static crossover the guide poses. NB the on-disk ZF series
has a gap between 90 K and 130 K (the intervening files are 100 G), so the
line-shape change is bracketed rather than densely traced through 150 K; the
change onsets on cooling through ~145–130 K (β departs from 2) consistent with
the ≈150 K transition, and the λ peak sits at ~60 K deep in the fluctuation
regime.

Scenarios registered:

* ``corpus_spinpeierls_zf_spectra`` — ZF spectra overlaid 185 K → 60 K: the
  line-shape changes from a slow static-Gaussian decay (high T) to a fast
  dynamic decay (low T). The core teaching image.
* ``corpus_spinpeierls_model_discrimination`` — one ZF run from each regime
  (185 K static, 90 K dynamic) each fitted with a **static** Gaussian
  Kubo–Toyabe and a **dynamic** stretched exponential, with residual panels and
  reduced-χ²: which model is physically meaningful flips across the transition.
* ``corpus_spinpeierls_beta_t`` — the stretching exponent β(T) across the ZF
  series: ≈2 (Gaussian / static) at high T falling toward ≈0.5 (dynamic) near
  the transition, the line-shape crossover as a single number.
* ``corpus_spinpeierls_lambda_t`` — the ZF relaxation rate λ(T), weak in the
  static high-T limit and rising to a peak near ~60 K as the electronic
  fluctuations enter the µSR window.

``requires_fit = True`` on every scenario that runs real iminuit fits at capture
time (all but the plain spectrum overlay). No numeric fit targets exist in the
guide (GT §6/§9 — the ground truth is the *qualitative* static↔dynamic
identification at ≈150 K); the β/λ magnitudes here reproduce the corpus working
note and are reported in ``NOTES_spinpeierls.md``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QWidget

from .._base import CaptureContext, _process_events_for
from ._corpus import CorpusScenario, load_corpus_datasets, register

EXAMPLE = "Magnetism/A spin-Peierls transition"
_DATA = EXAMPLE + "/Data"

# ZF (0 G) temperature series, run → measured T (K) from the .nxs headers
# (GT §2 file-header audit; the guide's 130–185 K rows match the headers). The
# contiguous 130–185 K block plus the lower-T ZF runs at 90/60/30 K. Run 29920
# is 300 K ZF (kept out of the trends — a large gap above 185 K — but part of
# the static high-T limit). The 110–125 K files (29926–29930) are 100 G
# transverse, not ZF, and are excluded (see module docstring / NOTES).
_ZF_SERIES: list[tuple[int, float]] = [
    (29944, 185.0),
    (29943, 180.0),
    (29941, 175.0),
    (29940, 170.0),
    (29939, 165.0),
    (29938, 160.0),
    (29937, 155.0),
    (29936, 150.0),
    (29934, 145.0),
    (29933, 140.0),
    (29932, 135.0),
    (29931, 130.0),
    (29924, 90.0),
    (29923, 60.0),
    (29921, 30.0),
]

# Spectrum-overlay runs: a descending-T sweep spanning both regimes so the
# line-shape change reads (slow static Gaussian at 185 K → fast dynamic at 60 K).
_SPECTRA_RUNS = [29944, 29934, 29931, 29924, 29923]  # 185/145/130/90/60 K

# Model-discrimination runs: one from each side of the transition.
_DISCRIM: list[tuple[int, float]] = [(29944, 185.0), (29924, 90.0)]

# Calibration run for the fixed t=0 asymmetry (highest-T ZF, well-determined).
_A1_CAL_RUN = 29944  # 185 K

# Candidate ZF relaxation models (GT §4 — "different ways to fit the data"):
#   static  = static Gaussian Kubo–Toyabe (nuclear-dipolar, time-independent)
#   dynamic = stretched exponential (electronic fluctuations in the µSR window)
_STATIC_MODEL = ["StaticGKT_ZF", "Constant"]
_DYNAMIC_MODEL = ["StretchedExponential", "Constant"]

# Fit window: skip the first ~0.1 µs (pulse/deadtime) out to 12 µs, before the
# ZF F−B asymmetry error fan (vanishing denominator) swamps the late-time data.
_T_MIN, _T_MAX = 0.1, 12.0


# --------------------------------------------------------------------------- #
#  Core-engine fitting helpers (drive the same FitEngine the GUI uses).
# --------------------------------------------------------------------------- #
def _fit_model(components, dataset, seed, *, fixed=()):
    """Fit *components* (+ flat bg already in the list) to one dataset.

    Returns ``(values, uncertainties, reduced_chi_squared)``. Positive physical
    parameters are bounded ≥ 0 and β ≤ 2 (matching the GUI's default bounds).
    """
    from asymmetry.core.fitting.composite import CompositeModel
    from asymmetry.core.fitting.engine import FitEngine
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    model = CompositeModel(components)
    params = []
    for name in model.param_names:
        p = Parameter(name, seed.get(name, 0.1))
        if name in ("A_1", "Lambda", "beta", "Delta"):
            p.min = 0.0
        if name == "beta":
            p.max = 2.0
        if name in fixed:
            p.fixed = True
        params.append(p)
    result = FitEngine().fit(
        dataset, model.function, ParameterSet(params), t_min=_T_MIN, t_max=_T_MAX
    )
    values = {p.name: p.value for p in result.parameters}
    return (
        values,
        result.uncertainties or {},
        float(getattr(result, "reduced_chi_squared", float("nan"))),
    )


def _calibrate_a1() -> float:
    """Fixed t=0 asymmetry from the highest-T ZF run (β, A_1 both free).

    The initial asymmetry is geometry-fixed (constant across T); pinning it lets
    the low-T dynamic fits determine β and λ cleanly. Left free, the stretched
    exponential's amplitude/rate/β degeneracy lets A_1 run away at 60–90 K
    (wave-1 lesson: fix the amplitude for real stretched-exponential series).
    """
    dataset = load_corpus_datasets([f"{_DATA}/emu000{_A1_CAL_RUN}.nxs"])[0]
    values, _unc, _chi = _fit_model(
        _DYNAMIC_MODEL, dataset, {"A_1": 18.0, "Lambda": 0.16, "beta": 1.9, "A_bg": 2.5}
    )
    return float(values["A_1"])


def _fit_series():
    """Warm-started ZF batch fit → (T, β, λ, β_err, λ_err) sorted ascending in T.

    Fits the stretched exponential + flat background with A_1 fixed to the
    calibrated t=0 asymmetry, in descending-temperature order so each fit
    warm-starts from the previous (higher-T) minimum — cold seeds on the
    stretched exponential walk to wrong β/λ minima on real data (wave-1 lesson).
    """
    a1 = _calibrate_a1()
    seed = {"A_1": a1, "Lambda": 0.05, "beta": 1.9, "A_bg": 2.5}
    rows: list[tuple[float, float, float, float, float]] = []
    for run, temp in sorted(_ZF_SERIES, key=lambda rt: -rt[1]):
        dataset = load_corpus_datasets([f"{_DATA}/emu000{run}.nxs"])[0]
        values, unc, _chi = _fit_model(_DYNAMIC_MODEL, dataset, seed, fixed=("A_1",))
        lam = abs(values["Lambda"])
        beta = values["beta"]
        rows.append(
            (
                temp,
                beta,
                lam,
                float(unc.get("beta", np.nan)),
                float(unc.get("Lambda", np.nan)),
            )
        )
        seed = {"A_1": a1, "Lambda": lam, "beta": beta, "A_bg": values["A_bg"]}
    rows.sort(key=lambda r: r[0])
    arr = np.array(rows)
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]


def _trend_panel(temps, values, errors, param_name, title):
    """Build a FitParametersPanel showing one fitted parameter vs temperature."""
    from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel

    order = np.argsort(temps)
    row_dicts = [
        {
            "run_number": int(_ZF_RUN_FOR_T.get(round(float(temps[j])), 1000 + i)),
            "run_label": f"{temps[j]:.0f} K",
            "field": 0.0,
            "temperature": float(temps[j]),
            "values": {param_name: float(values[j])},
            "errors": {param_name: float(errors[j]) if np.isfinite(errors[j]) else 0.0},
        }
        for i, j in enumerate(order)
    ]
    panel = FitParametersPanel()
    panel.load_representation_series(
        [(f"peierls-{param_name}", title, row_dicts)],
        select_id=f"peierls-{param_name}",
    )
    _process_events_for(milliseconds=80)
    return panel


_ZF_RUN_FOR_T = {int(t): r for r, t in _ZF_SERIES}


def _wait_until(predicate, *, timeout_ms: int, poll_ms: int = 40) -> None:
    elapsed = 0
    while elapsed < timeout_ms:
        if predicate():
            return
        _process_events_for(milliseconds=poll_ms)
        elapsed += poll_ms


def _pump(milliseconds: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(int(milliseconds), loop.quit)
    loop.exec()
    QApplication.processEvents()


# --------------------------------------------------------------------------- #
#  1. ZF spectrum evolution — the line-shape change across the transition.
# --------------------------------------------------------------------------- #
class SpinPeierlsZfSpectraScenario(CorpusScenario):
    name = "corpus_spinpeierls_zf_spectra"
    description = (
        "KTCNQF₄ zero-field spectra overlaid from 185 K down to 60 K. Above the "
        "≈150 K spin-Peierls transition the muon relaxes slowly with a static "
        "Gaussian (nuclear-dipolar) line-shape; on cooling the electronic spin "
        "fluctuations enter the µSR window and the relaxation becomes fast and "
        "dynamic — the line-shape change the exercise is built around."
    )
    example = EXAMPLE
    size = (1500, 920)

    def build(self) -> QWidget:
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window.resizeDocks([window._dock_data_browser], [340], Qt.Orientation.Horizontal)

        datasets = load_corpus_datasets([f"{_DATA}/emu000{r}.nxs" for r in _SPECTRA_RUNS])
        with window._data_browser.batch_updates():
            for dataset in datasets:
                window._data_browser.add_dataset(dataset)
        run_numbers = [int(ds.run_number) for ds in datasets]
        window._data_browser.create_data_group(run_numbers, name="KTCNQF₄ ZF — 185→60 K")

        window._plot_panel.set_overlay_enabled(True, emit_signal=True)
        # Bunch the bins so the fast low-T decays read through the ZF asymmetry
        # noise instead of being buried in it.
        window._plot_panel.set_bunch_factor(6, emit_signal=True)
        window._data_browser._table.selectAll()
        window._on_dataset_selected(run_numbers[0])
        _process_events_for(milliseconds=120)

        # First ~10 µs: the slow 185 K Gaussian and the fast 60 K dynamic decay
        # both read before the late-time ZF error fan takes over.
        x_min, x_max, y_min, y_max = window._plot_panel.get_view_limits()
        window._plot_panel.set_view_limits(0.0, 10.0, y_min, y_max)
        _process_events_for(milliseconds=80)
        return window


# --------------------------------------------------------------------------- #
#  2. Model discrimination — static KT vs dynamic stretched, either side.
# --------------------------------------------------------------------------- #
class SpinPeierlsModelDiscriminationScenario(CorpusScenario):
    name = "corpus_spinpeierls_model_discrimination"
    description = (
        "Which relaxation model describes the data flips across the transition. "
        "A 185 K (above) and a 90 K (below) KTCNQF₄ zero-field run are each "
        "fitted with a static Gaussian Kubo–Toyabe and a dynamic exponential — "
        "the guide's 'different ways to fit the data'. At 185 K the static "
        "Kubo–Toyabe fits (χ²ᵣ≈1.1) and the exponential misses the Gaussian "
        "shoulder; at 90 K it reverses — the static model fails badly (χ²ᵣ≈8, "
        "structured residuals) and the dynamic exponential is required."
    )
    example = EXAMPLE
    size = (1320, 860)
    requires_fit = True

    # Two simple candidate models the guide invites (GT §4): a static Gaussian
    # Kubo–Toyabe (nuclear dipolar) and a dynamic exponential (electronic
    # fluctuations). The stretched exponent that quantifies the crossover
    # continuously is the subject of corpus_spinpeierls_beta_t.
    _DYN_DISCRIM = ["Exponential", "Constant"]

    def capture(self, ctx: CaptureContext) -> Path:  # noqa: D401
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        from asymmetry.core.fitting.composite import CompositeModel

        static = CompositeModel(_STATIC_MODEL)
        dynamic = CompositeModel(self._DYN_DISCRIM)

        figure = Figure(figsize=(11.0, 7.2), dpi=120)
        # Two columns (185 K | 90 K), each a tall data+fit panel over a short
        # residual panel.
        gs = figure.add_gridspec(
            2,
            2,
            height_ratios=[3, 1.4],
            hspace=0.08,
            wspace=0.22,
            left=0.075,
            right=0.985,
            top=0.90,
            bottom=0.09,
        )
        c_static, c_dynamic, c_data = "#1f77b4", "#d62728", "#333333"

        for col, (run, temp) in enumerate(_DISCRIM):
            dataset = load_corpus_datasets([f"{_DATA}/emu000{run}.nxs"])[0]
            t = np.asarray(dataset.time)
            y = np.asarray(dataset.asymmetry)
            e = np.asarray(dataset.error) if dataset.error is not None else None
            mask = (t >= _T_MIN) & (t <= _T_MAX)

            sv, _su, s_chi = _fit_model(
                _STATIC_MODEL, dataset, {"A_1": 19.0, "Delta": 0.17, "A_bg": 2.5}
            )
            dv, _du, d_chi = _fit_model(
                self._DYN_DISCRIM,
                dataset,
                {"A_1": 19.0, "Lambda": 0.2, "A_bg": 2.5},
            )
            tfit = t[mask]
            y_static = static.function(tfit, **sv)
            y_dynamic = dynamic.function(tfit, **dv)

            ax = figure.add_subplot(gs[0, col])
            axr = figure.add_subplot(gs[1, col], sharex=ax)

            # Bin the raw asymmetry for a readable scatter.
            step = 8
            ax.errorbar(
                t[mask][::step],
                y[mask][::step],
                yerr=(e[mask][::step] if e is not None else None),
                fmt="o",
                ms=3,
                color=c_data,
                ecolor="#bbbbbb",
                elinewidth=0.6,
                alpha=0.55,
                zorder=1,
                label="ZF data",
            )
            ax.plot(
                tfit,
                y_static,
                color=c_static,
                lw=2.0,
                zorder=3,
                label=f"static Gaussian KT  (χ²ᵣ={s_chi:.2f})",
            )
            ax.plot(
                tfit,
                y_dynamic,
                color=c_dynamic,
                lw=2.0,
                ls="--",
                zorder=4,
                label=f"dynamic exponential  (χ²ᵣ={d_chi:.2f})",
            )
            regime = "above T$_{SP}$ — static" if temp > 150 else "below T$_{SP}$ — dynamic"
            ax.set_title(f"{temp:.0f} K  ({regime})", fontsize=12, pad=8)
            ax.legend(loc="upper right", fontsize=9, frameon=True)
            ax.grid(True, alpha=0.2)
            ax.tick_params(labelbottom=False)
            if col == 0:
                ax.set_ylabel("Asymmetry (%)")

            # Residuals in σ units (fall back to raw if no errors present).
            denom = e[mask] if e is not None else np.ones_like(tfit)
            denom = np.where(denom == 0, np.nan, denom)
            axr.axhline(0, color="0.6", lw=0.8)
            axr.plot(tfit[::step], ((y[mask] - y_static) / denom)[::step], color=c_static, lw=1.0)
            axr.plot(
                tfit[::step],
                ((y[mask] - y_dynamic) / denom)[::step],
                color=c_dynamic,
                lw=1.0,
                ls="--",
            )
            axr.set_ylim(-5, 5)
            axr.grid(True, alpha=0.2)
            axr.set_xlabel("Time  t (µs)")
            if col == 0:
                axr.set_ylabel("Resid. (σ)")

        figure.suptitle(
            "KTCNQF₄ ZF: static vs dynamic relaxation model, either side of the "
            "≈150 K spin-Peierls transition",
            fontsize=13,
        )

        canvas = FigureCanvasQTAgg(figure)
        canvas.draw()
        pix = QPixmap(canvas.size())
        canvas.render(pix)
        out_path = ctx.output_dir / f"{self.name}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not pix.save(str(out_path), "PNG"):
            raise RuntimeError(f"Failed to save screenshot to {out_path}")
        _pump(40)
        return out_path


# --------------------------------------------------------------------------- #
#  3. β(T) — the line-shape crossover as a single number.
# --------------------------------------------------------------------------- #
class SpinPeierlsBetaTScenario(CorpusScenario):
    name = "corpus_spinpeierls_beta_t"
    description = (
        "KTCNQF₄ ZF stretching exponent β(T): β ≈ 2 (Gaussian / static "
        "nuclear-dipolar line-shape) above the ≈150 K spin-Peierls transition, "
        "falling toward β ≈ 0.5 (dynamic electronic relaxation) as the "
        "fluctuations enter the µSR window on cooling."
    )
    example = EXAMPLE
    size = (1240, 760)
    requires_fit = True

    def build(self) -> QWidget:
        temps, beta, _lam, beta_err, _lam_err = _fit_series()
        self._temps, self._beta = temps, beta
        return _trend_panel(temps, beta, beta_err, "beta", "β(T) — KTCNQF₄ ZF")

    def settle(self, widget: QWidget) -> None:
        widget._refresh_plot()
        _process_events_for(milliseconds=200)
        axes = list(widget._figure.axes)
        if axes:
            ax = axes[0]
            ax.set_ylim(0.0, 2.2)
            # Reference levels the line-shape runs between, and the transition.
            for y, label in (
                (2.0, r"$\beta=2$  (Gaussian / static)"),
                (1.0, r"$\beta=1$  (exponential)"),
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
            ax.axvline(150.0, color="#d62728", linestyle=":", linewidth=1.2, zorder=1)
            ax.text(
                150.0, 0.12, " T$_{SP}$≈150 K", color="#d62728", fontsize=9, va="bottom", ha="left"
            )
            widget._canvas.draw()
        _process_events_for(milliseconds=150)


# --------------------------------------------------------------------------- #
#  4. λ(T) — dynamic relaxation-rate peak entering the µSR window.
# --------------------------------------------------------------------------- #
class SpinPeierlsLambdaTScenario(CorpusScenario):
    name = "corpus_spinpeierls_lambda_t"
    description = (
        "KTCNQF₄ ZF relaxation rate λ(T): weak in the static high-T limit "
        "(fast paramagnetic fluctuations averaged out) and rising to a peak "
        "near ~60 K as the electronic spin dynamics slow into the µSR window "
        "below the ≈150 K spin-Peierls transition."
    )
    example = EXAMPLE
    size = (1240, 760)
    requires_fit = True

    def build(self) -> QWidget:
        temps, _beta, lam, _beta_err, lam_err = _fit_series()
        self._temps, self._lam = temps, lam
        return _trend_panel(temps, lam, lam_err, "Lambda", "λ(T) — KTCNQF₄ ZF")

    def settle(self, widget: QWidget) -> None:
        widget._refresh_plot()
        _process_events_for(milliseconds=200)
        axes = list(widget._figure.axes)
        if axes:
            ax = axes[0]
            ax.axvline(150.0, color="#d62728", linestyle=":", linewidth=1.2, zorder=1)
            ax.text(
                150.0,
                ax.get_ylim()[1] * 0.9,
                " T$_{SP}$≈150 K",
                color="#d62728",
                fontsize=9,
                va="top",
                ha="left",
            )
            widget._canvas.draw()
        _process_events_for(milliseconds=150)


register(SpinPeierlsZfSpectraScenario())
register(SpinPeierlsModelDiscriminationScenario())
register(SpinPeierlsBetaTScenario())
register(SpinPeierlsLambdaTScenario())
