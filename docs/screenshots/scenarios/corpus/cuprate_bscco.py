"""Corpus scenarios — penetration depth in a high-Tc cuprate (BiSCCO).

Worked example ``Superconductivity/A high-Tc cuprate`` (WiMDA muon school
corpus), driving Asymmetry through the classic transverse-field vortex-lattice
σ(T) → penetration-depth workflow on Bi-2212 (Bi-Sr-Ca-Cu-O). Data are the ISIS
MUSR NeXus-v1 HDF4 ``.nxs`` runs 1274–1303 (a 400 G and a 200 G TF temperature
scan, 10–125 K), resolved through the corpus root. See the example's
``GROUND_TRUTH.md``; the shipped WiMDA ``.fit``/``.dat`` reference numbers grade
these scenarios.

Physics. Below T_c ≈ 105–110 K the applied transverse field enters the sample
as a vortex lattice, producing an inhomogeneous internal-field distribution
p(B). The TF-µSR line is approximated by a Gaussian relaxation
G(t) = exp(−σ²t²/2), whose second moment σ = γ_µ⟨ΔB²⟩^½ measures the width of
p(B). σ(T→0) connects to the London penetration depth by the guide relation
σ(µs⁻¹) = 75780 / λ_L²(nm). ⚠ For the *extremely anisotropic* Bi-2212 the three
teaching fields (150/200/400 G) all sit below the pancake-vortex crossover
B* ≈ 500 G, so a single λ_L is physically unreliable here (GROUND_TRUTH §6b);
these renders treat λ_L as indicative only and grade the σ(T) curve itself.

Scenarios registered (all on the MUSR 400 G scan, the authoritative deliverable):

* ``corpus_bscco_tf_damping`` — time-domain overlay of the 10 K (run 1277) and
  125 K normal-state (run 1276) 400 G spectra: the Gaussian vortex damping
  (σ ≈ 1.15 µs⁻¹, decayed by ~2 µs) against the essentially undamped
  normal-state precession. The field-distribution broadening, in time.
* ``corpus_bscco_vortex_fft`` — frequency-domain FFT of the 10 K run: the broad
  vortex p(B) line near the 400 G Larmor frequency (~5.4 MHz).
* ``corpus_bscco_tf_fit`` — converged Oscillatory × Gaussian fit on run 1277
  (10 K): σ ≈ 1.16 µs⁻¹ against the WiMDA reference 1.1467(75) µs⁻¹ (§7).
* ``corpus_bscco_sigma_t`` — the **headline**: per-run TF batch fit rendered as
  σ(T) 10–125 K, reproducing the 14-row WiMDA reference trend (σ falls from
  ≈1.15 µs⁻¹ at 10 K to ≈0.055 µs⁻¹ above T_c). λ_L ≈ 255 nm noted, with the
  §6b anisotropy caveat.
* ``corpus_bscco_field_compare`` — the guide's field-comparison task: σ(T) at
  400 G overlaid on σ(T) at 200 G. The 200 G plateau (~0.9 µs⁻¹) sits below the
  400 G plateau (~1.15 µs⁻¹) — the pancake-vortex field dependence of §6b, not
  fit scatter.

The guide also asks to compare FFT with Maximum-Entropy; MaxEnt does *not*
reconstruct cleanly on this real forward/back asymmetry (its large α = 1
baseline drives the solver past its χ² optimum by ~cycle 7 and it diverges into
spiky noise), so only the FFT frequency-domain render is shipped. ``requires_fit
= True`` on every scenario that runs a real iminuit fit at capture time (iminuit
trips numpy ≥ 2.3 in dev environments; CI pins numpy < 2.3).
"""

from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from .._base import _process_events_for
from ._corpus import CorpusScenario, load_corpus_datasets, register

EXAMPLE = "Superconductivity/A high-Tc cuprate"
_DATA = "Superconductivity/A high-Tc cuprate/Data/MUSR%08d.nxs"

# TF mixed-state precession (GROUND_TRUTH §7: Osc=Rotation Field, Rel=Gaussian):
#   G(t) = A·exp(−σ²t²/2)·cos(2πνt+φ) + A_bg.
_TF_MODEL = (["Oscillatory", "Gaussian", "Constant"], ["*", "+"])

# MUSR 400 G temperature scan, ascending T (GROUND_TRUTH §3 / §7 reference table).
# (run, T[K]); the σ(T) headline series. Run 1276 (125 K) is the normal-state
# reference run and sorts last by temperature.
_SCAN_400G: list[tuple[int, float]] = [
    (1277, 10.0),
    (1278, 30.0),
    (1279, 50.0),
    (1280, 70.0),
    (1281, 80.0),
    (1282, 85.0),
    (1283, 90.0),
    (1284, 95.0),
    (1285, 100.0),
    (1286, 105.0),
    (1287, 110.0),
    (1288, 115.0),
    (1289, 120.0),
    (1276, 125.0),
]
_RUN_SUPER = 1277  # 10 K, deep in the mixed state (broad vortex line)
_RUN_NORMAL = 1276  # 125 K, above T_c (narrow, undamped line)

# MUSR 200 G temperature scan (GROUND_TRUTH §7 second table). Run 1291 (10 K)
# is dropped: its reference fit is the documented negative-σ pathology (§9), a
# fit artefact rather than a plateau point. Runs 1295/1303 ship no reference
# and 1302 used a different model, so they are excluded too.
_SCAN_200G: list[tuple[int, float]] = [
    (1292, 30.0),
    (1293, 50.0),
    (1294, 70.0),
    (1296, 85.0),
    (1297, 90.0),
    (1298, 95.0),
    (1299, 100.0),
    (1300, 105.0),
    (1301, 110.0),
    (1290, 120.0),
]

# Guide σ↔λ relation, σ(µs⁻¹) = 75780 / λ_L²(nm) (GROUND_TRUTH §1/§6/§7).
_LAMBDA_COEFF = 75780.0


def _rel(run: int) -> str:
    return _DATA % run


# Larmor frequency seeds: γ_µ ≈ 0.013554 MHz/G, so 400 G ≈ 5.42 MHz and
# 200 G ≈ 2.71 MHz. Below T_c the fitted field drops (vortex diamagnetic shift),
# so the seed sits just below the applied-field value; bounds bracket it.
_FREQ_400G = (5.3, 4.8, 6.0)
_FREQ_200G = (2.7, 2.3, 3.2)


def _tf_seeds(dataset, sigma_seed: float, freq_seed: float = _FREQ_400G[0]) -> dict:
    """Seeds for the TF Gaussian model on one MUSR run.

    The loader's forward/back asymmetry carries the precession as a ~9 %
    oscillation riding on a large negative α = 1 baseline (~−23 %), which the
    additive ``Constant`` absorbs; ``A_bg`` is seeded from the good-bin median
    (the raw asymmetry saturates at ±100 % in the pre-t0 / late-time bad bins,
    so a plain mean would be poisoned).
    """
    a = np.asarray(dataset.asymmetry, dtype=float)
    baseline = float(np.nanmedian(a[np.abs(a) < 99.0]))
    return {
        "A_1": 8.0,
        "frequency": freq_seed,
        "phase": 0.0,
        "sigma": max(sigma_seed, 0.05),
        "A_bg": baseline,
    }


def _fit_sigma_series(scan, freq=_FREQ_400G, sigma_start=1.15):
    """Fit the TF Gaussian model to each run; return (T, |σ|, σ_err) arrays.

    Runs are fitted in ascending-temperature order through the same core
    :class:`FitEngine` the GUI drives, warm-starting σ downward from the plateau
    so every run lands in the correct minimum (σ falls with T). The Gaussian
    width enters squared, so its sign is a fit artefact — |σ| is reported
    (GROUND_TRUTH §9: the 200 G run-1291 fit famously returns −0.21).
    """
    from asymmetry.core.fitting.composite import CompositeModel
    from asymmetry.core.fitting.engine import FitEngine
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    model = CompositeModel(_TF_MODEL[0], operators=_TF_MODEL[1])
    engine = FitEngine()
    f_seed, f_lo, f_hi = freq

    temps, vals, errs = [], [], []
    sigma_seed = sigma_start
    for run, temp in scan:
        dataset = load_corpus_datasets([_rel(run)])[0]
        seeds = _tf_seeds(dataset, sigma_seed, f_seed)
        params = ParameterSet(
            [
                Parameter("A_1", seeds["A_1"], min=0.0, max=60.0),
                Parameter("frequency", seeds["frequency"], min=f_lo, max=f_hi),
                Parameter("phase", seeds["phase"]),
                Parameter("sigma", seeds["sigma"], min=0.0, max=3.0),
                Parameter("A_bg", seeds["A_bg"]),
            ]
        )
        result = engine.fit(dataset, model.function, params)
        by_name = {p.name: p.value for p in result.parameters}
        unc = result.uncertainties or {}
        sigma = abs(float(by_name["sigma"]))
        temps.append(float(temp))
        vals.append(sigma)
        errs.append(float(unc.get("sigma", np.nan)))
        sigma_seed = max(sigma * 0.9, 0.05)
    return np.array(temps), np.array(vals), np.array(errs)


def _sigma_rows(temps, sigma, sigma_err, field, base_run):
    """Build trend-panel row dicts for one σ(T) field scan (ascending T)."""
    order = np.argsort(temps)
    return [
        {
            "run_number": int(base_run + i),
            "run_label": f"{temps[j]:.0f} K",
            "field": float(field),
            "temperature": float(temps[j]),
            "values": {"sigma": float(sigma[j])},
            "errors": {"sigma": float(sigma_err[j])},
        }
        for i, j in enumerate(order)
    ]


def _load_trend_panel(temps, sigma, sigma_err, title):
    """Build a FitParametersPanel showing the σ(T) trend points."""
    from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel

    panel = FitParametersPanel()
    panel.load_representation_series(
        [("bscco-sigma-t", title, _sigma_rows(temps, sigma, sigma_err, 400.0, 1276))],
        select_id="bscco-sigma-t",
    )
    _process_events_for(milliseconds=80)
    return panel


def _configure_single_fit(window, seeds, positive):
    """Set up and run a single time-domain TF fit in the main window's fit panel."""
    from asymmetry.core.fitting.composite import CompositeModel
    from asymmetry.gui.panels.fit.tab_base import (
        _param_table_rows_by_name,
        _set_param_table_value,
    )

    single_tab = window._fit_panel._single_tab
    single_tab._set_composite_model(CompositeModel(_TF_MODEL[0], operators=_TF_MODEL[1]))
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


def _frame_y_to_window(window, x_min: float, x_max: float, pad_frac: float = 0.12):
    """Clamp the plot Y axis to the signal inside [x_min, x_max].

    The MUSR asymmetry saturates at ±100 % in the pre-t0 and late-time bad bins,
    so autoscaling over the full trace blows the y-axis apart and squashes the
    ~9 % oscillation into a flat line. Frame Y to the real signal in the visible
    window instead.
    """
    panel = window._plot_panel
    t = getattr(panel, "_last_plot_time", None)
    a = getattr(panel, "_last_plot_asymmetry", None)
    if t is None or a is None or not len(t):
        return
    t = np.asarray(t, dtype=float)
    a = np.asarray(a, dtype=float)
    m = (t >= x_min) & (t <= x_max) & (np.abs(a) < 99.0)
    if not np.any(m):
        return
    lo, hi = float(np.nanmin(a[m])), float(np.nanmax(a[m]))
    pad = pad_frac * (hi - lo or 1.0)
    panel.set_view_limits(x_min, x_max, lo - pad, hi + pad)


def _frame_spectrum(window, x_min: float, x_max: float, *, timeout_s: float = 12.0):
    """Poll the frequency plot until it renders, then frame [x_min, x_max]."""
    freq_panel = window._frequency_plot_panel
    deadline = time.monotonic() + timeout_s
    spectrum_x = spectrum_y = None
    while time.monotonic() < deadline:
        _process_events_for(milliseconds=100)
        x = freq_panel._last_plot_time
        y = freq_panel._last_plot_asymmetry
        if x is not None and y is not None and len(x) and float(np.nanmax(x)) >= x_max:
            spectrum_x = np.asarray(x, dtype=float)
            spectrum_y = np.asarray(y, dtype=float)
            break
    if spectrum_x is None:
        raise RuntimeError("BiSCCO frequency spectrum did not render in time")

    in_window = (spectrum_x >= x_min) & (spectrum_x <= x_max)
    peak = float(np.max(spectrum_y[in_window])) if np.any(in_window) else 1.0
    freq_panel._auto_x_btn.setChecked(False)
    freq_panel.set_view_limits(x_min, x_max, -0.05 * peak, 1.12 * peak)
    _process_events_for(milliseconds=120)


# --------------------------------------------------------------------------- #
#  1. TF vortex damping — time-domain overlay of 10 K vs 125 K (broadening).
# --------------------------------------------------------------------------- #
class BsccoTfDampingScenario(CorpusScenario):
    name = "corpus_bscco_tf_damping"
    description = (
        "Overlay of the BiSCCO 400 G TF spectra at 10 K (run 1277, mixed state) "
        "and 125 K (run 1276, normal state): the Gaussian vortex damping decays "
        "the 10 K precession by ~2 µs while the normal-state line persists."
    )
    example = EXAMPLE
    size = (1500, 920)

    def build(self) -> QWidget:
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window.resizeDocks([window._dock_data_browser], [320], Qt.Orientation.Horizontal)

        datasets = load_corpus_datasets([_rel(_RUN_SUPER), _rel(_RUN_NORMAL)])
        with window._data_browser.batch_updates():
            for dataset in datasets:
                window._data_browser.add_dataset(dataset)
        run_numbers = [int(ds.run_number) for ds in datasets]
        window._data_browser.create_data_group(run_numbers, name="BiSCCO 400 G — 10 K vs 125 K")

        window._plot_panel.set_overlay_enabled(True, emit_signal=True)
        window._data_browser._table.selectAll()
        window._on_dataset_selected(run_numbers[0])
        _process_events_for(milliseconds=150)

        # Frame the first ~4 µs, where the 10 K Gaussian envelope (σ ≈ 1.15
        # µs⁻¹ ⇒ 1/e at ~1.2 µs) collapses to the flat line while the 125 K
        # trace keeps its full amplitude. Y is clamped to the real ±9 %
        # oscillation (the raw asymmetry saturates at ±100 % in the bad bins).
        _frame_y_to_window(window, 0.1, 4.0)
        _process_events_for(milliseconds=80)
        return window


# --------------------------------------------------------------------------- #
#  2. Vortex FFT — frequency-domain broad line at base T.
# --------------------------------------------------------------------------- #
class BsccoVortexFftScenario(CorpusScenario):
    name = "corpus_bscco_vortex_fft"
    description = (
        "Frequency-domain FFT of the BiSCCO 10 K 400 G run (1277): the broad "
        "vortex-lattice field distribution p(B) near the 5.4 MHz Larmor line."
    )
    example = EXAMPLE
    size = (1500, 920)

    def build(self) -> QWidget:
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window._on_fourier()
        window.resizeDocks([window._dock_data_browser], [320], Qt.Orientation.Horizontal)

        datasets = load_corpus_datasets([_rel(_RUN_SUPER)])
        self.add_to_browser(window, datasets)
        window._on_dataset_selected(datasets[0].run_number)
        _process_events_for(milliseconds=150)

        window._on_domain_button_clicked("frequency")
        _process_events_for(milliseconds=80)

        # Frame onto the 400 G Larmor window (γ_µ·B ≈ 5.4 MHz) so the broad
        # vortex line — the frequency-domain image of the mixed-state p(B) —
        # fills the axis rather than sitting in a wide near-empty span.
        window._on_compute_fourier()
        _frame_spectrum(window, 3.5, 7.5)
        return window


# --------------------------------------------------------------------------- #
#  4. TF Gaussian-damped fit — converged σ on run 1277 (10 K).
# --------------------------------------------------------------------------- #
class BsccoTfFitScenario(CorpusScenario):
    name = "corpus_bscco_tf_fit"
    description = (
        "Converged Oscillatory × Gaussian fit on the BiSCCO 10 K 400 G run "
        "(1277): σ ≈ 1.16 µs⁻¹ vs the WiMDA reference 1.1467(75) µs⁻¹."
    )
    example = EXAMPLE
    size = (1500, 920)
    requires_fit = True

    def build(self) -> QWidget:
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window._on_fit()
        window.resizeDocks([window._dock_data_browser], [320], Qt.Orientation.Horizontal)

        datasets = load_corpus_datasets([_rel(_RUN_SUPER)])
        self.add_to_browser(window, datasets)
        window._on_dataset_selected(datasets[0].run_number)
        _process_events_for(milliseconds=80)

        _configure_single_fit(
            window,
            seeds=_tf_seeds(datasets[0], sigma_seed=1.1),
            positive=("A_1", "sigma"),
        )

        # Zoom to the first ~2.5 µs so individual precession cycles (period
        # ~0.19 µs at 5.4 MHz) and the Gaussian relaxation envelope are both
        # resolved, with Y clamped to the real oscillation.
        _frame_y_to_window(window, 0.1, 2.5)
        _process_events_for(milliseconds=80)
        return window


# --------------------------------------------------------------------------- #
#  5. σ(T) headline — vortex depolarisation rate through T_c.
# --------------------------------------------------------------------------- #
class BsccoSigmaTScenario(CorpusScenario):
    name = "corpus_bscco_sigma_t"
    description = (
        "BiSCCO 400 G vortex depolarisation rate σ(T), 10–125 K: falls from "
        "≈1.15 µs⁻¹ (λ_L ≈ 255 nm) toward ≈0.055 µs⁻¹ above T_c, reproducing "
        "the 14-row WiMDA reference trend."
    )
    example = EXAMPLE
    size = (1240, 760)
    requires_fit = True

    def build(self) -> QWidget:
        temps, sigma, sigma_err = _fit_sigma_series(_SCAN_400G)
        self._temps, self._sigma, self._sigma_err = temps, sigma, sigma_err
        return _load_trend_panel(temps, sigma, sigma_err, "σ(T) — BiSCCO 400 G TF (vortex state)")

    def settle(self, widget: QWidget) -> None:
        widget._refresh_plot()
        _process_events_for(milliseconds=200)
        axes = list(widget._figure.axes)
        if axes:
            ax = axes[0]
            ax.set_ylim(-0.05, 1.30)
            # T_c marker and the σ(T→0)→λ_L note (guide formula; ⚠ GROUND_TRUTH
            # §6b: λ is unreliable at 400 G for pancake-vortex Bi-2212).
            ax.axvline(107.0, color="0.5", linestyle="--", linewidth=1.0, zorder=1)
            ax.text(
                107.0,
                1.30,
                r"  $T_\mathrm{c}\approx$ 107 K",
                color="0.35",
                fontsize=9,
                va="top",
                ha="left",
            )
            if getattr(self, "_sigma", None) is not None and len(self._sigma):
                sigma0 = float(np.max(self._sigma))
                lam = float(np.sqrt(_LAMBDA_COEFF / sigma0))
                ax.text(
                    0.02,
                    0.06,
                    rf"$\sigma(10\,\mathrm{{K}})\approx${sigma0:.2f} µs$^{{-1}}$"
                    rf"  $\Rightarrow\ \lambda_L\approx${lam:.0f} nm"
                    "\n(indicative only — pancake vortices at 400 G, GT §6b)",
                    transform=ax.transAxes,
                    color="0.3",
                    fontsize=8,
                    va="bottom",
                    ha="left",
                )
            widget._canvas.draw()
        _process_events_for(milliseconds=120)


# --------------------------------------------------------------------------- #
#  5. Field comparison — σ(T) at 400 G vs 200 G (pancake-vortex physics).
# --------------------------------------------------------------------------- #
class BsccoFieldCompareScenario(CorpusScenario):
    name = "corpus_bscco_field_compare"
    description = (
        "BiSCCO σ(T) at 400 G overlaid on 200 G in the real Fit-Parameters trend "
        "panel (multi-series overlay): the 200 G plateau (~0.9 µs⁻¹) sits below "
        "the 400 G plateau (~1.15 µs⁻¹) — the pancake-vortex field dependence of "
        "the extremely anisotropic Bi-2212 (GROUND_TRUTH §6b)."
    )
    example = EXAMPLE
    size = (1240, 760)
    requires_fit = True

    def build(self) -> QWidget:
        # Native two-series overlay in the real FitParametersPanel (PR-248): both
        # σ(T) curves are genuine per-run TF Gaussian fits via the core FitEngine.
        # The panel distinguishes them by colour + a legend of the series names.
        from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel

        t400, s400, e400 = _fit_sigma_series(_SCAN_400G, _FREQ_400G, sigma_start=1.15)
        t200, s200, e200 = _fit_sigma_series(_SCAN_200G, _FREQ_200G, sigma_start=0.90)
        # Remember the plateau values for the settle() annotation.
        self._s400_plateau = float(np.max(s400))
        self._s200_plateau = float(np.max(s200))

        panel = FitParametersPanel()
        # Series names sort alphabetically for pill/colour order, so "200 G…"
        # takes C0 and "400 G…" C1; the 400 G scan is the active series (owns the
        # table + any model-fit overlay) via ``select_id``.
        panel.load_representation_series(
            [
                ("bscco-sig-400", "400 G — TF scan", _sigma_rows(t400, s400, e400, 400.0, 1276)),
                ("bscco-sig-200", "200 G — TF scan", _sigma_rows(t200, s200, e200, 200.0, 1292)),
            ],
            select_id="bscco-sig-400",
        )
        # Overlay: the public entry point single-selects, so arm the multi-select
        # the way a Shift+click on the second pill would (no public multi-select).
        panel._set_selected_group_ids(["bscco-sig-400", "bscco-sig-200"], emit=False)
        _process_events_for(milliseconds=80)
        return panel

    def settle(self, widget: QWidget) -> None:
        widget._refresh_plot()
        _process_events_for(milliseconds=200)
        axes = list(widget._figure.axes)
        if axes:
            ax = axes[0]
            ax.set_ylim(-0.05, 1.32)
            ax.set_title("BiSCCO vortex-state σ(T): 400 G vs 200 G transverse field", fontsize=10)
            ax.axvline(107.0, color="0.5", linestyle="--", linewidth=0.9, zorder=1)
            ax.text(
                106.0,
                0.62,
                r"$T_\mathrm{c}\approx$ 107 K",
                color="0.4",
                ha="right",
                va="center",
                fontsize=9,
            )
            ax.text(
                0.015,
                0.05,
                f"400 G plateau ≈ {getattr(self, '_s400_plateau', 1.15):.2f} µs⁻¹ sits above the "
                f"200 G plateau ≈ {getattr(self, '_s200_plateau', 0.9):.2f} µs⁻¹ — the field\n"
                "dependence of σ in Bi-2212 is pancake-vortex physics, so a single λ_L is\n"
                "unreliable here (both fields < B* ≈ 500 G; GROUND_TRUTH §6b).\n"
                "Run 1291 (200 G, 10 K) excluded: documented negative-σ fit (§9).",
                transform=ax.transAxes,
                color="0.35",
                fontsize=8,
                va="bottom",
                ha="left",
            )
            widget._canvas.draw()
        _process_events_for(milliseconds=120)


register(BsccoTfDampingScenario())
register(BsccoVortexFftScenario())
register(BsccoTfFitScenario())
register(BsccoSigmaTScenario())
register(BsccoFieldCompareScenario())
