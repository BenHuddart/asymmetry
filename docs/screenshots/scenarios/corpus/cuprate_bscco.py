"""Corpus figures for the Rigi school BiSCCO transverse-field scans.

The MuSR scans span 10–125 K at 400 G and 10–120 K at 200 G. The time-domain
fits use the Gaussian component's exp(−σ_G²t²) convention. The trend figures
show the fitted widths, including the normal-state runs. The FFT and MaxEnt
figures illustrate the broadening on cooling at 400 G.
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

# Asymmetry’s Gaussian component uses exp(−σ_G²t²).
_TF_MODEL = (["Oscillatory", "Gaussian", "Constant"], ["*", "+"])

# MuSR run selections in ascending temperature order.
_RUN_SUPER = 1277  # 10 K, broad mixed-state line
_RUN_NORMAL = 1276  # 125 K, narrow normal-state line
_SCAN_400G = (*range(_RUN_SUPER, 1290), _RUN_NORMAL)

# The 200 G subset excludes the poor reference fit at 10 K (1291), the
# two-component fit (1302) and the runs without reference fits (1295, 1303).
_SCAN_200G = (1292, 1293, 1294, *range(1296, 1302), 1290)


def _rel(run: int) -> str:
    return _DATA % run


# The seed and bounds account for the lower fitted frequency in the mixed state.
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


def _fit_sigma_series(scan, field, frequency_range, sigma_seed):
    """Fit each run and retain its identity alongside the Gaussian width."""
    from asymmetry.core.fitting.composite import CompositeModel
    from asymmetry.core.fitting.engine import FitEngine
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    model = CompositeModel(_TF_MODEL[0], operators=_TF_MODEL[1])
    engine = FitEngine()
    f_seed, f_lo, f_hi = frequency_range

    rows = []
    for run in scan:
        (dataset,) = load_corpus_datasets([_rel(run)])
        if dataset.run_number != run or dataset.field != field or dataset.temperature is None:
            raise ValueError(f"Unexpected run, applied field or temperature for MuSR file {run}")
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
        if not result.success:
            raise RuntimeError(f"MuSR run {dataset.run_number} TF fit failed: {result.message}")
        sigma = float(result.parameters["sigma"].value)
        rows.append(
            {
                "run_number": dataset.run_number,
                "run_label": dataset.run_label,
                "field": dataset.field,
                "temperature": dataset.temperature,
                "values": {"sigma": sigma},
                "errors": {"sigma": result.uncertainties["sigma"]},
                "reduced_chi_squared": result.reduced_chi_squared,
                "chi_squared": result.chi_squared,
            }
        )
        sigma_seed = max(sigma * 0.9, 0.05)
    return rows


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
    description = "Gaussian-damped precession fit to the BiSCCO 10 K, 400 G run (1277)."
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


class BsccoSigmaTScenario(CorpusScenario):
    name = "corpus_bscco_sigma_t"
    description = "Fitted BiSCCO Gaussian linewidths at 400 G from 10 to 125 K."
    example = EXAMPLE
    size = (1240, 760)
    requires_fit = True

    def build(self) -> QWidget:
        from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel

        rows = _fit_sigma_series(_SCAN_400G, 400.0, _FREQ_400G, 1.15)
        panel = FitParametersPanel()
        panel.load_representation_series(
            [("bscco-sig-400", "400 G — TF scan", rows)],
            select_id="bscco-sig-400",
        )
        return panel


class BsccoFieldCompareScenario(CorpusScenario):
    name = "corpus_bscco_field_compare"
    description = (
        "BiSCCO Gaussian linewidths at 400 G and 200 G overlaid in Fit Parameters. "
        "At the lowest shared temperatures, the 400 G series is broader."
    )
    example = EXAMPLE
    size = (1240, 760)
    requires_fit = True

    def build(self) -> QWidget:
        from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel

        rows_400 = _fit_sigma_series(_SCAN_400G, 400.0, _FREQ_400G, 1.15)
        rows_200 = _fit_sigma_series(_SCAN_200G, 200.0, _FREQ_200G, 0.90)
        panel = FitParametersPanel()
        panel.load_representation_series(
            [
                ("bscco-sig-400", "400 G — TF scan", rows_400),
                ("bscco-sig-200", "200 G — TF scan", rows_200),
            ],
            select_id="bscco-sig-400",
        )
        panel.select_series(["bscco-sig-400", "bscco-sig-200"])
        return panel


# --------------------------------------------------------------------------- #
#  6. MaxEnt vortex p(B) — 10 K broad distribution vs 125 K narrow line.
# --------------------------------------------------------------------------- #
def _grab_canvas_agg(canvas, name: str, output_dir):
    """Save a drawn matplotlib canvas from its Agg buffer (byte-deterministic).

    ``QWidget.grab`` on a fresh offscreen ``FigureCanvas`` settles the last pixel
    column non-deterministically offscreen; the Agg RGBA buffer is byte-stable
    (same pattern as ``benzene_multi``).
    """
    from PySide6.QtGui import QImage

    arr = np.asarray(canvas.buffer_rgba())
    height, width = arr.shape[:2]
    image = QImage(arr.tobytes(), width, height, QImage.Format.Format_RGBA8888)
    out_path = output_dir / f"{name}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not image.save(str(out_path), "PNG"):
        raise RuntimeError(f"Failed to save screenshot to {out_path}")
    return out_path


def _bscco_maxent_spectrum(run_number: int):
    """One out-of-the-box MaxEnt spectrum for a BiSCCO 400 G run.

    Defaults everywhere except a GUI-default 1024-point spectrum: the field-
    derived auto window centres on γ_µ·400 G ≈ 5.42 MHz, ``auto_phase_seed``
    recovers the four MUSR quadrant-group phases (~90° apart), and the
    reconstruction is unit-area normalised. Returns
    ``(frequencies, spectrum, χ²/N, peak MHz)``; the χ²/N is recomputed from the
    time-domain reconstruction so it equals the engine's by identity.
    """
    from asymmetry.core.maxent import (
        MaxEntConfig,
        maxent,
        reconstruct_group_signals,
    )

    dataset = load_corpus_datasets([_rel(run_number)])[0]
    config = MaxEntConfig(n_spectrum_points=1024)
    result = maxent(dataset.run, config, cycles=12)
    recon = reconstruct_group_signals(result.maxent_input, result.state)
    n_obs = sum(g.n_obs for g in recon.values()) or 1
    chi2_over_n = sum(g.chi2 for g in recon.values()) / n_obs
    spectrum = np.asarray(result.spectrum, dtype=float)
    frequencies = np.asarray(result.frequencies_mhz, dtype=float)
    peak_mhz = float(frequencies[int(np.argmax(spectrum))])
    return frequencies, spectrum, float(chi2_over_n), peak_mhz


class BsccoMaxEntScenario(CorpusScenario):
    name = "corpus_bscco_maxent"
    description = (
        "BiSCCO 400 G MaxEnt (PR #249): the 10 K broad vortex field distribution "
        "p(B) overlaid on the 125 K normal-state narrow line, both unit-area "
        "spectra near the 5.42 MHz Larmor frequency. The vortex lattice spreads "
        "the same area into a broad p(B), and the 10 K peak sits just below the "
        "normal-state line (diamagnetic shift) — the guide's FFT-vs-MaxEnt "
        "comparison, which previously diverged and had to be dropped."
    )
    example = EXAMPLE
    size = (1180, 720)
    requires_fit = True  # real MaxEnt reconstruction runs at capture time

    def __init__(self) -> None:
        super().__init__()
        self._summary: dict[str, float] = {}

    def capture(self, ctx):  # noqa: D401 - standalone-figure render
        from asymmetry.gui.styles import tokens
        from asymmetry.gui.widgets.mpl_canvas import create_canvas

        # Warm a throwaway canvas so the first real draw is deterministic.
        _warm_fig, _warm_canvas = create_canvas(layout="tight")
        _warm_canvas.draw()
        _process_events_for(milliseconds=60)

        f10, s10, chi10, pk10 = _bscco_maxent_spectrum(_RUN_SUPER)
        f125, s125, chi125, pk125 = _bscco_maxent_spectrum(_RUN_NORMAL)
        self._summary = {
            "chi2_over_n_10K": chi10,
            "chi2_over_n_125K": chi125,
            "peak_mhz_10K": pk10,
            "peak_mhz_125K": pk125,
        }

        larmor = 5.421  # γ_µ · 400 G
        figure, canvas = create_canvas(layout="tight")
        ax = figure.add_subplot(111)
        # 10 K vortex distribution — filled so it reads as p(B).
        ax.fill_between(f10, 0.0, s10, color=tokens.TRACE_BLUE, alpha=0.28, linewidth=0)
        ax.plot(
            f10,
            s10,
            color=tokens.TRACE_BLUE,
            lw=1.5,
            label=f"10 K vortex state — broad p(B)  (χ²/N = {chi10:.2f})",
        )
        # 125 K normal-state narrow line.
        ax.plot(
            f125,
            s125,
            color=tokens.TRACE_VERMILLION,
            lw=1.5,
            label=f"125 K normal state — narrow line  (χ²/N = {chi125:.2f})",
        )
        ax.axvline(larmor, color="0.55", linestyle="--", linewidth=1.0, zorder=1)
        ymax = float(max(s10.max(), s125.max()))
        ax.set_xlim(4.4, 6.4)
        ax.set_ylim(-0.03 * ymax, 1.10 * ymax)
        ax.text(
            larmor + 0.03,
            1.0 * ymax,
            r"$\gamma_\mu\!\cdot\!400\,$G $\approx$ 5.42 MHz",
            color="0.4",
            fontsize=8.5,
            va="top",
            ha="left",
        )
        ax.text(
            0.985,
            0.60,
            "Same unit area, spread by the\n"
            r"vortex lattice below $T_\mathrm{c}$."
            "\n"
            f"10 K peak {pk10:.2f} MHz sits below the\n"
            f"125 K line {pk125:.2f} MHz — diamagnetic shift.\n"
            "(MaxEnt line shape is an estimate,\nnot a calibrated width.)",
            transform=ax.transAxes,
            color="0.35",
            fontsize=8,
            va="center",
            ha="right",
        )
        ax.set_xlabel(r"Frequency $\nu$ (MHz)")
        ax.set_ylabel("MaxEnt spectral density (unit area)")
        ax.set_title(
            "BiSCCO 400 G MaxEnt: vortex broadening of the internal-field distribution p(B)",
            fontsize=10.5,
        )
        ax.legend(fontsize=8.5, loc="upper left")

        canvas.resize(*self.size)
        canvas.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        canvas.show()
        _process_events_for(milliseconds=200)
        canvas.draw()
        _process_events_for(milliseconds=80)
        canvas.draw()
        out = _grab_canvas_agg(canvas, self.name, ctx.output_dir)
        canvas.close()
        canvas.deleteLater()
        _process_events_for(milliseconds=40)
        return out


register(BsccoTfDampingScenario())
register(BsccoVortexFftScenario())
register(BsccoTfFitScenario())
register(BsccoSigmaTScenario())
register(BsccoMaxEntScenario())
register(BsccoFieldCompareScenario())
