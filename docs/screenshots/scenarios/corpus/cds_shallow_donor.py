"""Corpus scenarios — Shallow donor state in cadmium sulphide (Semiconductors).

Drives the Asymmetry GUI through the WiMDA muon-school **CdS shallow-donor**
example on the **real ISIS EMU HDF4 NeXus corpus** (``EMU00020711``–``20733``,
23 runs, TF **100 G**, logged sample T ≈ 5–285 K). The teaching guide is the
spec: J. M. Gil *et al.*, "Novel Muonium state in CdS," Phys. Rev. Lett. **83**
(1999) 5294; ionisation fitting in Phys. Rev. B **64** (2001) 075205. See the
example's ``GROUND_TRUTH.md`` (audit-corrected).

At low temperature the muon captures an electron to form neutral
shallow-donor muonium (Mu⁰). Its weak, anisotropic hyperfine interaction
produces satellite lines around the separate diamagnetic (Mu⁺) Larmor line.
The warm run provides a control with one dominant diamagnetic line. The
comparison uses the grouped FFT and MaxEnt on both runs, including the
weighted time-domain residuals that the MaxEnt overlay displays.

Coldest run: **20721** (Tlog = 5.175 K, 45.9 MEv — the highest-statistics run,
deepest in the Mu⁰ phase). ⚠ Runs 20711–20721 park the cryostat **setpoint** at
1.000 K while the sample actually cools 285→5 K; the **logged** temperature is
the physical axis (GROUND_TRUTH §3). The T-trend scenario therefore uses only
the **stable-setpoint** block 20722–20733 (Tset = Tlog, 10–50 K), which already
brackets the Mu⁰ onset — sidestepping the temperature pitfall entirely.

Scenarios registered:

* ``corpus_cds_low_t_lineshape`` — program FFT of the coldest run (20721,
  5.2 K) zoomed on the 100 G Larmor line: central Mu⁺ line + two symmetric
  Mu⁰ satellites, with the expected A∥ / A⊥ satellite positions marked.
* ``corpus_cds_cold_vs_warm`` — headline: the 5.2 K spectrum (split) overlaid
  on a 50 K spectrum with one dominant line.
* ``corpus_cds_maxent_triplet`` — matched cold and warm FFTs and MaxEnt
  spectra, with MaxEnt residuals for each run.
* ``corpus_cds_time_beats`` — the slow beat envelope in the cold run's
  time-domain asymmetry.
* ``corpus_cds_neutral_fraction_t`` — Mu⁰ neutral fraction vs T across the
  stable runs 20722–20733, fitted with the OrderParameter power law → the
  ionisation onset Tc ≈ 30 K (GROUND_TRUTH §6 "satellites appear below ~30 K").

``requires_fit = True`` where a real iminuit computation runs at capture time.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from ._corpus import (
    CorpusScenario,
    _process_events_for,
    corpus_path,
    load_corpus_datasets,
    register,
)

EXAMPLE = "Semiconductors/Shallow donor state in cadmium sulphide"
_DATA = EXAMPLE + "/Data/EMU000%d.nxs"

# Gil et al., PRL 83, 5294, Fig. 1 and Eq. (1). At 10 mT each pair is
# symmetric about the Larmor frequency and its separation depends on orientation.
_REFERENCE_CENTRE_MHZ = 1.38
_REFERENCE_INNER_PAIR_MHZ = 0.214
_A_PAR_MHZ = 0.335  # A∥, Θ = 0°
_A_PERP_MHZ = 0.199  # A⊥, Θ = 90°

_COLD_RUN = 20721  # Tlog 5.175 K, 45.9 MEv — coldest, highest statistics
_WARM_RUN = 20729  # Tlog 50.0 K — ionised, single Mu⁺ line

# Stable-setpoint block (Tset = Tlog): run → logged sample temperature (§3).
_STABLE_RUNS: list[tuple[int, float]] = [
    (20722, 10.013),
    (20730, 12.050),
    (20723, 14.983),
    (20731, 16.948),
    (20724, 17.987),
    (20732, 18.897),
    (20725, 20.038),
    (20733, 21.028),
    (20726, 21.996),
    (20727, 25.131),
    (20728, 29.948),
    (20729, 50.012),
]

# FFT band definitions (MHz): the central Mu⁺ line and the two satellite wings.
_CEN_BAND = (1.36, 1.42)
_SATL_BAND = (1.21, 1.32)
_SATU_BAND = (1.46, 1.57)
_COMPARISON_BAND_MHZ = (1.15, 1.65)
_PEAK_PROMINENCE_FRACTION = 0.20
_MIN_MARKER_SEPARATION_MHZ = 0.03


def _rel(run: int) -> str:
    return str(corpus_path(_DATA % run))


# --------------------------------------------------------------------------- #
#  Program-engine FFT (the real Fourier pipeline the panel drives).
# --------------------------------------------------------------------------- #
def _cds_grouped_fft(run, *, window: str = "lorentzian"):
    """Grouped-count FFT with a common 0–31 µs input window and padding."""
    from asymmetry.core.fourier import GroupSpectrumConfig, compute_average_group_spectrum

    cfg = GroupSpectrumConfig(
        display="(Power)^1/2",
        window=window,
        filter_time_constant_us=6.0,
        padding=4,
        t_min_us=0.0,
        t_max_us=31.0,
        subtract_average_signal=True,
    )
    spectrum = compute_average_group_spectrum(run, cfg)
    if spectrum is None:
        raise RuntimeError("CdS FFT requires the two detector groups")
    return spectrum


def _spectrum(run: int, label: str, *, window: str = "lorentzian", normalise_centre: bool = False):
    dataset = load_corpus_datasets([_DATA % run])[0]
    spectrum = _cds_grouped_fft(dataset.run, window=window)
    if normalise_centre:
        f = np.asarray(spectrum.time, float)
        peak = _band_peak(f, np.asarray(spectrum.asymmetry, float), _CEN_BAND)
        if peak:
            spectrum.asymmetry = np.asarray(spectrum.asymmetry, float) / peak
    if isinstance(spectrum.metadata, dict):
        spectrum.metadata["run_label"] = label
    return spectrum


def _band_peak(freq: np.ndarray, spec: np.ndarray, band: tuple[float, float]) -> float:
    m = (freq > band[0]) & (freq < band[1])
    return float(spec[m].max()) if np.any(m) else 0.0


def _neutral_fraction(run: int) -> float:
    """Mu⁰ neutral fraction proxy = satellite power / (satellite + central).

    The two Mu⁰ satellite peak heights over their sum with the central Mu⁺
    line: ~0.68 deep in the ordered phase (matching the program note's base-T
    ≈ 0.66) collapsing toward the warm background as the donor ionises. The
    complement of the central-line dominance (GROUND_TRUTH §4/§6).
    """
    sp = _spectrum(run, "", window="hann")
    f = np.asarray(sp.time, float)
    s = np.asarray(sp.asymmetry, float)
    c = _band_peak(f, s, _CEN_BAND)
    sl = _band_peak(f, s, _SATL_BAND)
    su = _band_peak(f, s, _SATU_BAND)
    return (sl + su) / (sl + su + c) if (sl + su + c) else 0.0


def _wait_until(predicate, *, timeout_ms: int, poll_ms: int = 40) -> None:
    elapsed = 0
    while elapsed < timeout_ms:
        if predicate():
            return
        _process_events_for(milliseconds=poll_ms)
        elapsed += poll_ms


def _annotate_satellites(panel, *, both_pairs: bool = True) -> None:
    """Mark the central Mu⁺ line and the expected A∥ / A⊥ satellite positions.

    Drawn straight onto the frequency PlotPanel's matplotlib axis after the
    spectra are plotted, so the marks travel with the program's own render.
    """
    ax = panel._figure.axes[0]
    _, y1 = ax.get_ylim()
    # Mathtext supplies the ⁺, ∥, and ⊥ glyphs used in the guide labels.
    ax.axvline(_REFERENCE_CENTRE_MHZ, color="0.45", lw=1.0, ls="--", zorder=1)
    ax.text(
        _REFERENCE_CENTRE_MHZ,
        y1 * 0.97,
        r" $\mathrm{Mu}^+$ reference",
        color="0.35",
        fontsize=9,
        ha="left",
        va="top",
    )
    pairs = [(_A_PERP_MHZ, r"$A_\perp$", "#1f77b4")]
    if both_pairs:
        pairs.append((_A_PAR_MHZ, r"$A_\parallel$", "#d62728"))
    for a, tag, colour in pairs:
        for sign in (-1.0, +1.0):
            x = _REFERENCE_CENTRE_MHZ + sign * a / 2.0
            ax.axvline(x, color=colour, lw=1.0, ls=":", alpha=0.8, zorder=1)
        ax.text(
            _REFERENCE_CENTRE_MHZ + a / 2.0,
            y1 * 0.80,
            f" {tag} guide",
            color=colour,
            fontsize=8,
            ha="left",
            va="top",
        )
    panel._figure.canvas.draw_idle()


def _comparison_grid_maxima(frequencies: np.ndarray, spectrum: np.ndarray) -> np.ndarray:
    """Select prominent grid maxima and leave room between their markers."""
    from scipy.signal import find_peaks

    lo, hi = _COMPARISON_BAND_MHZ
    in_view = np.flatnonzero((frequencies >= lo) & (frequencies <= hi))
    values = spectrum[in_view]
    grid_step = frequencies[1] - frequencies[0]
    peaks, _ = find_peaks(
        values,
        prominence=_PEAK_PROMINENCE_FRACTION * values.max(),
        distance=_MIN_MARKER_SEPARATION_MHZ / grid_step,
    )
    return in_view[peaks]


def _grab_canvas_agg(canvas, name: str, output_dir):
    """Save a standalone matplotlib canvas straight from its Agg buffer."""
    from PySide6.QtGui import QImage

    canvas.draw()
    arr = np.asarray(canvas.buffer_rgba())
    height, width = arr.shape[:2]
    image = QImage(arr.tobytes(), width, height, QImage.Format.Format_RGBA8888)
    out_path = output_dir / f"{name}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not image.save(str(out_path), "PNG"):
        raise RuntimeError(f"Failed to save screenshot to {out_path}")
    return out_path


def _cds_maxent_reconstruction(run):
    """Compare the runs with the same MaxEnt settings and overlay residuals."""
    from asymmetry.core.maxent import MaxEntConfig, maxent, reconstruct_group_signals

    config = MaxEntConfig(
        auto_window=False,
        f_min_mhz=0.9,
        f_max_mhz=1.9,
        time_binning_factor=4,
        n_spectrum_points=128,
    )
    result = maxent(run, config, cycles=20)
    return result, reconstruct_group_signals(result.maxent_input, result.state)


# --------------------------------------------------------------------------- #
#  1. Low-T lineshape — central Mu⁺ line + Mu⁰ satellites.
# --------------------------------------------------------------------------- #
class CdsLowTLineshapeScenario(CorpusScenario):
    name = "corpus_cds_low_t_lineshape"
    description = (
        "Program FFT of the coldest CdS run (20721, 5.2 K, TF 100 G) zoomed on "
        "the diamagnetic Mu⁺ line and candidate shallow-donor Mu⁰ satellites, "
        "with fixed A∥ and A⊥ guides from Gil et al."
    )
    example = EXAMPLE
    size = (1180, 760)

    def build(self) -> QWidget:
        from asymmetry.gui.panels.plot_panel import PlotPanel

        cold = _spectrum(_COLD_RUN, "5.2 K — Mu⁰ (ordered)")
        panel = PlotPanel(domain="frequency")
        panel.plot_datasets([cold])
        _process_events_for(milliseconds=120)

        f = np.asarray(cold.time, float)
        s = np.asarray(cold.asymmetry, float)
        x_min, x_max = 1.10, 1.66
        band = (f >= x_min) & (f <= x_max)
        peak = float(np.nanmax(s[band])) if np.any(band) else 1.0
        panel.set_view_limits(x_min, x_max, -0.04 * peak, 1.18 * peak)
        _process_events_for(milliseconds=80)
        _annotate_satellites(panel, both_pairs=True)
        _process_events_for(milliseconds=80)
        return panel

    def settle(self, widget: QWidget) -> None:
        _process_events_for(milliseconds=200)
        _annotate_satellites(widget, both_pairs=True)
        _process_events_for(milliseconds=120)


# --------------------------------------------------------------------------- #
#  2. Cold vs warm overlay — satellites disappear on ionisation (headline).
# --------------------------------------------------------------------------- #
class CdsColdVsWarmScenario(CorpusScenario):
    name = "corpus_cds_cold_vs_warm"
    description = (
        "The 5.2 K and 50 K CdS runs transformed with matched FFT settings. "
        "The cold trace has two satellite candidates around the central Mu⁺ "
        "line, while the warm trace has one dominant line."
    )
    example = EXAMPLE
    size = (1180, 760)

    def build(self) -> QWidget:
        from asymmetry.gui.panels.plot_panel import PlotPanel

        # Normalise each spectrum to its own central-line peak so the two
        # overlay on a common scale: the story is the satellites' presence
        # (cold) vs absence (warm), not the central line's absolute height.
        cold = _spectrum(_COLD_RUN, "5.2 K", normalise_centre=True)
        warm = _spectrum(_WARM_RUN, "50 K", normalise_centre=True)

        panel = PlotPanel(domain="frequency")
        panel.set_overlay_enabled(True)
        panel.plot_datasets([cold, warm])
        _process_events_for(milliseconds=150)

        x_min, x_max = 1.10, 1.66
        panel.set_view_limits(x_min, x_max, -0.06, 1.18)
        _process_events_for(milliseconds=120)
        _annotate_satellites(panel, both_pairs=False)
        _process_events_for(milliseconds=80)
        return panel

    def settle(self, widget: QWidget) -> None:
        _process_events_for(milliseconds=200)
        _annotate_satellites(widget, both_pairs=False)
        _process_events_for(milliseconds=120)


# --------------------------------------------------------------------------- #
#  3. Time-domain beating — the slow beat envelope (cold) vs flat (warm).
# --------------------------------------------------------------------------- #
class CdsTimeBeatsScenario(CorpusScenario):
    name = "corpus_cds_time_beats"
    description = (
        "Time-domain asymmetry of the 5.2 K CdS run: the central Mu⁺ line and "
        "the Mu⁰ satellite oscillations give a beat envelope on a slower "
        "time scale than the Larmor precession."
    )
    example = EXAMPLE
    size = (1440, 860)

    def build(self) -> QWidget:
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window.resizeDocks([window._dock_data_browser], [320], Qt.Orientation.Horizontal)

        # The cold run alone: the beat envelope reads cleanly against the raw
        # 1.39 MHz precession without a second overlaid oscillating trace.
        ds = load_corpus_datasets([_DATA % _COLD_RUN])[0]
        ds.metadata["run_label"] = "5.2 K — Mu0 beating"
        window._data_browser.add_dataset(ds)
        window._on_dataset_selected(int(ds.run_number))
        _process_events_for(milliseconds=120)
        # Frame the first ~12 µs — about 1.5 beat periods — where the envelope
        # modulation is legible before the counting statistics thin out.
        y = window._plot_panel.get_view_limits()[2:]
        window._plot_panel.set_view_limits(0.0, 12.0, *y)
        _process_events_for(milliseconds=80)
        return window


# --------------------------------------------------------------------------- #
#  4. Neutral fraction vs T — the ionisation onset (headline trend).
# --------------------------------------------------------------------------- #
class CdsNeutralFractionTScenario(CorpusScenario):
    name = "corpus_cds_neutral_fraction_t"
    description = (
        "Mu⁰ neutral fraction vs temperature across the stable-setpoint CdS "
        "runs 20722–20733 (10–50 K): the shallow-donor satellites collapse as "
        "the donor ionises. An OrderParameter fit puts the onset at Tc ≈ 30 K "
        "(GROUND_TRUTH §6: satellites appear below ~30 K)."
    )
    example = EXAMPLE
    size = (1240, 780)
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
            suggest_trend_seeds,
        )
        from asymmetry.core.fitting.parameters import Parameter, ParameterSet
        from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel

        temps = np.array([t for _r, t in _STABLE_RUNS])
        frac = np.array([_neutral_fraction(r) for r, _t in _STABLE_RUNS])
        order = np.argsort(temps)
        temps, frac = temps[order], frac[order]
        # Normalise to a 1 → 0 order parameter: warm plateau (>40 K) → 0,
        # cold plateau (<13 K) → 1, so the OrderParameter power law applies.
        f_warm = float(np.mean(frac[temps > 40]))
        f_cold = float(np.mean(frac[temps < 13]))
        amp = (frac - f_warm) / (f_cold - f_warm)
        amp_err = np.full_like(amp, 0.06)

        run_by_temp = {float(t): r for r, t in _STABLE_RUNS}
        row_dicts = [
            {
                "run_number": int(run_by_temp[float(temps[i])]),
                "run_label": f"{temps[i]:.1f} K",
                "field": 100.0,
                "temperature": float(temps[i]),
                "values": {"amplitude": float(amp[i])},
                "errors": {"amplitude": float(amp_err[i])},
            }
            for i in range(len(temps))
        ]

        model = ParameterCompositeModel(["OrderParameter"])
        seeds = suggest_trend_seeds(model, temps, amp)
        params = ParameterSet(
            [
                Parameter(name=p, value=float(seeds.get(p, model.param_defaults[p])))
                for p in model.param_names
            ]
        )
        result = fit_parameter_model(
            temps,
            amp,
            amp_err,
            model,
            params,
            x_min=float(temps.min()),
            x_max=float(temps.max()),
        )
        if not result.success:
            raise RuntimeError("CdS neutral-fraction OrderParameter fit did not converge")
        self._fit_summary = {n: float(result.parameters[n].value) for n in model.param_names}

        panel = FitParametersPanel()
        panel.load_representation_series(
            [("cds-neutral", "Mu⁰ neutral fraction — CdS", row_dicts)],
            select_id="cds-neutral",
        )
        fit_range = ModelFitRange(
            x_min=float(temps.min()),
            x_max=float(temps.max()),
            model=model,
            parameters=result.parameters,
            result=result,
        )
        panel._model_fits["amplitude"] = ParameterModelFit(
            parameter_name="amplitude",
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
            timeout_ms=20000,
        )
        _process_events_for(milliseconds=200)


# --------------------------------------------------------------------------- #
#  5. Matched spectra and time-domain residuals.
# --------------------------------------------------------------------------- #
class CdsMaxEntTripletScenario(CorpusScenario):
    name = "corpus_cds_maxent_triplet"
    description = (
        "Grouped FFT and MaxEnt for CdS runs 20721 and 20729, with matching settings "
        "within each method. Circles show prominent FFT and MaxEnt grid maxima, and "
        "dotted lines show the literature inner-pair guides. The bottom panels show "
        "the weighted MaxEnt residuals for both detector groups in each run."
    )
    example = EXAMPLE
    size = (1320, 780)
    requires_fit = True

    def capture(self, ctx):
        from asymmetry.gui.styles import tokens
        from asymmetry.gui.widgets.mpl_canvas import create_canvas

        figure, canvas = create_canvas(layout="tight")
        ax_fft = figure.add_subplot(2, 2, 1)
        ax_me = figure.add_subplot(2, 2, 2, sharey=ax_fft)
        ax_cold = figure.add_subplot(2, 2, 3)
        ax_warm = figure.add_subplot(2, 2, 4, sharex=ax_cold, sharey=ax_cold)
        lo, hi = _COMPARISON_BAND_MHZ
        for ax in (ax_fft, ax_me):
            for sign in (-1, 1):
                ax.axvline(
                    _REFERENCE_CENTRE_MHZ + sign * _REFERENCE_INNER_PAIR_MHZ / 2,
                    color="0.45",
                    ls=":",
                    lw=1.1,
                    label="Gil et al. inner pair" if sign == 1 else None,
                )
            ax.set(xlim=(lo, hi), ylim=(-0.04, 1.34), xlabel="Frequency (MHz)")
        ax_fft.set(title="Grouped FFT", ylabel="Normalised spectral amplitude")
        ax_me.set_title("MaxEnt spectrum")

        cases = (
            (_COLD_RUN, "5.2 K", tokens.TRACE_BLUE, ax_cold),
            (_WARM_RUN, "50 K", tokens.TRACE_VERMILLION, ax_warm),
        )
        for number, temperature, colour, ax_residual in cases:
            run = load_corpus_datasets([_DATA % number])[0].run
            fft = _cds_grouped_fft(run)
            result, groups = _cds_maxent_reconstruction(run)
            spectra = (
                (ax_fft, fft.time, fft.asymmetry),
                (ax_me, result.frequencies_mhz, result.spectrum),
            )
            for ax, frequency, values in spectra:
                keep = (frequency >= lo) & (frequency <= hi)
                normalised = values / values[keep].max()
                ax.plot(frequency, normalised, color=colour, label=f"{number} · {temperature}")
                peaks = _comparison_grid_maxima(frequency, values)
                ax.plot(
                    frequency[peaks],
                    normalised[peaks],
                    linestyle="none",
                    marker="o",
                    markersize=4.5,
                    markerfacecolor="white",
                    markeredgecolor=colour,
                    label="Grid maximum" if number == _COLD_RUN else None,
                )

            ax_residual.axhspan(-1, 1, color="0.92", zorder=0)
            ax_residual.axhline(0, color="0.55", lw=0.8)
            for group, group_colour in zip(
                groups.values(), (tokens.TRACE_BLUE, tokens.TRACE_VERMILLION), strict=True
            ):
                ax_residual.plot(
                    group.time_us,
                    group.residual,
                    color=group_colour,
                    lw=0.8,
                    label=group.group_name,
                )
            chi2_over_n = sum(g.chi2 for g in groups.values()) / sum(
                g.n_obs for g in groups.values()
            )
            ax_residual.set(
                title=rf"{number} · {temperature} · $\chi^2/N = {chi2_over_n:.2f}$",
                xlabel="Time (µs)",
            )
        ax_fft.legend(loc="upper center", ncol=2, fontsize=8)
        ax_cold.set_ylabel(r"Weighted residual $(d-m)/\sigma$")
        ax_warm.legend(loc="upper right", fontsize=8)
        figure.suptitle("CdS at 100 G · cold and warm spectra with MaxEnt residuals", fontsize=13)

        canvas.resize(*self.size)
        canvas.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        canvas.show()
        _process_events_for(milliseconds=200)
        out = _grab_canvas_agg(canvas, self.name, ctx.output_dir)
        canvas.close()
        canvas.deleteLater()
        _process_events_for(milliseconds=40)
        return out


register(CdsLowTLineshapeScenario())
register(CdsColdVsWarmScenario())
register(CdsTimeBeatsScenario())
register(CdsNeutralFractionTScenario())
register(CdsMaxEntTripletScenario())
