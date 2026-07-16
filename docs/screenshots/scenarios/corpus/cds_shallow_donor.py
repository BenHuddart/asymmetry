"""Corpus scenarios — Shallow donor state in cadmium sulphide (Semiconductors).

Drives the Asymmetry GUI through the WiMDA muon-school **CdS shallow-donor**
example on the **real ISIS EMU HDF4 NeXus corpus** (``EMU00020711``–``20733``,
23 runs, TF **100 G**, logged sample T ≈ 5–285 K). The teaching guide is the
spec: J. M. Gil *et al.*, "Novel Muonium state in CdS," Phys. Rev. Lett. **83**
(1999) 5294; ionisation fitting in Phys. Rev. B **64** (2001) 075205. See the
example's ``GROUND_TRUTH.md`` (audit-corrected).

Physics (GROUND_TRUTH §4/§5/§6). At low T the muon captures an electron and
forms a **neutral shallow-donor muonium** (Mu⁰) whose hyperfine coupling is
tiny — ~10⁻⁴ of vacuum Mu (4463 MHz) — and **anisotropic**: A∥ = 335(8) kHz,
A⊥ = 199(6) kHz (Gil *et al.*). At 100 G this is deep Paschen-Back, so the TF
line is a **central diamagnetic (Mu⁺) Larmor line** flanked by **satellites
split by the hyperfine constant A** (guide: satellite splitting = A). The
observed splitting ≈ 0.24 MHz sits between the A⊥ and A∥ orientation limits
(a wurtzite powder samples Δν(Θ) = A∥cos²Θ + A⊥sin²Θ, GROUND_TRUTH §6). On
**warming the donor ionises** (E_i ~ tens of meV): the satellites vanish and a
single sharp Mu⁺ line remains (onset ~30 K, GROUND_TRUTH §6). The satellites
also beat against the central line in the time domain (guide §4).

This is subtle spectroscopy near the FFT resolution limit: the loader delivers
1979 bins at 16 ns over a **31.75 µs** window ⇒ ~32 kHz raw frequency
resolution. Every FFT here runs through the program's own
``compute_average_group_spectrum`` with a **Hann** apodisation and **zero
padding** (the ``padding`` config field) so the ~0.12 MHz central-satellite
offset — ~4 raw bins — resolves cleanly. See ``NOTES_cds.md`` for the honest
resolution discussion and satellite-splitting comparison.

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
  on a 50 K spectrum (single sharp line) — the satellites disappear as the
  donor ionises.
* ``corpus_cds_time_beats`` — the corresponding slow beat envelope in the
  time-domain asymmetry (cold, beating) vs the warm run (single line, no beat).
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

# Observed diamagnetic (Mu⁺) Larmor line at 100 G. γ_µ/2π = 13.554 kHz/G gives
# 1.355 MHz at 100.00 G; the fitted line sits at ~1.39 MHz (a small calibration
# offset, GROUND_TRUTH §7). Annotations centre on the observed line.
_CENTRE_MHZ = 1.390
# Anisotropic hyperfine constants (Gil et al.; GROUND_TRUTH §6). The satellite
# splitting equals A, so satellites sit at centre ± A/2.
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


def _rel(run: int) -> str:
    return str(corpus_path(_DATA % run))


# --------------------------------------------------------------------------- #
#  Program-engine FFT (the real Fourier pipeline the panel drives).
# --------------------------------------------------------------------------- #
def _spectrum(run: int, label: str, *, padding: int = 4, normalise_centre: bool = False):
    """Hann-apodised, zero-padded FFT of *run* via the core Fourier routine.

    Runs through ``compute_average_group_spectrum`` — the same average grouped
    FFT the Fourier panel computes — so the render is the program's own output.
    A Hann window plus ``padding`` zero-padding lifts the effective frequency
    resolution from the raw ~32 kHz bin to ~8 kHz, enough to separate the
    ~0.12 MHz central-satellite offset. The overlay legend reads
    ``metadata['run_label']``, so a physics label replaces "20721 Average".

    With *normalise_centre* the spectrum is scaled to unit central-line peak, so
    a cold (3-line) and warm (1-line) run overlay on a common scale where the
    satellite presence/absence — not the absolute central-line height — is the
    visual message.
    """
    from asymmetry.core.fourier import GroupSpectrumConfig, compute_average_group_spectrum

    ds = load_corpus_datasets([_DATA % run])[0]
    run_obj = ds.run
    gids = list(run_obj.grouping["groups"].keys())
    cfg = GroupSpectrumConfig(
        display="(Power)^1/2",
        window="hann",
        padding=padding,
        selected_group_ids=gids,
        group_phase_degrees={g: 0.0 for g in gids},
        exclusion_ranges=[],
        t_min_us=0.0,
        t_max_us=31.0,
        subtract_average_signal=True,
    )
    spectrum = compute_average_group_spectrum(run_obj, cfg)
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
    sp = _spectrum(run, "", padding=4)
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
    y0, y1 = ax.get_ylim()
    # Central diamagnetic line. Mathtext labels so the ⁺/∥/⊥ physics symbols
    # render regardless of the panel's UI font (IBM Plex Mono lacks them).
    ax.axvline(_CENTRE_MHZ, color="0.45", lw=1.0, ls="--", zorder=1)
    ax.text(
        _CENTRE_MHZ, y1 * 0.97, r" $\mathrm{Mu}^+$", color="0.35", fontsize=9, ha="left", va="top"
    )
    pairs = [(_A_PERP_MHZ, r"$A_\perp$", "#1f77b4")]
    if both_pairs:
        pairs.append((_A_PAR_MHZ, r"$A_\parallel$", "#d62728"))
    for a, tag, colour in pairs:
        for sign in (-1.0, +1.0):
            x = _CENTRE_MHZ + sign * a / 2.0
            ax.axvline(x, color=colour, lw=1.0, ls=":", alpha=0.8, zorder=1)
        ax.text(
            _CENTRE_MHZ + a / 2.0,
            y1 * 0.80,
            f" {tag}",
            color=colour,
            fontsize=8,
            ha="left",
            va="top",
        )
    panel._figure.canvas.draw_idle()


# The satellite triplet the guide flags at 100 G (central Mu⁺ ± A/2, with the
# observed inner-pair splitting A ≈ 0.24 MHz, GROUND_TRUTH §6): 1.27 / 1.39 /
# 1.51 MHz.
_TRIPLET_MHZ = (_CENTRE_MHZ - 0.12, _CENTRE_MHZ, _CENTRE_MHZ + 0.12)


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


def _cds_maxent_triplet(
    run: int,
    *,
    f_lo: float = 0.9,
    f_hi: float = 1.9,
    binning: int = 4,
    points: int = 128,
    cycles: int = 20,
):
    """MaxEnt reconstruction of the CdS TF line over a tight window.

    The 100 G run's satellites straddle the ~1.39 MHz central line by only
    ~0.12 MHz — a few raw FFT bins. The field-derived auto window (±300 G ≈
    ±4 MHz) is far too wide at this low field (README lesson), so an **explicit
    tight window** 0.9–1.9 MHz plus modest **time binning** (×4, lifting the
    post-binning Nyquist just clear of the window) and a compact 128-point grid
    let MaxEnt super-resolve the triplet into three clean lobes the FFT can only
    blur. Stable at 10–25 cycles. Returns ``(frequencies, spectrum, χ²/N)`` —
    χ²/N recomputed from the time-domain reconstruction so it equals the
    engine's by identity. On this high-statistics real F/B run χ²/N plateaus
    ≈ 4.5 (README: expected on some real runs); the three lobes still land
    dead-on 1.27 / 1.39 / 1.51 MHz.
    """
    from asymmetry.core.maxent import MaxEntConfig, maxent, reconstruct_group_signals

    dataset = load_corpus_datasets([_DATA % run])[0]
    config = MaxEntConfig(
        auto_window=False,
        f_min_mhz=f_lo,
        f_max_mhz=f_hi,
        time_binning_factor=binning,
        n_spectrum_points=points,
    )
    result = maxent(dataset.run, config, cycles=cycles)
    recon = reconstruct_group_signals(result.maxent_input, result.state)
    n_obs = sum(g.n_obs for g in recon.values()) or 1
    chi2_over_n = sum(g.chi2 for g in recon.values()) / n_obs
    spectrum = np.asarray(result.spectrum, dtype=float)
    frequencies = np.asarray(result.frequencies_mhz, dtype=float)
    return frequencies, spectrum, float(chi2_over_n)


# --------------------------------------------------------------------------- #
#  1. Low-T lineshape — central Mu⁺ line + Mu⁰ satellites.
# --------------------------------------------------------------------------- #
class CdsLowTLineshapeScenario(CorpusScenario):
    name = "corpus_cds_low_t_lineshape"
    description = (
        "Program FFT of the coldest CdS run (20721, 5.2 K, TF 100 G) zoomed on "
        "the Larmor line: the central diamagnetic Mu⁺ line at ~1.39 MHz flanked "
        "by two symmetric shallow-donor Mu⁰ satellites (splitting ≈ 0.24 MHz = "
        "the hyperfine constant), with the expected A∥ / A⊥ positions marked."
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
        "Headline: the 5.2 K CdS spectrum (central Mu⁺ line + Mu⁰ satellites) "
        "overlaid on the 50 K spectrum (a single sharp Mu⁺ line). Warming "
        "ionises the shallow donor, so the satellites vanish — the CdS "
        "shallow-donor signature (GROUND_TRUTH §4/§6)."
    )
    example = EXAMPLE
    size = (1180, 760)

    def build(self) -> QWidget:
        from asymmetry.gui.panels.plot_panel import PlotPanel

        # Normalise each spectrum to its own central-line peak so the two
        # overlay on a common scale: the story is the satellites' presence
        # (cold) vs absence (warm), not the central line's absolute height.
        cold = _spectrum(_COLD_RUN, "5.2 K — Mu0 satellites", normalise_centre=True)
        warm = _spectrum(_WARM_RUN, "50 K — Mu+ only", normalise_centre=True)

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
        "its Mu⁰ satellites beat against each other, giving the pronounced "
        "envelope the guide flags (§4) — a node near ~3 µs and an antinode near "
        "~8 µs (beat period ~8 µs = 1/(central–satellite offset ≈ 0.12 MHz))."
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
#  5. MaxEnt super-resolution — FFT vs MaxEnt on the cold satellite triplet.
# --------------------------------------------------------------------------- #
class CdsMaxEntTripletScenario(CorpusScenario):
    name = "corpus_cds_maxent_triplet"
    description = (
        "MaxEnt super-resolution flagship (run 20721, 5.2 K, TF 100 G): the "
        "MaxEnt reconstruction resolves the 1.27 / 1.39 / 1.51 MHz shallow-donor "
        "satellite triplet (central Mu⁺ ± A/2) side-by-side with the raw program "
        "FFT, which at the native ~32 kHz bin cannot separate the lines. The "
        "frequency-domain MaxEnt-vs-FFT comparison the guide (§4) calls for."
    )
    example = EXAMPLE
    size = (1320, 640)
    requires_fit = True  # real MaxEnt reconstruction runs at capture time

    def __init__(self) -> None:
        super().__init__()
        self._summary: dict[str, float] = {}

    def capture(self, ctx):  # noqa: D401 - standalone side-by-side figure
        from asymmetry.gui.styles import tokens
        from asymmetry.gui.widgets.mpl_canvas import create_canvas

        # Warm a throwaway canvas so the first real draw is deterministic.
        _warm_fig, _warm_canvas = create_canvas(layout="tight")
        _warm_canvas.draw()
        _process_events_for(milliseconds=60)

        # The raw program FFT at native resolution (no zero-padding interpolation):
        # the satellites are only a few bins off the central line, so they blur
        # into shoulders — the FFT "cannot" resolve the triplet.
        fft = _spectrum(_COLD_RUN, "", padding=1)
        f_fft = np.asarray(fft.time, float)
        s_fft = np.asarray(fft.asymmetry, float)

        f_me, s_me, chi2 = _cds_maxent_triplet(_COLD_RUN)
        self._summary = {"chi2_over_n": chi2}

        x_min, x_max = 1.0, 1.8
        figure, canvas = create_canvas(layout="tight")
        ax_fft = figure.add_subplot(1, 2, 1)
        ax_me = figure.add_subplot(1, 2, 2)

        # Left: the noisy FFT that cannot separate the lines.
        band = (f_fft >= x_min) & (f_fft <= x_max)
        pk_fft = float(np.nanmax(s_fft[band])) if np.any(band) else 1.0
        ax_fft.plot(f_fft, s_fft / pk_fft, color=tokens.TRACE_VERMILLION, lw=1.3)
        ax_fft.set_title("Program FFT — triplet unresolved", fontsize=10.5)
        ax_fft.set_ylabel("Spectral amplitude (norm.)")

        # Right: MaxEnt resolves the three lines.
        band_me = (f_me >= x_min) & (f_me <= x_max)
        pk_me = float(np.nanmax(s_me[band_me])) if np.any(band_me) else 1.0
        ax_me.fill_between(f_me, 0.0, s_me / pk_me, color=tokens.TRACE_BLUE, alpha=0.25, lw=0)
        ax_me.plot(f_me, s_me / pk_me, color=tokens.TRACE_BLUE, lw=1.5)
        ax_me.set_title(f"MaxEnt — triplet resolved  (χ²/N = {chi2:.2f})", fontsize=10.5)

        for ax in (ax_fft, ax_me):
            for i, fx in enumerate(_TRIPLET_MHZ):
                colour = "0.45" if i == 1 else "#1f77b4"
                ax.axvline(fx, color=colour, ls="--" if i == 1 else ":", lw=1.0, alpha=0.8)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(-0.04, 1.16)
            ax.set_xlabel(r"Frequency $\nu$ (MHz)")
        ax_fft.text(
            _TRIPLET_MHZ[1],
            1.12,
            r" $\mathrm{Mu}^+$",
            color="0.35",
            fontsize=8.5,
            ha="left",
            va="top",
        )
        for fx in (_TRIPLET_MHZ[0], _TRIPLET_MHZ[2]):
            ax_me.text(fx, 1.12, f"{fx:.2f}", color="#1f77b4", fontsize=8, ha="center", va="top")
        figure.suptitle(
            "CdS shallow donor (20721, 5.2 K, 100 G): MaxEnt super-resolves the "
            "Mu⁰ satellite triplet the FFT blurs",
            fontsize=11,
        )

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


register(CdsLowTLineshapeScenario())
register(CdsColdVsWarmScenario())
register(CdsTimeBeatsScenario())
register(CdsNeutralFractionTScenario())
register(CdsMaxEntTripletScenario())
