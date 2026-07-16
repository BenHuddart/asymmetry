"""Corpus scenarios — Muon spectroscopy of benzene (Chemistry, multi-technique).

Drives the Asymmetry GUI through the WiMDA muon-school **benzene** example — the
corpus's four-technique showcase for the muoniated **cyclohexadienyl radical**
C6H6Mu (Mu addition to benzene). One radical, four measurements, three
instruments; each scenario reaches the same isotropic muon hyperfine coupling
``A_µ`` from a different angle. The example's ``GROUND_TRUTH.md`` is the spec
(McKenzie *et al.*, *J. Phys. Chem. B* **117**, 13614 (2013) = ``RFpaper.pdf``,
plus the teaching guide). Hard targets: **A_µ = 514.78(4) MHz**,
**A_p = 124.6(14) MHz** at 293 K (paper Table 1, RF-µSR).

Techniques and data (GT §2/§3):

* **High-TF muon spin rotation** — GPD@PSI ``.bin``, co-add of the five 3000 G
  runs ``deltat_tdc_gpd_3678–3682`` (298.5 K, high statistics). Frequency domain:
  the diamagnetic line (≈41 MHz) + the two radical precession lines (≈209 /
  ≈306 MHz) whose **sum is A_µ** (GT §3A).
* **ALC resonance** — HiFi@ISIS ``.nxs``, liquid 300 K. The ring-proton (Δ0)
  window ``hifi00029723–29798`` (≈28.5–30.0 kG): time-integral asymmetry vs
  field with the two Δ0 resonance dips fitted (GT §3B, §7 reference ``test.sel``).
* **Repolarisation** — EMU@ISIS ``.nxs``, LF 3–4000 G, runs
  ``EMU00015958–16001`` (GT §3C). Integral asymmetry vs field: the muonium
  repolarisation rise. GT §7 warns the shipped ``Repolarisation/test.sel`` is a
  **misfiled liquid-ALC scan** — there is no reference repolarisation trend, so
  the curve here is built directly from the EMU runs.
* **RF resonance** — DEVA/MUT@ISIS ``.nxs``, field scan at fixed ν_RF =
  218.5 MHz, runs ``56426–56462`` (≈560–1080 G, 293 K; GT §3D). The W-shaped
  double dip fitted with the muon+proton spin-Hamiltonian model ``RFResonanceMuP``
  → A_µ (dip mean) and A_p (dip splitting). (This is the model of GT's parity
  check **PC1** — present in Asymmetry.)

Scenarios registered:

* ``corpus_benzene_hightf_fft``   — GPD 3000 G co-add FFT: diamagnetic + two
  radical lines; ν1+ν2 = A_µ ≈ 514 MHz (real GUI frequency panel).
* ``corpus_benzene_correlation``  — WiMDA radical **correlation** spectrum of the
  same co-add: the Breit–Rabi line-pair collapsed onto a single A_µ peak at
  ≈514 MHz (the distinctive high-TF radical render).
* ``corpus_benzene_liquid_alc``   — liquid ring-proton ALC integral scan with a
  Cubic-background + 2-Lorentzian fit; Δ0 dips ≈28.94 / ≈29.54 kG (real GUI
  ALC integral-scan view, ``requires_fit``).
* ``corpus_benzene_repolarisation`` — EMU muonium repolarisation curve, integral
  asymmetry vs LF field (3–4000 G): the Mu-fraction repolarises above ~100 G.
* ``corpus_benzene_rf_resonance`` — RF field scan + ``RFResonanceMuP`` fit: the
  W-shaped double dip → A_µ ≈ 516 MHz, A_p ≈ 135 MHz (``requires_fit``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtWidgets import QApplication, QTableWidgetItem, QWidget

from asymmetry.gui.styles import tokens

from ._corpus import CorpusScenario, _process_events_for, load_corpus_datasets, register

EXAMPLE = "Chemistry/Muon spectroscopy of benzene"
_BASE = "Chemistry/Muon spectroscopy of benzene/data"
_GPD = f"{_BASE}/High TF rotation/deltat_tdc_gpd_%d.bin"
_ALC_LIQUID = f"{_BASE}/ALC resonance/liquid/hifi%08d.nxs"
_REPOL = f"{_BASE}/Repolarisation/EMU%08d.nxs"
_RF = f"{_BASE}/RF resonance/%d.nxs"

# ── High-TF (GT §3A) ────────────────────────────────────────────────────────
_GPD_COADD = range(3678, 3683)  # five 3000 G high-statistics runs
_GPD_FIELD_GAUSS = 3000.0
# Guide FFT line positions: diamagnetic ≈41 MHz, radicals ≈209 / ≈306 MHz,
# ν1+ν2 = A_µ ≈ 515 MHz; PSI cyclotron artefact ≈51 MHz.
_A_MU_TARGET = 514.78  # paper Table 1 (RF-µSR), MHz

# ── ALC liquid ring-proton (Δ0) window (GT §3B / §7) ────────────────────────
# Runs 29723–29798 span ≈28.5–30.0 kG; two Δ0 ring-proton dips. Baseline
# (non-resonant) edges bracket the pair; a Cubic background is the WiMDA ALC
# baseline (GT §3B). Reference test.sel dips ≈28.94 / ≈29.54 kG.
_ALC_RING_RUNS = range(29723, 29799)
_ALC_BASELINE_REGIONS = [(28500.0, 28850.0), (29650.0, 30000.0)]

# ── Repolarisation (GT §3C) ─────────────────────────────────────────────────
# The primary ascending LF sweep, one clean run per field (3–4000 G). The many
# interspersed 100 G "start of cycle" monitor runs read a distinct low value
# (~13.5 %, a different acquisition state) and are excluded; run 15972 is the
# physical 100 G point (~20 %). Reverse-sweep repeats (15994–16001) are dropped.
_REPOL_PRIMARY = [
    15959,
    15960,
    15961,
    15962,
    15963,
    15964,
    15965,
    15967,
    15968,
    15969,
    15970,
    15971,
    15972,
    15973,
    15974,
    15975,
    15977,
    15978,
    15979,
    15980,
    15981,
    15982,
    15983,
    15984,
    15985,
    15987,
    15988,
    15989,
    15990,
    15991,
    15992,
    15993,
]

# ── RF resonance (GT §3D) ───────────────────────────────────────────────────
_RF_RUNS = range(56426, 56463)  # 560–1080 G at ν_RF = 218.5 MHz
_RF_NU_MHZ = 218.5
# Digitised Fig. 3a dips (GT §11): left ≈773 G, right ≈865 G.
_RF_DIP_LEFT, _RF_DIP_RIGHT = 773.0, 865.0


def _pump(milliseconds: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(int(milliseconds), loop.quit)
    loop.exec()
    QApplication.processEvents()


def _grab_canvas_agg(canvas, name: str, output_dir: Path) -> Path:
    """Save a drawn matplotlib canvas from its Agg buffer (byte-deterministic).

    ``QWidget.grab`` on a fresh offscreen FigureCanvas settles the last pixel
    column non-deterministically; the Agg RGBA buffer is byte-stable.
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


def _coadd_gpd_run():
    """Co-add the five 3000 G GPD runs at the count level (GT §3A workflow)."""
    from asymmetry.core.data.combine import combine_runs

    datasets = load_corpus_datasets([_GPD % r for r in _GPD_COADD])
    runs = [d.run for d in datasets]
    return combine_runs(runs, sign=1, run_number=-3678, label="3678–3682 co-add · 3000 G")


def _radical_spectrum(display: str):
    """Averaged grouped spectrum of the GPD co-add (FFT or correlation).

    ``display='(Power)^1/2'`` gives the plain amplitude FFT (diamagnetic + two
    radical lines); ``display='correlation'`` gives the WiMDA radical
    correlation spectrum on the hyperfine ``A`` axis (single A_µ peak). Same
    apodisation (Lorentzian τ = 3 µs) and 8× zero-padding either way (GT §3A:
    apodise 1.5→7 µs, 8× pad).
    """
    from asymmetry.core.fourier.spectrum import (
        GroupSpectrumConfig,
        compute_average_group_spectrum,
    )

    run = _coadd_gpd_run()
    gids = list(run.grouping["groups"].keys())
    cfg = GroupSpectrumConfig(
        display=display,
        window="lorentzian",
        padding=8,
        filter_time_constant_us=3.0,
        selected_group_ids=gids,
        group_phase_degrees={g: 0.0 for g in gids},
        t_min_us=0.0,
        t_max_us=7.0,
        subtract_average_signal=True,
        correlation_reference_field_gauss=(_GPD_FIELD_GAUSS if display == "correlation" else None),
    )
    spectrum = compute_average_group_spectrum(run, cfg)
    return spectrum


def _peak_in(x, y, lo, hi):
    m = (x > lo) & (x < hi)
    if not np.any(m):
        return float("nan")
    return float(x[m][int(np.argmax(y[m]))])


# ══════════════════════════════════════════════════════════════════════════
# 1. High-TF FFT — diamagnetic + two radical lines (ν1+ν2 = A_µ)
# ══════════════════════════════════════════════════════════════════════════
class BenzeneHighTfFftScenario(CorpusScenario):
    name = "corpus_benzene_hightf_fft"
    description = (
        "GPD@PSI 3000 G co-add (runs 3678–3682) frequency domain: the "
        "diamagnetic line (~41 MHz) and the two radical precession lines "
        "(~209 / ~306 MHz) whose sum is the hyperfine coupling A_µ ≈ 515 MHz "
        "(target 514.78)."
    )
    example = EXAMPLE
    size = (1180, 760)

    def build(self) -> QWidget:
        from asymmetry.gui.panels.plot_panel import PlotPanel

        spectrum = _radical_spectrum("(Power)^1/2")
        if isinstance(spectrum.metadata, dict):
            spectrum.metadata["run_label"] = "3678–3682 co-add · 3000 G"

        panel = PlotPanel(domain="frequency")
        panel.plot_datasets([spectrum])
        _process_events_for(milliseconds=150)

        x = np.asarray(spectrum.time, dtype=float)
        y = np.asarray(spectrum.asymmetry, dtype=float)
        self._panel = panel
        self._f_dia = _peak_in(x, y, 30.0, 55.0)
        self._f_r1 = _peak_in(x, y, 190.0, 235.0)
        self._f_r2 = _peak_in(x, y, 285.0, 325.0)
        band = (x >= 0.0) & (x <= 360.0)
        self._peak = float(np.nanmax(y[band])) if np.any(band) else 1.0
        return panel

    def settle(self, widget: QWidget) -> None:
        # The panel replots on show(), wiping any build()-time artists — so
        # frame + annotate here, after the show-time replot has settled.
        _process_events_for(milliseconds=200)
        panel = self._panel
        peak = self._peak
        panel.set_view_limits(0.0, 360.0, -0.03 * peak, 1.14 * peak)
        _process_events_for(milliseconds=80)
        ax = panel._figure.axes[0]
        a_mu = self._f_r1 + self._f_r2
        for f, colour, label in (
            (self._f_dia, tokens.TRACE_BLUE, f"diamagnetic\n{self._f_dia:.0f} MHz"),
            (self._f_r1, tokens.TRACE_VERMILLION, f"radical ν₁\n{self._f_r1:.0f} MHz"),
            (self._f_r2, tokens.TRACE_VERMILLION, f"radical ν₂\n{self._f_r2:.0f} MHz"),
        ):
            ax.axvline(f, color=colour, ls="--", lw=1.0, alpha=0.8)
            ax.annotate(
                label,
                xy=(f, 0.86 * peak),
                ha="center",
                va="top",
                fontsize=8.5,
                color=colour,
            )
        ax.annotate(
            f"A_µ = ν₁ + ν₂ = {a_mu:.0f} MHz   (target {_A_MU_TARGET:.1f})",
            xy=(0.98, 0.96),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=10,
            fontweight="bold",
            color=tokens.TEXT,
        )
        panel._canvas.draw()
        _process_events_for(milliseconds=60)


# ══════════════════════════════════════════════════════════════════════════
# 2. Radical correlation spectrum — A_µ directly (distinctive render)
# ══════════════════════════════════════════════════════════════════════════
class BenzeneCorrelationScenario(CorpusScenario):
    name = "corpus_benzene_correlation"
    description = (
        "WiMDA radical correlation spectrum of the GPD 3000 G co-add: the "
        "Breit–Rabi line pair collapsed by the matched filter onto a single "
        "peak at the hyperfine coupling A_µ ≈ 514 MHz (target 514.78) — the "
        "frequency-domain route to the radical's A_µ."
    )
    example = EXAMPLE
    size = (1180, 760)

    def build(self) -> QWidget:
        from asymmetry.gui.panels.plot_panel import PlotPanel

        spectrum = _radical_spectrum("correlation")
        if isinstance(spectrum.metadata, dict):
            spectrum.metadata["run_label"] = "3678–3682 co-add · correlation"

        panel = PlotPanel(domain="frequency")
        panel.plot_datasets([spectrum])
        _process_events_for(milliseconds=150)

        x = np.asarray(spectrum.time, dtype=float)
        y = np.asarray(spectrum.asymmetry, dtype=float)
        self._panel = panel
        self._a_peak = _peak_in(x, y, 450.0, 560.0)
        band = (x >= 380.0) & (x <= 660.0)
        self._peak = float(np.nanmax(y[band])) if np.any(band) else 1.0
        return panel

    def settle(self, widget: QWidget) -> None:
        _process_events_for(milliseconds=200)
        panel = self._panel
        peak = self._peak
        a_peak = self._a_peak
        panel.set_view_limits(380.0, 660.0, -0.04 * peak, 1.15 * peak)
        _process_events_for(milliseconds=80)
        ax = panel._figure.axes[0]
        ax.axvline(a_peak, color=tokens.TRACE_VERMILLION, ls="--", lw=1.1, alpha=0.85)
        ax.axvline(_A_MU_TARGET, color=tokens.TEXT_DIM, ls=":", lw=1.0)
        ax.annotate(
            f"A_µ = {a_peak:.1f} MHz\n(target {_A_MU_TARGET:.2f})",
            xy=(a_peak, 0.92 * peak),
            xytext=(a_peak + 52.0, 0.80 * peak),
            fontsize=10,
            fontweight="bold",
            color=tokens.TRACE_VERMILLION,
            arrowprops=dict(arrowstyle="->", color=tokens.TRACE_VERMILLION, lw=1.0),
        )
        panel._canvas.draw()
        _process_events_for(milliseconds=60)


# ══════════════════════════════════════════════════════════════════════════
# 3. Liquid ALC — ring-proton Δ0 dips, Cubic BG + 2 Lorentzians (real GUI)
# ══════════════════════════════════════════════════════════════════════════
class BenzeneLiquidAlcScenario(CorpusScenario):
    name = "corpus_benzene_liquid_alc"
    description = (
        "Liquid ALC ring-proton (Δ0) window: 76 HiFi runs (~28.5–30.0 kG) "
        "reduced to integral asymmetry vs field in the ALC integral-scan view, "
        "with a Cubic background + two Lorentzians. The two Δ0 dips sit at "
        "≈28.94 / ≈29.54 kG (ref test.sel 28.94 / 29.54 kG)."
    )
    example = EXAMPLE
    size = (1500, 900)
    requires_fit = True

    def build(self) -> QWidget:
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window.resizeDocks([window._dock_data_browser], [320], Qt.Orientation.Horizontal)

        datasets = load_corpus_datasets([_ALC_LIQUID % r for r in _ALC_RING_RUNS])
        with window._data_browser.batch_updates():
            for dataset in datasets:
                window._data_browser.add_dataset(dataset)
        window._data_browser.create_data_group(
            [int(ds.run_number) for ds in datasets],
            name="Benzene ALC liquid — ring-proton Δ0 window",
        )
        window._data_browser._table.selectAll()
        _pump(250)
        window._plot_workspace.set_active_view("integral_scan")
        _pump(150)
        window._alc_fit_panel.build_requested.emit()
        _pump(500)

        view = window._alc_scan_view

        # Cubic baseline over the two non-resonant edges (brackets the dip pair).
        view._baseline_model_combo.setCurrentText("Cubic")
        for _ in _ALC_BASELINE_REGIONS:
            view._add_region()
        for row, (lo, hi) in enumerate(_ALC_BASELINE_REGIONS):
            view._regions_table.setItem(row, 0, QTableWidgetItem(f"{lo:.0f}"))
            view._regions_table.setItem(row, 1, QTableWidgetItem(f"{hi:.0f}"))
        _pump(150)
        view.baseline_fit_requested.emit()
        _pump(350)

        # Two Lorentzian dips seeded a little off the resonance fields so the fit
        # visibly settles onto ≈28.94 / ≈29.54 kG. Cols: 1=B0, 2=Width, 3=Amp.
        for b0_seed in (28920.0, 29520.0):
            view._add_peak("Lorentzian")
        for row, b0_seed in enumerate((28920.0, 29520.0)):
            view._peaks_table.setItem(row, 1, QTableWidgetItem(f"{b0_seed:.0f}"))
            view._peaks_table.setItem(row, 2, QTableWidgetItem("40"))
            view._peaks_table.setItem(row, 3, QTableWidgetItem("-1.5"))
        _pump(150)
        view.peaks_fit_requested.emit()
        _pump(400)
        return window


# ══════════════════════════════════════════════════════════════════════════
# 4. Repolarisation — muonium repolarisation curve (EMU, 3–4000 G)
# ══════════════════════════════════════════════════════════════════════════
class BenzeneRepolarisationScenario(CorpusScenario):
    name = "corpus_benzene_repolarisation"
    description = (
        "EMU@ISIS muonium repolarisation: integral asymmetry vs longitudinal "
        "field (3–4000 G) for the benzene liquid. The recoverable Mu-fraction "
        "asymmetry is flat below ~100 G and repolarises through the hyperfine "
        "field toward high LF — the coupling-strength overview (GT §3C)."
    )
    example = EXAMPLE
    size = (1120, 720)

    def capture(self, ctx) -> Path:  # noqa: D401
        from asymmetry.core.transform.integral import build_field_scan
        from asymmetry.gui.widgets.mpl_canvas import create_canvas

        _warm_fig, _warm_canvas = create_canvas(layout="tight")
        _warm_canvas.draw()
        _pump(60)

        datasets = load_corpus_datasets([_REPOL % r for r in _REPOL_PRIMARY])
        scan = build_field_scan([d.run for d in datasets], method="integral", order_key="field")
        x = np.asarray(scan.x, dtype=float)
        y = np.asarray(scan.value, dtype=float) * 100.0
        e = np.asarray(scan.error, dtype=float) * 100.0

        figure, canvas = create_canvas(layout="tight")
        ax = figure.add_subplot(111)
        ax.errorbar(
            x,
            y,
            yerr=e,
            fmt="o-",
            ms=4.5,
            lw=1.3,
            color=tokens.TRACE_BLUE,
            elinewidth=0.8,
            capsize=2.0,
        )
        ax.set_xscale("log")
        ax.set_xlabel("Longitudinal field B (G, log scale)")
        ax.set_ylabel("Integral asymmetry (%)")
        ax.set_title("Benzene muonium repolarisation — recoverable asymmetry vs LF field")
        ax.grid(True, which="both", color=tokens.PLOT_GRID, linewidth=0.8)
        ax.annotate(
            "Mu-fraction repolarises\nthrough the hyperfine field",
            xy=(600.0, float(np.interp(600.0, x, y))),
            xytext=(120.0, float(y.max()) - 1.0),
            fontsize=9.5,
            color=tokens.TEXT,
            arrowprops=dict(arrowstyle="->", color=tokens.TEXT_DIM, lw=1.0),
        )

        canvas.resize(*self.size)
        canvas.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        canvas.show()
        _pump(200)
        canvas.draw()
        _pump(80)
        canvas.draw()
        out = _grab_canvas_agg(canvas, self.name, ctx.output_dir)
        canvas.close()
        canvas.deleteLater()
        _pump(40)
        return out


# ══════════════════════════════════════════════════════════════════════════
# 5. RF resonance — W-shaped double dip + RFResonanceMuP fit (A_µ, A_p)
# ══════════════════════════════════════════════════════════════════════════
class BenzeneRfResonanceScenario(CorpusScenario):
    name = "corpus_benzene_rf_resonance"
    description = (
        "RF-µSR field scan at ν_RF = 218.5 MHz (runs 56426–56462, 560–1080 G): "
        "integral asymmetry vs field with the W-shaped double dip fitted by the "
        "muon+proton spin-Hamiltonian model RFResonanceMuP → A_µ ≈ 516 MHz "
        "(mean, target 514.78) and A_p ≈ 135 MHz (splitting, target 124.6)."
    )
    example = EXAMPLE
    size = (1120, 720)
    requires_fit = True

    def __init__(self) -> None:
        super().__init__()
        self._fit_summary: dict[str, float] = {}

    def capture(self, ctx) -> Path:  # noqa: D401
        from asymmetry.core.fitting.field_scan import (
            RF_RESONANCE_COMPONENT,
            as_composite_model,
            fit_rf_resonance,
        )
        from asymmetry.core.transform.integral import build_field_scan
        from asymmetry.gui.widgets.mpl_canvas import create_canvas

        _warm_fig, _warm_canvas = create_canvas(layout="tight")
        _warm_canvas.draw()
        _pump(60)

        datasets = load_corpus_datasets([_RF % r for r in _RF_RUNS])
        scan = build_field_scan([d.run for d in datasets], method="integral", order_key="field")
        result = fit_rf_resonance(scan, nu_rf=_RF_NU_MHZ, a_mu=515.0, a_p=124.0)
        if not result.success:
            raise RuntimeError(f"RFResonanceMuP fit did not converge: {result.message}")
        a_mu = float(result.parameters["A_mu"].value)
        a_p = float(result.parameters["A_p"].value)
        self._fit_summary = {"A_mu": a_mu, "A_p": a_p}

        model = as_composite_model(RF_RESONANCE_COMPONENT)
        params = {n: result.parameters[n].value for n in model.param_names}
        x = np.asarray(scan.x, dtype=float)
        y = np.asarray(scan.value, dtype=float) * 100.0
        e = np.asarray(scan.error, dtype=float) * 100.0
        xf = np.linspace(float(x.min()), float(x.max()), 500)
        yf = np.asarray(model.function(xf, **params), dtype=float) * 100.0

        figure, canvas = create_canvas(layout="tight")
        ax = figure.add_subplot(111)
        ax.errorbar(
            x,
            y,
            yerr=e,
            fmt="o",
            ms=4.5,
            lw=0.0,
            color=tokens.TRACE_BLUE,
            elinewidth=0.8,
            capsize=2.0,
            label="integral asymmetry",
        )
        ax.plot(xf, yf, color=tokens.TRACE_VERMILLION, lw=1.8, label="RFResonanceMuP fit")
        for b in (_RF_DIP_LEFT, _RF_DIP_RIGHT):
            ax.axvline(b, color=tokens.TEXT_DIM, ls=":", lw=1.0)
        ax.set_xlabel("Longitudinal field B (G)")
        ax.set_ylabel("Integral asymmetry (%)")
        ax.set_title(
            "Benzene RF-µSR resonance (ν_RF = 218.5 MHz) — muon+proton spin-Hamiltonian fit"
        )
        ax.grid(True, color=tokens.PLOT_GRID, linewidth=0.8)
        ax.legend(loc="upper left", fontsize="small", framealpha=0.92)
        # The migrad errors here are unrealistically tight (the flat-BG model is
        # overconfident on the sloping wings), so quote the values, not spurious
        # sub-0.1 MHz precisions — A_µ is the robust read-off, A_p the weak one.
        ax.annotate(
            f"A_µ = {a_mu:.1f} MHz  [dip mean]\n"
            f"A_p = {a_p:.0f} MHz  [splitting]\n"
            f"targets 514.78 / 124.6",
            xy=(0.985, 0.04),
            xycoords="axes fraction",
            ha="right",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color=tokens.TEXT,
            bbox=dict(boxstyle="round,pad=0.4", fc=tokens.SURFACE, ec=tokens.TEXT_DIM, lw=0.8),
        )

        canvas.resize(*self.size)
        canvas.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        canvas.show()
        _pump(200)
        canvas.draw()
        _pump(80)
        canvas.draw()
        out = _grab_canvas_agg(canvas, self.name, ctx.output_dir)
        canvas.close()
        canvas.deleteLater()
        _pump(40)
        return out


register(BenzeneHighTfFftScenario())
register(BenzeneCorrelationScenario())
register(BenzeneLiquidAlcScenario())
register(BenzeneRepolarisationScenario())
register(BenzeneRfResonanceScenario())
