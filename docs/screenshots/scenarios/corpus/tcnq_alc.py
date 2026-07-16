"""Corpus scenarios — ALC resonance in TCNQ (Chemistry).

Drives the Asymmetry GUI through the WiMDA muon-school **ALC-µSR** example on the
real EMU ``.nxs`` corpus runs (``emu00019485–19612``): avoided-level-crossing
longitudinal-field scans of a muoniated TCNQ radical, repeated at 350 / 100 /
50 / 10 K. Each temperature scan is 31 runs stepped 2000 → 5000 G in 100 G steps
(GROUND_TRUTH §3). The observable is the **time-integral asymmetry vs field** —
a field-domain (not time-domain) technique — and the D1 resonance shows up as a
dip in that integral scan near the expected ≈2.94 kG (A_µ ≈ 80 MHz).

The example's ``GROUND_TRUTH.md`` is the spec. Workflow followed (GT §4):
batch-build the 350 K integral scan (runs 19489–19519), fit a **cubic
background** over the non-resonant edges, then a **Lorentzian** line to the
resonance, and read the resonance field B_res off the fitted centre. The
hyperfine coupling inverts from B_res via the guide relation
``A_µ[MHz] ≈ B_res[G] / 36.71`` (GT §6) and the muon dipolar coupling from
``D_µ[MHz] ≈ FWHM[G] / 68`` (GT §4D).

Scenarios registered:

* ``corpus_tcnq_integral_scan`` — the raw 350 K integral-asymmetry field scan in
  the integral-scan view; the ALC dip near ~3.1 kG stands clear of a flat ~25 %
  baseline.
* ``corpus_tcnq_alc_fit``       — converged Cubic-background + Lorentzian fit on
  the 350 K scan; B_res ≈ 3104 G readable → A_µ ≈ 84.6 MHz.
* ``corpus_tcnq_temperature``   — all four T scans (350/100/50/10 K) overlaid:
  the dip deepens and narrows as T rises (motional narrowing).
* ``corpus_tcnq_dmu_trend``     — headline hyperfine deliverable: A_µ(T) and the
  muon dipolar coupling D_µ(T) from the four fitted resonances.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtWidgets import QApplication, QTableWidgetItem, QWidget

from asymmetry.gui.styles import tokens

from ._corpus import CorpusScenario, load_corpus_datasets, register

EXAMPLE = "Chemistry/ALC resonance in TCNQ"
_DATA = "Chemistry/ALC resonance in TCNQ/Data/emu000%d.nxs"

# Four longitudinal-field ALC scans, each 31 runs stepped 2000→5000 G / 100 G
# (GROUND_TRUTH §3). Labelled by setpoint T (measured T drifts, GT §9).
_SCANS: dict[int, range] = {
    350: range(19489, 19520),  # 350 K scan start = 19489 (19488 excluded, GT §9)
    100: range(19520, 19551),
    50: range(19582, 19613),
    10: range(19551, 19582),
}

# Baseline (non-resonant) field windows either side of the dip (~2700–3400 G);
# a Cubic background is the WiMDA/Mantid-prescribed ALC baseline (GT §4C).
_BASELINE_REGIONS = [(2000.0, 2600.0), (3400.0, 5000.0)]
_BASELINE_MODEL = "Cubic"

# Guide inversion constants (GROUND_TRUTH §6 / §4D):
#   A_µ[MHz] = 2·B_res / (γ_µ⁻¹ − γ_e⁻¹) ≈ B_res[G] / 36.71
#   D_µ[MHz] = FWHM[G] / 68
_A_MU_PER_GAUSS = 1.0 / 36.71
_D_MU_PER_GAUSS = 1.0 / 68.0

# Okabe–Ito trace palette (tokens), warm→cool with temperature.
_T_COLOURS: dict[int, str] = {
    350: tokens.TRACE_VERMILLION,
    100: tokens.TRACE_ORANGE,
    50: tokens.TRACE_GREEN,
    10: tokens.TRACE_BLUE,
}


def _pump_events(milliseconds: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(int(milliseconds), loop.quit)
    loop.exec()
    QApplication.processEvents()


def _grab_widget(widget: QWidget, name: str, output_dir: Path) -> Path:
    out_path = output_dir / f"{name}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pix = widget.grab()
    if not pix.save(str(out_path), "PNG"):
        raise RuntimeError(f"Failed to save screenshot to {out_path}")
    return out_path


def _grab_canvas_agg(canvas, name: str, output_dir: Path) -> Path:
    """Save a drawn matplotlib canvas from its Agg buffer (byte-deterministic).

    ``QWidget.grab`` on a fresh offscreen FigureCanvas settles the last pixel
    column two ways ~1/3 of the time (a first-paint edge artifact). The Agg
    RGBA buffer the figure already rendered is byte-stable, so read it directly
    rather than round-tripping through Qt's grab.
    """
    from PySide6.QtGui import QImage

    buf = canvas.buffer_rgba()
    arr = np.asarray(buf)
    height, width = arr.shape[:2]
    image = QImage(arr.tobytes(), width, height, QImage.Format.Format_RGBA8888)
    out_path = output_dir / f"{name}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not image.save(str(out_path), "PNG"):
        raise RuntimeError(f"Failed to save screenshot to {out_path}")
    return out_path


def _scan_rels(runs: range) -> list[str]:
    return [_DATA % r for r in runs]


def _build_scan_pct(runs: range):
    """Build one temperature's integral field scan in percent units.

    Returns ``FieldScan`` (percent) — the same reduction the GUI's
    ``_on_scan_requested`` runs (``build_field_scan`` with the WiMDA integral
    method), ordered by field.
    """
    from asymmetry.core.transform.integral import FieldScan, build_field_scan

    datasets = load_corpus_datasets(_scan_rels(runs))
    scan = build_field_scan([d.run for d in datasets], method="integral", order_key="field")
    return FieldScan(
        x=np.asarray(scan.x, dtype=float),
        value=np.asarray(scan.value, dtype=float) * 100.0,
        error=np.asarray(scan.error, dtype=float) * 100.0,
        run_numbers=list(scan.run_numbers),
        order_key="field",
        method="integral",
        x_label="B (G)",
    )


def _fit_alc(scan_pct):
    """Cubic baseline + single Lorentzian on a percent scan (GT §4C).

    Returns a dict with the raw scan arrays, the fitted total curve, and the
    read-off resonance field / FWHM / hyperfine numbers. Uses the same core
    ``fit_scan_baseline`` / ``fit_scan_model`` the GUI's Baseline/Peaks buttons
    drive. Raises on a failed fit so a broken capture never ships a bad number.
    """
    from asymmetry.core.fitting.field_scan import fit_scan_baseline, fit_scan_model

    base = fit_scan_baseline(scan_pct, _BASELINE_REGIONS, model=_BASELINE_MODEL)
    if not base.success:
        raise RuntimeError("TCNQ ALC cubic-baseline fit did not converge")
    fit = fit_scan_model(
        base.corrected,
        ["LorentzianLCR"],
        initial={"f": -3.0, "B0": 3100.0, "Bwid": 120.0},
    )
    if not fit.success:
        raise RuntimeError(f"TCNQ ALC Lorentzian fit did not converge: {fit.message}")
    b0 = float(fit.parameters["B0"].value)
    b0_err = float(fit.uncertainties.get("B0", 0.0))
    bwid = abs(float(fit.parameters["Bwid"].value))
    amp = float(fit.parameters["f"].value)
    fwhm = 2.0 * bwid  # LorentzianLCR fwhm_factor = 2
    # Total (baseline + peak) on the raw scan for overlay.
    from asymmetry.core.fitting.field_scan import as_composite_model

    model = as_composite_model(["LorentzianLCR"])
    params = {n: fit.parameters[n].value for n in model.param_names}
    x = np.asarray(scan_pct.x, dtype=float)
    baseline = np.asarray(base.baseline, dtype=float)
    total = baseline + np.asarray(model.function(x, **params), dtype=float)
    # A dense curve for a smooth overlay: the cubic baseline is smooth, so
    # linear-interpolating it onto a fine grid and adding the analytic
    # Lorentzian avoids the angular V that a 100 G-spaced polyline draws.
    x_fine = np.linspace(float(x.min()), float(x.max()), 600)
    total_fine = np.interp(x_fine, x, baseline) + np.asarray(
        model.function(x_fine, **params), dtype=float
    )
    return {
        "x": x,
        "value": np.asarray(scan_pct.value, dtype=float),
        "error": np.asarray(scan_pct.error, dtype=float),
        "total": total,
        "x_fine": x_fine,
        "total_fine": total_fine,
        "baseline": baseline,
        "B_res": b0,
        "B_res_err": b0_err,
        "FWHM": fwhm,
        "amp": amp,
        "A_mu": b0 * _A_MU_PER_GAUSS,
        "D_mu": fwhm * _D_MU_PER_GAUSS,
    }


# ══════════════════════════════════════════════════════════════════════════
# 1. Raw integral scan — the ALC dip in the integral-scan view (real GUI)
# ══════════════════════════════════════════════════════════════════════════
class TcnqIntegralScanScenario(CorpusScenario):
    name = "corpus_tcnq_integral_scan"
    description = (
        "350 K TCNQ ALC scan: 31 EMU runs (2000–5000 G) reduced to integral "
        "asymmetry vs longitudinal field in the integral-scan view — the D1 "
        "resonance dips near ~3.1 kG out of a flat ~25 % baseline."
    )
    example = EXAMPLE
    size = (1500, 900)

    def build(self) -> QWidget:
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window.resizeDocks([window._dock_data_browser], [340], Qt.Orientation.Horizontal)

        datasets = load_corpus_datasets(_scan_rels(_SCANS[350]))
        with window._data_browser.batch_updates():
            for dataset in datasets:
                window._data_browser.add_dataset(dataset)
        window._data_browser.create_data_group(
            [int(ds.run_number) for ds in datasets],
            name="TCNQ ALC 350 K — 2000–5000 G",
        )
        # Multi-select every run so the fit panel's batch (which the scan build
        # reads) is populated, then enter the integral-scan view and build.
        window._data_browser._table.selectAll()
        _pump_events(200)
        window._plot_workspace.set_active_view("integral_scan")
        _pump_events(150)
        window._alc_fit_panel.build_requested.emit()
        _pump_events(400)
        return window


# ══════════════════════════════════════════════════════════════════════════
# 2. Converged Cubic-background + Lorentzian fit (real GUI, requires_fit)
# ══════════════════════════════════════════════════════════════════════════
class TcnqAlcFitScenario(CorpusScenario):
    name = "corpus_tcnq_alc_fit"
    description = (
        "Converged ALC fit on the 350 K TCNQ scan: Cubic background over the "
        "non-resonant edges + a Lorentzian resonance line. B_res ≈ 3104 G is "
        "read off the fitted centre → A_µ ≈ 84.6 MHz (target ≈ 80 MHz)."
    )
    example = EXAMPLE
    size = (1500, 900)
    requires_fit = True

    def build(self) -> QWidget:
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window.resizeDocks([window._dock_data_browser], [320], Qt.Orientation.Horizontal)

        datasets = load_corpus_datasets(_scan_rels(_SCANS[350]))
        with window._data_browser.batch_updates():
            for dataset in datasets:
                window._data_browser.add_dataset(dataset)
        window._data_browser.create_data_group(
            [int(ds.run_number) for ds in datasets],
            name="TCNQ ALC 350 K — 2000–5000 G",
        )
        window._data_browser._table.selectAll()
        _pump_events(200)
        window._plot_workspace.set_active_view("integral_scan")
        _pump_events(150)
        window._alc_fit_panel.build_requested.emit()
        _pump_events(400)

        view = window._alc_scan_view

        # Baseline: Cubic over the two non-resonant edges (GT §4C). The dip lives
        # ~2700–3400 G, so the edge windows bracket it. Both ALC fits are
        # synchronous (core iminuit fit on the ~31-point scan).
        view._baseline_model_combo.setCurrentText(_BASELINE_MODEL)
        for lo, hi in _BASELINE_REGIONS:
            view._add_region()
        for row, (lo, hi) in enumerate(_BASELINE_REGIONS):
            view._regions_table.setItem(row, 0, QTableWidgetItem(f"{lo:.0f}"))
            view._regions_table.setItem(row, 1, QTableWidgetItem(f"{hi:.0f}"))
        _pump_events(120)
        view.baseline_fit_requested.emit()
        _pump_events(300)

        # Peak: a Lorentzian seeded a little off the resonance (3050 G) so the
        # fit visibly moves onto B_res ≈ 3104 G. Cols: 1=B0, 2=Width, 3=Amp.
        view._add_peak("Lorentzian")
        view._peaks_table.setItem(0, 1, QTableWidgetItem("3050"))
        view._peaks_table.setItem(0, 2, QTableWidgetItem("140"))
        view._peaks_table.setItem(0, 3, QTableWidgetItem("-5"))
        _pump_events(120)
        view.peaks_fit_requested.emit()
        _pump_events(350)
        return window


# ══════════════════════════════════════════════════════════════════════════
# 3. Temperature comparison — all four ALC scans overlaid (motional narrowing)
# ══════════════════════════════════════════════════════════════════════════
class TcnqTemperatureScenario(CorpusScenario):
    name = "corpus_tcnq_temperature"
    description = (
        "TCNQ ALC dip vs temperature: the 350/100/50/10 K integral-asymmetry "
        "field scans overlaid. As T rises the D1 dip deepens and narrows — the "
        "signature of motional averaging of the dipolar hyperfine coupling."
    )
    example = EXAMPLE
    size = (1120, 720)
    requires_fit = True  # a Lorentzian fit per temperature draws the overlaid curves

    def capture(self, ctx) -> Path:  # noqa: D401
        from asymmetry.gui.widgets.mpl_canvas import create_canvas

        # Warm-up draw: the first matplotlib canvas painted in a fresh process
        # settles one right-margin column non-deterministically (offscreen
        # first-paint artifact); prime it so the real capture is byte-stable.
        _warm_fig, _warm_canvas = create_canvas(layout="tight")
        _warm_canvas.draw()
        _pump_events(60)

        figure, canvas = create_canvas(layout="tight")
        axes = figure.add_subplot(111)
        # Warm (350 K) → cool (10 K): fit each scan so the smooth Lorentzian+
        # cubic curve guides the eye through the raw points.
        for temp in (350, 100, 50, 10):
            scan = _build_scan_pct(_SCANS[temp])
            fit = _fit_alc(scan)
            colour = _T_COLOURS[temp]
            axes.errorbar(
                fit["x"],
                fit["value"],
                yerr=fit["error"],
                fmt="o",
                ms=3.0,
                lw=0.0,
                elinewidth=0.7,
                color=colour,
                alpha=0.85,
            )
            axes.plot(
                fit["x_fine"],
                fit["total_fine"],
                color=colour,
                linewidth=1.6,
                label=f"{temp} K  (FWHM {fit['FWHM']:.0f} G, depth {abs(fit['amp']):.1f}%)",
            )
        axes.set_xlabel("Longitudinal field B (G)")
        axes.set_ylabel("Integral asymmetry (%)")
        axes.set_title("TCNQ ALC D1 resonance vs temperature — dip narrows/deepens as T rises")
        axes.set_xlim(2000.0, 5000.0)
        axes.legend(loc="lower right", fontsize="small", framealpha=0.92, title="Setpoint T")
        axes.grid(True, color=tokens.PLOT_GRID, linewidth=0.8)

        canvas.resize(*self.size)
        canvas.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        canvas.show()
        _pump_events(200)
        canvas.draw()
        _pump_events(80)
        canvas.draw()
        out = _grab_widget(canvas, self.name, ctx.output_dir)
        canvas.close()
        canvas.deleteLater()
        _pump_events(40)
        return out


# ══════════════════════════════════════════════════════════════════════════
# 4. Headline — A_µ(T) and D_µ(T) hyperfine deliverable from the four fits
# ══════════════════════════════════════════════════════════════════════════
class TcnqDMuTrendScenario(CorpusScenario):
    name = "corpus_tcnq_dmu_trend"
    description = (
        "TCNQ hyperfine deliverable: muon coupling A_µ(T) (≈80–85 MHz, near the "
        "≈80 MHz target) and dipolar coupling D_µ(T)=FWHM/68 from the four fitted "
        "ALC resonances — D_µ falls as T rises (motional narrowing, GT Q5)."
    )
    example = EXAMPLE
    size = (1020, 720)
    requires_fit = True

    def capture(self, ctx) -> Path:  # noqa: D401
        from matplotlib.ticker import FuncFormatter

        from asymmetry.gui.widgets.mpl_canvas import create_canvas

        _warm_fig, _warm_canvas = create_canvas(layout="tight")
        _warm_canvas.draw()
        _pump_events(60)

        temps: list[int] = []
        a_mu: list[float] = []
        d_mu: list[float] = []
        for temp in (10, 50, 100, 350):
            fit = _fit_alc(_build_scan_pct(_SCANS[temp]))
            temps.append(temp)
            a_mu.append(fit["A_mu"])
            d_mu.append(fit["D_mu"])
        t = np.array(temps, dtype=float)

        # Two stacked single-axis panels sharing the log-temperature x. A twin-y
        # axis was byte-flaky offscreen (the twinned right spine settled the last
        # canvas column two ways ~1/3 of the time); two plain subplots under
        # tight layout render deterministically.
        figure, canvas = create_canvas(layout="tight")
        ax_a = figure.add_subplot(211)
        ax_d = figure.add_subplot(212, sharex=ax_a)

        ax_a.plot(t, a_mu, "o-", color=tokens.TRACE_BLUE, lw=1.6, ms=6)
        # The ≈80 MHz guide target the radical's resonance is set to match (GT §6).
        ax_a.axhline(80.0, color=tokens.TEXT_DIM, ls="--", lw=1.1, label="target ≈ 80 MHz")
        ax_a.set_ylabel("A_µ (MHz)")
        ax_a.set_ylim(min(a_mu) - 3.0, max(a_mu) + 3.0)
        ax_a.legend(loc="lower right", fontsize="small", framealpha=0.92)
        ax_a.grid(True, which="both", color=tokens.PLOT_GRID, linewidth=0.8)
        ax_a.tick_params(labelbottom=False)
        ax_a.set_title("TCNQ hyperfine parameters vs temperature — D_µ narrows as motion grows")

        ax_d.plot(t, d_mu, "s-", color=tokens.TRACE_VERMILLION, lw=1.6, ms=6)
        ax_d.set_ylabel("D_µ = FWHM/68 (MHz)")
        ax_d.set_ylim(min(d_mu) - 0.6, max(d_mu) + 0.6)
        ax_d.set_xscale("log")
        ax_d.set_xlabel("Temperature (K, log scale)")
        ax_d.set_xticks(t)
        ax_d.get_xaxis().set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
        ax_d.grid(True, which="both", color=tokens.PLOT_GRID, linewidth=0.8)

        canvas.resize(*self.size)
        canvas.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        canvas.show()
        _pump_events(200)
        canvas.draw()
        _pump_events(80)
        canvas.draw()
        out = _grab_canvas_agg(canvas, self.name, ctx.output_dir)
        canvas.close()
        canvas.deleteLater()
        _pump_events(40)
        return out


register(TcnqIntegralScanScenario())
register(TcnqAlcFitScenario())
register(TcnqTemperatureScenario())
register(TcnqDMuTrendScenario())
