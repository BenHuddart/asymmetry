"""Corpus scenarios — Molecular dynamics of corannulene (Chemistry).

Drives the Asymmetry GUI through the WiMDA muon-school **µLCR / avoided-level-
crossing** example on the real **HiFi ``HIFI00118xxx.nxs`` (HDF4)** corpus runs
(118133–118515, 383 files — the largest set in the corpus). Corannulene
(C₂₀H₁₀, a cup-shaped PAH ≈ 1/3 of a C₆₀ ball) chemisorbs muonium to form four
long-lived muoniated radical adducts R1–R4; the observable is the **time-
integral asymmetry vs longitudinal field** and each |ΔM|=1 resonance shows as a
dip in that field scan. The reference paper is the spec: M. Gaboardi *et al.*,
"The interaction of hydrogen with corannulene," *Carbon* **155** (2019) 432–437.
See the example's ``GROUND_TRUTH.md``.

Two dense µLCR field scans dominate the run table (GT §3):

* **40 K** (setpoint 50 K; measured sample-T ≈ 42.7 K) — runs **118242–118416**,
  0–3.0 T. The low-field log-spaced block 118242–118259 (0–5000 G) is the LF
  **repolarisation** curve; 118259–118416 (0.5–3.0 T, 100–200 G steps) is the
  **wide µLCR scan** carrying all four resonances.
* **410 K** (setpoint 420 K; measured ≈ 410 K) — repolarisation block
  118204–118221 (0–5000 G) plus the wide µLCR scan **118417–118515**
  (0.5–2.46 T, 200 G steps).

Resonance fields → hyperfine couplings via **eq. 2: B_r = (Aᵤ/2)(1/γᵤ − 1/γₑ)**,
i.e. ``A_µ[MHz] = B_r[G] / 36.713`` (equivalently B_r[T] = 0.0036713·A_µ). Paper
targets (Table 1, 40 K): R4 192(11), R3 419(10), R2 484(20), R1 665(15) MHz at
B_r = 0.7 / 1.53 / 1.8 / 2.44 T (GT §6).

Scenarios registered:

* ``corpus_corannulene_ulcr_scan``     — the 40 K wide µLCR scan (118259–118416,
  0.5–3.0 T) in the GUI integral-scan view: the corpus's widest field scan, the
  R3 dip carved out of the rising repolarisation baseline (program-in-action).
* ``corpus_corannulene_resonance_fit`` — background-subtracted 40 K Δα scan with
  the **four |ΔM|=1 Gaussian dips** fitted (R2/R3 as a joint doublet); B_r read
  off each centre → A_µ = 190 / 418 / 485 / 667 MHz vs the paper's 192/419/484/665.
* ``corpus_corannulene_temperature``   — 40 K vs 410 K subtracted scans overlaid:
  R4 and R3 **narrow sharply** (Lorentzian, molecular rotation) while R1/R2
  broaden and weaken — the "molecular dynamics" signature (GT §6/§6b).
* ``corpus_corannulene_repolarisation`` — LF repolarisation P_µ(B) at 40 K and
  410 K (log field): the muonium step with **B½ ≈ 100 G (40 K) / ≈ 200 G (410 K)**,
  matching the paper's stated 100–400 G half-polarisation range (GT §6d).

``requires_fit = True`` where a real iminuit fit runs at capture time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtWidgets import QApplication, QWidget

from asymmetry.gui.styles import tokens

from ._corpus import CorpusScenario, load_corpus_datasets, register

EXAMPLE = "Chemistry/Molecular dynamics of corannulene"
_DATA = "Chemistry/Molecular dynamics of corannulene/data/HIFI00%d.nxs"

# --- Run selection (GROUND_TRUTH §3; confirmed against the loader field/T log) --
# 40 K = setpoint 50 K (measured sample-T ≈ 42.7 K); 410 K = setpoint 420 K
# (measured ≈ 410 K). Wide µLCR scans start at 5000 G (0.5 T); the log-spaced
# 0–5000 G blocks are the repolarisation curves.
_WIDE_40K = range(118259, 118417)  # 0.5–3.0 T, 158 runs (100–200 G steps)
_WIDE_410K = range(118417, 118516)  # 0.5–2.46 T, 99 runs (200 G steps)
_REPOL_40K = range(118242, 118260)  # 0–5000 G, log-spaced (40 K repolarisation)
_REPOL_410K = range(118204, 118222)  # 0–5000 G, log-spaced (410 K repolarisation)

# Guide inversion (GT §1 eq. 2 / §6): A_µ[MHz] = B_r[G] / 36.713.
_A_MU_PER_GAUSS = 1.0 / 36.713

# Paper resonance fields (T) and 40 K hyperfine couplings (MHz), GT §6a/§6b.
_PAPER = {
    "R4": {"B_r": 0.70, "A_mu": 192, "A_err": 11},
    "R3": {"B_r": 1.53, "A_mu": 419, "A_err": 10},
    "R2": {"B_r": 1.80, "A_mu": 484, "A_err": 20},
    "R1": {"B_r": 2.44, "A_mu": 665, "A_err": 15},
}

# Off-resonance windows (Gauss) for the smooth repolarisation-baseline fit —
# they bracket the four dips at 7000 / 15300 / 18000 / 24400 G. A quartic is the
# lowest order that tracks the S-shaped repolarisation background (GT §4.2).
_BASELINE_40K = [(5000, 6200), (9000, 13500), (20200, 22800), (26500, 30000)]
_BASELINE_410K = [(5000, 6200), (9000, 13500), (20000, 22600)]
_BASELINE_MODEL = "Quartic"

_RES_COLOUR = {
    "R4": tokens.TRACE_BLUE,
    "R3": tokens.TRACE_VERMILLION,
    "R2": tokens.TRACE_GREEN,
    "R1": tokens.TRACE_ORANGE,
}


# ── event / grab helpers (mirror tcnq_alc.py) ──────────────────────────────
def _pump_events(milliseconds: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(int(milliseconds), loop.quit)
    loop.exec()
    QApplication.processEvents()


def _grab_canvas_agg(canvas, name: str, output_dir: Path) -> Path:
    """Save a drawn matplotlib canvas from its Agg buffer (byte-deterministic)."""
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


def _scan_rels(runs) -> list[str]:
    return [_DATA % r for r in runs]


# ── scan construction (cached across scenarios in one capture process) ─────
_SCAN_CACHE: dict[tuple, object] = {}


def _build_scan_pct(runs):
    """Build one field's integral-asymmetry scan (percent units), field-ordered.

    Same reduction the GUI's ``_on_scan_requested`` runs (``build_field_scan``
    with the WiMDA integral method). Cached by run-range so the 40 K wide scan
    is loaded once even though several scenarios consume it (383 runs total —
    loading dominates, so the cache keeps the flock capture under budget).
    """
    from asymmetry.core.transform.integral import FieldScan, build_field_scan

    key = (runs.start, runs.stop, runs.step)
    if key in _SCAN_CACHE:
        return _SCAN_CACHE[key]
    datasets = load_corpus_datasets(_scan_rels(runs))
    scan = build_field_scan([d.run for d in datasets], method="integral", order_key="field")
    out = FieldScan(
        x=np.asarray(scan.x, dtype=float),
        value=np.asarray(scan.value, dtype=float) * 100.0,
        error=np.asarray(scan.error, dtype=float) * 100.0,
        run_numbers=list(scan.run_numbers),
        order_key="field",
        method="integral",
        x_label="B (G)",
    )
    _SCAN_CACHE[key] = out
    return out


def _subtract_baseline(scan, regions, *, x_max=None):
    """Fit + subtract the quartic repolarisation baseline; return corrected scan.

    Uses the same core ``fit_scan_baseline`` the GUI's Baseline button drives.
    Raises on a failed fit so a broken capture never ships a bad figure.
    """
    from asymmetry.core.fitting.field_scan import fit_scan_baseline
    from asymmetry.core.transform.integral import FieldScan

    x = np.asarray(scan.x, dtype=float)
    v = np.asarray(scan.value, dtype=float)
    e = np.asarray(scan.error, dtype=float)
    if x_max is not None:
        sel = x <= x_max
        x, v, e = x[sel], v[sel], e[sel]
    trimmed = FieldScan(
        x=x,
        value=v,
        error=e,
        run_numbers=list(range(len(x))),
        order_key="field",
        method="integral",
        x_label="B (G)",
    )
    base = fit_scan_baseline(trimmed, regions, model=_BASELINE_MODEL)
    if not base.success:
        raise RuntimeError("corannulene repolarisation-baseline fit did not converge")
    baseline = np.asarray(base.baseline, dtype=float)
    return {"x": x, "raw": v, "error": e, "baseline": baseline, "corrected": v - baseline}


def _fit_resonances_40k():
    """Fit the four 40 K |ΔM|=1 dips on the background-subtracted scan.

    R4 (0.7 T) and R1 (2.44 T) fit as single Gaussians; the overlapping R2/R3
    doublet (1.53 / 1.8 T) fits jointly (two GaussianLCR lines) so the shoulder
    separates cleanly (GT §4.3 — all four Gaussian at 40 K). Returns per-radical
    B_r / A_µ plus a dense overlay curve. Raises on any non-convergence.
    """
    from asymmetry.core.fitting.field_scan import as_composite_model, fit_scan_model
    from asymmetry.core.transform.integral import FieldScan

    corr = _subtract_baseline(_build_scan_pct(_WIDE_40K), _BASELINE_40K)
    x, y, e = corr["x"], corr["corrected"], corr["error"]
    cscan = FieldScan(
        x=x,
        value=y,
        error=e,
        run_numbers=list(range(len(x))),
        order_key="field",
        method="integral",
        x_label="B (G)",
    )

    results: dict[str, dict] = {}
    x_fine = np.linspace(float(x.min()), float(x.max()), 800)
    total_fine = np.zeros_like(x_fine)

    # --- singles: R4, R1 ---
    single_model = as_composite_model(["GaussianLCR"])
    for name, (lo, hi, b0, f0, w0) in {
        "R4": (6000.0, 9000.0, 7000.0, -0.5, 500.0),
        "R1": (22800.0, 26500.0, 24400.0, -0.8, 1000.0),
    }.items():
        fit = fit_scan_model(
            cscan,
            ["GaussianLCR"],
            initial={"f": f0, "B0": b0, "Bwid": w0},
            x_min=lo,
            x_max=hi,
        )
        if not fit.success:
            raise RuntimeError(f"corannulene {name} Gaussian fit did not converge")
        b0f = float(fit.parameters["B0"].value)
        params = {n: fit.parameters[n].value for n in single_model.param_names}
        results[name] = {
            "B_r": b0f,
            "A_mu": b0f * _A_MU_PER_GAUSS,
            "Bwid": abs(float(fit.parameters["Bwid"].value)),
            "amp": float(fit.parameters["f"].value),
        }
        total_fine += np.asarray(single_model.function(x_fine, **params), dtype=float)

    # --- joint doublet: R3 + R2 ---
    doublet = as_composite_model(["GaussianLCR", "GaussianLCR"])
    dfit = fit_scan_model(
        cscan,
        ["GaussianLCR", "GaussianLCR"],
        initial={
            "f_1": -1.8,
            "B0_1": 15300.0,
            "Bwid_1": 600.0,
            "f_2": -0.8,
            "B0_2": 18000.0,
            "Bwid_2": 800.0,
        },
        x_min=13000.0,
        x_max=20000.0,
    )
    if not dfit.success:
        raise RuntimeError("corannulene R2/R3 doublet fit did not converge")
    dparams = {n: dfit.parameters[n].value for n in doublet.param_names}
    total_fine += np.asarray(doublet.function(x_fine, **dparams), dtype=float)
    for name, suffix in (("R3", "_1"), ("R2", "_2")):
        b0f = float(dfit.parameters[f"B0{suffix}"].value)
        results[name] = {
            "B_r": b0f,
            "A_mu": b0f * _A_MU_PER_GAUSS,
            "Bwid": abs(float(dfit.parameters[f"Bwid{suffix}"].value)),
            "amp": float(dfit.parameters[f"f{suffix}"].value),
        }

    return {
        "x": x,
        "corrected": y,
        "error": e,
        "x_fine": x_fine,
        "total_fine": total_fine,
        "res": results,
    }


# ══════════════════════════════════════════════════════════════════════════
# 1. Wide µLCR scan (40 K) — the corpus's widest field scan, in the GUI
# ══════════════════════════════════════════════════════════════════════════
class CorannuleneUlcrScanScenario(CorpusScenario):
    name = "corpus_corannulene_ulcr_scan"
    description = (
        "Corannulene 40 K wide µLCR scan: 158 HiFi runs (0.5–3.0 T, the corpus's "
        "widest field scan) reduced to integral asymmetry vs longitudinal field "
        "in the integral-scan view — the strong R3 |ΔM|=1 dip near ~1.53 T is "
        "carved out of the rising LF repolarisation baseline."
    )
    example = EXAMPLE
    size = (1500, 900)

    def build(self) -> QWidget:
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window.resizeDocks([window._dock_data_browser], [340], Qt.Orientation.Horizontal)

        datasets = load_corpus_datasets(_scan_rels(_WIDE_40K))
        with window._data_browser.batch_updates():
            for dataset in datasets:
                window._data_browser.add_dataset(dataset)
        window._data_browser.create_data_group(
            [int(ds.run_number) for ds in datasets],
            name="Corannulene µLCR 40 K — 0.5–3.0 T",
        )
        window._data_browser._table.selectAll()
        _pump_events(250)
        window._plot_workspace.set_active_view("integral_scan")
        _pump_events(150)
        window._alc_fit_panel.build_requested.emit()
        _pump_events(500)
        return window


# ══════════════════════════════════════════════════════════════════════════
# 2. Resonance fit — four Gaussian dips → A_µ vs paper (headline analysis)
# ══════════════════════════════════════════════════════════════════════════
class CorannuleneResonanceFitScenario(CorpusScenario):
    name = "corpus_corannulene_resonance_fit"
    description = (
        "Background-subtracted 40 K Δα µLCR scan with the four |ΔM|=1 Gaussian "
        "dips fitted (R2/R3 as a joint doublet). Resonance fields → hyperfine "
        "couplings A_µ = 190 / 418 / 485 / 667 MHz for R4/R3/R2/R1, reproducing "
        "the paper's 192(11) / 419(10) / 484(20) / 665(15) MHz (Table 1)."
    )
    example = EXAMPLE
    size = (1180, 760)
    requires_fit = True

    def capture(self, ctx) -> Path:  # noqa: D401
        from asymmetry.gui.widgets.mpl_canvas import create_canvas

        _warm_fig, _warm_canvas = create_canvas(layout="tight")
        _warm_canvas.draw()
        _pump_events(60)

        fit = _fit_resonances_40k()
        x = fit["x"] / 1e4  # Tesla
        figure, canvas = create_canvas(layout="tight")
        axes = figure.add_subplot(111)
        axes.errorbar(
            x,
            fit["corrected"],
            yerr=fit["error"],
            fmt="o",
            ms=3.0,
            lw=0.0,
            elinewidth=0.7,
            color=tokens.PLOT_DATA,
            alpha=0.8,
            label="Δα (40 K)",
        )
        axes.plot(
            fit["x_fine"] / 1e4,
            fit["total_fine"],
            color=tokens.PLOT_FIT,
            lw=1.8,
            label="4× Gaussian |ΔM|=1 fit",
        )
        axes.axhline(0.0, color=tokens.PLOT_ZERO_LINE, lw=0.8)
        # Mark each fitted resonance field and annotate A_µ vs the paper.
        ymin = float(np.nanmin(fit["corrected"]))
        for name in ("R4", "R3", "R2", "R1"):
            r = fit["res"][name]
            br_t = r["B_r"] / 1e4
            colour = _RES_COLOUR[name]
            axes.axvline(br_t, color=colour, ls=":", lw=1.2)
            axes.annotate(
                f"{name}: {r['A_mu']:.0f} MHz\n(paper {_PAPER[name]['A_mu']})",
                xy=(br_t, ymin * 0.5),
                xytext=(br_t + 0.04, ymin * 1.16),
                ha="left",
                va="top",
                fontsize="small",
                color=colour,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=colour, alpha=0.85),
            )
        axes.set_xlabel("Longitudinal field B (T)")
        axes.set_ylabel("Δα integral asymmetry (%, baseline-subtracted)")
        axes.set_title("Corannulene 40 K µLCR — four muoniated-radical |ΔM|=1 resonances")
        axes.set_xlim(0.5, 3.0)
        axes.set_ylim(ymin * 1.30, max(0.6, -ymin * 0.35))
        axes.legend(loc="upper left", fontsize="small", framealpha=0.92)
        axes.grid(True, color=tokens.PLOT_GRID, linewidth=0.8)

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


# ══════════════════════════════════════════════════════════════════════════
# 3. Temperature comparison — 40 K vs 410 K (molecular dynamics)
# ══════════════════════════════════════════════════════════════════════════
class CorannuleneTemperatureScenario(CorpusScenario):
    name = "corpus_corannulene_temperature"
    description = (
        "Corannulene µLCR vs temperature: 40 K and 410 K background-subtracted "
        "Δα scans overlaid (offset for clarity, paper Fig. 4 style). At 410 K "
        "the low-field R4 and R3 lines narrow sharply (fast molecular rotation / "
        "pre-melting) while R1 and R2 broaden and weaken — the molecular-dynamics "
        "signature."
    )
    example = EXAMPLE
    size = (1180, 760)
    requires_fit = True

    def capture(self, ctx) -> Path:  # noqa: D401
        from asymmetry.gui.widgets.mpl_canvas import create_canvas

        _warm_fig, _warm_canvas = create_canvas(layout="tight")
        _warm_canvas.draw()
        _pump_events(60)

        c40 = _subtract_baseline(_build_scan_pct(_WIDE_40K), _BASELINE_40K, x_max=24600)
        c41 = _subtract_baseline(_build_scan_pct(_WIDE_410K), _BASELINE_410K, x_max=24600)
        offset = 2.4  # vertical separation (paper offsets the two traces ±1)

        figure, canvas = create_canvas(layout="tight")
        axes = figure.add_subplot(111)
        for corr, temp, colour, off in (
            (c40, "40 K", tokens.TRACE_BLUE, offset),
            (c41, "410 K", tokens.TRACE_VERMILLION, 0.0),
        ):
            axes.errorbar(
                corr["x"] / 1e4,
                corr["corrected"] + off,
                yerr=corr["error"],
                fmt="o",
                ms=3.0,
                lw=0.0,
                elinewidth=0.6,
                color=colour,
                alpha=0.55,
            )
            axes.plot(
                corr["x"] / 1e4,
                corr["corrected"] + off,
                color=colour,
                lw=1.0,
                alpha=0.9,
                label=f"{temp}  (Δα offset {off:+.1f}%)",
            )
            axes.axhline(off, color=colour, lw=0.5, ls="--", alpha=0.4)
        for name in ("R4", "R3", "R2", "R1"):
            br_t = _PAPER[name]["B_r"]
            axes.axvline(br_t, color=tokens.TEXT_DIM, ls=":", lw=1.0)
            axes.text(
                br_t, offset + 0.7, name, ha="center", fontsize="small", color=tokens.TEXT_DIM
            )
        axes.set_xlabel("Longitudinal field B (T)")
        axes.set_ylabel("Δα integral asymmetry (%, offset)")
        axes.set_title("Corannulene µLCR 40 K vs 410 K — R4/R3 narrow, R1/R2 broaden on warming")
        axes.set_xlim(0.5, 2.55)
        axes.legend(loc="lower right", fontsize="small", framealpha=0.92)
        axes.grid(True, color=tokens.PLOT_GRID, linewidth=0.8)

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


# ══════════════════════════════════════════════════════════════════════════
# 4. Repolarisation — the muonium step, B½ vs paper
# ══════════════════════════════════════════════════════════════════════════
class CorannuleneRepolarisationScenario(CorpusScenario):
    name = "corpus_corannulene_repolarisation"
    description = (
        "LF muon repolarisation P_µ(B) at 40 K and 410 K (log field, 0–5000 G): "
        "the integral asymmetry rises through the muonium step, half-recovered at "
        "B½ ≈ 100 G (40 K) / ≈ 200 G (410 K) — inside the paper's stated "
        "100–400 G range and signalling the ≈ 80 % muonium fraction (GT §6d)."
    )
    example = EXAMPLE
    size = (1120, 720)

    @staticmethod
    def _repol(runs):
        scan = _build_scan_pct(runs)
        x = np.asarray(scan.x, dtype=float)
        v = np.asarray(scan.value, dtype=float)
        order = np.argsort(x)
        x, v = x[order], v[order]
        # Plateau anchors: low-field (B→0) and the high-field repolarisation top.
        lo = float(np.mean(v[x <= 5.0]))
        hi = float(np.max(v))
        p = (v - lo) / (hi - lo)  # normalised P_µ: 0 at B→0, 1 at the plateau
        # B½: where P crosses 0.5 on the monotonic rising branch (up to the
        # plateau peak); np.interp needs an increasing sample vector, so cut at
        # the argmax before interpolating.
        peak = int(np.argmax(v))
        rise = (x > 5.0) & (np.arange(len(x)) <= peak)
        xr, pr = x[rise], p[rise]
        b_half = float(np.interp(0.5, pr, xr))
        return x, p, b_half, lo, hi

    def capture(self, ctx) -> Path:  # noqa: D401
        from asymmetry.gui.widgets.mpl_canvas import create_canvas

        _warm_fig, _warm_canvas = create_canvas(layout="tight")
        _warm_canvas.draw()
        _pump_events(60)

        x40, p40, bh40, _, _ = self._repol(_REPOL_40K)
        x41, p41, bh41, _, _ = self._repol(_REPOL_410K)

        figure, canvas = create_canvas(layout="tight")
        axes = figure.add_subplot(111)
        for x, p, bh, temp, colour in (
            (x40, p40, bh40, "40 K", tokens.TRACE_BLUE),
            (x41, p41, bh41, "410 K", tokens.TRACE_VERMILLION),
        ):
            m = x > 0.0  # log axis: drop the true-zero-field point
            axes.plot(
                x[m], p[m], "o-", ms=4.0, lw=1.4, color=colour, label=f"{temp}  (B½ ≈ {bh:.0f} G)"
            )
            axes.axvline(bh, color=colour, ls=":", lw=1.1)
        axes.axhline(0.5, color=tokens.PLOT_ZERO_LINE, lw=0.8, ls="--")
        axes.axvspan(
            100.0, 400.0, color=tokens.TRACE_GREEN, alpha=0.10, label="paper 100–400 G B½ range"
        )
        axes.set_xscale("log")
        axes.set_xlabel("Longitudinal field B (G, log scale)")
        axes.set_ylabel("Normalised repolarisation P_µ(B)")
        axes.set_title("Corannulene LF repolarisation — muonium step (≈ 80 % Mu fraction)")
        axes.set_xlim(8.0, 2200.0)
        axes.set_ylim(-0.05, 1.08)
        axes.legend(loc="upper left", fontsize="small", framealpha=0.92)
        axes.grid(True, which="both", color=tokens.PLOT_GRID, linewidth=0.7)

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


register(CorannuleneUlcrScanScenario())
register(CorannuleneResonanceFitScenario())
register(CorannuleneTemperatureScenario())
register(CorannuleneRepolarisationScenario())
