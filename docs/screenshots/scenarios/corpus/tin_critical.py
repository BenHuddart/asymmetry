"""Corpus scenarios — critical fields in type-I tin (Sn).

Worked example ``Superconductivity/Critical fields in Sn`` (WiMDA muon school
corpus). HIFI (ISIS) NeXus-v1 HDF4 ``.nxs`` runs 91488–91529 — a 2015
cryostat-commissioning teaching run (``experiment_number = −1000``). Thin Sn
foil at 45°, applied field 20/40/80/160 G, nominal setpoints 1.0–2.9 K. See the
example's ``GROUND_TRUTH.md``; guidance physics (GT §10, Karl et al., PRB 99,
184515): T_c = 3.717(3) K, B_c(0) = 30.578(6) mT ≈ 306 G, parabolic
H_c(T) = B_c(0)[1 − (T/T_c)²].

What the data actually shows (established here from the raw counts; see
NOTES_sn.md for the full audit trail):

* **The intermediate state is present and maps H_c(T)** — but displaced by a
  large thermometer error, the guide's own headline question. In the 40 G
  temperature scan (91516–91529) a single transverse line marches monotonically
  from ≈141 G at nominal 1.6 K down to the applied 40 G at nominal 2.8 K
  (6–23σ per run). In the intermediate state the normal domains carry exactly
  B_c(T), so B_int = B_c(T_real): inverting the guidance parabola gives
  T_real ≈ 2.73–3.40 K, i.e. the thermometer reads **0.7–1.1 K low** (and
  ~1.7 K low at the "1 K" setpoints — the commissioning cryostat's true base was
  ≈2.7 K). At nominal 2.8–2.9 K the line sits at the applied field with doubled
  amplitude: the sample has crossed H_c(T) = 40 G and gone normal.
* **The domain field is independent of the applied field**, the type-I
  signature: at base temperature the 20 G and 80 G runs both show the line at
  ≈138–141 G (= B_c(T_real ≈ 2.7 K)), while the 160 G run — for which
  B_app > B_c(T_real) — shows a single line at the applied field (fully normal).
  The "sensor-fault" block 91501–91515 (nominal 1.95–2.35 K, sensors ~8 K) shows
  strong applied-field lines at all three fields: the sample was normal, i.e.
  the ~8 K sensor readings were *right* and the setpoints wrong.
* **No line near 3.85 MHz (B_c at a true 1 K ≈ 284 G) exists** in the 160 G
  run: searched 3.2–4.6 MHz over several time windows; nothing above ~1.3σ,
  versus the 21σ applied-field line. Consistent with the thermometer error —
  the sample never got near 1 K, so B_c never exceeded ≈145 G.

⚠ Loader / geometry caveat. HIFI's 64 detectors form two longitudinal rings and
the file's grouping array reduces them to a forward/backward pair, which cancels
a *transverse* precession — the standard-loader F−B asymmetry is noise here
(both time-domain and GUI-FFT renders were checked and rejected). All the
physics above is recovered by re-grouping the **raw counts** into left/right
ring halves (two quadrature pairs per ring); the loaded ``MuonDataset`` retains
no histograms, so this is not reachable through the normal pipeline — the
scenarios read the HDF4 counts directly (still via ``corpus_path``) and render
standalone Matplotlib figures (the ``cuprate_bscco`` ``field_compare`` pattern).

Scenarios registered:

* ``corpus_sn_hc_dome`` — the **headline**: the guidance H_c(T) dome with the
  dataset's (T, B) coverage and the *measured* intermediate-state B_int points
  from the 40 G scan (computed from the raw counts at capture time), falling
  systematically left of the parabola — the thermometer error made visible, and
  the boundary crossing at nominal ≈2.8 K vs the parabola's 3.47 K.
* ``corpus_sn_intermediate_lines`` — the spectral evidence: (a) base-T spectra
  at 20/80/160 G — the 20 and 80 G lines coincide at ≈140 G (domain field
  B_c, independent of applied field) while 160 G sits at the applied field;
  (b) the 40 G-scan spectra marching from 141 G down to the applied field.
* ``corpus_sn_transverse_recovery`` — the loader caveat: the 160 G run's
  standard F−B asymmetry (noise) vs the left/right regrouping of the same raw
  counts (a clean 2.14 MHz ≈ 158 G line — the normal-state applied-field
  precession).
"""

from __future__ import annotations

import numpy as np

from ._corpus import CorpusScenario, corpus_path, register
from .._base import CaptureContext

EXAMPLE = "Superconductivity/Critical fields in Sn"
_DATA = "Superconductivity/Critical fields in Sn/Data/HIFI000%d.nxs"

# Literature guidance (GROUND_TRUTH §10; Karl et al. PRB 99, 184515):
_TC = 3.717          # K
_BC0 = 306.0         # G  (≈ 30.6 mT)
_GAMMA = 0.013554    # MHz/G  (muon)

# Applied fields present in the dataset (GROUND_TRUTH §3).
_FIELDS = [20.0, 40.0, 80.0, 160.0]

# Full run → (nominal T, applied B) grid (GROUND_TRUTH §3 table).
_RUN_GRID: list[tuple[float, float]] = [
    (1.0, 20), (1.0, 40), (1.0, 80), (1.0, 160),
    (1.65, 20), (1.65, 40), (1.65, 80),
    (1.75, 20), (1.75, 40), (1.75, 80),
    (1.85, 20), (1.85, 40), (1.85, 80),
    (1.95, 20), (1.95, 40), (1.95, 80),
    (2.05, 20), (2.05, 40), (2.05, 80),
    (2.15, 20), (2.15, 40), (2.15, 80),
    (2.25, 20), (2.25, 40), (2.25, 80),
    (2.35, 20), (2.35, 40), (2.35, 80),
    # 40 G temperature scan 91516–91529:
    (1.6, 40), (1.7, 40), (1.8, 40), (1.9, 40), (2.0, 40), (2.1, 40), (2.2, 40),
    (2.3, 40), (2.4, 40), (2.5, 40), (2.6, 40), (2.7, 40), (2.8, 40), (2.9, 40),
]

# 40 G temperature scan (GROUND_TRUTH §3): run → nominal setpoint T.
_SCAN_40G: list[tuple[int, float]] = [
    (91516 + i, round(1.6 + 0.1 * i, 1)) for i in range(14)
]

_RUN_160G = 91491    # base T; B_app > B_c(T_real) ⇒ sample normal, clean line


def _hc(temp) -> np.ndarray:
    """Guidance parabolic critical field H_c(T) in Gauss."""
    t = np.asarray(temp, dtype=float)
    return _BC0 * (1.0 - (t / _TC) ** 2)


def _crossing_temperature(field: float) -> float:
    """Temperature where H_c(T) = *field* (the field's boundary crossing)."""
    return _TC * float(np.sqrt(max(0.0, 1.0 - field / _BC0)))


def _read_counts(run: int):
    """Read raw HIFI detector counts / grouping / time from the NeXus HDF4 file."""
    from asymmetry.core.io import hdf4

    tree = hdf4.read_tree(str(corpus_path(_DATA % run)))

    def get(path: str):
        node = tree
        for part in path.split("/"):
            if part:
                node = node.children[part]
        return np.asarray(node.data)

    counts = get("run/histogram_data_1/counts").astype(float)
    grouping = get("run/histogram_data_1/grouping")
    raw_time = get("run/histogram_data_1/raw_time")
    return counts, grouping, raw_time


def _transverse_asymmetry(run: int, ring: int = 2, shift: int = 0):
    """Left/right (transverse) asymmetry from one detector ring's raw counts.

    The 32 detectors of a ring span the azimuth around the beam; a transverse
    field precesses them with an index-dependent phase that the file's ring-sum
    (forward/backward) grouping cancels. Splitting the ring into index-ordered
    halves (optionally rolled by *shift* for the quadrature pair) restores the
    transverse projection.
    """
    counts, grouping, t = _read_counts(run)
    idx = np.roll(np.where(grouping == ring)[0], shift)
    n = len(idx)
    left = counts[idx[: n // 2]].sum(0)
    right = counts[idx[n // 2:]].sum(0)
    total = left + right
    with np.errstate(invalid="ignore", divide="ignore"):
        asym = np.where(total > 0, (left - right) / total, np.nan)
    return np.asarray(t, dtype=float), asym


def _line_spectrum(run: int, f_min=0.15, f_max=3.2, n_f=900, t_min=0.15, t_max=10.0):
    """Quadrature amplitude spectrum of the transverse-regrouped asymmetry.

    For each ring the two orthogonal left/right splits (0 and quarter-ring roll)
    form a quadrature pair, making the amplitude estimate phase-insensitive; the
    two rings are averaged in quadrature. A cubic detrend removes the slow
    baseline (decay/detector imbalance). Returns (freqs [MHz], amplitude).
    """
    counts, grouping, t = _read_counts(run)
    mask = (t > t_min) & (t < t_max)
    tt = t[mask]
    freqs = np.linspace(f_min, f_max, n_f)
    cos_m = np.cos(2.0 * np.pi * np.outer(freqs, tt))
    sin_m = np.sin(2.0 * np.pi * np.outer(freqs, tt))

    total = np.zeros(n_f)
    n_terms = 0
    for ring in (1, 2):
        idx = np.where(grouping == ring)[0]
        quarter = len(idx) // 4
        for shift in (0, quarter):
            rolled = np.roll(idx, shift)
            n = len(rolled)
            left = counts[rolled[: n // 2]][:, mask].sum(0)
            right = counts[rolled[n // 2:]][:, mask].sum(0)
            asym = (left - right) / (left + right)
            resid = asym - np.polyval(np.polyfit(tt, asym, 3), tt)
            amp = 2.0 * np.hypot(cos_m @ resid, sin_m @ resid) / len(tt)
            total += amp**2
            n_terms += 1
    return freqs, np.sqrt(total / n_terms)


def _peak(freqs: np.ndarray, amp: np.ndarray):
    """Interpolated peak (freq, amplitude, significance) of a line spectrum."""
    j = int(np.argmax(amp))
    f_pk = float(freqs[j])
    if 0 < j < len(freqs) - 1:
        denom = amp[j - 1] - 2 * amp[j] + amp[j + 1]
        if denom != 0:
            f_pk += float(
                0.5 * (amp[j - 1] - amp[j + 1]) / denom * (freqs[1] - freqs[0])
            )
    floor = float(np.median(amp))
    mad = float(np.median(np.abs(amp - floor))) or 1e-12
    return f_pk, float(amp[j]), (float(amp[j]) - floor) / (1.4826 * mad)


def _save_figure(figure, ctx: CaptureContext, name: str):
    """Render *figure* to the scenario's PNG via a Qt pixmap (bscco pattern)."""
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from PySide6.QtCore import QEventLoop, QTimer
    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import QApplication

    canvas = FigureCanvasQTAgg(figure)
    canvas.draw()
    pix = QPixmap(canvas.size())
    canvas.render(pix)

    out_path = ctx.output_dir / f"{name}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not pix.save(str(out_path), "PNG"):
        raise RuntimeError(f"Failed to save screenshot to {out_path}")

    loop = QEventLoop()
    QTimer.singleShot(40, loop.quit)
    loop.exec()
    QApplication.processEvents()
    return out_path


# --------------------------------------------------------------------------- #
#  1. Headline — the H_c(T) dome, the coverage, and the measured B_int points.
# --------------------------------------------------------------------------- #
class SnHcDomeScenario(CorpusScenario):
    name = "corpus_sn_hc_dome"
    description = (
        "Type-I Sn H_c(T) phase boundary (guidance parabola, T_c = 3.717 K, "
        "B_c(0) = 306 G) with the corpus (T, B) coverage and the measured "
        "intermediate-state B_int from the 40 G scan: the domain field maps "
        "H_c(T) but sits 0.7–1.1 K left of the parabola — the thermometer error."
    )
    example = EXAMPLE

    def capture(self, ctx: CaptureContext):  # noqa: D401
        from matplotlib.figure import Figure

        # Measure the intermediate-state line for every 40 G-scan run from the
        # raw counts (genuine per-run measurements, not hard-coded numbers).
        scan_t, scan_b = [], []
        for run, t_nom in _SCAN_40G:
            freqs, amp = _line_spectrum(run)
            f_pk, _a_pk, _sig = _peak(freqs, amp)
            scan_t.append(t_nom)
            scan_b.append(f_pk / _GAMMA)
        scan_t = np.array(scan_t)
        scan_b = np.array(scan_b)

        # Intermediate-state points (line above the applied field) vs normal
        # points (line at the applied field: boundary crossed).
        inter = scan_b > 45.0
        t_cross_nom = float(scan_t[~inter].min()) if np.any(~inter) else np.nan
        t_cross_parab = _crossing_temperature(40.0)

        figure = Figure(figsize=(10.0, 6.8), dpi=120)
        figure.subplots_adjust(left=0.08, right=0.975, top=0.94, bottom=0.20)
        ax = figure.add_subplot(1, 1, 1)

        # Guidance parabola H_c(T).
        tt = np.linspace(0.0, _TC, 400)
        ax.plot(tt, _hc(tt), color="#1f77b4", lw=2.2, zorder=3,
                label=r"$H_c(T)=B_c(0)\,[1-(T/T_c)^2]$  (guidance, Karl 2019)")
        ax.fill_between(tt, 0.0, _hc(tt), color="#1f77b4", alpha=0.07, zorder=0)
        ax.text(0.15, 270, "superconducting\n(Meissner / intermediate state)",
                color="#1f77b4", fontsize=10, ha="left", va="center")
        ax.text(3.08, 262, "normal", color="0.4", fontsize=10, ha="left",
                va="center")

        # T_c and B_c(0) anchors.
        ax.scatter([_TC], [0.0], s=55, color="#1f77b4", zorder=5)
        ax.annotate(rf"$T_c={_TC:.3f}$ K", (_TC, 0.0), xytext=(_TC - 0.02, 18),
                    ha="right", fontsize=10, color="#1f77b4")
        ax.axhline(_BC0, color="0.7", ls=":", lw=1.0)
        ax.text(0.05, _BC0 + 4, rf"$B_c(0)={_BC0:.0f}$ G ($\approx$30.6 mT)",
                fontsize=9, color="0.4")

        # Applied fields + their parabola-crossing temperatures.
        for field in _FIELDS:
            tc_field = _crossing_temperature(field)
            ax.axhline(field, color="0.6", ls="--", lw=0.8, alpha=0.6)
            ax.text(0.05, field + 3, f"{field:.0f} G applied",
                    fontsize=8, color="0.35")
            ax.scatter([tc_field], [field], marker="v", s=55, color="#1f77b4",
                       zorder=6, alpha=0.85)

        # Dataset run coverage (nominal setpoint T, applied B).
        grid = np.array(_RUN_GRID, dtype=float)
        ax.scatter(grid[:, 0], grid[:, 1], marker="o", s=22, facecolor="none",
                   edgecolor="0.55", linewidths=0.8, zorder=4,
                   label="corpus runs (nominal setpoint T, applied B)")

        # Measured intermediate-state line positions from the 40 G scan.
        ax.plot(scan_t[inter], scan_b[inter], color="#d62728", lw=1.0,
                alpha=0.6, zorder=6)
        ax.scatter(scan_t[inter], scan_b[inter], marker="D", s=42,
                   color="#d62728", zorder=7,
                   label=r"measured $B_\mathrm{int}=B_c(T_\mathrm{real})$ "
                         "(40 G scan, intermediate state)")
        ax.scatter(scan_t[~inter], scan_b[~inter], marker="s", s=42,
                   color="#7f7f7f", zorder=7,
                   label="line at applied field (boundary crossed — normal)")

        # The thermometer-error arrow: observed crossing vs parabola crossing.
        y_arrow = 54.0
        ax.annotate(
            "", xy=(t_cross_parab, y_arrow), xytext=(t_cross_nom, y_arrow),
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.4),
        )
        ax.text(0.5 * (t_cross_nom + t_cross_parab), y_arrow + 5,
                rf"thermometer reads $\approx${t_cross_parab - t_cross_nom:.1f} K low",
                color="#d62728", fontsize=9, ha="center", va="bottom")

        ax.set_xlim(0.0, 3.9)
        ax.set_ylim(0.0, 320)
        ax.set_xlabel("Temperature  T (K)   —   measured points at *nominal* setpoint T")
        ax.set_ylabel(r"Field  $B$  (G)")
        ax.set_title("Critical fields in Sn — H_c(T) mapped by the "
                     "intermediate-state line (40 G scan)")
        ax.legend(loc="upper right", frameon=True, fontsize=8)
        ax.grid(True, alpha=0.2)
        figure.text(
            0.5, 0.015,
            "In the intermediate state the normal domains carry exactly B_c(T), so the measured line (red ◆) maps the phase boundary.\n"
            "Plotted at the nominal setpoint T it falls systematically left of the accepted parabola: inverting the parabola gives\n"
            f"T_real ≈ 2.73–3.40 K, i.e. the commissioning cryostat reads 0.7–1.1 K low (boundary crossing observed at nominal "
            f"{t_cross_nom:.1f} K\nvs {t_cross_parab:.2f} K expected) — the guide's \"how much is the thermometer in error?\" question, answered from the data.",
            color="0.35", fontsize=8, va="bottom", ha="center",
        )
        return _save_figure(figure, ctx, self.name)


# --------------------------------------------------------------------------- #
#  2. Spectral evidence — domain field independent of B_app; the marching line.
# --------------------------------------------------------------------------- #
class SnIntermediateLinesScenario(CorpusScenario):
    name = "corpus_sn_intermediate_lines"
    description = (
        "Transverse-regrouped line spectra: (top) base-T runs at 20/80/160 G — "
        "the 20 and 80 G lines coincide near 140 G (domain field = B_c, "
        "independent of applied field) while 160 G > B_c sits at the applied "
        "field (normal); (bottom) the 40 G scan line marching from 141 G down "
        "to the applied field at nominal 2.8 K."
    )
    example = EXAMPLE

    def capture(self, ctx: CaptureContext):  # noqa: D401
        from matplotlib.figure import Figure

        figure = Figure(figsize=(10.0, 7.6), dpi=120)
        figure.subplots_adjust(left=0.09, right=0.97, top=0.95, bottom=0.07,
                               hspace=0.32)

        # (a) Base-T field series: 20 / 80 / 160 G.
        ax1 = figure.add_subplot(2, 1, 1)
        colors = {20: "#2ca02c", 80: "#1f77b4", 160: "#d62728"}
        for run, field in [(91488, 20), (91490, 80), (91491, 160)]:
            freqs, amp = _line_spectrum(run)
            f_pk, _, sig = _peak(freqs, amp)
            ax1.plot(freqs / _GAMMA, amp * 100, color=colors[field], lw=1.2,
                     label=f"{field} G applied — line at {f_pk / _GAMMA:.0f} G "
                           f"({sig:.0f}σ)")
        ax1.axvline(140.0, color="0.5", ls=":", lw=1.0)
        ax1.text(141.0, 1.30, r"$B_c(T_\mathrm{real})\approx$140 G",
                 fontsize=9, color="0.35", va="top")
        ax1.set_xlim(0, 235)
        ax1.set_ylim(0, 1.45)
        ax1.set_xlabel("Internal field  B (G)")
        ax1.set_ylabel("Line amplitude (%)")
        ax1.set_title(
            "Base T (nominal 1 K): 20 G and 80 G share the domain field "
            "≈140 G = $B_c$; 160 G > $B_c$ → normal (line at applied field)")
        ax1.legend(loc="upper left", frameon=True, fontsize=9)
        ax1.grid(True, alpha=0.2)

        # (b) 40 G scan: stacked spectra, line marching down to 40 G.
        ax2 = figure.add_subplot(2, 1, 2)
        subset = [(91516, 1.6), (91520, 2.0), (91524, 2.4), (91526, 2.6),
                  (91527, 2.7), (91528, 2.8)]
        offset = 0.0
        step = 1.1
        for run, t_nom in subset:
            freqs, amp = _line_spectrum(run)
            f_pk, a_pk, _sig = _peak(freqs, amp)
            ax2.plot(freqs / _GAMMA, amp * 100 + offset, color="#1f77b4", lw=1.0)
            ax2.scatter([f_pk / _GAMMA], [a_pk * 100 + offset + 0.12],
                        marker="v", s=34, color="#d62728", zorder=5)
            ax2.text(231, offset + 0.18, f"nominal {t_nom:.1f} K",
                     fontsize=8.5, color="0.3", ha="right")
            offset += step
        ax2.axvline(40.0, color="0.5", ls="--", lw=1.0)
        ax2.text(42.0, offset + 0.75, "applied 40 G", fontsize=9, color="0.35",
                 va="top")
        ax2.set_xlim(0, 235)
        ax2.set_ylim(-0.1, offset + 1.0)
        ax2.set_xlabel("Internal field  B (G)")
        ax2.set_ylabel("Line amplitude (%, offset per run)")
        ax2.set_title(
            "40 G temperature scan: the intermediate-state line (red ▼) marches "
            r"$B_c(T)\rightarrow B_\mathrm{app}$ and doubles in amplitude at the "
            "crossing")
        ax2.grid(True, alpha=0.2)
        return _save_figure(figure, ctx, self.name)


# --------------------------------------------------------------------------- #
#  3. Transverse recovery — loader cancellation vs regrouped precession.
# --------------------------------------------------------------------------- #
class SnTransverseRecoveryScenario(CorpusScenario):
    name = "corpus_sn_transverse_recovery"
    description = (
        "HIFI 160 G Sn run 91491: the standard forward/backward asymmetry (noise) "
        "vs the left/right transverse regrouping of the same raw counts — a clean "
        "~2.14 MHz (≈158 G) applied-field precession the ring-sum grouping hides."
    )
    example = EXAMPLE

    def capture(self, ctx: CaptureContext):  # noqa: D401
        from matplotlib.figure import Figure
        from ._corpus import load_corpus_datasets

        # Standard-loader forward/backward asymmetry (what Asymmetry shows).
        ds = load_corpus_datasets([_DATA % _RUN_160G])[0]
        t_fb = np.asarray(ds.time, dtype=float)
        a_fb = np.asarray(ds.asymmetry, dtype=float)
        good = (np.abs(a_fb) < 99.0) & (t_fb > 0.15) & (t_fb < 8.0)

        # Left/right transverse regrouping of the raw counts (same run).
        t_tr, a_tr = _transverse_asymmetry(_RUN_160G, ring=2)
        m = (t_tr > 0.15) & (t_tr < 8.0) & np.isfinite(a_tr)
        tt, aa = t_tr[m], a_tr[m] - np.nanmean(a_tr[m])

        # Fit the recovered precession (A e^{-λt} cos(2πft+φ)) for the annotation.
        from scipy.optimize import curve_fit

        def osc(t, A, f, phi, lam, c):
            return A * np.exp(-lam * t) * np.cos(2 * np.pi * f * t + phi) + c

        popt, _ = curve_fit(
            osc, tt, aa, p0=[0.03, 160 * _GAMMA, 0.0, 0.1, 0.0], maxfev=20000
        )
        f_fit = abs(popt[1])
        b_fit = f_fit / _GAMMA

        figure = Figure(figsize=(10.0, 7.2), dpi=120)
        figure.subplots_adjust(left=0.10, right=0.97, top=0.94, bottom=0.14,
                               hspace=0.32)
        ax1 = figure.add_subplot(2, 1, 1)
        ax1.errorbar(
            t_fb[good], a_fb[good] - np.nanmedian(a_fb[good]),
            fmt=".", ms=3, color="#7f7f7f", alpha=0.7,
        )
        ax1.set_ylabel("F−B asymmetry\n(baseline-subtracted, %)")
        ax1.set_title(
            "Sn 160 G run 91491 — standard forward/backward grouping: "
            "transverse precession cancelled (noise)")
        ax1.set_ylim(-4.5, 4.5)
        ax1.grid(True, alpha=0.2)

        ax2 = figure.add_subplot(2, 1, 2, sharex=ax1)
        ax2.plot(tt, aa, color="#1f77b4", lw=0.8, alpha=0.9, label="L/R regrouped")
        tf = np.linspace(tt.min(), tt.max(), 1000)
        ax2.plot(tf, osc(tf, *popt), color="#d62728", lw=1.6,
                 label=rf"fit: $f={f_fit:.3f}$ MHz $\Rightarrow$ {b_fit:.0f} G "
                       rf"($B_\mathrm{{app}}=160$ G)")
        ax2.set_xlabel("Time  t (µs)")
        ax2.set_ylabel("L/R asymmetry\n(mean-subtracted)")
        ax2.set_title(
            "Same raw counts, left/right (transverse) regrouping: "
            "clean applied-field precession recovered")
        ax2.set_xlim(0.15, 8.0)
        ax2.legend(loc="upper right", frameon=True, fontsize=9)
        ax2.grid(True, alpha=0.2)
        figure.text(
            0.5, 0.025,
            "The precession is present in the raw HIFI counts but the file's forward/backward ring grouping cancels it. The recovered\n"
            "line sits at the applied field because 160 G exceeds B_c(T_real) ≈ 140 G — the sample is fully normal in this run; the\n"
            "intermediate-state B_c line appears instead in the 20/80/40 G runs (see corpus_sn_intermediate_lines).",
            color="0.35", fontsize=8, va="bottom", ha="center",
        )
        return _save_figure(figure, ctx, self.name)


register(SnHcDomeScenario())
register(SnIntermediateLinesScenario())
register(SnTransverseRecoveryScenario())
