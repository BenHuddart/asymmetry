"""Corpus scenarios — penetration depth in the pnictide LiFeAs (Superconductivity).

Worked example ``Superconductivity/LiFeAs`` (WiMDA muon school corpus), driving
Asymmetry through the transverse-field vortex-lattice **B_rms(T) → penetration
depth** workflow on the "111" iron-arsenide superconductor LiFeAs. Data are the
real PSI GPS ``.bin`` runs (``deltat_pta_gps_<run>.bin``), two samples:
**Sample 1 "LFA"** (runs 3366–3387, T_c = 16 K) and **Sample 2 "LFA_2"** (runs
3662–3697, T_c ≈ 12 K). The paper is the spec: F. L. Pratt *et al.*, "Enhanced
superfluid stiffness, lowered superconducting transition temperature, and
field-induced magnetic state of the pnictide superconductor LiFeAs," Phys. Rev.
B **79**, 052508 (2009). See the example's ``GROUND_TRUTH.md``.

Physics. In a transverse field B₀ the sample below T_c admits a vortex lattice,
whose inhomogeneous field distribution p(B) dephases the muon precession. The
extra (super-conducting) broadening adds in quadrature to the temperature-
independent nuclear width: **σ² = σ_VL² + σ_n²** (paper Eq. 2). The VL field
width is **B_rms = σ_VL/γ_µ** (γ_µ = 0.8516 µs⁻¹ mT⁻¹), and for a powder the
London limit gives (paper Eq. 3)

    B_rms = σ_VL/γ_µ = √0.00371 · φ₀ / (3^{1/4} λ_ab)²

so B_rms(T→0) → λ_ab. Sample 1's low-T plateau B_rms ≈ 1.9 mT ⇒ λ_ab = 195(2)
nm; Sample 2's ≈ 1.2 mT ⇒ 244(2) nm (GROUND_TRUTH §5/§6/§11).

Two corpus-specific data facts drive every scenario here:

* **Detector pairing.** These are spin-rotated ("TF WED") GPS runs: the muon
  spin is rotated transverse, so the precession appears in the **Up/Down**
  detector pair. The loader's *default* Forward/Back pairing sees the two
  detectors *in phase* and the precession **cancels** (leaving only a weak 2ν
  artefact). Every quantitative render regroups onto the Up/Down transverse
  pair — the essential data-reduction choice for this example (see
  ``_regroup``). This mirrors selecting Up/Down in the grouping dialog.
* **Gaussian convention.** Asymmetry's ``Gaussian`` component is
  A·exp(−(σt)²) (no ½), whereas the paper's relaxation is exp(−σ²t²/2); hence
  **σ_paper = √2 · σ_Asymmetry**. B_rms in mT is therefore
  √2·σ_VL[µs⁻¹]/0.8516 (see ``_brms_mT``).

* **Weakly-relaxing background.** <20 % of the signal is muons stopping in the
  Ag sample holder, precessing at the same ν with only slow nuclear damping
  (paper §II). A single Gaussian is diluted by this persistent tail (base-T
  B_rms falls to ~1.1 mT); the physical σ_VL is recovered with a **two-Gaussian
  signal+background** model, which lands base-T B_rms ≈ 1.79 mT — the paper's
  1.8–1.9 mT plateau. That two-component model is used for the fits.

Scenarios registered:

* ``corpus_lifeas_pair_select`` — data-handling: the WED pairing insight, a
  two-panel figure contrasting the cancelled default Forward/Back spectrum with
  the clean Up/Down transverse precession (base-T Sample 1, run 3366).
* ``corpus_lifeas_tf_fit`` — converged two-component TF fit on the base-T
  Sample-1 run (3366, 1.5 K, 40 mT), Up/Down pair: σ_VL → B_rms ≈ 1.8 mT
  (λ_ab ≈ 195 nm).
* ``corpus_lifeas_brms_t`` — **headline**: both samples' B_rms(T) — the Fig. 1
  digitised paper curves (GROUND_TRUTH §11) with the real Asymmetry Sample-1
  fits overlaid, plateaus ≈ 1.9 / ≈ 1.2 mT, T_c ≈ 16 / 12 K, and the λ_ab =
  195/244 nm conversions annotated via Eq. (3).
* ``corpus_lifeas_vortex_lineshape`` — normal vs superconducting: time-domain
  overlay of the 1.5 K (broad vortex damping) and 18 K (narrow, normal-state)
  40 mT spectra, the field-distribution broadening in time.

``requires_fit = True`` on every scenario that runs a real iminuit fit at
capture time.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from ._corpus import CorpusScenario, register, _process_events_for
from .._base import CaptureContext

EXAMPLE = "Superconductivity/LiFeAs"
_DATA = "Superconductivity/LiFeAs/data/deltat_pta_gps_%d.bin"

# --- physical constants (GROUND_TRUTH §6) ---------------------------------- #
_GAMMA = 0.8516          # µs⁻¹ per mT: σ_paper[µs⁻¹] = γ_µ · B_rms[mT]
_SQRT2 = np.sqrt(2.0)    # Asymmetry Gaussian exp(−(σt)²) → σ_paper = √2 σ_A
_PHI0 = 2.067833848e-15  # Wb, flux quantum

# --- Sample-1 (LFA) 40 mT (400 G) temperature scan, ascending T ------------ #
# Runs 3366–3373 are the 400 G T-scan; 3373 (18 K) is above T_c = 16 K, i.e.
# the normal-state σ_n reference at 40 mT (GROUND_TRUTH §3 note).
_SCAN_S1: list[tuple[int, float]] = [
    (3366, 1.5), (3367, 4.0), (3368, 7.0), (3369, 10.0),
    (3370, 12.0), (3371, 14.0), (3372, 16.0), (3373, 18.0),
]
_S1_BASE = 3366   # 1.5 K, deep in the mixed state (broad vortex line)
_S1_NORMAL = 3373  # 18 K, above T_c (narrow, nuclear-only line)
_S1_FREQ = 5.44    # 400 G Larmor line (MHz)
_S2_BASE = 3663    # Sample 2, 1.5 K, 200 G

# Transverse (Up/Down) detector pair for the spin-rotated GPS runs (0-based
# histogram indices: 0 Forw, 1 Back, 2 Up, 3 Down, 4 Righ).
_FWD_IDX = [2]
_BWD_IDX = [3]

# Digitised Fig. 1 B_rms(T) at B₀ = 40 mT (GROUND_TRUTH §11, 600 dpi). The
# corpus grading target for the σ(T)/B_rms(T) trend shape and magnitude.
_FIG1_S1: list[tuple[float, float]] = [
    (1.6, 1.93), (3.9, 1.77), (7.3, 1.48), (10.4, 1.04),
    (12.6, 0.74), (14.7, 0.43), (16.9, 0.11), (19.0, 0.02),
]
_FIG1_S2: list[tuple[float, float]] = [
    (1.4, 1.31), (2.9, 1.10), (4.0, 1.01), (5.1, 0.89), (6.1, 0.86),
    (6.8, 0.73), (8.3, 0.45), (9.3, 0.29), (10.4, 0.17), (11.5, 0.09),
    (12.6, 0.04), (14.7, 0.02),
]

# Two-component signal+background TF model:
#   [A_1·osc·exp(−(σ_2 t)²)] + [A_3·osc·exp(−(σ_4 t)²)] + A_bg
# component 1/2 = the vortex-lattice signal, 3/4 = the weakly-relaxing Ag
# background at the same Larmor frequency (GROUND_TRUTH §4/§6).
_TF2_COMPONENTS = ["Oscillatory", "Gaussian", "Oscillatory", "Gaussian", "Constant"]
_TF2_OPERATORS = ["*", "+", "*", "+"]


def _rel(run: int) -> str:
    return _DATA % run


def _brms_mT(sigma_a: float, sigma_n: float) -> float:
    """Convert an Asymmetry Gaussian σ (main + nuclear) to VL field width B_rms.

    ``sigma_a`` and ``sigma_n`` are the fitted *Asymmetry* Gaussian widths of the
    signal and of the normal-state nuclear reference (both µs⁻¹). Subtract in
    quadrature (paper Eq. 2), apply the √2 convention factor, then divide by γ_µ.
    """
    svl_a = np.sqrt(max(sigma_a**2 - sigma_n**2, 0.0))
    return _SQRT2 * svl_a / _GAMMA


def _brms_from_lambda(lambda_nm: float) -> float:
    """Paper Eq. (3): B_rms (mT) for an ab-plane penetration depth λ_ab (nm)."""
    lam = lambda_nm * 1e-9
    return (0.00371**0.5) * _PHI0 / ((3.0**0.25 * lam) ** 2) * 1e3


def _regroup(run: int, rebin: int = 10):
    """Load a run and rebuild its asymmetry from the Up/Down transverse pair.

    The default loader pairs Forward/Back, which cancels the precession for
    these spin-rotated (TF WED) runs. This reconstructs the asymmetry from the
    Up/Down pair with the *same* transform primitives the loader/grouping dialog
    use, then rebins (PSI 1.25 ns bins are far finer than the physics needs).
    """
    from asymmetry.core.data.dataset import MuonDataset
    from asymmetry.core.io import load
    from asymmetry.core.transform import (
        apply_grouping_aligned,
        common_t0_for_groups,
        compute_asymmetry,
    )

    from ._corpus import corpus_path

    dataset = load(str(corpus_path(_rel(run))))
    hists = dataset.run.histograms
    ct0 = common_t0_for_groups(hists, _FWD_IDX, _BWD_IDX)
    fwd = apply_grouping_aligned(hists, _FWD_IDX, common_t0_bin=ct0)
    bwd = apply_grouping_aligned(hists, _BWD_IDX, common_t0_bin=ct0)
    n = min(len(fwd), len(bwd))
    asym, err = compute_asymmetry(fwd[:n], bwd[:n], alpha=1.0)
    asym *= 100.0
    err *= 100.0

    pair = _FWD_IDX + _BWD_IDX
    good_off = [max(0, hists[i].good_bin_start - hists[i].t0_bin) for i in pair]
    last_off = [max(0, hists[i].good_bin_end - hists[i].t0_bin) for i in pair]
    first = min(n - 1, int(ct0) + max(good_off))
    last = min(n - 1, int(ct0) + min(last_off))
    if last < first:
        last = first
    bw = hists[0].bin_width
    time = (np.arange(n, dtype=float) - float(ct0)) * bw
    sl = slice(first, last + 1)
    ds = MuonDataset(
        time=time[sl], asymmetry=asym[sl], error=err[sl],
        metadata=dict(dataset.metadata), run=dataset.run,
    )
    return ds.rebin(rebin) if rebin > 1 else ds


def _tf2_seeds(dataset, sigma_seed: float, freq: float) -> dict:
    """Seeds for the two-component TF model on one regrouped run."""
    a = np.asarray(dataset.asymmetry, dtype=float)
    baseline = float(np.nanmedian(a[np.abs(a) < 99.0]))
    return {
        "A_1": 12.0, "frequency_1": freq, "phase_1": 0.0,
        "sigma_2": max(sigma_seed, 0.1),
        "A_3": 3.0, "frequency_3": freq, "phase_3": 0.0,
        "sigma_4": 0.12,
        "A_bg": baseline,
    }


def _tf2_paramset(seeds: dict, freq: float):
    """Build the bounded ParameterSet for the two-component TF fit."""
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    return ParameterSet([
        Parameter("A_1", seeds["A_1"], min=0.0, max=40.0),
        Parameter("frequency_1", seeds["frequency_1"], min=freq - 0.25, max=freq + 0.25),
        Parameter("phase_1", seeds["phase_1"], min=-3.2, max=3.2),
        Parameter("sigma_2", seeds["sigma_2"], min=0.05, max=4.0),
        Parameter("A_3", seeds["A_3"], min=0.0, max=20.0),
        Parameter("frequency_3", seeds["frequency_3"], min=freq - 0.25, max=freq + 0.25),
        Parameter("phase_3", seeds["phase_3"], min=-3.2, max=3.2),
        Parameter("sigma_4", seeds["sigma_4"], min=0.0, max=0.5),
        Parameter("A_bg", seeds["A_bg"]),
    ])


def _fit_tf2(dataset, sigma_seed: float, freq: float):
    """Fit the two-component TF model; return (sigma_main, A_main, A_bg_amp)."""
    from asymmetry.core.fitting.composite import CompositeModel
    from asymmetry.core.fitting.engine import FitEngine

    model = CompositeModel(_TF2_COMPONENTS, operators=_TF2_OPERATORS)
    params = _tf2_paramset(_tf2_seeds(dataset, sigma_seed, freq), freq)
    result = FitEngine().fit(dataset, model.function, params)
    by_name = {p.name: p.value for p in result.parameters}
    s2, s4 = abs(by_name["sigma_2"]), abs(by_name["sigma_4"])
    a1, a3 = abs(by_name["A_1"]), abs(by_name["A_3"])
    # Order so the *main* (larger-σ, vortex) component is reported first.
    if s4 > s2:
        s2, s4, a1, a3 = s4, s2, a3, a1
    return s2, a1, a3


def _fit_single_sigma(dataset, sigma_seed: float, freq: float) -> float:
    """Fit a single Oscillatory×Gaussian; return the Asymmetry Gaussian σ.

    Used for the normal-state (18 K) nuclear reference σ_n, where there is no
    vortex broadening and the two-component split is degenerate.
    """
    from asymmetry.core.fitting.composite import CompositeModel
    from asymmetry.core.fitting.engine import FitEngine
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    model = CompositeModel(["Oscillatory", "Gaussian", "Constant"], operators=["*", "+"])
    a = np.asarray(dataset.asymmetry, dtype=float)
    baseline = float(np.nanmedian(a[np.abs(a) < 99.0]))
    params = ParameterSet([
        Parameter("A_1", 12.0, min=0.0, max=40.0),
        Parameter("frequency", freq, min=freq - 0.3, max=freq + 0.3),
        Parameter("phase", 0.0),
        Parameter("sigma", max(sigma_seed, 0.05), min=0.0, max=4.0),
        Parameter("A_bg", baseline),
    ])
    result = FitEngine().fit(dataset, model.function, params)
    return abs(float({p.name: p.value for p in result.parameters}["sigma"]))


def _fit_s1_brms_series():
    """Fit the Sample-1 40 mT T-scan; return (T, B_rms, keep_mask).

    Warm-starts σ downward in temperature. The two-Gaussian model is stable
    while the VL signal dominates (low T); as the SC signal collapses toward
    T_c the signal/background split degenerates, so ``keep_mask`` flags the
    points where the main component still dominates (A_main ≳ A_bg).
    """
    temps, brms, keep = [], [], []
    sigma_seed = 1.05
    # Nuclear reference σ_n: the 18 K normal-state run has no vortex broadening,
    # so a *single* Gaussian gives a clean nuclear width (the two-component split
    # is degenerate there).
    sigma_n = _fit_single_sigma(_regroup(_S1_NORMAL), 0.2, _S1_FREQ)
    for run, temp in _SCAN_S1:
        ds = _regroup(run)
        s_main, a_main, a_bg = _fit_tf2(ds, sigma_seed, _S1_FREQ)
        signal_frac = a_main / (a_main + a_bg + 1e-9)
        # Keep points where the broad vortex component is well-defined: its σ
        # clearly exceeds the nuclear width and it still dominates the signal.
        well_defined = (s_main > 1.3 * sigma_n) and (signal_frac > 0.55)
        temps.append(float(temp))
        brms.append(_brms_mT(s_main, sigma_n))
        keep.append(bool(well_defined))
        sigma_seed = max(s_main * 0.9, 0.15)
    return np.array(temps), np.array(brms), np.array(keep, dtype=bool)


# --------------------------------------------------------------------------- #
#  1. Detector pairing — the WED transverse-pair data-reduction choice.
# --------------------------------------------------------------------------- #
class LifeasPairSelectScenario(CorpusScenario):
    name = "corpus_lifeas_pair_select"
    description = (
        "LiFeAs base-T Sample-1 run (3366, 1.5 K, 40 mT): the default "
        "Forward/Back pairing cancels the spin-rotated (WED) precession (top), "
        "while the Up/Down transverse pair recovers the clean 5.44 MHz vortex-"
        "damped signal (bottom) — the essential grouping choice for this example."
    )
    example = EXAMPLE
    size = (1200, 820)

    def capture(self, ctx: CaptureContext):
        from matplotlib.figure import Figure

        # Same run, two detector pairings.
        ud = _regroup(_S1_BASE, rebin=10)
        fb = _regroup_pair(_S1_BASE, [0], [1], rebin=10)

        figure = Figure(figsize=(9.6, 6.6), dpi=120, tight_layout=True)
        for ax, ds, title, color in (
            (figure.add_subplot(2, 1, 1), fb,
             "Default pairing  Forward / Back  —  precession cancels (WED spin rotation)",
             "#888888"),
            (figure.add_subplot(2, 1, 2), ud,
             "Transverse pairing  Up / Down  —  clean 400 G vortex-damped precession",
             "#1f77b4"),
        ):
            t = np.asarray(ds.time, dtype=float)
            a = np.asarray(ds.asymmetry, dtype=float)
            m = (t >= 0.0) & (t <= 2.5) & (np.abs(a) < 99.0)
            ax.plot(t[m], a[m], color=color, lw=1.0)
            ax.set_title(title, fontsize=10)
            ax.set_ylabel("Asymmetry (%)")
            ax.grid(True, alpha=0.25)
            ax.set_xlim(0.0, 2.5)
        figure.axes[-1].set_xlabel("Time  t (µs)")

        return _save_canvas(figure, ctx, self.name)


# --------------------------------------------------------------------------- #
#  2. Converged two-component TF fit on the base-T Sample-1 run.
# --------------------------------------------------------------------------- #
class LifeasTfFitScenario(CorpusScenario):
    name = "corpus_lifeas_tf_fit"
    description = (
        "Converged two-Gaussian TF fit on the LiFeAs Sample-1 base-T run "
        "(3366, 1.5 K, 40 mT, Up/Down pair): the vortex signal σ_2 with the "
        "weakly-relaxing Ag background σ_4 → B_rms ≈ 1.8 mT (λ_ab ≈ 195 nm)."
    )
    example = EXAMPLE
    size = (1500, 920)
    requires_fit = True

    def build(self) -> QWidget:
        from asymmetry.core.fitting.composite import CompositeModel
        from asymmetry.gui.mainwindow import MainWindow
        from asymmetry.gui.panels.fit.tab_base import (
            _param_table_rows_by_name,
            _set_param_table_value,
        )

        window = MainWindow()
        window._on_fit()
        window.resizeDocks(
            [window._dock_data_browser], [320], Qt.Orientation.Horizontal
        )

        dataset = _regroup(_S1_BASE)
        self.add_to_browser(window, [dataset])
        window._on_dataset_selected(dataset.run_number)
        _process_events_for(milliseconds=80)

        single_tab = window._fit_panel._single_tab
        single_tab._set_composite_model(
            CompositeModel(_TF2_COMPONENTS, operators=_TF2_OPERATORS)
        )
        _process_events_for(milliseconds=80)

        table = single_tab._param_table
        rows = _param_table_rows_by_name(table)
        seeds = _tf2_seeds(dataset, sigma_seed=1.05, freq=_S1_FREQ)
        # Tight frequency bounds keep the two oscillators on the 400 G line.
        bounds = {
            "frequency_1": (_S1_FREQ - 0.25, _S1_FREQ + 0.25),
            "frequency_3": (_S1_FREQ - 0.25, _S1_FREQ + 0.25),
            "sigma_2": (0.05, 4.0), "sigma_4": (0.0, 0.5),
            "A_1": (0.0, 40.0), "A_3": (0.0, 20.0),
        }
        for name, value in seeds.items():
            if name in rows:
                _set_param_table_value(table, rows[name], value)
        for name, (lo, hi) in bounds.items():
            if name in rows:
                lo_item = table.item(rows[name], table.COL_MIN)
                hi_item = table.item(rows[name], table.COL_MAX)
                if lo_item is not None:
                    lo_item.setText(f"{lo}")
                if hi_item is not None:
                    hi_item.setText(f"{hi}")
        _process_events_for(milliseconds=60)

        single_tab._run_fit()
        single_tab.wait_for_fit()
        _process_events_for(milliseconds=80)

        # Zoom to the first ~2.5 µs so precession cycles (~0.18 µs at 5.44 MHz)
        # and the Gaussian vortex envelope are both resolved.
        self._frame_y(window, 0.0, 2.5)
        _process_events_for(milliseconds=80)
        return window

    @staticmethod
    def _frame_y(window, x_min: float, x_max: float) -> None:
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
        pad = 0.12 * (hi - lo or 1.0)
        panel.set_view_limits(x_min, x_max, lo - pad, hi + pad)


# --------------------------------------------------------------------------- #
#  3. Headline — two-sample B_rms(T) with λ_ab conversions.
# --------------------------------------------------------------------------- #
class LifeasBrmsTScenario(CorpusScenario):
    name = "corpus_lifeas_brms_t"
    description = (
        "LiFeAs vortex-lattice field width B_rms(T) at 40 mT for both samples: "
        "Sample 1 (T_c = 16 K, plateau ≈ 1.9 mT ⇒ λ_ab = 195 nm) and Sample 2 "
        "(T_c ≈ 12 K, ≈ 1.2 mT ⇒ 244 nm). Paper Fig. 1 (digitised) with the real "
        "Asymmetry Sample-1 two-Gaussian fits overlaid."
    )
    example = EXAMPLE
    size = (1240, 820)
    requires_fit = True

    def capture(self, ctx: CaptureContext):
        from matplotlib.figure import Figure

        s1 = np.array(_FIG1_S1)
        s2 = np.array(_FIG1_S2)
        ft, fb, keep = _fit_s1_brms_series()

        figure = Figure(figsize=(9.6, 6.4), dpi=120, tight_layout=True)
        ax = figure.add_subplot(1, 1, 1)

        # Paper Fig. 1 digitised reference (guide-to-the-eye curves).
        ax.plot(s1[:, 0], s1[:, 1], "-", color="#1f77b4", lw=1.0, alpha=0.5)
        ax.plot(s1[:, 0], s1[:, 1], "o", color="#1f77b4", ms=6,
                label="Sample 1 — LFA, 40 mT (Pratt 2009 Fig. 1)")
        ax.plot(s2[:, 0], s2[:, 1], "-", color="#d62728", lw=1.0, alpha=0.5)
        ax.plot(s2[:, 0], s2[:, 1], "s", color="#d62728", ms=6,
                label="Sample 2 — LFA_2, 40 mT (Pratt 2009 Fig. 1)")

        # Real Asymmetry Sample-1 fits (open markers), where the VL signal
        # dominates the two-Gaussian split.
        ax.plot(ft[keep], fb[keep], "D", mfc="none", mec="#0b3d63", mew=1.6,
                ms=8, label="Sample 1 — Asymmetry two-Gaussian fit (this work)")

        # λ_ab conversions via Eq. (3), drawn as plateau guide lines.
        for lam, color, sample in ((195.0, "#1f77b4", "1"), (244.0, "#d62728", "2")):
            b = _brms_from_lambda(lam)
            ax.axhline(b, color=color, ls=":", lw=1.0, alpha=0.7)
            ax.text(19.6, b, f"  λ_ab({sample}) = {lam:.0f} nm", color=color,
                    fontsize=8, va="center", ha="left")

        ax.axvline(16.0, color="#1f77b4", ls="--", lw=0.8, alpha=0.5)
        ax.axvline(12.0, color="#d62728", ls="--", lw=0.8, alpha=0.5)
        ax.text(16.0, 2.02, r" $T_\mathrm{c}$=16 K", color="#1f77b4",
                fontsize=8, va="bottom", ha="left")
        ax.text(12.0, 2.02, r" 12 K", color="#d62728",
                fontsize=8, va="bottom", ha="right")

        ax.set_xlabel("Temperature  T (K)")
        ax.set_ylabel(r"VL field width  $B_\mathrm{rms}$  (mT)")
        ax.set_title("LiFeAs vortex-lattice linewidth B_rms(T) at B₀ = 40 mT — two samples")
        ax.set_xlim(0.0, 25.0)
        ax.set_ylim(0.0, 2.15)
        ax.legend(loc="upper right", frameon=True, fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.text(
            0.015, 0.03,
            "Eq. (3):  B_rms = √0.00371·φ₀/(3^{1/4}λ_ab)²   ⇒   195 nm→1.91 mT, 244 nm→1.22 mT.\n"
            "Asymmetry σ (Gaussian exp(−(σt)²)) → σ_paper = √2·σ; B_rms = √2·σ_VL/γ_µ, γ_µ=0.8516 µs⁻¹mT⁻¹.\n"
            "Sample-2 corpus runs are 1.5/20 K field pairs, not a T-scan — its curve is the digitised Fig. 1.",
            transform=ax.transAxes, color="0.35", fontsize=7.5, va="bottom", ha="left",
        )
        return _save_canvas_agg(figure, ctx, self.name)


# --------------------------------------------------------------------------- #
#  4. Normal vs superconducting lineshape — vortex broadening in time.
# --------------------------------------------------------------------------- #
class LifeasVortexLineshapeScenario(CorpusScenario):
    name = "corpus_lifeas_vortex_lineshape"
    description = (
        "LiFeAs Sample-1 40 mT lineshape p(B) (FFT of the Up/Down pair): at 18 K "
        "(normal state) a narrow nuclear-only line at the 5.44 MHz Larmor "
        "frequency; at 1.5 K (mixed state) the vortex lattice broadens it — the "
        "field-distribution width whose second moment gives B_rms."
    )
    example = EXAMPLE
    size = (1160, 760)

    def capture(self, ctx: CaptureContext):
        from matplotlib.figure import Figure

        # Light rebin only (keep frequency resolution); FFT the good-bin span.
        base = _regroup(_S1_BASE, rebin=4)
        normal = _regroup(_S1_NORMAL, rebin=4)

        figure = Figure(figsize=(9.0, 5.9), dpi=120, tight_layout=True)
        ax = figure.add_subplot(1, 1, 1)
        for ds, color, label in (
            (normal, "#d62728", "18 K — normal state (nuclear-only, narrow line)"),
            (base, "#1f77b4", "1.5 K — mixed state (vortex-broadened line)"),
        ):
            f, amp = _fft_lineshape(ds)
            band = (f >= 3.0) & (f <= 8.0)
            peak = float(np.max(amp[band])) if np.any(band) else 1.0
            ax.plot(f[band], amp[band] / peak, color=color, lw=1.6, label=label)
        ax.axvline(_S1_FREQ, color="0.5", ls="--", lw=0.9)
        ax.text(_S1_FREQ - 0.06, 0.45, f"400 G Larmor = {_S1_FREQ:.2f} MHz",
                color="0.4", fontsize=8, va="center", ha="right", rotation=90)
        ax.set_xlabel("Frequency  ν (MHz)")
        ax.set_ylabel("FFT amplitude  (normalised)")
        ax.set_title("LiFeAs Sample-1 field distribution p(B) at 40 mT — normal vs superconducting")
        ax.set_xlim(3.0, 8.0)
        ax.set_ylim(0.0, 1.12)
        ax.legend(loc="upper right", frameon=True, fontsize=9)
        ax.grid(True, alpha=0.25)
        return _save_canvas_agg(figure, ctx, self.name)


# --------------------------------------------------------------------------- #
#  Shared helpers (regroup with an arbitrary pair; matplotlib → PNG).
# --------------------------------------------------------------------------- #
def _regroup_pair(run: int, fwd_idx, bwd_idx, rebin: int = 10):
    """Like :func:`_regroup` but for an explicit forward/backward index pair."""
    from asymmetry.core.data.dataset import MuonDataset
    from asymmetry.core.io import load
    from asymmetry.core.transform import (
        apply_grouping_aligned,
        common_t0_for_groups,
        compute_asymmetry,
    )

    from ._corpus import corpus_path

    dataset = load(str(corpus_path(_rel(run))))
    hists = dataset.run.histograms
    ct0 = common_t0_for_groups(hists, fwd_idx, bwd_idx)
    fwd = apply_grouping_aligned(hists, fwd_idx, common_t0_bin=ct0)
    bwd = apply_grouping_aligned(hists, bwd_idx, common_t0_bin=ct0)
    n = min(len(fwd), len(bwd))
    asym, err = compute_asymmetry(fwd[:n], bwd[:n], alpha=1.0)
    asym *= 100.0
    err *= 100.0
    pair = list(fwd_idx) + list(bwd_idx)
    good_off = [max(0, hists[i].good_bin_start - hists[i].t0_bin) for i in pair]
    last_off = [max(0, hists[i].good_bin_end - hists[i].t0_bin) for i in pair]
    first = min(n - 1, int(ct0) + max(good_off))
    last = min(n - 1, int(ct0) + min(last_off))
    if last < first:
        last = first
    bw = hists[0].bin_width
    time = (np.arange(n, dtype=float) - float(ct0)) * bw
    sl = slice(first, last + 1)
    ds = MuonDataset(
        time=time[sl], asymmetry=asym[sl], error=err[sl],
        metadata=dict(dataset.metadata), run=dataset.run,
    )
    return ds.rebin(rebin) if rebin > 1 else ds


def _fft_lineshape(dataset, t_lo: float = 0.05, t_hi: float = 6.0):
    """Return (freq_MHz, amplitude) FFT of a regrouped run over the good span."""
    t = np.asarray(dataset.time, dtype=float)
    a = np.asarray(dataset.asymmetry, dtype=float)
    m = (t >= t_lo) & (t <= t_hi) & (np.abs(a) < 99.0)
    tt, aa = t[m], a[m]
    aa = aa - np.mean(aa)
    dt = float(np.median(np.diff(tt)))
    n = len(aa)
    windowed = aa * np.hanning(n)
    # Zero-pad ×8 so the (physically damping-limited) linewidths are sampled
    # smoothly rather than in coarse 1/T bins.
    n_fft = 1
    while n_fft < n * 8:
        n_fft *= 2
    freq = np.fft.rfftfreq(n_fft, d=dt)
    amp = np.abs(np.fft.rfft(windowed, n=n_fft))
    return freq, amp


def _save_canvas(figure, ctx: CaptureContext, name: str):
    """Render a Matplotlib figure to a QPixmap PNG (QWidget.grab-free)."""
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


def _save_canvas_agg(figure, ctx: CaptureContext, name: str):
    """Render a Matplotlib figure straight from the Agg buffer to PNG."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    out_path = ctx.output_dir / f"{name}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(out_path), format="png", dpi=120)
    return out_path


register(LifeasPairSelectScenario())
register(LifeasTfFitScenario())
register(LifeasBrmsTScenario())
register(LifeasVortexLineshapeScenario())
