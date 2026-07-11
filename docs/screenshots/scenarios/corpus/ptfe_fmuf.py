"""Corpus scenarios — the FµF state in PTFE (Nuclear magnetism & ionic motion).

Drives the Asymmetry GUI through the WiMDA muon-school **F–µ–F in PTFE**
example on the real MuSR ``.nxs`` corpus files (ISIS NeXus-v1 HDF4, runs
17293–17322, Teflon 2008). See the example's ``GROUND_TRUTH.md``.

The implanted µ⁺ stays diamagnetic and sits between two F⁻, forming a closely
coupled collinear **F⁻–µ⁺–F⁻** three-spin unit. Its zero-field polarisation
carries the characteristic non-exponential FµF oscillation — a dip and partial
recovery at the three combination frequencies ≈ (0.63, 1.73, 2.37)·ν_d, where
ν_d = µ₀γ_µγ_F ℏ/(16π²r³) maps directly to the µ–F distance r (GROUND_TRUTH §5,
§6, §10). The deliverable named by the guide is that dipolar coupling frequency
(→ bond length); the guide gives no numeric target, and the literature ballpark
is r ≈ 1.1–1.2 Å (Brewer 1986 CaF₂: 1.172 Å).

Scenarios registered:

* ``corpus_ptfe_zf_signature``   — the raw ZF FµF signature: base-T (20 K),
  highest-statistics ZF run 17294, zoomed to the first ~8 µs so the
  dip → recovery → dip beating reads clearly (no fit).
* ``corpus_ptfe_fmuf_fit``       — the headline: ``FmuF_Linear * Gaussian +
  Constant`` converged on run 17294, r_µF ≈ 1.30 Å visible in the fit table.
  A bare ``FmuF_Linear`` pins r_µF at its bound (GROUND_TRUTH §7); the Gaussian
  envelope (damping by more distant fluorines) is what makes it fit.
* ``corpus_ptfe_fft``            — frequency-domain view: the sub-MHz FµF line
  cluster (broad combination lines near ~0.3–0.5 MHz on the DC skirt).
* ``corpus_ptfe_tf_calibration`` — the prescribed calibration step
  (GROUND_TRUTH §4): the TF 20 G run 17293, showing the slow ~0.3 MHz
  precession used to fix the detector balance.

``requires_fit = True`` on the fit scenario, which runs a real iminuit fit at
capture time (iminuit/numba trips on numpy ≥ 2.3 in dev environments; CI pins
numpy < 2.3).
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTabBar, QWidget

from ._corpus import CorpusScenario, _process_events_for, load_corpus_datasets, register

EXAMPLE = "Nuclear magnetism and ionic motion/The FmuF state in PTFE"
_DATA = "Nuclear magnetism and ionic motion/The FmuF state in PTFE/Data/musr000%d.nxs"

# FµF collinear three-spin ZF polarisation × Gaussian damping + background:
#   G(t) = A₁·G_FµF(t; r_µF)·exp(−σ²t²/2) + A_bg.
# The bare FmuF_Linear+Constant pins r_µF at its bound (χ²ᵣ ≈ 62, GROUND_TRUTH
# §7); the more distant fluorines damp the oscillation, so a Gaussian envelope
# is required to make the model fit.
_FMUF_MODEL = (["FmuF_Linear", "Gaussian", "Constant"], ["*", "+"])

# Base temperature, highest-statistics zero-field run (20 K, 41.6 MEv;
# GROUND_TRUTH §3): the cleanest FµF beating in the set.
_ZF_BASE_RUN = 17294
# TF 20 G calibration run taken while cooling (GROUND_TRUTH §4).
_TF_CAL_RUN = 17293


def _rel(run: int) -> str:
    return _DATA % run


def _raise_inspector_tab(window, tab_label: str) -> None:
    """Select *tab_label* in the right inspector deck's tab bar.

    ``QDockWidget.raise_()`` is a silent no-op for tabified docks under the
    offscreen QPA platform, so the deck's ``QTabBar`` — identified by carrying
    both *tab_label* and the always-present "Fit" tab — is driven directly
    (same approach as the synthetic ``fourier_tf`` scenario).
    """
    for tab_bar in window.findChildren(QTabBar):
        labels = [tab_bar.tabText(i) for i in range(tab_bar.count())]
        if tab_label in labels and "Fit" in labels:
            tab_bar.setCurrentIndex(labels.index(tab_label))
            _process_events_for(milliseconds=80)
            return


def _frame_y_to_window(window, x_min: float, x_max: float, *, pad_frac: float = 0.10) -> None:
    """Clamp the plot Y-axis to the asymmetry spread inside [x_min, x_max].

    The ZF FµF amplitude (~15 % at t=0) decays into its baseline within a few
    µs; autoscaling over the whole record flattens the beating against the
    late-time error fan. Reframing Y to the visible window makes the dip and
    recovery sit large on screen (same trick as the EuO ZF-fit scenario).
    """
    panel = window._plot_panel
    t = getattr(panel, "_last_plot_time", None)
    a = getattr(panel, "_last_plot_asymmetry", None)
    if t is None or a is None or not len(t):
        return
    t = np.asarray(t, dtype=float)
    a = np.asarray(a, dtype=float)
    m = (t >= x_min) & (t <= x_max)
    if not np.any(m):
        return
    lo, hi = float(np.nanmin(a[m])), float(np.nanmax(a[m]))
    pad = pad_frac * (hi - lo or 1.0)
    panel.set_view_limits(x_min, x_max, lo - pad, hi + pad)


# --------------------------------------------------------------------------- #
#  1. The raw ZF FµF signature — dip → recovery → dip (no fit).
# --------------------------------------------------------------------------- #
class PtfeZfSignatureScenario(CorpusScenario):
    name = "corpus_ptfe_zf_signature"
    description = (
        "Raw zero-field FµF signature in PTFE: base-T (20 K) run 17294, zoomed "
        "to the first ~8 µs so the characteristic non-exponential dip and "
        "partial recovery of the F–µ–F three-spin oscillation read clearly."
    )
    example = EXAMPLE
    size = (1500, 920)

    def build(self) -> QWidget:
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window.resizeDocks([window._dock_data_browser], [320], Qt.Orientation.Horizontal)

        datasets = load_corpus_datasets([_rel(_ZF_BASE_RUN)])
        self.add_to_browser(window, datasets)
        window._on_dataset_selected(datasets[0].run_number)
        _process_events_for(milliseconds=120)

        # The FµF beat lives in the first ~5 µs (dip near 1.4 µs, recovery bump
        # near 2.4 µs, second dip near 3.8 µs); frame the first 8 µs so the full
        # pattern reads before it decays into the baseline and the ZF error fan.
        window._plot_panel.set_view_limits(0.0, 8.0, *window._plot_panel.get_view_limits()[2:])
        _process_events_for(milliseconds=60)
        _frame_y_to_window(window, 0.0, 8.0)
        _process_events_for(milliseconds=80)
        return window


# --------------------------------------------------------------------------- #
#  2. The FµF model fit — r_µF from FmuF_Linear × Gaussian + Constant.
# --------------------------------------------------------------------------- #
class PtfeFmuFFitScenario(CorpusScenario):
    name = "corpus_ptfe_fmuf_fit"
    description = (
        "Converged FmuF_Linear × Gaussian + Constant fit on the PTFE 20 K ZF "
        "run 17294: r_µF ≈ 1.30 Å (χ²ᵣ ≈ 1.4), the collinear F–µ–F distance "
        "from the dipolar oscillation."
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
        window.resizeDocks([window._dock_data_browser], [320], Qt.Orientation.Horizontal)

        datasets = load_corpus_datasets([_rel(_ZF_BASE_RUN)])
        self.add_to_browser(window, datasets)
        window._on_dataset_selected(datasets[0].run_number)
        _process_events_for(milliseconds=80)

        single_tab = window._fit_panel._single_tab
        single_tab._set_composite_model(CompositeModel(*_FMUF_MODEL))
        _process_events_for(milliseconds=80)

        # Seed r_µF at the literature guidance value (1.15 Å); the ZF FµF beat
        # is over-damped by the neighbouring fluorines, so the fit walks it out
        # to ≈ 1.30 Å (matching the reference-program result, GROUND_TRUTH §7).
        # Seed the amplitude near the ~15 % t=0 asymmetry and the Gaussian width
        # near the ~0.4 µs⁻¹ envelope the beat decay implies.
        table = single_tab._param_table
        rows = _param_table_rows_by_name(table)
        seeds = {"A_1": 14.0, "r_muF": 1.15, "sigma": 0.35, "A_bg": 0.5}
        for name, value in seeds.items():
            if name in rows:
                _set_param_table_value(table, rows[name], value)
        # Pin the amplitude and Gaussian width positive so the fit does not
        # settle in the sign-degenerate mirror minimum.
        for name in ("A_1", "sigma"):
            if name in rows:
                item = table.item(rows[name], table.COL_MIN)
                if item is not None:
                    item.setText("0.0")
        _process_events_for(milliseconds=60)

        single_tab._run_fit()
        single_tab.wait_for_fit()
        _process_events_for(milliseconds=80)

        # Zoom to the first ~10 µs, where the resolved beats and the converged
        # fit overlay both read, before the late-time ZF error fan swamps them.
        _x0, _x1, y0, y1 = window._plot_panel.get_view_limits()
        window._plot_panel.set_view_limits(0.0, 10.0, y0, y1)
        _process_events_for(milliseconds=60)
        _frame_y_to_window(window, 0.0, 10.0)
        _process_events_for(milliseconds=80)
        return window


# --------------------------------------------------------------------------- #
#  3. FFT — the sub-MHz FµF line cluster.
# --------------------------------------------------------------------------- #
class PtfeFftScenario(CorpusScenario):
    name = "corpus_ptfe_fft"
    description = (
        "Frequency-domain view of the PTFE 20 K ZF run: the sub-MHz F–µ–F line "
        "cluster (broad combination lines near ~0.3–0.5 MHz) that encodes the "
        "dipolar coupling frequency."
    )
    example = EXAMPLE
    size = (1500, 920)

    def build(self) -> QWidget:
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window._on_fourier()
        window.resizeDocks([window._dock_data_browser], [320], Qt.Orientation.Horizontal)

        datasets = load_corpus_datasets([_rel(_ZF_BASE_RUN)])
        self.add_to_browser(window, datasets)
        window._on_dataset_selected(datasets[0].run_number)
        _process_events_for(milliseconds=120)

        window._on_domain_button_clicked("frequency")
        _process_events_for(milliseconds=80)

        # The FµF combination lines sit sub-MHz on top of the decaying-envelope
        # DC skirt. A Lorentzian apodisation (τ ≈ 4 µs, matched to the beat's
        # coherence time) trims the record's noisy tail so the cluster near
        # ~0.3–0.5 MHz stands out from the DC peak.
        fp = window._fourier_panel
        fp._filter_lorentzian_radio.setChecked(True)
        fp._filter_time_constant_edit.setText("4.0")
        _process_events_for(milliseconds=40)

        freq_panel = window._frequency_plot_panel
        x_min, x_max = 0.0, 1.2
        window._on_compute_fourier()

        spectrum_x = spectrum_y = None
        for _ in range(100):  # bounded ~10 s
            _process_events_for(milliseconds=100)
            x = freq_panel._last_plot_time
            y = freq_panel._last_plot_asymmetry
            if x is not None and y is not None and len(x) and float(np.nanmax(x)) >= x_max:
                spectrum_x = np.asarray(x, dtype=float)
                spectrum_y = np.asarray(y, dtype=float)
                break
        if spectrum_x is None:
            raise RuntimeError("PTFE Fourier recompute did not render within 10 s")

        # Frame Y to the FµF cluster, not the DC spike, so the sub-MHz line
        # structure is legible: peak of the spectrum away from the 0 MHz bin.
        cluster = (spectrum_x >= 0.15) & (spectrum_x <= x_max)
        peak = float(np.max(spectrum_y[cluster])) if np.any(cluster) else 1.0
        freq_panel.set_view_limits(x_min, x_max, -0.04 * peak, 1.25 * peak)
        _process_events_for(milliseconds=120)
        return window

    def settle(self, widget: QWidget) -> None:
        # Show the Fourier inspector tab (apodisation / padding / units — the
        # transform's own controls, which the caption discusses) rather than
        # the Fit tab, whose carried-over Gaussian model is irrelevant here.
        _process_events_for(milliseconds=100)
        _raise_inspector_tab(widget, widget._dock_fourier.windowTitle())
        super().settle(widget)


# --------------------------------------------------------------------------- #
#  4. TF 20 G calibration run — the prescribed detector-balance step.
# --------------------------------------------------------------------------- #
class PtfeTfCalibrationScenario(CorpusScenario):
    name = "corpus_ptfe_tf_calibration"
    description = (
        "The prescribed calibration step (GROUND_TRUTH §4): the TF 20 G run "
        "17293, showing the slow ~0.3 MHz transverse precession used to fix the "
        "detector balance before the zero-field FµF analysis."
    )
    example = EXAMPLE
    size = (1500, 920)

    def build(self) -> QWidget:
        from asymmetry.gui.mainwindow import MainWindow

        window = MainWindow()
        window.resizeDocks([window._dock_data_browser], [320], Qt.Orientation.Horizontal)

        datasets = load_corpus_datasets([_rel(_TF_CAL_RUN)])
        self.add_to_browser(window, datasets)
        window._on_dataset_selected(datasets[0].run_number)
        _process_events_for(milliseconds=120)

        # 20 G ⇒ γ_µ·B ≈ 0.27 MHz (~3.7 µs period); frame the first ~12 µs so
        # several precession cycles are legible.
        window._plot_panel.set_view_limits(0.0, 12.0, *window._plot_panel.get_view_limits()[2:])
        _process_events_for(milliseconds=60)
        _frame_y_to_window(window, 0.0, 12.0)
        _process_events_for(milliseconds=80)
        return window


register(PtfeZfSignatureScenario())
register(PtfeFmuFFitScenario())
register(PtfeFftScenario())
register(PtfeTfCalibrationScenario())
