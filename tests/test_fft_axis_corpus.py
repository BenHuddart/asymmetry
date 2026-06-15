"""Regression guards for the ``fix/fft-render`` branch.

Round-2 GUI finding (CdS, ``_findings/windows-gui/CdS_MaxEnt.md``): the FFT of an
ISIS NeXus run rendered as an *empty plot* for EMU *and* MUSR, and the EMU axis
was reported auto-ranging to -18887 .. 396639 MHz (Nyquist for ~16 ns EMU bins is
~31 MHz).

Diagnosis on this branch:

* **Empty plot (root cause, fixed).** Both core FFT paths
  (``fft_complex_asymmetry`` and ``compute_average_group_spectrum``) are correct
  *when given a good-statistics window* — the two ``..._axis_is_physical`` guards
  below pin that. The GUI, however, inherits the time-domain fit range, which
  defaults to the full ~32 µs span; transforming the ±100 %-saturated late-time
  tail buries the physical line under low-frequency leakage, so the dominant
  spectral feature collapses onto DC and the plot reads as empty. The fix excludes
  the statistically-spent tail in the ``MainWindow`` FFT worker
  (``_fourier_time_window_excluding_tail``); ``test_fft_spectrum_renders_in_gui``
  drives that real render path and is the acceptance gate.
* **EMU axis corruption (separate, not reproduced).** Capping the time window
  changes the sample count, not the sample spacing, so it cannot move the
  ``rfftfreq`` Nyquist limit — and the ~396639 MHz axis does **not** reproduce on
  this build through any headless path (every path yields <= ~31 MHz). The render
  test keeps a Nyquist axis assertion as a cheap guard, but the reported axis
  blow-up remains an open, separately-tracked item pending a live-GUI repro.

All tests here are **corpus-conditional**: they skip cleanly when the WiMDA
muon-school corpus is absent (always in CI). Run locally to exercise them.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from asymmetry.core.fourier.fft import fft_complex_asymmetry
from asymmetry.core.fourier.spectrum import (
    GroupSpectrumConfig,
    compute_average_group_spectrum,
)


def _corpus_root() -> Path | None:
    candidates = [
        os.environ.get("WIMDA_CORPUS_ROOT"),
        r"C:\Users\benhu\Source\wimda-corpus",
        str(Path.home() / "Documents" / "WiMDA muon school"),
    ]
    for cand in candidates:
        if cand and Path(cand).exists():
            return Path(cand)
    return None


def _emu_run() -> Path | None:
    root = _corpus_root()
    if root is None:
        return None
    matches = sorted(root.rglob("EMU00020711.nxs")) or sorted(
        (root / "Semiconductors").rglob("EMU*.nxs")
    )
    # The legacy ``Data/`` copies are HDF4 (not loadable); prefer the HDF5 v2
    # copies under ``data_hdf5/`` (case-insensitive).
    hdf5 = [m for m in matches if "hdf5" in str(m).lower()]
    matches = hdf5 or matches
    return matches[0] if matches else None


def _musr_run() -> Path | None:
    """A TF MUSR (BiSCCO, 400 G ⇒ ~5.4 MHz) run — the second ISIS NeXus case.

    Round-2 saw the same empty FFT plot for MUSR as for EMU, so the GUI render
    regression covers both instruments.
    """
    root = _corpus_root()
    if root is None:
        return None
    matches = sorted(root.rglob("MUSR00001277.nxs"))
    hdf5 = [m for m in matches if "hdf5" in str(m).lower()]
    matches = hdf5 or matches
    return matches[0] if matches else None


_EMU = _emu_run()
_MUSR = _musr_run()


@pytest.mark.skipif(_EMU is None, reason="WiMDA corpus EMU run not present")
def test_emu_fft_complex_axis_is_physical() -> None:
    """Characterisation: the single-dataset primitive is correct for EMU.

    This currently PASSES — it pins the bug *out* of ``core/fourier/fft.py`` so the
    fix targets the GUI grouped-spectrum path, not this primitive.
    """
    from asymmetry.core.io import load

    dataset = load(str(_EMU))
    freqs, spectrum = fft_complex_asymmetry(dataset, t_min=0.1, t_max=12.0)

    assert np.isfinite(freqs).all()
    assert float(np.max(np.abs(freqs))) < 100.0
    peak_freq = float(freqs[int(np.argmax(np.abs(spectrum)))])
    assert 1.0 < peak_freq < 2.0, f"peak at {peak_freq:.3f} MHz, expected ~1.36"


@pytest.mark.skipif(_EMU is None, reason="WiMDA corpus EMU run not present")
def test_emu_grouped_spectrum_axis_is_physical() -> None:
    """The GUI's averaged grouped-FFT path must produce a physical axis for EMU.

    This is the path the GUI Fourier panel drives (``compute_average_group_spectrum``),
    where the Round-2 finding saw the axis blow up to ~396639 MHz. RED if the
    corruption lives here; if GREEN, the corruption is purely in the Qt display
    layer (``fourier_panel`` / ``plot_panel``) — record that and gate on the GUI
    render test instead.
    """
    from asymmetry.core.io import load

    dataset = load(str(_EMU))
    run = getattr(dataset, "run", None) or dataset
    spectrum = compute_average_group_spectrum(run, GroupSpectrumConfig(t_min_us=0.1, t_max_us=12.0))
    assert spectrum is not None, "grouped spectrum returned None for EMU run"

    freqs = np.asarray(spectrum.time, dtype=float)
    assert np.isfinite(freqs).all()
    assert float(np.max(np.abs(freqs))) < 100.0, (
        f"EMU grouped-FFT axis spans to {np.max(np.abs(freqs)):.0f} MHz "
        "(expected <= ~31 MHz Nyquist)."
    )
    peak_freq = float(freqs[int(np.argmax(np.abs(np.asarray(spectrum.asymmetry))))])
    assert 1.0 < peak_freq < 2.0, f"peak at {peak_freq:.3f} MHz, expected ~1.36"


#: ISIS NeXus runs whose Round-2 FFT rendered empty, with the physical line each
#: should resolve (γµ·B): EMU 100 G ⇒ 1.36 MHz, MUSR (BiSCCO) 400 G ⇒ ~5.42 MHz.
_RENDER_CASES = [
    pytest.param(_EMU, 1.36, id="emu-100G"),
    pytest.param(_MUSR, 5.42, id="musr-biscco-400G"),
]


@pytest.mark.gui
@pytest.mark.parametrize("run_path, expected_mhz", _RENDER_CASES)
def test_fft_spectrum_renders_in_gui(run_path: Path | None, expected_mhz: float) -> None:
    """The GUI FFT must draw the physical line for ISIS NeXus data (EMU + MUSR).

    Round-2 acceptance gate. Drives the *real* render entry point — the
    :class:`MainWindow` FFT worker (:meth:`_on_compute_fourier`) — on a corpus
    run whose time-domain fit range defaults to the full ~32 µs span. Before the
    fix that full window transforms the ±100 %-saturated tail, so the spectrum's
    dominant feature collapses onto DC and the plot reads as empty; the worker
    now excludes the spent tail, so the physical line resolves and renders.

    Asserts (a) a non-empty spectrum curve is drawn, (b) the displayed frequency
    axis stays within the ~31 MHz Nyquist limit, and (c) the rendered spectrum's
    dominant non-DC peak is the physical line.
    """
    if run_path is None:
        pytest.skip("WiMDA corpus run not present")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from asymmetry.core.io import load
    from asymmetry.gui.mainwindow import MainWindow
    from tests._qt_helpers import wait_for

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        dataset = load(str(run_path))
        window._data_browser.add_dataset(dataset)
        window._current_dataset = dataset
        # Plotting on the time panel seeds the full-range fit window the FFT
        # inherits — i.e. the exact condition that produced the empty plot.
        window._plot_panel.plot_dataset(dataset)

        window._on_compute_fourier()
        wait_for(lambda: not window._fourier_compute_active, app, timeout_s=30.0)

        panel = window._frequency_plot_panel
        if not getattr(panel, "_has_mpl", False):
            pytest.skip("matplotlib not available")

        # Guard against a validation short-circuit (no groups, etc.): without this
        # the compute would never start and ``wait_for`` would return instantly,
        # so the render asserts below would fail as if the curve had not drawn.
        run_number = int(dataset.run_number)
        assert window._frequency_spectra_by_run.get(run_number), (
            "FFT compute produced no spectrum for the run (validation short-circuit?)"
        )
        assert panel._last_plot_asymmetry is not None, "the FFT spectrum never rendered"

        # (a) a non-empty spectrum curve is drawn.
        assert panel._ax.collections, "no spectrum curve was drawn on the FFT canvas"
        freqs = np.asarray(panel._last_plot_time, dtype=float)
        values = np.abs(np.asarray(panel._last_plot_asymmetry, dtype=float))
        assert freqs.size > 1 and np.isfinite(freqs).all()
        assert np.any(values > 0.0), "the plotted spectrum is entirely flat"

        # (b) the displayed frequency axis is within the ~31 MHz Nyquist limit
        # (would catch the Round-2 ~396639 MHz EMU axis corruption).
        assert float(np.max(np.abs(freqs))) < 100.0, (
            f"FFT axis spans to {np.max(np.abs(freqs)):.0f} MHz (expected <= ~31 MHz Nyquist)"
        )
        x_lo, x_hi = panel._ax.get_xlim()
        assert max(abs(x_lo), abs(x_hi)) < 100.0

        # (c) the dominant non-DC feature is the physical line — the empty-plot
        # gate. Before the fix this peak collapses onto DC (~0.06 MHz).
        non_dc = values.copy()
        non_dc[freqs < 0.05] = 0.0
        peak_freq = float(freqs[int(np.argmax(non_dc))])
        assert abs(peak_freq - expected_mhz) < 0.3, (
            f"dominant peak at {peak_freq:.3f} MHz, expected the line near {expected_mhz} MHz"
        )
    finally:
        window.close()
        window.deleteLater()


@pytest.mark.gui
def test_fourier_tail_exclusion_heuristic() -> None:
    """Unit-cover the saturated-tail window cap (runs in CI, no corpus needed).

    Exercises the worker helpers directly with synthetic count profiles: an
    exponential pulsed-source decay is capped to ≈4.6 lifetimes, a tighter user
    window is honoured, a late-starting window is never inverted, and a
    continuous (flat) source or a histogram-less run is left untouched.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from types import SimpleNamespace

    from PySide6.QtWidgets import QApplication

    from asymmetry.gui.mainwindow import MainWindow

    def _dataset(counts: np.ndarray, time_axis: np.ndarray) -> object:
        hist = SimpleNamespace(counts=counts, time_axis=time_axis)
        return SimpleNamespace(run=SimpleNamespace(histograms=[hist]))

    if QApplication.instance() is None:
        QApplication([])
    window = MainWindow()
    try:
        time_axis = np.linspace(0.0, 32.0, 2000)
        decay = _dataset(1.0e5 * np.exp(-time_axis / 2.197), time_axis)

        tail = window._fourier_good_statistics_t_max(decay)
        assert tail is not None and 9.0 < tail < 12.0  # ≈4.6 muon lifetimes

        # Full inherited window is capped to the tail; a tighter user window wins.
        assert window._fourier_time_window_excluding_tail(decay, 0.1, 32.0) == (0.1, tail)
        assert window._fourier_time_window_excluding_tail(decay, 0.1, 6.0) == (0.1, 6.0)
        # A window starting past the tail must not be inverted/emptied.
        assert window._fourier_time_window_excluding_tail(decay, 20.0, 30.0) == (20.0, 30.0)

        # Continuous source (flat counts never reach 1 %) and a histogram-less run
        # are both left untouched.
        flat = _dataset(np.full(2000, 5.0e4), time_axis)
        assert window._fourier_good_statistics_t_max(flat) is None
        assert window._fourier_time_window_excluding_tail(flat, 0.1, 32.0) == (0.1, 32.0)
        empty = SimpleNamespace(run=SimpleNamespace(histograms=[]))
        assert window._fourier_time_window_excluding_tail(empty, 0.1, 32.0) == (0.1, 32.0)
    finally:
        window.close()
        window.deleteLater()
