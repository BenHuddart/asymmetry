"""RED target for branch ``fix/fft-render``.

Round-2 GUI finding (CdS, ``_findings/windows-gui/CdS_MaxEnt.md``): the FFT of an
ISIS EMU TF run produces a *corrupt frequency axis* — it auto-ranged to
-18887 .. 396639 MHz, although the Nyquist limit for ~16 ns EMU bins is ~31 MHz.
(Separately, the GUI never draws the spectrum curve for EMU *and* MUSR; that
render-path failure is exercised by ``test_fft_render_gui`` below.)

These are **corpus-conditional**: they skip cleanly when the WiMDA muon-school
corpus is absent (always in CI). Run locally to see them RED on the EMU run.

When fixing: the axis test pins the bug location. If it is already GREEN at the
core ``fft_complex_asymmetry`` level, the corruption lives purely in the GUI
Fourier path (``fourier_panel`` / ``plot_panel``) — note that in the branch and
keep the GUI render test as the acceptance gate.
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


_EMU = _emu_run()


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
    spectrum = compute_average_group_spectrum(
        run, GroupSpectrumConfig(t_min_us=0.1, t_max_us=12.0)
    )
    assert spectrum is not None, "grouped spectrum returned None for EMU run"

    freqs = np.asarray(spectrum.time, dtype=float)
    assert np.isfinite(freqs).all()
    assert float(np.max(np.abs(freqs))) < 100.0, (
        f"EMU grouped-FFT axis spans to {np.max(np.abs(freqs)):.0f} MHz "
        "(expected <= ~31 MHz Nyquist)."
    )
    peak_freq = float(freqs[int(np.argmax(np.abs(np.asarray(spectrum.asymmetry))))])
    assert 1.0 < peak_freq < 2.0, f"peak at {peak_freq:.3f} MHz, expected ~1.36"


@pytest.mark.skip(
    reason=(
        "fix/fft-render: the two tests above PASS, proving both core FFT paths "
        "(fft_complex_asymmetry AND compute_average_group_spectrum) are correct "
        "for EMU. So the Round-2 failures — the empty FFT plot (EMU + MUSR) and "
        "the ~396639 MHz axis (EMU) — live ENTIRELY in the Qt display layer "
        "(fourier_panel / plot_panel / mainwindow FFT worker), not in core. "
        "Acceptance: add an offscreen GUI regression that drives the panel's "
        "FFT compute+plot (mirror tests/test_maxent_corpus_smoke.py, which uses "
        "PlotPanel) and asserts (a) a non-empty spectrum curve is drawn and "
        "(b) the displayed frequency axis is <= ~31 MHz Nyquist. Wire it to the "
        "real render entry point as part of the fix, then un-skip."
    )
)
def test_fft_spectrum_renders_in_gui() -> None:  # pragma: no cover - placeholder
    raise AssertionError("wire to the GUI FFT render path during the fix")
