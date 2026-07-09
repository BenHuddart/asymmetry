"""Unit tests for the shared preview-plot decimation helper.

Pure-function coverage (no Qt) of the point-count bound shared by the
grouping preview pane (``gui/windows/grouping/preview_pane.py``) and the fit
wizard's fingerprint plot (``gui/windows/fit_wizard_window.py``). Both draw a
matplotlib ``errorbar`` synchronously on the GUI thread over an entire
reduced/raw curve; without a bound, a long high-resolution run freezes the
GUI for seconds building the error-bar collection's data limits.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.gui.utils.plot_decimation import decimate_for_preview

pytestmark = [pytest.mark.gui]


def test_decimate_for_preview_passes_through_small_curves() -> None:
    """A curve at/below the cap is returned unchanged (no copy needed)."""
    n = 2000
    time = np.arange(n, dtype=float)
    y = np.linspace(0.0, 1.0, n)
    yerr = np.full(n, 0.01)
    out_t, out_y, out_e = decimate_for_preview(time, y, yerr, 2000)
    assert out_t is time
    assert out_y is y
    assert out_e is yerr


def test_decimate_for_preview_strides_large_curve_to_cap() -> None:
    """A ~1M-point curve (the pathological case that froze the GUI) is bounded."""
    n = 1_000_000
    time = np.arange(n, dtype=float)
    y = np.sin(time)
    yerr = np.full(n, 0.01)
    out_t, out_y, out_e = decimate_for_preview(time, y, yerr, 2000)
    assert out_t.size <= 2000
    assert out_y.size == out_t.size
    assert out_e.size == out_t.size
    # Uniform stride: the decimated time values stay monotonically increasing
    # and are a subsequence of the original.
    assert np.all(np.diff(out_t) > 0)
    assert np.isin(out_t, time).all()


def test_decimate_for_preview_nonpositive_cap_returns_input_unchanged() -> None:
    time = np.arange(10, dtype=float)
    y = np.zeros(10)
    yerr = np.zeros(10)
    out_t, out_y, out_e = decimate_for_preview(time, y, yerr, 0)
    assert out_t is time
    assert out_y is y
    assert out_e is yerr


def test_decimate_for_preview_handles_empty_arrays() -> None:
    time = np.array([], dtype=float)
    y = np.array([], dtype=float)
    yerr = np.array([], dtype=float)
    out_t, out_y, out_e = decimate_for_preview(time, y, yerr, 2000)
    assert out_t.size == 0
    assert out_y.size == 0
    assert out_e.size == 0
