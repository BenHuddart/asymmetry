"""Tests for the 3-point parabolic peak readout."""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.transform.peakfit import parabolic_peak


def test_parabolic_peak_symmetric_vertex() -> None:
    # y = 1 - x²  sampled at -1, 0, 1 → vertex at (0, 1).
    peak = parabolic_peak([-1.0, 0.0, 1.0], [0.0, 1.0, 0.0])
    assert peak is not None
    x_pk, y_pk = peak
    assert x_pk == pytest.approx(0.0)
    assert y_pk == pytest.approx(1.0)


def test_parabolic_peak_sub_bin_offset() -> None:
    # A downward parabola whose true peak sits between the centre and a bin.
    def f(x: float) -> float:
        return -2.0 * (x - 0.3) ** 2 + 5.0

    xs = np.array([0.0, 1.0, 2.0])
    peak = parabolic_peak(xs, [f(v) for v in xs])
    assert peak is not None
    x_pk, y_pk = peak
    assert x_pk == pytest.approx(0.3, abs=1e-9)
    assert y_pk == pytest.approx(5.0, abs=1e-9)


def test_parabolic_peak_rejects_upward_parabola() -> None:
    # y = x² opens upward → not a maximum → None.
    assert parabolic_peak([-1.0, 0.0, 1.0], [1.0, 0.0, 1.0]) is None


def test_parabolic_peak_rejects_monotonic_run() -> None:
    # Strictly increasing samples: the parabola's vertex (if any) lies outside
    # the span, so no in-span maximum is reported.
    assert parabolic_peak([0.0, 1.0, 2.0], [1.0, 2.0, 3.0]) is None


def test_parabolic_peak_rejects_degenerate_input() -> None:
    assert parabolic_peak([0.0, 0.0, 1.0], [1.0, 2.0, 3.0]) is None
    assert parabolic_peak([0.0, 1.0], [1.0, 2.0]) is None
