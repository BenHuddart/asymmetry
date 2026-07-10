"""Tests for the pure waterfall auto-spacing helper (no Qt needed)."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

from asymmetry.gui.utils.waterfall import auto_waterfall_delta


def test_uniform_traces_use_span_times_factor() -> None:
    # Two identical traces spanning ~1.0 (robust 2nd..98th ~ 0.96): Δ = 1.4 * span.
    y = np.linspace(0.0, 1.0, 101)
    delta = auto_waterfall_delta([y, y.copy()])
    span = float(np.percentile(y, 98) - np.percentile(y, 2))
    assert delta == pytest.approx(1.4 * span)


def test_median_over_traces_not_max() -> None:
    small = np.linspace(0.0, 1.0, 101)
    large = np.linspace(0.0, 10.0, 101)
    # Three traces: spans ~1, ~1, ~10 -> median span ~1, so Δ tracks the median.
    delta = auto_waterfall_delta([small, small.copy(), large])
    span_small = float(np.percentile(small, 98) - np.percentile(small, 2))
    assert delta == pytest.approx(1.4 * span_small)


def test_nans_are_ignored() -> None:
    y = np.linspace(0.0, 1.0, 101)
    dirty = y.copy()
    dirty[::7] = np.nan
    clean_delta = auto_waterfall_delta([y])
    dirty_delta = auto_waterfall_delta([dirty])
    # Dropping NaNs leaves the span essentially unchanged.
    assert dirty_delta == pytest.approx(clean_delta, rel=0.05)


def test_empty_and_flat_traces_fall_back_to_unit_delta() -> None:
    assert auto_waterfall_delta([]) == pytest.approx(1.0)
    assert auto_waterfall_delta([np.array([])]) == pytest.approx(1.0)
    flat = np.full(50, 3.0)
    assert auto_waterfall_delta([flat, flat.copy()]) == pytest.approx(1.0)


def test_median_zero_falls_back_to_max_span() -> None:
    # Majority flat (span 0) so the median span is 0; the one real span drives Δ.
    flat = np.full(50, 2.0)
    active = np.linspace(0.0, 4.0, 101)
    delta = auto_waterfall_delta([flat, flat.copy(), active])
    span_active = float(np.percentile(active, 98) - np.percentile(active, 2))
    assert delta == pytest.approx(1.4 * span_active)


def test_delta_is_always_positive() -> None:
    for traces in ([], [np.array([])], [np.full(10, 5.0)], [np.linspace(-3, 3, 20)]):
        assert auto_waterfall_delta(traces) > 0.0


def _fft_like_trace(x: np.ndarray, peak: float) -> np.ndarray:
    # Dominant narrow region near x=1 over a tiny flat tail — the FFT-magnitude
    # shape whose full-array percentiles land in the tail.
    return peak * np.exp(-(((x - 1.0) / 0.3) ** 2)) + 1.0e-4 * peak


def test_windowed_span_ignores_out_of_window_tail() -> None:
    # A huge in-window region plus a long near-zero tail: Δ must track the
    # median IN-WINDOW robust span (× the 1.4 factor), not the deflated
    # full-array span.
    x = np.linspace(0.0, 50.0, 2000)
    traces = [_fft_like_trace(x, peak) for peak in (1.0e7, 1.2e7, 0.9e7)]
    window = (0.0, 3.0)

    delta = auto_waterfall_delta(traces, x_arrays=[x, x, x], x_window=window)

    in_window = (x >= window[0]) & (x <= window[1])
    spans = [
        float(np.percentile(t[in_window], 98) - np.percentile(t[in_window], 2)) for t in traces
    ]
    median_span = float(np.median(spans))
    assert delta == pytest.approx(1.4 * median_span)
    # The full-array spacing (the old behavior) is several times smaller.
    assert delta > 3.0 * auto_waterfall_delta(traces)


def test_window_with_too_few_samples_falls_back_to_full_span() -> None:
    x = np.linspace(0.0, 50.0, 100)
    y = np.linspace(0.0, 1.0, 100)
    # A sliver window holding ~2 samples: the trace uses its full-array span.
    delta = auto_waterfall_delta([y], x_arrays=[x], x_window=(0.0, 0.6))
    assert delta == pytest.approx(auto_waterfall_delta([y]))


def test_window_without_x_arrays_uses_full_span() -> None:
    y = np.linspace(0.0, 1.0, 100)
    delta = auto_waterfall_delta([y], x_window=(0.0, 0.5))
    assert delta == pytest.approx(auto_waterfall_delta([y]))


def test_mismatched_x_shape_falls_back_to_full_span() -> None:
    y = np.linspace(0.0, 1.0, 100)
    delta = auto_waterfall_delta([y], x_arrays=[np.linspace(0, 1, 7)], x_window=(0.0, 0.5))
    assert delta == pytest.approx(auto_waterfall_delta([y]))
