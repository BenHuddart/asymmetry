"""Auto-spacing for waterfall overlay plots.

When several traces are overlaid on a single axis in *waterfall* mode, each is
shifted vertically by a uniform per-trace offset ``i * Δ`` so the curves are
cleanly resolved rather than piled on top of one another. :func:`auto_waterfall_delta`
picks that Δ from the data so the default spacing looks reasonable without the
user tuning it: a trace's vertical extent is measured by a robust span (the
98th minus the 2nd percentile of its finite samples, which ignores a handful of
saturation/outlier bins), and Δ is 1.4× the median span across traces so
neighbours clear each other with a little breathing room.

The span is measured over the samples that will actually be *displayed* when
the caller supplies the trace x-axes and the shown x-window: an FFT magnitude
spectrum has a dominant low-frequency region and a long near-zero tail, so
percentiles over the full array would deflate Δ to a small fraction of the
visible spans and the stack would look like a no-op.

This is a pure helper (no Qt, no matplotlib): the panel gathers the display
arrays it is about to draw and asks for the spacing.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

#: Percentiles bounding the robust per-trace span. The 2nd/98th pair drops the
#: few saturated or divergent bins that a plain peak-to-peak would otherwise let
#: dominate the spacing.
_SPAN_LOW_PERCENTILE = 2.0
_SPAN_HIGH_PERCENTILE = 98.0

#: Multiplier applied to the median span so adjacent traces clear each other.
_DELTA_FACTOR = 1.4

#: Spacing used when no trace carries a measurable span (all empty/flat).
_FALLBACK_DELTA = 1.0

#: Minimum number of finite in-window samples for a trace's windowed span to be
#: trusted; below this the trace falls back to its full-array span.
_MIN_WINDOW_SAMPLES = 8


def _robust_span(values: np.ndarray) -> float:
    """Return the 98th − 2nd percentile span of the finite entries, or 0.0."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    lo, hi = np.percentile(finite, [_SPAN_LOW_PERCENTILE, _SPAN_HIGH_PERCENTILE])
    span = float(hi - lo)
    return span if span > 0.0 else 0.0


def _trace_span(
    values: np.ndarray,
    x: np.ndarray | None,
    x_window: tuple[float, float] | None,
) -> float:
    """Robust span of *values*, windowed to ``x ∈ x_window`` when possible.

    Falls back to the full-array span when no window/x-axis is given, the
    shapes disagree, or the window holds fewer than ``_MIN_WINDOW_SAMPLES``
    finite samples (too few points to estimate percentiles meaningfully).
    """
    if x is not None and x_window is not None:
        x = np.asarray(x, dtype=float)
        if x.shape == values.shape:
            lo, hi = min(x_window), max(x_window)
            windowed = values[np.isfinite(x) & (x >= lo) & (x <= hi)]
            if np.count_nonzero(np.isfinite(windowed)) >= _MIN_WINDOW_SAMPLES:
                return _robust_span(windowed)
    return _robust_span(values)


def auto_waterfall_delta(
    traces: Sequence[np.ndarray],
    *,
    x_arrays: Sequence[np.ndarray] | None = None,
    x_window: tuple[float, float] | None = None,
) -> float:
    """Return the automatic per-trace vertical offset Δ for a set of traces.

    Δ = ``1.4 × median`` of the per-trace robust spans (98th − 2nd percentile of
    finite samples). When that median is zero (e.g. most traces are flat or
    empty) it falls back to the maximum robust span, and to ``1.0`` when even
    that is zero, so the spacing is always strictly positive.

    When *x_arrays* (one x-axis per trace, same order) and *x_window* are
    given, each trace's span is measured only over samples whose x lies inside
    the window — the samples the plot will actually show — with a per-trace
    fallback to the full array when fewer than ``_MIN_WINDOW_SAMPLES`` finite
    samples fall inside. Callers resolve Δ once at plot time from the window
    then shown (the first-paint frame, or the manually set limits); a later
    interactive zoom — including the decimation re-render it schedules —
    deliberately does not re-resolve Δ (the plot panel caches the plot-time Δ
    per content identity), so the stack never re-spaces under the user
    mid-inspection.
    """
    if not traces:
        return _FALLBACK_DELTA
    spans = np.array(
        [
            _trace_span(
                np.asarray(trace, dtype=float),
                x_arrays[i] if x_arrays is not None and i < len(x_arrays) else None,
                x_window,
            )
            for i, trace in enumerate(traces)
        ],
        dtype=float,
    )
    median_span = float(np.median(spans))
    if median_span > 0.0:
        return _DELTA_FACTOR * median_span
    max_span = float(np.max(spans))
    if max_span > 0.0:
        return _DELTA_FACTOR * max_span
    return _FALLBACK_DELTA
