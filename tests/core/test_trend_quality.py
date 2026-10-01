"""Trend quality of a local parameter's trace (plan D6) and the pass diagnostic (D15)."""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.fitting.trend_quality import (
    PASS_AXIS_ZIGZAG_MIN,
    PASS_RUN_ZIGZAG_MAX,
    CandidateTrend,
    PassDiagnostic,
    TracePoint,
    TraceQuality,
    pass_diagnostic,
    trace_quality,
)


def _trace(values, errors, x=None) -> list[TracePoint]:
    """Points at ``x`` (default 1, 2, 3…) with run numbers 1, 2, 3… in the order given."""
    positions = range(1, len(values) + 1) if x is None else x
    errors = [errors] * len(values) if isinstance(errors, float) else errors
    return [
        TracePoint(x=float(position), run=run, value=float(value), error=error)
        for run, (position, value, error) in enumerate(
            zip(positions, values, errors, strict=True), start=1
        )
    ]


def _noise(n: int, sigma: float) -> np.ndarray:
    return np.random.default_rng(20261001).normal(0.0, sigma, n)


def test_steep_smooth_rise_over_three_decades_is_a_good_trend() -> None:
    # The case that ruled out interpolation-residual measures: a hop rate
    # rising monotonically by 10³ is as smooth as a trace gets.
    values = np.logspace(-2, 1, 16)
    quality = trace_quality(_trace(values, list(0.01 * values)))

    assert quality.zigzag == 0.0
    assert quality.determined == 1.0
    assert quality.signal > 0.99
    assert quality.quality > 0.99


def test_single_sharp_peak_costs_one_point() -> None:
    values = [1.0, 1.1, 1.2, 1.3, 9.0, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1]
    quality = trace_quality(_trace(values, 0.05))

    # Only the peak itself lies outside its neighbours' interval.
    assert quality.zigzag == pytest.approx(1 / 10)


def test_step_is_not_a_zigzag() -> None:
    values = [5.0] * 6 + [1.0] * 6
    quality = trace_quality(_trace(values, 0.05))

    assert quality.zigzag == 0.0
    assert quality.quality > 0.9


def test_noise_within_errors_is_smooth_but_carries_no_signal() -> None:
    values = 2.0 + _noise(40, 0.02)
    quality = trace_quality(_trace(values, 0.1))

    assert quality.zigzag == 0.0
    assert quality.determined == 1.0
    assert quality.signal < 0.2
    assert quality.quality < 0.2


def test_noise_well_beyond_errors_zigzags() -> None:
    values = 2.0 + _noise(40, 1.0)
    quality = trace_quality(_trace(values, 0.01))

    # An interior point of white noise is a local extremum two times in three.
    assert quality.zigzag > 0.5
    assert quality.quality < 0.5


def test_order_parameter_going_to_zero_stays_determined() -> None:
    values = [5.0, 4.9, 4.7, 4.3, 3.6, 2.4, 0.9, 0.1, 0.05, 0.0, 0.02]
    points = _trace(values, 0.2)
    # Above the transition the error exceeds the value itself …
    assert all(point.error > abs(point.value) for point in points[-4:])

    quality = trace_quality(points)

    # … but it is small against the span, so those points are well determined.
    assert quality.determined == 1.0
    assert quality.zigzag == 0.0
    assert quality.quality > 0.8


def test_small_values_with_large_errors_and_no_span_are_undetermined() -> None:
    quality = trace_quality(_trace([0.1, 0.12, 0.09, 0.11, 0.1], 0.3))

    assert quality.determined == 0.0
    assert quality.quality == 0.0


def test_flat_parameter_within_errors_scores_low_through_signal() -> None:
    quality = trace_quality(_trace([3.0, 3.02, 2.99, 3.01, 3.0, 2.98, 3.01], 0.05))

    assert quality.determined == 1.0
    assert quality.zigzag == 0.0
    assert quality.quality == quality.signal < 0.2


def test_run_without_uncertainty_is_undetermined_and_outside_the_trace() -> None:
    values = [1.0, 2.0, 50.0, 4.0, 5.0, 6.0]
    errors = [0.05, 0.05, None, 0.05, 0.05, 0.05]
    quality = trace_quality(_trace(values, errors))
    without_it = trace_quality(_trace([1.0, 2.0, 4.0, 5.0, 6.0], 0.05, x=[1, 2, 4, 5, 6]))

    assert quality.determined == pytest.approx(5 / 6)
    # The wild value has no uncertainty: it is no extremum, it does not make
    # run 4 one, and it does not widen the span.
    assert quality.zigzag == without_it.zigzag == 0.0
    assert quality.signal == without_it.signal


def test_trace_with_no_uncertainties_has_no_quality() -> None:
    quality = trace_quality(_trace([1.0, 2.0, 3.0], [None, None, None]))

    assert quality == TraceQuality(determined=0.0, signal=0.0, zigzag=0.0)


def test_tied_axis_values_are_ordered_by_run_number() -> None:
    # Two runs at x = 2: run 2 then run 5. In (x, run) order the values read
    # 1, 2, 2.1, 3, 4 — smooth — whatever order the points arrive in.
    points = [
        TracePoint(x=1.0, run=1, value=1.0, error=0.01),
        TracePoint(x=2.0, run=2, value=2.0, error=0.01),
        TracePoint(x=3.0, run=3, value=3.0, error=0.01),
        TracePoint(x=4.0, run=4, value=4.0, error=0.01),
        TracePoint(x=2.0, run=5, value=2.1, error=0.01),
    ]

    quality = trace_quality(points)

    assert quality.zigzag == 0.0
    assert trace_quality(points[::-1]) == quality
    # The reverse tie order would make both tied runs extrema; run order rules it out.
    swapped = [
        TracePoint(
            x=point.x,
            run={2: 5, 5: 2}.get(point.run, point.run),
            value=point.value,
            error=point.error,
        )
        for point in points
    ]
    assert trace_quality(swapped).zigzag == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    ("values", "determined", "signal"),
    [
        # One run: no span, so no signal.
        ([4.0], 1.0, 0.0),
        # Two runs: span is the 10–90 % range, 0.8 of the difference; snr = 8.
        ([4.0, 5.0], 1.0, 8.0 / 11.0),
    ],
)
def test_fewer_than_three_points_have_no_zigzag(values, determined, signal) -> None:
    quality = trace_quality(_trace(values, 0.1))

    assert quality.zigzag == 0.0
    assert quality.determined == determined
    assert quality.signal == pytest.approx(signal)


def test_empty_trace_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one run"):
        trace_quality([])


@pytest.mark.parametrize("error", [0.0, -0.1, float("nan"), float("inf")])
def test_trace_point_rejects_an_uncertainty_that_is_not_positive_and_finite(error) -> None:
    with pytest.raises(ValueError, match="uncertainty"):
        TracePoint(x=1.0, run=1, value=1.0, error=error)


def test_quality_is_the_product_of_its_factors() -> None:
    assert TraceQuality(determined=0.8, signal=0.5, zigzag=0.25).quality == pytest.approx(0.3)


def test_candidate_orders_by_worst_then_mean_quality() -> None:
    good = TraceQuality(determined=1.0, signal=0.9, zigzag=0.0)
    fair = TraceQuality(determined=1.0, signal=0.6, zigzag=0.0)
    poor = TraceQuality(determined=1.0, signal=0.2, zigzag=0.0)

    even = CandidateTrend({"Lambda": fair, "beta": fair})
    lopsided = CandidateTrend({"Lambda": good, "beta": poor})
    better_mean = CandidateTrend({"Lambda": good, "beta": fair})

    assert even.ordering_key == (True, pytest.approx(0.6), pytest.approx(0.6))
    assert sorted([lopsided, even, better_mean], key=lambda c: c.ordering_key) == [
        lopsided,
        even,
        better_mean,
    ]


def test_candidate_with_no_local_parameters_is_not_a_trend() -> None:
    all_shared = CandidateTrend({})
    poor = CandidateTrend({"Lambda": TraceQuality(determined=0.1, signal=0.1, zigzag=0.9)})

    assert not all_shared.is_trend
    assert poor.is_trend
    # Any trend, however poor, ranks above a candidate in which nothing varies.
    assert all_shared.ordering_key < poor.ordering_key


def _two_pass_trace(offset: float) -> list[TracePoint]:
    """A coarse pass at x = 10, 20, … then an infill pass at x = 15, 25, …"""
    coarse_x = [10.0 * k for k in range(1, 14)]
    infill_x = [position + 5.0 for position in coarse_x]
    x = coarse_x + infill_x
    values = [0.01 * position for position in coarse_x] + [
        0.01 * position + offset for position in infill_x
    ]
    return _trace(values, 0.005, x=x)


def test_interleaved_passes_offset_from_each_other_disagree() -> None:
    diagnostic = pass_diagnostic(_two_pass_trace(offset=0.3))

    assert diagnostic.zigzag_axis_order >= PASS_AXIS_ZIGZAG_MIN
    # In acquisition order each pass is smooth; only the join between them costs.
    assert diagnostic.zigzag_run_order == pytest.approx(2 / 24)
    assert diagnostic.zigzag_run_order <= PASS_RUN_ZIGZAG_MAX
    assert diagnostic.passes_disagree


def test_interleaved_passes_that_agree_do_not_disagree() -> None:
    diagnostic = pass_diagnostic(_two_pass_trace(offset=0.0))

    assert diagnostic.zigzag_axis_order == 0.0
    assert not diagnostic.passes_disagree


def test_trace_rough_in_both_orders_is_not_a_pass_disagreement() -> None:
    diagnostic = pass_diagnostic(_trace(2.0 + _noise(40, 1.0), 0.01))

    assert diagnostic.zigzag_axis_order >= PASS_AXIS_ZIGZAG_MIN
    assert diagnostic.zigzag_run_order > PASS_RUN_ZIGZAG_MAX
    assert not diagnostic.passes_disagree


def test_pass_diagnostic_thresholds() -> None:
    assert PassDiagnostic(zigzag_axis_order=0.3, zigzag_run_order=0.1).passes_disagree
    assert not PassDiagnostic(zigzag_axis_order=0.29, zigzag_run_order=0.0).passes_disagree
    assert not PassDiagnostic(zigzag_axis_order=0.9, zigzag_run_order=0.11).passes_disagree
