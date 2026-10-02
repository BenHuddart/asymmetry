"""How well a series fit's local parameters trend along the scan axis.

A local parameter is worth having when its values vary by more than their
errors, are pinned by the data in most runs, and do not zigzag from one run to
the next. :func:`trace_quality` scores one parameter's trace on those three
counts; :class:`CandidateTrend` orders candidates by their worst parameter.
The design and the rejected alternatives are in
``docs/plans/global-wizard-trend-objective.md`` (D6, D15, D20).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean, median

import numpy as np

#: A point is undetermined when its error exceeds this share of its own value …
_UNDETERMINED_VALUE_SHARE = 0.5
#: … and this share of the trace's span, so an order parameter at zero still counts.
_UNDETERMINED_SPAN_SHARE = 0.25
#: Signal-to-noise at which the signal factor is one half.
_SIGNAL_HALF_SNR = 3.0
#: Combined standard deviations by which a point must overshoot its neighbours.
_EXTREMUM_SIGMA = 2.0
#: Zigzag in axis order at or above which a trace is called rough.
PASS_AXIS_ZIGZAG_MIN = 0.3
#: Zigzag in run order at or below which the same trace is called smooth.
PASS_RUN_ZIGZAG_MAX = 0.1


@dataclass(frozen=True)
class TracePoint:
    """One run's fitted value of a local parameter.

    ``error`` is the 1σ uncertainty, or ``None`` when the fit reported none
    (the engine gives no entry for a parameter the data do not fix).
    """

    x: float
    run: int
    value: float
    error: float | None

    def __post_init__(self) -> None:
        if not (math.isfinite(self.x) and math.isfinite(self.value)):
            raise ValueError(f"run {self.run}: trace position and value must be finite")
        if self.error is not None and not (math.isfinite(self.error) and self.error > 0.0):
            raise ValueError(
                f"run {self.run}: an uncertainty must be finite and positive, "
                f"or None when the fit gave none (got {self.error!r})"
            )


@dataclass(frozen=True)
class TraceQuality:
    """The three factors of one local parameter's trend quality, each in [0, 1]."""

    #: Share of runs whose value the data pin down.
    determined: float
    #: ``snr / (snr + 3)``, with ``snr`` the trace's span over its median error.
    signal: float
    #: Share of interior points that are a significant local extremum.
    zigzag: float

    @property
    def quality(self) -> float:
        return self.determined * self.signal * (1.0 - self.zigzag)


@dataclass(frozen=True)
class CandidateTrend:
    """The trend qualities of a candidate's local parameters, by parameter name."""

    parameters: Mapping[str, TraceQuality]

    @property
    def is_trend(self) -> bool:
        """False for a candidate with no local parameter: nothing in it varies."""
        return bool(self.parameters)

    @property
    def ordering_key(self) -> tuple[bool, float, float]:
        """Sort key, larger is better: any trend first, then worst and mean quality."""
        qualities = [item.quality for item in self.parameters.values()]
        if not qualities:
            return (False, 0.0, 0.0)
        return (True, min(qualities), fmean(qualities))


@dataclass(frozen=True)
class PassDiagnostic:
    """One trace's zigzag along the scan axis and within each acquisition pass."""

    zigzag_axis_order: float
    #: Extrema over interior points, both counted inside each pass and summed.
    zigzag_run_order: float
    #: Run numbers of each pass, in the order the passes were taken.
    passes: tuple[tuple[int, ...], ...]

    @property
    def passes_disagree(self) -> bool:
        """Rough along the axis, smooth inside each pass: the passes do not agree.

        It takes two passes of three or more runs to say so: a shorter pass has
        no interior point, so nothing shows that it is smooth.
        """
        return (
            sum(len(runs) >= 3 for runs in self.passes) >= 2
            and self.zigzag_axis_order >= PASS_AXIS_ZIGZAG_MIN
            and self.zigzag_run_order <= PASS_RUN_ZIGZAG_MAX
        )


def _axis_order(point: TracePoint) -> tuple[float, int]:
    return (point.x, point.run)


def _run_order(point: TracePoint) -> int:
    return point.run


def _extrema(
    points: Sequence[TracePoint], order: Callable[[TracePoint], object]
) -> tuple[int, int]:
    """Count interior points lying outside their two neighbours' interval by > 2σ.

    Returns ``(extrema, interior points)``. σ combines the point's error with
    that of the neighbour it overshoots. The test compares values only with
    their neighbours' values, so it is scale-free and a monotone trace of any
    steepness has no extremum. Runs without an uncertainty are left out of the
    sequence: their values fix nothing, so they are neither extrema nor
    neighbours.
    """
    ordered = sorted((point for point in points if point.error is not None), key=order)
    extrema = 0
    for before, point, after in zip(ordered, ordered[1:], ordered[2:], strict=False):
        low, high = sorted((before, after), key=lambda neighbour: neighbour.value)
        if point.value < low.value:
            overshot, excess = low, low.value - point.value
        elif point.value > high.value:
            overshot, excess = high, point.value - high.value
        else:
            continue
        if excess > _EXTREMUM_SIGMA * math.hypot(point.error, overshot.error):
            extrema += 1
    return extrema, max(len(ordered) - 2, 0)


def _zigzag(extrema: int, interior: int) -> float:
    """Share of interior points that are extrema; zero when there is no interior point."""
    return extrema / interior if interior else 0.0


def acquisition_passes(points: Sequence[TracePoint]) -> tuple[tuple[TracePoint, ...], ...]:
    """Split a series into passes: maximal stretches, in run order, monotone in the axis.

    A run at the same axis position as the one before it continues the pass, and
    the run that reverses the direction starts the next one.
    """
    passes: list[list[TracePoint]] = []
    direction = 0.0
    for point in sorted(points, key=_run_order):
        step = point.x - passes[-1][-1].x if passes else 0.0
        if passes and step * direction >= 0.0:
            passes[-1].append(point)
            direction = direction or step
        else:
            passes.append([point])
            direction = 0.0
    return tuple(tuple(members) for members in passes)


def trace_quality(points: Sequence[TracePoint]) -> TraceQuality:
    """Score one local parameter's trace (plan D6).

    Points are taken in axis order; runs at the same axis position are ordered
    by run number, so the result does not depend on the order of ``points``.

    A run without an uncertainty counts as undetermined and is left out of the
    span (the 10–90 % range of the values), the median error and the zigzag: a
    value the data do not fix says nothing about how the parameter moves. A
    trace in which no run has an uncertainty therefore has no signal. With fewer
    than three runs that have one there is no interior point and no zigzag.
    """
    if not points:
        raise ValueError("a parameter trace needs at least one run")
    known = [(point.value, point.error) for point in points if point.error is not None]
    if not known:
        return TraceQuality(determined=0.0, signal=0.0, zigzag=0.0)
    low, high = np.percentile([value for value, _error in known], [10.0, 90.0])
    span = float(high - low)
    snr = span / median(error for _value, error in known)
    determined = sum(
        not (
            error > _UNDETERMINED_VALUE_SHARE * abs(value)
            and error > _UNDETERMINED_SPAN_SHARE * span
        )
        for value, error in known
    )
    return TraceQuality(
        determined=determined / len(points),
        signal=snr / (snr + _SIGNAL_HALF_SNR),
        zigzag=_zigzag(*_extrema(points, _axis_order)),
    )


def pass_diagnostic(points: Sequence[TracePoint]) -> PassDiagnostic:
    """Compare one trace's zigzag along the axis and inside each pass (plan D15, D20).

    The run-order zigzag is counted pass by pass, so the join between two
    passes is never an extremum and a short interleaved series can be judged.
    """
    passes = acquisition_passes(points)
    counts = [_extrema(members, _run_order) for members in passes]
    return PassDiagnostic(
        zigzag_axis_order=_zigzag(*_extrema(points, _axis_order)),
        zigzag_run_order=_zigzag(
            sum(extrema for extrema, _interior in counts),
            sum(interior for _extrema_count, interior in counts),
        ),
        passes=tuple(tuple(point.run for point in members) for members in passes),
    )
