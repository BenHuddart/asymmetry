"""Robustness heuristics for batch (series) fits.

A near-transition oscillatory scan — the canonical case being an EuO ZF
temperature scan approaching ``T_C`` — is *bistable*: each per-run composite fit
lands either on the real, slowly decreasing precession frequency or on a spurious
high-frequency "noise" branch with the oscillation amplitude collapsed to ~0.
Neither "Auto" nor "Chain from previous run" seeding lands consistently across the
scan, so a user reading the ``ν(T)`` trend sees wild outliers with no signpost to
the cure.

This module is the shared, GUI-free definition of *"this batch went wrong"* and the
recipe that fixes it. It provides:

* :func:`detect_amplitude_collapse` — runs whose fitted amplitude collapsed far
  below the series scale (the spurious-branch signature).
* :func:`detect_frequency_outliers` — runs whose fitted frequency is discontinuous
  with its order-neighbours (a jump onto the spurious branch).
* :func:`suggest_series_seeds` — per-run frequency (and restored amplitude) seeds
  interpolated from the *good* runs, i.e. the WiMDA descending-frequency warm-start
  that defeats the bistability.
* :func:`diagnose_series` — the combined verdict + human reason + suggested seeds.
* :func:`recommend_series_seeding` — the shared "Auto" seeding-mode policy.

Everything operates on plain per-run summaries (:class:`SeriesPoint`), so the core
series engines (which reseed mid-scan) and the GUI batch panel (which signposts and
offers one-click seeds after the fact) share exactly one set of thresholds.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from asymmetry.core.fitting.global_search.heuristics import is_amplitude_parameter

#: A frequency-continuity residual is only treated as an outlier when it exceeds
#: ``threshold * robust_spread`` *and* this fraction of the typical frequency, so a
#: genuinely smooth ``ν(T)`` whose tiny residuals merely have a tiny spread is not
#: flagged.
_DEFAULT_OUTLIER_THRESHOLD = 4.0
_DEFAULT_OUTLIER_REL_FLOOR = 0.20

#: An amplitude is "collapsed" when ``|A|`` falls below this fraction of the median
#: ``|A|`` across the scan (the spurious branch drives the oscillation amplitude to
#: ~0 while the real branch keeps a finite amplitude).
_DEFAULT_COLLAPSE_FRACTION = 0.15

#: Minimum members for a meaningful robust scale.
_MIN_COLLAPSE_POINTS = 3
_MIN_OUTLIER_POINTS = 4

#: Auto chains only on an ordered scan of at least this many members.
_CHAIN_MIN_MEMBERS = 3


def is_frequency_parameter(name: str) -> bool:
    """Return whether a parameter looks like an oscillation frequency / field term.

    Matches the composite-model conventions (``frequency``, ``frequency_2``, …) and
    the field-parameterised oscillators (``field``), plus the bare ``f``/``nu``/``ν``
    aliases. Used to pick the parameter whose continuity defines the scan trend.
    """
    lower = name.lower()
    if lower in {"f", "nu", "ν", "frequency", "freq", "field"}:
        return True
    return "frequency" in lower or "freq" in lower or lower.startswith("field")


def resolve_series_params(param_names: Iterable[str]) -> tuple[str | None, str | None]:
    """Pick the leading amplitude and frequency parameter names from a model.

    Returns ``(amplitude_name, frequency_name)``; either may be ``None`` when the
    model has no such parameter (e.g. a pure-relaxation model has no frequency, so
    continuity diagnostics do not apply). The *first* match in declaration order is
    used — for a composite the leading oscillatory term is the dominant signal.
    """
    amplitude_name: str | None = None
    frequency_name: str | None = None
    for name in param_names:
        if amplitude_name is None and is_amplitude_parameter(name):
            amplitude_name = name
        if frequency_name is None and is_frequency_parameter(name):
            frequency_name = name
    return amplitude_name, frequency_name


@dataclass(frozen=True)
class SeriesPoint:
    """One run's batch-fit summary, ordered along the physical scan.

    ``order`` is the temperature/field scan coordinate (falling back to the run
    number when no scan metadata exists). ``amplitude``/``frequency`` are the fitted
    values of the leading oscillatory term, or ``None`` when unavailable. ``success``
    is the optimiser's convergence flag.
    """

    run: int
    order: float
    amplitude: float | None = None
    frequency: float | None = None
    success: bool = True
    reduced_chi2: float | None = None


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(float(value))


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return float("nan")
    mid = n // 2
    if n % 2:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def detect_amplitude_collapse(
    amplitudes: Sequence[float | None],
    *,
    floor_fraction: float = _DEFAULT_COLLAPSE_FRACTION,
    min_points: int = _MIN_COLLAPSE_POINTS,
) -> list[int]:
    """Return indices whose ``|amplitude|`` collapsed far below the series scale.

    The scale is the median ``|A|`` over the finite amplitudes; an index ``i`` is
    flagged when ``|A_i| < floor_fraction * scale``. This is the spurious-branch
    signature — the oscillation amplitude pinned near zero while its neighbours keep
    a healthy amplitude. With fewer than ``min_points`` finite amplitudes the scale is
    not meaningful and nothing is flagged.
    """
    finite = [abs(float(a)) for a in amplitudes if _finite(a)]
    if len(finite) < min_points:
        return []
    scale = _median(finite)
    if not math.isfinite(scale) or scale <= 0.0:
        return []
    cutoff = floor_fraction * scale
    flagged: list[int] = []
    for i, a in enumerate(amplitudes):
        if _finite(a) and abs(float(a)) < cutoff:
            flagged.append(i)
    return flagged


def _predict_at(
    orders: Sequence[float],
    values: Sequence[float],
    k: int,
    exclude: set[int],
) -> float | None:
    """Predict ``values[k]`` from the two nearest points not in ``exclude`` (nor ``k``).

    Linear interpolation when ``k`` is bracketed by the usable points, linear
    extrapolation from the nearest two otherwise. Returns ``None`` when fewer than two
    usable points exist.
    """
    usable = [
        (orders[j], values[j])
        for j in range(len(orders))
        if j != k and j not in exclude and math.isfinite(orders[j]) and math.isfinite(values[j])
    ]
    if len(usable) < 2:
        return None
    usable.sort(key=lambda ov: ov[0])
    x = orders[k]
    lower = [ov for ov in usable if ov[0] <= x]
    upper = [ov for ov in usable if ov[0] > x]
    if lower and upper:
        (x0, y0), (x1, y1) = lower[-1], upper[0]
    elif len(lower) >= 2:
        (x0, y0), (x1, y1) = lower[-2], lower[-1]
    else:
        (x0, y0), (x1, y1) = upper[0], upper[1]
    if x1 == x0:
        return float((y0 + y1) / 2.0)
    slope = (y1 - y0) / (x1 - x0)
    return float(y0 + slope * (x - x0))


def detect_frequency_outliers(
    orders: Sequence[float],
    frequencies: Sequence[float | None],
    *,
    threshold: float = _DEFAULT_OUTLIER_THRESHOLD,
    rel_floor: float = _DEFAULT_OUTLIER_REL_FLOOR,
    min_points: int = _MIN_OUTLIER_POINTS,
) -> list[int]:
    """Return indices whose frequency is discontinuous with its order-neighbours.

    Each point's frequency is predicted from its order-neighbours and the residual is
    judged against the robust spread (MAD × 1.4826) of all residuals; a point is
    flagged only when the residual clears both ``threshold × spread`` and
    ``rel_floor × median|f|`` (the second guard stops a smooth ``ν(T)`` whose tiny
    residuals merely have a tiny spread from tripping). A run that jumped to the
    spurious branch yields a large residual.

    Detection is two-pass so a real outlier does not drag its neighbours down with it:
    a first pass nominates *candidates*, then each candidate is re-judged with the
    *other* candidates excluded from its neighbour set — a neighbour only flagged
    because the outlier corrupted its prediction is cleared, while the genuine outlier
    (predicted from clean points) stays flagged. Needs ≥ ``min_points`` finite freqs.
    """
    pts = [
        (float(orders[i]), float(frequencies[i]), i)
        for i in range(min(len(orders), len(frequencies)))
        if _finite(orders[i]) and _finite(frequencies[i])
    ]
    if len(pts) < min_points:
        return []
    pts.sort(key=lambda p: p[0])
    o = [p[0] for p in pts]
    f = [p[1] for p in pts]
    original_index = [p[2] for p in pts]
    n = len(pts)

    freq_scale = _median([abs(v) for v in f])
    abs_floor = rel_floor * freq_scale if math.isfinite(freq_scale) else 0.0

    def residual(k: int, exclude: set[int]) -> float:
        predicted = _predict_at(o, f, k, exclude)
        return 0.0 if predicted is None else f[k] - predicted

    # Pass 1: nominate candidates from leave-one-out residuals.
    resid1 = [residual(k, set()) for k in range(n)]
    spread1 = 1.4826 * _median([abs(r) for r in resid1])
    cut1 = threshold * spread1 if spread1 > 0.0 else 0.0
    candidates = {k for k in range(n) if abs(resid1[k]) > abs_floor and abs(resid1[k]) > cut1}
    if not candidates:
        return []

    # Robust scale from the clean (non-candidate) residuals so a cluster of outliers
    # cannot inflate the cut and hide itself.
    clean = [abs(resid1[k]) for k in range(n) if k not in candidates]
    spread2 = 1.4826 * _median(clean) if clean else spread1
    cut2 = threshold * spread2 if spread2 > 0.0 else 0.0

    # Pass 2: re-judge each candidate with the other candidates removed.
    flagged: list[int] = []
    for k in candidates:
        magnitude = abs(residual(k, candidates - {k}))
        if magnitude > abs_floor and magnitude > cut2:
            flagged.append(original_index[k])
    return sorted(flagged)


def suggest_series_seeds(
    points: Sequence[SeriesPoint],
    flagged_runs: Iterable[int],
    *,
    amplitude_param: str | None,
    frequency_param: str | None,
) -> dict[int, dict[str, float]]:
    """Propose per-run Local seed overrides for the flagged runs.

    Each flagged run's frequency seed is interpolated from the *good* (unflagged,
    finite-frequency) runs along the scan — the smooth-trend warm-start that steers
    the fit back onto the real branch. The amplitude seed is restored to the median
    ``|A|`` of the good runs (undoing the collapse). Returns
    ``{run: {param: value, ...}}`` containing only the parameters that could be
    estimated; runs with no usable estimate are omitted.
    """
    flagged = {int(r) for r in flagged_runs}
    if not flagged:
        return {}
    good = [p for p in points if p.run not in flagged]
    # Good points carrying a finite frequency, sorted by scan order.
    good_freq = sorted(
        ((p.order, float(p.frequency)) for p in good if _finite(p.frequency)),
        key=lambda of: of[0],
    )
    good_amp = [abs(float(p.amplitude)) for p in good if _finite(p.amplitude)]
    amp_seed = _median(good_amp) if good_amp else None

    def predict_freq(order: float) -> float | None:
        if not good_freq:
            return None
        if len(good_freq) == 1:
            return good_freq[0][1]
        xs = [of[0] for of in good_freq]
        ys = [of[1] for of in good_freq]
        if order <= xs[0]:
            return ys[0]
        if order >= xs[-1]:
            return ys[-1]
        for j in range(1, len(xs)):
            if order <= xs[j]:
                x0, x1 = xs[j - 1], xs[j]
                y0, y1 = ys[j - 1], ys[j]
                if x1 == x0:
                    return float((y0 + y1) / 2.0)
                return float(y0 + (y1 - y0) * (order - x0) / (x1 - x0))
        return ys[-1]

    by_run = {p.run: p for p in points}
    seeds: dict[int, dict[str, float]] = {}
    for run in sorted(flagged):
        point = by_run.get(run)
        if point is None:
            continue
        run_seed: dict[str, float] = {}
        if frequency_param is not None:
            predicted = predict_freq(point.order)
            if predicted is not None and math.isfinite(predicted):
                run_seed[frequency_param] = float(predicted)
        if amplitude_param is not None and amp_seed is not None and math.isfinite(amp_seed):
            run_seed[amplitude_param] = float(amp_seed)
        if run_seed:
            seeds[run] = run_seed
    return seeds


@dataclass(frozen=True)
class SeriesDiagnostics:
    """The verdict on a finished batch fit, plus the per-run cure."""

    collapsed_runs: tuple[int, ...] = ()
    outlier_runs: tuple[int, ...] = ()
    failed_runs: tuple[int, ...] = ()
    flagged_runs: tuple[int, ...] = ()
    reason: str = ""
    suggested_seeds: dict[int, dict[str, float]] = field(default_factory=dict)
    amplitude_param: str | None = None
    frequency_param: str | None = None

    @property
    def has_issues(self) -> bool:
        """Whether anything is worth signposting (a flagged or failed run)."""
        return bool(self.flagged_runs or self.failed_runs)


def _format_runs(runs: Sequence[int], by_run: Mapping[int, SeriesPoint]) -> str:
    """Render a short ``run`` list, annotating with the scan coordinate when distinct."""
    return ", ".join(str(r) for r in runs)


def diagnose_series(
    points: Sequence[SeriesPoint],
    *,
    amplitude_param: str | None,
    frequency_param: str | None,
    floor_fraction: float = _DEFAULT_COLLAPSE_FRACTION,
    outlier_threshold: float = _DEFAULT_OUTLIER_THRESHOLD,
) -> SeriesDiagnostics:
    """Diagnose a finished batch fit for collapse / discontinuity, with a cure.

    Combines amplitude-collapse and frequency-outlier detection across the per-run
    summaries, lists the runs that failed to converge, and (when a frequency
    parameter exists) computes the per-run descending-frequency seeds that steer the
    flagged runs back onto the smooth trend. ``reason`` is a one-line, user-facing
    explanation for the batch panel's signpost.
    """
    ordered = sorted(points, key=lambda p: (p.order, p.run))
    by_run = {p.run: p for p in ordered}

    amplitudes = [p.amplitude for p in ordered]
    orders = [p.order for p in ordered]
    frequencies = [p.frequency for p in ordered]

    collapsed_idx = (
        detect_amplitude_collapse(amplitudes, floor_fraction=floor_fraction)
        if amplitude_param is not None
        else []
    )
    outlier_idx = (
        detect_frequency_outliers(orders, frequencies, threshold=outlier_threshold)
        if frequency_param is not None
        else []
    )
    collapsed_runs = tuple(ordered[i].run for i in collapsed_idx)
    outlier_runs = tuple(ordered[i].run for i in outlier_idx)
    failed_runs = tuple(p.run for p in ordered if not p.success)

    flagged = sorted(
        {*collapsed_runs, *outlier_runs, *failed_runs},
        key=lambda r: (by_run[r].order if r in by_run else math.inf, r),
    )

    suggested_seeds = suggest_series_seeds(
        ordered,
        flagged,
        amplitude_param=amplitude_param,
        frequency_param=frequency_param,
    )

    parts: list[str] = []
    if collapsed_runs:
        parts.append(f"amplitude collapsed to ~0 on run(s) {_format_runs(collapsed_runs, by_run)}")
    if outlier_runs:
        parts.append(
            f"frequency jumped off the trend on run(s) {_format_runs(outlier_runs, by_run)}"
        )
    if failed_runs:
        parts.append(f"fit did not converge on run(s) {_format_runs(failed_runs, by_run)}")
    reason = "; ".join(parts)

    return SeriesDiagnostics(
        collapsed_runs=collapsed_runs,
        outlier_runs=outlier_runs,
        failed_runs=failed_runs,
        flagged_runs=tuple(flagged),
        reason=reason,
        suggested_seeds=suggested_seeds,
        amplitude_param=amplitude_param,
        frequency_param=frequency_param,
    )


@dataclass(frozen=True)
class SeedingRecommendation:
    """The seeding mode Auto picked for a series, with a human-readable reason."""

    mode: str  # "as_provided" or "chain"
    reason: str


def recommend_series_seeding(
    member_runs: Sequence[int],
    order_key: Mapping[int, float] | None,
    *,
    min_members: int = _CHAIN_MIN_MEMBERS,
) -> SeedingRecommendation:
    """Pick a seeding mode for an independent series (the shared "Auto" policy).

    Chaining from the previous run wins on **ordered scans** — a temperature or field
    sweep where each member's best seed is its neighbour, especially across a
    transition. It is pointless or harmful on unordered/repeat collections (a diverged
    member would poison the chain). So Auto chains only when a usable numeric
    ``order_key`` (run → temperature/field) spans a real range over at least
    ``min_members`` members; otherwise it leaves the caller's seeds in place. The
    returned ``reason`` is meant to be surfaced to the user — Auto is never silent.
    """
    runs = [int(r) for r in member_runs]
    n = len(runs)
    if n < min_members:
        return SeedingRecommendation(
            "as_provided",
            f"{n} member(s): too few to chain (need ≥ {min_members}) — using independent seeds",
        )
    if not order_key:
        return SeedingRecommendation(
            "as_provided",
            "no temperature/field order key — members are unordered, using independent seeds",
        )
    values = [
        float(order_key[r])
        for r in runs
        if r in order_key and order_key[r] is not None and math.isfinite(float(order_key[r]))
    ]
    if len(values) < n or len(set(values)) < 2:
        return SeedingRecommendation(
            "as_provided",
            "order key missing or constant across members — using independent seeds",
        )
    lo, hi = min(values), max(values)
    return SeedingRecommendation(
        "chain",
        f"ordered scan over {n} members spanning {lo:g}–{hi:g} — chaining from previous run",
    )
