"""Unified chained series fit for block-separable (F-B asymmetry) batches.

A batch of per-run composite fits where nothing is shared across runs (every free
parameter is Local) is *block-separable*: the joint objective factorises into one
independent minimisation per run. The historical path solved each run from a static
seed, so the menu's "Chain from previous run" mode was a silent no-op for F-B
asymmetry batches — only the grouped time-domain series actually chained.

This module gives that path a real, robust chain:

* runs are visited in physical-scan order (the ``order_key``);
* with ``seeding="chain"`` each run is warm-started from the previous **good** run's
  fitted Local values;
* a run that converges onto the spurious branch — amplitude collapsed to ~0 or
  frequency discontinuous with the trend (the near-``T_C`` bistability) — is
  *detected and reseeded* once from the smooth trend of the good runs so far, then
  refit; the better of the two attempts is kept;
* a run that fails outright, or stays bad after reseeding, does not poison the
  chain: the next run is seeded from the last good run instead.

The per-run fit kernel is :meth:`FitEngine.fit`, so this is exactly the
block-separable branch of :meth:`FitEngine.global_fit` plus the chain/reseed
orchestration. The collapse/continuity thresholds come from
:mod:`asymmetry.core.fitting.series_seeding`, shared with the GUI's post-batch
signpost so both agree on what "went wrong".
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.engine import CostFactory, FitEngine, FitResult
from asymmetry.core.fitting.member_quality import MemberQuality, assess_member_quality
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.fitting.series_seeding import (
    SeriesPoint,
    diagnose_series,
    recommend_series_seeding,
)

#: Online (single-pass) thresholds for spotting a run that landed on the spurious
#: branch, judged against the good runs seen so far. Deliberately a touch looser than
#: the batch-wide :func:`diagnose_series` cuts: mid-scan we have only the prefix of
#: good points, so we reseed only on an unmistakable departure.
_ONLINE_COLLAPSE_FRACTION = 0.2
_ONLINE_FREQ_REL_TOL = 0.5
_MIN_HISTORY = 2


@dataclass
class AsymmetrySeriesResult:
    """Per-run results of a chained block-separable series fit."""

    results: dict[int, FitResult]
    fitted_global: ParameterSet
    seeding_used: str = "as_provided"
    seeding_reason: str = ""
    #: Runs that were reseeded mid-scan (converged-but-bad → refit from the trend).
    reseeded_runs: tuple[int, ...] = ()
    order: tuple[int, ...] = field(default_factory=tuple)
    #: Per-run advisory fit-quality (χ²ᵣ, σ, flags) — the shared trend-gating
    #: contract. Diagnostic only: it never mutates a member's trend inclusion.
    member_quality: dict[int, MemberQuality] = field(default_factory=dict)


def _local_value(result: FitResult, name: str) -> float | None:
    try:
        parameter = result.parameters[name]
    except (KeyError, TypeError):
        return None
    value = float(parameter.value)
    return value if math.isfinite(value) else None


def _summarize(
    run: int,
    order: float,
    result: FitResult,
    amplitude_param: str | None,
    frequency_param: str | None,
) -> SeriesPoint:
    return SeriesPoint(
        run=run,
        order=order,
        amplitude=_local_value(result, amplitude_param) if amplitude_param else None,
        frequency=_local_value(result, frequency_param) if frequency_param else None,
        success=bool(result.success),
        reduced_chi2=float(getattr(result, "reduced_chi_squared", 0.0) or 0.0),
    )


def _predict_from_history(history: Sequence[SeriesPoint], order: float) -> float | None:
    """Linearly extrapolate the frequency trend of the good runs to ``order``."""
    pts = sorted(
        ((p.order, float(p.frequency)) for p in history if p.frequency is not None),
        key=lambda of: of[0],
    )
    if len(pts) < 2:
        return pts[0][1] if pts else None
    # Use the two history points nearest ``order`` for a local linear estimate.
    pts.sort(key=lambda of: abs(of[0] - order))
    (x0, y0), (x1, y1) = pts[0], pts[1]
    if x1 == x0:
        return float((y0 + y1) / 2.0)
    return float(y0 + (y1 - y0) * (order - x0) / (x1 - x0))


def _is_spurious(
    point: SeriesPoint,
    history: Sequence[SeriesPoint],
    amplitude_param: str | None,
    frequency_param: str | None,
) -> bool:
    """Whether a *converged* run looks like the spurious branch, given the trend.

    Needs at least :data:`_MIN_HISTORY` good runs to have a trend to judge against.
    """
    good = [p for p in history if p.success]
    if len(good) < _MIN_HISTORY:
        return False
    if amplitude_param and point.amplitude is not None:
        amps = [abs(p.amplitude) for p in good if p.amplitude is not None]
        if amps:
            scale = sorted(amps)[len(amps) // 2]
            if scale > 0 and abs(point.amplitude) < _ONLINE_COLLAPSE_FRACTION * scale:
                return True
    if frequency_param and point.frequency is not None:
        predicted = _predict_from_history(good, point.order)
        if predicted is not None and math.isfinite(predicted):
            tol = _ONLINE_FREQ_REL_TOL * max(abs(predicted), 1e-9)
            if abs(point.frequency - predicted) > tol:
                return True
    return False


def _chain_seed(
    previous: FitResult,
    provided: ParameterSet,
    local_params: Sequence[str],
) -> ParameterSet:
    """Warm-start the next run: carry the previous good run's free Local values.

    Keeps the provided structure (names, bounds, fixed flags) but overrides each free
    Local parameter with the previous run's fitted value. Fixed and Global parameters
    keep their provided values.
    """
    local = set(local_params)
    rebuilt = ParameterSet()
    for parameter in provided:
        value = parameter.value
        if not parameter.fixed and parameter.name in local:
            carried = _local_value(previous, parameter.name)
            if carried is not None:
                value = carried
        rebuilt.add(
            Parameter(
                name=parameter.name,
                value=value,
                min=parameter.min,
                max=parameter.max,
                fixed=parameter.fixed,
            )
        )
    return rebuilt


def _reseed_from_trend(
    provided: ParameterSet,
    history: Sequence[SeriesPoint],
    order: float,
    local_params: Sequence[str],
    amplitude_param: str | None,
    frequency_param: str | None,
) -> ParameterSet | None:
    """Build a continuity warm-start for a spurious run from the good-run trend.

    The frequency seed is the trend extrapolated to this run's scan coordinate and
    the amplitude seed is the median good amplitude (restoring it away from the
    collapse). Returns ``None`` when there is no usable trend to extrapolate.
    """
    good = [p for p in history if p.success]
    if not good:
        return None
    local = set(local_params)
    overrides: dict[str, float] = {}
    if frequency_param and frequency_param in local:
        predicted = _predict_from_history(good, order)
        if predicted is not None and math.isfinite(predicted):
            overrides[frequency_param] = float(predicted)
    if amplitude_param and amplitude_param in local:
        amps = [abs(p.amplitude) for p in good if p.amplitude is not None]
        if amps:
            overrides[amplitude_param] = float(sorted(amps)[len(amps) // 2])
    if not overrides:
        return None
    rebuilt = ParameterSet()
    for parameter in provided:
        value = overrides.get(parameter.name, parameter.value)
        rebuilt.add(
            Parameter(
                name=parameter.name,
                value=value,
                min=parameter.min,
                max=parameter.max,
                fixed=parameter.fixed,
            )
        )
    return rebuilt


def _better_result(
    first: FitResult,
    first_point: SeriesPoint,
    second: FitResult,
    second_point: SeriesPoint,
    history: Sequence[SeriesPoint],
    amplitude_param: str | None,
    frequency_param: str | None,
) -> tuple[FitResult, SeriesPoint]:
    """Choose between the original fit and the reseeded refit.

    Prefer a converged, non-spurious result; break ties (and the all-bad case) on the
    smaller reduced χ².
    """

    def rank(result: FitResult, point: SeriesPoint) -> tuple[int, float]:
        spurious = _is_spurious(point, history, amplitude_param, frequency_param)
        # Lower is better: converged-and-clean (0) < converged-spurious (1) < failed (2).
        if not result.success:
            tier = 2
        elif spurious:
            tier = 1
        else:
            tier = 0
        chi2 = float(getattr(result, "reduced_chi_squared", math.inf) or math.inf)
        return tier, chi2

    if rank(second, second_point) < rank(first, first_point):
        return second, second_point
    return first, first_point


def fit_asymmetry_series(
    datasets: Sequence[MuonDataset],
    model_fn: Callable[..., object],
    global_params: Sequence[str],
    local_params: Sequence[str],
    initial_params: dict[int, ParameterSet],
    *,
    fit_engine: FitEngine | None = None,
    t_min: float | None = None,
    t_max: float | None = None,
    method: str = "migrad",
    minos: bool = False,
    cancel_callback: Callable[[], bool] | None = None,
    cost_factory: CostFactory | None = None,
    seeding: str = "as_provided",
    order_key: dict[int, float] | None = None,
    amplitude_param: str | None = None,
    frequency_param: str | None = None,
) -> AsymmetrySeriesResult:
    """Fit a block-separable F-B asymmetry batch with optional robust chaining.

    Each run is fit independently via :meth:`FitEngine.fit`. With
    ``seeding="as_provided"`` the runs share no state (equivalent to the
    block-separable branch of :meth:`FitEngine.global_fit`). With ``"chain"`` each run
    warm-starts from the previous good run and a converged-but-spurious run is
    reseeded from the good-run trend and refit. ``"auto"`` resolves to one of those via
    :func:`recommend_series_seeding` over the ``order_key``.
    """
    engine = fit_engine or FitEngine()
    if not datasets:
        raise ValueError("No datasets provided for series fitting")

    run_numbers = [int(ds.run_number) for ds in datasets]
    dataset_by_run = {int(ds.run_number): ds for ds in datasets}

    resolved = seeding
    reason = ""
    if seeding == "auto":
        recommendation = recommend_series_seeding(run_numbers, order_key)
        resolved = recommendation.mode
        reason = recommendation.reason

    # Echo the global parameters (block-separable ⇒ they are fixed) as the
    # "fitted globals", matching FitEngine.global_fit's contract.
    first_params = initial_params[run_numbers[0]]
    fitted_global = ParameterSet()
    for name in global_params:
        parameter = first_params[name]
        fitted_global.add(
            Parameter(
                name=name,
                value=parameter.value,
                min=parameter.min,
                max=parameter.max,
                fixed=parameter.fixed,
            )
        )

    run_order = list(run_numbers)
    if resolved == "chain" and order_key:
        run_order.sort(key=lambda r: (order_key.get(r, math.inf), r))

    def _fit_one(run: int, seed: ParameterSet) -> FitResult:
        return engine.fit(
            dataset_by_run[run],
            model_fn,
            seed,
            t_min=t_min,
            t_max=t_max,
            method=method,
            minos=minos,
            cancel_callback=cancel_callback,
            cost_factory=cost_factory,
        )

    results: dict[int, FitResult] = {}
    reseeded: list[int] = []
    history: list[SeriesPoint] = []
    # Final chosen per-run summary points, for the batch-wide off-trend diagnosis
    # that feeds the ``spurious_reseeded`` quality flag.
    final_points: dict[int, SeriesPoint] = {}
    last_good: FitResult | None = None

    for run in run_order:
        provided = initial_params[run]
        order = float(order_key.get(run, run)) if order_key else float(run)

        if resolved == "chain" and last_good is not None:
            seed = _chain_seed(last_good, provided, local_params)
        else:
            seed = provided

        result = _fit_one(run, seed)
        point = _summarize(run, order, result, amplitude_param, frequency_param)

        # Detect-and-reseed: only for a converged run that looks spurious — a hard
        # non-convergence is handled by the chain-reset path below, not reseeded.
        if (
            resolved == "chain"
            and result.success
            and _is_spurious(point, history, amplitude_param, frequency_param)
        ):
            reseed = _reseed_from_trend(
                provided, history, order, local_params, amplitude_param, frequency_param
            )
            if reseed is not None:
                retry = _fit_one(run, reseed)
                retry_point = _summarize(run, order, retry, amplitude_param, frequency_param)
                chosen, point = _better_result(
                    result, point, retry, retry_point, history, amplitude_param, frequency_param
                )
                if chosen is retry:
                    reseeded.append(run)
                result = chosen

        results[run] = result
        final_points[run] = point

        spurious = _is_spurious(point, history, amplitude_param, frequency_param)
        if result.success and not spurious:
            last_good = result
            history.append(point)
        elif not result.success:
            # Failed outright → do not chain a diverged fit into the next run.
            last_good = None

    # Batch-wide off-trend diagnosis over the final chosen points (independent of
    # seeding mode, so a plain block-separable batch is diagnosed too). Runs whose
    # amplitude collapsed or whose frequency jumped off the trend — plus any run
    # reseeded mid-scan — carry the ``spurious_reseeded`` advisory flag.
    diagnostics = diagnose_series(
        list(final_points.values()),
        amplitude_param=amplitude_param,
        frequency_param=frequency_param,
    )
    spurious_runs = {*diagnostics.collapsed_runs, *diagnostics.outlier_runs, *reseeded}
    member_quality = {
        run: assess_member_quality(
            result,
            extra_flags=("spurious_reseeded",) if run in spurious_runs else (),
        )
        for run, result in results.items()
    }

    return AsymmetrySeriesResult(
        results=results,
        fitted_global=fitted_global,
        seeding_used=resolved,
        seeding_reason=reason,
        reseeded_runs=tuple(reseeded),
        order=tuple(run_order),
        member_quality=member_quality,
    )
