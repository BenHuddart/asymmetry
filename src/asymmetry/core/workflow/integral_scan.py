"""Agent-facing integral-asymmetry scan assembly and fitting.

This is the workflow façade over the established field-scan transform and
parameter-model fitter.  It keeps CLI serialization and data-aware starting
values out of the command module while remaining GUI-free.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

import numpy as np

from asymmetry.core.data.dataset import MuonDataset, Run
from asymmetry.core.fitting.field_scan import (
    as_composite_model,
    fit_scan_baseline,
    fit_scan_model,
    parameter_set_for_model,
)
from asymmetry.core.fitting.parameter_models import suggest_model_seeds
from asymmetry.core.fitting.parameters import ParameterSet
from asymmetry.core.io.periods import GREEN_INDEX, RED_INDEX, period_count, period_run
from asymmetry.core.transform.integral import FieldScan, build_field_scan
from asymmetry.core.workflow.reduction import (
    GREEN_MINUS_RED,
    ReductionSettings,
    resolve_reduction_grouping,
)


def build_integral_scan(
    datasets: Iterable[MuonDataset],
    settings: ReductionSettings,
    *,
    t_min: float | None = None,
    t_max: float | None = None,
    method: str = "integral",
    order_key: str = "field",
) -> FieldScan:
    """Build one integral scan from loaded datasets, reduced under *settings*.

    Each run's counts are grouped and corrected under the settings' pair,
    deadtime, t0 and good window. With :data:`GREEN_MINUS_RED` the datasets are
    combined two-period runs and each point is the green period's integral
    asymmetry less the red one's — the RF-resonance and differential-ALC
    observable, with the two periods' errors added in quadrature.
    """
    runs = [dataset.run for dataset in datasets]
    if settings.period != GREEN_MINUS_RED:
        resolved = [_resolved(run, settings) for run in runs]
        scan = build_field_scan(
            resolved,
            t_min=t_min,
            t_max=t_max,
            method=method,
            order_key=order_key,
        )
        return _decoded(scan, _source_run_numbers(resolved))
    two_period = [run for run in runs if period_count(run) == 2]
    # Each period is reduced as the single-period run it is.
    per_period = replace(settings, period=None)
    red_periods = [period_run(run, RED_INDEX) for run in two_period]
    green_periods = [period_run(run, GREEN_INDEX) for run in two_period]
    sources = _source_run_numbers(red_periods + green_periods)
    red, green = (
        build_field_scan(
            [_resolved(period, per_period) for period in periods],
            t_min=t_min,
            t_max=t_max,
            method=method,
            order_key=order_key,
        )
        for periods in (red_periods, green_periods)
    )
    # Both periods of a run share its field, temperature and window, so the two
    # scans list the same runs in the same order once decoded to their source
    # run number (encode_period_run_number).
    red_sources = [sources[encoded] for encoded in red.run_numbers]
    green_sources = [sources[encoded] for encoded in green.run_numbers]
    if red_sources != green_sources:
        raise ValueError("The red and green scans of the same runs came out in different orders.")
    return FieldScan(
        x=red.x,
        value=green.value - red.value,
        error=np.hypot(red.error, green.error),
        run_numbers=red_sources,
        order_key=red.order_key,
        method=red.method,
        x_label=red.x_label,
        y_label="Integral asymmetry (Green − Red)",
        excluded=[
            (int(run.run_number), "not a two-period (red/green) run")
            for run in runs
            if period_count(run) != 2
        ]
        + [(sources[encoded], reason) for encoded, reason in (*red.excluded, *green.excluded)],
        units=red.units,
    )


def _resolved(run: Run, settings: ReductionSettings) -> Run:
    """*run* carrying the grouping *settings* resolve for it."""
    return replace(run, grouping=resolve_reduction_grouping(run, settings))


def _source_run_numbers(runs: Iterable[Run]) -> dict[int, int]:
    """Map each *run*'s own number (period-encoded or not) to its source run number.

    A period-selected run carries ``metadata["source_run_number"]``
    (:func:`asymmetry.core.io.periods.period_run`); any other run's own number
    already *is* its source run number.
    """
    return {
        int(run.run_number): int(run.metadata.get("source_run_number", run.run_number))
        for run in runs
    }


def _decoded(scan: FieldScan, sources: Mapping[int, int]) -> FieldScan:
    """*scan* with every point's and exclusion's run number mapped to its source run.

    Every number appearing in ``scan.run_numbers``/``scan.excluded`` was drawn
    from the same runs *sources* was built from, so the lookup cannot miss.
    """
    return replace(
        scan,
        run_numbers=[sources[number] for number in scan.run_numbers],
        excluded=[(sources[number], reason) for number, reason in scan.excluded],
    )


def field_scan_payload(scan: FieldScan) -> dict[str, Any]:
    """Serialize a :class:`FieldScan` to JSON-safe values."""
    return {
        "order_key": scan.order_key,
        "method": scan.method,
        "units": scan.units.value,
        "x_label": scan.x_label,
        "y_label": scan.y_label,
        "points": [
            {
                "run": point.run_number,
                "x": point.x,
                "value": point.value,
                "error": point.error,
            }
            for point in scan.points
        ],
        "excluded": [
            {"run": int(run_number), "reason": str(reason)} for run_number, reason in scan.excluded
        ],
    }


def _parameters(
    scan: FieldScan,
    expression: str,
    *,
    initial: Mapping[str, float] | None,
    fixed: Mapping[str, float] | None,
) -> tuple[Any, ParameterSet]:
    model = as_composite_model(expression)
    starts = suggest_model_seeds(model, scan.x, scan.value, scan.error)
    starts.update({str(name): float(value) for name, value in (initial or {}).items()})
    parameters = parameter_set_for_model(model, starts)
    # Every resonance sits inside the scan with a width between a thousandth and
    # a quarter of it: two LCR components otherwise trade places, one running
    # off the axis with a negative width, and a resonance wider than a quarter
    # of the scan is indistinguishable from the polynomial background it then
    # impersonates.
    x_lo = float(np.min(scan.x))
    x_hi = float(np.max(scan.x))
    span = x_hi - x_lo
    for index, component in enumerate(model.components):
        if {"B0", "Bwid"}.issubset(component.param_names):
            centre = parameters[model.component_param_name(index, "B0")]
            width = parameters[model.component_param_name(index, "Bwid")]
            centre.min, centre.max = x_lo, x_hi
            width.min = max(span / 1000.0, np.finfo(float).eps)
            width.max = max(span / 4.0, width.min)
    fixed = {str(name): float(value) for name, value in (fixed or {}).items()}
    unknown = set(fixed) - set(model.param_names)
    if unknown:
        raise ValueError(
            f"Unknown fixed parameter(s) {sorted(unknown)}; model parameters are {model.param_names}."
        )
    for name, value in fixed.items():
        parameters[name].value = value
        parameters[name].fixed = True
    return model, parameters


def fit_integral_scan(
    scan: FieldScan,
    expression: str,
    *,
    initial: Mapping[str, float] | None = None,
    fixed: Mapping[str, float] | None = None,
    baseline_model: str | None = None,
    baseline_regions: list[tuple[float, float]] | None = None,
    x_min: float | None = None,
    x_max: float | None = None,
) -> tuple[FieldScan, dict[str, Any]]:
    """Optionally subtract a baseline, then fit *expression* to the scan.

    *x_min*/*x_max* crop the scan to that window first, so the seeds, the
    resonance bounds and the fit all see only the resonances inside it.
    """
    if x_min is not None or x_max is not None:
        inside = (scan.x >= (-np.inf if x_min is None else x_min)) & (
            scan.x <= (np.inf if x_max is None else x_max)
        )
        scan = replace(
            scan,
            x=scan.x[inside],
            value=scan.value[inside],
            error=scan.error[inside],
            run_numbers=[run for run, kept in zip(scan.run_numbers, inside, strict=True) if kept],
        )
    fitted_scan = scan
    baseline_payload = None
    if baseline_model is not None:
        _baseline_composite, baseline_parameters = _parameters(
            scan,
            baseline_model,
            initial=None,
            fixed=None,
        )
        baseline = fit_scan_baseline(
            scan,
            baseline_regions or [],
            model=baseline_model,
            parameters=baseline_parameters,
        )
        fitted_scan = baseline.corrected
        baseline_payload = {
            "model": baseline_model,
            "regions": [[lo, hi] for lo, hi in baseline.regions],
            "parameters": _parameter_values(baseline.fit.parameters),
            "uncertainties": dict(baseline.fit.uncertainties),
            "reduced_chi_squared": float(baseline.fit.reduced_chi_squared),
        }

    model, parameters = _parameters(
        fitted_scan,
        expression,
        initial=initial,
        fixed=fixed,
    )
    free = sum(1 for parameter in parameters if not parameter.fixed)
    if fitted_scan.n_points <= free:
        raise ValueError(
            f"The scan has {fitted_scan.n_points} point(s) to fit for {free} free "
            f"parameter(s) of {expression}; widen the window or hold parameters with --fix."
        )
    result = fit_scan_model(fitted_scan, model, parameters=parameters)
    fit_payload = {
        "success": bool(result.success),
        "message": str(result.message),
        "expression": expression,
        "parameters": _parameter_values(result.parameters),
        "uncertainties": {name: float(value) for name, value in result.uncertainties.items()},
        "chi_squared": float(result.chi_squared),
        "reduced_chi_squared": float(result.reduced_chi_squared),
        "n_points": int(result.n_points),
        "params_at_bound": list(result.params_at_bound),
        "baseline": baseline_payload,
        "x_min": x_min,
        "x_max": x_max,
    }
    return fitted_scan, fit_payload


def _parameter_values(parameters: ParameterSet) -> dict[str, float]:
    return {parameter.name: float(parameter.value) for parameter in parameters}


__all__ = [
    "build_integral_scan",
    "field_scan_payload",
    "fit_integral_scan",
]
