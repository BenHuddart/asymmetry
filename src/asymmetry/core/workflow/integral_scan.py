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

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.field_scan import (
    as_composite_model,
    fit_scan_baseline,
    fit_scan_model,
    parameter_set_for_model,
)
from asymmetry.core.fitting.parameter_models import suggest_model_seeds
from asymmetry.core.fitting.parameters import ParameterSet
from asymmetry.core.io.periods import build_rf_difference_scan
from asymmetry.core.transform.integral import FieldScan, build_field_scan
from asymmetry.core.workflow.reduction import (
    GREEN_MINUS_RED,
    ReductionSettings,
    red_green_curves,
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
    combined two-period runs and each point is the mean of the run's green − red
    difference over the window — the RF-resonance observable.
    """
    runs = [dataset.run for dataset in datasets]
    if settings.period == GREEN_MINUS_RED:
        if method != "integral":
            raise ValueError(
                "The green − red scan averages each run's difference curve; "
                f"method {method!r} does not apply."
            )
        return build_rf_difference_scan(
            runs,
            t_min=t_min,
            t_max=t_max,
            order_key=order_key,
            red_green=lambda run: red_green_curves(run, settings),
        )
    return build_field_scan(
        [replace(run, grouping=resolve_reduction_grouping(run, settings)) for run in runs],
        t_min=t_min,
        t_max=t_max,
        method=method,
        order_key=order_key,
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
    if {"B0", "Bwid"}.issubset(model.param_names) and scan.n_points:
        x_lo = float(np.min(scan.x))
        x_hi = float(np.max(scan.x))
        span = x_hi - x_lo
        parameters["B0"].min = x_lo
        parameters["B0"].max = x_hi
        parameters["Bwid"].min = max(span / 1000.0, np.finfo(float).eps)
        parameters["Bwid"].max = max(span, parameters["Bwid"].min)
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
) -> tuple[FieldScan, dict[str, Any]]:
    """Optionally subtract a baseline, then fit *expression* to the scan."""
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
    }
    return fitted_scan, fit_payload


def _parameter_values(parameters: ParameterSet) -> dict[str, float]:
    return {parameter.name: float(parameter.value) for parameter in parameters}


__all__ = [
    "build_integral_scan",
    "field_scan_payload",
    "fit_integral_scan",
]
