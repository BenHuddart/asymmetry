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
from asymmetry.core.fitting.parameter_models import ParameterCompositeModel, suggest_model_seeds
from asymmetry.core.fitting.parameters import ParameterSet
from asymmetry.core.io.nexus import active_series_mean
from asymmetry.core.io.periods import (
    GREEN_INDEX,
    RED_INDEX,
    period_count,
    period_run,
    source_run_of,
)
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
        _decoded(
            build_field_scan(
                [_resolved(period, per_period) for period in periods],
                t_min=t_min,
                t_max=t_max,
                method=method,
                order_key=order_key,
            ),
            sources,
        )
        for periods in (red_periods, green_periods)
    )
    # Both periods of a run share its field, temperature and window, so the two
    # scans list the same source runs in the same order.
    if red.run_numbers != green.run_numbers:
        raise ValueError("The red and green scans of the same runs came out in different orders.")
    return FieldScan(
        x=red.x,
        value=green.value - red.value,
        error=np.hypot(red.error, green.error),
        run_numbers=red.run_numbers,
        order_key=red.order_key,
        method=red.method,
        x_label=red.x_label,
        y_label="Integral asymmetry (Green − Red)",
        excluded=[
            (int(run.run_number), "not a two-period (red/green) run")
            for run in runs
            if period_count(run) != 2
        ]
        + [*red.excluded, *green.excluded],
        units=red.units,
    )


def period_field_offset_gauss(runs: Iterable[Run]) -> tuple[float, int] | None:
    """The scan's mean red − green field offset in gauss, and how many runs it averages.

    Each run logs the step in Hall-probe units (``period_hall_offset``). The
    probe reads the main field through a linear response with a zero offset of
    order a kilogauss, so a *difference* converts by the slope
    ``dField_Main/dField_Hall_Z`` — regressed over the runs' active means, not
    one run's ratio of means, which carries the zero offset. ``None`` when fewer
    than two runs log the step at distinct fields, where no slope is measured.
    """
    logged = [run.metadata for run in runs if "period_hall_offset" in run.metadata]
    hall = np.array([active_series_mean(m["nexus_time_series"]["Field_Hall_Z"]) for m in logged])
    main = np.array([active_series_mean(m["nexus_time_series"]["Field_Main"]) for m in logged])
    if np.unique(hall).size < 2:
        return None
    slope = float(np.polyfit(hall, main, 1)[0])
    step = float(np.mean([m["period_hall_offset"] for m in logged]))
    return slope * step, len(logged)


def _resolved(run: Run, settings: ReductionSettings) -> Run:
    """*run* carrying the grouping *settings* resolve for it."""
    return replace(run, grouping=resolve_reduction_grouping(run, settings))


def _source_run_numbers(runs: Iterable[Run]) -> dict[int, int]:
    """Map each *run*'s own number (period-encoded or not) to its source run number."""
    return {int(run.run_number): source_run_of(run) for run in runs}


def _decoded(scan: FieldScan, sources: Mapping[int, int]) -> FieldScan:
    """*scan* with every run number — points, exclusions and a run-ordered x — at its source run.

    Every number in the scan was drawn from the runs *sources* was built from,
    so the lookup cannot miss.
    """
    run_numbers = [sources[number] for number in scan.run_numbers]
    return replace(
        scan,
        x=np.asarray(run_numbers, dtype=float) if scan.order_key == "run" else scan.x,
        run_numbers=run_numbers,
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
    fixed = {str(name): float(value) for name, value in (fixed or {}).items()}
    unknown = set(fixed) - set(model.param_names)
    if unknown:
        raise ValueError(
            f"Unknown fixed parameter(s) {sorted(unknown)}; model parameters are {model.param_names}."
        )
    known = {**{str(name): float(value) for name, value in (initial or {}).items()}, **fixed}
    parameters = parameter_set_for_model(
        model, suggest_model_seeds(model, scan.x, scan.value, scan.error, known=known)
    )
    # Every resonance sits inside the scan with a width between a thousandth and
    # a quarter of it: two LCR components otherwise trade places, one running
    # off the axis with a negative width, and a resonance wider than a quarter
    # of the scan is indistinguishable from the polynomial background it then
    # impersonates. A differential pair's copy sits a positive dB above its line,
    # inside the scan.
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
        if "dB" in component.param_names:
            offset = parameters[model.component_param_name(index, "dB")]
            offset.min = max(span / 1000.0, np.finfo(float).eps)
            offset.max = max(span, offset.min)
    for name in fixed:
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
    # Dips outside a --xmin/--xmax window are still the scan's: they are
    # searched for on the whole of it.
    whole_scan = scan
    if x_min is not None or x_max is not None:
        scan = _cropped(scan, x_min, x_max)
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
    # One extra start is the scan's own data seed (fixed values alone known),
    # which rescues a hand-given start the fit cannot converge from.
    result = fit_scan_model(fitted_scan, model, parameters=parameters, extra_starts=1)
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
        "initial": {str(name): float(value) for name, value in (initial or {}).items()},
        "fixed": [parameter.name for parameter in parameters if parameter.fixed],
        "baseline": baseline_payload,
        "x_min": x_min,
        "x_max": x_max,
        "resonance_windows": (
            [] if result.success else resonance_windows(fitted_scan, model, fixed or {})
        ),
        # Where one more line would go: the seeder's window around the next dip
        # (its centre can sit off the resonance on a curved background; the
        # window still holds it).
        "x_range": [float(np.min(fitted_scan.x)), float(np.max(fitted_scan.x))],
        "next_dip_windows": (
            [
                window
                for window in resonance_windows(
                    whole_scan, as_composite_model(f"LorentzianLCR + {expression}"), {}
                )
                if _holds_a_line(whole_scan, window)
            ]
            if result.success and any(c.name in _LCR_LINES for c in model.components)
            else []
        ),
    }
    return fitted_scan, fit_payload


def _cropped(scan: FieldScan, x_min: float | None, x_max: float | None) -> FieldScan:
    """*scan* restricted to ``x_min <= x <= x_max`` (an open end where ``None``)."""
    inside = (scan.x >= (-np.inf if x_min is None else x_min)) & (
        scan.x <= (np.inf if x_max is None else x_max)
    )
    return replace(
        scan,
        x=scan.x[inside],
        value=scan.value[inside],
        error=scan.error[inside],
        run_numbers=[run for run, kept in zip(scan.run_numbers, inside, strict=True) if kept],
    )


#: A dip's depth must exceed this many of its errors to be named: a line's
#: wing or a bump in the background fits a line of depth near zero.
_DIP_SIGNIFICANCE = 5.0

#: A window must hold this many points to be tried with one line on a slope.
_WINDOW_MIN_POINTS = 8


def _holds_a_line(scan: FieldScan, window: Mapping[str, Any]) -> bool:
    """Whether one line on a straight background fits inside *window* as a resonance.

    A seeder's window on a curved background can hold only the background's
    rise or step, or a line's wing: there a line's fit runs its centre to the
    window's edge or its width to a bound, or finds no significant depth.
    """
    part = _cropped(scan, window["x_min"], window["x_max"])
    if part.x.size < _WINDOW_MIN_POINTS:
        return False
    model, parameters = _parameters(part, "LorentzianLCR + Linear", initial=None, fixed=None)
    result = fit_scan_model(part, model, parameters=parameters, extra_starts=1)
    centre = float(result.parameters["B0"].value)
    depth = float(result.parameters["f"].value)
    return (
        bool(result.success)
        and not {"B0", "Bwid"} & set(result.params_at_bound)
        and window["x_min"] < centre < window["x_max"]
        # An ALC dip lowers the integral asymmetry, well beyond its error.
        and depth < -_DIP_SIGNIFICANCE * float(result.uncertainties["f"])
    )


#: Single-line resonance shapes a scan can be searched for one more of.
_LCR_LINES = frozenset({"LorentzianLCR", "GaussianLCR"})

#: Half-widths either side of a resonance's seeded centre that its own window
#: spans: a Lorentzian there has fallen to 1/26 of its depth, leaving baseline
#: on both sides for the background to fit.
_WINDOW_HALF_WIDTHS = 5.0


def resonance_windows(
    scan: FieldScan, model: ParameterCompositeModel, fixed: Mapping[str, float]
) -> list[dict[str, Any]]:
    """One x window per resonance of *model*, around the dip the scan's own seeding puts it on.

    The centres and widths are :func:`suggest_model_seeds`' starts with only
    *fixed* known, in x order. Each window spans :data:`_WINDOW_HALF_WIDTHS`
    half-widths of its resonance, cut at the midpoints to its neighbours and
    at the scan's ends. A resonance the seeding could not place has none.
    """
    seeds = suggest_model_seeds(model, scan.x, scan.value, scan.error, known=dict(fixed))
    resonances = []
    for index, component in enumerate(model.components):
        if not {"B0", "Bwid"}.issubset(component.param_names):
            continue
        name = model.component_param_name(index, "B0")
        width = seeds.get(model.component_param_name(index, "Bwid"))
        if name in seeds and width is not None:
            resonances.append((seeds[name], width, name, component.name))
    resonances.sort()
    centres = [centre for centre, *_ in resonances]
    edges = [
        float(np.min(scan.x)),
        *((low + high) / 2.0 for low, high in zip(centres, centres[1:])),
        float(np.max(scan.x)),
    ]
    return [
        {
            "parameter": name,
            "component": component,
            "centre": float(centre),
            "x_min": float(max(low, centre - _WINDOW_HALF_WIDTHS * abs(width))),
            "x_max": float(min(high, centre + _WINDOW_HALF_WIDTHS * abs(width))),
        }
        for (centre, width, name, component), low, high in zip(resonances, edges, edges[1:])
    ]


def _parameter_values(parameters: ParameterSet) -> dict[str, float]:
    return {parameter.name: float(parameter.value) for parameter in parameters}


__all__ = [
    "period_field_offset_gauss",
    "build_integral_scan",
    "field_scan_payload",
    "fit_integral_scan",
    "resonance_windows",
]
