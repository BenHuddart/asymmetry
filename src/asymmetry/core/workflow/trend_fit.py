"""Fit a parameter-vs-x model to one column of a stored trend.

The scripted form of the desktop trend dialog's fit: the same
:func:`~asymmetry.core.fitting.parameter_models.fit_parameter_model`, seeded by
the same :func:`~asymmetry.core.fitting.seeding.seed_trend_parameters` and with
the same four extra deterministic starts, so a critical-temperature or Redfield
fit converges here exactly when it converges there. Bounds are the model's
static defaults (a rate's floor at zero), as for a fit recipe.

Which points enter is explicit. Excluding a run is the analyst's call, never
automatic (as for a series, see :mod:`asymmetry.core.fitting.member_quality`):
every row with a value enters unless its run is named in ``exclude`` or it
falls outside the x range. The outcome reports every row left out with the
reason, and every flagged row that entered with its flags, so the analyst
sees exactly what the fit rests on.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from asymmetry.core.fitting.parameter_models import ParameterCompositeModel, fit_parameter_model
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.fitting.seeding import seed_trend_parameters
from asymmetry.core.workflow.series import TrendTable

#: Extra deterministic starts beyond the seeded one — the trend dialog's value.
_EXTRA_STARTS = 4


@dataclass(frozen=True)
class TrendFitOutcome:
    """One model fitted to one trend column, with the points it rests on."""

    param: str
    expression: str
    order_key: str
    x_min: float | None
    x_max: float | None
    #: Runs whose rows entered the fit, in trend order.
    runs: list[int]
    #: ``{"run", "reason"}`` for every row that did not.
    excluded: list[dict[str, Any]]
    #: ``{"run", "flags"}`` for every row that entered carrying quality flags.
    flagged: list[dict[str, Any]]
    success: bool
    message: str
    parameters: dict[str, float]
    uncertainties: dict[str, float]
    fixed: list[str]
    chi_squared: float
    reduced_chi_squared: float
    params_at_bound: list[str]
    #: The x range the fitted points actually span.
    x_fitted: tuple[float, float]
    #: Each parameter's unit, where the law's component declares one.
    units: dict[str, str | None]
    #: x of an interior extremum the fitted points turn through, or ``None``.
    turning_point: float | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain, JSON-safe dict."""
        return {
            "param": self.param,
            "expression": self.expression,
            "order_key": self.order_key,
            "x_min": self.x_min,
            "x_max": self.x_max,
            "runs": list(self.runs),
            "n_points": len(self.runs),
            "excluded": [dict(entry) for entry in self.excluded],
            "flagged": [dict(entry) for entry in self.flagged],
            "success": self.success,
            "message": self.message,
            "parameters": dict(self.parameters),
            "uncertainties": dict(self.uncertainties),
            "fixed": list(self.fixed),
            "chi_squared": self.chi_squared,
            "reduced_chi_squared": self.reduced_chi_squared,
            "params_at_bound": list(self.params_at_bound),
            "x_fitted": list(self.x_fitted),
            "units": dict(self.units),
            "turning_point": self.turning_point,
        }


def fit_trend(
    trend: TrendTable,
    param: str,
    expression: str,
    *,
    x_min: float | None = None,
    x_max: float | None = None,
    initial: Mapping[str, float] | None = None,
    fixed: Mapping[str, float] | None = None,
    exclude: Iterable[int] = (),
) -> TrendFitOutcome:
    """Fit *expression* to the *param* column of *trend* against its x.

    ``initial`` replaces seeded start values, ``fixed`` holds parameters at a
    value, and ``exclude`` names runs to leave out. Raises :class:`ValueError`
    for a column the trend does not carry, an unknown model or parameter name,
    an excluded run the trend does not hold, or too few points left to fit.
    """
    value_columns = [
        column
        for column in trend.columns
        if column not in ("run", "x", "flags", "survey_line_mhz") and not column.endswith("_err")
    ]
    if param not in value_columns:
        raise ValueError(
            f"The trend has no column {param!r} (it tabulates {', '.join(value_columns)})."
        )
    model = ParameterCompositeModel.from_expression(expression)
    initial = {str(name): float(value) for name, value in (initial or {}).items()}
    fixed = {str(name): float(value) for name, value in (fixed or {}).items()}
    unknown = sorted((set(initial) | set(fixed)) - set(model.param_names))
    if unknown:
        raise ValueError(
            f"{', '.join(unknown)} is not a parameter of {expression!r} "
            f"(it has {', '.join(model.param_names)})."
        )

    rows_by_run = {int(row["run"]): row for row in trend.rows}
    exclude = {int(run) for run in exclude}
    strangers = sorted(exclude - set(rows_by_run))
    if strangers:
        raise ValueError(f"Run(s) {', '.join(str(run) for run in strangers)} are not in the trend.")

    runs: list[int] = []
    excluded: list[dict[str, Any]] = []
    flagged: list[dict[str, Any]] = []
    for row in trend.rows:
        run = int(row["run"])
        reason = _exclusion_reason(row, param, x_min, x_max, exclude)
        if reason is not None:
            excluded.append({"run": run, "reason": reason})
            continue
        runs.append(run)
        if row["flags"]:
            flagged.append({"run": run, "flags": list(row["flags"])})

    x = np.asarray([rows_by_run[run]["x"] for run in runs], dtype=float)
    y = np.asarray([rows_by_run[run][param] for run in runs], dtype=float)
    y_err = np.asarray([rows_by_run[run][f"{param}_err"] for run in runs], dtype=float)
    seeds = seed_trend_parameters(model, x, y)
    parameters = ParameterSet()
    for name in model.param_names:
        seed = seeds[name]
        parameters.add(
            Parameter(
                name=name,
                value=fixed.get(name, initial.get(name, seed.value)),
                min=seed.min,
                max=seed.max,
                fixed=name in fixed or seed.fixed,
            )
        )
    n_free = len(parameters.free_parameters)
    if len(runs) <= n_free:
        raise ValueError(
            f"{len(runs)} point(s) are left to fit {n_free} free parameter(s) of "
            f"{expression!r}; widen the range or readmit runs."
        )

    result = fit_parameter_model(x, y, y_err, model, parameters, extra_starts=_EXTRA_STARTS, seed=0)
    return TrendFitOutcome(
        param=param,
        expression=expression,
        order_key=trend.order_key,
        x_min=x_min,
        x_max=x_max,
        runs=runs,
        excluded=excluded,
        flagged=flagged,
        success=bool(result.success),
        message=str(result.message),
        parameters={parameter.name: float(parameter.value) for parameter in result.parameters},
        uncertainties={name: float(value) for name, value in result.uncertainties.items()},
        fixed=[parameter.name for parameter in result.parameters if parameter.fixed],
        chi_squared=float(result.chi_squared),
        reduced_chi_squared=float(result.reduced_chi_squared),
        params_at_bound=list(result.params_at_bound),
        x_fitted=(float(np.min(x)), float(np.max(x))),
        units={name: model.param_info[name].unit for name in model.param_names},
        turning_point=turning_point(x, y, y_err),
    )


def turning_point(x: np.ndarray, y: np.ndarray, y_err: np.ndarray) -> float | None:
    """x of an interior minimum or maximum both ends clear by two errors, else ``None``.

    A monotonic law (Arrhenius, an order parameter, a one-sided divergence)
    fitted across such a point averages two regimes into one.
    """
    order = np.argsort(x)
    xs, ys, es = x[order], y[order], np.abs(y_err[order])
    for index in (int(np.argmin(ys)), int(np.argmax(ys))):
        if 0 < index < ys.size - 1:
            depth = np.abs(ys[[0, -1]] - ys[index]) - 2.0 * np.hypot(es[[0, -1]], es[index])
            if np.all(depth > 0.0):
                return float(xs[index])
    return None


def _exclusion_reason(
    row: Mapping[str, Any],
    param: str,
    x_min: float | None,
    x_max: float | None,
    exclude: set[int],
) -> str | None:
    """Why *row* stays out of the fit, or ``None`` when it enters."""
    if int(row["run"]) in exclude:
        return "excluded"
    if row[param] is None or row[f"{param}_err"] is None:
        return f"no {param} uncertainty"
    if (x_min is not None and row["x"] < x_min) or (x_max is not None and row["x"] > x_max):
        return "outside the x range"
    return None


__all__ = ["TrendFitOutcome", "fit_trend", "turning_point"]
