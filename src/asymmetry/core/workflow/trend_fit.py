"""Fit a parameter-vs-x model to one column of a stored trend.

The scripted form of the desktop trend dialog's fit: the same
:func:`~asymmetry.core.fitting.parameter_models.fit_parameter_model`, seeded by
the same :func:`~asymmetry.core.fitting.seeding.seed_trend_parameters` and with
the same four extra deterministic starts, so a critical-temperature or Redfield
fit converges here exactly when it converges there. Bounds are the model's
static defaults (a rate's floor at zero), as for a fit recipe.

Which points enter is explicit. Excluding a point is the analyst's call, never
automatic (as for a series, see :mod:`asymmetry.core.fitting.member_quality`):
every row with a value enters unless its key is named in ``exclude`` or it
falls outside the x range. The outcome reports every row left out with the
reason, and every flagged row that entered with its flags, so the analyst
sees exactly what the fit rests on.

A stored fit's parameters can themselves be trended across series
(:func:`fit_trend_table`) — a rate constant fitted at each temperature, then an
Arrhenius law through them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from asymmetry.core.fitting.parameter_models import ParameterCompositeModel, fit_parameter_model
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.fitting.seeding import seed_trend_parameters
from asymmetry.core.workflow.series import ScanAxis, TrendTable, build_trend_table

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
    #: Keys of the rows that entered the fit, in trend order.
    keys: list[str]
    #: ``{"key", "reason"}`` for every row that did not.
    excluded: list[dict[str, Any]]
    #: ``{"key", "flags"}`` for every row that entered carrying quality flags.
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
            "keys": list(self.keys),
            "n_points": len(self.keys),
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
    exclude: Iterable[str] = (),
) -> TrendFitOutcome:
    """Fit *expression* to the *param* column of *trend* against its x.

    ``initial`` replaces seeded start values, ``fixed`` holds parameters at a
    value, and ``exclude`` names row keys to leave out. Raises
    :class:`ValueError` for a column the trend does not carry, an unknown model
    or parameter name, an excluded key the trend does not hold, or too few
    points left to fit.
    """
    value_columns = [
        column
        for column in trend.columns
        if column not in ("key", "x", "flags", "survey_line_mhz") and not column.endswith("_err")
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

    rows_by_key = {row["key"]: row for row in trend.rows}
    exclude = set(exclude)
    strangers = sorted(exclude - set(rows_by_key))
    if strangers:
        raise ValueError(
            f"Not in the trend: {', '.join(strangers)} (it holds {', '.join(rows_by_key)})."
        )

    keys: list[str] = []
    excluded: list[dict[str, Any]] = []
    flagged: list[dict[str, Any]] = []
    for row in trend.rows:
        key = row["key"]
        reason = _exclusion_reason(row, param, x_min, x_max, exclude)
        if reason is not None:
            excluded.append({"key": key, "reason": reason})
            continue
        keys.append(key)
        if row["flags"]:
            flagged.append({"key": key, "flags": list(row["flags"])})

    x = np.asarray([rows_by_key[key]["x"] for key in keys], dtype=float)
    y = np.asarray([rows_by_key[key][param] for key in keys], dtype=float)
    y_err = np.asarray([rows_by_key[key][f"{param}_err"] for key in keys], dtype=float)
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
    if len(keys) <= n_free:
        raise ValueError(
            f"{len(keys)} point(s) are left to fit {n_free} free parameter(s) of "
            f"{expression!r}; widen the range or readmit points."
        )

    result = fit_parameter_model(x, y, y_err, model, parameters, extra_starts=_EXTRA_STARTS, seed=0)
    return TrendFitOutcome(
        param=param,
        expression=expression,
        order_key=trend.order_key,
        x_min=x_min,
        x_max=x_max,
        keys=keys,
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


def trend_fit_key(param: str, expression: str) -> str:
    """The key a series stores a trend fit under: one per law on each column."""
    return f"{param}:{expression}"


def error_scale(fit: Mapping[str, Any]) -> float:
    """√χ²ᵣ for a converged stored fit whose points scatter beyond their errors, else 1.

    The fit's own errors assume the points' errors are right; when χ²ᵣ > 1 the
    scatter says they are too small, and the errors scaled by √χ²ᵣ are the
    honest ones.
    """
    return max(1.0, fit["reduced_chi_squared"]) ** 0.5 if fit["success"] else 1.0


def fit_trend_table(
    fits: Mapping[str, Mapping[str, Any]], param: str, axis: ScanAxis[str]
) -> TrendTable:
    """*param* of each member series' stored trend fit, against *axis*.

    *fits* maps each member series to the stored :class:`TrendFitOutcome` dict
    it contributes; the rows follow *axis*. Each row carries the parameter's
    error scaled by :func:`error_scale`, so a law fitted through them weighs
    each point by the honest error, and is flagged ``failed`` for a fit that
    did not converge and ``bound_pinned`` for a parameter at a bound. Raises
    :class:`ValueError` naming a member whose law has no parameter *param*.
    """
    lacking = sorted(name for name, fit in fits.items() if param not in fit["parameters"])
    if lacking:
        raise ValueError(
            f"The trend fit of {', '.join(lacking)} has no parameter {param!r} "
            f"(it has {', '.join(fits[lacking[0]]['parameters'])})."
        )
    entries: dict[str, dict[str, Any]] = {}
    for name in sorted(fits, key=lambda name: (axis.values[name], name)):
        fit = fits[name]
        entries[name] = {
            "x": axis.values[name],
            "parameters": {param: fit["parameters"][param]},
            # A parameter the law held has no error, and its row no weight.
            "uncertainties": (
                {param: fit["uncertainties"][param] * error_scale(fit)}
                if param in fit["uncertainties"]
                else {}
            ),
            "quality_flags": (["failed"] if not fit["success"] else [])
            + (["bound_pinned"] if param in fit["params_at_bound"] else []),
        }
    return build_trend_table(entries, [param], axis.name)


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
    exclude: set[str],
) -> str | None:
    """Why *row* stays out of the fit, or ``None`` when it enters."""
    if row["key"] in exclude:
        return "excluded"
    if row[param] is None or row[f"{param}_err"] is None:
        return f"no {param} uncertainty"
    if (x_min is not None and row["x"] < x_min) or (x_max is not None and row["x"] > x_max):
        return "outside the x range"
    return None


__all__ = [
    "TrendFitOutcome",
    "error_scale",
    "fit_trend",
    "fit_trend_table",
    "trend_fit_key",
    "turning_point",
]
