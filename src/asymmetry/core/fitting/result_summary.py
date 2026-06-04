"""Shared, JSON-serialisable summary of a fit result.

Both the run-batch and grouped-series recording paths convert a
:class:`~asymmetry.core.fitting.engine.FitResult` into the same compact shape so
a :attr:`~asymmetry.core.representation.series.FitSeries.results_by_run` entry
has one canonical structure for parameter trending.
"""

from __future__ import annotations

from typing import Any


def fit_result_summary(fit_result: Any) -> dict:
    """Return a JSON-serialisable summary of *fit_result*.

    Includes the fitted parameter values and uncertainties so a series'
    ``results_by_run`` can drive parameter trending.
    """
    parameters: dict[str, float] = {}
    parameter_set = getattr(fit_result, "parameters", None)
    if parameter_set is not None:
        for name in getattr(parameter_set, "names", []):
            try:
                parameters[str(name)] = float(parameter_set[name].value)
            except (KeyError, TypeError, ValueError, AttributeError):
                continue
    uncertainties = {
        str(k): float(v) for k, v in (getattr(fit_result, "uncertainties", {}) or {}).items()
    }
    return {
        "success": bool(getattr(fit_result, "success", False)),
        "chi_squared": float(getattr(fit_result, "chi_squared", 0.0)),
        "reduced_chi_squared": float(getattr(fit_result, "reduced_chi_squared", 0.0)),
        "parameters": parameters,
        "uncertainties": uncertainties,
    }
