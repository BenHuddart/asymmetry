"""Shared wizard fit payloads retain scalar reports and optional exact residual support."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.data.ratio_validity import ObservationReport
from asymmetry.core.fitting.likelihood import Objective
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.fitting.results import FitResult


def _serialize_parameter_set(parameters: ParameterSet) -> list[dict[str, object]]:
    return [
        {
            "name": parameter.name,
            "value": float(parameter.value),
            "min": float(parameter.min),
            "max": float(parameter.max),
            "fixed": bool(parameter.fixed),
            "expr": parameter.expr,
        }
        for parameter in parameters
    ]


def _deserialize_parameter_set(payload: object) -> ParameterSet:
    parameters = ParameterSet()
    if not isinstance(payload, list):
        return parameters
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str):
            continue
        try:
            parameters.add(
                Parameter(
                    name=name,
                    value=float(entry.get("value", 0.0)),
                    min=float(entry.get("min", -float("inf"))),
                    max=float(entry.get("max", float("inf"))),
                    fixed=bool(entry.get("fixed", False)),
                    expr=str(entry["expr"]) if entry.get("expr") is not None else None,
                )
            )
        except (TypeError, ValueError):
            continue
    return parameters


def _serialize_fit_result(
    result: FitResult, *, include_residuals: bool = True
) -> dict[str, object]:
    """Persist parameters, advisories and corrections; optionally omit residuals.

    Cached diagnostics own their scalar residual statistics. Decimating residuals
    would mismatch the cached time grid, so compact caches omit that array.
    """
    return {
        "success": bool(result.success),
        "warnings": list(result.warnings),
        "correction_metadata": dict(result.correction_metadata),
        "observation_report": result.observation_report.to_dict()
        if result.observation_report is not None
        else None,
        "dof": result.dof,
        "chi_squared": float(result.chi_squared),
        "reduced_chi_squared": float(result.reduced_chi_squared),
        "parameters": _serialize_parameter_set(result.parameters),
        "uncertainties": {name: float(value) for name, value in result.uncertainties.items()},
        "residual_sigma": result.residual_sigma.tolist()
        if include_residuals and result.residual_sigma is not None
        else None,
        "residual_time": result.residual_time.tolist()
        if include_residuals and result.residual_time is not None
        else None,
        "residuals": (
            np.asarray(result.residuals, dtype=float).tolist()
            if (include_residuals and result.residuals is not None)
            else None
        ),
        "message": result.message,
    }


def _deserialize_fit_result(payload: object) -> FitResult | None:
    if not isinstance(payload, dict):
        return None
    try:
        uncertainties = {
            str(name): float(value)
            for name, value in (payload.get("uncertainties", {}) or {}).items()
        }
    except (TypeError, ValueError):
        uncertainties = {}
    residuals_payload = payload.get("residuals")
    residuals: NDArray[np.float64] | None
    if residuals_payload is None:
        residuals = None
    else:
        try:
            residuals = np.asarray(residuals_payload, dtype=float)
        except (TypeError, ValueError):
            residuals = None
    return FitResult(
        objective=Objective.GAUSSIAN,
        observation_report=(
            ObservationReport(
                **{k: v for k, v in payload["observation_report"].items() if k != "omitted"}
            )
            if payload.get("observation_report") is not None
            else None
        ),
        dof=int(payload.get("dof", 0)),
        warnings=list(payload.get("warnings", [])),
        correction_metadata=dict(payload.get("correction_metadata", {})),
        success=bool(payload.get("success", False)),
        chi_squared=float(payload.get("chi_squared", 0.0)),
        reduced_chi_squared=float(payload.get("reduced_chi_squared", 0.0)),
        parameters=_deserialize_parameter_set(payload.get("parameters", [])),
        uncertainties=uncertainties,
        residuals=residuals,
        residual_sigma=np.asarray(payload["residual_sigma"], dtype=float)
        if payload.get("residual_sigma") is not None
        else None,
        residual_time=np.asarray(payload["residual_time"], dtype=float)
        if payload.get("residual_time") is not None
        else None,
        message=str(payload.get("message", "")),
    )
