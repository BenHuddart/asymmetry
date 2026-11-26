"""Uncertainty weighting and quality metadata for parameter trend fits."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.fitting.fit_quality import (
    FitQuality,
    assess_fit_quality,
    quality_inapplicability_reason,
    recorded_statistic,
)
from asymmetry.core.fitting.parameters import ParameterSet
from asymmetry.core.fitting.result_payload import (
    _deserialize_parameter_set,
    _serialize_parameter_set,
)

if TYPE_CHECKING:
    from asymmetry.core.fitting.parameter_models import CrossGroupFitResult


class ErrorMode(str, Enum):
    """How per-point σ values are assigned when weighting a model fit.

    ``COLUMN`` uses the propagated errors of the trended parameter (the
    default and the only mode where the stabilisation floor applies);
    ``PERCENT`` sets σᵢ = (value/100)·|yᵢ|; ``ABSOLUTE`` a constant σ =
    value; ``NONE`` unit weights; ``SCATTER`` fits with unit weights and
    rescales the parameter errors by √(χ²/ν) afterwards — the standard
    estimate of errors from the scatter of the points (and the fixed point
    of WiMDA's iterated Estimate mode).
    """

    COLUMN = "column"
    PERCENT = "percent"
    ABSOLUTE = "absolute"
    NONE = "none"
    SCATTER = "scatter"


class ColumnErrorSource(str, Enum):
    """Source of the retained vertical errors under a Column request."""

    SUPPLIED = "supplied"
    ASSUMED_UNIT = "assumed_unit"
    INCLUDES_ESTIMATES = "includes_estimates"


def column_error_source(
    yerr: NDArray | None, estimated: NDArray[np.bool_] | None, mask: NDArray[np.bool_]
) -> ColumnErrorSource:
    """Describe contributions to the actual objective after windowing and masking."""
    if yerr is None:
        return ColumnErrorSource.ASSUMED_UNIT
    if estimated is not None and np.any(estimated[mask]):
        return ColumnErrorSource.INCLUDES_ESTIMATES
    return ColumnErrorSource.SUPPLIED


def estimate_missing_trend_errors(
    y: NDArray, yerr: NDArray
) -> tuple[NDArray, NDArray[np.bool_] | None]:
    """Apply the trend editor's exploratory fallback and identify affected points."""
    missing = ~np.isfinite(yerr) | (yerr <= 0)
    if not np.any(missing):
        return yerr, None
    finite = np.abs(y[np.isfinite(y)])
    fallback = max(float(np.median(finite)) * 0.02, 1e-9) if finite.size else 1e-3
    return np.where(missing, fallback, yerr), missing


@dataclass
class ParameterModelFitResult:
    """Result of fitting a parameter-vs-x model."""

    success: bool
    chi_squared: float = 0.0
    reduced_chi_squared: float = 0.0
    parameters: ParameterSet = field(default_factory=ParameterSet)
    uncertainties: dict[str, float] = field(default_factory=dict)
    message: str = ""
    #: Requested error mode, retained for refitting.
    error_mode: str = ErrorMode.COLUMN.value
    #: Number of points that entered the fit (0 when unknown/legacy).
    n_points: int = 0
    #: How many initial-value candidates the multi-start machinery actually
    #: tried (1 for the default single-start path; more with ``extra_starts``).
    n_starts_tried: int = 0
    #: ``True`` iff the winning fit came from a *non-user* start (a data-aware or
    #: perturbed seed beat the user's provided parameters). Advisory only.
    seed_beat_user_start: bool = False
    #: Names of free parameters whose fitted value sits within tolerance of a
    #: finite ``min``/``max`` bound — a sign the data did not determine them.
    params_at_bound: tuple[str, ...] = ()
    #: ``(names, matrix)`` of the free-parameter covariance from Minuit, in
    #: the same order as ``names`` (free parameters only, fixed parameters
    #: excluded). ``None`` when unavailable: the fit failed/invalid or Minuit
    #: produced no covariance (e.g. HESSE did not run/converge).
    covariance: tuple[list[str], list[list[float]]] | None = None
    computation_revision: str | None = None
    input_width_context: dict | None = None
    #: None on legacy results where the Column weighting was not recorded.
    column_error_source: ColumnErrorSource | None = None
    uses_effective_variance: bool | None = None

    def __post_init__(self) -> None:
        dof = self.dof
        if dof is not None and dof <= 0:
            self.reduced_chi_squared = float("nan")

    def to_dict(self) -> dict:
        """Serialize a solved range without inferring historical weighting."""
        covariance = None
        if self.covariance is not None:
            names, matrix = self.covariance
            covariance = {
                "names": list(names),
                "matrix": [[float(v) for v in row] for row in matrix],
            }
        return {
            "success": self.success,
            "computation_revision": self.computation_revision,
            "input_width_context": self.input_width_context,
            "chi_squared": self.chi_squared,
            "reduced_chi_squared": recorded_statistic(self.reduced_chi_squared),
            "message": self.message,
            "error_mode": self.error_mode,
            "column_error_source": self.column_error_source,
            "uses_effective_variance": self.uses_effective_variance,
            "n_points": self.n_points,
            "parameters": _serialize_parameter_set(self.parameters),
            "uncertainties": dict(self.uncertainties),
            "params_at_bound": list(self.params_at_bound),
            **({"covariance": covariance} if covariance is not None else {}),
        }

    @classmethod
    def from_dict(
        cls, data: dict, *, legacy_diffusion_name: bool = False
    ) -> ParameterModelFitResult:
        """Read a stored range whose weighting and point count may be unavailable."""
        raw_parameters = data.get("parameters", [])
        if legacy_diffusion_name and isinstance(raw_parameters, list):
            raw_parameters = [
                dict(p, name="A") if isinstance(p, dict) and p.get("name") == "C" else p
                for p in raw_parameters
            ]
        covariance = None
        raw_covariance = data.get("covariance")
        if isinstance(raw_covariance, dict):
            names, matrix = raw_covariance.get("names"), raw_covariance.get("matrix")
            if isinstance(names, (list, tuple)) and isinstance(matrix, (list, tuple)):
                try:
                    covariance = (
                        [str(name) for name in names],
                        [[float(v) for v in row] for row in matrix],
                    )
                except (TypeError, ValueError):
                    pass
        bounds = data.get("params_at_bound")
        reduced = data.get("reduced_chi_squared", 0.0)
        source = data.get("column_error_source")
        return cls(
            success=bool(data.get("success", False)),
            computation_revision=data.get("computation_revision"),
            input_width_context=data.get("input_width_context"),
            chi_squared=float(data.get("chi_squared", 0.0)),
            reduced_chi_squared=float("nan") if reduced is None else float(reduced),
            message=str(data.get("message", "")),
            error_mode=str(data.get("error_mode", ErrorMode.COLUMN.value)),
            column_error_source=None if source is None else ColumnErrorSource(source),
            uses_effective_variance=data.get("uses_effective_variance"),
            n_points=int(data.get("n_points", 0)),
            parameters=_deserialize_parameter_set(raw_parameters),
            uncertainties={str(k): float(v) for k, v in data.get("uncertainties", {}).items()},
            params_at_bound=tuple(str(name) for name in bounds)
            if isinstance(bounds, (list, tuple))
            else (),
            covariance=covariance,
        )

    @property
    def effective_error_mode(self) -> str | None:
        """Effective y weighting; None when historical Column weighting is unknown."""
        if self.error_mode == ErrorMode.COLUMN:
            if self.column_error_source is None:
                return None
            if self.column_error_source is ColumnErrorSource.ASSUMED_UNIT:
                return ErrorMode.NONE.value
        return self.error_mode

    @property
    def dof(self) -> int | None:
        """Residual degrees of freedom where the observation count is known."""
        return self.n_points - len(self.parameters.free_parameters) if self.n_points else None

    @property
    def quality_reason(self) -> str | None:
        """Reason a χ² verdict is unavailable, or None when applicable."""
        sources = () if self.column_error_source is None else (self.column_error_source,)
        reason = trend_weighting_reason(self.error_mode, sources, self.uses_effective_variance)
        return quality_inapplicability_reason(self.dof, reason)

    def assess_quality(self, confidence: float = 0.95) -> FitQuality | None:
        """Evaluate the reference band when the weighting and degree count allow it."""
        dof = self.dof
        if dof is None or self.quality_reason is not None:
            return None
        quality = assess_fit_quality(self.chi_squared, dof, confidence)
        return quality if quality.verdict is not None else None


def trend_weighting_reason(
    error_mode: str,
    column_sources: tuple[ColumnErrorSource, ...],
    uses_effective_variance: bool | None = None,
) -> str | None:
    """Explain when a trend's weighting fails to support a χ² verdict."""
    if error_mode == ErrorMode.COLUMN:
        if not column_sources:
            return "No χ² quality verdict is available because the effective weighting is unknown."
        if ColumnErrorSource.INCLUDES_ESTIMATES in column_sources:
            return (
                "No χ² quality verdict is available when fitted y errors include estimates. "
                "Parameter uncertainties are conditional on those estimates."
            )
        if ColumnErrorSource.ASSUMED_UNIT in column_sources:
            weights = (
                "assumed unit y errors and supplied x errors"
                if uses_effective_variance
                else "assumed unit y errors"
            )
            return (
                f"No χ² quality verdict is available with {weights}. "
                "Parameter uncertainties are conditional on the assumed y scale."
            )
    if error_mode == ErrorMode.NONE:
        return "No χ² quality verdict is available for unit weights."
    if error_mode == ErrorMode.SCATTER:
        return "No χ² quality verdict is available for scatter-estimated errors."
    return None


def trend_fit_statistics(result: ParameterModelFitResult | CrossGroupFitResult) -> dict:
    """Record objective, residual count, and weighting for a derived trend series."""
    summary = {
        "success": result.success,
        "computation_revision": result.computation_revision,
        "chi_squared": result.chi_squared,
        "reduced_chi_squared": recorded_statistic(result.reduced_chi_squared),
        "ndof": result.dof,
        "quality_reason": result.quality_reason,
        "error_mode": result.error_mode,
        "uses_effective_variance": result.uses_effective_variance,
    }
    if isinstance(result, ParameterModelFitResult):
        source = result.column_error_source
        summary["column_error_source"] = None if source is None else source.value
        summary["effective_y_error_mode"] = result.effective_error_mode
    else:
        summary["column_error_sources"] = {
            group: source.value for group, source in result.column_error_sources.items()
        }
    return summary


def reduced_chi_squared_parameter(summary: dict) -> dict[str, float]:
    """Offer the reduced statistic as a synthetic trend column where defined."""
    value = summary["reduced_chi_squared"]
    return {} if value is None else {"chi2_r": value}
