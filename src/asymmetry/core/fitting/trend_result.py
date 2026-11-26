"""Uncertainty weighting and quality metadata for parameter trend fits."""

from __future__ import annotations

from collections.abc import Callable
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
    default). ``COLUMN_STABILIZED`` adds a floor of half the median retained
    positive uncertainty;
    ``PERCENT`` sets σᵢ = (value/100)·|yᵢ|; ``ABSOLUTE`` a constant σ =
    value; ``NONE`` unit weights; ``SCATTER`` fits with unit weights and
    rescales the parameter errors by √(χ²/ν) afterwards — the standard
    estimate of errors from the scatter of the points (and the fixed point
    of WiMDA's iterated Estimate mode).
    """

    COLUMN = "column"
    COLUMN_STABILIZED = "column_stabilized"
    PERCENT = "percent"
    ABSOLUTE = "absolute"
    NONE = "none"
    SCATTER = "scatter"

    @property
    def uses_column(self) -> bool:
        """Whether this mode starts from the supplied uncertainties."""
        return self in (ErrorMode.COLUMN, ErrorMode.COLUMN_STABILIZED)


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


def apply_error_mode(
    y: NDArray,
    yerr: NDArray | None,
    mode: ErrorMode | str = ErrorMode.COLUMN,
    value: float | None = None,
) -> NDArray[np.float64] | None:
    """Assign per-point σ before point selection and any selected-range floor.

    ``value`` is the percentage for ``PERCENT`` and the constant σ for
    ``ABSOLUTE``. A missing, non-finite, or non-positive value falls back to
    σ = 1 for ``ABSOLUTE`` (matching WiMDA) and to 1 % for ``PERCENT`` (so a
    zero percentage cannot zero out every error and mask the whole fit).
    Returns ``None`` for ``COLUMN`` with no error column, which callers treat
    as unit weights. Percent-mode points with y = 0 get σ = 0 and are
    excluded by the standard validity mask — they carry no error information.
    """
    mode = ErrorMode(mode)
    yy = np.asarray(y, dtype=float)
    if mode.uses_column:
        return None if yerr is None else np.asarray(yerr, dtype=float)
    if mode is ErrorMode.PERCENT:
        pct = abs(float(value)) if value is not None and np.isfinite(value) else 1.0
        if pct <= 0.0:
            pct = 1.0
        return (pct / 100.0) * np.abs(yy)
    if mode is ErrorMode.ABSOLUTE:
        const = abs(float(value)) if value is not None and np.isfinite(value) else 1.0
        if const <= 0.0:
            const = 1.0
        return np.full_like(yy, const)
    # NONE and SCATTER both weight uniformly; SCATTER additionally rescales
    # the resulting parameter errors after the fit.
    return np.ones_like(yy)


@dataclass(frozen=True)
class TrendWeights:
    """Fit selection and its aligned vertical uncertainties."""

    mask: NDArray[np.bool_]
    sigma: NDArray[np.float64]
    floor_applied: bool


def selected_point_errors(errors: NDArray[np.float64], mode: ErrorMode) -> NDArray[np.float64]:
    """Apply the selected-range floor to finite positive uncertainties."""
    if mode is ErrorMode.COLUMN_STABILIZED and errors.size:
        return np.maximum(errors, np.median(0.5 * errors))
    return errors


def resolve_trend_weights(
    x: NDArray,
    y: NDArray,
    yerr: NDArray | None,
    selection: NDArray[np.bool_],
    mode: ErrorMode,
    value: float | None = None,
) -> TrendWeights:
    """Select usable points before computing weights for this fit or group."""
    sigma = apply_error_mode(y, yerr, mode, value)
    if sigma is None:
        sigma = np.ones_like(y, dtype=float)
    mask = selection & np.isfinite(x) & np.isfinite(y) & np.isfinite(sigma) & (sigma > 0)
    selected = sigma[mask]
    effective = selected_point_errors(selected, mode)
    return TrendWeights(mask, effective, bool(np.any(effective != selected)))


def trend_x_derivative_step(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Choose the central-difference probe for the effective-variance objective."""
    return np.maximum(np.abs(x), 1.0) * 1e-6


def effective_variance_residual(
    predict: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    sigma_y2: NDArray[np.float64],
    xerr2: NDArray[np.float64],
    x_step: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Orear/York effective-variance residual when both x and y are uncertain.

    ``σ²_eff = σ_y² + (∂f/∂x)²·σ_x²``, with the local slope evaluated by a
    central finite difference ``(f(x+h) − f(x−h)) / 2h`` (no analytic
    derivative needed). A model undefined just outside the data range yields a
    non-finite probe; the slope falls back to zero there rather than poisoning
    the whole cost (``σ_y² > 0`` is guaranteed by the caller's validity mask, so
    the denominator stays positive). Shared by the single-series and
    cross-group fits so the two paths cannot drift numerically.
    """
    pred = np.asarray(predict(x), dtype=float)
    slope = (
        np.asarray(predict(x + x_step), dtype=float) - np.asarray(predict(x - x_step), dtype=float)
    ) / (2.0 * x_step)
    slope = np.where(np.isfinite(slope), slope, 0.0)
    sigma2 = sigma_y2 + (slope**2) * xerr2
    return (y - pred) / np.sqrt(sigma2)


def trend_residuals(
    predict: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    sigma: NDArray[np.float64],
    xerr: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """Evaluate the fitted objective's residuals at the actual observations."""
    if xerr is None:
        return (y - predict(x)) / sigma
    xerr = np.where(np.isfinite(xerr) & (xerr > 0.0), xerr, 0.0)
    return effective_variance_residual(predict, x, y, sigma**2, xerr**2, trend_x_derivative_step(x))


def no_trend_points_message(
    x: NDArray, y: NDArray, selection: NDArray[np.bool_], mode: ErrorMode
) -> str:
    """Explain an empty fit selection in terms of the current error mode."""
    if mode.uses_column and np.any(selection & np.isfinite(x) & np.isfinite(y)):
        return (
            "No finite positive column errors in the selected range. "
            "Choose None (unit weights) or Estimate from scatter to use these points."
        )
    return "No valid points in selected range"


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
    error_value: float | None = None
    column_error_floor_applied: bool | None = None
    shares_joint_parameters: bool = False

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
            "error_value": self.error_value,
            "column_error_floor_applied": self.column_error_floor_applied,
            "shares_joint_parameters": self.shares_joint_parameters,
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
            error_value=data.get("error_value"),
            column_error_source=None if source is None else ColumnErrorSource(source),
            column_error_floor_applied=data.get("column_error_floor_applied"),
            shares_joint_parameters=bool(data.get("shares_joint_parameters", False)),
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
        if ErrorMode(self.error_mode).uses_column:
            if self.column_error_source is None:
                return None
            if self.column_error_source is ColumnErrorSource.ASSUMED_UNIT:
                return ErrorMode.NONE.value
        return self.error_mode

    @property
    def dof(self) -> int | None:
        """Residual degrees of freedom where the observation count is known."""
        if self.shares_joint_parameters or not self.n_points:
            return None
        return self.n_points - len(self.parameters.free_parameters)

    @property
    def quality_reason(self) -> str | None:
        """Reason a χ² verdict is unavailable, or None when applicable."""
        sources = () if self.column_error_source is None else (self.column_error_source,)
        reason = trend_weighting_reason(
            self.error_mode, sources, self.uses_effective_variance, self.column_error_floor_applied
        )
        if reason is None and self.shares_joint_parameters:
            reason = (
                "This residual χ² is one group's contribution to a fit with shared parameters. "
                "The quality verdict applies to the joint fit."
            )
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
    floor_applied: bool | None = None,
) -> str | None:
    """Explain when a trend's weighting fails to support a χ² verdict."""
    if ErrorMode(error_mode).uses_column:
        if floor_applied:
            return (
                "No χ² quality verdict is available when the column error floor changes the weights. "
                "Parameter uncertainties are conditional on the floored errors."
            )
        if not column_sources or (
            error_mode == ErrorMode.COLUMN_STABILIZED and floor_applied is None
        ):
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
        "column_error_floor_applied": result.column_error_floor_applied,
        "uses_effective_variance": result.uses_effective_variance,
    }
    if isinstance(result, ParameterModelFitResult):
        source = result.column_error_source
        summary["shares_joint_parameters"] = result.shares_joint_parameters
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
