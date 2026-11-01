r"""Fit baseline + peak models to an ALC / integral-asymmetry field scan.

The field scan (integral asymmetry vs a run variable) produced by
:func:`asymmetry.core.transform.build_field_scan` is fitted with the **existing**
parameter-model machinery — the same `fit_parameter_model` /
`ParameterCompositeModel` engine the trending panel uses — so no new fitter is
introduced.

The baseline-first workflow fits non-resonant regions, subtracts the fitted
baseline, and propagates its uncertainty into subsequent peak inference.

Scale: a :class:`FieldScan`'s ``value``/``error`` are the integral asymmetry **as
a fraction** (``A ∈ [-1, 1]`` — see its ``units`` field), *not* the percent of a
:class:`~asymmetry.core.data.dataset.MuonDataset`. So every amplitude fitted here,
and every amplitude seed in the ALC / LCR parameter models (``GaussianLCR`` seeds
``f`` near 0.1), is on that fractional scale. These functions pass the scan's
values through unchanged, so a scan scaled to percent fits equally well and
simply reports percent-scale amplitudes. This is the one corner of the fitting
layer whose asymmetry convention is *not* percent — see "Asymmetry units across
the API" in the documentation.

Layering: this lives in the fitting package because it depends on both
:mod:`asymmetry.core.transform` (the :class:`FieldScan`) and
:mod:`asymmetry.core.fitting.parameter_models`. It must stay free of Qt /
matplotlib / ``asymmetry.gui`` imports.

Peak-model note: the built-in ``Lorentzian`` component is centred at zero
(``a/(1+(x/B0)²)+c``); for an off-zero ALC resonance use ``GaussianLCR``
(``f·exp(-(x-B0)²/2·Bwid²)``, centred at ``B0``). A centred Lorentzian peak is a
follow-up model-library addition; these helpers are model-agnostic and work with
whatever component is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.fitting.parameter_models import (
    ParameterCompositeModel,
    ParameterModelFitResult,
    fit_parameter_model,
    stabilized_point_errors,
    suggest_model_seeds,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.transform import FieldScan

__all__ = [
    "parameter_set_for_model",
    "as_composite_model",
    "fit_scan_model",
    "fit_scan_baseline",
    "fit_scan_peaks",
    "ScanPeakFit",
    "fit_rf_resonance",
    "rf_resonance_seeds",
    "ScanBaselineResult",
]

#: Component name of the RF-µSR muon+proton resonance model (registered field-scope
#: in :mod:`asymmetry.core.fitting.parameter_models`).
RF_RESONANCE_COMPONENT = "RFResonanceMuP"


def as_composite_model(
    model: str | list[str] | ParameterCompositeModel,
) -> ParameterCompositeModel:
    """Coerce a model spec to a :class:`ParameterCompositeModel`.

    Accepts an existing model, a single component name, a component-name
    expression (``"GaussianLCR + Constant"``), or a list of component names.
    """
    if isinstance(model, ParameterCompositeModel):
        return model
    if isinstance(model, str):
        return ParameterCompositeModel.from_expression(model)
    if isinstance(model, list):
        return ParameterCompositeModel(model)
    raise TypeError(
        f"Expected a model name/expression/list/ParameterCompositeModel, got {type(model).__name__}"
    )


def parameter_set_for_model(
    model: ParameterCompositeModel,
    overrides: dict[str, float] | None = None,
) -> ParameterSet:
    """Build a :class:`ParameterSet` seeded from the model's defaults.

    ``overrides`` replaces individual starting values (the GUI passes the user's
    edited guesses). Mirrors how the trending panel seeds parameters.
    """
    overrides = overrides or {}
    unknown = set(overrides) - set(model.param_names)
    if unknown:
        raise ValueError(
            f"Unknown parameter override(s) {sorted(unknown)}; "
            f"model parameters are {model.param_names}."
        )
    params = ParameterSet()
    for name in model.param_names:
        value = float(overrides.get(name, model.param_defaults.get(name, 0.0)))
        params.add(Parameter(name=name, value=value))
    return params


def fit_scan_model(
    scan: FieldScan,
    model: str | list[str] | ParameterCompositeModel,
    *,
    parameters: ParameterSet | None = None,
    initial: dict[str, float] | None = None,
    x_min: float | None = None,
    x_max: float | None = None,
    method: str = "migrad",
    extra_starts: int = 0,
) -> ParameterModelFitResult:
    """Fit a parameter model to a field scan's ``(x, value, error)``.

    Amplitude parameters come back on the **scan's own asymmetry scale** — a
    fraction (``A ∈ [-1, 1]``) for a scan straight out of
    :func:`~asymmetry.core.transform.build_field_scan`, per the module docstring.

    The thin adapter that lets a :class:`FieldScan` flow into the existing
    ``fit_parameter_model``. Used for the peak-fit step on a (typically
    baseline-corrected) scan; **check ``result.success`` before** reading the
    resonance field and width from :attr:`ParameterModelFitResult.parameters`
    (e.g. ``B0`` / ``Bwid`` for ``GaussianLCR``) — on a failed fit the parameter
    set may be empty.

    Pass starting values via *either* an explicit ``parameters`` set *or* an
    ``initial`` override dict, not both. ``extra_starts`` is
    :func:`~asymmetry.core.fitting.parameter_models.fit_parameter_model`'s
    multi-start, which also tries the scan's own data seed.
    """
    composite = as_composite_model(model)
    if parameters is not None and initial is not None:
        raise ValueError("Pass either `parameters` or `initial`, not both.")
    if parameters is None:
        parameters = parameter_set_for_model(composite, initial)
    return fit_parameter_model(
        scan.x,
        scan.value,
        scan.error,
        composite,
        parameters,
        x_min=x_min,
        x_max=x_max,
        method=method,
        extra_starts=extra_starts,
    )


def rf_resonance_seeds(
    scan: FieldScan,
    *,
    nu_rf: float,
    a_mu: float = 515.0,
    a_p: float = 124.0,
) -> dict[str, float]:
    """Seed values for an :data:`RF_RESONANCE_COMPONENT` fit of *scan*.

    The two resonance fields ``B1, B2`` are derived inside the model from
    ``A_µ``/``A_p``/``ν_RF``, so those three seed the *position* and *splitting*
    (defaults 515/124 MHz put the dips near the benzene 866/772 G). The depths,
    widths and background are seeded from the data by the model's own estimator
    in :func:`~asymmetry.core.fitting.parameter_models.suggest_model_seeds`, so
    the GUI's RF fit and a scripted one start from the same place.
    """
    composite = as_composite_model(RF_RESONANCE_COMPONENT)
    known = {"A_mu": float(a_mu), "A_p": float(a_p), "nu_RF": float(nu_rf)}
    return {
        **composite.param_defaults,
        **suggest_model_seeds(composite, scan.x, scan.value, scan.error, known=known),
    }


def fit_rf_resonance(
    scan: FieldScan,
    *,
    nu_rf: float,
    a_mu: float = 515.0,
    a_p: float = 124.0,
    fix_nu_rf: bool = True,
    x_min: float | None = None,
    x_max: float | None = None,
    method: str = "migrad",
) -> ParameterModelFitResult:
    """Fit the RF-µSR muon+proton resonance model to an RF field scan.

    Wraps :func:`fit_scan_model` for the :data:`RF_RESONANCE_COMPONENT` model so
    the GUI and scripts share one seeding + fixing rule. The scan is the
    **(Green − Red)** integral asymmetry vs field built by
    :func:`asymmetry.core.io.periods.build_rf_difference_scan`. ``ν_RF`` is a
    known acquisition constant and is **held fixed by default**; ``A_µ`` (mean dip
    position) and ``A_p`` (dip splitting) are the couplings read off the result —
    **check ``result.success`` first**. Seeds come from :func:`rf_resonance_seeds`.
    """
    composite = as_composite_model(RF_RESONANCE_COMPONENT)
    seeds = rf_resonance_seeds(scan, nu_rf=nu_rf, a_mu=a_mu, a_p=a_p)
    parameters = ParameterSet()
    for name in composite.param_names:
        parameters.add(
            Parameter(
                name=name,
                value=float(seeds.get(name, composite.param_defaults.get(name, 0.0))),
                fixed=(name == "nu_RF" and fix_nu_rf),
            )
        )
    return fit_scan_model(
        scan, composite, parameters=parameters, x_min=x_min, x_max=x_max, method=method
    )


@dataclass(frozen=True)
class ScanBaselineResult:
    """Baseline estimate and first-order correction of independent scan errors.

    The correction operator includes the covariance between the baseline
    estimate and observations in its fitting regions. Marginal errors alone
    do not describe the correlations in ``corrected_covariance``.
    """

    scan: FieldScan
    model: ParameterCompositeModel
    regions: tuple[tuple[float, float], ...]
    fit: ParameterModelFitResult

    @property
    def success(self) -> bool:
        return bool(self.fit.success)

    @property
    def selection(self) -> NDArray[np.bool_]:
        mask = np.zeros(self.scan.x.size, dtype=bool)
        for lo, hi in self.regions:
            mask |= (self.scan.x >= lo) & (self.scan.x <= hi)
        return (
            mask
            & np.isfinite(self.scan.value)
            & np.isfinite(self.scan.error)
            & (self.scan.error > 0)
        )

    @property
    def n_points(self) -> int:
        return int(np.count_nonzero(self.selection))

    @property
    def baseline(self) -> NDArray[np.float64]:
        return _evaluate_model(self.model, self.fit.parameters, self.scan.x)

    @property
    def correction_operator(self) -> NDArray[np.float64]:
        names, covariance = _fit_covariance(self.fit)
        design = _model_jacobian(self.model, self.fit.parameters, self.scan.x, names, covariance)
        mask = self.selection
        weights = stabilized_point_errors(self.scan.error[mask]) ** -2
        operator = np.eye(self.scan.x.size)
        operator[:, mask] -= design @ covariance @ (design[mask].T * weights)
        return operator

    @property
    def corrected_covariance(self) -> NDArray[np.float64]:
        residual = self.correction_operator * self.scan.error
        return residual @ residual.T

    @property
    def corrected(self) -> FieldScan:
        return replace(
            self.scan,
            value=self.scan.value - self.baseline,
            error=np.sqrt(np.diag(self.corrected_covariance)),
            y_label=self.scan.y_label.removesuffix(" (baseline-subtracted)")
            + " (baseline-subtracted)",
        )


@dataclass(frozen=True)
class ScanPeakFit:
    """Two-stage peak estimator with baseline-propagated covariance.

    ``fit`` retains the conditional Minuit result, weighted by source errors.
    The propagated covariance is first order, conditional on the chosen
    baseline model and regions; it does not account for peak leakage or bias.
    """

    fit: ParameterModelFitResult
    covariance: tuple[list[str], list[list[float]]]
    effective_dof: float

    @property
    def uncertainties(self) -> dict[str, float]:
        names, matrix = self.covariance
        return dict(zip(names, np.sqrt(np.diag(matrix)), strict=True))

    @property
    def reduced_chi_squared(self) -> float:
        return self.fit.chi_squared / self.effective_dof if self.effective_dof > 0 else float("nan")


def fit_scan_peaks(
    baseline: ScanBaselineResult,
    model: str | list[str] | ParameterCompositeModel,
    *,
    parameters: ParameterSet | None = None,
    initial: dict[str, float] | None = None,
    extra_starts: int = 0,
) -> ScanPeakFit:
    """Fit peaks after subtraction, propagating the estimated baseline.

    Uses independent source scan errors for Gaussian least squares. Following
    two-stage estimating-equation propagation (Murphy & Topel, JBES 3, 370,
    1985), the covariance adds the baseline correction to the conditional
    Minuit covariance. The fixed-baseline limit preserves that covariance.
    """
    scan = baseline.scan
    if not (
        np.all(np.isfinite(scan.x))
        and np.all(np.isfinite(scan.value))
        and np.all(np.isfinite(scan.error))
        and np.all(scan.error > 0)
    ):
        raise ValueError(
            "Peak propagation requires finite observations and positive source errors."
        )
    composite = as_composite_model(model)
    corrected = replace(scan, value=scan.value - baseline.baseline)
    fit = fit_scan_model(
        corrected, composite, parameters=parameters, initial=initial, extra_starts=extra_starts
    )
    names, conditional = _fit_covariance(fit)
    jacobian = _model_jacobian(composite, fit.parameters, scan.x, names, conditional)
    errors = stabilized_point_errors(scan.error)
    weighted = jacobian / errors[:, None]
    # Column scaling keeps polynomial/peak parameters with different units
    # numerically comparable when solving the normal equations.
    scales = np.linalg.norm(weighted, axis=0)
    if np.any(scales == 0):
        raise ValueError("Peak Jacobian must have full column rank.")
    normalised = weighted / scales
    if np.linalg.matrix_rank(normalised) != len(names):
        raise ValueError("Peak Jacobian must have full column rank.")
    influence = np.linalg.solve(normalised.T @ normalised, normalised.T)
    influence = influence / scales[:, None] / errors
    corrected_covariance = baseline.corrected_covariance
    covariance = (
        conditional + influence @ (corrected_covariance - np.diag(scan.error**2)) @ influence.T
    )
    covariance = (covariance + covariance.T) / 2
    if not np.all(np.isfinite(covariance)) or np.linalg.eigvalsh(covariance).min() < 0:
        raise ValueError("Baseline-propagated peak covariance must be positive semidefinite.")
    residual = np.eye(scan.x.size) - jacobian @ influence
    effective_dof = float(
        np.trace((residual @ corrected_covariance @ residual.T) / errors[:, None] ** 2)
    )
    return ScanPeakFit(fit, (names, covariance.tolist()), effective_dof)


def fit_scan_baseline(
    scan: FieldScan,
    regions: list[tuple[float, float]],
    *,
    model: str | list[str] | ParameterCompositeModel = "Linear",
    parameters: ParameterSet | None = None,
    method: str = "migrad",
) -> ScanBaselineResult:
    """Fit a baseline over non-resonant *regions* and subtract it.

    Parameters
    ----------
    scan
        The field scan to correct.
    regions
        Inclusive ``(x_lo, x_hi)`` windows on the scan's x-axis marking the
        non-resonant baseline regions (their union is fitted).
    model
        Baseline model — ``"Linear"`` (default), ``"Constant"``, or any
        component/expression.
    parameters
        Optional starting :class:`ParameterSet`; defaults are used otherwise.

    Returns
    -------
    ScanBaselineResult
        The fitted baseline sampled over **all** scan points, a corrected
        :class:`FieldScan` (``value − baseline``, marginal propagated errors), and the fit.

    Raises
    ------
    ValueError
        If no valid regions are given; if the regions select fewer *usable*
        points (finite value, error > 0) than the baseline has free parameters
        (an underdetermined fit); or if the baseline fit does not converge. A
        failed fit never returns a (silently wrong) corrected scan.
    """
    composite = as_composite_model(model)
    clean_regions = _validate_regions(regions)

    x = np.asarray(scan.x, dtype=np.float64)
    value = np.asarray(scan.value, dtype=np.float64)
    error = np.asarray(scan.error, dtype=np.float64)

    in_region = np.zeros(x.size, dtype=bool)
    for lo, hi in clean_regions:
        in_region |= (x >= lo) & (x <= hi)
    # Count only the points the fitter will actually use — in a region AND
    # usable (finite value, positive error). fit_parameter_model re-masks on the
    # same criteria, so counting here keeps the underdetermined guard sound and
    # keeps the fit off its empty-input failure path.
    mask = in_region & np.isfinite(value) & np.isfinite(error) & (error > 0.0)

    n_sel = int(np.count_nonzero(mask))
    if n_sel == 0:
        raise ValueError("Baseline regions select no usable scan points (finite value, error > 0).")

    if parameters is None:
        parameters = parameter_set_for_model(composite)
    n_free = len(parameters.free_parameters)
    if n_free == 0:
        raise ValueError("Baseline fit needs at least one free parameter.")
    if n_sel < n_free:
        raise ValueError(
            f"Baseline fit needs at least {n_free} usable point(s) in the regions, got {n_sel}."
        )

    fit = fit_parameter_model(
        x[mask], value[mask], error[mask], composite, parameters, method=method
    )
    if not fit.success:
        raise ValueError(f"Baseline fit did not converge: {fit.message or 'unknown error'}")

    _fit_covariance(fit)
    snapshot = replace(
        scan,
        x=x.copy(),
        value=value.copy(),
        error=error.copy(),
        run_numbers=list(scan.run_numbers),
        excluded=list(scan.excluded),
    )
    for array in (snapshot.x, snapshot.value, snapshot.error):
        array.setflags(write=False)
    return ScanBaselineResult(scan=snapshot, model=composite, regions=tuple(clean_regions), fit=fit)


# --- internal -----------------------------------------------------------------


def _validate_regions(regions: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not regions:
        raise ValueError("At least one baseline region is required.")
    clean: list[tuple[float, float]] = []
    for region in regions:
        try:
            lo, hi = float(region[0]), float(region[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError(f"Invalid baseline region {region!r}; expected (x_lo, x_hi).") from exc
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo >= hi:
            raise ValueError(f"Baseline region {region!r} must have x_lo < x_hi and be finite.")
        clean.append((lo, hi))
    return clean


def _evaluate_model(
    model: ParameterCompositeModel,
    parameters: ParameterSet,
    x: NDArray[np.float64],
) -> NDArray[np.float64]:
    values = {param.name: float(param.value) for param in parameters}
    return np.asarray(model.function(x, **values), dtype=np.float64)


def _model_jacobian(model, parameters, x, names, covariance) -> NDArray[np.float64]:
    """Five-point derivatives in parameter coordinates at the fitted model."""
    values = {param.name: float(param.value) for param in parameters}
    columns = []
    for name, variance in zip(names, np.diag(covariance), strict=True):
        value = values[name]
        scale = np.sqrt(variance) if variance > 0 else (abs(value) or 1.0)
        step = max(np.finfo(float).eps ** 0.2 * scale, 10 * np.spacing(abs(value)))
        samples = [
            np.asarray(model.function(x, **{**values, name: value + k * step}))
            for k in (-2, -1, 1, 2)
        ]
        columns.append((samples[0] - 8 * samples[1] + 8 * samples[2] - samples[3]) / (12 * step))
    return np.column_stack(columns)


def _fit_covariance(fit) -> tuple[list[str], NDArray[np.float64]]:
    """Require a converged fit with finite positive-semidefinite covariance."""
    if not fit.success or fit.covariance is None:
        raise ValueError("Baseline propagation requires a converged fit with covariance.")
    names, matrix = fit.covariance
    covariance = np.asarray(matrix, dtype=float)
    if (
        not names
        or covariance.shape != (len(names), len(names))
        or not np.all(np.isfinite(covariance))
        or not np.allclose(covariance, covariance.T)
        or np.linalg.eigvalsh(covariance).min() < 0
    ):
        raise ValueError("Fit covariance must be finite and positive semidefinite.")
    return names, covariance
