r"""Fit baseline + peak models to an ALC / integral-asymmetry field scan.

The field scan (integral asymmetry vs a run variable) produced by
:func:`asymmetry.core.transform.build_field_scan` is fitted with the **existing**
parameter-model machinery — the same `fit_parameter_model` /
`ParameterCompositeModel` engine the trending panel uses — so no new fitter is
introduced.

This module provides the two-step ALC workflow chosen for the GUI port (see
``docs/porting/time-integral-asymmetry/implementation-options.md``):

1. :func:`fit_scan_baseline` — fit a baseline (Linear/Constant/…) over the
   user-marked **non-resonant regions**, then subtract it to yield a corrected
   scan. This mirrors Mantid's "Baseline modelling" step.
2. :func:`fit_scan_model` — fit a peak (e.g. ``GaussianLCR``, centred at the
   resonance field) to the corrected scan, reading off the resonance position
   and width. This mirrors Mantid's "Peak fitting" step.

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

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.fitting.parameter_models import (
    ParameterCompositeModel,
    ParameterModelFitResult,
    fit_parameter_model,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.transform import FieldScan

__all__ = [
    "parameter_set_for_model",
    "as_composite_model",
    "fit_scan_model",
    "fit_scan_baseline",
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
) -> ParameterModelFitResult:
    """Fit a parameter model to a field scan's ``(x, value, error)``.

    The thin adapter that lets a :class:`FieldScan` flow into the existing
    ``fit_parameter_model``. Used for the peak-fit step on a (typically
    baseline-corrected) scan; **check ``result.success`` before** reading the
    resonance field and width from :attr:`ParameterModelFitResult.parameters`
    (e.g. ``B0`` / ``Bwid`` for ``GaussianLCR``) — on a failed fit the parameter
    set may be empty.

    Pass starting values via *either* an explicit ``parameters`` set *or* an
    ``initial`` override dict, not both.
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
    (defaults 515/124 MHz put the dips near the benzene 866/772 G). The peak
    amplitudes, widths and background are seeded **from the data** because the
    registered defaults assume a paper-graded dip depth that an integrated scan
    does not have: ``BG`` is the median value, ``ampl1 = ampl2`` is the signed
    largest excursion from it (so the sign follows whether the observable shows
    peaks or dips), and the widths are a small fraction of the field span. This
    makes the fit robust to the scan's units (fractional vs percent) and depth.
    """
    value = np.asarray(scan.value, dtype=np.float64)
    x = np.asarray(scan.x, dtype=np.float64)
    finite = np.isfinite(value)
    if np.any(finite):
        bg = float(np.median(value[finite]))
        deviations = value[finite] - bg
        ampl = float(deviations[int(np.argmax(np.abs(deviations)))])
    else:
        bg, ampl = 0.0, 0.0
    if ampl == 0.0:
        ampl = 1.0
    span = float(x.max() - x.min()) if x.size >= 2 else 0.0
    width = span / 20.0 if span > 0.0 else 25.0
    return {
        "A_mu": float(a_mu),
        "A_p": float(a_p),
        "nu_RF": float(nu_rf),
        "ampl1": ampl,
        "wid1": width,
        "ampl2": ampl,
        "wid2": width,
        "BG": bg,
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


@dataclass
class ScanBaselineResult:
    """Outcome of fitting + subtracting a baseline from a field scan."""

    baseline: NDArray[np.float64]
    corrected: FieldScan
    model: ParameterCompositeModel
    regions: list[tuple[float, float]]
    fit: ParameterModelFitResult
    n_points: int

    @property
    def success(self) -> bool:
        return bool(self.fit.success)


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
        non-resonant baseline regions (their union is fitted). Mantid's
        "Sections".
    model
        Baseline model — ``"Linear"`` (default), ``"Constant"``, or any
        component/expression.
    parameters
        Optional starting :class:`ParameterSet`; defaults are used otherwise.

    Returns
    -------
    ScanBaselineResult
        The fitted baseline sampled over **all** scan points, a corrected
        :class:`FieldScan` (``value − baseline``, errors preserved), and the fit.

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
    if n_sel < n_free:
        raise ValueError(
            f"Baseline fit needs at least {n_free} usable point(s) in the regions, got {n_sel}."
        )

    fit = fit_parameter_model(
        x[mask], value[mask], error[mask], composite, parameters, method=method
    )
    if not fit.success:
        raise ValueError(f"Baseline fit did not converge: {fit.message or 'unknown error'}")

    baseline = _evaluate_model(composite, fit.parameters, x)
    corrected = _subtract_baseline(scan, baseline)
    return ScanBaselineResult(
        baseline=baseline,
        corrected=corrected,
        model=composite,
        regions=clean_regions,
        fit=fit,
        n_points=n_sel,
    )


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
    # On the success path `parameters` already holds every model parameter; the
    # defaults merge is belt-and-suspenders so a partial set can never reach
    # model.function() (which would KeyError) — callers reach here only after a
    # converged fit.
    values = dict(model.param_defaults)
    for param in parameters:
        if param.name in values:
            values[param.name] = float(param.value)
    return np.asarray(model.function(x, **values), dtype=np.float64)


def _subtract_baseline(scan: FieldScan, baseline: NDArray[np.float64]) -> FieldScan:
    # Errors carry through unchanged: the baseline fit's covariance is
    # intentionally not propagated into the corrected curve (matching Mantid's
    # ALC baseline subtraction). Downstream peak-fit error bars are therefore
    # mildly optimistic where the baseline is extrapolated outside its regions.
    corrected_value = np.asarray(scan.value, dtype=np.float64) - baseline
    y_label = scan.y_label
    suffix = " (baseline-subtracted)"
    if not y_label.endswith(suffix):
        y_label = f"{y_label}{suffix}"
    return FieldScan(
        x=np.asarray(scan.x, dtype=np.float64).copy(),
        value=corrected_value,
        error=np.asarray(scan.error, dtype=np.float64).copy(),
        run_numbers=list(scan.run_numbers),
        order_key=scan.order_key,
        method=scan.method,
        derivative=scan.derivative,
        x_label=scan.x_label,
        y_label=y_label,
        excluded=list(scan.excluded),
    )
