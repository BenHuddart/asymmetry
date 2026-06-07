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

from dataclasses import dataclass, field

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
    "ScanBaselineResult",
]


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
    baseline-corrected) scan; the resonance field and width are read from the
    returned :attr:`ParameterModelFitResult.parameters` (e.g. ``B0`` / ``Bwid``
    for ``GaussianLCR``).
    """
    composite = as_composite_model(model)
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


@dataclass
class ScanBaselineResult:
    """Outcome of fitting + subtracting a baseline from a field scan."""

    baseline: NDArray[np.float64]
    corrected: FieldScan
    model: ParameterCompositeModel
    regions: list[tuple[float, float]]
    fit: ParameterModelFitResult
    n_points: int
    excluded_regions: list[tuple[float, float]] = field(default_factory=list)

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
        If no valid regions are given, or the regions select fewer points than
        the model has parameters (an underdetermined baseline fit).
    """
    composite = as_composite_model(model)
    clean_regions = _validate_regions(regions)

    x = np.asarray(scan.x, dtype=np.float64)
    mask = np.zeros(x.size, dtype=bool)
    for lo, hi in clean_regions:
        mask |= (x >= lo) & (x <= hi)

    n_sel = int(np.count_nonzero(mask))
    if n_sel == 0:
        raise ValueError("Baseline regions select no scan points.")
    n_params = len(composite.param_names)
    if n_sel < n_params:
        raise ValueError(
            f"Baseline fit needs at least {n_params} point(s) in the regions, got {n_sel}."
        )

    if parameters is None:
        parameters = parameter_set_for_model(composite)

    fit = fit_parameter_model(
        x[mask],
        np.asarray(scan.value, dtype=np.float64)[mask],
        np.asarray(scan.error, dtype=np.float64)[mask],
        composite,
        parameters,
        method=method,
    )

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
    # Start from defaults so every parameter is present, then apply fitted values.
    values = dict(model.param_defaults)
    for param in parameters:
        if param.name in values:
            values[param.name] = float(param.value)
    return np.asarray(model.function(x, **values), dtype=np.float64)


def _subtract_baseline(scan: FieldScan, baseline: NDArray[np.float64]) -> FieldScan:
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
