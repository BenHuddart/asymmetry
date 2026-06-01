"""Frequency-domain fitting helpers."""

from __future__ import annotations

import numpy as np

from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.utils.constants import GAUSS_TO_TESLA, MUON_GYROMAGNETIC_RATIO_MHZ_PER_T

FREQUENCY_COMPONENT_NAMES: tuple[str, ...] = (
    "GaussianPeak",
    "LorentzianPeak",
    "ConstantBackground",
    "LinearBackground",
)


def default_frequency_model() -> CompositeModel:
    """Return the default V1 frequency-domain peak model."""
    return CompositeModel(["GaussianPeak", "ConstantBackground"], operators=["+"])


def frequency_mhz_to_field_gauss(value_mhz: float) -> float:
    """Convert a muon precession frequency or width in MHz to Gauss."""
    scale = MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * GAUSS_TO_TESLA
    return float(value_mhz) / scale


def field_gauss_to_frequency_mhz(value_gauss: float) -> float:
    """Convert a magnetic field or width in Gauss to MHz."""
    scale = MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * GAUSS_TO_TESLA
    return float(value_gauss) * scale


def append_frequency_field_derived_parameters(
    parameters: ParameterSet,
    uncertainties: dict[str, float] | None = None,
) -> tuple[ParameterSet, dict[str, float]]:
    """Return a copy with derived centre/width field parameters appended.

    The canonical fitted quantities remain ``nu0`` and ``fwhm`` in MHz.  The
    derived ``B0`` and ``Bwid`` values are appended for trend tables and exports.
    """
    result = ParameterSet()
    result_uncertainties: dict[str, float] = dict(uncertainties or {})

    for parameter in parameters:
        result.add(
            Parameter(
                name=parameter.name,
                value=parameter.value,
                min=parameter.min,
                max=parameter.max,
                fixed=parameter.fixed,
            )
        )

    existing = {parameter.name for parameter in result}
    if "nu0" in parameters and "B0" not in existing:
        result.add(Parameter("B0", value=frequency_mhz_to_field_gauss(parameters["nu0"].value)))
        if "nu0" in result_uncertainties:
            result_uncertainties["B0"] = abs(
                frequency_mhz_to_field_gauss(result_uncertainties["nu0"])
            )
    if "fwhm" in parameters and "Bwid" not in existing:
        result.add(Parameter("Bwid", value=frequency_mhz_to_field_gauss(parameters["fwhm"].value)))
        if "fwhm" in result_uncertainties:
            result_uncertainties["Bwid"] = abs(
                frequency_mhz_to_field_gauss(result_uncertainties["fwhm"])
            )
    return result, result_uncertainties


def seed_peak_parameters_from_dataset(dataset, model: CompositeModel) -> dict[str, float]:
    """Return simple peak/background seeds for one displayed Fourier spectrum."""
    x = np.asarray(getattr(dataset, "time", []), dtype=float)
    y = np.asarray(getattr(dataset, "asymmetry", []), dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size == 0 or y.size == 0:
        return {}

    baseline = float(np.nanpercentile(y, 10.0))
    peak_index = int(np.nanargmax(y))
    peak_y = float(y[peak_index])
    height = max(peak_y - baseline, 1e-12)
    nu0 = float(x[peak_index])
    half_height = baseline + 0.5 * height
    above = np.flatnonzero(y >= half_height)
    if above.size >= 2:
        fwhm = max(float(x[above[-1]] - x[above[0]]), np.finfo(float).eps)
    elif x.size >= 2:
        fwhm = max(float((np.nanmax(x) - np.nanmin(x)) / 20.0), np.finfo(float).eps)
    else:
        fwhm = 0.1

    seeds = {"height": height, "nu0": nu0, "fwhm": fwhm, "bg": baseline}
    if "slope" in model.param_names:
        edge = max(1, min(10, x.size // 10))
        dx = float(np.mean(x[-edge:]) - np.mean(x[:edge]))
        dy = float(np.mean(y[-edge:]) - np.mean(y[:edge]))
        seeds["slope"] = dy / dx if abs(dx) > 1e-12 else 0.0
    return {name: value for name, value in seeds.items() if name in model.param_names}


__all__ = [
    "FREQUENCY_COMPONENT_NAMES",
    "append_frequency_field_derived_parameters",
    "default_frequency_model",
    "field_gauss_to_frequency_mhz",
    "frequency_mhz_to_field_gauss",
    "seed_peak_parameters_from_dataset",
]
