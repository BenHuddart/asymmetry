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


def seed_peak_parameters_from_dataset(
    dataset, model: CompositeModel, *, guard_bins: int = 3, guard_freq_mhz: float = 2.0
) -> dict[str, float]:
    """Return simple peak/background seeds for one displayed Fourier spectrum.

    The dominant-magnitude bin of a (Power)^1/2 spectrum is usually the
    DC/apodisation spike, not the physical precession peak, so the peak
    search excludes a low-frequency guard band before taking the argmax.
    The guard width is ``max(guard_bins * df, guard_freq_mhz)``, where ``df``
    is the spectrum's bin spacing (a proxy for ``1/T_obs``); it falls back to
    the unguarded global argmax if the guard would empty the search array.
    """
    x = np.asarray(getattr(dataset, "time", []), dtype=float)
    y = np.asarray(getattr(dataset, "asymmetry", []), dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size == 0 or y.size == 0:
        return {}

    candidates = np.arange(x.size)
    if x.size >= 2:
        df = float(np.median(np.diff(np.sort(x))))
        guard = max(guard_bins * df, guard_freq_mhz)
        guarded = np.flatnonzero(np.abs(x) >= guard)
        if guarded.size > 0:
            candidates = guarded
    # ``candidates`` is never empty here: it starts as the full (non-empty,
    # per the size check above) index range and is only narrowed when the
    # narrowed set itself is non-empty.
    candidate_mask = np.zeros(x.size, dtype=bool)
    candidate_mask[candidates] = True

    baseline = float(np.nanmedian(y[candidates]))
    peak_index = int(candidates[np.nanargmax(y[candidates])])

    peak_y = float(y[peak_index])
    height = max(peak_y - baseline, 1e-12)
    nu0 = float(x[peak_index])
    half_height = baseline + 0.5 * height
    # Restrict the half-max crossing search to the same guarded region as the
    # peak search — otherwise a DC/apodisation spike that also exceeds
    # half_height (common, since it dwarfs the physical peak) drags the FWHM
    # span out to the DC spike's edge, producing a wildly inflated seed even
    # though nu0 itself correctly skipped it.
    above = np.flatnonzero((y >= half_height) & candidate_mask)
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
