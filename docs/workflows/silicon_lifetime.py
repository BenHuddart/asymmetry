"""Replay the school silicon example from tab-separated Asymmetry fit results.

Usage: python silicon_lifetime.py school-rates.tsv
Columns: run, lambda_per_us, sigma_lambda_per_us (with a header).
Method: Yokoyama et al., Phys. Rev. Lett. 119, 226601 (2017), Fig. 1(d,e).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from asymmetry.core.fitting.parameter_models import (
    ParameterCompositeModel,
    ParameterModelFitResult,
    fit_parameter_model,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet

# Run coordinates from the Rigi school Carrier Recombination in Silicon guide.
CALIBRATION_RUNS = {
    103277: 8.9e13,
    103278: 7.9e13,
    103279: 7.1e13,
    103280: 6.3e13,
    103281: 5.6e13,
    103282: 4.7e13,
    103283: 3.7e13,
    103284: 2.7e13,
    103285: 1.8e13,
    103286: 9.3e12,
}
DELAY_RUNS = {
    103287: 0.1,
    103288: 3.0,
    103289: 5.0,
    103290: 10.0,
    103291: 15.0,
    103292: 20.0,
    103293: 25.0,
    103294: 30.0,
    103295: 40.0,
    103296: 50.0,
    103297: 60.0,
    103298: 70.0,
}

REFERENCE_DENSITY_CM3 = CALIBRATION_RUNS[103277]
DENSITY_UNIT_CM3 = 1e13
MAX_DELAY_US = 20.0
CALIBRATION_MODEL = ParameterCompositeModel(["PowerLaw"])
DECAY_MODEL = ParameterCompositeModel(["ExponentialDecay"])


@dataclass(frozen=True)
class CarrierDecay:
    """Densities and rate-only errors in 10¹³ cm⁻³, with the fitted lifetime."""

    density: np.ndarray
    error: np.ndarray
    fit: ParameterModelFitResult
    calibration_tau_error: float


def _require_rate_errors(rate_error: np.ndarray) -> None:
    if np.any(~np.isfinite(rate_error) | (rate_error <= 0)):
        raise ValueError("The silicon analysis requires positive finite rate standard errors")


def fit_calibration(
    density_cm3: np.ndarray, rate: np.ndarray, rate_error: np.ndarray
) -> ParameterModelFitResult:
    """Fit lambda = a * (density / reference density)**n with c fixed at zero."""
    _require_rate_errors(rate_error)
    result = fit_parameter_model(
        density_cm3 / REFERENCE_DENSITY_CM3,
        rate,
        rate_error,
        CALIBRATION_MODEL,
        ParameterSet(
            [
                Parameter("a", 1.4, min=0.0),
                Parameter("n", 0.68, min=0.0),
                Parameter("c", 0.0, fixed=True),
            ]
        ),
    )
    if not result.success or result.covariance is None:
        raise RuntimeError(
            f"Silicon calibration needs a converged fit with covariance ({result.message})"
        )
    return result


def fit_carrier_decay(
    calibration: ParameterModelFitResult,
    delay_us: np.ndarray,
    rate: np.ndarray,
    rate_error: np.ndarray,
    *,
    min_delay_us: float = 0.0,
    max_delay_us: float = MAX_DELAY_US,
) -> CarrierDecay:
    """Invert the calibration and refit after shifts along its covariance axes."""
    _require_rate_errors(rate_error)
    names, covariance = calibration.covariance
    coefficients = np.array([calibration.parameters[name].value for name in names])
    offset = calibration.parameters["c"].value
    excess_rate = rate - offset
    if np.any(~np.isfinite(excess_rate) | (excess_rate <= 0)):
        raise ValueError("All delay rates must be finite and exceed the calibration offset")

    def invert_and_fit(
        values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, ParameterModelFitResult]:
        beta, exponent = (values[names.index(name)] for name in ("a", "n"))
        if beta <= 0 or exponent <= 0:
            raise ValueError("Calibration prefactor and exponent must be positive")
        density = (REFERENCE_DENSITY_CM3 / DENSITY_UNIT_CM3) * (excess_rate / beta) ** (
            1 / exponent
        )
        # d(density)/d(rate) from the inversion; calibration held fixed.
        error = density * rate_error / (exponent * excess_rate)
        result = fit_parameter_model(
            delay_us,
            density,
            error,
            DECAY_MODEL,
            ParameterSet(
                [
                    Parameter("a", float(density.max()), min=0.0),
                    Parameter("tau", 11.0, min=0.0),
                    Parameter("c", 0.0, fixed=True),
                ]
            ),
            x_min=min_delay_us,
            x_max=max_delay_us,
        )
        if not result.success or result.covariance is None:
            raise RuntimeError(
                f"Silicon decay needs a converged fit with covariance ({result.message})"
            )
        return density, error, result

    density, error, fit = invert_and_fit(coefficients)
    variances, axes = np.linalg.eigh(covariance)
    if np.any(variances <= 0):
        raise RuntimeError("Silicon calibration covariance must be positive definite")
    tau_shifts = []
    for shift in (axes * np.sqrt(variances)).T:
        # Each shift applies the same calibration to every delay.
        lower, upper = (
            invert_and_fit(coefficients + sign * shift)[2].parameters["tau"].value
            for sign in (-1, 1)
        )
        tau_shifts.append((upper - lower) / 2)
    return CarrierDecay(density, error, fit, float(np.linalg.norm(tau_shifts)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rates_tsv", type=Path)
    args = parser.parse_args()
    runs, rates, errors = np.loadtxt(
        args.rates_tsv, delimiter="\t", skiprows=1, unpack=True, ndmin=2
    )
    by_run = dict(zip(runs, zip(rates, errors, strict=True), strict=True))
    if len(by_run) != len(runs) or by_run.keys() != CALIBRATION_RUNS.keys() | DELAY_RUNS.keys():
        parser.error("supply one illuminated rate and standard error for each run 103277–103298")
    cal_rate, cal_error = np.array([by_run[run] for run in CALIBRATION_RUNS]).T
    delay_us = np.array(list(DELAY_RUNS.values()))
    delay_rate, delay_error = np.array([by_run[run] for run in DELAY_RUNS]).T
    calibration = fit_calibration(np.array(list(CALIBRATION_RUNS.values())), cal_rate, cal_error)
    decay = fit_carrier_decay(calibration, delay_us, delay_rate, delay_error)
    without_first = fit_carrier_decay(
        calibration, delay_us, delay_rate, delay_error, min_delay_us=3.0
    )
    print("Calibration prefactor (µs⁻¹), exponent, offset (µs⁻¹):")
    print(*(calibration.parameters[name].value for name in ("a", "n", "c")))
    print("Calibration covariance (names, matrix):", calibration.covariance)
    print("Delay (µs), density and rate-only standard error (10¹³ cm⁻³):")
    print(np.column_stack((delay_us, decay.density, decay.error)))
    print("Delay at most", MAX_DELAY_US, "µs enters the lifetime fit.")
    print("Initial density (10¹³ cm⁻³):", decay.fit.parameters["a"].value)
    print("Lifetime (µs):", decay.fit.parameters["tau"].value)
    print("Rate-only lifetime standard error (µs):", decay.fit.uncertainties["tau"])
    print("Shared calibration lifetime sensitivity (µs):", decay.calibration_tau_error)
    print("Lifetime for delays of 3–20 µs (µs):", without_first.fit.parameters["tau"].value)


if __name__ == "__main__":
    main()
