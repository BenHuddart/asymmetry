"""Recover a known exponential relaxation rate from synthetic asymmetry."""

import numpy as np

from asymmetry.core.fitting import FitEngine, Parameter, ParameterSet
from asymmetry.core.fitting.models import exponential_relaxation


def main() -> None:
    time_us = np.linspace(0.0, 10.0, 201)
    sigma = np.full_like(time_us, 0.3)
    amplitude_percent = 24.0
    rate_per_us = 0.3
    signal = amplitude_percent * np.exp(-rate_per_us * time_us) + 2.0
    rng = np.random.default_rng(23)
    asymmetry = signal + rng.normal(0.0, sigma)

    parameters = ParameterSet(
        [
            Parameter("A0", 20.0, min=0.0),
            Parameter("Lambda", 0.2, min=0.0),
            Parameter("baseline", 1.0),
        ]
    )
    result = FitEngine().fit_arrays(time_us, asymmetry, sigma, exponential_relaxation, parameters)

    assert result.success, result.message
    fitted_rate = result.parameters["Lambda"].value
    np.testing.assert_allclose(fitted_rate, rate_per_us, atol=0.02, rtol=0.0)
    np.testing.assert_allclose(result.parameters["A0"].value, amplitude_percent, atol=0.5, rtol=0.0)
    assert np.all(np.isfinite(result.residuals))
    print(f"Relaxation rate = {fitted_rate:.3f} μs⁻¹")


if __name__ == "__main__":
    main()
