"""Fit synthetic asymmetry data with the built-in fit engine."""

from __future__ import annotations

import numpy as np

from asymmetry.core.data import MuonDataset
from asymmetry.core.fitting import FitEngine, MODELS, Parameter, ParameterSet


def main() -> None:
    rng = np.random.default_rng(21)
    time = np.linspace(0.0, 8.0, 240)
    model = MODELS["ExponentialRelaxation"]

    truth = {"A0": 24.0, "Lambda": 0.42, "baseline": 0.8}
    clean = model.function(time, **truth)
    error = np.full_like(time, 0.35)
    noise = rng.normal(0.0, error)

    dataset = MuonDataset(
        time=time,
        asymmetry=clean + noise,
        error=error,
        metadata={"run_number": 2001, "title": "Synthetic fit run"},
    )

    params = ParameterSet(
        [
            Parameter("A0", value=20.0, min=0.0, max=40.0),
            Parameter("Lambda", value=0.3, min=0.0, max=2.0),
            Parameter("baseline", value=0.0, min=-5.0, max=5.0),
        ]
    )

    result = FitEngine().fit(dataset, model.function, params)
    print(f"success = {result.success}")
    print(f"chi2_red = {result.reduced_chi_squared:.3f}")
    for param in result.parameters:
        sigma = result.uncertainties.get(param.name)
        if sigma is None:
            print(f"{param.name} = {param.value:.4f}")
        else:
            print(f"{param.name} = {param.value:.4f} +/- {sigma:.4f}")


if __name__ == "__main__":
    main()
