"""Synthetic rate sweep for the dynamic Kubo–Toyabe documentation."""

import numpy as np

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.models import MODELS
from asymmetry.core.simulate_presets import DELTA_AG_PER_US

from .archetypes import _poisson_errors, _time_axis


def make_ag_dynamic_rate_scan(seed: int = 71) -> list[MuonDataset]:
    """Ag-width Gaussian fields with a synthetic reorientation-rate sweep.

    Δ=0.39 μs⁻¹ is the Ag nuclear-field archetype; the imposed rates demonstrate
    strong-collision physics and are not a measurement of Ag diffusion.
    """
    rng = np.random.default_rng(seed)
    time = _time_axis()
    model = MODELS["DynamicGaussianKT"].function
    datasets = []
    for index, rate in enumerate((0.0, 0.1, 0.5, 2.0, 20.0)):
        clean = model(time, A0=24.0, Delta=DELTA_AG_PER_US, nu=rate, B_L=0.0, baseline=0.3)
        error = _poisson_errors(clean, counts_per_bin=1.0e5)
        datasets.append(
            MuonDataset(
                time=time,
                asymmetry=clean + rng.normal(0.0, error),
                error=error,
                metadata={
                    "run_number": 5701 + index,
                    "title": f"Gaussian fields: ν={rate:g} MHz",
                    "comment": f"Rate = {rate:g} MHz",
                    "field": 0.0,
                    "temperature": 20.0,
                },
            )
        )
    return datasets
