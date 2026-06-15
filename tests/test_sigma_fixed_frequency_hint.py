"""RED target for branch ``fix/sigma-fixed-freq-hint``.

Round-2 GUI finding (BiSCCO, ``_findings/windows-gui/BiSCCO_penetration_depth.md``):
a TF Gaussian fit that *fixes* the precession frequency at the nominal applied
field inflates the fitted ``sigma`` by ~8%, because below T_c the vortex-state
diamagnetic shift moves the true precession frequency (e.g. 5.42 MHz nominal at
400 G vs 5.24 MHz actual at 10 K). Letting ``frequency`` float removes the bias.

Desired behaviour: when a fit pins ``frequency`` (``fixed=True``) to a value that
sits far (say >2%) from gamma_mu * B implied by the run's ``field`` metadata, the
engine should emit a discoverable ``UserWarning`` pointing the user to float it
(mirrors the discoverability-via-pointing-warning convention, not prose docs).

No such warning exists yet, so ``pytest.warns`` below fails -> this test is RED
until ``fix/sigma-fixed-freq-hint`` lands. Remove nothing here when fixing; make
it pass by emitting the warning.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.engine import FitEngine
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.fourier.units import gauss_to_mhz

pytest.importorskip("iminuit")


def _tf_dataset(field_gauss: float = 400.0) -> MuonDataset:
    """Synthetic TF asymmetry oscillating at gamma_mu * B with Gaussian damping."""
    t = np.linspace(0.05, 8.0, 400)
    f_true = float(gauss_to_mhz(field_gauss))  # ~5.42 MHz at 400 G
    asym = 9.0 * np.cos(2 * np.pi * f_true * t) * np.exp(-((1.2 * t) ** 2)) - 23.0
    err = np.full_like(t, 0.3)
    return MuonDataset(t, asym, err, {"field": field_gauss, "run_number": 1277})


def _seeded_params(model) -> ParameterSet:
    """Seed the composite, pinning the frequency ~10% off gamma_mu * B (the trap)."""
    params: list[Parameter] = []
    for name in model.param_names:
        low = name.lower()
        is_freq = "freq" in low
        if is_freq:
            value = 6.0  # far from 5.42 MHz at 400 G
        elif low.startswith("a_1") or low == "a1":
            value = 9.0
        elif "sigma" in low:
            value = 1.0
        elif "phase" in low:
            value = 0.0
        elif "bg" in low:
            value = -23.0
        else:
            value = float(model.param_defaults.get(name, 1.0))
        params.append(Parameter(name, value, fixed=is_freq))
    return ParameterSet(params)


def test_fixed_frequency_far_from_field_warns() -> None:
    dataset = _tf_dataset(400.0)
    model = CompositeModel.from_expression(
        "Oscillatory * Gaussian + Constant"
    ).to_model_definition()
    params = _seeded_params(model)

    with pytest.warns(UserWarning, match=r"(?i)frequenc|field|gamma|γ"):
        FitEngine().fit(dataset, model.function, params, 0.05, 8.0)
