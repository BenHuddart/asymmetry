"""Stored explicit capture lifetimes survive corrected authority defaults."""

import json

import numpy as np

from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.fitting.result_payload import _deserialize_fit_result, _serialize_fit_result
from asymmetry.core.negmu.fit import (
    CaptureModelSpec,
    default_capture_parameters,
    fit_capture_histogram,
)
from asymmetry.core.negmu.model import CaptureComponent, build_capture_count_model
from asymmetry.core.project.json_values import decode_non_finite, encode_non_finite


def test_stored_fixed_lifetime_roundtrip_controls_evaluation_and_refit():
    old_tau = 0.0885
    spec = CaptureModelSpec(elements=("Pd",), include_decay_background=False)
    assert spec.components()[0].tau_us == 0.096
    stored_component = CaptureComponent("Pd", old_tau)
    assert stored_component.tau_us == old_tau
    time = np.linspace(0, 0.6, 120)
    counts = 1300 * np.exp(-time / old_tau) + 5
    parameters = ParameterSet()
    parameters.add(Parameter("amp_Pd", 1200, min=0))
    parameters.add(Parameter("tau_Pd", old_tau, fixed=True))
    parameters.add(Parameter("background", 3, min=0))
    fitted = fit_capture_histogram(time, counts, spec, parameters=parameters, cost="gaussian")
    assert fitted.success
    # A stored fixed lifetime can carry explicitly authored bounds in microseconds.
    fitted.parameters["tau_Pd"].min = old_tau / 2
    fitted.parameters["tau_Pd"].max = old_tau * 2
    payload = json.loads(
        json.dumps(encode_non_finite(_serialize_fit_result(fitted)), allow_nan=False)
    )
    restored = _deserialize_fit_result(decode_non_finite(payload))
    assert restored.parameters["tau_Pd"].value == old_tau
    assert restored.parameters["tau_Pd"].fixed
    assert restored.parameters["tau_Pd"].min == old_tau / 2
    assert restored.parameters["tau_Pd"].max == old_tau * 2
    values = {parameter.name: parameter.value for parameter in restored.parameters}
    replay = build_capture_count_model(spec.components())(time, **values)
    assert np.allclose(replay, counts, rtol=1e-4, atol=0.01)
    refitted = fit_capture_histogram(
        time, counts, spec, parameters=restored.parameters, cost="gaussian"
    )
    assert refitted.success
    assert refitted.parameters["tau_Pd"].value == old_tau
    assert refitted.parameters["tau_Pd"].fixed


def test_explicit_seed_overrides_new_fixed_lifetime_default():
    time = np.linspace(0, 0.6, 120)
    counts = np.ones(120)
    spec = CaptureModelSpec(elements=("Pd",), include_decay_background=False)
    defaults = default_capture_parameters(spec, time=time, counts=counts)
    retained = default_capture_parameters(spec, time=time, counts=counts, seeds={"tau_Pd": 0.0885})
    assert defaults["tau_Pd"].value == 0.096
    assert retained["tau_Pd"].value == 0.0885
    assert defaults["tau_Pd"].fixed and retained["tau_Pd"].fixed
