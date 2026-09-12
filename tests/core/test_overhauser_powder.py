"""Tests for the powder Overhauser and two-cut-off Overhauser fit components."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.special import j0

from asymmetry.core.fitting.composite import COMPONENTS, CompositeModel

PLAIN = COMPONENTS["OverhauserPowder"].function
CENTRE = COMPONENTS["OverhauserPowderCentre"].function
CUTOFF = COMPONENTS["OverhauserPowderCutoff"].function


def _exact_two_cutoff_transform(t: np.ndarray, f_max: float, ratio: float) -> np.ndarray:
    """Cosine transform of (2/pi) B / [sqrt(B^2-B_min^2) sqrt(B_max^2-B^2)]."""
    f_min = ratio * f_max
    u_mid = (f_max**2 + f_min**2) / 2.0
    u_half = (f_max**2 - f_min**2) / 2.0
    theta = np.linspace(0.0, np.pi, 20001)
    freqs = np.sqrt(u_mid + u_half * np.cos(theta))
    return np.trapezoid(np.cos(2.0 * np.pi * np.outer(t, freqs)), theta, axis=1) / np.pi


@pytest.mark.parametrize(
    ("function", "extra"),
    [
        (PLAIN, {}),
        (CENTRE, {"delta_frequency": 0.3, "phase": 0.0}),
        (CUTOFF, {"ratio": 0.4, "phase": 0.0}),
    ],
)
def test_all_components_start_at_amplitude(function, extra: dict[str, float]) -> None:
    t = np.array([0.0])
    out = function(t, A=25.0, frequency=1.4, lambda_T=0.7, lambda_L=0.2, **extra)
    assert out[0] == pytest.approx(25.0)


@pytest.mark.parametrize("f_max", [0.3, 1.0, 4.0])
@pytest.mark.parametrize("ratio", [0.0, 0.25, 0.5, 0.9, 1.0])
def test_cutoff_and_centre_parametrisations_agree(f_max: float, ratio: float) -> None:
    t = np.linspace(0.0, 8.0, 401)
    cutoff = CUTOFF(t, A=18.0, frequency=f_max, ratio=ratio, phase=0.4, lambda_T=0.6, lambda_L=0.15)
    centre = CENTRE(
        t,
        A=18.0,
        frequency=f_max * (1.0 + ratio) / 2.0,
        delta_frequency=f_max * (1.0 - ratio) / 2.0,
        phase=0.4,
        lambda_T=0.6,
        lambda_L=0.15,
    )
    assert np.allclose(cutoff, centre, atol=1e-12)


def test_unit_ratio_reduces_to_a_single_cosine() -> None:
    t = np.linspace(0.0, 6.0, 301)
    out = CUTOFF(t, A=20.0, frequency=1.3, ratio=1.0, phase=0.4, lambda_T=0.6, lambda_L=0.15)
    expected = 20.0 * (
        np.exp(-0.15 * t) / 3.0 + 2.0 / 3.0 * np.cos(2.0 * np.pi * 1.3 * t + 0.4) * np.exp(-0.6 * t)
    )
    assert np.allclose(out, expected, atol=1e-12)


def test_first_minimum_sits_at_the_first_bessel_minimum() -> None:
    freq = 1.0
    t = np.linspace(0.0, 2.0, 200001)
    out = PLAIN(t, A=30.0, frequency=freq, lambda_T=0.0, lambda_L=0.0)
    argument = 2.0 * np.pi * freq * t[int(np.argmin(out))]
    assert argument == pytest.approx(3.8317, abs=1e-4)
    assert out.min() == pytest.approx(30.0 * (1.0 / 3.0 + 2.0 / 3.0 * j0(3.8317)), abs=1e-6)


def test_undamped_precessing_part_is_the_overhauser_transform() -> None:
    freq = 0.7
    t = np.linspace(0.0, 8.0, 50)
    phi = np.linspace(0.0, np.pi, 20001)  # B = B_max cos(phi) substitution
    transform = (
        np.trapezoid(np.cos(2.0 * np.pi * freq * np.outer(t, np.cos(phi))), phi, axis=1) / np.pi
    )
    out = PLAIN(t, A=1.0, frequency=freq, lambda_T=0.0, lambda_L=0.0)
    assert np.allclose(out, 1.0 / 3.0 + 2.0 / 3.0 * transform, atol=1e-10)


# The closed form is Amato's narrow-distribution approximation to the exact
# two-cut-off density, so it converges on the exact transform as r -> 1.
@pytest.mark.parametrize(
    ("ratio", "tolerance"),
    [(0.0, 0.32), (0.2, 0.18), (0.4, 0.13), (0.6, 0.08), (0.8, 0.035)],
)
def test_closed_form_approximates_the_exact_two_cutoff_transform(
    ratio: float, tolerance: float
) -> None:
    t = np.linspace(0.0, 10.0, 2001)
    out = CUTOFF(t, A=1.0, frequency=1.0, ratio=ratio, phase=0.0, lambda_T=0.0, lambda_L=0.0)
    precessing = (out - 1.0 / 3.0) * 3.0 / 2.0
    assert np.max(np.abs(precessing - _exact_two_cutoff_transform(t, 1.0, ratio))) < tolerance


def test_composite_parsing_and_serialisation_round_trip() -> None:
    model = CompositeModel.from_expression("OverhauserPowderCutoff + Constant")

    assert {"ratio", "lambda_T", "lambda_L", "A_bg"} <= set(model.param_names)

    restored = CompositeModel.from_dict(model.to_dict())
    assert restored.component_names == model.component_names
    assert restored.operators == model.operators

    values = {name: 1.0 for name in model.param_names}
    assert np.all(np.isfinite(model.function(np.linspace(0.0, 5.0, 50), **values)))


def test_centre_component_survives_fit_slot_round_trip() -> None:
    from asymmetry.core.representation.base import FitSlot

    model = CompositeModel(["OverhauserPowderCentre", "Constant"], operators=["+"])
    slot = FitSlot(
        model=model.to_dict(),
        parameters=[{"name": "delta_frequency", "value": 0.42}],
        provenance="single",
    )
    restored = FitSlot.from_dict(json.loads(json.dumps(slot.to_dict())))

    assert CompositeModel.from_dict(restored.model).component_names == [
        "OverhauserPowderCentre",
        "Constant",
    ]
    assert restored.parameters[0]["value"] == pytest.approx(0.42)
