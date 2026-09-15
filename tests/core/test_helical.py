"""Tests for the exact helical field-distribution lineshape and its fit components."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.special import j0, struve

from asymmetry.core.fitting.component_tags import ComputationalCost, FieldGeometry, PhysicsClass
from asymmetry.core.fitting.composite import COMPONENTS, CompositeModel
from asymmetry.core.fitting.helical import helical_crystal_line, helical_line

POWDER = COMPONENTS["HelicalPowder"].function
CRYSTAL = COMPONENTS["HelicalCrystal"].function

#: Polarization orientation whose squared components along both ellipse axes are 1/3.
MAGIC_THETA = float(np.degrees(np.arccos(1.0 / np.sqrt(3.0))))


def _axes(theta_h: float, phi_h: float) -> tuple[float, float]:
    sin_theta = np.sin(np.radians(theta_h))
    return sin_theta * np.cos(np.radians(phi_h)), sin_theta * np.sin(np.radians(phi_h))


def _helix_phase_average(
    t: np.ndarray,
    frequency: float,
    ratio: float,
    phase: float,
    axes: tuple[float, float] | None = None,
    n: int = 8192,
) -> tuple[float, np.ndarray]:
    """Average over the helix phase of a field on the ellipse (f cos ψ, r f sin ψ).

    Returns the non-precessing fraction and the precessing line; without
    ``axes`` every field counts as precessing.
    """
    helix = (np.arange(n) + 0.5) * (2.0 * np.pi / n)
    along_max = np.cos(helix)
    along_min = ratio * np.sin(helix)
    magnitude = np.hypot(along_max, along_min)
    if axes is None:
        weight = np.zeros(n)
    else:
        a, c = axes
        weight = ((a * along_max + c * along_min) / magnitude) ** 2
    angles = np.multiply.outer(t, 2.0 * np.pi * frequency * magnitude) + phase
    return float(weight.mean()), np.mean((1.0 - weight) * np.cos(angles), axis=1)


def _field_squared_average(
    t: np.ndarray, frequency: float, ratio: float, axes: tuple[float, float], n: int = 60000
) -> tuple[float, np.ndarray]:
    """Zero-phase line from the arcsine density of B² — an independent route to small ratios."""
    theta = (np.arange(n) + 0.5) * (np.pi / n)
    r2 = ratio * ratio
    u = 0.5 * (1.0 + r2) + 0.5 * (1.0 - r2) * np.cos(theta)
    a, c = axes
    weight = (a * a - c * c * r2) / (1.0 - r2) + r2 * (c * c - a * a) / ((1.0 - r2) * u)
    cosines = np.cos(np.multiply.outer(t, 2.0 * np.pi * frequency * np.sqrt(u)))
    return float(weight.mean()), cosines @ (1.0 - weight) / n


# The window reaches Δω t ≈ 200, so each ratio exercises the early-time
# quadrature and, where its series is short enough, the Bessel recurrence.
TIMES = np.linspace(0.0, 6.0, 241)


@pytest.mark.parametrize("ratio", [0.05, 0.3, 0.46, 0.8, 0.999])
@pytest.mark.parametrize("phase", [0.0, 0.7])
def test_powder_line_is_the_helix_phase_average(ratio: float, phase: float) -> None:
    _, expected = _helix_phase_average(TIMES, 12.0, ratio, phase)
    assert np.allclose(helical_line(TIMES, 12.0, ratio, phase), expected, rtol=0.0, atol=1e-11)


@pytest.mark.parametrize(
    ("ratio", "theta_h", "phi_h"),
    [
        (0.05, 90.0, 0.0),
        (0.3, 35.0, 70.0),
        (0.46, 61.0, 20.0),
        (0.8, 90.0, 90.0),
        (0.999, 12.0, 40.0),
    ],
)
def test_crystal_line_is_the_helix_phase_average(
    ratio: float, theta_h: float, phi_h: float
) -> None:
    expected_fraction, expected_line = _helix_phase_average(
        TIMES, 12.0, ratio, -0.4, axes=_axes(theta_h, phi_h)
    )
    fraction, line = helical_crystal_line(TIMES, 12.0, ratio, -0.4, theta_h, phi_h)
    assert fraction == pytest.approx(expected_fraction, abs=1e-12)
    assert np.allclose(line, expected_line, rtol=0.0, atol=1e-11)


@pytest.mark.parametrize("ratio", [1e-5, 1e-3, 0.01])
def test_small_ratio_powder_line_matches_the_field_squared_reference(ratio: float) -> None:
    t = np.linspace(0.0, 4.0, 61)
    _, expected = _field_squared_average(t, 20.0, ratio, (1.0 / np.sqrt(3.0),) * 2)
    assert np.allclose(helical_line(t, 20.0, ratio, 0.0), 1.5 * expected, rtol=0.0, atol=1e-10)


# The crystal weight's 1/B² term varies on the scale B_min near the lower
# cut-off, which the B² reference only resolves for ratios well above 1/60000.
@pytest.mark.parametrize("ratio", [1e-3, 0.01])
def test_small_ratio_crystal_line_matches_the_field_squared_reference(ratio: float) -> None:
    t = np.linspace(0.0, 4.0, 61)
    expected_fraction, expected_line = _field_squared_average(t, 20.0, ratio, _axes(50.0, 30.0))
    fraction, line = helical_crystal_line(t, 20.0, ratio, 0.0, 50.0, 30.0)
    assert fraction == pytest.approx(expected_fraction, abs=1e-10)
    assert np.allclose(line, expected_line, rtol=0.0, atol=1e-10)


def test_non_precessing_fraction_has_its_closed_form() -> None:
    ratio = 0.37
    a, c = _axes(64.0, 25.0)
    fraction, _ = helical_crystal_line(np.zeros(1), 5.0, ratio, 0.0, 64.0, 25.0)
    assert fraction == pytest.approx((a * a + c * c * ratio) / (1.0 + ratio), abs=1e-15)


def test_zero_ratio_is_the_overhauser_transform() -> None:
    t = np.linspace(0.0, 3.0, 400)
    argument = 2.0 * np.pi * 7.0 * t
    assert np.allclose(helical_line(t, 7.0, 0.0, 0.0), j0(argument), atol=1e-15)
    expected = np.cos(0.6) * j0(argument) - np.sin(0.6) * struve(0, argument)
    assert np.allclose(helical_line(t, 7.0, 0.0, 0.6), expected, atol=1e-15)
    fraction, line = helical_crystal_line(t, 7.0, 0.0, 0.0, 40.0, 10.0)
    assert fraction == pytest.approx(_axes(40.0, 10.0)[0] ** 2)
    assert np.allclose(line, (1.0 - fraction) * j0(argument), atol=1e-15)


def test_ratio_below_the_closed_form_threshold_is_continuous_with_it() -> None:
    t = np.linspace(0.0, 10.0, 500)
    assert np.allclose(helical_line(t, 30.0, 2e-6, 0.3), helical_line(t, 30.0, 0.0, 0.3), atol=1e-8)
    assert np.allclose(
        helical_line(t, 30.0, 0.9e-6, 0.3), helical_line(t, 30.0, 2e-6, 0.3), atol=1e-8
    )


def test_unit_ratio_is_a_single_cosine() -> None:
    t = np.linspace(0.0, 2.0, 300)
    expected = np.cos(2.0 * np.pi * 9.0 * t + 0.25)
    assert np.allclose(helical_line(t, 9.0, 1.0, 0.25), expected, atol=1e-15)
    fraction, line = helical_crystal_line(t, 9.0, 1.0, 0.25, 90.0, 0.0)
    assert fraction == pytest.approx(0.5)
    assert np.allclose(line, 0.5 * expected, atol=1e-15)


def test_ratio_above_one_relabels_the_ellipse_axes() -> None:
    t = np.linspace(0.0, 5.0, 300)
    assert np.allclose(helical_line(t, 4.0, 2.5, 0.2), helical_line(t, 10.0, 0.4, 0.2), atol=1e-13)
    swapped = helical_crystal_line(t, 4.0, 2.5, 0.2, 70.0, 20.0)
    relabelled = helical_crystal_line(t, 10.0, 0.4, 0.2, 70.0, 70.0)
    assert swapped[0] == pytest.approx(relabelled[0], abs=1e-15)
    assert np.allclose(swapped[1], relabelled[1], atol=1e-13)


def test_arbitrary_time_order_and_sign() -> None:
    t = np.linspace(0.0, 6.0, 500)
    forward = helical_line(t, 15.0, 0.3, 0.4)
    shuffled = np.random.default_rng(5).permutation(t.size)
    assert np.allclose(helical_line(t[shuffled], 15.0, 0.3, 0.4), forward[shuffled], atol=1e-14)
    # cos(−x + φ) = cos(x − φ): negative times mirror the phase.
    assert np.allclose(helical_line(-t, 15.0, 0.3, -0.4), forward, atol=1e-14)


def test_magic_angle_crystal_reproduces_the_powder_split() -> None:
    t = np.linspace(0.0, 5.0, 200)
    fraction, line = helical_crystal_line(t, 20.0, 0.4, 0.0, MAGIC_THETA, 45.0)
    assert fraction == pytest.approx(1.0 / 3.0, abs=1e-15)
    assert np.allclose(line, 2.0 / 3.0 * helical_line(t, 20.0, 0.4, 0.0), atol=1e-13)
    powder = POWDER(t, A=18.0, frequency=20.0, ratio=0.4, phase=0.0, lambda_T=0.6, lambda_L=0.1)
    crystal = CRYSTAL(
        t,
        A=18.0,
        frequency=20.0,
        ratio=0.4,
        theta_h=MAGIC_THETA,
        phi_h=45.0,
        phase=0.0,
        lambda_T=0.6,
        lambda_L=0.1,
    )
    assert np.allclose(crystal, powder, atol=1e-12)


def test_polarization_along_the_major_axis_removes_the_upper_edge() -> None:
    # Fields at B_max lie along the polarization and do not precess, so the
    # precessing weight vanishes there and the line carries only the lower edge.
    _, line = helical_crystal_line(np.zeros(1), 10.0, 0.5, 0.0, 90.0, 0.0)
    fraction, _ = helical_crystal_line(np.zeros(1), 10.0, 0.5, 0.0, 90.0, 0.0)
    assert fraction == pytest.approx(1.0 / 1.5)
    assert line[0] == pytest.approx(1.0 - fraction)


@pytest.mark.parametrize("function", [POWDER, CRYSTAL])
def test_components_start_at_amplitude(function) -> None:
    params = {
        "A": 22.0,
        "frequency": 3.0,
        "ratio": 0.35,
        "phase": 0.0,
        "lambda_T": 0.8,
        "lambda_L": 0.2,
    }
    if function is CRYSTAL:
        params |= {"theta_h": 30.0, "phi_h": 60.0}
    assert function(np.array([0.0]), **params)[0] == pytest.approx(22.0)


def test_zero_ratio_powder_is_overhauser_powder() -> None:
    t = np.linspace(0.0, 4.0, 300)
    common = {"A": 20.0, "frequency": 2.5, "lambda_T": 0.4, "lambda_L": 0.05}
    helical = POWDER(t, ratio=0.0, phase=0.0, **common)
    overhauser = COMPONENTS["OverhauserPowder"].function(t, **common)
    assert np.allclose(helical, overhauser, atol=1e-13)


def test_registry_tags_and_fixed_phase() -> None:
    for name in ("HelicalPowder", "HelicalCrystal"):
        definition = COMPONENTS[name]
        assert definition.field_geometries == frozenset({FieldGeometry.ZF})
        assert definition.physics_classes == frozenset({PhysicsClass.MAGNETISM})
        assert definition.cost is ComputationalCost.MODERATE
        assert definition.fixed_params == ("phase",)
        assert definition.category == "Oscillation"


def test_composite_parsing_and_serialisation_round_trip() -> None:
    model = CompositeModel.from_expression("HelicalCrystal + Constant")
    assert {"theta_h", "phi_h", "ratio", "A_bg"} <= set(model.param_names)

    restored = CompositeModel.from_dict(json.loads(json.dumps(model.to_dict())))
    assert restored.component_names == model.component_names

    values = {name: 1.0 for name in model.param_names}
    assert np.all(np.isfinite(model.function(np.linspace(0.0, 5.0, 50), **values)))
