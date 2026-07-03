"""Tests for the Brandt field-dependent vortex-lattice line width sigma(B0).

Covers the analytic anchors (field factor limits, tie to the existing London
helper, powder factor, quadrature background, monotonicity, guards) and a
synthetic round-trip recovery of (lambda_ab, Bc2) through the registered
field-scope parameter-trend components.
"""

from __future__ import annotations

import numpy as np

from asymmetry.core.fitting import ParameterCompositeModel
from asymmetry.core.fitting.parameter_models import component_names_for_x, fit_parameter_model
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.fitting.sc import constants, models

# muon gyromagnetic ratio in us^-1 mT^-1 (= 2*pi*135.5 MHz/T), per Pratt 2009.
_GAMMA_MU_US_PER_MT = 0.8516


def test_field_factor_limits() -> None:
    assert float(models.brandt_field_factor(0.0)) == 1.0
    assert float(models.brandt_field_factor(1.0)) == 0.0
    # Clamped above Bc2 and for unphysical negative reduced field.
    assert float(models.brandt_field_factor(1.5)) == 0.0
    assert float(models.brandt_field_factor(-0.2)) == 1.0


def test_field_factor_monotonic_decrease() -> None:
    b = np.linspace(0.0, 1.0, 200)
    g = models.brandt_field_factor(b)
    assert np.all(np.diff(g) <= 1e-12)
    assert np.all((g >= 0.0) & (g <= 1.0))


def test_zero_field_limit_ties_to_london_helper() -> None:
    # At B0 -> 0 the bracket is 1 + 1.21 = 2.21 and g -> 1, so the model must
    # reproduce the existing field-independent London conversion exactly.
    lam = 195.0
    sigma0 = float(constants.lambda_nm_to_sigma_us(lam))
    model_value = float(models.brandt_field_width_sigma(1e-9, lam, 25.0))
    assert np.isclose(model_value, sigma0, rtol=1e-6)


def test_powder_is_single_crystal_over_sqrt3() -> None:
    lam, bc2 = 195.0, 25.0
    for b0 in (1e-3, 400.0, 4000.0):
        single = float(models.brandt_field_width_sigma(b0, lam, bc2, powder=False))
        powder = float(models.brandt_field_width_sigma(b0, lam, bc2, powder=True))
        assert np.isclose(single / powder, np.sqrt(3.0), rtol=1e-9)
    # The dedicated wrapper matches the powder=True keyword path.
    assert np.isclose(
        float(models.brandt_field_width_sigma_powder(400.0, lam, bc2)),
        float(models.brandt_field_width_sigma(400.0, lam, bc2, powder=True)),
    )


def test_background_adds_in_quadrature() -> None:
    lam, bc2 = 195.0, 25.0
    sigma_vl = float(models.brandt_field_width_sigma(400.0, lam, bc2, sigma_bg=0.0))
    sigma_tot = float(models.brandt_field_width_sigma(400.0, lam, bc2, sigma_bg=0.3))
    assert np.isclose(sigma_tot, np.hypot(sigma_vl, 0.3), rtol=1e-9)
    # Exactly at and above Bc2 only the background channel survives.
    bc2_gauss = bc2 * 1.0e4  # tesla -> gauss (1 T = 1e4 G)
    assert np.isclose(
        float(models.brandt_field_width_sigma(bc2_gauss, lam, bc2, sigma_bg=0.25)),
        0.25,
        rtol=1e-9,
    )


def test_powder_brms_plateau_matches_lifeas_figure() -> None:
    # Pratt 2009 Fig. 1: powder B_rms low-T plateau ~1.9 mT (195 nm) and
    # ~1.0-1.3 mT (244 nm). B_rms = sigma / gamma_mu.
    s195 = float(models.brandt_field_width_sigma(1e-9, 195.0, 25.0, powder=True))
    s244 = float(models.brandt_field_width_sigma(1e-9, 244.0, 25.0, powder=True))
    assert np.isclose(s195 / _GAMMA_MU_US_PER_MT, 1.91, atol=0.05)
    assert np.isclose(s244 / _GAMMA_MU_US_PER_MT, 1.22, atol=0.05)


def test_guards_against_nonpositive_inputs() -> None:
    # Non-positive lambda/Bc2 must stay finite (no NaN/inf in the residual path).
    for value in models.brandt_field_width_sigma(np.array([100.0, 400.0]), lambda_ab=0.0, Bc2=0.0):
        assert np.isfinite(value)
    arr = models.brandt_field_width_sigma(np.array([0.0, 100.0, 1e9]), 200.0, 0.5)
    assert np.all(np.isfinite(arr))
    assert arr.dtype == np.float64


def test_scalar_and_array_inputs_agree() -> None:
    fields = [100.0, 400.0, 1600.0]
    arr = models.brandt_field_width_sigma(np.asarray(fields), 200.0, 0.5)
    for value, field in zip(arr, fields, strict=True):
        assert np.isclose(value, float(models.brandt_field_width_sigma(field, 200.0, 0.5)))


def test_components_registered_in_field_scope() -> None:
    field_names = component_names_for_x("field")
    assert "SC_Brandt_VortexLattice" in field_names
    assert "SC_Brandt_VortexLattice_Powder" in field_names
    # These are field-domain models; they must not leak into the temperature scope.
    temperature_names = component_names_for_x("temperature")
    assert "SC_Brandt_VortexLattice" not in temperature_names


def _fit_recovery(powder: bool, component: str) -> None:
    rng = np.random.default_rng(12345)
    lam_true, bc2_true = 195.0, 0.5
    # Field grid (gauss) reaching b ~ 0.2 so Bc2 is constrained by the curvature.
    b0 = np.array([100.0, 200.0, 400.0, 800.0, 1600.0, 4000.0, 7000.0, 10000.0])
    truth = models.brandt_field_width_sigma(b0, lam_true, bc2_true, powder=powder)
    noise = 0.01
    y = truth + rng.normal(0.0, noise, size=b0.shape)
    yerr = np.full_like(b0, noise)

    model = ParameterCompositeModel([component])
    params = ParameterSet(
        [
            Parameter("lambda_ab", 150.0, min=0.0),
            Parameter("Bc2", 0.3, min=0.0),
            Parameter("sigma_bg", 0.0, min=0.0, fixed=True),
        ]
    )
    result = fit_parameter_model(b0, y, yerr, model, params, error_mode="column")
    assert result.success
    fitted = dict(zip(result.parameters.names, result.parameters.values_array(), strict=True))
    assert np.isclose(fitted["lambda_ab"], lam_true, rtol=0.03)
    assert np.isclose(fitted["Bc2"], bc2_true, rtol=0.15)


def test_synthetic_round_trip_single_crystal() -> None:
    _fit_recovery(powder=False, component="SC_Brandt_VortexLattice")


def test_synthetic_round_trip_powder() -> None:
    _fit_recovery(powder=True, component="SC_Brandt_VortexLattice_Powder")
