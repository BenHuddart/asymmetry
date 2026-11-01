"""Independent Gaussian-characteristic-function and Brandt width checks."""

import numpy as np
import pytest
from scipy.integrate import quad

from asymmetry.core.fitting.models import gaussian_relaxation
from asymmetry.core.fitting.sc.constants import lambda_nm_to_sigma_us, sigma_to_lambda_nm
from asymmetry.core.utils.constants import MUON_GYROMAGNETIC_RATIO_MHZ_PER_T


@pytest.mark.parametrize("function", [sigma_to_lambda_nm, lambda_nm_to_sigma_us])
def test_width_conversion_requires_an_explicit_convention(function):
    with pytest.raises(TypeError, match="width"):
        function(1.0)


def test_gaussian_field_distribution_fixes_sqrt2():
    from asymmetry.core.fitting.sc import GaussianWidth

    field_rms_tesla = 0.001
    angular_width = 2 * np.pi * MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * field_rms_tesla
    for time in (0.1, 0.4, 1.2):
        characteristic, _ = quad(
            lambda z: np.exp(-z * z / 2) * np.cos(angular_width * time * z) / np.sqrt(2 * np.pi),
            -np.inf,
            np.inf,
            epsabs=1e-13,
            epsrel=1e-13,
        )
        fitted_sigma = np.sqrt(-np.log(characteristic)) / time
        assert angular_width / fitted_sigma == pytest.approx(np.sqrt(2), rel=1e-10)
        assert GaussianWidth.GAUSSIAN_COMPONENT.value == pytest.approx(
            angular_width / fitted_sigma, rel=1e-10
        )
        assert gaussian_relaxation(time, 1.0, fitted_sigma) == pytest.approx(
            characteristic, abs=1e-12
        )


def test_workflow_relation_matches_library_after_conversion():
    from asymmetry.core.fitting.sc import GaussianWidth

    sigma_g = 1.164
    gaussian_lambda = sigma_to_lambda_nm(sigma_g, width=GaussianWidth.GAUSSIAN_COMPONENT)
    second_moment_lambda = sigma_to_lambda_nm(sigma_g, width=GaussianWidth.SECOND_MOMENT)
    # Rounded teaching-guide coefficient, for exp(-(sigma_G*t)^2).
    assert gaussian_lambda == pytest.approx(np.sqrt(75780 / sigma_g), rel=1e-3)
    assert second_moment_lambda / gaussian_lambda == pytest.approx(2**0.25, rel=1e-12)


@pytest.mark.parametrize("width_name", ["SECOND_MOMENT", "GAUSSIAN_COMPONENT"])
def test_width_conversion_round_trip(width_name):
    from asymmetry.core.fitting.sc import GaussianWidth

    width = GaussianWidth[width_name]
    sigma = np.array([0.2, 1.0, 2.0])
    penetration = sigma_to_lambda_nm(sigma, width=width)
    np.testing.assert_allclose(lambda_nm_to_sigma_us(penetration, width=width), sigma, rtol=1e-14)


@pytest.mark.parametrize("function", [sigma_to_lambda_nm, lambda_nm_to_sigma_us])
def test_unknown_width_convention_is_rejected(function):
    with pytest.raises(ValueError, match="GaussianWidth"):
        function(1.0, width="unrecognised")


@pytest.mark.parametrize("name", ["SC_Brandt_VortexLattice", "SC_Brandt_VortexLattice_Powder"])
def test_brandt_help_identifies_second_moment_input(name):
    from asymmetry.core.fitting.component_docs import (
        get_component_applicability,
        get_component_references,
    )

    help_text = get_component_applicability(name)
    assert "second-moment" in help_text
    assert "√2" in help_text
    assert get_component_references(name)
