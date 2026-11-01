"""Independent SI, CODATA 2018, PDG 2024, and muonium reference values."""

from __future__ import annotations

from math import pi

import pytest

from asymmetry.core.utils.constants import (
    AVOGADRO_CONSTANT_PER_MOL,
    BOLTZMANN_CONSTANT_J_PER_K,
    BOLTZMANN_CONSTANT_MEV_PER_K,
    ELECTRON_GYROMAGNETIC_RATIO_RAD_PER_S_PER_T,
    ELECTRON_GYROMAGNETIC_RATIO_RAD_PER_US_PER_G,
    ELEMENTARY_CHARGE_C,
    GAUSS_TO_TESLA,
    MAGNETIC_FLUX_QUANTUM_WB,
    MOLAR_GAS_CONSTANT_J_PER_MOL_K,
    MUON_GYROMAGNETIC_RATIO_MHZ_PER_T,
    MUON_LIFETIME_US,
    MUON_MAGNETIC_MOMENT_J_PER_T,
    MUONIUM_VACUUM_HYPERFINE_MHZ,
    PION_LIFETIME_US,
    PLANCK_CONSTANT_J_S,
    PROTON_GYROMAGNETIC_RATIO_MHZ_PER_T,
    REDUCED_PLANCK_CONSTANT_J_S,
    VACUUM_PERMEABILITY_OVER_4PI_H_PER_M,
)


def test_si_defining_constants_are_exact() -> None:
    # Revised SI definitions, as tabulated by Tiesinga et al. (2021), Table XXXI.
    assert PLANCK_CONSTANT_J_S == 6.62607015e-34
    assert ELEMENTARY_CHARGE_C == 1.602176634e-19
    assert BOLTZMANN_CONSTANT_J_PER_K == 1.380649e-23
    assert AVOGADRO_CONSTANT_PER_MOL == 6.02214076e23


@pytest.mark.parametrize(
    ("value", "reference"),
    [
        (REDUCED_PLANCK_CONSTANT_J_S, 6.62607015e-34 / (2 * pi)),
        (MAGNETIC_FLUX_QUANTUM_WB, 6.62607015e-34 / (2 * 1.602176634e-19)),
        (BOLTZMANN_CONSTANT_MEV_PER_K, 1.380649e-23 / 1.602176634e-19 * 1e3),
        (MOLAR_GAS_CONSTANT_J_PER_MOL_K, 6.02214076e23 * 1.380649e-23),
    ],
)
def test_derived_constants_follow_exact_si_definitions(value: float, reference: float) -> None:
    assert value == pytest.approx(reference, rel=1e-15, abs=0)


def test_magnetic_constants_match_codata_2018() -> None:
    # Tiesinga et al., Rev. Mod. Phys. 93, 025010 (2021), Table XXXI.
    assert MUON_MAGNETIC_MOMENT_J_PER_T == -4.49044830e-26
    assert VACUUM_PERMEABILITY_OVER_4PI_H_PER_M == 1.00000000055e-7
    assert ELECTRON_GYROMAGNETIC_RATIO_RAD_PER_S_PER_T == 1.76085963023e11
    assert ELECTRON_GYROMAGNETIC_RATIO_RAD_PER_US_PER_G == pytest.approx(17.6085963023, rel=1e-15)
    assert PROTON_GYROMAGNETIC_RATIO_MHZ_PER_T == pytest.approx(42.577478518, rel=2e-8)


def test_muon_frequency_and_field_units_match_codata_precision() -> None:
    # The retained muon approximation differs from 2|mu_mu|/h by less than 6e-8.
    reference = 2 * 4.49044830e-26 / 6.62607015e-34 * 1e-6
    assert MUON_GYROMAGNETIC_RATIO_MHZ_PER_T == pytest.approx(reference, rel=6e-8)
    assert GAUSS_TO_TESLA == 1e-4
    assert MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * GAUSS_TO_TESLA == pytest.approx(
        0.01355388095, rel=6e-8
    )


def test_lifetimes_match_pdg_2024_with_documented_pion_rounding() -> None:
    assert MUON_LIFETIME_US == 2.1969811
    assert PION_LIFETIME_US == round(0.026033, 3)


def test_muonium_default_is_a_truncation_of_the_measured_interval() -> None:
    # Liu et al., Phys. Rev. Lett. 82, 711 (1999): 4,463,302,765(53) Hz.
    assert MUONIUM_VACUUM_HYPERFINE_MHZ == pytest.approx(4463.302765, abs=0.001, rel=0)


def test_consumer_values_preserve_reference_precision() -> None:
    from asymmetry.core.fitting.muon_fluorine.dipolar import (
        omega_d_f_f_rad_per_us,
        omega_d_mu_f_rad_per_us,
    )
    from asymmetry.core.fitting.muonium import G_E_MHZ_PER_G
    from asymmetry.core.fitting.sc.bcs import gap_ratio_from_mev
    from asymmetry.core.fitting.sc.constants import lambda_nm_to_sigma_us

    # Exact ħ and CODATA μ₀ together change the rounded dipolar baseline by 1.163e-9.
    assert omega_d_mu_f_rad_per_us(1.17) == pytest.approx(1.4111625863735597, rel=2e-9)
    assert omega_d_f_f_rad_per_us(2.34) == pytest.approx(0.052126483323242566, rel=2e-9)
    assert lambda_nm_to_sigma_us(200.0) == pytest.approx(2.6811215311944148, rel=1e-9)
    assert G_E_MHZ_PER_G == pytest.approx(2.802495142420715, rel=1e-12)
    assert gap_ratio_from_mev(1.0, 10.0) == pytest.approx(1.0 / (8.617333262e-2 * 10), rel=1e-9)
