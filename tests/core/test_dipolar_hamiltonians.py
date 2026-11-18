"""Independent point-dipole Hamiltonians and isotropic detector traces.

Lancaster et al., PRL 99, 267601 (2007), Eqs. (1) and (2), define the
dipole interaction and detected spin observable. Constants are shared;
these checks establish coupling units, spin algebra and selected geometries.
"""

from functools import reduce
from itertools import combinations

import numpy as np
import pytest

from asymmetry.core.fitting.composite import COMPONENTS
from asymmetry.core.fitting.muon_fluorine import polarization
from asymmetry.core.utils.constants import (
    ELECTRON_GYROMAGNETIC_RATIO_RAD_PER_US_PER_G,
    FLUORINE_19_GYROMAGNETIC_RATIO_MHZ_PER_T,
    GAUSS_TO_TESLA,
    MUON_GYROMAGNETIC_RATIO_MHZ_PER_T,
    PROTON_GYROMAGNETIC_RATIO_MHZ_PER_T,
    REDUCED_PLANCK_CONSTANT_J_S,
    VACUUM_PERMEABILITY_OVER_4PI_H_PER_M,
)


def dipolar_powder(times, positions, gyromagnetic_ratios, nuclear_pairs=True):
    """Hamiltonian in rad/μs; position in Å, gamma/2pi in MHz/T, muon first."""
    spins = [
        np.array([[0, 1], [1, 0]]) / 2,
        np.array([[0, -1j], [1j, 0]]) / 2,
        np.diag([1, -1]) / 2,
    ]
    count = len(positions)
    operators = []
    for site in range(count):
        operators.append(
            [
                reduce(np.kron, [spin if index == site else np.eye(2) for index in range(count)])
                for spin in spins
            ]
        )
    hamiltonian = np.zeros((2**count, 2**count), dtype=complex)
    for left, right in combinations(range(count), 2):
        if not nuclear_pairs and left != 0:
            continue
        displacement = np.asarray(positions[right]) - positions[left]
        distance = np.linalg.norm(displacement)
        direction = displacement / distance
        gamma_left, gamma_right = np.asarray(gyromagnetic_ratios)[[left, right]] * (2 * np.pi * 1e6)
        coupling = (
            VACUUM_PERMEABILITY_OVER_4PI_H_PER_M
            * REDUCED_PLANCK_CONSTANT_J_S
            * gamma_left
            * gamma_right
            / (distance * 1e-10) ** 3
            * 1e-6
        )
        longitudinal_left = sum(direction[axis] * operators[left][axis] for axis in range(3))
        longitudinal_right = sum(direction[axis] * operators[right][axis] for axis in range(3))
        isotropic = sum(operators[left][axis] @ operators[right][axis] for axis in range(3))
        hamiltonian += coupling * (isotropic - 3 * longitudinal_left @ longitudinal_right)
    energies, vectors = np.linalg.eigh(hamiltonian)
    values = np.zeros_like(times)
    # Averaging three orthogonal initial/detector axes is exact for an isotropic powder.
    for muon_spin in operators[0]:
        detected = vectors.conj().T @ (2 * muon_spin) @ vectors
        weights = abs(detected) ** 2 / (3 * 2**count)
        gaps = energies[:, None] - energies[None, :]
        values += np.einsum("ij,ijt->t", weights, np.cos(gaps[..., None] * times))
    return values


@pytest.mark.parametrize("distance", [0.9, 1.17, 2.0])
def test_mu_f_and_linear_fmuf_match_independent_spin_trace(distance):
    times = np.linspace(0, 8, 41)
    gamma_mu = MUON_GYROMAGNETIC_RATIO_MHZ_PER_T
    gamma_f = FLUORINE_19_GYROMAGNETIC_RATIO_MHZ_PER_T
    for positions, gammas, function in (
        ([[0, 0, 0], [0, 0, distance]], [gamma_mu, gamma_f], polarization.mu_f_polarization),
        (
            [[0, 0, 0], [0, 0, distance], [0, 0, -distance]],
            [gamma_mu, gamma_f, gamma_f],
            polarization.linear_fmuf_polarization,
        ),
    ):
        independent = dipolar_powder(times, positions, gammas, nuclear_pairs=False)
        np.testing.assert_allclose(function(times, distance), independent, atol=1e-11, rtol=0)


@pytest.mark.parametrize("angle", [60, 120, 180])
def test_general_and_triangle_geometries_match_independent_spin_trace(angle):
    times = np.linspace(0, 8, 31)
    theta = np.deg2rad(angle)
    gamma_mu = MUON_GYROMAGNETIC_RATIO_MHZ_PER_T
    gamma_f = FLUORINE_19_GYROMAGNETIC_RATIO_MHZ_PER_T
    positions = [[0, 0, 0], [0, 0, 1.17], [1.3 * np.sin(theta), 0, 1.3 * np.cos(theta)]]
    independent = dipolar_powder(times, positions, [gamma_mu, gamma_f, gamma_f])
    actual = polarization.general_fmuf_polarization(times, 1.17, 1.3, angle)
    np.testing.assert_allclose(actual, independent, atol=2e-9, rtol=0)
    positions.insert(2, [0, 0, -1.17])
    independent = dipolar_powder(times, positions, [gamma_mu, gamma_f, gamma_f, gamma_f])
    actual = polarization.fmuf_triangle_polarization(times, 1.17, 1.3, angle)
    np.testing.assert_allclose(actual, independent, atol=2e-9, rtol=0)


@pytest.mark.parametrize(
    "name,partner_gamma,distance_parameter",
    [
        ("ProtonDipole", PROTON_GYROMAGNETIC_RATIO_MHZ_PER_T, "r_muH"),
        (
            "ElectronDipole",
            ELECTRON_GYROMAGNETIC_RATIO_RAD_PER_US_PER_G / (2 * np.pi * GAUSS_TO_TESLA),
            "r_mue",
        ),
    ],
)
def test_distance_and_field_pair_parameterizations_match_spin_trace(
    name, partner_gamma, distance_parameter
):
    times = np.linspace(0, 0.8, 41)
    distance, amplitude = 2.0, 17.0
    independent = dipolar_powder(
        times, [[0, 0, 0], [0, 0, distance]], [MUON_GYROMAGNETIC_RATIO_MHZ_PER_T, partner_gamma]
    )
    actual = COMPONENTS[name].function(
        times, A=amplitude, lambda_T=0, **{distance_parameter: distance}
    )
    np.testing.assert_allclose(actual / amplitude, independent, atol=1e-11, rtol=0)
    dipolar_field_tesla = (
        VACUUM_PERMEABILITY_OVER_4PI_H_PER_M
        * REDUCED_PLANCK_CONSTANT_J_S
        * (2 * np.pi * partner_gamma * 1e6)
        / (distance * 1e-10) ** 3
    )
    field_pair = COMPONENTS["DipolarPairField"].function(
        times, A=amplitude, B_dip=dipolar_field_tesla / GAUSS_TO_TESLA, lambda_T=0
    )
    np.testing.assert_allclose(field_pair / amplitude, independent, atol=1e-11, rtol=0)
