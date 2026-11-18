"""Independent normalized spin observables, with Hamiltonian frequencies in MHz."""

import numpy as np
import pytest

from asymmetry.core.fitting import muonium
from asymmetry.core.fitting.composite import COMPONENTS

SPINS = tuple(
    0.5 * np.asarray(matrix, dtype=complex)
    for matrix in (
        [[0, 1], [1, 0]],
        [[0, -1j], [1j, 0]],
        [[1, 0], [0, -1]],
    )
)
ELECTRON = tuple(np.kron(spin, np.eye(2)) for spin in SPINS)
MUON = tuple(np.kron(np.eye(2), spin) for spin in SPINS)


def hyperfine_hamiltonian(hyperfine, field):
    """H/h in MHz: positive electron and negative muon Zeeman coefficients."""
    hamiltonian = (
        muonium.G_E_MHZ_PER_G * field * ELECTRON[2] - muonium.G_MU_MHZ_PER_G * field * MUON[2]
    )
    for axis, coupling in enumerate(hyperfine):
        hamiltonian = hamiltonian + coupling * ELECTRON[axis] @ MUON[axis]
    return hamiltonian


def spin_evolution(times, hyperfine, field, direction):
    """Initially polarized muon, unpolarized electron, matching detector axis."""
    energies, vectors = np.linalg.eigh(hyperfine_hamiltonian(hyperfine, field))
    observable = 2 * MUON[direction]
    density = (np.eye(4) + observable) / 4
    values = []
    for time in times:
        evolution = (vectors * np.exp(-2j * np.pi * energies * time)) @ vectors.conj().T
        values.append(np.trace(evolution @ density @ evolution.conj().T @ observable).real)
    return np.asarray(values)


@pytest.mark.parametrize("field,a_hf", [(0, 50), (10, 50), (-10, 50), (1000, 50), (30, 4463.302)])
def test_full_tf_matches_normalized_density_matrix(field, a_hf):
    times = np.linspace(0, 2 / a_hf, 51)
    independent = spin_evolution(times, [a_hf] * 3, field, 0)
    actual = muonium.tf_muonium(times, field, a_hf, 0)
    np.testing.assert_allclose(actual, independent, atol=1e-11, rtol=0)


@pytest.mark.parametrize("a_hf,anisotropy", [(3, 1), (3, 0), (0.5, -0.2)])
def test_zf_three_lines_omit_additional_static_sixth(a_hf, anisotropy):
    times = np.linspace(0, 8, 101)
    tensor = [a_hf - anisotropy, a_hf - anisotropy, a_hf + 2 * anisotropy]
    independent = sum(spin_evolution(times, tensor, 0, axis) for axis in range(3)) / 3
    actual = muonium.zf_muonium(times, a_hf, anisotropy, 0, 0)
    np.testing.assert_allclose(actual + 1 / 6, independent, atol=1e-11, rtol=0)


@pytest.mark.parametrize("regime,field", [("low", 1.0), ("high", 1000.0)])
def test_low_and_high_field_pairs_match_independent_transition_selection(regime, field):
    a_hf = 50.0
    times = np.linspace(0, 1, 101)
    energies, vectors = np.linalg.eigh(hyperfine_hamiltonian([a_hf] * 3, field))
    detected = vectors.conj().T @ (2 * MUON[0]) @ vectors
    if regime == "low":
        # At low field the three upper eigenstates are the triplet manifold.
        reference = np.zeros_like(times)
        for left in range(1, 4):
            for right in range(1, 4):
                reference += (
                    abs(detected[left, right]) ** 2
                    / 4
                    * np.cos(2 * np.pi * (energies[left] - energies[right]) * times)
                )
        actual = muonium.low_tf_muonium(times, field, a_hf, 0)
        np.testing.assert_allclose(actual, reference, atol=1e-11, rtol=0)
        assert actual[0] == pytest.approx(0.5)
    else:
        # The high-field convention keeps the two strongest transitions with equal weights.
        pairs = [(left, right) for left in range(4) for right in range(left + 1, 4)]
        strongest = sorted(pairs, key=lambda pair: abs(detected[pair]) ** 2)[-2:]
        reference = sum(
            0.5 * np.cos(2 * np.pi * (energies[left] - energies[right]) * times)
            for left, right in strongest
        )
        actual = muonium.high_tf_muonium(times, field, a_hf, 0)
        np.testing.assert_allclose(actual, reference, atol=1e-11, rtol=0)
        full = spin_evolution(times, [a_hf] * 3, field, 0)
        np.testing.assert_allclose(actual, full, atol=1e-3, rtol=0)


@pytest.mark.parametrize("field", [0, 100, 1000, 5000])
def test_lf_rate_matches_eq22_using_independent_energy_gap(field):
    # Kadono and Ito (2025), Eq.22; Eq.19 has a separate repolarization amplitude.
    a_hf, delta_ex, tau_c = 50.0, 0.5, 0.01
    energies = np.linalg.eigvalsh(hyperfine_hamiltonian([a_hf] * 3, field))
    # The intratriplet gap connects |up,up> to the triplet-like m=0 state.
    parallel_energy = a_hf / 4 + (muonium.G_E_MHZ_PER_G - muonium.G_MU_MHZ_PER_G) * field / 2
    triplet_zero = max(np.linalg.eigvalsh(hyperfine_hamiltonian([a_hf] * 3, field)[1:3, 1:3]))
    assert np.min(np.abs(energies - parallel_energy)) < 1e-10
    angular_gap = 2 * np.pi * abs(parallel_energy - triplet_zero)
    rate = delta_ex**2 * tau_c / (1 + (angular_gap * tau_c) ** 2)
    time = np.array([0, 0.5, 3])
    actual = muonium.muonium_lf_relaxation(time, delta_ex, tau_c, field, a_hf)
    np.testing.assert_allclose(actual, np.exp(-rate * time), atol=1e-14)


def test_lf_displayed_equation_has_no_repolarization_prefactor():
    assert r"(1-\delta)" not in COMPONENTS["MuoniumLFRelax"].latex_equation
