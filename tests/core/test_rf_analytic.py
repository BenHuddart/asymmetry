"""First-order RF gaps checked with independent spin-matrix expectations.

The operator is the isotropic contact Hamiltonian documented in the RF model
reference (McKenzie et al., J. Phys. Chem. B 117, 13614 (2013), for context).
These checks concern signed branch energies, not transition visibility or the
accuracy of first-order perturbation theory at finite proton coupling.
"""

import numpy as np
import pytest

from asymmetry.core.fitting.muon_proton import analytic_rf_transition_freqs
from asymmetry.core.utils.constants import (
    ELECTRON_GYROMAGNETIC_RATIO_RAD_PER_US_PER_G,
    GAUSS_TO_TESLA,
    MUON_GYROMAGNETIC_RATIO_MHZ_PER_T,
    PROTON_GYROMAGNETIC_RATIO_MHZ_PER_T,
)

G_E = ELECTRON_GYROMAGNETIC_RATIO_RAD_PER_US_PER_G / (2 * np.pi)
G_MU = MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * GAUSS_TO_TESLA
G_P = PROTON_GYROMAGNETIC_RATIO_MHZ_PER_T * GAUSS_TO_TESLA


def first_order_spin_gaps(field, a_mu, a_p):
    """Diagonalise the electron-muon m_z=0 block; perturb each proton sector."""
    spins = (
        np.array([[0, 1], [1, 0]], dtype=complex) / 2,
        np.array([[0, -1j], [1j, 0]], dtype=complex) / 2,
        np.diag([1, -1]) / 2,
    )
    electron = [np.kron(spin, np.eye(2)) for spin in spins]
    muon = [np.kron(np.eye(2), spin) for spin in spins]
    hamiltonian = field * (G_E * electron[2] - G_MU * muon[2])
    for se, sm in zip(electron, muon, strict=True):
        hamiltonian = hamiltonian + a_mu * se @ sm
    energies, vectors = np.linalg.eigh(hamiltonian[1:3, 1:3])
    mixed = np.zeros(4, dtype=complex)
    mixed[1:3] = vectors[:, -1]
    aligned = np.array([1, 0, 0, 0])
    gaps = []
    for proton_projection in (0.5, -0.5):
        perturbation = a_p * proton_projection * electron[
            2
        ] - G_P * field * proton_projection * np.eye(4)
        initial = np.vdot(aligned, (hamiltonian + perturbation) @ aligned).real
        final = energies[-1] + np.vdot(mixed, perturbation @ mixed).real
        gaps.append(initial - final)
    return gaps


@pytest.mark.parametrize(
    "field,a_mu,a_p",
    [
        (0.0, 50.0, 2.0),
        (10.0, 4000.0, 50.0),
        (-10.0, 4000.0, 50.0),
        (100.0, 50.0, 2.0),
        (800.0, 514.78, 124.6),
        (-800.0, 514.78, 124.6),
        (1200.0, 400.0, -20.0),
        (800.0, 514.78, 0.0),
        (800.0, 0.0, 2.0),
        (-800.0, 0.0, 2.0),
        (800.0, -514.78, 20.0),
    ],
)
def test_analytic_rf_matches_spin_expectation_perturbation(field, a_mu, a_p):
    np.testing.assert_allclose(
        analytic_rf_transition_freqs(field, a_mu, a_p),
        first_order_spin_gaps(field, a_mu, a_p),
        atol=2e-12,
        rtol=0,
    )


def test_analytic_rf_keeps_signed_proton_sector_order():
    # At B=0 and positive A_mu, the two triplet levels coincide before the
    # diagonal proton correction. The outputs label m_p=+1/2 and -1/2.
    assert analytic_rf_transition_freqs(0.0, 500.0, 8.0) == (2.0, -2.0)
    assert analytic_rf_transition_freqs(0.0, 500.0, -8.0) == (-2.0, 2.0)


@pytest.mark.parametrize("a_mu", [0.0, 1e-30, 1e-300])
def test_analytic_rf_zero_muon_coupling_limit(a_mu):
    field = 800.0
    # At positive field, both selected electron states have S_ez=+1/2;
    # the proton shift cancels and the signed gap is the muon Zeeman gap.
    np.testing.assert_allclose(
        analytic_rf_transition_freqs(field, a_mu, 100.0),
        [-G_MU * field] * 2,
        atol=1e-12,
        rtol=0,
    )


def test_analytic_rf_joint_zero_uses_triplet_continuation():
    # The joint degeneracy has no unique eigenbasis. The convention is the
    # B=0, A_mu -> 0+ triplet continuation, whose electron projection is zero.
    assert analytic_rf_transition_freqs(0.0, 0.0, 8.0) == (2.0, -2.0)


def test_analytic_rf_small_field_retains_linear_zeeman_gap():
    field = 1e-12
    expected = (G_E - G_MU) * field / 2
    np.testing.assert_allclose(
        analytic_rf_transition_freqs(field, 4000.0, 0.0),
        [expected] * 2,
        atol=0,
        rtol=1e-12,
    )
