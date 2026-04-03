"""Muon-fluorine polarization functions for entangled spin states."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.fitting.muon_fluorine.dipolar import (
    MUON_SIGMA_Z_THREE_SPIN,
    omega_d_f_f_rad_per_us,
    omega_d_mu_f_rad_per_us,
    three_spin_hamiltonian_rad_per_us,
)

_DEFAULT_NUM_BETA = 8
_DEFAULT_NUM_ALPHA = 8
_DEFAULT_NUM_GAMMA = 6
_CACHE_KEY_DECIMALS = 9
_SPECTRUM_BIN_DECIMALS = 10


def mu_f_polarization(t: NDArray[np.float64], r_muF: float) -> NDArray[np.float64]:
    """Analytical mu-F longitudinal polarization, D_z(t), for one fluorine."""
    t_arr = np.asarray(t, dtype=float)
    omega_d = omega_d_mu_f_rad_per_us(r_muF)
    return (
        1.0
        + 2.0 * np.cos(0.5 * omega_d * t_arr)
        + np.cos(omega_d * t_arr)
        + 2.0 * np.cos(1.5 * omega_d * t_arr)
    ) / 6.0


def linear_fmuf_polarization(t: NDArray[np.float64], r_muF: float) -> NDArray[np.float64]:
    """Analytical collinear F-mu-F polarization from the classic ionic-crystal model."""
    t_arr = np.asarray(t, dtype=float)
    omega_d = omega_d_mu_f_rad_per_us(r_muF)
    sqrt3 = np.sqrt(3.0)

    return (
        3.0
        + np.cos(sqrt3 * omega_d * t_arr)
        + (1.0 - 1.0 / sqrt3) * np.cos(0.5 * (3.0 - sqrt3) * omega_d * t_arr)
        + (1.0 + 1.0 / sqrt3) * np.cos(0.5 * (3.0 + sqrt3) * omega_d * t_arr)
    ) / 6.0


def _rz(angle: float) -> NDArray[np.float64]:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _ry(angle: float) -> NDArray[np.float64]:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=float,
    )


@lru_cache(maxsize=16)
def _powder_rotations(
    num_beta: int,
    num_alpha: int,
    num_gamma: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    beta_nodes, beta_weights = np.polynomial.legendre.leggauss(num_beta)
    alphas = np.linspace(0.0, 2.0 * np.pi, num_alpha, endpoint=False)
    gammas = np.linspace(0.0, 2.0 * np.pi, num_gamma, endpoint=False)

    rotations: list[NDArray[np.float64]] = []
    weights: list[float] = []
    alpha_gamma_norm = float(num_alpha * num_gamma)

    for node, node_weight in zip(beta_nodes, beta_weights, strict=True):
        beta = float(np.arccos(np.clip(node, -1.0, 1.0)))
        for alpha in alphas:
            rz_alpha = _rz(float(alpha))
            for gamma in gammas:
                rotation = rz_alpha @ _ry(beta) @ _rz(float(gamma))
                rotations.append(rotation)
                weights.append(0.5 * float(node_weight) / alpha_gamma_norm)

    return np.asarray(rotations, dtype=float), np.asarray(weights, dtype=float)


def _validate_general_geometry(r1: float, r2: float, theta_deg: float) -> None:
    if r1 <= 0.0 or r2 <= 0.0:
        raise ValueError("r1 and r2 must be positive")
    if theta_deg <= 0.0 or theta_deg > 180.0:
        raise ValueError("theta must be in the range (0, 180] degrees")


@lru_cache(maxsize=256)
def _general_spectral_terms_cached(
    r1_key: float,
    r2_key: float,
    theta_key: float,
    num_beta: int,
    num_alpha: int,
    num_gamma: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r1 = float(r1_key)
    r2 = float(r2_key)
    theta_deg = float(theta_key)
    _validate_general_geometry(r1, r2, theta_deg)

    theta = np.deg2rad(theta_deg)
    v_f1 = np.array([0.0, 0.0, 1.0], dtype=float)
    v_f2 = np.array([np.sin(theta), 0.0, np.cos(theta)], dtype=float)

    rotations, orientation_weights = _powder_rotations(num_beta, num_alpha, num_gamma)
    n_mu_f1 = rotations @ v_f1
    n_mu_f2 = rotations @ v_f2

    f1_vectors = r1 * n_mu_f1
    f2_vectors = r2 * n_mu_f2
    f1_to_f2 = f2_vectors - f1_vectors
    d_f1_f2 = np.linalg.norm(f1_to_f2, axis=1)
    if float(np.min(d_f1_f2)) <= 1.0e-9:
        raise ValueError("Invalid geometry: F-F distance is zero or too small")
    n_f1_f2 = f1_to_f2 / d_f1_f2[:, None]

    coupling_mu_f1 = omega_d_mu_f_rad_per_us(r1)
    coupling_mu_f2 = omega_d_mu_f_rad_per_us(r2)
    coupling_f1_f2 = omega_d_f_f_rad_per_us(float(np.mean(d_f1_f2)))

    frequencies_list: list[NDArray[np.float64]] = []
    amplitudes_list: list[NDArray[np.float64]] = []
    dim = MUON_SIGMA_Z_THREE_SPIN.shape[0]

    for idx, orient_weight in enumerate(orientation_weights):
        hamiltonian = three_spin_hamiltonian_rad_per_us(
            coupling_mu_f1,
            coupling_mu_f2,
            coupling_f1_f2,
            n_mu_f1[idx],
            n_mu_f2[idx],
            n_f1_f2[idx],
        )
        evals, evecs = np.linalg.eigh(hamiltonian)

        sigma_mu_z_eigenbasis = evecs.conj().T @ MUON_SIGMA_Z_THREE_SPIN @ evecs
        transition_weights = (np.abs(sigma_mu_z_eigenbasis) ** 2) / float(dim)

        omega_mn = (evals[:, None] - evals[None, :]).real
        frequencies_list.append(omega_mn.ravel())
        amplitudes_list.append((float(orient_weight) * transition_weights).ravel().real)

    frequencies = np.concatenate(frequencies_list)
    amplitudes = np.concatenate(amplitudes_list)

    binned_frequencies = np.round(frequencies, decimals=_SPECTRUM_BIN_DECIMALS)
    unique_freq, inverse = np.unique(binned_frequencies, return_inverse=True)
    binned_amplitudes = np.zeros_like(unique_freq, dtype=float)
    np.add.at(binned_amplitudes, inverse, amplitudes)

    total_weight = float(np.sum(binned_amplitudes))
    if total_weight > 0.0:
        binned_amplitudes /= total_weight

    return unique_freq, binned_amplitudes


def general_fmuf_polarization(
    t: NDArray[np.float64],
    r1: float,
    r2: float,
    theta: float,
) -> NDArray[np.float64]:
    """Numerical powder-averaged polarization for a general F-mu-F geometry.

    The geometry is parameterized by two mu-F distances (r1, r2) in Angstrom and
    a bond angle theta in degrees. The eigenspectrum for each geometry is cached
    to keep fitting workloads feasible when the same geometry is re-evaluated.
    """
    t_arr = np.asarray(t, dtype=float)

    r1_key = round(float(r1), _CACHE_KEY_DECIMALS)
    r2_key = round(float(r2), _CACHE_KEY_DECIMALS)
    theta_key = round(float(theta), _CACHE_KEY_DECIMALS)

    freqs, amps = _general_spectral_terms_cached(
        r1_key,
        r2_key,
        theta_key,
        _DEFAULT_NUM_BETA,
        _DEFAULT_NUM_ALPHA,
        _DEFAULT_NUM_GAMMA,
    )
    cos_terms = np.cos(np.outer(freqs, t_arr))
    return np.asarray(amps @ cos_terms, dtype=float)


__all__ = [
    "mu_f_polarization",
    "linear_fmuf_polarization",
    "general_fmuf_polarization",
]
