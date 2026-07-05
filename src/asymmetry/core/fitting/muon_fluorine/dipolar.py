"""Dipolar Hamiltonian helpers for muon-fluorine spin systems."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.utils.constants import (
    FLUORINE_19_GYROMAGNETIC_RATIO_MHZ_PER_T,
    MUON_GYROMAGNETIC_RATIO_MHZ_PER_T,
)

MU_0_OVER_4PI = 1.0e-7
HBAR_J_S = 1.054_571_817e-34
ANGSTROM_TO_M = 1.0e-10
RAD_PER_S_TO_RAD_PER_US = 1.0e-6

_PAIR_MU_F1 = (0, 1)
_PAIR_MU_F2 = (1, 2)
_PAIR_F1_F2 = (0, 2)

_SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
_SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
_ID2 = np.eye(2, dtype=complex)

_SX = 0.5 * _SIGMA_X
_SY = 0.5 * _SIGMA_Y
_SZ = 0.5 * _SIGMA_Z


def _kron_chain(mats: list[NDArray[np.complex128]]) -> NDArray[np.complex128]:
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def _spin_half_operators(n_spins: int) -> dict[tuple[int, int], NDArray[np.complex128]]:
    """Spin-1/2 operators S_x/S_y/S_z for each site of an n-spin product space."""
    ops: dict[tuple[int, int], NDArray[np.complex128]] = {}
    for site in range(n_spins):
        for axis, mat in enumerate((_SX, _SY, _SZ)):
            factors: list[NDArray[np.complex128]] = [_ID2] * n_spins
            factors[site] = mat
            ops[(site, axis)] = _kron_chain(factors)
    return ops


def _pair_operators(
    spin_ops: dict[tuple[int, int], NDArray[np.complex128]],
    pair: tuple[int, int],
    dim: int,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Isotropic S_i.S_j and the (3, 3, dim, dim) tensor S_i^a S_j^b for a pair."""
    tensor = np.empty((3, 3, dim, dim), dtype=complex)
    for i in range(3):
        for j in range(3):
            tensor[i, j] = spin_ops[(pair[0], i)] @ spin_ops[(pair[1], j)]
    isotropic = tensor[0, 0] + tensor[1, 1] + tensor[2, 2]
    return isotropic, tensor


# --- three-spin (F-mu-F) operator tables -------------------------------------

_SPIN_OPS = _spin_half_operators(3)

MUON_SIGMA_Z_THREE_SPIN = _kron_chain([_ID2, _SIGMA_Z, _ID2])

_PAIR_ISO: dict[tuple[int, int], NDArray[np.complex128]] = {}
_PAIR_TENSOR: dict[tuple[int, int], NDArray[np.complex128]] = {}
for _pair in (_PAIR_MU_F1, _PAIR_MU_F2, _PAIR_F1_F2):
    _iso, _tensor = _pair_operators(_SPIN_OPS, _pair, 8)
    _PAIR_ISO[_pair] = _iso
    _PAIR_TENSOR[_pair] = _tensor


def _gamma_rad_per_s_per_t(gamma_mhz_per_t: float) -> float:
    return 2.0 * np.pi * gamma_mhz_per_t * 1.0e6


def omega_dipolar_rad_per_us(
    distance_angstrom: float,
    gamma_i_mhz_per_t: float,
    gamma_j_mhz_per_t: float,
) -> float:
    """Return dipolar angular frequency in rad/us for a spin pair."""
    r_m = float(distance_angstrom) * ANGSTROM_TO_M
    if r_m <= 0.0:
        raise ValueError("distance_angstrom must be positive")

    gamma_i = _gamma_rad_per_s_per_t(gamma_i_mhz_per_t)
    gamma_j = _gamma_rad_per_s_per_t(gamma_j_mhz_per_t)
    omega_rad_per_s = MU_0_OVER_4PI * gamma_i * gamma_j * HBAR_J_S / (r_m**3)
    return omega_rad_per_s * RAD_PER_S_TO_RAD_PER_US


def omega_d_mu_f_rad_per_us(distance_angstrom: float) -> float:
    """Return mu-F dipolar angular frequency in rad/us."""
    return omega_dipolar_rad_per_us(
        distance_angstrom,
        MUON_GYROMAGNETIC_RATIO_MHZ_PER_T,
        FLUORINE_19_GYROMAGNETIC_RATIO_MHZ_PER_T,
    )


def omega_d_f_f_rad_per_us(distance_angstrom: float) -> float:
    """Return F-F dipolar angular frequency in rad/us."""
    return omega_dipolar_rad_per_us(
        distance_angstrom,
        FLUORINE_19_GYROMAGNETIC_RATIO_MHZ_PER_T,
        FLUORINE_19_GYROMAGNETIC_RATIO_MHZ_PER_T,
    )


def r_mu_f_from_omega_d(omega_d_rad_per_us: float) -> float:
    """Invert :func:`omega_d_mu_f_rad_per_us`: mu-F distance in angstrom.

    ``omega_d`` scales as ``r^-3``, so ``r = (omega_d(1 A) / omega_d)^(1/3)``.
    """
    omega = float(omega_d_rad_per_us)
    if omega <= 0.0:
        raise ValueError("omega_d_rad_per_us must be positive")
    return (omega_d_mu_f_rad_per_us(1.0) / omega) ** (1.0 / 3.0)


def pair_dipolar_hamiltonian_three_spin(
    coupling_rad_per_us: float,
    unit_vector: NDArray[np.float64],
    pair: tuple[int, int],
) -> NDArray[np.complex128]:
    """Build one pair contribution for a three-spin dipolar Hamiltonian."""
    if pair not in _PAIR_TENSOR:
        raise ValueError(f"Unsupported pair index: {pair}")

    n = np.asarray(unit_vector, dtype=float)
    if n.shape != (3,):
        raise ValueError("unit_vector must have shape (3,)")

    norm = float(np.linalg.norm(n))
    if norm <= 0.0:
        raise ValueError("unit_vector must be non-zero")
    n /= norm

    anisotropic = np.tensordot(np.outer(n, n), _PAIR_TENSOR[pair], axes=((0, 1), (0, 1)))
    return float(coupling_rad_per_us) * (_PAIR_ISO[pair] - 3.0 * anisotropic)


def three_spin_hamiltonian_rad_per_us(
    coupling_mu_f1: float,
    coupling_mu_f2: float,
    coupling_f1_f2: float,
    n_mu_f1: NDArray[np.float64],
    n_mu_f2: NDArray[np.float64],
    n_f1_f2: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """Construct the full 8x8 dipolar Hamiltonian (angular-frequency units)."""
    return (
        pair_dipolar_hamiltonian_three_spin(coupling_mu_f1, n_mu_f1, _PAIR_MU_F1)
        + pair_dipolar_hamiltonian_three_spin(coupling_mu_f2, n_mu_f2, _PAIR_MU_F2)
        + pair_dipolar_hamiltonian_three_spin(coupling_f1_f2, n_f1_f2, _PAIR_F1_F2)
    )


# --- four-spin (muon + three fluorines) operator tables ----------------------
#
# Spin index 0 is the muon; indices 1..3 are the fluorines.  The per-pair
# (3, 3, 16, 16) tensors are consumed by the batched Hamiltonian build in
# polarization._triangle_spectral_terms_cached.

_N_FOUR = 4
_DIM_FOUR = 2**_N_FOUR

_SPIN_OPS_FOUR = _spin_half_operators(_N_FOUR)

MUON_SIGMA_Z_FOUR_SPIN = _kron_chain([_SIGMA_Z, _ID2, _ID2, _ID2])

_PAIRS_FOUR: list[tuple[int, int]] = [(i, j) for i in range(_N_FOUR) for j in range(i + 1, _N_FOUR)]

_PAIR_ISO_FOUR: dict[tuple[int, int], NDArray[np.complex128]] = {}
_PAIR_TENSOR_FOUR: dict[tuple[int, int], NDArray[np.complex128]] = {}
for _pair4 in _PAIRS_FOUR:
    _iso4, _tensor4 = _pair_operators(_SPIN_OPS_FOUR, _pair4, _DIM_FOUR)
    _PAIR_ISO_FOUR[_pair4] = _iso4
    _PAIR_TENSOR_FOUR[_pair4] = _tensor4


__all__ = [
    "MUON_SIGMA_Z_THREE_SPIN",
    "MUON_SIGMA_Z_FOUR_SPIN",
    "omega_d_mu_f_rad_per_us",
    "omega_d_f_f_rad_per_us",
    "omega_dipolar_rad_per_us",
    "r_mu_f_from_omega_d",
    "three_spin_hamiltonian_rad_per_us",
]
