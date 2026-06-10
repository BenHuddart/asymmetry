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


def _kron3(
    a: NDArray[np.complex128], b: NDArray[np.complex128], c: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    return np.kron(np.kron(a, b), c)


_SPIN_OPS: dict[tuple[int, int], NDArray[np.complex128]] = {
    (0, 0): _kron3(_SX, _ID2, _ID2),
    (0, 1): _kron3(_SY, _ID2, _ID2),
    (0, 2): _kron3(_SZ, _ID2, _ID2),
    (1, 0): _kron3(_ID2, _SX, _ID2),
    (1, 1): _kron3(_ID2, _SY, _ID2),
    (1, 2): _kron3(_ID2, _SZ, _ID2),
    (2, 0): _kron3(_ID2, _ID2, _SX),
    (2, 1): _kron3(_ID2, _ID2, _SY),
    (2, 2): _kron3(_ID2, _ID2, _SZ),
}


MUON_SIGMA_Z_THREE_SPIN = _kron3(_ID2, _SIGMA_Z, _ID2)


def _build_pair_operators(
    pair: tuple[int, int],
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    tensor = np.empty((3, 3, 8, 8), dtype=complex)
    for i in range(3):
        for j in range(3):
            tensor[i, j] = _SPIN_OPS[(pair[0], i)] @ _SPIN_OPS[(pair[1], j)]
    isotropic = tensor[0, 0] + tensor[1, 1] + tensor[2, 2]
    return isotropic, tensor


_PAIR_ISO: dict[tuple[int, int], NDArray[np.complex128]] = {}
_PAIR_TENSOR: dict[tuple[int, int], NDArray[np.complex128]] = {}
for _pair in (_PAIR_MU_F1, _PAIR_MU_F2, _PAIR_F1_F2):
    _iso, _tensor = _build_pair_operators(_pair)
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


# --- four-spin (muon + three fluorines) machinery ---------------------------
#
# Spin index 0 is the muon; indices 1..3 are the fluorines.  Operators are
# built once at import; the per-pair (3, 3, 16, 16) tensors stay small enough
# (~100 kB total) to precompute for all six pairs.

_N_FOUR = 4
_DIM_FOUR = 2**_N_FOUR


def _kron_chain(mats: list[NDArray[np.complex128]]) -> NDArray[np.complex128]:
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


_SPIN_OPS_FOUR: dict[tuple[int, int], NDArray[np.complex128]] = {}
for _s in range(_N_FOUR):
    for _a, _mat in enumerate((_SX, _SY, _SZ)):
        _factors: list[NDArray[np.complex128]] = [_ID2] * _N_FOUR
        _factors[_s] = _mat
        _SPIN_OPS_FOUR[(_s, _a)] = _kron_chain(_factors)

MUON_SIGMA_Z_FOUR_SPIN = _kron_chain([_SIGMA_Z, _ID2, _ID2, _ID2])

_PAIRS_FOUR: list[tuple[int, int]] = [(i, j) for i in range(_N_FOUR) for j in range(i + 1, _N_FOUR)]

_PAIR_ISO_FOUR: dict[tuple[int, int], NDArray[np.complex128]] = {}
_PAIR_TENSOR_FOUR: dict[tuple[int, int], NDArray[np.complex128]] = {}
for _pair4 in _PAIRS_FOUR:
    _tensor4 = np.empty((3, 3, _DIM_FOUR, _DIM_FOUR), dtype=complex)
    for _i in range(3):
        for _j in range(3):
            _tensor4[_i, _j] = _SPIN_OPS_FOUR[(_pair4[0], _i)] @ _SPIN_OPS_FOUR[(_pair4[1], _j)]
    _PAIR_ISO_FOUR[_pair4] = _tensor4[0, 0] + _tensor4[1, 1] + _tensor4[2, 2]
    _PAIR_TENSOR_FOUR[_pair4] = _tensor4


def four_spin_hamiltonian_rad_per_us(
    couplings: list[tuple[tuple[int, int], float, NDArray[np.float64]]],
) -> NDArray[np.complex128]:
    """Construct the 16x16 dipolar Hamiltonian for the muon + 3F system.

    ``couplings`` is a list of ``(pair, coupling_rad_per_us, unit_vector)``
    entries, one per spin pair (pair indices: 0 = muon, 1-3 = fluorines).
    Each pair contributes ``coupling * [S_i . S_j - 3 (S_i . n)(S_j . n)]``.
    """
    h = np.zeros((_DIM_FOUR, _DIM_FOUR), dtype=complex)
    for pair, coupling, n_vec in couplings:
        if pair not in _PAIR_TENSOR_FOUR:
            raise ValueError(f"Unsupported four-spin pair index: {pair}")
        n = np.asarray(n_vec, dtype=float)
        norm = float(np.linalg.norm(n))
        if norm <= 0.0:
            raise ValueError("unit_vector must be non-zero")
        n = n / norm
        anisotropic = np.tensordot(np.outer(n, n), _PAIR_TENSOR_FOUR[pair], axes=((0, 1), (0, 1)))
        h = h + float(coupling) * (_PAIR_ISO_FOUR[pair] - 3.0 * anisotropic)
    return h


__all__ = [
    "MUON_SIGMA_Z_THREE_SPIN",
    "MUON_SIGMA_Z_FOUR_SPIN",
    "omega_d_mu_f_rad_per_us",
    "omega_d_f_f_rad_per_us",
    "omega_dipolar_rad_per_us",
    "three_spin_hamiltonian_rad_per_us",
    "four_spin_hamiltonian_rad_per_us",
]
