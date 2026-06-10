"""Muon-fluorine polarization functions for entangled spin states."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.fitting.models import _strong_collision_solve
from asymmetry.core.fitting.muon_fluorine.dipolar import (
    MUON_SIGMA_Z_FOUR_SPIN,
    MUON_SIGMA_Z_THREE_SPIN,
    four_spin_hamiltonian_rad_per_us,
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


# Cache of dynamic-FmuF solutions keyed by quantised (r_muF, nu, tmax).
_DYN_FMUF_CACHE: dict[tuple, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
_DYN_FMUF_CACHE_MAX = 128

# Above this fluctuation rate (µs⁻¹) the trapezoidal strong-collision solver
# would need a prohibitively fine grid (same crossover as the dynamic KT); the
# motional-narrowing limit exp(-2 omega_d^2 t / nu) is used instead.  The static
# F-mu-F second moment is M2 = 2 omega_d^2, so the narrowed rate is M2/nu.
_DYN_FMUF_NU_SWITCH = 12.0


def dynamic_fmuf_polarization(
    t: NDArray[np.float64], r_muF: float, nu: float
) -> NDArray[np.float64]:
    """Strong-collision dynamicized linear F-mu-F polarization (WiMDA ``dyn F-u-F``).

    The static collinear F-mu-F polarization ``G_s`` (eqn 4.81 of Blundell, De
    Renzi, Lancaster & Pratt, *Muon Spectroscopy*, OUP 2022) dynamicized by the
    strong-collision integral equation (eqn 5.30):

        G_d(t) = e^{-nu t} G_s(t) + nu * integral_0^t G_d(t - t') e^{-nu t'} G_s(t') dt'

    modelling muon hopping away from the F-mu-F site (or fluctuation of the
    coupling) at rate ``nu`` (µs⁻¹).  ``nu = 0`` reduces exactly to the static
    :func:`linear_fmuf_polarization`; large ``nu`` gives motional narrowing
    ``exp(-2 omega_d^2 t / nu)``.  Unlike WiMDA, the integration horizon is
    derived from the requested time range rather than a user-visible ``tmax``
    parameter, and the solution is cached per ``(r_muF, nu, tmax)``.
    """
    t_arr = np.asarray(t, dtype=float)
    scalar = t_arr.ndim == 0
    tt = np.atleast_1d(np.abs(t_arr))
    nu = abs(float(nu))
    if nu <= 1e-9:
        gd = np.asarray(linear_fmuf_polarization(tt, r_muF), dtype=float)
        return float(gd[0]) if scalar else gd

    omega_d = omega_d_mu_f_rad_per_us(r_muF)
    if nu > _DYN_FMUF_NU_SWITCH:
        gd = np.exp(np.clip(-2.0 * omega_d * omega_d * tt / nu, -700.0, 0.0))
        return float(gd[0]) if scalar else gd

    tmax = float(max(tt.max(), 1e-6))
    key = (round(float(r_muF), 9), round(nu, 6), round(tmax, 5))
    cached = _DYN_FMUF_CACHE.get(key)
    if cached is None:
        # Step resolves both the collision rate and the fastest static
        # oscillation, (3+sqrt(3))/2 * omega_d.
        h_des = min(0.02, 0.02 / max(nu, 1e-3), 0.1 / max(omega_d, 1e-3))
        n = int(min(max(round(tmax / h_des) + 1, 64), 20001))
        grid = np.linspace(0.0, tmax, n)
        h = grid[1] - grid[0] if n > 1 else tmax
        gs = np.asarray(linear_fmuf_polarization(grid, r_muF), dtype=float)
        gd_grid = _strong_collision_solve(gs, nu, h)
        if len(_DYN_FMUF_CACHE) >= _DYN_FMUF_CACHE_MAX:
            _DYN_FMUF_CACHE.pop(next(iter(_DYN_FMUF_CACHE)))
        _DYN_FMUF_CACHE[key] = (grid, gd_grid)
        cached = (grid, gd_grid)
    grid, gd_grid = cached
    gd = np.interp(tt, grid, gd_grid)
    return float(gd[0]) if scalar else gd


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


def _validate_triangle_geometry(r_muF: float, r3: float, phi3_deg: float) -> None:
    if r_muF <= 0.0 or r3 <= 0.0:
        raise ValueError("r_muF and r3 must be positive")
    if phi3_deg <= 0.0 or phi3_deg > 180.0:
        raise ValueError("phi3 must be in the range (0, 180] degrees")


@lru_cache(maxsize=256)
def _triangle_spectral_terms_cached(
    r_key: float,
    r3_key: float,
    phi3_key: float,
    num_beta: int,
    num_alpha: int,
    num_gamma: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r_muF = float(r_key)
    r3 = float(r3_key)
    phi3_deg = float(phi3_key)
    _validate_triangle_geometry(r_muF, r3, phi3_deg)

    phi3 = np.deg2rad(phi3_deg)
    # Muon at the origin; collinear F-mu-F pair along z; third fluorine in the
    # x-z plane at angle phi3 from the F-mu-F axis.
    positions = np.array(
        [
            [0.0, 0.0, r_muF],  # F1
            [0.0, 0.0, -r_muF],  # F2
            [r3 * np.sin(phi3), 0.0, r3 * np.cos(phi3)],  # F3
        ],
        dtype=float,
    )

    # Pair list in four-spin indices (0 = muon, 1..3 = fluorines), with
    # couplings from the pair separations (all six pairs included).
    mu_pairs = [
        ((0, k + 1), omega_d_mu_f_rad_per_us(float(np.linalg.norm(positions[k])))) for k in range(3)
    ]
    ff_pairs = []
    for a in range(3):
        for b in range(a + 1, 3):
            sep = float(np.linalg.norm(positions[b] - positions[a]))
            if sep <= 1.0e-9:
                raise ValueError("Invalid geometry: F-F distance is zero or too small")
            ff_pairs.append(((a + 1, b + 1), omega_d_f_f_rad_per_us(sep)))

    # Direction vectors in the crystal frame (constant), rotated per orientation.
    directions = {
        (0, 1): positions[0],
        (0, 2): positions[1],
        (0, 3): positions[2],
        (1, 2): positions[1] - positions[0],
        (1, 3): positions[2] - positions[0],
        (2, 3): positions[2] - positions[1],
    }
    couplings = dict(mu_pairs + ff_pairs)

    rotations, orientation_weights = _powder_rotations(num_beta, num_alpha, num_gamma)

    frequencies_list: list[NDArray[np.float64]] = []
    amplitudes_list: list[NDArray[np.float64]] = []
    dim = MUON_SIGMA_Z_FOUR_SPIN.shape[0]

    for idx, orient_weight in enumerate(orientation_weights):
        rot = rotations[idx]
        pair_terms = [
            (pair, couplings[pair], rot @ direction) for pair, direction in directions.items()
        ]
        hamiltonian = four_spin_hamiltonian_rad_per_us(pair_terms)
        evals, evecs = np.linalg.eigh(hamiltonian)

        sigma_mu_z_eigenbasis = evecs.conj().T @ MUON_SIGMA_Z_FOUR_SPIN @ evecs
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


def fmuf_triangle_polarization(
    t: NDArray[np.float64],
    r_muF: float,
    r3: float,
    phi3: float,
) -> NDArray[np.float64]:
    """Powder-averaged polarization for F-mu-F plus a third fluorine (16-dim).

    A collinear F-mu-F pair (both fluorines at ``r_muF``, as in
    :func:`linear_fmuf_polarization`) plus a third fluorine at distance ``r3``
    and angle ``phi3`` (degrees) from the F-mu-F axis, solved exactly in the
    16-dimensional muon + 3F Hilbert space with **all** pairwise dipolar
    couplings (mu-F and F-F) and a proper powder average.

    This supersedes WiMDA's ``F-u-F-F ZF PCR``, which neglects the F-F
    couplings and approximates the powder average by (P_z + 2 P_x)/3 for a
    single crystal orientation pair; see
    ``docs/porting/wimda-fit-function-parity/comparison.md``.  As
    ``r3 -> infinity`` the result approaches the linear F-mu-F polarization.
    """
    t_arr = np.asarray(t, dtype=float)

    r_key = round(float(r_muF), _CACHE_KEY_DECIMALS)
    r3_key = round(float(r3), _CACHE_KEY_DECIMALS)
    phi3_key = round(float(phi3), _CACHE_KEY_DECIMALS)

    freqs, amps = _triangle_spectral_terms_cached(
        r_key,
        r3_key,
        phi3_key,
        _DEFAULT_NUM_BETA,
        _DEFAULT_NUM_ALPHA,
        _DEFAULT_NUM_GAMMA,
    )
    cos_terms = np.cos(np.outer(freqs, t_arr))
    return np.asarray(amps @ cos_terms, dtype=float)


__all__ = [
    "mu_f_polarization",
    "linear_fmuf_polarization",
    "dynamic_fmuf_polarization",
    "general_fmuf_polarization",
    "fmuf_triangle_polarization",
]
