"""Muon-fluorine polarization functions for entangled spin states."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.fitting.models import _exponential_sum, _strong_collision_modes
from asymmetry.core.fitting.muon_fluorine.dipolar import (
    _PAIR_F1_F2,
    _PAIR_ISO,
    _PAIR_ISO_FOUR,
    _PAIR_MU_F1,
    _PAIR_MU_F2,
    _PAIR_TENSOR,
    _PAIR_TENSOR_FOUR,
    MUON_SIGMA_Z_FOUR_SPIN,
    MUON_SIGMA_Z_THREE_SPIN,
    omega_d_f_f_rad_per_us,
    omega_d_mu_f_rad_per_us,
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


def _linear_fmuf_realisation(
    r_muF: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """State-space ``(A, b, c)`` of the static collinear F-mu-F polarization.

    :func:`linear_fmuf_polarization` is a constant plus three cosines, so it is
    ``c^T exp(A t) b`` for a seven-state system: one constant mode and a 2x2
    rotation block per frequency (``x_k = cos``, ``x_{k+1} = sin``).
    """
    omega_d = omega_d_mu_f_rad_per_us(r_muF)
    sqrt3 = np.sqrt(3.0)
    frequencies = (sqrt3 * omega_d, 0.5 * (3.0 - sqrt3) * omega_d, 0.5 * (3.0 + sqrt3) * omega_d)
    amplitudes = (1.0, 1.0 - 1.0 / sqrt3, 1.0 + 1.0 / sqrt3)
    a_mat = np.zeros((7, 7))
    b = np.zeros(7)
    c = np.zeros(7)
    b[0] = 0.5
    c[0] = 1.0
    for i, (omega, amplitude) in enumerate(zip(frequencies, amplitudes, strict=True)):
        k = 1 + 2 * i
        a_mat[k, k + 1] = -omega
        a_mat[k + 1, k] = omega
        b[k] = amplitude / 6.0
        c[k] = 1.0
    return a_mat, b, c


def dynamic_fmuf_polarization(
    t: NDArray[np.float64], r_muF: float, nu: float
) -> NDArray[np.float64]:
    """Strong-collision dynamicized linear F-mu-F polarization (WiMDA ``dyn F-u-F``).

    The static collinear F-mu-F polarization ``G_s`` (eqn 4.81 of Blundell, De
    Renzi, Lancaster & Pratt, *Muon Spectroscopy*, OUP 2022) dynamicized by the
    strong-collision integral equation (eqn 5.30):

        G_d(t) = e^{-nu t} G_s(t) + nu * integral_0^t G_d(t - t') e^{-nu t'} G_s(t') dt'

    modelling muon hopping away from the F-mu-F site (or fluctuation of the
    coupling) at rate ``nu`` (µs⁻¹).  Because ``G_s`` is a constant plus three
    cosines, the equation has an **exact closed-form solution**: a sum of seven
    exponentials whose rates and weights come from a 7x7 eigenproblem
    (:func:`asymmetry.core.fitting.models._strong_collision_modes`).  It is
    evaluated directly at the requested times -- no integration grid, cache,
    call-count-dependent accuracy or fast-fluctuation crossover -- and is valid
    at every rate: ``nu = 0`` reduces exactly to the static
    :func:`linear_fmuf_polarization`, and for ``nu >> omega_d`` it tends to the
    motional-narrowing exponential ``exp(-2 omega_d^2 t / nu)`` (with the
    Abragam-form quadratic onset at short times).  WiMDA instead integrates on
    a user-visible ``tmax`` grid and jumps to the bare narrowing exponential at
    a fixed rate, leaving a discontinuity in the model.
    """
    t_arr = np.asarray(t, dtype=float)
    scalar = t_arr.ndim == 0
    tt = np.atleast_1d(np.abs(t_arr))
    nu = abs(float(nu))
    if nu <= 1e-9:
        gd = np.asarray(linear_fmuf_polarization(tt, r_muF), dtype=float)
        return float(gd[0]) if scalar else gd
    lam, weights = _strong_collision_modes(*_linear_fmuf_realisation(r_muF), nu)
    gd = np.asarray(_exponential_sum(tt, lam, weights), dtype=float)
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

    # Build all orientations' 8x8 Hamiltonians in one shot, mirroring the
    # four-spin triangle path (`_triangle_spectral_terms_cached`): for each
    # pair the coupling term is c * [iso - 3 (n.S_i)(n.S_j)] with n the
    # per-orientation unit direction, so stacking the outer products n n^T
    # over orientations turns the per-orientation Python loop into a single
    # einsum against the precomputed (3, 3, 8, 8) pair tensors, followed by
    # one batched eigh — ~4x faster per cache miss than the loop (this is the
    # hot path of a fit, since every minimizer step changes the geometry and
    # misses the cache).
    n_orient = rotations.shape[0]
    dim = MUON_SIGMA_Z_THREE_SPIN.shape[0]
    hamiltonians = np.zeros((n_orient, dim, dim), dtype=complex)
    for coupling, n_vecs, pair in (
        (coupling_mu_f1, n_mu_f1, _PAIR_MU_F1),
        (coupling_mu_f2, n_mu_f2, _PAIR_MU_F2),
        (coupling_f1_f2, n_f1_f2, _PAIR_F1_F2),
    ):
        outer = (n_vecs[:, :, None] * n_vecs[:, None, :]).reshape(n_orient, 9)
        aniso = (outer @ _PAIR_TENSOR[pair].reshape(9, dim * dim)).reshape(n_orient, dim, dim)
        hamiltonians += coupling * (_PAIR_ISO[pair][None, :, :] - 3.0 * aniso)

    evals, evecs = np.linalg.eigh(hamiltonians)  # (n, 8), (n, 8, 8)
    sigma_eig = evecs.conj().transpose(0, 2, 1) @ MUON_SIGMA_Z_THREE_SPIN @ evecs
    transition_weights = (np.abs(sigma_eig) ** 2) / float(dim)  # (n, 8, 8)
    omega_mn = (evals[:, :, None] - evals[:, None, :]).real

    frequencies = omega_mn.reshape(-1)
    amplitudes = (orientation_weights[:, None, None] * transition_weights).reshape(-1).real

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


def _validate_triangle_geometry(r_muF: float, r3: float) -> None:
    """Reject only genuinely unphysical trial geometries.

    Any ``phi3`` is geometrically valid (the powder average makes the angle
    periodic and mirror-symmetric, with [0, 180] the canonical range), so no
    angular range is enforced — a range rejection would hand the minimiser a
    flat penalty plateau just past the boundary.  Coincident nuclei are caught
    separately via the pair-separation guard.
    """
    if r_muF <= 0.0 or r3 <= 0.0:
        raise ValueError("r_muF and r3 must be positive")


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
    _validate_triangle_geometry(r_muF, r3)

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
    dim = MUON_SIGMA_Z_FOUR_SPIN.shape[0]

    # Build all orientations' Hamiltonians in one shot: for each pair the
    # coupling term is c * [iso - 3 (n.S_i)(n.S_j)] with n the rotated
    # direction, so stacking the outer products n n^T over orientations turns
    # the per-orientation Python loop into a single einsum against the
    # precomputed (3, 3, 16, 16) pair tensors, followed by one batched eigh —
    # ~4x faster per cache miss than the loop (this is the hot path of a fit,
    # since every minimizer step changes the geometry and misses the cache).
    n_orient = rotations.shape[0]
    hamiltonians = np.zeros((n_orient, dim, dim), dtype=complex)
    for pair, direction in directions.items():
        coupling = couplings[pair]
        n_vecs = rotations @ (np.asarray(direction, dtype=float) / np.linalg.norm(direction))
        outer = (n_vecs[:, :, None] * n_vecs[:, None, :]).reshape(n_orient, 9)
        aniso = (outer @ _PAIR_TENSOR_FOUR[pair].reshape(9, dim * dim)).reshape(n_orient, dim, dim)
        hamiltonians += coupling * (_PAIR_ISO_FOUR[pair][None, :, :] - 3.0 * aniso)

    evals, evecs = np.linalg.eigh(hamiltonians)  # (n, 16), (n, 16, 16)
    sigma_eig = evecs.conj().transpose(0, 2, 1) @ MUON_SIGMA_Z_FOUR_SPIN @ evecs
    transition_weights = (np.abs(sigma_eig) ** 2) / float(dim)  # (n, 16, 16)
    omega_mn = (evals[:, :, None] - evals[:, None, :]).real

    frequencies = omega_mn.reshape(-1)
    amplitudes = (orientation_weights[:, None, None] * transition_weights).reshape(-1).real

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
