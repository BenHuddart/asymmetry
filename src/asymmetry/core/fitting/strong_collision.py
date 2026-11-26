"""Exponential-weighted product integration of the strong-collision renewal equation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_triangular, toeplitz

if TYPE_CHECKING:
    from scipy.interpolate import CubicSpline

_BLOCK_SIZE = 512
_GAUSS_RULES = {order: np.polynomial.legendre.leggauss(order) for order in (8, 32)}


class EvaluationWorkLimitError(ValueError):
    """The selected model needs more numerical work than its evaluation-work budget."""


KT_GRID_SAMPLE_BUDGET = 1 << 19
KT_CACHE_ENTRIES = 16


def _renewal_toeplitz(diagonal: float, kernel: NDArray, forcing: NDArray) -> NDArray:
    """Solve a causal Toeplitz system with recursively blocked FFT substitution.

    Every coupling from time j to a later time i is added once, at the first
    interval split separating j and i. Leaves solve the same triangular block;
    the recursive schedule avoids repeatedly transforming the entire history.
    """
    result = np.empty_like(forcing)
    rhs = forcing.copy()

    def solve_interval(start: int, end: int) -> None:
        size = end - start
        if size <= _BLOCK_SIZE:
            column = np.r_[diagonal, -kernel[: size - 1]]
            row = np.r_[diagonal, np.zeros(size - 1)]
            result[start:end] = solve_triangular(
                toeplitz(column, row), rhs[start:end], lower=True, check_finite=False
            )
            return
        middle = start + size // 2
        solve_interval(start, middle)
        left = middle - start
        fft_size = 1 << (size + left - 2).bit_length()
        convolution = np.fft.irfft(
            np.fft.rfft(result[start:middle], fft_size) * np.fft.rfft(kernel[: size - 1], fft_size),
            fft_size,
        )
        rhs[middle:end] += convolution[left - 1 : size - 1]
        solve_interval(middle, end)

    solve_interval(0, forcing.size)
    return result


def _collision_kernel_intervals(
    static: CubicSpline, n: int, nu: float, h: float
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Sample static polarisation with exp(-nu*s) integrated on each interval.

    Each weight includes nu*ds. Rescaling the quadrature by nu*h resolves the
    exponential even when its decay is much faster than the dynamic time grid.
    Beyond 40 lifetimes its omitted normalised mass is below 5e-18.
    """
    from scipy.interpolate import PPoly

    z = nu * h
    x, weights = _GAUSS_RULES[8 if z <= 1.0 else 32]
    limit = min(z, 40.0)
    fraction = (x + 1.0) * (limit / (2.0 * z))
    weights = weights * (limit / 2.0) * np.exp(-z * fraction)
    index = np.arange(n - 1)
    coefficients = static.c.copy()
    coefficients[-1] -= 1.0
    deficit = PPoly(coefficients, static.x)((index[:, None] + fraction) * h)
    attenuation = np.exp(-z * index)
    return fraction, weights, deficit, attenuation


def _linear_collision_startup(static: CubicSpline, nu: float, h: float) -> NDArray:
    """Resolve the two initial steps before quadratic history is available."""
    subdivisions = 128
    step = h / subdivisions
    n = 2 * subdivisions + 1
    fraction, weights, deficit, attenuation = _collision_kernel_intervals(static, n, nu, step)
    values = 1.0 + deficit
    alpha = attenuation * (values @ (weights * (1.0 - fraction)))
    beta = attenuation * (values @ (weights * fraction))
    z = nu * step
    diagonal = -np.expm1(-z) / z - np.sum(weights * (1.0 - fraction) * deficit[0])
    time = np.arange(1, n) * step
    forcing = np.exp(-nu * time) * static(time) + beta
    solution = _renewal_toeplitz(diagonal, alpha[1:] + beta[:-1], forcing)
    return np.r_[1.0, solution][::subdivisions]


def strong_collision_polarisation(
    gs: NDArray, nu: float, h: float, *, initial_derivative: float | None = None
) -> NDArray:
    r"""Solve G=f+nu*(f*G), f=exp(-nu*t)*Gs, at t=i*h.

    Quadratic interpolation of G under the convolution uses exact exponential
    weighting and a cubic interpolant of the supplied static kernel. Unlike
    direct trapezoidal integration of exp(-nu*t), it preserves Gs=1 exactly
    and retains the physical rate at large nu. Its stiff limit is BDF2,
    rather than a frozen rate or a different physical relaxation function.
    A known static initial derivative clamps its spline, preserving the
    Gaussian zero slope or the Lorentzian cusp in the fast-collision limit.
    """
    gs = np.asarray(gs, dtype=float)
    if gs.size == 1 or nu == 0.0 or nu * h == 0.0:
        return gs.copy()
    if gs.size < 4:
        raise ValueError(
            "Strong-collision product integration requires at least four static samples"
        )
    from scipy.interpolate import CubicSpline

    n = gs.size
    boundary = (
        "not-a-knot" if initial_derivative is None else ((1, initial_derivative), "not-a-knot")
    )
    static = CubicSpline(np.arange(n) * h, gs, bc_type=boundary)
    fraction, weights, deficit, attenuation = _collision_kernel_intervals(static, n, nu, h)
    values = 1.0 + deficit
    l0 = (fraction - 1.0) * (fraction - 2.0) / 2.0
    l1 = fraction * (2.0 - fraction)
    l2 = fraction * (fraction - 1.0) / 2.0
    alpha = attenuation * (values @ (weights * l0))
    beta = attenuation * (values @ (weights * l1))
    gamma = attenuation * (values @ (weights * l2))
    # Write 1-alpha[0] as positive constant-kernel moments plus its deficit,
    # so the denominator remains resolved when nu*h is very large.
    diagonal = (
        np.exp(-nu * h)
        + np.sum(weights * (1.5 * fraction - 0.5 * fraction**2))
        - np.sum(weights * l0 * deficit[0])
    )
    _, first, second = _linear_collision_startup(static, nu, h)
    kernel = alpha[1:] + beta[:-1]
    kernel[1:] += gamma[:-2]
    time = np.arange(3, n) * h
    forcing = (
        np.exp(-nu * time) * gs[3:]
        + beta[2:]
        + gamma[1:-1]
        + gamma[2:] * (3.0 + second - 3.0 * first)
        + kernel[1:] * first
        + kernel[:-1] * second
    )
    solution = _renewal_toeplitz(diagonal, kernel[: n - 4], forcing)
    return np.r_[1.0, first, second, solution]
