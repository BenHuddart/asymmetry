"""Burg all-poles (maximum-entropy / autoregressive) spectral estimation.

The all-poles method models the spectrum as

.. math::

    P(\\nu) = \\frac{P_m}{\\left|1 - \\sum_{k=1}^{m} a_k z^k\\right|^2},
    \\qquad z = e^{2\\pi i \\nu \\Delta t},

a function with *m* poles in the z-plane (Blundell, De Renzi, Lancaster & Pratt,
*Muon Spectroscopy*, OUP 2022, §15.5).  Because *m* can be far smaller than the
number of time points *N*, the method places poles exactly where the data demand
sharp lines, giving better intrinsic frequency resolution than the all-zeroes FFT
on short windows.  Burg's recursion (J. P. Burg, *Geophysics* **37**, 375, 1972)
estimates the coefficients from the forward/backward prediction residuals; the
Final Prediction Error (FPE) passes through a minimum versus *m*, which selects
the pole count.

This is a **diagnostic** estimator: it qualitatively super-resolves close lines
and its FPE-optimal pole count hints at the number of lines, but it can
spuriously split strong features and seed false baseline peaks, it offsets line
positions slightly, and it carries no uncertainties.  It is never the
quantitative result — frequency-domain fitting and MaxEnt are.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

#: Default pole-scan range for the FPE order search.
DEFAULT_ORDER_RANGE = (2, 40)


def burg_coefficients(signal: ArrayLike, order: int) -> tuple[NDArray[np.float64], float]:
    """Return the order-*m* Burg AR coefficients ``a_k`` and the power ``P_m``.

    The predictor is ``x[t] ≈ Σ aᵢ x[t−i]``.  From ``f = b = x`` and
    ``P₀ = ⟨x²⟩``, each order *k* takes the reflection coefficient
    ``κ = 2 Σ f[t] b[t−1] / Σ (f[t]² + b[t−1]²)`` over ``t = k…n−1`` (zero when the
    denominator vanishes), then sets ``P_k = P_{k−1}(1 − κ²)``, updates the
    coefficients by Levinson (``a_k = κ``, ``aᵢ −= κ a_{k−i}``) and the residuals by
    ``f[t] −= κ b[t−1]``, ``b[t] = b[t−1] − κ f[t]``.  The order is clamped to
    ``n − 1``; for ``n < 2`` or ``order < 1`` the coefficients are zero and the
    power is ``⟨x²⟩``.
    """
    x = np.asarray(signal, dtype=np.float64)
    order = int(order)
    n = x.size
    if n < 2 or order < 1:
        return np.zeros(max(order, 0), dtype=np.float64), float(np.mean(x**2)) if n else 0.0

    order = min(order, n - 1)
    a = np.zeros(order, dtype=np.float64)
    power = float(np.mean(x**2))
    # f[i] and b[i] are the residuals f[t] and b[t−1] for the same t, so they stay aligned.
    f, b = x[1:], x[:-1]
    for k in range(order):
        denominator = np.dot(f, f) + np.dot(b, b)
        kappa = 2.0 * np.dot(f, b) / denominator if denominator != 0.0 else 0.0
        power *= 1.0 - kappa**2
        a[:k] -= kappa * a[:k][::-1]
        a[k] = kappa
        f, b = (f - kappa * b)[1:], (b - kappa * f)[:-1]
    return a, float(power)


def ar_power_spectrum(
    a_coeffs: ArrayLike,
    power: float,
    freqs_mhz: ArrayLike,
    dt_us: float,
) -> NDArray[np.float64]:
    """Evaluate the all-poles spectrum amplitude ``√(P_m / |1 − Σ aₖzᵏ|²)``.

    The amplitude (square root of power) matches the ``(Power)^1/2`` display
    convention.  Frequencies are in MHz and ``dt_us`` in µs, so ``ν·Δt`` is the
    dimensionless cycle count per bin.
    """
    a = np.asarray(a_coeffs, dtype=np.float64)
    freqs = np.asarray(freqs_mhz, dtype=np.float64)
    if a.size == 0:
        return np.sqrt(np.full(freqs.shape, max(power, 0.0)))
    k = np.arange(1, a.size + 1)
    theta = 2.0 * np.pi * freqs[:, np.newaxis] * float(dt_us) * k[np.newaxis, :]
    transfer = 1.0 - np.sum(a[np.newaxis, :] * np.exp(1j * theta), axis=1)
    denom = np.abs(transfer) ** 2
    denom = np.where(denom > 0.0, denom, np.finfo(np.float64).tiny)
    return np.sqrt(np.maximum(power, 0.0) / denom)


def fpe_order_scan(signal: ArrayLike, orders: Iterable[int]) -> tuple[int | None, dict[int, float]]:
    """Return the FPE-optimal pole count and the FPE for each scanned order.

    The Final Prediction Error is ``FPE_m = P_m·(N+m) / ((N−m)·P₀·(N+1)/(N−1))``;
    the optimum is the order minimising it.  Returns ``(best_order, {order: fpe})``.
    """
    x = np.asarray(signal, dtype=np.float64)
    n = x.size
    p0 = float(np.dot(x, x)) / n if n else 0.0
    fp1 = p0 * (n + 1) / (n - 1) if n > 1 else 0.0
    fpe: dict[int, float] = {}
    best: int | None = None
    best_val = np.inf
    for m in orders:
        m = int(m)
        _, power = burg_coefficients(x, m)
        if n > m and fp1 > 0.0:
            value = power * (n + m) / ((n - m) * fp1)
        else:
            value = np.inf
        fpe[m] = float(value)
        if value < best_val:
            best_val = value
            best = m
    return best, fpe


def burg_spectrum(
    signal: ArrayLike,
    freqs_mhz: ArrayLike,
    dt_us: float,
    *,
    order_range: tuple[int, int] = DEFAULT_ORDER_RANGE,
) -> tuple[NDArray[np.float64], int, bool]:
    """Return ``(spectrum, best_order, hit_boundary)`` for the Burg estimate.

    Scans the pole count over ``order_range`` (inclusive), selects the FPE
    minimum, and evaluates the AR spectrum at *freqs_mhz*.  ``hit_boundary`` is
    true when the optimum lands on a scan endpoint — a sign the range is too
    narrow (WiMDA warns in the same case).
    """
    x = np.asarray(signal, dtype=np.float64)
    n = x.size
    lo = max(1, int(order_range[0]))
    hi = max(lo, int(order_range[1]))
    if n >= 2:
        hi = min(hi, n - 1)
    orders = list(range(lo, hi + 1))
    if not orders:
        orders = [lo]

    best, _fpe = fpe_order_scan(x, orders)
    if best is None:
        best = orders[0]
    a, power = burg_coefficients(x, best)
    spectrum = ar_power_spectrum(a, power, freqs_mhz, dt_us)
    hit_boundary = len(orders) > 1 and best in (orders[0], orders[-1])
    return spectrum, int(best), bool(hit_boundary)
