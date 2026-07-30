"""Apodization / window functions for Fourier analysis."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_WINDOWS: dict[str, callable] = {}
_FFT_FILTER_MODES = frozenset({"none", "lorentzian", "gaussian"})


def _register_window(name: str):
    def decorator(fn):
        _WINDOWS[name] = fn
        return fn

    return decorator


def apply_window(signal: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    """Apply a named window function to *signal*.

    Supported names: ``"gaussian"``, ``"hann"``, ``"cosine"``, ``"lorentzian"``.

    .. warning::

       ``"hann"`` and ``"cosine"`` are **symmetric tapers**: they are zero at
       the first sample and rise as ``(t/T)²``, so they DELETE early-time
       signal. They are for late-time / narrow-line work, where the record is
       long compared with the signal's lifetime. A heavily damped oscillation
       (lifetime ≪ record length) can vanish entirely under a Hann window while
       being many-sigma without it. For that case use no window with a time
       crop, or the exponential filter (:func:`apply_fft_filter` with
       ``mode="lorentzian"`` and ``time_constant_us ≈ 1/λ``), which is weight-1
       at ``t = 0`` and is the matched apodisation for a Lorentzian line. The
       FFT prepare path warns about this automatically —
       :class:`~asymmetry.core.fourier.apodisation.ApodisationEarlySignalWarning`.
    """
    fn = _WINDOWS.get(name.lower())
    if fn is None:
        known = ", ".join(sorted(_WINDOWS))
        raise ValueError(f"Unknown window {name!r}. Available: {known}")
    return fn(signal)


def apply_fft_filter(
    signal: NDArray[np.float64],
    time_us: NDArray[np.float64],
    *,
    mode: str = "none",
    start_time_us: float = 0.0,
    time_constant_us: float = 1.5,
) -> NDArray[np.float64]:
    """Apply WiMDA-style FFT apodisation to a time-domain signal.

    Unlike the symmetric tapers of :func:`apply_window`, these filters are
    weight-1 at ``t = 0`` (with ``start_time_us = 0``) and decay from there, so
    they preserve early-time signal. ``mode="lorentzian"`` with
    ``time_constant_us = 1/λ`` is the **matched** apodisation for a
    Lorentzian-broadened line relaxing at ``λ``: it maximises the line's peak
    signal-to-noise, at the cost of roughly doubling its apparent width.

    Parameters
    ----------
    signal
        Time-domain signal values.
    time_us
        Matching time axis in microseconds.
    mode
        One of ``"none"``, ``"lorentzian"``, or ``"gaussian"``.
    start_time_us
        Filter start time. A value above zero creates WiMDA's softened step at
        the chosen start time.
    time_constant_us
        Filter time constant ``tau`` in microseconds.
    """
    values = np.asarray(signal, dtype=np.float64)
    times = np.asarray(time_us, dtype=np.float64)
    if values.shape != times.shape:
        raise ValueError("FFT filter time axis must match the signal shape.")

    mode_key = str(mode).strip().lower()
    if mode_key not in _FFT_FILTER_MODES:
        known = ", ".join(sorted(_FFT_FILTER_MODES))
        raise ValueError(f"Unknown FFT filter mode {mode!r}. Available: {known}")
    if mode_key == "none":
        return values.copy()

    tau = float(time_constant_us)
    if not np.isfinite(tau) or tau <= 0.0:
        return values.copy()

    start = float(start_time_us)
    if mode_key == "lorentzian":
        if start > 0.0:
            numerator = 1.0 + np.exp(-start / tau)
            weights = numerator / (1.0 + np.exp((times - start) / tau))
        else:
            weights = np.exp(-times / tau)
    else:
        if start > 0.0:
            numerator = 1.0 + np.exp((start / tau) ** 2)
            weights = numerator / (1.0 + np.exp(((times - start) / tau) ** 2))
        else:
            weights = np.exp(-np.square(times / tau))

    return values * weights


@_register_window("gaussian")
def _gaussian(signal: NDArray) -> NDArray:
    n = len(signal)
    sigma = n / 6.0
    w = np.exp(-0.5 * ((np.arange(n) - n / 2) / sigma) ** 2)
    return signal * w


@_register_window("hann")
def _hann(signal: NDArray) -> NDArray:
    return signal * np.hanning(len(signal))


@_register_window("cosine")
def _cosine(signal: NDArray) -> NDArray:
    n = len(signal)
    return signal * np.sin(np.pi * np.arange(n) / n)


@_register_window("lorentzian")
def _lorentzian(signal: NDArray) -> NDArray:
    n = len(signal)
    gamma = n / 6.0
    w = gamma / (gamma + np.abs(np.arange(n) - n / 2))
    return signal * w
