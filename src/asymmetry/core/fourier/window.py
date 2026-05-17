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
