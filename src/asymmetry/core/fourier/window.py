"""Apodization / window functions for Fourier analysis."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_WINDOWS: dict[str, callable] = {}


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
