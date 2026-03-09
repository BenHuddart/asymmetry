"""Tests for Fourier analysis modules."""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fourier.fft import fft_asymmetry
from asymmetry.core.fourier.maxent import maxent
from asymmetry.core.fourier.window import apply_window


def _dataset(n: int = 128) -> MuonDataset:
    t = np.linspace(0.0, 10.0, n)
    a = 0.2 * np.cos(2 * np.pi * 0.4 * t)
    e = np.full(n, 0.01)
    return MuonDataset(time=t, asymmetry=a, error=e, metadata={"run_number": 1})


@pytest.mark.parametrize("name", ["gaussian", "hann", "cosine", "lorentzian"])
def test_apply_window_supported(name: str) -> None:
    signal = np.ones(32, dtype=float)
    out = apply_window(signal, name)
    assert out.shape == signal.shape
    assert np.all(np.isfinite(out))


def test_apply_window_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown window"):
        apply_window(np.ones(16, dtype=float), "not-a-window")


def test_fft_asymmetry_basic_shape() -> None:
    ds = _dataset(128)
    f, real, mag = fft_asymmetry(ds)
    assert len(f) == 65
    assert len(real) == 65
    assert len(mag) == 65
    assert np.all(mag >= 0)


def test_fft_asymmetry_padding_and_time_window() -> None:
    ds = _dataset(120)
    f, real, mag = fft_asymmetry(ds, window="hann", padding_factor=2, t_min=2.0, t_max=8.0)
    assert len(f) == len(real) == len(mag)
    assert len(f) > 60


def test_maxent_stub_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        maxent(_dataset(32))
