"""Cancellation preserves the seeded spectral-moment estimator."""

import numpy as np
import pytest

from asymmetry.core.fourier import moments


def test_cancel_during_bootstrap(monkeypatch):
    x = np.linspace(1, 10, 101)
    amplitude = np.exp(-((x - 5) ** 2))
    calls = []
    original = moments._moment_scalars

    def counted(*args):
        calls.append(None)
        return original(*args)

    monkeypatch.setattr(moments, "_moment_scalars", counted)
    with pytest.raises(moments.MomentsCancelledError):
        moments.spectrum_moments(
            x,
            amplitude,
            errors=np.full_like(x, 0.01),
            x_range=None,
            cutoff_fraction=0.05,
            cancel_callback=lambda: len(calls) >= 4,
        )
    assert len(calls) == 4


def test_uncancelled_callback_preserves_seeded_output():
    x = np.linspace(1, 10, 101)
    kwargs = dict(errors=np.full_like(x, 0.01), x_range=None, cutoff_fraction=0.05, seed=31)
    expected = moments.spectrum_moments(x, np.exp(-((x - 5) ** 2)), **kwargs)
    actual = moments.spectrum_moments(
        x,
        np.exp(-((x - 5) ** 2)),
        **kwargs,
        cancel_callback=lambda: False,
    )
    assert actual == expected
