"""FFT of μSR asymmetry data."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fourier.window import apply_window


def fft_asymmetry(
    dataset: MuonDataset,
    window: str = "none",
    padding_factor: int = 1,
    t_min: float | None = None,
    t_max: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute the FFT of the asymmetry signal.

    Parameters
    ----------
    dataset
        The time-domain data.
    window
        Apodization window name (``"none"``, ``"gaussian"``, ``"hann"``, ``"cosine"``).
    padding_factor
        Zero-pad the signal to ``padding_factor × N``.
    t_min, t_max
        Restrict the time range before transforming.

    Returns
    -------
    frequencies, real_part, magnitude
        Frequency axis (MHz) and the real and magnitude spectra.
    """
    ds = dataset.time_range(t_min, t_max) if (t_min is not None or t_max is not None) else dataset

    signal = ds.asymmetry.copy()
    dt = np.mean(np.diff(ds.time)) if len(ds.time) > 1 else 1.0

    # Apply window
    if window != "none":
        signal = apply_window(signal, window)

    # Zero-pad
    n = len(signal)
    n_padded = n * max(padding_factor, 1)

    spectrum = np.fft.rfft(signal, n=n_padded)
    freqs = np.fft.rfftfreq(n_padded, d=dt)  # MHz (since dt is in μs)

    return freqs, spectrum.real, np.abs(spectrum)
