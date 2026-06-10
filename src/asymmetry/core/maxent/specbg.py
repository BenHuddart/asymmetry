"""Zero-frequency lineshape subtraction for ZF/LF field-distribution displays.

In zero/longitudinal field the MaxEnt spectrum is dominated by a strong central
(zero-frequency) feature — the static peak of the field distribution.  Weak
satellite or precession structure sits on its flank and is hard to see.  This
display-time tool subtracts a model of that central peak, a zero-centred
pseudo-Voigt scaled to the measured central amplitude, so the surrounding
lineshape stands out (after WiMDA's ``SpecBG``).

It is **display-only**: it operates on a copy of the plotted spectrum, never the
engine's reconstructed spectrum.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.data.dataset import MuonDataset

if TYPE_CHECKING:
    from asymmetry.core.maxent.engine import MaxEntConfig

# Empirical factor relating the edit's Gaussian-width value to the 1/e width of
# the subtracted Gaussian (carried verbatim from WiMDA's SpecBG; magic number).
_GAUSSIAN_WIDTH_FACTOR = 1.201

_MIN_WIDTH = 1.0e-3


def subtract_zero_frequency(
    frequencies: NDArray[np.float64],
    spectrum: NDArray[np.float64],
    *,
    gaussian_width: float,
    lorentzian_width: float,
    lorentzian_fraction: float,
    anchor: float | None = None,
) -> NDArray[np.float64]:
    """Return *spectrum* with a zero-centred pseudo-Voigt central peak removed.

    The subtracted shape is

        Δ(x) = a₀·[ (1−η)·exp(−(x/(w_G·1.201))²) + η/(1 + (x/w_L)²) ],

    where ``x`` is the axis coordinate (the spectrum is centred so the static
    peak is near zero), ``w_G``/``w_L`` are the Gaussian/Lorentzian widths in the
    axis units, ``η`` the Lorentzian fraction (clamped to [0, 1]), and ``a₀`` the
    central amplitude — the spectrum value at the bin nearest zero unless an
    explicit *anchor* is given.  Returns a new array; the input is unchanged.
    """
    freqs = np.asarray(frequencies, dtype=np.float64)
    values = np.asarray(spectrum, dtype=np.float64).copy()
    if freqs.size == 0 or values.size == 0:
        return values

    fraction = float(np.clip(lorentzian_fraction, 0.0, 1.0))
    gauss = max(float(gaussian_width), _MIN_WIDTH) * _GAUSSIAN_WIDTH_FACTOR
    lorentz = max(float(lorentzian_width), _MIN_WIDTH)

    if anchor is None:
        anchor = float(values[int(np.argmin(np.abs(freqs)))])

    gaussian = np.exp(-((freqs / gauss) ** 2))
    lorentzian = 1.0 / (1.0 + (freqs / lorentz) ** 2)
    background = anchor * ((1.0 - fraction) * gaussian + fraction * lorentzian)
    return values - background


def apply_maxent_specbg(dataset: MuonDataset, config: MaxEntConfig) -> MuonDataset:
    """Subtract the SpecBG zero-frequency model from a MaxEnt spectrum dataset.

    Display-only and a no-op unless SpecBG is enabled in ZF/LF mode; mutates and
    returns the (freshly built) spectrum dataset in place.  This is the **single
    application point** for SpecBG — both the on-demand representation
    (:class:`FrequencyMaxEnt`) and the live worker path reach it through
    :meth:`MaxEntResult.as_dataset`, so the two cannot drift.
    """
    if not (config.specbg_enabled and config.mode == "zf_lf"):
        return dataset
    frequencies = np.asarray(dataset.time, dtype=float)
    if frequencies.size == 0:
        return dataset
    # SpecBG subtracts a *zero-centred* model of the static peak, so it only
    # makes sense when the spectrum window actually reaches zero frequency (true
    # ZF, window from 0).  For an LF window centred on the Larmor line the peak
    # is not at zero, so skip rather than subtract a meaningless edge model.
    width = max(config.specbg_gaussian_width_mhz, config.specbg_lorentzian_width_mhz, 1.0e-3)
    if float(np.min(np.abs(frequencies))) > width:
        return dataset
    dataset.asymmetry = np.asarray(
        subtract_zero_frequency(
            frequencies,
            dataset.asymmetry,
            gaussian_width=config.specbg_gaussian_width_mhz,
            lorentzian_width=config.specbg_lorentzian_width_mhz,
            lorentzian_fraction=config.specbg_lorentzian_fraction,
        ),
        dtype=float,
    )
    return dataset
