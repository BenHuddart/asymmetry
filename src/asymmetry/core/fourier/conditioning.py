"""Post-FFT spectrum conditioning: pulse compensation, baseline, exclusions.

These are the WiMDA spectrum-conditioning steps applied to a finished Fourier
display channel, on the canonical frequency axis (MHz).  They are pure NumPy and
carry no Qt dependency; the GUI panel and the project-recompute path both reach
them through :func:`apply_spectrum_conditioning`.

Three steps, applied in WiMDA's order (compensation → baseline → exclusions):

* **Pulse frequency-response compensation.** A pulsed muon source spreads the
  muon arrival over a finite pulse, which suppresses high frequencies before they
  are recorded — the pulse acts as a passband filter (Blundell, De Renzi,
  Lancaster & Pratt, *Muon Spectroscopy*, OUP 2022, §15.5).  Compensation divides
  each bin by the per-frequency pulse amplitude ``R(ν)`` from
  :mod:`asymmetry.core.maxent.pulse` — the inverse of the same response the
  MaxEnt forward model folds into its kernel, so the FFT and MaxEnt spectra share
  one pulse model.  ``1/R`` diverges as ``R → 0`` near the first node of the pulse
  transform, so the gain is capped and the spectrum is cut off at the node: beyond
  it the pulse has destroyed the information and no correction is physical.  This
  replaces WiMDA's unbounded Gaussian ``exp((πfτ)²)`` factor.

* **Robust baseline offset.** Iterative σ-clipping (the literature-standard robust
  continuum estimator — Sánchez-Monge *et al.*, *Astron. Astrophys.* **609**, A101
  (2018), STATCONT) finds the spectrum's baseline by repeatedly re-estimating the
  location and width of the inlier set until σ converges.  WiMDA's single-pass 2σ
  clip is the one-iteration, mean-location special case (``"wimda"`` mode).

* **Frequency-range exclusions.** Zeroing of bins inside symmetric
  ``(centre, half-width)`` windows, delegated to
  :func:`asymmetry.core.fourier.fft.exclude_frequency_ranges`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from asymmetry.core.fourier.fft import exclude_frequency_ranges
from asymmetry.core.maxent.pulse import pulse_amplitude_phase

#: Baseline estimator modes accepted by :func:`apply_spectrum_conditioning`.
BASELINE_MODES = ("none", "sigma_clip", "wimda")


def sigma_clip_baseline(
    values: ArrayLike,
    *,
    kappa: float = 2.0,
    max_iter: int = 10,
    location: str = "median",
) -> tuple[float, float]:
    """Return the ``(baseline, noise_sigma)`` of *values* by σ-clipping.

    Iteratively discards points more than ``kappa`` standard deviations from the
    running location and re-estimates the location/width over the survivors until
    the inlier set stops changing or ``max_iter`` is reached.  ``location`` is the
    robust ``"median"`` (default) or the ``"mean"`` used for WiMDA parity.

    The single-pass WiMDA baseline is ``location="mean", max_iter=1``: it clips
    once at 2σ about the full-spectrum mean and returns the mean of the
    survivors.  The converged σ doubles as a baseline-noise estimate.
    """
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 0.0
    loc_fn = np.median if str(location).lower() == "median" else np.mean

    loc = float(loc_fn(finite))
    sigma = float(np.std(finite))
    mask = np.ones(finite.shape, dtype=bool)
    for _ in range(max(1, int(max_iter))):
        if sigma <= 0.0:
            break
        new_mask = np.abs(finite - loc) <= float(kappa) * sigma
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask
        survivors = finite[mask]
        if survivors.size == 0:
            break
        loc = float(loc_fn(survivors))
        sigma = float(np.std(survivors))
    return loc, sigma


def pulse_compensation_gain(
    freqs_mhz: ArrayLike,
    *,
    half_width_us: float,
    separation_us: float = 0.0,
    n_pulses: int = 1,
    max_gain: float = 25.0,
) -> NDArray[np.float64]:
    """Return the per-frequency compensation gain ``1/R(ν)``, capped and cut off.

    ``R(ν)`` is the pulse amplitude from
    :func:`asymmetry.core.maxent.pulse.pulse_amplitude_phase`.  The gain is
    ``1/R`` up to ``max_gain``; at and above the first node — the first bin where
    ``R`` would demand more than ``max_gain`` — the gain is ``0`` (hard cutoff),
    because the pulse has suppressed that frequency below recovery.  A
    non-positive ``half_width_us`` or ``n_pulses`` means no shaping → unit gain.
    """
    freqs = np.asarray(freqs_mhz, dtype=np.float64)
    amplitude, _phase = pulse_amplitude_phase(
        freqs,
        half_width_us=half_width_us,
        separation_us=separation_us,
        n_pulses=n_pulses,
    )
    if amplitude is None:
        return np.ones_like(freqs)

    cap = max(1.0, float(max_gain))
    floor = 1.0 / cap
    gain = np.ones_like(freqs)
    above_floor = amplitude >= floor
    np.divide(1.0, amplitude, out=gain, where=above_floor)
    gain[~above_floor] = 0.0

    # Hard cutoff at and beyond the first node: once the pulse amplitude has
    # fallen below the floor (above DC), everything past it is unrecoverable.
    order = np.argsort(freqs, kind="stable")
    below = ~above_floor
    below_positive = below & (freqs > 0.0)
    if np.any(below_positive):
        first_node_pos = int(np.argmax(below_positive[order]))
        cut = order[first_node_pos:]
        gain[cut] = 0.0
    return gain


@dataclass
class ConditioningResult:
    """Conditioned channels plus diagnostics for the panel readouts."""

    display: NDArray[np.float64]
    error: NDArray[np.float64]
    baseline: float = 0.0
    noise_sigma: float = 0.0
    cutoff_frequency_mhz: float | None = None
    #: The pulse-compensation gain applied to ``display`` (``None`` if no
    #: compensation ran), so a companion channel can reuse the same gain.
    gain: NDArray[np.float64] | None = None


def apply_spectrum_conditioning(
    freqs_mhz: ArrayLike,
    display: ArrayLike,
    error: ArrayLike | None = None,
    *,
    pulse_compensation: bool = False,
    pulse_half_width_us: float = 0.0,
    pulse_separation_us: float = 0.0,
    pulse_n_pulses: int = 1,
    pulse_max_gain: float = 25.0,
    baseline_mode: str = "none",
    baseline_kappa: float = 2.0,
    exclusion_ranges: list[tuple[float, float]] | None = None,
) -> ConditioningResult:
    """Apply pulse compensation, baseline offset, and exclusions to a spectrum.

    Operates on the real display channel and its error on the canonical MHz axis,
    in WiMDA's order (compensation → baseline → exclusions).  Returns the
    conditioned channels and the baseline/noise/cutoff diagnostics the readouts
    surface.  Inputs are not mutated.
    """
    freqs = np.asarray(freqs_mhz, dtype=np.float64)
    values = np.asarray(display, dtype=np.float64).copy()
    errors = (
        np.asarray(error, dtype=np.float64).copy() if error is not None else np.zeros_like(values)
    )
    result = ConditioningResult(display=values, error=errors)

    if pulse_compensation and float(pulse_half_width_us) > 0.0:
        gain = pulse_compensation_gain(
            freqs,
            half_width_us=pulse_half_width_us,
            separation_us=pulse_separation_us,
            n_pulses=pulse_n_pulses,
            max_gain=pulse_max_gain,
        )
        values *= gain
        errors *= gain
        result.gain = gain
        cut = freqs[gain == 0.0]
        cut_positive = cut[cut > 0.0]
        if cut_positive.size:
            result.cutoff_frequency_mhz = float(np.min(cut_positive))

    mode = str(baseline_mode).lower()
    if mode in {"sigma_clip", "wimda"}:
        if mode == "wimda":
            baseline, sigma = sigma_clip_baseline(
                values, kappa=baseline_kappa, max_iter=1, location="mean"
            )
        else:
            baseline, sigma = sigma_clip_baseline(
                values, kappa=baseline_kappa, max_iter=10, location="median"
            )
        values = values - baseline
        result.baseline = baseline
        result.noise_sigma = sigma

    if exclusion_ranges:
        values = exclude_frequency_ranges(freqs, values, exclusion_ranges)

    result.display = values
    result.error = errors
    return result
