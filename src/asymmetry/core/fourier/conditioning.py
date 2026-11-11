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
  one pulse model. Amplitudes receive ``1/R`` and Power receives ``1/R²``.
  The first sampled positive frequency below ``1/max_gain`` and all following
  bins are removed. This grid-dependent numerical safeguard is not an analytic
  node or a statement that later response lobes contain no information.
  WiMDA uses a different, Gaussian Fourier response.

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

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from asymmetry.core.fourier.fft import exclude_frequency_ranges
from asymmetry.core.maxent.pulse import pulse_amplitude_phase

CONDITIONING_REVISION = "quantity_gain_sampled_support_v1"

#: Baseline estimator modes accepted by :func:`apply_spectrum_conditioning`.
BASELINE_MODES = ("none", "sigma_clip", "wimda")

#: Robust noise-floor block count for the unit-area estimator: the spectrum is
#: split into this many frequency blocks and a σ-clipped median floor is fitted
#: per block, so a slowly-varying (coloured) floor is tracked rather than
#: assumed flat.  Capped so short spectra keep several bins per block.
_UNIT_AREA_FLOOR_BLOCKS = 8
#: σ-clip half-width for the per-block floor estimate.
_UNIT_AREA_FLOOR_KAPPA = 2.0
#: Significance multiple ``k``: the floor-subtracted area must exceed ``k`` times
#: its noise scatter ``σ_area = σ_floor·√N·Δν`` for unit-area to be applied,
#: refusing to normalise pure noise (whose true integral is ~0).
_UNIT_AREA_SIGNIFICANCE_K = 5.0


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


def _amplitude_gain_limit(max_gain: float) -> float:
    """Retain at least unity amplitude gain, including an unavailable limit."""
    return max(1.0, float(max_gain))


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
    ``1/R`` up to ``max_gain``. The first sampled positive frequency with
    ``R < 1/max_gain`` and every following bin has zero gain, marking removed
    support. The grid and limit determine this numerical safeguard. A
    zero ``n_pulses`` is the identity. Active shaping requires a positive finite
    support half-width; double pulses require a nonnegative finite separation.
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

    cap = _amplitude_gain_limit(max_gain)
    floor = 1.0 / cap
    gain = np.ones_like(freqs)
    above_floor = amplitude >= floor
    np.divide(1.0, amplitude, out=gain, where=above_floor)
    gain[~above_floor] = 0.0

    # A sampled amplitude floor bounds amplification, including later lobes.
    order = np.argsort(freqs, kind="stable")
    below = ~above_floor
    below_positive = below & (freqs > 0.0)
    if np.any(below_positive):
        first_cut_pos = int(np.argmax(below_positive[order]))
        cut = order[first_cut_pos:]
        gain[cut] = 0.0
    return gain


def _block_median_floor(
    values: NDArray[np.float64], *, kappa: float
) -> tuple[NDArray[np.float64], float]:
    """Return a per-bin robust noise floor and the overall inlier scatter.

    The spectrum is split into up to :data:`_UNIT_AREA_FLOOR_BLOCKS` contiguous
    frequency blocks; each block's floor is the σ-clipped median of its values
    (:func:`sigma_clip_baseline` with ``location="median"``), assigned to every
    bin of that block — a piecewise-constant floor tolerant of a slowly-varying
    (coloured) continuum.  The returned scatter ``σ_floor`` is the σ-clipped
    width over the whole spectrum, used by the significance guard.
    """
    n = values.size
    if n == 0:
        return np.zeros(0, dtype=np.float64), 0.0
    n_blocks = int(min(_UNIT_AREA_FLOOR_BLOCKS, max(1, n)))
    edges = np.linspace(0, n, n_blocks + 1, dtype=int)
    floor = np.zeros(n, dtype=np.float64)
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        block_floor, _sigma = sigma_clip_baseline(
            values[lo:hi], kappa=kappa, max_iter=10, location="median"
        )
        floor[lo:hi] = block_floor
    _loc, sigma = sigma_clip_baseline(values, kappa=kappa, max_iter=10, location="median")
    return floor, float(sigma)


@dataclass
class UnitAreaResult:
    """Outcome of a unit-area (field-distribution) normalisation attempt."""

    #: Normalised density spectrum when ``applied``; the input unchanged otherwise.
    display: NDArray[np.float64]
    error: NDArray[np.float64]
    applied: bool
    #: Floor-subtracted integral ``Σ residual·Δν`` (the pre-normalisation area).
    area: float = 0.0
    #: Noise scatter of that area used by the significance guard.
    area_sigma: float = 0.0
    #: Human-readable reason unit-area was refused (``""`` when applied).
    reason: str = ""


def unit_area_normalise(
    freqs_mhz: ArrayLike,
    display: ArrayLike,
    error: ArrayLike | None = None,
    *,
    kappa: float = _UNIT_AREA_FLOOR_KAPPA,
    significance_k: float = _UNIT_AREA_SIGNIFICANCE_K,
) -> UnitAreaResult:
    """Present a magnitude spectrum as a field distribution ``p(ν)``.

    A robust noise floor (:func:`_block_median_floor`) is fitted and subtracted;
    the residual is integrated **unclipped** over the full one-sided range (so
    noise integrates to ~0 and the result is range-independent by construction),
    and the floor-subtracted spectrum and its error are divided by that area so
    ``∫ p dν = 1`` numerically on the MHz grid.

    A significance guard refuses the normalisation when the area does not exceed
    ``significance_k`` times its noise scatter ``σ_area = σ_floor·√N·Δν`` (or is
    non-positive) — a pure-noise spectrum keeps its calibrated scale and reports
    the reason instead of being blown up by division by a near-zero area.
    """
    freqs = np.asarray(freqs_mhz, dtype=np.float64)
    values = np.asarray(display, dtype=np.float64).copy()
    errors = (
        np.asarray(error, dtype=np.float64).copy() if error is not None else np.zeros_like(values)
    )
    n = values.size
    if n < 2:
        return UnitAreaResult(values, errors, False, reason="the spectrum has too few bins")

    diffs = np.diff(freqs[np.isfinite(freqs)])
    bin_width = float(np.median(diffs)) if diffs.size else 0.0
    if not np.isfinite(bin_width) or bin_width <= 0.0:
        return UnitAreaResult(values, errors, False, reason="the frequency grid is degenerate")

    floor, sigma = _block_median_floor(values, kappa=kappa)
    residual = values - floor
    area = float(np.sum(residual, dtype=np.float64) * bin_width)
    area_sigma = float(sigma) * math.sqrt(float(n)) * bin_width

    if not np.isfinite(area) or area <= 0.0 or area <= float(significance_k) * area_sigma:
        return UnitAreaResult(
            values,
            errors,
            False,
            area=area,
            area_sigma=area_sigma,
            reason="the integrated signal is not significant above the noise floor",
        )

    density = residual / area
    density_error = errors / area
    return UnitAreaResult(density, density_error, True, area=area, area_sigma=area_sigma)


@dataclass
class ConditioningResult:
    """Conditioned channels plus diagnostics for the panel readouts."""

    display: NDArray[np.float64]
    error: NDArray[np.float64]
    baseline: float = 0.0
    noise_sigma: float = 0.0
    cutoff_frequency_mhz: float | None = None
    quantity: Literal["amplitude", "power", "angle"] = "amplitude"
    #: Amplitude gain, shared with companion quadratures; zero marks removed bins.
    gain: NDArray[np.float64] | None = None
    amplitude_max_gain: float | None = None


def apply_spectrum_conditioning(
    freqs_mhz: ArrayLike,
    display: ArrayLike,
    error: ArrayLike | None = None,
    *,
    quantity: Literal["amplitude", "power", "angle"] = "amplitude",
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
    surface. ``quantity`` specifies the observable, never inferred from values.
    Power receives squared amplitude gain; angles receive no pulse gain.
    Pulse-removed bins retain shape as NaN values/errors, with zero amplitude
    gain marking their support. Inputs are not mutated.
    """
    if quantity not in {"amplitude", "power", "angle"}:
        raise ValueError("Spectrum quantity must be 'amplitude', 'power' or 'angle'.")
    freqs = np.asarray(freqs_mhz, dtype=np.float64)
    values = np.asarray(display, dtype=np.float64).copy()
    errors = (
        np.asarray(error, dtype=np.float64).copy() if error is not None else np.zeros_like(values)
    )
    result = ConditioningResult(display=values, error=errors, quantity=quantity)

    if pulse_compensation and quantity != "angle":
        result.amplitude_max_gain = _amplitude_gain_limit(pulse_max_gain)
        gain = pulse_compensation_gain(
            freqs,
            half_width_us=pulse_half_width_us,
            separation_us=pulse_separation_us,
            n_pulses=pulse_n_pulses,
            max_gain=result.amplitude_max_gain,
        )
        multiplier = gain**2 if quantity == "power" else gain
        values *= multiplier
        errors *= multiplier
        values[gain == 0.0] = np.nan
        errors[gain == 0.0] = np.nan
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

    if result.gain is not None:
        values[result.gain == 0.0] = np.nan
    result.display = values
    result.error = errors
    return result
