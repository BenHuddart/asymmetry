"""Spectral peak detection and multiplet pattern matching for the fit wizard.

This Qt-free module provides the peak-detection layer beneath the fit wizard's
model recommendation: given a time-domain :class:`MuonDataset` (or a raw FFT
magnitude spectrum), it locates oscillation lines with sub-bin frequency
refinement, local-noise-floor SNR, FWHM widths, and an optional Burg (all-poles)
cross-check that confirms — but never adds — peaks.

It reuses the existing spectral estimators rather than reimplementing them:
:func:`asymmetry.core.fourier.fft.fft_asymmetry` for the all-zeroes FFT and
:func:`asymmetry.core.fourier.burg.burg_spectrum` for the super-resolving
all-poles diagnostic.  ``scipy.signal`` is imported lazily inside the functions
that need it (mirroring the ``_scipy_fit_fallback`` pattern in ``fit_wizard``),
so importing this module never pulls SciPy in.

The module also hosts multiplet pattern matching (``MultipletMatch`` /
``match_multiplets``, added by the orchestrating layer): the detected-peak set
here is the input that pattern matcher consumes.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import median_filter

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fourier.apodisation import ApodisationEarlySignalWarning
from asymmetry.core.fourier.burg import burg_spectrum
from asymmetry.core.fourier.fft import fft_arrays

if TYPE_CHECKING:  # pragma: no cover - typing only
    # ``damped_line_scan`` imports ``effective_analysis_window`` from this
    # module, so the runtime import lives inside the functions that need it.
    from asymmetry.core.fitting.damped_line_scan import DampedLineAnalysis

_EPS = 1e-12

#: Sentinel SNR assigned to user-declared peaks so they always sort first and are
#: never dropped by a ``max_peaks`` cap.
USER_PEAK_SNR_SENTINEL = 1e6

#: Expected number of false noise peaks tolerated per spectrum: the SNR gate is
#: raised so that Rayleigh noise across all resolution elements clears it this
#: often (see ``detect_peaks_in_spectrum``).
_FALSE_PEAK_RATE = 0.01

#: SNR-truncation factor for the detection window (see
#: :func:`effective_analysis_window`).  Real μSR error bars grow exponentially
#: with time (dying-muon statistics, capped at 100 %); the pure-noise late tail
#: otherwise whitens the FFT and buries even strong lines.  The window is cut
#: where the per-point σ first exceeds ``_SNR_TRUNCATION_FACTOR`` times the
#: early-time σ — i.e. where per-point information (∝ 1/σ²) has dropped to
#: ~1/25 of its early value.  Chosen so a low-f line still retains ≳2 cycles
#: (keeping the fingerprint's ``cycles_in_window`` hint alive) while the
#: noise-dominated tail is discarded.
_SNR_TRUNCATION_FACTOR = 5.0

#: Never truncate below this many points — a short record has no meaningful
#: tail to shed and needs its full resolution.
_MIN_WINDOW_POINTS = 32

#: Sidelobe guard anchor: the Hann window's worst sidelobe is -31.5 dB (~2.7 %
#: of the main lobe) at ~2.5 resolution elements, decaying at -18 dB/octave.
#: The ceiling carries headroom over the textbook level because noise and
#: overlapping leakage tails routinely lift the first sidelobe past it.
_SIDELOBE_CEILING = 0.05
_SIDELOBE_ANCHOR_RESOLUTIONS = 3.0

#: Leakage profiles selectable on :func:`detect_peaks_in_spectrum`, as
#: ``(ceiling, anchor_resolutions, exponent)`` — see :func:`_sidelobe_ceiling`.
#:
#: ``"rect"`` is the unwindowed early-window pass's.  A boxcar's worst sidelobe
#: is -13.3 dB (~22 % of the main lobe) at ~1.4 resolution elements and decays
#: at only -6 dB/octave — two orders of magnitude worse than Hann near the line,
#: and far slower far from it.  Applying the Hann profile to an unwindowed crop
#: mistook a strong slow line's sinc tails for real lines (the first two
#: peak-detection regression tests caught exactly this).  As with the Hann
#: ceiling the level carries headroom over the textbook envelope
#: (``1/(pi·Δ/resolution)``), here ~3×: measured leakage ran at 1.5-2.2× the
#: single-line envelope on crops holding a strong line that completes only one
#: or two cycles, because two truncated lines' tails add and the first-order
#: detrend leaves its own step-like residual.
_LEAKAGE_PROFILES: dict[str, tuple[float, float, float]] = {
    "hann": (_SIDELOBE_CEILING, _SIDELOBE_ANCHOR_RESOLUTIONS, 3.0),
    "rect": (0.65, 1.5, 1.0),
}

# --------------------------------------------------------------------------- #
# Early-window (damped-line) pass — constants settled by the synthetic study
# documented on :func:`analyze_early_window_peaks`.
# --------------------------------------------------------------------------- #

#: Crop ladder for the early pass, as divisors of the SNR-truncated record.
#: Fractions of the record rather than absolute times, so the ladder scales with
#: the run's duration instead of its binning.  Two rungs: study step 1 on
#: :func:`analyze_early_window_peaks` records why both longer and shorter rungs
#: were dropped.
_EARLY_CROP_DIVISORS = (16, 64)

#: Never crop below this many points.  This is really a floor on the *width of
#: the search band*, which is what governs the pass's null: a crop of ``n``
#: points spans ``n/2`` resolution elements up to Nyquist, of which the guards
#: take four — so 64 points leaves ~28 independent elements, enough for the
#: clipped median floor to be a floor rather than an estimate of a handful of
#: correlated bins.  Study step 1 measured the cost of going below it: a rung
#: leaving ~10 elements raised the pure-noise maximum from 5.07 to 6.85 while
#: detecting nothing extra.
_EARLY_MIN_POINTS = 64

#: A ladder rung is kept only if it is at most this fraction of the previous one.
#: The divisors are a factor of four apart by construction, but
#: ``_EARLY_MIN_POINTS`` can collapse two of them onto nearly the same crop on a
#: short record — two transforms of the same window, two looks at the same noise,
#: no extra reach.
_EARLY_CROP_SHRINK = 0.5

#: DC/Nyquist guard band for the early pass, in resolution elements (the Hann
#: pass keeps its historical 0.5).  On a short unwindowed crop the running
#: median floor is edge-biased low at both ends and the final ``rfft`` bin is
#: real-only, so noise excursions in the outermost resolution elements cleared
#: the gate far more often than the look-elsewhere correction allows for.
_EARLY_GUARD_RESOLUTIONS = 3.0

#: A line must complete this many cycles inside the crop to be an early-pass
#: candidate.  Below that it is not a resolvable oscillation in that crop, and
#: the region is exactly where the residual curvature of the relaxing tail —
#: what a first-order detrend leaves behind — piles up.
_EARLY_MIN_CYCLES_IN_CROP = 4.0

#: Peaks kept per crop before the cross-crop dedupe.
_EARLY_MAX_PEAKS_PER_CROP = 3

#: Re-derived SNR gate for the early pass — NOT inherited from the Hann pass.
#: The early pass runs on a short, noisy, unwindowed crop with far fewer
#: resolution elements, so the look-elsewhere correction that governs the Hann
#: pass does not transfer.  Set from the null study (step 3): across 1400 draws
#: of three signal-free / oscillation-free record types the largest early-pass
#: SNR was 5.31, so the gate sits above it with margin and the measured
#: false-seed count at this threshold is zero on every null family.
_EARLY_MIN_SNR = 6.0

#: Early-pass additions guaranteed a slot in the merged peak set even when the
#: Hann pass already filled ``max_peaks``.  A full Hann peak set must not be
#: able to starve the pass that exists precisely to see what Hann cannot.
_EARLY_RESERVED_PEAKS = 2

#: Fraction of Nyquist guarded at the top of every reported band, on top of the
#: resolution-element guard.  On finely binned records the two differ by orders
#: of magnitude — see the note in :func:`detect_peaks_in_spectrum`.
_NYQUIST_GUARD_FRACTION = 0.02

#: Cap on damped lines the scan pass may contribute, mirroring
#: ``damped_line_scan.detect_damped_lines``'s own ``max_lines`` default.
_DAMPED_SCAN_MAX_LINES = 3


def _sidelobe_ceiling(
    delta_mhz: float,
    resolution_mhz: float,
    profile: tuple[float, float, float] = _LEAKAGE_PROFILES["hann"],
) -> float:
    """Max amplitude ratio a genuine line needs at ``delta_mhz`` from a stronger one.

    Anchored just above the analysis window's worst sidelobe and rolled off at
    that window's sidelobe decay, so leakage structure is rejected while genuine
    weak lines — which sit above the local sidelobe level — survive at any
    separation.  The Hann profile (-31.5 dB, -18 dB/octave, hence ``delta^-3``)
    is the default; the early-window pass selects the rectangular one.
    """
    ceiling, anchor_resolutions, exponent = profile
    anchor = anchor_resolutions * resolution_mhz
    return ceiling * (anchor / max(delta_mhz, anchor)) ** exponent


@dataclass(frozen=True)
class DetectedPeak:
    """A single detected spectral line.

    Attributes
    ----------
    frequency_mhz
        Sub-bin (parabolically interpolated) line frequency in MHz.
    amplitude
        Interpolated magnitude at the peak.
    snr
        ``amplitude`` divided by the local noise floor at the peak bin.
    width_mhz
        FWHM estimated via ``scipy.signal.peak_widths`` (``rel_height=0.5``),
        converted from bins to MHz.
    prominence
        Peak prominence from ``scipy.signal.find_peaks``.
    source
        Provenance: ``"fft"``, ``"residual_fft"``, ``"early_fft"``,
        ``"damped_scan"`` or ``"user"``.  An ``"early_fft"`` SNR is measured on
        a short unwindowed crop against a different noise floor, and a
        ``"damped_scan"`` SNR is a MAD excess over a matched-apodisation rung's
        own floor; neither is comparable with an ``"fft"`` SNR from the full
        record — rank within a pass, never across.
    burg_confirmed
        ``True``/``False`` when a Burg cross-check ran and did / did not find a
        matching all-poles local maximum; ``None`` when no cross-check ran.
        Always ``None`` for ``"early_fft"`` / ``"damped_scan"`` peaks: Burg is
        scoped to the Hann pass (it is unreliable on short, heavily damped
        windows).
    crop_us
        Duration of the early-window crop that found this peak, in µs; its
        spectral resolution is ``1/crop_us``.  ``None`` for every other source —
        the matched-apodisation scan needs no crop, so its peaks carry ``None``
        here and report their envelope directly in ``damping_rate_per_us``.
    damping_rate_per_us, amplitude_percent, phase_rad, delta_chi_squared
        The measured line parameters a
        :class:`~asymmetry.core.fitting.damped_line_scan.DampedLine` carries:
        envelope rate λ (µs⁻¹) of ``exp(-λt)``, amplitude ``A`` of
        ``A·exp(-λt)·cos(2πft + φ)`` on the input's asymmetry scale, phase φ in
        radians, and the weighted χ² improvement the line bought against the
        scan's nuisance basis.  All ``None`` for a peak that carries no such
        measurement (every pass but ``"damped_scan"``, and a ``"user"`` peak
        that inherited nothing).  Additive — old payloads deserialize to
        ``None``.
    """

    frequency_mhz: float
    amplitude: float
    snr: float
    width_mhz: float
    prominence: float
    source: str
    burg_confirmed: bool | None = None
    crop_us: float | None = None
    damping_rate_per_us: float | None = None
    amplitude_percent: float | None = None
    phase_rad: float | None = None
    delta_chi_squared: float | None = None


@dataclass(frozen=True)
class PeakAnalysis:
    """The outcome of a peak-detection pass over one spectrum.

    ``peaks`` is ordered by SNR descending *within a detection pass*, with any
    user-declared peaks first and each pass's block following: an ``early_fft``
    SNR is measured on a short unwindowed crop against a different noise floor,
    so it is not comparable with an ``fft`` SNR from the whole record.  A single
    pass's output is therefore plainly SNR-descending; a merged one is blocks of
    it (see :func:`merge_early_peaks` and :func:`merge_user_peaks`).
    """

    peaks: tuple[DetectedPeak, ...]
    noise_floor: float
    resolution_mhz: float
    nyquist_mhz: float
    detrended: bool
    detrend_template_key: str | None = None
    burg_order: int | None = None
    burg_hit_boundary: bool = False
    #: Full outcome of the matched-apodisation damped-line scan that produced
    #: this analysis's ``"damped_scan"`` peaks — the Δχ² threshold, the trial
    #: count and the τ ladder, which the peaks themselves do not carry.
    #: ``None`` when the scan did not run.  Additive — old payloads
    #: deserialize to ``None``.
    damped_lines: DampedLineAnalysis | None = None


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #


def serialize_detected_peak(peak: DetectedPeak) -> dict[str, object]:
    """Return a JSON-safe dict snapshot of a :class:`DetectedPeak`."""
    return {
        "frequency_mhz": float(peak.frequency_mhz),
        "amplitude": float(peak.amplitude),
        "snr": float(peak.snr),
        "width_mhz": float(peak.width_mhz),
        "prominence": float(peak.prominence),
        "source": str(peak.source),
        "burg_confirmed": peak.burg_confirmed,
        "crop_us": (float(peak.crop_us) if peak.crop_us is not None else None),
        # Additive damped-scan measurements; ``None`` for every other pass.
        "damping_rate_per_us": _optional_float(peak.damping_rate_per_us),
        "amplitude_percent": _optional_float(peak.amplitude_percent),
        "phase_rad": _optional_float(peak.phase_rad),
        "delta_chi_squared": _optional_float(peak.delta_chi_squared),
    }


def _serialize_damped_lines(analysis: DampedLineAnalysis | None) -> dict[str, object] | None:
    """Serialize the attached scan result, importing the scan module lazily.

    ``damped_line_scan`` imports :func:`effective_analysis_window` from this
    module, so the dependency can only run one way at import time.
    """
    if analysis is None:
        return None
    from asymmetry.core.fitting.damped_line_scan import serialize_damped_line_analysis

    return serialize_damped_line_analysis(analysis)


def _deserialize_damped_lines(payload: object) -> DampedLineAnalysis | None:
    """Rebuild an attached scan result; ``None`` for payloads that predate it."""
    if payload is None:
        return None
    from asymmetry.core.fitting.damped_line_scan import deserialize_damped_line_analysis

    return deserialize_damped_line_analysis(payload)


def _optional_float(value: object) -> float | None:
    """``float(value)`` or ``None`` — the additive-field (de)serialisation rule."""
    if value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def deserialize_detected_peak(payload: object) -> DetectedPeak | None:
    """Rebuild a :class:`DetectedPeak` from a persisted dict, tolerating gaps."""
    if not isinstance(payload, dict):
        return None
    burg = payload.get("burg_confirmed", None)
    crop = payload.get("crop_us", None)
    return DetectedPeak(
        frequency_mhz=float(payload.get("frequency_mhz", 0.0)),
        amplitude=float(payload.get("amplitude", 0.0)),
        snr=float(payload.get("snr", 0.0)),
        width_mhz=float(payload.get("width_mhz", 0.0)),
        prominence=float(payload.get("prominence", 0.0)),
        source=str(payload.get("source", "fft")),
        burg_confirmed=(bool(burg) if burg is not None else None),
        crop_us=(float(crop) if crop is not None else None),
        damping_rate_per_us=_optional_float(payload.get("damping_rate_per_us")),
        amplitude_percent=_optional_float(payload.get("amplitude_percent")),
        phase_rad=_optional_float(payload.get("phase_rad")),
        delta_chi_squared=_optional_float(payload.get("delta_chi_squared")),
    )


def serialize_peak_analysis(analysis: PeakAnalysis) -> dict[str, object]:
    """Return a JSON-safe dict snapshot of a :class:`PeakAnalysis`."""
    return {
        "peaks": [serialize_detected_peak(peak) for peak in analysis.peaks],
        "noise_floor": float(analysis.noise_floor),
        "resolution_mhz": float(analysis.resolution_mhz),
        "nyquist_mhz": float(analysis.nyquist_mhz),
        "detrended": bool(analysis.detrended),
        "detrend_template_key": analysis.detrend_template_key,
        "burg_order": analysis.burg_order,
        "burg_hit_boundary": bool(analysis.burg_hit_boundary),
        "damped_lines": _serialize_damped_lines(analysis.damped_lines),
    }


def deserialize_peak_analysis(payload: object) -> PeakAnalysis | None:
    """Rebuild a :class:`PeakAnalysis` from a persisted dict, tolerating gaps."""
    if not isinstance(payload, dict):
        return None
    peaks = tuple(
        peak
        for entry in payload.get("peaks", [])
        if (peak := deserialize_detected_peak(entry)) is not None
    )
    burg_order = payload.get("burg_order", None)
    template_key = payload.get("detrend_template_key", None)
    return PeakAnalysis(
        peaks=peaks,
        noise_floor=float(payload.get("noise_floor", 0.0)),
        resolution_mhz=float(payload.get("resolution_mhz", 0.0)),
        nyquist_mhz=float(payload.get("nyquist_mhz", 0.0)),
        detrended=bool(payload.get("detrended", False)),
        detrend_template_key=(str(template_key) if template_key is not None else None),
        burg_order=(int(burg_order) if burg_order is not None else None),
        burg_hit_boundary=bool(payload.get("burg_hit_boundary", False)),
        damped_lines=_deserialize_damped_lines(payload.get("damped_lines")),
    )


# --------------------------------------------------------------------------- #
# Noise floor and interpolation helpers
# --------------------------------------------------------------------------- #


def _running_median(values: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """Return a same-length running median with edge padding.

    ``window`` is coerced odd and clamped to the array size.
    """
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n == 0:
        return arr.copy()
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    window = min(window, n if n % 2 == 1 else n - 1)
    if window <= 1:
        return arr.copy()
    # scipy's rank filter, not a sliding_window_view + np.median(axis=1): the
    # windowed matrix materialises n×window floats (27 GB for a 256k-bin
    # padded spectrum — froze an 8 GB machine) and is ~1000× slower for the
    # same result. mode="nearest" matches the previous edge padding exactly.
    return median_filter(arr, size=window, mode="nearest")


def _mad_sigma(residual: NDArray[np.float64]) -> float:
    """Robust σ of ``residual`` from its median absolute deviation."""
    mad = float(np.median(np.abs(residual - np.median(residual))))
    return 1.4826 * mad


def _local_noise_floor(
    magnitude: NDArray[np.float64], bins_per_resolution: int = 1
) -> NDArray[np.float64]:
    """Estimate the local noise floor via a running median + one sigma-clip pass.

    The running median is robust to isolated lines; a single MAD-based
    sigma-clip refinement then recomputes the floor over bins not exceeding
    ``floor + 3·σ``, so a spectrum with several strong lines does not have its
    floor pulled up by them.  The window spans at least eight resolution
    elements (``bins_per_resolution`` accounts for zero-padding oversampling) so
    that a broad line — e.g. an unresolved doublet in a short record — cannot
    dominate its own median window and suppress its SNR.
    """
    mags = np.asarray(magnitude, dtype=float)
    n = mags.size
    if n == 0:
        return mags.copy()
    window = max(9, int(round(0.05 * n)), 8 * max(1, int(bins_per_resolution)) + 1)
    floor = _running_median(mags, window)

    residual = mags - floor
    sigma = _mad_sigma(residual)
    if sigma > _EPS:
        clipped = mags.copy()
        outliers = residual > 3.0 * sigma
        # Replace strong lines with the current floor estimate so the refined
        # running median is not polluted by them, then recompute.
        clipped[outliers] = floor[outliers]
        floor = _running_median(clipped, window)
    return floor


def _global_noise_floor(in_band: NDArray[np.float64]) -> float:
    """One sigma-clipped median floor for a whole (short) spectrum.

    The running-median floor of :func:`_local_noise_floor` estimates the noise
    *near* each bin, which is right for a long record where the floor varies
    across the band.  On a short early-window crop it is wrong twice over: the
    band holds only ~10-50 resolution elements, so a median window spanning
    eight of them sits largely inside the line it is supposed to be a floor
    *for*, and a heavily damped line is a broad Lorentzian whose own skirt then
    becomes its "noise".  Measured consequence (study step 2): the local-floor
    SNR of a damped line saturated near 5 however much its amplitude was
    raised — it had become a line-shape ratio, not a significance — while pure
    noise reached 6.4, so no threshold separated signal from noise at all.  A
    single clipped median over the guarded band restores an
    amplitude-proportional statistic (SNR ∝ amplitude across an 8× sweep) and
    drops the pure-noise maximum below the weakest real detection.

    Not :func:`asymmetry.core.fourier.conditioning.sigma_clip_baseline`: that
    clips ``|x − loc|`` symmetrically and estimates σ with ``np.std``, and both
    are wrong here.  Lines on a magnitude spectrum are positive-only
    excursions, so a symmetric clip eats the low tail as well; and ``std`` over
    a ~30-element band holding a broad Lorentzian is not robust to the very
    feature the floor has to see past.
    """
    values = np.asarray(in_band, dtype=float)
    if values.size == 0:
        return 0.0
    # The clip width is a property of the data, not of the running estimate:
    # ``(values − floor) − median(values − floor)`` is ``values − median(values)``
    # for any ``floor``, so σ is loop-invariant and is computed once.
    sigma = _mad_sigma(values)
    floor = float(np.median(values))
    if sigma <= _EPS:
        return floor
    for _ in range(3):
        kept = values[values - floor <= 3.0 * sigma]
        if kept.size < 5:
            break
        updated = float(np.median(kept))
        if abs(updated - floor) <= _EPS:
            break
        floor = updated
    return floor


def _parabolic_interpolation(log_mag: NDArray[np.float64], idx: int) -> tuple[float, float]:
    """3-point parabolic vertex offset and interpolated log-amplitude at ``idx``.

    Returns ``(delta_bins, log_amplitude)`` where ``delta_bins`` is the sub-bin
    offset of the vertex from ``idx`` in [-0.5, 0.5]; falls back to ``(0, y1)``
    at array edges.
    """
    n = log_mag.size
    if idx <= 0 or idx >= n - 1:
        return 0.0, float(log_mag[idx])
    y0 = float(log_mag[idx - 1])
    y1 = float(log_mag[idx])
    y2 = float(log_mag[idx + 1])
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < _EPS:
        return 0.0, y1
    delta = 0.5 * (y0 - y2) / denom
    if not np.isfinite(delta):
        return 0.0, y1
    delta = float(np.clip(delta, -0.5, 0.5))
    peak_log = y1 - 0.25 * (y0 - y2) * delta
    return delta, float(peak_log)


# --------------------------------------------------------------------------- #
# Peak detection on a magnitude spectrum
# --------------------------------------------------------------------------- #


def detect_peaks_in_spectrum(
    frequencies_mhz: NDArray[np.float64],
    magnitude: NDArray[np.float64],
    *,
    resolution_mhz: float,
    max_peaks: int = 6,
    min_snr: float = 2.5,
    source: str = "fft",
    low_guard_resolutions: float = 0.5,
    high_guard_resolutions: float = 0.5,
    global_noise_floor: bool = False,
    leakage_profile: str = "hann",
    crop_us: float | None = None,
) -> PeakAnalysis:
    """Detect spectral lines in a positive-frequency magnitude spectrum.

    Parameters
    ----------
    frequencies_mhz, magnitude
        The (possibly zero-padded / oversampled) frequency axis and magnitude.
    resolution_mhz
        The true spectral resolution ``1/T`` — distinct from the bin spacing
        ``df`` when the spectrum is zero-padded.
    max_peaks, min_snr, source
        Cap, SNR threshold and provenance tag.
    low_guard_resolutions, high_guard_resolutions
        Widths of the DC and Nyquist guard bands, in resolution elements.  Both
        default to the historical Hann-pass ``0.5``; the early-window pass
        widens both (see :func:`analyze_early_window_peaks`).  *Low:* an
        unwindowed short crop of a relaxing record leaves residual trend
        curvature in the first few resolution elements, and a line completing
        only a cycle or two inside such a crop is not a resolvable oscillation
        there in any case.  *High:* the running-median floor is edge-biased low
        at both ends, and the last bin of an ``rfft`` is real-only (half-normal,
        not Rayleigh), so noise excursions in the outermost resolution elements
        clear the SNR gate far more often than the look-elsewhere correction
        allows for.
    global_noise_floor
        Estimate one sigma-clipped median floor over the whole guarded band
        instead of the historical running-median floor — what a short crop needs
        (see :func:`_global_noise_floor`).
    leakage_profile
        ``"hann"`` (default) or ``"rect"`` — which window's sidelobe envelope the
        leakage guard rejects against (see :func:`_sidelobe_ceiling`).  The
        early-window pass transforms unwindowed and must use ``"rect"``.
    crop_us
        Stamped onto every returned peak's :attr:`DetectedPeak.crop_us` for
        provenance; the early-window pass passes its crop duration here.

    Returns
    -------
    PeakAnalysis
        Peaks SNR-descending, with the representative noise floor, resolution
        and Nyquist recorded.
    """
    from scipy.signal import find_peaks, peak_widths

    freqs = np.asarray(frequencies_mhz, dtype=float)
    mags = np.asarray(magnitude, dtype=float)
    resolution_mhz = float(max(resolution_mhz, _EPS))
    nyquist = float(freqs[-1]) if freqs.size else 0.0

    if freqs.size < 4 or mags.size < 4:
        return PeakAnalysis(
            peaks=(),
            noise_floor=0.0,
            resolution_mhz=resolution_mhz,
            nyquist_mhz=nyquist,
            detrended=False,
        )

    df = float(np.median(np.diff(freqs)))
    if df <= 0.0:
        df = _EPS

    # Positive frequencies inside the guard bands.  The DC guard rejects the
    # relaxation/leakage hump at the bottom edge; the mirrored top-edge guard
    # rejects artifact lines hard against Nyquist (aliased structure, filter
    # roll-off), which — like DC — carry no genuine oscillation frequency.
    dc_bin_guard = 3.0 * df
    low_guard = max(dc_bin_guard, float(low_guard_resolutions) * resolution_mhz)
    requested_high_guard = max(dc_bin_guard, float(high_guard_resolutions) * resolution_mhz)
    # A finely binned record (0.1 ns bins => ~5000 MHz Nyquist over a few µs)
    # has a resolution element four orders of magnitude narrower than its
    # Nyquist, so a guard counted in resolution elements is effectively no
    # guard at all up there — and that is exactly where the anti-aliasing
    # roll-off, the real-only last rfft bin and the edge-biased running median
    # manufacture clusters of junk peaks. Floor the guard at a FRACTION OF
    # NYQUIST so it scales with the binning rather than with the duration.
    high_guard = max(requested_high_guard, _NYQUIST_GUARD_FRACTION * nyquist)
    valid = (freqs > low_guard) & (freqs < nyquist - high_guard)
    # Leakage parents may live BELOW the reporting band: once the guards widen
    # (the early pass), the strong low line — or the residual trend hump at DC —
    # whose sidelobes we are rejecting is itself outside the reported band, and a
    # guard that only knows about reported peaks cannot attribute the leakage to
    # anything.  So the leakage guard sees everything down to the bare DC-bin
    # guard.  ``parent_valid ⊇ valid`` holds by construction at any padding
    # factor, and the two coincide whenever the guards are not widened — which
    # is what keeps the Hann pass unchanged.
    # Deliberately keyed on the REQUESTED guards, not the Nyquist-fraction
    # floor: the floor applies on every record, and letting it flip this switch
    # would silently change the Hann pass's leakage-parent set (and so which
    # low-frequency peaks survive the leakage guard) on data whose top edge is
    # not in question.
    widened = low_guard > dc_bin_guard or requested_high_guard > dc_bin_guard
    parent_valid = (freqs > dc_bin_guard) & (freqs < nyquist - dc_bin_guard) if widened else valid
    empty = PeakAnalysis(
        peaks=(),
        noise_floor=0.0,
        resolution_mhz=resolution_mhz,
        nyquist_mhz=nyquist,
        detrended=False,
    )
    if not np.any(valid):
        return empty

    bins_per_resolution = max(1, int(round(resolution_mhz / df)))
    if global_noise_floor:
        # One number for the whole band, so there is nothing to take a median
        # of and nothing to materialise a full-length array for.
        flat_floor = _global_noise_floor(mags[valid])
        local_floor = None
        representative_floor = flat_floor
    else:
        local_floor = _local_noise_floor(mags, bins_per_resolution)
        representative_floor = float(np.median(local_floor[valid]))

    # Look-elsewhere-corrected SNR gate.  For Gaussian time-domain noise the
    # magnitude bins are Rayleigh with P(X > k*median) = 2^(-k^2), so across
    # ~n_res independent resolution elements the tallest noise excursion
    # reaches ~sqrt(log2(n_res)) median units; gate at the level where the
    # expected false-peak count is _FALSE_PEAK_RATE.
    span = float(freqs[-1] - low_guard)
    n_res = max(2.0, span / resolution_mhz)
    adaptive_min_snr = float(np.sqrt(np.log2(n_res / _FALSE_PEAK_RATE)))
    effective_min_snr = max(float(min_snr), adaptive_min_snr)

    prominence = 2.0 * representative_floor
    distance = bins_per_resolution

    find_kwargs: dict[str, object] = {"distance": distance}
    if prominence > _EPS:
        find_kwargs["prominence"] = prominence
    peak_indices, properties = find_peaks(mags, **find_kwargs)

    # Restrict to the guarded positive band (plus any sub-guard leakage parents).
    keep = parent_valid[peak_indices]
    peak_indices = peak_indices[keep]
    reported = valid[peak_indices]
    if peak_indices.size == 0:
        return replace(empty, noise_floor=representative_floor)

    prominences = (
        np.asarray(properties.get("prominences", np.zeros(keep.size)))[keep]
        if "prominences" in properties
        else np.zeros(peak_indices.size)
    )

    # FWHM widths (in samples) at half the peak's prominence-relative height.
    widths_samples, _wh, _lips, _rips = peak_widths(mags, peak_indices, rel_height=0.5)

    log_mag = np.log(np.maximum(mags, _EPS))
    # Deliberately not a silent fallback: giving an unwindowed spectrum the Hann
    # sidelobe envelope is exactly the failure this profile exists to prevent.
    profile = _LEAKAGE_PROFILES[leakage_profile]

    candidates: list[tuple[DetectedPeak, bool]] = []
    for k, idx in enumerate(peak_indices):
        idx = int(idx)
        delta_bins, peak_log = _parabolic_interpolation(log_mag, idx)
        freq = float(freqs[idx] + delta_bins * df)
        amplitude = float(np.exp(peak_log))
        floor_here = float(max(flat_floor if local_floor is None else local_floor[idx], _EPS))
        snr = amplitude / floor_here
        if snr < effective_min_snr:
            continue
        width_mhz = float(max(widths_samples[k], 0.0) * df)
        candidates.append(
            (
                DetectedPeak(
                    frequency_mhz=freq,
                    amplitude=amplitude,
                    snr=float(snr),
                    width_mhz=width_mhz,
                    prominence=float(prominences[k]),
                    source=source,
                    crop_us=(float(crop_us) if crop_us is not None else None),
                ),
                bool(reported[k]),
            )
        )

    # Windowing-leakage guard: walk peaks strongest-first and drop any peak
    # sitting below the sidelobe ceiling of an already-accepted stronger line.
    # A rejected parent cannot shelter its own children, so the accumulator
    # carries every survivor; only the reported band is returned.
    candidates.sort(key=lambda entry: entry[0].amplitude, reverse=True)
    accepted: list[tuple[DetectedPeak, bool]] = []
    for peak, is_reported in candidates:
        is_sidelobe = any(
            peak.amplitude
            < other.amplitude
            * _sidelobe_ceiling(
                abs(peak.frequency_mhz - other.frequency_mhz), resolution_mhz, profile
            )
            for other, _ in accepted
        )
        if not is_sidelobe:
            accepted.append((peak, is_reported))

    surviving = sorted(
        (peak for peak, is_reported in accepted if is_reported),
        key=lambda p: p.snr,
        reverse=True,
    )
    detected = surviving[: max(0, int(max_peaks))]

    return PeakAnalysis(
        peaks=tuple(detected),
        noise_floor=representative_floor,
        resolution_mhz=resolution_mhz,
        nyquist_mhz=nyquist,
        detrended=False,
    )


# --------------------------------------------------------------------------- #
# Dataset-level analysis
# --------------------------------------------------------------------------- #


def effective_analysis_window(
    time: NDArray[np.float64],
    error: NDArray[np.float64],
    *,
    factor: float = _SNR_TRUNCATION_FACTOR,
    min_points: int = _MIN_WINDOW_POINTS,
) -> int:
    """Return the exclusive end index of the noise-truncated analysis window.

    μSR error bars grow roughly exponentially with time (dying-muon statistics)
    and are capped at 100 %; a full-window FFT is then dominated by the late-time
    pure-noise tail, whitening the spectrum so even clean lines vanish.  This
    truncates the record at the first point whose per-point error exceeds
    ``factor`` times the early-time error — i.e. where the per-point information
    ``1/σ²`` has fallen below ``1/factor²`` of its early value — keeping the
    statistically informative early window and shedding the noise tail.

    The criterion is strictly **per-point**, so flat-error records (constant σ,
    the synthetic/test convention) are never truncated: the returned index is the
    full length and the reported resolution is unchanged.  Returns the full
    length for short records (``≤ min_points``) or degenerate error arrays.
    """
    err = np.asarray(error, dtype=float)
    n = err.size
    if n <= int(min_points) or int(time.size) != n:
        return n
    finite = np.isfinite(err) & (err > 0.0)
    if not np.any(finite):
        return n
    early = max(5, n // 20)
    sigma_early = float(np.median(err[:early][finite[:early]])) if np.any(finite[:early]) else 0.0
    if not np.isfinite(sigma_early) or sigma_early <= 0.0:
        return n
    # First point that is finite and exceeds the SNR-truncation threshold.
    exceeds = finite & (err > float(factor) * sigma_early)
    if not np.any(exceeds):
        return n
    end = int(np.argmax(exceeds))
    return max(end, int(min_points))


def _centered_signal(
    dataset: MuonDataset, detrend_curve: NDArray[np.float64] | None
) -> tuple[NDArray[np.float64], bool]:
    """Return ``(signal, detrended)`` — the residual to transform.

    With ``detrend_curve`` the residual is ``asymmetry − detrend_curve``;
    otherwise it is ``asymmetry −`` tail estimate (mean of the last ~20 %),
    mirroring the fingerprint centering in ``fit_wizard``.
    """
    y = np.asarray(dataset.asymmetry, dtype=float)
    if detrend_curve is not None:
        curve = np.asarray(detrend_curve, dtype=float)
        if curve.shape != y.shape:
            raise ValueError(
                f"detrend_curve shape {curve.shape} does not match asymmetry {y.shape}"
            )
        return y - curve, True
    n = y.size
    late = min(n, max(5, n // 5))  # last ~20 %
    tail = float(np.mean(y[-late:])) if n else 0.0
    return y - tail, False


def early_window_crops(n_points: int) -> tuple[int, ...]:
    """Point counts of the early-window crop ladder for an ``n_points`` record.

    Each rung is ``n_points // divisor`` for the divisors in
    ``_EARLY_CROP_DIVISORS``, floored at ``_EARLY_MIN_POINTS``.  A rung that is
    not at least ``_EARLY_CROP_SHRINK`` shorter than the one before it is
    dropped, so a short record collapses the ladder rather than transforming
    nearly the same crop twice.
    """
    crops: list[int] = []
    total = int(n_points)
    for divisor in _EARLY_CROP_DIVISORS:
        crop = int(round(total / divisor))
        crop = min(max(crop, _EARLY_MIN_POINTS), total)
        if crops and crop > _EARLY_CROP_SHRINK * crops[-1]:
            continue
        crops.append(crop)
    return tuple(crops)


def analyze_early_window_peaks(
    time: NDArray[np.float64],
    signal: NDArray[np.float64],
    error: NDArray[np.float64],
    *,
    max_peaks: int = _EARLY_MAX_PEAKS_PER_CROP,
    min_snr: float = _EARLY_MIN_SNR,
) -> PeakAnalysis:
    """Find heavily damped lines the Hann pass structurally cannot see.

    Every windowed transform (``hann``/``cosine``/``gaussian``) is zero at the
    first sample, so it deletes exactly the leading nanoseconds where a heavily
    damped oscillation lives — the failure the library's
    :class:`~asymmetry.core.fourier.apodisation.ApodisationEarlySignalWarning`
    exists to flag.  This pass is the seeding-side answer: an **unwindowed**
    (``window="none"``), first-order-detrended transform of a short leading crop
    of the record, repeated over a small ladder of crops
    (:func:`early_window_crops`), each with its own honest
    ``resolution_mhz = 1/T_crop``.

    ``time``/``signal``/``error`` are the already-centred, already SNR-truncated
    arrays :func:`analyze_dataset_peaks` transforms; the returned analysis
    carries the peaks from every rung, de-duplicated across rungs, tagged
    ``source="early_fft"`` and stamped with the crop that found them
    (:attr:`DetectedPeak.crop_us`).  ``resolution_mhz`` is the ladder's finest
    (longest crop); a peak's own resolution is ``1/peak.crop_us``.

    **The synthetic study behind the constants.**  All parameters invented; the
    record is a damped cosine at 300 MHz on a relaxing tail, 1 ns binning over
    8 µs, with μSR-like errors growing as ``exp(t/2τ_μ)`` capped at 100 %.

    1. *Which crops — a fixed ladder, not an envelope-estimated crop.*  Both
       shapes the plan called for were built and scored.  The adaptive one —
       high-pass with the apodisation guard's moving-mean machinery, fit the
       decay of the high-passed envelope, crop to ~3/λ_est — tracks λ well while
       λ is *slow* (10.7 against a true 10, 2.2 against 2) and fails exactly
       where the feature is needed: at λ = 60 µs⁻¹ it returned an estimate on
       only 3 draws in 30 and read ~35, because a moving-mean kernel wide enough
       to be a baseline is already wider than the lifetime being measured.
       Recovering that would mean scanning kernel scales — which *is* the crop
       ladder, with extra machinery in front.  It was also the worse citizen on
       a conventional record, contributing a spurious peak on ~1 draw in 3 where
       the ladder contributed none.  So: a fixed ladder.  Ladders of two to four
       rungs drawn from 1/8 … 1/512 of the record were then scored.  Detection
       is flat across all of them (median SNR within ~5 % at every λ), so the
       null decides, and short rungs are what damage it: a crop below ~1/128 of
       the record leaves a band only ~10 resolution elements wide, where the
       clipped median floor is itself noisy.  Adding a 1/256 rung raised the
       pure-noise maximum from 5.07 to 6.85 and a 1/512 rung to 9.12, neither
       detecting anything the surviving rungs missed.  1/16 and 1/64 — ~440 ns
       and ~110 ns on an 8 µs informative window — cover envelope rates from
       ~2 µs⁻¹ (where the Hann pass sees the line too) to ~100 µs⁻¹ (where it is
       entirely blind).  Beyond ~120 µs⁻¹ the line is not recoverable at any
       crop at these amplitudes; that is a limit of the pass, not a claim.
    2. *Which noise floor.*  With the default running-median floor the reported
       SNR of the damped line **saturated near 5 whatever its amplitude** — on a
       band ten resolution elements wide the median window sits inside the
       line's own Lorentzian skirt, so the ratio measures line shape, not
       significance — while pure noise reached 6.4.  Nothing separated them.
       The global clipped floor (:func:`_global_noise_floor`) restores
       SNR ∝ amplitude over an 8× amplitude sweep.
    3. *Which threshold* (``_EARLY_MIN_SNR``, re-derived, not inherited).  Three
       null families were drawn through the shipped ladder with the gate open:
       500 pure-noise records, 500 relaxing tails with no oscillation, and 400
       conventional narrow-line records whose line is alive across the whole
       record.  Raw peaks arrive at 0.08-0.14 per draw and the largest SNR any
       of them reached was 5.31.  The gate sits at 6.0.  At that gate all three
       families contribute exactly zero early-pass peaks, the conventional
       record's Hann-pass peaks are bit-identical with the pass on and off, and
       the damped line is still found (median SNR 9.3 at λ = 60 µs⁻¹).
    """
    t = np.asarray(time, dtype=float)
    y = np.asarray(signal, dtype=float)
    err = np.asarray(error, dtype=float)
    total = t.size
    empty = PeakAnalysis(
        peaks=(),
        noise_floor=0.0,
        resolution_mhz=_EPS,
        nyquist_mhz=0.0,
        detrended=False,
    )
    if total < _EARLY_MIN_POINTS:
        return empty

    collected: list[DetectedPeak] = []
    floors: list[float] = []
    finest_resolution = float("inf")
    nyquist_mhz = 0.0
    for crop in early_window_crops(total):
        t_crop = t[:crop]
        duration = float(t_crop[-1] - t_crop[0])
        if duration <= 0.0:
            continue
        resolution = 1.0 / duration
        # window="none" is the whole point; detrend=1 removes the slow tail's
        # local slope across the crop without touching an oscillation that
        # completes _EARLY_MIN_CYCLES_IN_CROP cycles inside it.
        frequencies, _real, magnitude = fft_arrays(
            t_crop,
            y[:crop],
            err[:crop],
            window="none",
            padding_factor=4,
            detrend=1,
        )
        nyquist_mhz = max(nyquist_mhz, float(frequencies[-1]) if frequencies.size else 0.0)
        crop_analysis = detect_peaks_in_spectrum(
            frequencies,
            magnitude,
            resolution_mhz=resolution,
            max_peaks=max_peaks,
            min_snr=min_snr,
            source="early_fft",
            low_guard_resolutions=_EARLY_MIN_CYCLES_IN_CROP,
            high_guard_resolutions=_EARLY_GUARD_RESOLUTIONS,
            global_noise_floor=True,
            leakage_profile="rect",
            crop_us=duration,
        )
        if crop_analysis.peaks:
            finest_resolution = min(finest_resolution, resolution)
        collected.extend(crop_analysis.peaks)
        floors.append(crop_analysis.noise_floor)

    if not collected:
        return replace(empty, nyquist_mhz=nyquist_mhz)

    # Cross-rung dedupe: the same line seen at two crops is one line.  Keep the
    # strongest (its crop is the one best matched to the envelope) and drop
    # anything within the coarser of the two crops' resolutions.
    collected.sort(key=lambda p: p.snr, reverse=True)
    kept: list[DetectedPeak] = []
    for peak in collected:
        if any(_same_line(peak, other) for other in kept):
            continue
        kept.append(peak)

    return PeakAnalysis(
        peaks=tuple(kept),
        noise_floor=float(np.median(floors)) if floors else 0.0,
        resolution_mhz=float(finest_resolution),
        nyquist_mhz=nyquist_mhz,
        detrended=False,
    )


def _peak_resolution_mhz(peak: DetectedPeak, fallback_mhz: float) -> float:
    """Resolution of the pass that found ``peak`` — its crop's, or ``fallback``."""
    if peak.crop_us is not None and peak.crop_us > 0.0:
        return 1.0 / peak.crop_us
    return float(fallback_mhz)


def _same_line(
    peak: DetectedPeak,
    other: DetectedPeak,
    *,
    peak_fallback_mhz: float = _EPS,
    other_fallback_mhz: float = _EPS,
) -> bool:
    """True when two peaks are the same line at the coarser pass's resolution.

    The one collision rule the merge policies share: each peak is matched at the
    resolution of the pass that found it (see :func:`_peak_resolution_mhz`), and
    the coarser of the two decides.
    """
    coarser = max(
        _peak_resolution_mhz(peak, peak_fallback_mhz),
        _peak_resolution_mhz(other, other_fallback_mhz),
    )
    return abs(peak.frequency_mhz - other.frequency_mhz) <= coarser


def merge_early_peaks(
    analysis: PeakAnalysis,
    early: PeakAnalysis,
    *,
    max_peaks: int = 6,
) -> PeakAnalysis:
    """Fold early-window peaks into a Hann-pass analysis.

    Merge policy, in full:

    * **The Hann pass wins a collision.**  An early-pass peak within the
      *coarser* of the two resolutions of an existing Hann-pass peak is dropped:
      the Hann pass looked at the whole informative record, so for a line it can
      see at all its frequency estimate is the better one.  Early-pass peaks are
      only ever *additions* — lines the Hann pass missed.
    * **SNRs are never compared across passes.**  An early-pass SNR is measured
      on a short crop against a different noise floor; the merged tuple keeps the
      Hann peaks in their own SNR order, then the early additions in theirs, and
      each peak keeps its ``source`` (and ``crop_us``) so downstream consumers
      can rank within a pass.
    * **The ``max_peaks`` cap survives the merge**, with ``_EARLY_RESERVED_PEAKS``
      slots reserved for early additions — otherwise a full Hann peak set could
      starve the pass that exists to see what Hann cannot.
    """
    additions = [
        peak
        for peak in early.peaks
        if not any(
            _same_line(
                peak,
                existing,
                peak_fallback_mhz=early.resolution_mhz,
                other_fallback_mhz=analysis.resolution_mhz,
            )
            for existing in analysis.peaks
        )
    ]
    if not additions:
        return analysis

    cap = max(0, int(max_peaks))
    reserved = min(len(additions), _EARLY_RESERVED_PEAKS)
    kept_existing = list(analysis.peaks)[: max(0, cap - reserved)]
    merged = kept_existing + additions[: max(0, cap - len(kept_existing))]
    return replace(analysis, peaks=tuple(merged))


def analyze_damped_scan_peaks(
    time: NDArray[np.float64],
    signal: NDArray[np.float64],
    error: NDArray[np.float64],
    *,
    max_lines: int = _DAMPED_SCAN_MAX_LINES,
) -> PeakAnalysis:
    """Run the matched-apodisation damped-line scan and express it as peaks.

    The scan (:func:`~asymmetry.core.fitting.damped_line_scan.detect_damped_lines`)
    is the wizard's damped-line pass: it needs no user-chosen crop, it measures
    the envelope rate rather than inferring it from a crop length, and it
    accepts a line on a look-elsewhere-corrected Δχ² test rather than on an SNR
    gate.  Each accepted line becomes one ``source="damped_scan"``
    :class:`DetectedPeak`, Δχ²-descending, carrying the measured λ, amplitude,
    phase and Δχ² so the seeding path can use them directly.

    ``width_mhz`` is the Lorentzian FWHM ``λ/π`` of the line's own envelope —
    the honest width of a peak the scan never resolved with a window — and
    ``crop_us`` is ``None``: there is no crop.  The full
    :class:`~asymmetry.core.fitting.damped_line_scan.DampedLineAnalysis` is
    attached to :attr:`PeakAnalysis.damped_lines`, since the Δχ² threshold and
    the trial count live there and not on the peaks.

    ``signal`` should be the record the lines are actually in — the raw
    asymmetry, or the residual when a detrend curve is available.  The scan
    carries its own slow-decay nuisance dictionary, so a constant offset or a
    slow relaxation left in ``signal`` is absorbed rather than fitted.
    """
    from asymmetry.core.fitting.damped_line_scan import detect_damped_lines

    t = np.asarray(time, dtype=float)
    y = np.asarray(signal, dtype=float)
    err = np.asarray(error, dtype=float)
    analysis = detect_damped_lines(t, y, err, max_lines=int(max_lines))

    scan_peaks = [
        DetectedPeak(
            frequency_mhz=float(line.frequency_mhz),
            # No windowed magnitude exists for a line the scan measured
            # directly; the fitted amplitude is the honest stand-in and is what
            # the multiplet amplitude-share seeding consumes.
            amplitude=abs(float(line.amplitude_percent)),
            snr=float(line.snr),
            width_mhz=float(line.damping_rate_per_us / np.pi),
            prominence=0.0,
            source="damped_scan",
            burg_confirmed=None,
            crop_us=None,
            damping_rate_per_us=float(line.damping_rate_per_us),
            amplitude_percent=float(line.amplitude_percent),
            phase_rad=float(line.phase_rad),
            delta_chi_squared=float(line.delta_chi_squared),
        )
        for line in analysis.lines
    ]

    # Two accepted lines closer together than the wider one's own FWHM are one
    # line as far as *seeding* is concerned: the consumer builds one damped
    # cosine per peak, and two cosines inside a single linewidth are not a
    # multiplet the fit can separate — they are one broad line the scan's
    # peeling split in two, and seeding both spends a whole component on the
    # duplicate. The scan's own separation rule is deliberately looser (it is
    # deciding significance, where an over-split pair is the safe error); this
    # is the tighter rule the seeding side needs. Lines arrive Δχ²-descending,
    # so the survivor is always the better-determined of a pair.
    peaks: list[DetectedPeak] = []
    for peak in scan_peaks:
        if any(
            abs(peak.frequency_mhz - kept.frequency_mhz) < max(peak.width_mhz, kept.width_mhz)
            for kept in peaks
        ):
            continue
        peaks.append(peak)

    nyquist = 0.0
    if t.size > 1:
        dt = float(np.median(np.diff(t)))
        if dt > 0.0:
            nyquist = 0.5 / dt
    return PeakAnalysis(
        peaks=tuple(peaks),
        noise_floor=0.0,
        # A scan peak is matched at its own linewidth, not at a global
        # resolution; this is only the fallback for a peak with no width.
        resolution_mhz=float(max(min((peak.width_mhz for peak in peaks), default=_EPS), _EPS)),
        nyquist_mhz=nyquist,
        detrended=False,
        damped_lines=analysis,
    )


def merge_damped_scan_peaks(
    analysis: PeakAnalysis,
    scan: PeakAnalysis,
    *,
    max_peaks: int = 6,
) -> PeakAnalysis:
    """Fold damped-scan peaks into a Hann-pass analysis.

    The merge policy is :func:`merge_early_peaks`'s, with one addition that only
    the scan pass makes possible:

    * **The Hann pass still wins a collision on frequency** — for a line it can
      see at all, a transform of the whole informative record localises it
      better than a scan rung does.  The collision radius is the scan line's own
      linewidth ``λ/π`` (or the Hann resolution, whichever is coarser), because
      that is the width of the thing being matched.
    * **...but it inherits the measurement.**  The scan's λ, amplitude, phase
      and Δχ² are copied onto the surviving Hann peak.  Dropping them would
      throw away the only envelope estimate in the pipeline for a line both
      passes can see — precisely the seeding information the wizard needs.
    * Scan peaks that collide with nothing are **additions**, capped by
      ``max_peaks`` with ``_EARLY_RESERVED_PEAKS`` slots reserved so a full Hann
      peak set cannot starve the pass that exists to see what Hann cannot.
    * ``damped_lines`` is always attached, even when the scan found nothing —
      the threshold and trial count are the evidence that it looked.
    """
    kept = list(analysis.peaks)
    additions: list[DetectedPeak] = []
    for peak in scan.peaks:
        radius = max(float(peak.width_mhz), float(analysis.resolution_mhz))
        match_idx: int | None = None
        best = math.inf
        for i, existing in enumerate(kept):
            if existing.source == "damped_scan":
                continue
            distance = abs(existing.frequency_mhz - peak.frequency_mhz)
            if distance <= radius and distance < best:
                best = distance
                match_idx = i
        if match_idx is None:
            additions.append(peak)
            continue
        kept[match_idx] = replace(
            kept[match_idx],
            damping_rate_per_us=peak.damping_rate_per_us,
            amplitude_percent=peak.amplitude_percent,
            phase_rad=peak.phase_rad,
            delta_chi_squared=peak.delta_chi_squared,
        )

    if additions:
        cap = max(0, int(max_peaks))
        reserved = min(len(additions), _EARLY_RESERVED_PEAKS)
        kept_existing = kept[: max(0, cap - reserved)]
        kept = kept_existing + additions[: max(0, cap - len(kept_existing))]

    return replace(analysis, peaks=tuple(kept), damped_lines=scan.damped_lines)


def analyze_dataset_peaks(
    dataset: MuonDataset,
    *,
    detrend_curve: NDArray[np.float64] | None = None,
    detrend_template_key: str | None = None,
    max_peaks: int = 6,
    min_snr: float = 2.5,
    burg_check: str = "auto",
    damped_pass: bool = True,
) -> PeakAnalysis:
    """Detect oscillation lines in a time-domain dataset via FFT + Burg check.

    The residual (``asymmetry − detrend_curve`` when given, else tail-subtracted)
    is transformed with a Hann window and 4× zero-padding, then passed to
    :func:`detect_peaks_in_spectrum`.  A Burg (all-poles) cross-check may confirm
    but never add peaks — see ``burg_check``.

    A second, **matched-apodisation damped-line scan** then runs over the record
    (:func:`analyze_damped_scan_peaks`) and its peaks are merged in
    (:func:`merge_damped_scan_peaks`).  That pass is what lets the wizard see an
    oscillation whose lifetime is a small fraction of the record — the case the
    Hann window structurally deletes — and, unlike the crop ladder it replaced
    (:func:`analyze_early_window_peaks`, kept importable but no longer wired
    here), it *measures* the envelope rate instead of inferring it from a crop
    length.

    Parameters
    ----------
    detrend_curve
        Optional model curve aligned with ``dataset.time`` to subtract before
        transforming (the "residual FFT" path).
    detrend_template_key
        Provenance label recorded on the analysis when ``detrend_curve`` is used.
    burg_check
        ``"auto"`` (run when ``n_points < 512`` or two peaks fall within
        ``2·resolution``), ``"always"`` or ``"never"``.  The cross-check is
        scoped to the Hann pass: Burg is known to return a featureless
        1/f-like spectrum on a short, heavily damped window, so letting it
        judge damped-scan peaks would veto exactly the lines that pass exists
        to find.
    damped_pass
        Set ``False`` to run the historical Hann-only detection.
    """
    t_full = np.asarray(dataset.time, dtype=float)
    err_full = np.asarray(dataset.error, dtype=float)

    # SNR-truncate the window before transforming: the late-time error blow-up
    # (capped at 100 %) otherwise whitens the FFT and buries even strong lines.
    # dt (hence Nyquist) is unchanged; only the duration — and thus resolution —
    # reflects the effective window actually used.
    end = effective_analysis_window(t_full, err_full)
    t = t_full[:end]
    signal_full, detrended = _centered_signal(dataset, detrend_curve)
    signal = signal_full[:end]
    error = err_full[:end]

    n = t.size
    if n > 1:
        duration = float(t[-1] - t[0])
        dt = float(np.median(np.diff(t)))
    else:
        duration = 1.0
        dt = 1.0
    resolution_mhz = 1.0 / max(abs(duration), _EPS)
    nyquist_mhz = 1.0 / (2.0 * max(abs(dt), _EPS))

    source = "residual_fft" if detrend_curve is not None else "fft"

    # The Hann window here is an internal seeding choice, not the user's
    # apodisation, so the early-signal guard's advice ("use window='none'")
    # is not actionable at this call site and is suppressed. The known
    # consequence — this seeding pass is blind to an oscillation whose
    # lifetime is a small fraction of the record — is a property of the
    # detector, not of the user's settings.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ApodisationEarlySignalWarning)
        frequencies, _real, magnitude = fft_arrays(
            t,
            signal,
            error,
            window="hann",
            padding_factor=4,
        )

    analysis = detect_peaks_in_spectrum(
        frequencies,
        magnitude,
        resolution_mhz=resolution_mhz,
        max_peaks=max_peaks,
        min_snr=min_snr,
        source=source,
    )
    analysis = replace(
        analysis,
        nyquist_mhz=nyquist_mhz,
        detrended=detrended,
        detrend_template_key=(detrend_template_key if detrend_curve is not None else None),
    )

    if _should_run_burg(burg_check, n, analysis.peaks, resolution_mhz):
        analysis = _apply_burg_cross_check(analysis, signal, frequencies, dt, resolution_mhz)

    if not damped_pass:
        return analysis

    # The scan sees the same residual the Hann pass does — ``asymmetry −
    # detrend_curve`` when a curve was given, tail-subtracted otherwise — but
    # over the FULL record: it applies its own informative-window truncation,
    # and its nuisance basis already carries a constant, so the tail
    # subtraction is a no-op for it either way.
    #
    # Merged AFTER the Burg cross-check so scan peaks never reach it: Burg is
    # unreliable on short damped windows and would veto them (see the
    # ``burg_check`` note above).
    scan = analyze_damped_scan_peaks(t_full, signal_full, err_full)
    return merge_damped_scan_peaks(analysis, scan, max_peaks=max_peaks)


def _should_run_burg(
    burg_check: str,
    n_points: int,
    peaks: tuple[DetectedPeak, ...],
    resolution_mhz: float,
) -> bool:
    """Decide whether the Burg cross-check runs for this analysis."""
    mode = str(burg_check).strip().lower()
    if mode == "always":
        return True
    if mode == "never":
        return False
    # "auto": short record, or any two detected peaks closer than 2·resolution.
    if n_points < 512:
        return True
    freqs = sorted(peak.frequency_mhz for peak in peaks)
    for lo, hi in zip(freqs, freqs[1:]):
        if abs(hi - lo) < 2.0 * resolution_mhz:
            return True
    return False


def _apply_burg_cross_check(
    analysis: PeakAnalysis,
    signal: NDArray[np.float64],
    frequencies: NDArray[np.float64],
    dt_us: float,
    resolution_mhz: float,
) -> PeakAnalysis:
    """Confirm (never add) detected peaks against a Burg all-poles spectrum."""
    from scipy.signal import find_peaks

    burg_mag, burg_order, hit_boundary = burg_spectrum(signal, frequencies, float(dt_us))

    burg_peak_idx, _props = find_peaks(np.asarray(burg_mag, dtype=float))
    burg_peak_freqs = np.asarray(frequencies, dtype=float)[burg_peak_idx]

    confirmed_peaks: list[DetectedPeak] = []
    for peak in analysis.peaks:
        tol = max(resolution_mhz, peak.width_mhz)
        if burg_peak_freqs.size:
            nearest = float(np.min(np.abs(burg_peak_freqs - peak.frequency_mhz)))
            confirmed = nearest <= tol
        else:
            confirmed = False
        confirmed_peaks.append(replace(peak, burg_confirmed=confirmed))

    return replace(
        analysis,
        peaks=tuple(confirmed_peaks),
        burg_order=int(burg_order),
        burg_hit_boundary=bool(hit_boundary),
    )


# --------------------------------------------------------------------------- #
# User-declared peaks
# --------------------------------------------------------------------------- #


def _inherit_damped_scan_measurement(
    peak: DetectedPeak, candidates: Sequence[DetectedPeak]
) -> DetectedPeak:
    """Copy the nearest damped-scan line's measurement onto ``peak``.

    Only a ``"damped_scan"`` line whose own width ``λ/π`` brackets the offset
    qualifies, and the nearest such line wins.  The width comes along with the
    rest: it is what the seeding path's frequency bounds are cut from, and a
    user click's nominal width (the Hann resolution) would bound a heavily
    damped line far tighter than the line is wide.
    """
    best: DetectedPeak | None = None
    best_distance = math.inf
    for candidate in candidates:
        if candidate.source != "damped_scan" or candidate.damping_rate_per_us is None:
            continue
        distance = abs(candidate.frequency_mhz - peak.frequency_mhz)
        if distance <= max(candidate.width_mhz, _EPS) and distance < best_distance:
            best_distance = distance
            best = candidate
    if best is None:
        return peak
    return replace(
        peak,
        width_mhz=best.width_mhz,
        damping_rate_per_us=best.damping_rate_per_us,
        amplitude_percent=best.amplitude_percent,
        phase_rad=best.phase_rad,
        delta_chi_squared=best.delta_chi_squared,
    )


def merge_user_peaks(
    analysis: PeakAnalysis, user_frequencies_mhz: NDArray[np.float64]
) -> PeakAnalysis:
    """Fold user-declared frequencies into an analysis.

    A user frequency within one resolution element of an existing detected peak
    *replaces* it — keeping the detected amplitude/width but flagging
    ``source="user"`` with the sentinel SNR.  The radius is the matched peak's
    OWN pass resolution (an early-window peak's crop resolution, not the Hann
    pass's), so a user frequency and the early-pass line it corresponds to
    collapse into one seed rather than two.  Otherwise the frequency is added as
    a fresh user peak.  User peaks sort first and are never dropped; no
    ``max_peaks`` cap is re-applied here, and the relative order of the detected
    peaks — which is per-pass, since SNRs are not comparable across passes — is
    preserved.

    A *fresh* user frequency (one that matched nothing at that radius) still
    inherits the λ/amplitude/phase/Δχ² of a ``"damped_scan"`` peak lying within
    one linewidth of it, and that peak's width with them.  The radius above is
    the Hann pass's spectral resolution, orders of magnitude finer than a
    heavily damped line's own ``λ/π`` width, so a user click on such a line
    lands "fresh" while plainly naming the same oscillation — and the envelope
    the scan measured is the one piece of seeding information the click itself
    cannot carry.
    """
    user_freqs = [float(f) for f in np.atleast_1d(np.asarray(user_frequencies_mhz, dtype=float))]
    resolution = float(max(analysis.resolution_mhz, _EPS))

    remaining = list(analysis.peaks)
    merged: list[DetectedPeak] = []
    for freq in user_freqs:
        match_idx: int | None = None
        best = 0.0
        for i, peak in enumerate(remaining):
            radius = _peak_resolution_mhz(peak, resolution)
            distance = abs(peak.frequency_mhz - freq)
            if distance <= radius and (match_idx is None or distance < best):
                best = distance
                match_idx = i
        if match_idx is not None:
            existing = remaining.pop(match_idx)
            merged.append(
                replace(
                    existing,
                    frequency_mhz=freq,
                    snr=USER_PEAK_SNR_SENTINEL,
                    source="user",
                    burg_confirmed=None,
                )
            )
        else:
            fresh = DetectedPeak(
                frequency_mhz=freq,
                amplitude=0.0,
                snr=USER_PEAK_SNR_SENTINEL,
                width_mhz=resolution,
                prominence=0.0,
                source="user",
                burg_confirmed=None,
            )
            merged.append(_inherit_damped_scan_measurement(fresh, remaining))

    # Stable: user peaks first, everything else in the order the passes set.
    combined = merged + remaining
    combined.sort(key=lambda p: p.source != "user")
    return replace(analysis, peaks=tuple(combined))


# --------------------------------------------------------------------------- #
# Multiplet pattern matching
# --------------------------------------------------------------------------- #

#: Relative frequency tolerance for single-line and pair matching (combined with
#: the spectral resolution: ``tol(f) = max(2*resolution, _MULTIPLET_REL_TOL*f)``).
_MULTIPLET_REL_TOL = 0.04

#: Relative tolerance on the frequency *ratios* of three-line signatures.
_TRIPLET_RATIO_TOL = 0.05

#: F-mu-F collinear line positions in units of ``omega_d / 2*pi`` (see
#: ``muon_fluorine.polarization.linear_fmuf_polarization``):
#: ``(3-sqrt(3))/2, sqrt(3), (3+sqrt(3))/2`` — ratios ``1 : 1+sqrt(3) : 2+sqrt(3)``.
_FMUF_LINE_FACTORS = (
    0.5 * (3.0 - np.sqrt(3.0)),
    np.sqrt(3.0),
    0.5 * (3.0 + np.sqrt(3.0)),
)

#: Single-fluorine mu-F line positions in units of ``omega_d / 2*pi``
#: (``mu_f_polarization``): ratios ``1 : 2 : 3``.
_MUF_LINE_FACTORS = (0.5, 1.0, 1.5)

#: Physical bracket for the muonium hyperfine constant (MHz); vacuum muonium is
#: 4463.302 and shallow-donor/radical states reach far below it.
_A_HF_BRACKET_MHZ = (10.0, 4700.0)

#: SNR cap used when weighting multi-line frequency estimates, so a
#: user-declared peak (sentinel SNR) guides but does not annihilate the
#: detected lines' contributions.
_MATCH_WEIGHT_SNR_CAP = 100.0


@dataclass(frozen=True)
class MultipletMatch:
    """A recognised physical line pattern within a :class:`PeakAnalysis`.

    Attributes
    ----------
    kind
        ``"larmor"`` | ``"muonium_low_tf"`` | ``"muonium_high_tf"`` |
        ``"muonium_zf"`` | ``"fmuf_linear"`` | ``"muf"``.
    family_key
        The wizard candidate family this match promotes (``"oscillatory"``,
        ``"muonium"`` or ``"fmuf"``).
    peak_indices
        Indices into ``PeakAnalysis.peaks`` of the constituent lines.
    quality
        ``1 - mismatch/tolerance`` of the worst constituent line, in [0, 1].
    derived_values
        Physics quantities implied by the match, as ``(name, value)`` pairs —
        e.g. ``("a_hf_mhz", ...)``, ``("r_muF_angstrom", ...)``,
        ``("field_gauss", ...)`` — kept as a tuple so the dataclass stays
        frozen/hashable and trivially serializable.
    note
        Human-readable explanation for GUI display.
    """

    kind: str
    family_key: str
    peak_indices: tuple[int, ...]
    quality: float
    derived_values: tuple[tuple[str, float], ...]
    note: str

    def derived(self, name: str) -> float | None:
        """Return the derived value called ``name``, or ``None``."""
        for key, value in self.derived_values:
            if key == name:
                return value
        return None


def serialize_multiplet_match(match: MultipletMatch) -> dict[str, object]:
    """Return a JSON-safe dict snapshot of a :class:`MultipletMatch`."""
    return {
        "kind": str(match.kind),
        "family_key": str(match.family_key),
        "peak_indices": [int(i) for i in match.peak_indices],
        "quality": float(match.quality),
        "derived_values": [[str(k), float(v)] for k, v in match.derived_values],
        "note": str(match.note),
    }


def deserialize_multiplet_match(payload: object) -> MultipletMatch | None:
    """Rebuild a :class:`MultipletMatch` from a persisted dict, tolerating gaps."""
    if not isinstance(payload, dict):
        return None
    derived = tuple(
        (str(entry[0]), float(entry[1]))
        for entry in payload.get("derived_values", [])
        if isinstance(entry, (list, tuple)) and len(entry) == 2
    )
    return MultipletMatch(
        kind=str(payload.get("kind", "")),
        family_key=str(payload.get("family_key", "")),
        peak_indices=tuple(int(i) for i in payload.get("peak_indices", [])),
        quality=float(payload.get("quality", 0.0)),
        derived_values=derived,
        note=str(payload.get("note", "")),
    )


def _tolerance_mhz(frequency_mhz: float, resolution_mhz: float) -> float:
    """Line-position tolerance: two resolution elements or 4 % of the frequency."""
    return max(2.0 * resolution_mhz, _MULTIPLET_REL_TOL * abs(frequency_mhz))


def _quality(mismatches_over_tolerances: list[float]) -> float:
    """Map the worst relative mismatch onto a [0, 1] quality score."""
    worst = max(mismatches_over_tolerances) if mismatches_over_tolerances else 1.0
    return float(np.clip(1.0 - worst, 0.0, 1.0))


def _weighted_mean(values: list[float], snrs: list[float]) -> float:
    weights = [min(max(s, 1.0), _MATCH_WEIGHT_SNR_CAP) for s in snrs]
    total = sum(weights)
    return sum(v * w for v, w in zip(values, weights)) / total


def _match_larmor(
    peaks: tuple[DetectedPeak, ...], resolution_mhz: float, field_gauss: float
) -> list[MultipletMatch]:
    from asymmetry.core.fitting.spectral import (
        field_gauss_to_frequency_mhz,
        frequency_mhz_to_field_gauss,
    )

    nu_d = field_gauss_to_frequency_mhz(field_gauss)
    if nu_d <= 0.0:
        return []
    tol = _tolerance_mhz(nu_d, resolution_mhz)
    matches: list[MultipletMatch] = []
    for i, peak in enumerate(peaks):
        mismatch = abs(peak.frequency_mhz - nu_d)
        if mismatch > tol:
            continue
        matches.append(
            MultipletMatch(
                kind="larmor",
                family_key="oscillatory",
                peak_indices=(i,),
                quality=_quality([mismatch / tol]),
                derived_values=(("field_gauss", frequency_mhz_to_field_gauss(peak.frequency_mhz)),),
                note=(
                    f"line at {peak.frequency_mhz:.4g} MHz matches the muon Larmor "
                    f"frequency for {field_gauss:.4g} G — diamagnetic precession"
                ),
            )
        )
    return matches


def _match_muonium_low_tf(
    peaks: tuple[DetectedPeak, ...], resolution_mhz: float, field_gauss: float
) -> list[MultipletMatch]:
    from asymmetry.core.fitting.muonium import (
        a_hf_from_low_tf_pair,
        low_tf_pair_frequencies,
    )

    matches: list[MultipletMatch] = []
    n = len(peaks)
    for i in range(n):
        for j in range(i + 1, n):
            f_a, f_b = peaks[i].frequency_mhz, peaks[j].frequency_mhz
            f_lo, f_hi = min(f_a, f_b), max(f_a, f_b)
            a_hf = a_hf_from_low_tf_pair(field_gauss, f_lo, f_hi, a_hf_range_mhz=_A_HF_BRACKET_MHZ)
            if a_hf is None:
                continue
            pred_lo, pred_hi = low_tf_pair_frequencies(field_gauss, a_hf)
            checks = []
            ok = True
            for observed, predicted in ((f_lo, pred_lo), (f_hi, pred_hi)):
                tol = _tolerance_mhz(predicted, resolution_mhz)
                mismatch = abs(observed - predicted)
                if mismatch > tol:
                    ok = False
                    break
                checks.append(mismatch / tol)
            if not ok:
                continue
            matches.append(
                MultipletMatch(
                    kind="muonium_low_tf",
                    family_key="muonium",
                    peak_indices=(i, j),
                    quality=_quality(checks),
                    derived_values=(("a_hf_mhz", a_hf),),
                    note=(
                        f"pair at {f_lo:.4g}/{f_hi:.4g} MHz fits the low-TF muonium "
                        f"doublet at {field_gauss:.4g} G with A_hf ≈ {a_hf:.4g} MHz"
                    ),
                )
            )
    return matches


def _match_muonium_high_tf(
    peaks: tuple[DetectedPeak, ...], resolution_mhz: float, field_gauss: float
) -> list[MultipletMatch]:
    from asymmetry.core.fitting.muonium import high_tf_pair_frequencies

    a_min, a_max = _A_HF_BRACKET_MHZ
    matches: list[MultipletMatch] = []
    n = len(peaks)
    for i in range(n):
        for j in range(i + 1, n):
            f_a, f_b = peaks[i].frequency_mhz, peaks[j].frequency_mhz
            f_lo, f_hi = min(f_a, f_b), max(f_a, f_b)
            a_hf = f_lo + f_hi  # nu_12 + nu_34 = A_hf exactly
            if not (a_min <= a_hf <= a_max):
                continue
            pred = sorted(high_tf_pair_frequencies(field_gauss, a_hf))
            checks = []
            ok = True
            for observed, predicted in zip((f_lo, f_hi), pred):
                tol = _tolerance_mhz(max(predicted, 1.0), resolution_mhz)
                mismatch = abs(observed - predicted)
                if mismatch > tol:
                    ok = False
                    break
                checks.append(mismatch / tol)
            if not ok:
                continue
            matches.append(
                MultipletMatch(
                    kind="muonium_high_tf",
                    family_key="muonium",
                    peak_indices=(i, j),
                    quality=_quality(checks),
                    derived_values=(("a_hf_mhz", a_hf),),
                    note=(
                        f"pair at {f_lo:.4g}/{f_hi:.4g} MHz sums to "
                        f"A_hf ≈ {a_hf:.4g} MHz — high-TF muonium "
                        f"(nu_12 + nu_34 = A_hf)"
                    ),
                )
            )
    return matches


def _match_muonium_zf(
    peaks: tuple[DetectedPeak, ...], resolution_mhz: float
) -> list[MultipletMatch]:
    a_min, a_max = _A_HF_BRACKET_MHZ
    matches: list[MultipletMatch] = []
    n = len(peaks)
    for combo in _three_subsets(n):
        g = sorted(peaks[k].frequency_mhz for k in combo)
        # zf_muonium: f1 = A-D, f2 = A+D/2, f3 = 3D/2, so f3 == f2 - f1.
        f3, f1, f2 = g[0], g[1], g[2]
        tol = _tolerance_mhz(f3, resolution_mhz)
        mismatch = abs((f2 - f1) - f3)
        if mismatch > tol:
            continue
        d_mhz = 2.0 * f3 / 3.0
        a_hf = f1 + d_mhz
        if not (a_min <= a_hf <= a_max):
            continue
        matches.append(
            MultipletMatch(
                kind="muonium_zf",
                family_key="muonium",
                peak_indices=tuple(combo),
                quality=_quality([mismatch / tol]),
                derived_values=(("a_hf_mhz", a_hf), ("d_mhz", d_mhz)),
                note=(
                    f"lines at {f3:.4g}/{f1:.4g}/{f2:.4g} MHz satisfy the axial "
                    f"ZF-muonium relation f3 = f2 - f1 with A_hf ≈ {a_hf:.4g} MHz, "
                    f"D ≈ {d_mhz:.4g} MHz"
                ),
            )
        )
    return matches


def _three_subsets(n: int) -> list[tuple[int, int, int]]:
    return [(i, j, k) for i in range(n) for j in range(i + 1, n) for k in range(j + 1, n)]


def _match_dipolar_triplet(
    peaks: tuple[DetectedPeak, ...],
    factors: tuple[float, float, float],
    kind: str,
    note_label: str,
) -> list[MultipletMatch]:
    from asymmetry.core.fitting.muon_fluorine.dipolar import r_mu_f_from_omega_d

    t2 = factors[1] / factors[0]
    t3 = factors[2] / factors[0]
    matches: list[MultipletMatch] = []
    for combo in _three_subsets(len(peaks)):
        chosen = sorted((peaks[k] for k in combo), key=lambda p: p.frequency_mhz)
        g1, g2, g3 = (p.frequency_mhz for p in chosen)
        if g1 <= 0.0:
            continue
        r2_mismatch = abs(g2 / g1 - t2) / t2
        r3_mismatch = abs(g3 / g1 - t3) / t3
        if r2_mismatch > _TRIPLET_RATIO_TOL or r3_mismatch > _TRIPLET_RATIO_TOL:
            continue
        # Each line independently estimates omega_d/2pi; SNR-weight them.
        omega_tilde = _weighted_mean(
            [p.frequency_mhz / f for p, f in zip(chosen, factors)],
            [p.snr for p in chosen],
        )
        omega_d_rad_per_us = 2.0 * np.pi * omega_tilde
        r_muf = r_mu_f_from_omega_d(omega_d_rad_per_us)
        matches.append(
            MultipletMatch(
                kind=kind,
                family_key="fmuf",
                peak_indices=tuple(combo),
                quality=_quality(
                    [
                        r2_mismatch / _TRIPLET_RATIO_TOL,
                        r3_mismatch / _TRIPLET_RATIO_TOL,
                    ]
                ),
                derived_values=(
                    ("omega_d_mhz", omega_tilde),
                    ("r_muF_angstrom", r_muf),
                ),
                note=(
                    f"lines at {g1:.4g}/{g2:.4g}/{g3:.4g} MHz match the "
                    f"{note_label} signature (ratios 1 : {t2:.3f} : {t3:.3f}) — "
                    f"r_muF ≈ {r_muf:.3g} Å"
                ),
            )
        )
    return matches


def match_multiplets(
    analysis: PeakAnalysis,
    *,
    field_gauss: float | None,
    geometry: str | None,
) -> tuple[MultipletMatch, ...]:
    """Recognise physical line patterns among the detected peaks.

    Parameters
    ----------
    analysis
        A peak analysis (typically the detrended pass, optionally merged with
        user peaks).
    field_gauss
        Applied field from run metadata, or ``None`` when unknown.
    geometry
        ``"ZF"``, ``"TF"``, ``"LF"`` or ``None`` when the run geometry is not
        recorded.  Unknown geometry runs *all* rules (metadata-poor data must
        not lose pattern hints); a recorded geometry gates the rules to the
        physically meaningful subset.

    Returns
    -------
    tuple[MultipletMatch, ...]
        All recognised patterns, quality-descending.  A peak may participate in
        several matches; family promotion downstream is per family, and the GUI
        lists every match.
    """
    peaks = analysis.peaks
    if not peaks:
        return ()
    resolution = float(max(analysis.resolution_mhz, _EPS))
    geometry_token = geometry.strip().upper() if isinstance(geometry, str) else None
    if geometry_token not in ("ZF", "TF", "LF"):
        geometry_token = None

    transverse = geometry_token in ("TF", None)
    zero_or_longitudinal = geometry_token in ("ZF", "LF", None)

    matches: list[MultipletMatch] = []
    has_field = field_gauss is not None and field_gauss > 0.0
    if transverse and has_field:
        matches.extend(_match_larmor(peaks, resolution, float(field_gauss)))
        matches.extend(_match_muonium_low_tf(peaks, resolution, float(field_gauss)))
        matches.extend(_match_muonium_high_tf(peaks, resolution, float(field_gauss)))
    if zero_or_longitudinal:
        matches.extend(
            _match_dipolar_triplet(peaks, _FMUF_LINE_FACTORS, "fmuf_linear", "collinear F-mu-F")
        )
        matches.extend(
            _match_dipolar_triplet(peaks, _MUF_LINE_FACTORS, "muf", "single-fluorine mu-F")
        )
    if geometry_token in ("ZF", None):
        matches.extend(_match_muonium_zf(peaks, resolution))

    matches.sort(key=lambda m: m.quality, reverse=True)
    return tuple(matches)
