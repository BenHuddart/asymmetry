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

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import median_filter

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fourier.burg import burg_spectrum
from asymmetry.core.fourier.fft import fft_asymmetry

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


def _sidelobe_ceiling(delta_mhz: float, resolution_mhz: float) -> float:
    """Max amplitude ratio a genuine line needs at ``delta_mhz`` from a stronger one.

    Anchored just above the Hann window's worst sidelobe (-31.5 dB near the main
    lobe) and rolled off as ``delta^-3`` (the window's -18 dB/octave sidelobe
    decay), so leakage structure is rejected while genuine weak lines — which sit
    above the local sidelobe level — survive at any separation.
    """
    anchor = _SIDELOBE_ANCHOR_RESOLUTIONS * resolution_mhz
    return _SIDELOBE_CEILING * (anchor / max(delta_mhz, anchor)) ** 3


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
        Provenance: ``"fft"``, ``"residual_fft"`` or ``"user"``.
    burg_confirmed
        ``True``/``False`` when a Burg cross-check ran and did / did not find a
        matching all-poles local maximum; ``None`` when no cross-check ran.
    """

    frequency_mhz: float
    amplitude: float
    snr: float
    width_mhz: float
    prominence: float
    source: str
    burg_confirmed: bool | None = None


@dataclass(frozen=True)
class PeakAnalysis:
    """The outcome of a peak-detection pass over one spectrum.

    ``peaks`` is ordered by SNR descending.
    """

    peaks: tuple[DetectedPeak, ...]
    noise_floor: float
    resolution_mhz: float
    nyquist_mhz: float
    detrended: bool
    detrend_template_key: str | None = None
    burg_order: int | None = None
    burg_hit_boundary: bool = False


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
    }


def deserialize_detected_peak(payload: object) -> DetectedPeak | None:
    """Rebuild a :class:`DetectedPeak` from a persisted dict, tolerating gaps."""
    if not isinstance(payload, dict):
        return None
    burg = payload.get("burg_confirmed", None)
    return DetectedPeak(
        frequency_mhz=float(payload.get("frequency_mhz", 0.0)),
        amplitude=float(payload.get("amplitude", 0.0)),
        snr=float(payload.get("snr", 0.0)),
        width_mhz=float(payload.get("width_mhz", 0.0)),
        prominence=float(payload.get("prominence", 0.0)),
        source=str(payload.get("source", "fft")),
        burg_confirmed=(bool(burg) if burg is not None else None),
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
    mad = float(np.median(np.abs(residual - np.median(residual))))
    sigma = 1.4826 * mad
    if sigma > _EPS:
        clipped = mags.copy()
        outliers = residual > 3.0 * sigma
        # Replace strong lines with the current floor estimate so the refined
        # running median is not polluted by them, then recompute.
        clipped[outliers] = floor[outliers]
        floor = _running_median(clipped, window)
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
    guard = max(3.0 * df, 0.5 * resolution_mhz)
    valid = (freqs > guard) & (freqs < nyquist - guard)
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
    local_floor = _local_noise_floor(mags, bins_per_resolution)
    representative_floor = float(np.median(local_floor[valid]))

    # Look-elsewhere-corrected SNR gate.  For Gaussian time-domain noise the
    # magnitude bins are Rayleigh with P(X > k*median) = 2^(-k^2), so across
    # ~n_res independent resolution elements the tallest noise excursion
    # reaches ~sqrt(log2(n_res)) median units; gate at the level where the
    # expected false-peak count is _FALSE_PEAK_RATE.
    span = float(freqs[-1] - guard)
    n_res = max(2.0, span / resolution_mhz)
    adaptive_min_snr = float(np.sqrt(np.log2(n_res / _FALSE_PEAK_RATE)))
    effective_min_snr = max(float(min_snr), adaptive_min_snr)

    prominence = 2.0 * representative_floor
    distance = bins_per_resolution

    find_kwargs: dict[str, object] = {"distance": distance}
    if prominence > _EPS:
        find_kwargs["prominence"] = prominence
    peak_indices, properties = find_peaks(mags, **find_kwargs)

    # Restrict to the guarded positive band.
    keep = valid[peak_indices]
    peak_indices = peak_indices[keep]
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

    detected: list[DetectedPeak] = []
    for k, idx in enumerate(peak_indices):
        idx = int(idx)
        delta_bins, peak_log = _parabolic_interpolation(log_mag, idx)
        freq = float(freqs[idx] + delta_bins * df)
        amplitude = float(np.exp(peak_log))
        floor_here = float(max(local_floor[idx], _EPS))
        snr = amplitude / floor_here
        if snr < effective_min_snr:
            continue
        width_mhz = float(max(widths_samples[k], 0.0) * df)
        detected.append(
            DetectedPeak(
                frequency_mhz=freq,
                amplitude=amplitude,
                snr=float(snr),
                width_mhz=width_mhz,
                prominence=float(prominences[k]),
                source=source,
            )
        )

    # Windowing-leakage guard: walk peaks strongest-first and drop any peak
    # sitting below the sidelobe ceiling of an already-accepted stronger line.
    detected.sort(key=lambda p: p.amplitude, reverse=True)
    accepted: list[DetectedPeak] = []
    for peak in detected:
        is_sidelobe = any(
            peak.amplitude
            < other.amplitude
            * _sidelobe_ceiling(abs(peak.frequency_mhz - other.frequency_mhz), resolution_mhz)
            for other in accepted
        )
        if not is_sidelobe:
            accepted.append(peak)

    accepted.sort(key=lambda p: p.snr, reverse=True)
    detected = accepted[: max(0, int(max_peaks))]

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


def analyze_dataset_peaks(
    dataset: MuonDataset,
    *,
    detrend_curve: NDArray[np.float64] | None = None,
    detrend_template_key: str | None = None,
    max_peaks: int = 6,
    min_snr: float = 2.5,
    burg_check: str = "auto",
) -> PeakAnalysis:
    """Detect oscillation lines in a time-domain dataset via FFT + Burg check.

    The residual (``asymmetry − detrend_curve`` when given, else tail-subtracted)
    is transformed with a Hann window and 4× zero-padding, then passed to
    :func:`detect_peaks_in_spectrum`.  A Burg (all-poles) cross-check may confirm
    but never add peaks — see ``burg_check``.

    Parameters
    ----------
    detrend_curve
        Optional model curve aligned with ``dataset.time`` to subtract before
        transforming (the "residual FFT" path).
    detrend_template_key
        Provenance label recorded on the analysis when ``detrend_curve`` is used.
    burg_check
        ``"auto"`` (run when ``n_points < 512`` or two peaks fall within
        ``2·resolution``), ``"always"`` or ``"never"``.
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

    fft_dataset = MuonDataset(
        time=t.copy(),
        asymmetry=signal.copy(),
        error=error.copy(),
        metadata=dict(dataset.metadata),
        run=dataset.run,
    )
    frequencies, _real, magnitude = fft_asymmetry(
        fft_dataset,
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

    if not _should_run_burg(burg_check, n, analysis.peaks, resolution_mhz):
        return analysis

    return _apply_burg_cross_check(analysis, signal, frequencies, dt, resolution_mhz)


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


def merge_user_peaks(
    analysis: PeakAnalysis, user_frequencies_mhz: NDArray[np.float64]
) -> PeakAnalysis:
    """Fold user-declared frequencies into an analysis.

    A user frequency within one ``resolution_mhz`` of an existing detected peak
    *replaces* it — keeping the detected amplitude/width but flagging
    ``source="user"`` with the sentinel SNR.  Otherwise it is added as a fresh
    user peak.  User peaks sort first (sentinel SNR) and are never dropped; no
    ``max_peaks`` cap is re-applied here.
    """
    user_freqs = [float(f) for f in np.atleast_1d(np.asarray(user_frequencies_mhz, dtype=float))]
    resolution = float(max(analysis.resolution_mhz, _EPS))

    remaining = list(analysis.peaks)
    merged: list[DetectedPeak] = []
    for freq in user_freqs:
        match_idx: int | None = None
        best = resolution
        for i, peak in enumerate(remaining):
            distance = abs(peak.frequency_mhz - freq)
            if distance <= best:
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
            merged.append(
                DetectedPeak(
                    frequency_mhz=freq,
                    amplitude=0.0,
                    snr=USER_PEAK_SNR_SENTINEL,
                    width_mhz=resolution,
                    prominence=0.0,
                    source="user",
                    burg_confirmed=None,
                )
            )

    combined = merged + remaining
    combined.sort(key=lambda p: p.snr, reverse=True)
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
