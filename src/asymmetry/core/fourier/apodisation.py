"""Data-driven matched-apodisation suggestion, and the early-signal guard.

Two things live here. :func:`suggest_matched_apodisation` computes the matched
filter for a spectrum's dominant line. :func:`early_signal_apodisation_loss` is
the opposite check — it measures how much early-time signal power an
apodisation already in force has *removed*, and backs
:class:`ApodisationEarlySignalWarning`, which the FFT prepare path raises when a
symmetric taper (``hann``/``cosine``, zero at the record's ends) has deleted a
heavily damped oscillation. That failure mode is concrete: a signal whose
lifetime is a small fraction of the record can read as flat noise under Hann
while being many-sigma with no window and a crop, or under the exponential
("lorentzian") filter with ``filter_time_constant_us ≈ 1/λ``.

Advisory only — nothing in the core (or the GUI) ever applies a suggested
filter automatically. A matched filter maximises a line's peak S/N at the cost
of roughly doubling its apparent width, so applying one is a decision the user
must make knowingly; this module only computes what the matched values *would
be* from the unapodised spectrum.

Widths are measured on the POWER spectrum ``|F|^2``, not the magnitude: the
one-sided transform of a damped cosine carries a dispersion part alongside the
absorption line, and at half-maximum the *magnitude* of a Lorentzian is a
factor ``sqrt(3)`` wider than the absorption shape (the first implementation
measured magnitude and recovered relaxation rates ~1.7x too large). On the
power spectrum both shapes have closed-form half-widths.

The window parameterisation matches :func:`asymmetry.core.fourier.window.
apply_fft_filter` (``start_time_us=0``); with the power-spectrum FWHM
``Gamma`` in MHz and time constants in µs:

* Lorentzian weight ``exp(-t/tau)``. The one-sided transform of
  ``exp(-lambda t) cos(w0 t)`` has power ``1/4 / (lambda^2 + dw^2)`` exactly —
  a true Lorentzian of FWHM ``Gamma = lambda / pi`` (MHz) — so
  ``tau = 1 / (pi Gamma)``.
* Gaussian weight ``exp(-(t/tau)^2)``. For a Gaussian envelope
  ``exp(-sigma^2 t^2 / 2)`` the one-sided power line is
  ``(pi/2) exp(-u^2) + 2 Dawson(u / sqrt 2)^2`` (in ``u = dw / sigma`` units);
  its half-maximum falls at ``u* = 1.42294`` (numerical root — the Dawson
  dispersion tail broadens it well past the pure-Gaussian ``sqrt(ln 2)``), so
  ``sigma = pi Gamma / u*`` and the matched
  ``tau = sqrt(2) / sigma = sqrt(2) u* / (pi Gamma)``.

Detection is two-stage. The fast path (:func:`_prominence_line`) tests the
raw, unsmoothed peak against the window's median power — cheap, and correct
whenever the line already towers over the noise. A line that is genuinely
present but sits below that raw threshold (e.g. an un-windowed,
lifetime-corrected record whose late-time noise is amplified by the
``e^{t/tau}`` correction) needs the power concentrated before it is visible:
the fallback (:func:`_matched_scan_fallback`) convolves the windowed power
spectrum with a family of normalised kernels (Lorentzian or Gaussian,
matching the requested filter kind) spanning roughly 10 geometrically-spaced
widths, and keeps the width whose robust SNR — median/MAD floor, with the
peak region excluded so a real line cannot inflate its own floor — is
highest.

The scanned kernel widths are anchored to the spectrum's INTRINSIC
resolution (``1 / T_window`` MHz, the unpadded transform's bin spacing), not
to the padded grid's bin width: on a zero-padded spectrum adjacent grid bins
are correlated over one resolution element, so a kernel a few *grid* bins
wide smooths nothing, and the unsmoothed power-spectrum noise is
exponentially distributed — its maximum over thousands of bins hugely
exceeds a median + 8 x MAD floor, which would yield false detections on pure
noise. With every kernel at least ~4 independent resolution elements wide
the smoothed noise is near-Gaussian and the SNR threshold has real headroom.
Callers should pass ``intrinsic_resolution_mhz``; when they cannot, the
resolution is estimated from the half-maximum lag of the windowed power
spectrum's autocorrelation (the correlation length of the noise).

Smoothing broadens whatever it detects, so the kernel's own width is
then removed from the measured FWHM before any physical quantity is derived
from it: linearly for a Lorentzian kernel (widths add), in quadrature for a
Gaussian one (variances add). The same resolution-limited guard
(``_MIN_FWHM_BINS``) is applied to that deconvolved width, not the observed
one, so a smoothed noise ripple cannot masquerade as a physical line.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: A candidate line's POWER must rise this far above the search window's
#: baseline power to be worth matching — 4x in amplitude, squared because the
#: search runs on the power spectrum (the plot framing's amplitude convention
#: is 4.0). Below it, "matching" would chase a noise spike.
_LINE_PROMINENCE_POWER = 16.0

#: Half-maximum of the one-sided Gaussian-envelope power line in units of
#: ``dw / sigma`` — the numerical root of
#: ``(pi/2) exp(-u^2) + 2 Dawson(u / sqrt 2)^2 = pi / 4``.
_GAUSSIAN_POWER_HALF_WIDTH = 1.42294

#: Fraction of the frequency span treated as the DC region (zero-frequency
#: peak plus filter rolloff) and excluded from the line search, matching the
#: plot-framing convention.
_DC_CUT_FRACTION = 0.02

#: A measured FWHM narrower than this many frequency bins is resolution-limited
#: — the width is the transform's, not the sample's — so there is no physical
#: relaxation to match a filter to.
_MIN_FWHM_BINS = 2.0

#: Matched-scan fallback (see the module docstring): number of kernel widths
#: scanned, geometrically spaced between the low and high bounds below.
_MATCHED_SCAN_KERNEL_COUNT = 10

#: Narrowest kernel FWHM tried, as a multiple of the larger of the spectrum's
#: intrinsic resolution and its (padded) grid bin width — narrower than this
#: the kernel barely smooths anything (grid) or smooths within a single
#: resolution element, where padded-grid noise is not yet independent (see
#: module docstring).
_MATCHED_SCAN_MIN_KERNEL_BINS = 4.0

#: Conservative safety factor applied to the autocorrelation-based intrinsic
#: resolution estimate (see ``_estimate_intrinsic_resolution``) when a caller
#: cannot supply ``intrinsic_resolution_mhz`` directly. The half-maximum lag
#: of a zero-padded noise spectrum's autocorrelation systematically runs
#: ~0.42-0.5x the true intrinsic resolution (measured empirically across
#: padding factors 4-64 on white-noise power spectra — the mainlobe of the
#: padding kernel is wider than its own half-maximum lag), so the raw lag
#: under-estimates unless corrected. Under-estimating the resolution reopens
#: the sub-resolution false-positive the anchoring exists to prevent;
#: over-estimating only costs sensitivity to marginal lines, which is the
#: safe direction for an advisory suggestion — so this factor is set above
#: the empirical worst case (1 / 0.42 ~= 2.4), not at its center. Pinned
#: against the heavy-padding pure-noise case in test_apodisation_suggestion.py.
_RESOLUTION_ESTIMATE_SAFETY_FACTOR = 2.5

#: Widest kernel FWHM tried, in MHz — a matched filter for a line this broad
#: is not a useful suggestion regardless of detectability.
_MATCHED_SCAN_MAX_KERNEL_MHZ = 10.0

#: The widest kernel is also capped at (search-window span) / this fraction,
#: so the scan never smooths over the whole search window.
_MATCHED_SCAN_MAX_KERNEL_SPAN_FRACTION = 8.0

#: Kernel support, in units of its own half-width-at-half-max, used when
#: building the discrete convolution kernel (wide enough to include the
#: Lorentzian's slowly-decaying tail).
_MATCHED_SCAN_KERNEL_HALF_SPAN = 8.0

#: Region excluded from the robust median/MAD floor estimate, in units of the
#: detecting kernel's own FWHM either side of its peak — otherwise a strong
#: line inflates the floor computed "under" it and suppresses its own SNR.
_MATCHED_SCAN_EXCLUSION_KERNELS = 3.0

#: Minimum robust SNR (median/MAD floor) a smoothed candidate must clear to
#: be treated as a detection. Chosen from the real high-TF TDC dataset that
#: validated the fix (a
#: genuine ~1.6 MHz line buried below the raw-prominence threshold scans at
#: SNR ~13.7) with headroom above typical noise-only fluctuations of a
#: MAD-normalised scan (most pure-noise draws stay below ~6; see
#: test_apodisation_suggestion.py for the pinned values). Scanning ~10 kernel
#: widths against the peak of a thousands-of-bins window is itself a
#: look-elsewhere search, so a residual false-positive rate against
#: adversarial noise draws is expected — empirically a few percent of draws
#: produce a spurious detection with SNR comparable to (occasionally above)
#: that validating case, so no threshold cleanly separates every noise draw from
#: every genuine line. This mirrors the raw-prominence fast path, which has
#: the same residual risk against pure noise (hence that path's own test
#: pins a specific seed rather than asserting over an arbitrary one). This
#: is acceptable because the suggestion is advisory only — nothing applies
#: it automatically — so a false suggestion costs the user one look, not a
#: silent change to their analysis.
_MATCHED_SCAN_SNR_THRESHOLD = 8.0


class ApodisationEarlySignalWarning(UserWarning):
    """The apodisation in force has deleted the early-time signal.

    Raised by the shared FFT prepare core (so both the dataset and the array
    front doors carry it) when a front-loaded signal meets a symmetric taper
    window. Advisory only: no returned value changes.
    """


#: Leading share of the record treated as "early" by
#: :func:`early_signal_apodisation_loss`. A symmetric taper (hann/cosine) is
#: exactly zero at the first sample and climbs as ``(t/T)²``, so its damage is
#: concentrated in the first tenth or so of the record; 0.15 is wide enough
#: that a handful of bins cannot dominate the statistic and narrow enough that
#: an undamped signal puts only 15 % of its power inside it.
_EARLY_WINDOW_FRACTION = 0.15

#: Trigger condition 1: this much of the record's error-weighted power must
#: live in the early window. With flat errors a signal alive across the whole
#: record puts exactly ``_EARLY_WINDOW_FRACTION`` (0.15) of its power there —
#: but real μSR errors grow with time as the counts decay, and the 1/σ²
#: weighting tilts the statistic early on its own: an *undamped* line with
#: ``σ ∝ e^{t/2τ_μ}`` over an 8 µs record already reads 0.43. The threshold has
#: to clear that pure-weighting tilt, so it sits at 0.75, which an undamped or
#: weakly damped line cannot reach and a lifetime well inside the record does
#: comfortably (λ = 0.4 µs⁻¹ over 8 µs reads 0.78, λ = 60 µs⁻¹ over 0.3 µs
#: reads 0.996).
_EARLY_POWER_CONCENTRATION_TRIGGER = 0.75

#: Trigger condition 2: the window must keep no more than this share of that
#: early power. ``window="none"`` keeps 1.0. The *matched* exponential filter
#: (``window="lorentzian"``, ``filter_time_constant_us = 1/λ``) keeps exactly
#: 0.5 in the long-record limit — weight ``e^{-λt}`` against power ``e^{-2λt}``
#: gives ``∫e^{-4λt}/∫e^{-2λt} → 1/2`` — so the threshold has to sit well below
#: 0.5, or the very cure the warning recommends would trip it. A Hann window on
#: the same signal keeps ~1e-3. 0.2 splits those with more than a factor of two
#: of margin on each side. Both thresholds are pinned by the synthetic study in
#: ``tests/core/test_fourier_apodisation_guard.py``.
_EARLY_RETAINED_POWER_TRIGGER = 0.2

#: Below this many samples the early window holds too few bins for the power
#: statistics to mean anything, and the guard stands down.
_EARLY_GUARD_MIN_POINTS = 16

#: Fast bail-out: when the window's smallest weight over the early portion is
#: above this, it cannot have removed most of the early power, so the rest of
#: the statistic is skipped. Keeps the guard off the hot path for
#: ``window="none"`` and for gentle filters.
_EARLY_GUARD_MIN_WEIGHT_TO_CHECK = 0.9


@dataclass(frozen=True)
class EarlySignalApodisationLoss:
    """How much early-time signal power the applied apodisation removed."""

    #: Leading share of the record measured (``_EARLY_WINDOW_FRACTION``).
    early_window_fraction: float
    #: Share of the pre-window signal's error-weighted power living there.
    early_power_fraction: float
    #: Share of *that* power the window keeps (1.0 = untouched, 0.0 = deleted).
    early_retained_fraction: float
    #: True when both trigger conditions are met.
    triggered: bool


def early_signal_apodisation_loss(
    time_us: np.ndarray,
    signal: np.ndarray,
    weights: np.ndarray,
    error: np.ndarray | None = None,
) -> EarlySignalApodisationLoss | None:
    """Measure the early-time signal power an apodisation removed.

    *signal* is the pre-window signal (after any cropping, fractional footing
    and baseline removal) and *weights* the apodisation weights about to
    multiply it — the two the FFT prepare core has in hand. Power is
    ``|signal/σ|²`` when usable errors are supplied and ``|signal|²`` otherwise,
    so the well-measured bins dominate exactly as they do in a fit.

    Two numbers are computed over the leading ``_EARLY_WINDOW_FRACTION`` of the
    record: how much of the signal's total power lives there
    (``early_power_fraction``) and how much of that the window keeps
    (``early_retained_fraction``). ``triggered`` is set when the power is
    concentrated early *and* the window removes it — a heavily damped
    oscillation under a symmetric taper, which the taper's zero at ``t = 0``
    can erase entirely.

    Returns ``None`` when the measurement is not meaningful: too few samples,
    mismatched shapes, no finite power, or an apodisation that plainly cannot
    have done damage.
    """
    times = np.asarray(time_us, dtype=np.float64)
    values = np.asarray(signal, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    n = values.size
    if n < _EARLY_GUARD_MIN_POINTS or times.size != n or w.size != n:
        return None

    n_early = max(1, int(round(_EARLY_WINDOW_FRACTION * n)))
    early_weights = np.abs(w[:n_early])
    if not np.all(np.isfinite(early_weights)):
        return None
    if float(np.min(early_weights)) > _EARLY_GUARD_MIN_WEIGHT_TO_CHECK:
        return None

    power = np.square(values)
    if error is not None:
        sigma = np.asarray(error, dtype=np.float64)
        if sigma.shape == values.shape:
            usable = np.isfinite(sigma) & (sigma > 0.0)
            if np.any(usable):
                power = np.where(usable, np.square(values / np.where(usable, sigma, 1.0)), 0.0)
    power = np.where(np.isfinite(power), power, 0.0)

    total = float(np.sum(power, dtype=np.float64))
    early = float(np.sum(power[:n_early], dtype=np.float64))
    if not np.isfinite(total) or total <= 0.0 or early <= 0.0:
        return None

    retained = float(np.sum(power[:n_early] * np.square(early_weights), dtype=np.float64)) / early
    early_fraction = early / total
    return EarlySignalApodisationLoss(
        early_window_fraction=float(_EARLY_WINDOW_FRACTION),
        early_power_fraction=early_fraction,
        early_retained_fraction=retained,
        triggered=(
            early_fraction >= _EARLY_POWER_CONCENTRATION_TRIGGER
            and retained <= _EARLY_RETAINED_POWER_TRIGGER
        ),
    )


@dataclass(frozen=True)
class ApodisationSuggestion:
    """A matched-filter suggestion derived from one spectral line."""

    #: ``"lorentzian"`` or ``"gaussian"`` — the window kind that was matched.
    window: str
    #: Matched filter time constant in µs (``apply_fft_filter`` convention).
    time_constant_us: float
    #: Frequency (MHz) of the line the suggestion was matched to.
    line_frequency_mhz: float
    #: Measured POWER-spectrum FWHM (MHz) of that line, unapodised.
    line_fwhm_mhz: float


def _half_maximum_crossing(
    freqs: np.ndarray,
    values: np.ndarray,
    peak_index: int,
    half_level: float,
    step: int,
) -> float | None:
    """Interpolated frequency where *values* first crosses *half_level*.

    Walks from the peak in *step* direction (±1); ``None`` when the edge of the
    search window is reached first (the width cannot be measured).
    """
    index = peak_index
    while 0 <= index + step < freqs.size:
        nxt = index + step
        if values[nxt] <= half_level:
            v0, v1 = values[index], values[nxt]
            if v1 == v0:
                return float(freqs[nxt])
            fraction = (v0 - half_level) / (v0 - v1)
            return float(freqs[index] + fraction * (freqs[nxt] - freqs[index]))
        index = nxt
    return None


def _matched_time_constant(window_key: str, fwhm: float) -> float:
    """Matched filter time constant (µs) for a power-spectrum FWHM (MHz)."""
    if window_key == "lorentzian":
        return float(1.0 / (math.pi * fwhm))
    return float(math.sqrt(2.0) * _GAUSSIAN_POWER_HALF_WIDTH / (math.pi * fwhm))


def _prominence_line(
    f_win: np.ndarray, v_win: np.ndarray, window_key: str, bin_width: float
) -> ApodisationSuggestion | None:
    """Fast path: match the dominant RAW peak of the windowed power spectrum.

    Fires whenever the unsmoothed peak already towers over the window's
    median power (see ``_LINE_PROMINENCE_POWER``) — the cheap, common case.
    Returns ``None`` for anything else (no prominent peak, unmeasurable
    width, or a resolution-limited width), leaving the matched-scan fallback
    in :func:`suggest_matched_apodisation` to try harder.
    """
    baseline = float(np.median(v_win))
    if baseline <= 0.0:
        baseline = float(np.mean(v_win))
    peak_index = int(np.argmax(v_win))
    peak = float(v_win[peak_index])
    if baseline <= 0.0 or peak <= baseline * _LINE_PROMINENCE_POWER:
        return None

    half_level = baseline + 0.5 * (peak - baseline)
    left = _half_maximum_crossing(f_win, v_win, peak_index, half_level, -1)
    right = _half_maximum_crossing(f_win, v_win, peak_index, half_level, +1)
    if left is None or right is None:
        return None
    fwhm = float(right - left)
    if not np.isfinite(fwhm) or fwhm < _MIN_FWHM_BINS * bin_width:
        return None

    return ApodisationSuggestion(
        window=window_key,
        time_constant_us=_matched_time_constant(window_key, fwhm),
        line_frequency_mhz=float(f_win[peak_index]),
        line_fwhm_mhz=fwhm,
    )


def _matched_scan_kernel(window_key: str, kernel_fwhm: float, bin_width: float) -> np.ndarray:
    """Normalised, discretely-sampled smoothing kernel of the given FWHM."""
    half = kernel_fwhm / 2.0
    x = np.arange(
        -_MATCHED_SCAN_KERNEL_HALF_SPAN * half,
        _MATCHED_SCAN_KERNEL_HALF_SPAN * half + bin_width,
        bin_width,
    )
    if window_key == "lorentzian":
        kernel = (half**2) / (x**2 + half**2)
    else:
        sigma = kernel_fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel_sum = float(kernel.sum())
    if kernel_sum <= 0.0:
        return np.array([])
    return kernel / kernel_sum


def _matched_scan_best(
    f_win: np.ndarray,
    v_win: np.ndarray,
    window_key: str,
    bin_width: float,
    resolution: float,
) -> tuple[float, float, np.ndarray, int, float] | None:
    """Scan candidate kernel widths, returning the highest-SNR candidate.

    *resolution* is the spectrum's intrinsic resolution (MHz) — see the
    module docstring's "anchored to the spectrum's INTRINSIC resolution"
    paragraph. The scan's low bound is anchored to it (not just the padded
    grid's bin width) so that on a heavily zero-padded spectrum the kernel is
    always wide enough to span independent noise.

    Returns ``(snr, kernel_fwhm, smoothed, peak_index, floor_median)`` for
    the best-scoring width, or ``None`` when the search window is too narrow
    to scan (or every width fails to yield a usable robust floor).
    """
    span = float(f_win[-1] - f_win[0])
    low = _MATCHED_SCAN_MIN_KERNEL_BINS * max(resolution, bin_width)
    high = min(_MATCHED_SCAN_MAX_KERNEL_MHZ, span / _MATCHED_SCAN_MAX_KERNEL_SPAN_FRACTION)
    if not (low > 0.0 and high > low):
        return None

    best: tuple[float, float, np.ndarray, int, float] | None = None
    for kernel_fwhm in np.geomspace(low, high, _MATCHED_SCAN_KERNEL_COUNT):
        kernel_fwhm = float(kernel_fwhm)
        kernel = _matched_scan_kernel(window_key, kernel_fwhm, bin_width)
        if kernel.size < 3:
            continue
        smoothed = np.convolve(v_win, kernel, mode="same")
        peak_index = int(np.argmax(smoothed))

        exclude_bins = int(math.ceil(_MATCHED_SCAN_EXCLUSION_KERNELS * kernel_fwhm / bin_width))
        lo = max(0, peak_index - exclude_bins)
        hi = min(smoothed.size, peak_index + exclude_bins + 1)
        floor_mask = np.ones(smoothed.size, dtype=bool)
        floor_mask[lo:hi] = False
        if np.count_nonzero(floor_mask) < 8:
            continue
        floor_values = smoothed[floor_mask]
        median = float(np.median(floor_values))
        mad = float(np.median(np.abs(floor_values - median))) * 1.4826
        if mad <= 0.0:
            continue
        snr = (float(smoothed[peak_index]) - median) / mad
        if best is None or snr > best[0]:
            best = (snr, kernel_fwhm, smoothed, peak_index, median)
    return best


def _matched_scan_fallback(
    f_win: np.ndarray,
    v_win: np.ndarray,
    window_key: str,
    bin_width: float,
    resolution: float,
) -> ApodisationSuggestion | None:
    """Matched-filter detection: smooth the spectrum at candidate linewidths.

    See the module docstring's "Detection is two-stage" paragraph. Called by
    :func:`suggest_matched_apodisation` only after :func:`_prominence_line`
    fails. Scans a family of kernel widths, keeps the one with the highest
    robust SNR, and — if it clears ``_MATCHED_SCAN_SNR_THRESHOLD`` —
    deconvolves the kernel's own width from the measured FWHM before
    deriving a matched time constant from it. *resolution* anchors the
    scan's low bound (see :func:`_matched_scan_best`).
    """
    best = _matched_scan_best(f_win, v_win, window_key, bin_width, resolution)
    if best is None or best[0] < _MATCHED_SCAN_SNR_THRESHOLD:
        return None
    _snr, kernel_fwhm, smoothed, peak_index, median = best

    half_level = median + 0.5 * (float(smoothed[peak_index]) - median)
    left = _half_maximum_crossing(f_win, smoothed, peak_index, half_level, -1)
    right = _half_maximum_crossing(f_win, smoothed, peak_index, half_level, +1)
    if left is None or right is None:
        return None
    fwhm_observed = float(right - left)
    if not np.isfinite(fwhm_observed):
        return None

    # Smoothing broadens whatever it detects — remove the kernel's own
    # contribution before treating the width as physical (module docstring).
    if window_key == "lorentzian":
        fwhm_line = fwhm_observed - kernel_fwhm
    else:
        fwhm_line = math.sqrt(max(fwhm_observed**2 - kernel_fwhm**2, 0.0))
    if fwhm_line < _MIN_FWHM_BINS * bin_width:
        return None

    return ApodisationSuggestion(
        window=window_key,
        time_constant_us=_matched_time_constant(window_key, fwhm_line),
        line_frequency_mhz=float(f_win[peak_index]),
        line_fwhm_mhz=fwhm_line,
    )


def _estimate_intrinsic_resolution(v_win: np.ndarray, bin_width: float) -> float | None:
    """Fallback intrinsic-resolution estimate from the spectrum's own autocorrelation.

    Used only when the caller cannot supply ``intrinsic_resolution_mhz``
    directly (module docstring). Zero-padding correlates adjacent
    power-spectrum bins over one resolution element, so the lag at which the
    (mean-subtracted) autocorrelation first decays to half its zero-lag value
    is a proxy for the unpadded bin spacing — scaled by
    ``_RESOLUTION_ESTIMATE_SAFETY_FACTOR`` to bias the estimate upward rather
    than risk under-estimating it (see that constant's docstring for why
    over-estimation is the safe failure mode here).

    Returns ``None`` when the window is too short or too flat (zero
    variance) to estimate from at all; the caller declines the fallback scan
    entirely in that case rather than guessing a resolution.
    """
    n = v_win.size
    if n < 32:
        return None
    centered = v_win - float(np.mean(v_win))
    corr = np.correlate(centered, centered, mode="full")
    corr = corr[corr.size // 2 :]
    zero_lag = float(corr[0])
    if zero_lag <= 0.0:
        return None
    half = 0.5 * zero_lag
    below = np.nonzero(corr <= half)[0]
    lag = int(below[0]) if below.size else n - 1
    lag = max(lag, 1)
    return float(_RESOLUTION_ESTIMATE_SAFETY_FACTOR * lag * bin_width)


def suggest_matched_apodisation(
    freqs: np.ndarray,
    magnitude: np.ndarray,
    *,
    window: str = "lorentzian",
    min_frequency_mhz: float | None = None,
    max_frequency_mhz: float | None = None,
    intrinsic_resolution_mhz: float | None = None,
) -> ApodisationSuggestion | None:
    """Suggest the matched apodisation for the dominant line of a spectrum.

    *freqs*/*magnitude* are the UNAPODISED magnitude spectrum (MHz axis); an
    already-filtered spectrum would match the filter, not the sample. The
    optional frequency window restricts the line search (callers narrow it
    around the field-expected region, as phase estimation does).
    *intrinsic_resolution_mhz* is the unpadded transform's bin spacing
    (``1 / (t_max - t_min)``); callers that zero-pad should pass it so the
    matched-scan fallback's kernels are anchored to real resolution elements
    rather than the padded grid (module docstring). When omitted, it is
    estimated from the spectrum itself (:func:`_estimate_intrinsic_resolution`);
    when even that is not possible the fallback scan is skipped. Detection
    is two-stage (module docstring): a cheap raw-prominence fast path, then
    a matched-filter scan for lines buried below it. Returns ``None`` —
    meaning "leave apodisation off" — when neither stage finds a line, when
    the dominant line is resolution-limited, or when its width cannot be
    measured inside the window.

    .. caution::

       This reasons from the **spectrum only** and cannot see a dead
       time-domain envelope. If the spectrum handed to it was itself computed
       under a symmetric taper that already deleted the early-time signal, the
       line it is asked to match may simply not be there, and it will report
       "no clear line" on data that is many-sigma without the taper. It also
       never suggests a taper: the two kinds it returns
       (``"lorentzian"``/``"gaussian"``) are the weight-1-at-``t = 0``
       :func:`~asymmetry.core.fourier.window.apply_fft_filter` filters, which
       are safe for early-time work. The active check for the dead-envelope
       case is :class:`ApodisationEarlySignalWarning`, raised by the FFT
       prepare path; a signal-aware suggester is deliberately not built.
    """
    window_key = str(window).strip().lower()
    if window_key not in {"lorentzian", "gaussian"}:
        raise ValueError(f"Unknown apodisation window {window!r}.")

    f = np.asarray(freqs, dtype=float)
    # Measure on the power spectrum — see the module docstring for why the
    # magnitude's half-width is the wrong observable.
    v = np.square(np.abs(np.asarray(magnitude, dtype=float)))
    finite = np.isfinite(f) & np.isfinite(v)
    f = f[finite]
    v = v[finite]
    if f.size < 8:
        return None
    order = np.argsort(f)
    f = f[order]
    v = v[order]

    f_max = float(np.max(f))
    if f_max <= 0.0:
        return None
    lower = max(float(min_frequency_mhz or 0.0), f_max * _DC_CUT_FRACTION)
    upper = float(max_frequency_mhz) if max_frequency_mhz is not None else f_max
    in_window = (f > lower) & (f <= upper)
    if np.count_nonzero(in_window) < 8:
        return None
    f_win = f[in_window]
    v_win = v[in_window]
    bin_width = float(np.median(np.diff(f_win)))

    suggestion = _prominence_line(f_win, v_win, window_key, bin_width)
    if suggestion is not None:
        return suggestion

    if intrinsic_resolution_mhz is not None and intrinsic_resolution_mhz > 0.0:
        resolution = float(intrinsic_resolution_mhz)
    else:
        estimated = _estimate_intrinsic_resolution(v_win, bin_width)
        if estimated is None:
            return None
        resolution = estimated
    return _matched_scan_fallback(f_win, v_win, window_key, bin_width, resolution)
