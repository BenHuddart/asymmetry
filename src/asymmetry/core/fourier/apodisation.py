"""Data-driven matched-apodisation suggestion, and the early-signal guard.

Two things live here. :func:`suggest_matched_apodisation` computes the matched
filter for a spectrum's dominant line. :func:`early_signal_apodisation_loss` is
the opposite check — it measures how much early-time signal power an
apodisation already in force has *removed*, and backs
:class:`ApodisationEarlySignalWarning`, which the FFT prepare path raises when a
symmetric taper (``hann``/``cosine``, zero at the record's ends) has deleted a
heavily damped oscillation — including when that oscillation rides on a slowly
relaxing tail whose own power would otherwise mask the loss. That failure mode
is concrete: a signal whose lifetime is a small fraction of the record can read
as flat noise under Hann while being many-sigma with no window and a crop, or
under the exponential ("lorentzian") filter with
``filter_time_constant_us ≈ 1/λ``.

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
    window — including the standard case where the damped oscillation rides on a
    slowly relaxing tail that hides it from a whole-signal power statistic (see
    :func:`early_signal_apodisation_loss`). Advisory only: no returned value
    changes.
    """


#: Leading share of the record treated as "early" by
#: :func:`early_signal_apodisation_loss`. A symmetric taper (hann/cosine) is
#: exactly zero at the first sample and climbs as ``(t/T)²``, so its damage is
#: concentrated in the first tenth or so of the record; 0.15 is wide enough
#: that a handful of bins cannot dominate the statistic and narrow enough that
#: an undamped signal puts only 15 % of its power inside it.
_EARLY_WINDOW_FRACTION = 0.15

#: Trigger condition 1: this much of the error-weighted power must live in the
#: early window, on either pass. With flat errors a signal alive across the
#: whole record puts exactly ``_EARLY_WINDOW_FRACTION`` (0.15) of its power
#: there — but real μSR errors grow with time as the counts decay, and the 1/σ²
#: weighting tilts the statistic early on its own: an *undamped* line with
#: ``σ ∝ e^{t/2τ_μ}`` over a long record already reads 0.43, and 0.64 at a
#: relaxation of 0.2 µs⁻¹. The threshold has to clear that pure-weighting tilt,
#: so it sits at 0.75. The oscillatory pass leaves those benign cases untouched
#: (a long-record line is unaffected by the slow-baseline subtraction, reading
#: within 0.001 of its signal-pass value), so the same threshold serves both
#: passes; what the second pass changes is the *damped* side, where a
#: tail-diluted case moves from ~0.4 to ~0.99.
_EARLY_POWER_CONCENTRATION_TRIGGER = 0.75

#: Trigger condition 2: the window must keep no more than this share of that
#: early power. ``window="none"`` keeps 1.0. The *matched* exponential filter
#: (``window="lorentzian"``, ``filter_time_constant_us = 1/λ``) keeps exactly
#: 0.5 in the long-record limit — weight ``e^{-λt}`` against power ``e^{-2λt}``
#: gives ``∫e^{-4λt}/∫e^{-2λt} → 1/2`` — so the threshold has to sit well below
#: 0.5, or the very cure the warning recommends would trip it. A Hann window on
#: the same signal keeps ~1e-3. 0.2 splits those with more than a factor of two
#: of margin on each side; on the oscillatory pass the matched filter reads
#: higher still (~0.54), so that pass has more margin, not less. Both thresholds
#: are pinned by the synthetic study in
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

#: How many standard deviations of the null distribution the measured excess
#: power must clear before the guard is willing to decide at all.
#:
#: Under pure noise each bin contributes ``|n/σ|² − 1``, mean zero and variance
#: 2, so a record of ``n`` bins has an excess-power null of ``0 ± √(2n)``. The
#: gate is that scale times this factor; below it the guard reports
#: ``"insufficient_statistics"`` rather than a confident verdict, because a
#: concentration ratio computed from a sum consistent with zero is an arbitrary
#: number. Measured on pure noise (200 draws, both passes, n = 750 and 3000) the
#: realised excess sits at ~0.4 √(2n) with a spread of ~0.7 √(2n), so 4 is
#: comfortably clear of the null while costing nothing on a record carrying real
#: signal — where the excess runs orders of magnitude above the gate.
_EXCESS_SIGNIFICANCE_SIGMA = 4.0


@dataclass(frozen=True)
class EarlySignalApodisationLoss:
    """How much early-time signal power the applied apodisation removed."""

    #: Leading share of the record measured (``_EARLY_WINDOW_FRACTION``).
    early_window_fraction: float
    #: Share of the error-weighted power living there, on the deciding pass.
    early_power_fraction: float
    #: Share of *that* power the window keeps (1.0 = untouched, 0.0 = deleted).
    early_retained_fraction: float
    #: Which pass the reported numbers come from: ``"signal"`` for the signal's
    #: own power, ``"oscillatory"`` for the power left after the slow baseline
    #: is removed (see :func:`early_signal_apodisation_loss`).
    measured_on: str
    #: ``"triggered"``, ``"clear"``, or ``"insufficient_statistics"`` — the last
    #: meaning the record's excess power did not clear the significance gate, so
    #: the guard declined to decide rather than deciding on noise. Distinct from
    #: ``"clear"``, which is a confident "this apodisation is fine".
    verdict: str
    #: Total noise-excess power the verdict rests on (the larger of the whole
    #: record's and the early window's), in units of ``|s/σ|²``.
    excess_power: float
    #: The significance gate that excess was compared against,
    #: ``_EXCESS_SIGNIFICANCE_SIGMA · √(2n)``.
    significance_floor: float

    @property
    def triggered(self) -> bool:
        """True when both trigger conditions were met on the reported pass."""
        return self.verdict == "triggered"

    @property
    def assessable(self) -> bool:
        """False when there was too little excess power to judge either way."""
        return self.verdict != "insufficient_statistics"


def _slow_baseline(
    values: np.ndarray, weights: np.ndarray, kernel_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return an error-weighted centred moving mean, and each bin's self-weight.

    The slow-baseline estimate the oscillatory pass subtracts. The kernel is
    ``kernel_points`` wide in *samples*, and each sample is weighted by
    ``weights`` so poorly-measured bins do not drag the baseline — the same
    1/σ² weighting the power statistic itself uses.

    The second return value is ``wᵢ / Σ_{j∈window} wⱼ``, the share of its own
    window each bin carries. It is what the oscillatory pass needs to know how
    much noise power the subtraction removed: for independent noise the
    high-passed residual has variance ``σᵢ²(1 − wᵢ/Wᵢ)``, so that is the noise
    pedestal to subtract on that pass rather than a flat 1.

    Edges are handled by **odd padding about a locally fitted intercept**: the
    record is extended antisymmetrically, ``s[−i] = 2·L(0) − s[i]``, where
    ``L`` is a straight line least-squares fitted to the leading (or trailing)
    kernel. Two things make this the right anchor, and both were measured:

    * Against a *truncated* window, odd padding removes a systematic
      ``slope · k/4`` bias at the first sample. That bias lands squarely inside
      the early window and is indistinguishable from early-time signal, so a
      plain straight line — no curvature at all — produced a residual
      concentrated at the start and tripped the guard. Odd padding reproduces a
      linear continuation exactly, cutting that residual ~150×.
    * Against reflecting about the endpoint *sample* ``s[0]``, the fitted
      intercept averages down that sample's own noise instead of doubling it
      into the whole pad. Reflecting about a single noisy sample inflated the
      high-passed noise pedestal by ~3× (0.056 vs 0.019 per bin) and, worse,
      *amplified* a damped oscillation's power by 2.7× when ``s[0]`` happened to
      sit on a peak. The fitted anchor is neutral on both (0.996 of the
      oscillation's power kept) while keeping the straight-line residual
      identical.

    Computed from cumulative sums, so it stays O(n) on records that can reach
    several hundred thousand bins.
    """
    n = values.size
    k = int(np.clip(kernel_points, 1, n))
    pad = int(min(k // 2, n - 1))
    if pad > 0:
        anchor_left = _fitted_edge_intercept(values[:k])
        anchor_right = _fitted_edge_intercept(values[-k:][::-1])
        padded_values = np.concatenate(
            (
                2.0 * anchor_left - values[pad:0:-1],
                values,
                2.0 * anchor_right - values[-2 : -pad - 2 : -1],
            )
        )
        padded_weights = np.concatenate((weights[pad:0:-1], weights, weights[-2 : -pad - 2 : -1]))
        offset = pad
    else:
        padded_values, padded_weights, offset = values, weights, 0

    cum_w = np.concatenate(([0.0], np.cumsum(padded_weights, dtype=np.float64)))
    cum_ws = np.concatenate(([0.0], np.cumsum(padded_weights * padded_values, dtype=np.float64)))
    total = padded_values.size
    starts = np.clip(np.arange(offset, offset + n) - k // 2, 0, total)
    stops = np.clip(starts + k, 0, total)
    denominator = cum_w[stops] - cum_w[starts]
    numerator = cum_ws[stops] - cum_ws[starts]
    safe = denominator > 0.0
    baseline = np.where(safe, numerator / np.where(safe, denominator, 1.0), values)
    self_weight = np.where(safe, weights / np.where(safe, denominator, 1.0), 0.0)
    return baseline, self_weight


def _fitted_edge_intercept(edge: np.ndarray) -> float:
    """Least-squares intercept of a straight line through *edge*.

    ``edge`` runs outward from the record's end, so the intercept is the fitted
    value *at* that end. Averaging the edge kernel this way keeps the odd-padding
    anchor from carrying one sample's full noise (see :func:`_slow_baseline`).
    """
    m = edge.size
    if m < 2:
        return float(edge[0]) if m else 0.0
    x = np.arange(m, dtype=np.float64)
    x_mean = x.mean()
    centred = x - x_mean
    denominator = float(np.dot(centred, centred))
    y_mean = float(edge.mean())
    if denominator <= 0.0:
        return y_mean
    slope = float(np.dot(centred, edge)) / denominator
    return y_mean - slope * x_mean


def _measure_early_loss(
    excess: np.ndarray,
    early_weights: np.ndarray,
    n_early: int,
    measured_on: str,
    *,
    calibrated: bool = True,
) -> EarlySignalApodisationLoss:
    """Run the two-condition test over one profile of noise-excess power.

    *excess* is ``|s/σ|² − pedestal`` per bin: the power the data carries *above
    what noise alone would produce*. Working in excess rather than raw power is
    what makes the statistic binning-invariant — a real signal's excess is
    unchanged by rebinning (σ falls as ``1/√f`` while the bin count falls as
    ``f``) whereas raw power carries a ``+1`` per bin, so the noise pedestal, and
    therefore the dilution, grew with however finely the record happened to be
    binned.

    Individual bins may be negative (a noise dip); that is correct and keeps the
    estimator unbiased. The verdict rests on the larger of the whole record's
    and the early window's excess: a signal confined to the early window can
    leave the record's total *below* the early sum, because the empty late
    portion contributes its own negative fluctuation.
    """
    n = excess.size
    total = float(np.sum(excess, dtype=np.float64))
    early = float(np.sum(excess[:n_early], dtype=np.float64))
    reference = max(total, early)
    # Without calibrated errors there is no noise scale, so no pedestal was
    # subtracted and √(2n) is not in the same units as the power: the gate is
    # meaningless and is stood down rather than applied wrongly.
    floor = _EXCESS_SIGNIFICANCE_SIGMA * np.sqrt(2.0 * n) if calibrated else 0.0

    def _result(verdict: str, concentration: float, retained: float):
        return EarlySignalApodisationLoss(
            early_window_fraction=float(_EARLY_WINDOW_FRACTION),
            early_power_fraction=concentration,
            early_retained_fraction=retained,
            measured_on=measured_on,
            verdict=verdict,
            excess_power=reference,
            significance_floor=float(floor),
        )

    if not np.isfinite(reference) or reference < floor or early <= 0.0:
        return _result("insufficient_statistics", float("nan"), float("nan"))

    concentration = float(np.clip(early / reference, 0.0, 1.0))
    # Individual excess bins can be negative, so the weighted sum can dip just
    # below zero on a record whose early window is almost entirely signal-free
    # after the window. Report it as the fraction it is meant to be.
    retained = float(
        np.clip(
            np.sum(excess[:n_early] * np.square(early_weights), dtype=np.float64) / early,
            0.0,
            1.0,
        )
    )
    triggered = (
        concentration >= _EARLY_POWER_CONCENTRATION_TRIGGER
        and retained <= _EARLY_RETAINED_POWER_TRIGGER
    )
    return _result("triggered" if triggered else "clear", concentration, retained)


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
    record: how much of the power lives there (``early_power_fraction``) and how
    much of that the window keeps (``early_retained_fraction``). Both conditions
    must hold to trigger — a heavily damped oscillation under a symmetric taper,
    which the taper's zero at ``t = 0`` can erase entirely.

    **The test runs twice**, and triggers if either pass fires:

    ``"signal"``
        The signal's own power, as measured.
    ``"oscillatory"``
        The power left after an error-weighted moving mean
        (:func:`_slow_baseline`, kernel ``_EARLY_WINDOW_FRACTION`` of the record)
        is subtracted — a high-pass that removes the slow component and leaves
        the oscillation.

    The second pass exists because a real ordered-state μSR curve is not a bare
    damped cosine: it carries a slowly relaxing tail (the powder 1/3 tail) whose
    amplitude can exceed the oscillation's. A constant or low-order baseline
    removal cannot take out that tail's residual curvature, so its power spreads
    across the whole record and *dilutes* the concentration statistic — the
    signal pass then reads well under the trigger even when the window has
    deleted essentially all of the oscillatory content. Subtracting the slow
    baseline first asks the question the guard means to ask: where does the
    *oscillatory* power live, and how much of it survives?

    Either pass may return ``"insufficient_statistics"`` when its excess power
    does not clear the significance gate — there was not enough signal above the
    noise to judge, and a ratio computed from a sum consistent with zero would be
    meaningless. That is reported distinctly from a confident ``"clear"``.

    The kernel is wide enough to pass any oscillation with more than a handful
    of cycles across the record. A very slow oscillation — fewer than about
    ``1 / _EARLY_WINDOW_FRACTION`` (~7) cycles — is partly absorbed into the
    baseline, which costs sensitivity on that pass only; the signal pass still
    sees it, and a false negative is the safe direction for an advisory warning.

    Returns ``None`` when the measurement is not meaningful: too few samples,
    mismatched shapes, no finite power on either pass, or an apodisation that
    plainly cannot have done damage.
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

    finite = np.isfinite(values)
    clean = np.where(finite, values, 0.0)
    inverse_variance = np.ones(n, dtype=np.float64)
    calibrated = False
    if error is not None:
        sigma = np.asarray(error, dtype=np.float64)
        if sigma.shape == values.shape:
            usable = finite & np.isfinite(sigma) & (sigma > 0.0)
            if np.any(usable):
                inverse_variance = np.where(
                    usable, 1.0 / np.square(np.where(usable, sigma, 1.0)), 0.0
                )
                calibrated = True
    power = np.where(np.isfinite(clean), np.square(clean) * inverse_variance, 0.0)
    # Pure noise contributes exactly 1 per bin to |s/σ|²; subtracting it leaves
    # the power the data carries above the noise expectation. Without calibrated
    # errors there is no such expectation to subtract, and the statistic falls
    # back to raw power — losing the noise robustness and the binning invariance
    # that the subtraction buys, which is the price of not supplying σ.
    signal_excess = power - 1.0 if calibrated else power

    baseline, self_weight = _slow_baseline(clean, inverse_variance, n_early)
    oscillatory = clean - baseline
    oscillatory_power = np.where(
        np.isfinite(oscillatory), np.square(oscillatory) * inverse_variance, 0.0
    )
    # The high-pass removes some of the noise along with the baseline: for
    # independent noise the residual has variance σ²(1 − wᵢ/Wᵢ), so that — not a
    # flat 1 — is this pass's pedestal.
    oscillatory_excess = (
        oscillatory_power - (1.0 - self_weight) if calibrated else oscillatory_power
    )

    passes = [
        _measure_early_loss(signal_excess, early_weights, n_early, "signal", calibrated=calibrated),
        _measure_early_loss(
            oscillatory_excess, early_weights, n_early, "oscillatory", calibrated=calibrated
        ),
    ]
    # Report the pass that fired; failing that, a confident "clear" from either
    # pass; failing that, the honest "there was not enough excess power to say".
    for entry in passes:
        if entry.verdict == "triggered":
            return entry
    for entry in passes:
        if entry.verdict == "clear":
            return entry
    return passes[0]


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
