"""Matched-apodisation scan for heavily damped oscillation lines.

This Qt-free module detects oscillations whose envelope dies within tens of
nanoseconds to a few microseconds — the regime a windowed FFT structurally
cannot see, because every taper (``hann``/``cosine``/``gaussian``) is zero at
the first sample and so deletes exactly the leading nanoseconds where such a
line lives.  Unlike the early-window crop ladder in
:mod:`~asymmetry.core.fitting.peak_detection`, it needs **no user-chosen time
crop**: the damping itself is the scan variable.

Why a matched apodisation
-------------------------
Multiplying the record by ``exp(-t/τ)`` before an unwindowed transform is a
matched filter for a line whose own envelope is ``exp(-λt)``.  The apodised
line contributes coherently over ``~1/(λ + 1/τ)`` while the noise it competes
with is scaled down by the same weight, so the peak height over the noise floor
behaves as

    SNR(τ) ∝ √τ / (λτ + 1),

which is maximised exactly at ``τ = 1/λ``.  Scanning a geometric ladder of τ
therefore does two jobs at once: it makes a heavily damped line visible at all,
and the rung of maximum SNR *measures* its damping (``λ₀ = 1/τ*``).  The
maximum is broad (half the peak SNR is still reached at ``τ = 1/(6λ)`` and
``τ = 6/λ``), so four rungs per decade is ample sampling and a line is normally
seen on several neighbouring rungs — which is what makes clustering across
rungs meaningful.

Significance
------------
A scan peak is only a candidate.  Acceptance is decided by a **weighted linear**
Δχ² test (:func:`damped_line_delta_chi2`): the record is fitted, with weights
``1/σ``, against a small dictionary of slow monotonic decays ``exp(-λ_k t)``
(plus every already-accepted line) with and without the pair
``e^{-λt}cos(2πft)``, ``e^{-λt}sin(2πft)``.  Because the pair enters linearly,
Δχ² is available in closed form and a local (f, λ) grid can be swept cheaply
(:func:`refine_line`).

For pure Gaussian noise the improvement from two extra linear degrees of
freedom is ``χ²₂``-distributed, so ``P(Δχ² > x) = exp(-x/2)``.  Searching
``N`` independent (f, λ) cells and tolerating an expected ``α`` false lines per
record gives ``N·exp(-x/2) = α``, i.e.

    x = 2·ln(N / α)                      (:func:`look_elsewhere_threshold`)

with ``α = 0.01``, the same false-rate philosophy as
``peak_detection._FALSE_PEAK_RATE``.  ``N`` is counted as the scan actually
searches: summed over rungs, the width of that rung's guarded search band
divided by the Lorentzian FWHM ``1/(πτ)`` of its own matched envelope, times
``_TRIALS_REFINEMENT_FACTOR`` for the continuous refinement each shortlisted
candidate then gets.

Δχ² is a *statistical* gate; it does not know that a shape is unphysical.  A
static Gaussian Kubo-Toyabe minimum, for instance, is a single dip-and-recover
excursion that no sum of monotonic exponentials can reproduce, so a damped
cosine completing about half a cycle within its envelope will always show a
large Δχ².  Such a feature is not an oscillation in any useful sense, so an
accepted line must also complete at least ``_MIN_CYCLES_PER_LIFETIME`` full
cycles within its own 1/e lifetime (``f ≥ λ``, one cycle per lifetime — half
the ``min_cycles = 3`` the scan band already demands at the matched rung, so
genuinely low-Q lines such as ``f = 200 MHz, λ = 100 µs⁻¹`` are kept).

Cost control
------------
Finely binned records are large: 0.1 ns bins over 10 µs is 10⁵ points, and a
padded transform per rung on all of them would dominate the wizard's runtime.
Two reductions apply, both driven by the physics of the rung rather than by a
blanket decimation:

1. each rung is cropped to ``~10τ``, beyond which its own apodisation has
   suppressed the record to ``e^{-10}``;
2. the crop is value-rebinned (:func:`asymmetry.core.transform.rebin.rebin`)
   so no rung exceeds ``sample_budget`` samples.

Short-τ rungs are short, so they keep full resolution; long-τ rungs are
rebinned, which lowers their Nyquist frequency.  **The consequence is
deliberate and worth stating**: a high-frequency line that is only *slowly*
damped is seen on the rungs whose rebinned Nyquist still covers it (short and
mid τ, where it is also matched well enough — the SNR maximum is broad), not
on the longest rungs.  A high-frequency line that is slowly damped is in any
case the easy case the ordinary windowed pass already handles; the rungs that
matter here are the short ones, and those are never rebinned.

Amplitudes are reported on **the scale of the input** — the library convention
is percent asymmetry, so ``amplitude_percent`` is percent when the caller
passes percent — and are corrected for the bin-averaging attenuation
introduced by the rebinning this module performs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.fitting.peak_detection import effective_analysis_window
from asymmetry.core.transform.rebin import rebin

_EPS = 1e-12

# --------------------------------------------------------------------------- #
# τ ladder
# --------------------------------------------------------------------------- #

#: Rungs per decade of τ.  The matched-filter SNR curve ``√τ/(λτ+1)`` is broad
#: (half maximum at τ = 1/(6λ) and 6/λ), so four rungs per decade lose at most
#: a few percent of the peak SNR between rungs while keeping the ladder short.
_TAU_RUNGS_PER_DECADE = 4

#: Shortest rung, in bins: below ~20 samples per lifetime the apodised support
#: is too short for a transform to resolve anything inside its own guard band.
_TAU_MIN_BINS = 20

#: Absolute floor on the shortest rung (1 ns).  A record binned far finer than
#: any physical µSR envelope should not spend rungs on lifetimes no muon signal
#: can have.
_TAU_MIN_US = 1e-3

# --------------------------------------------------------------------------- #
# Scan
# --------------------------------------------------------------------------- #

#: Zero-padding factor for each rung's transform.  Padding does not add
#: information; it interpolates the spectrum finely enough that ``find_peaks``
#: sees a smooth line shape and the peak bin is close to the true frequency.
_SCAN_PAD_FACTOR = 4

#: Guard band: a rung only reports lines completing at least this many cycles
#: within its own matched lifetime.  Below it, a "line" is a single excursion
#: of the envelope — indistinguishable from the relaxation shape itself.
_SCAN_MIN_CYCLES = 3.0

#: Upper edge of the search band as a fraction of the rung's Nyquist frequency.
_SCAN_F_MAX_FRACTION = 0.9

#: A rung needs at least this many in-band bins for its noise floor to be a
#: floor rather than an estimate of a handful of correlated bins.
_SCAN_MIN_BAND_BINS = 16

#: Peak prominence required by ``find_peaks``, in MAD units of the band.
_SCAN_PROMINENCE_MADS = 3.0

#: Peaks kept per rung (strongest first).  A rung that reports more than a few
#: lines is reporting noise structure.
_SCAN_MAX_PEAKS_PER_RUNG = 5

#: Minimum scan SNR for a peak to become a Δχ² candidate.  Acceptance is the
#: look-elsewhere-corrected Δχ², so this is only a shortlist gate — but the
#: shortlist is finite (``_MAX_CANDIDATES``), and a noise peak that takes a slot
#: costs a real line its verification.  Calibrated on the null: over 260
#: synthetic records with no oscillation (pure noise and a relaxing background,
#: both binnings, µSR-like exploding errors) the largest scan SNR anywhere on
#: the ladder was 6.9, with a median of 5.1-5.9 — the ``3·MAD`` prominence
#: requirement already puts every record's best noise peak near 5.  Eight
#: clears that maximum with margin while sitting far below the real lines the
#: study records carry (17-36 for a 2-5 % damped line, 357 for a 20 % TF line).
_SCAN_MIN_SNR = 8.0

#: Lifetimes of record retained per rung before rebinning (see module docstring).
_RUNG_LIFETIMES = 10.0

#: Never shorten a rung below this many samples.
_MIN_RUNG_POINTS = 64

#: Default per-rung sample budget after cropping and rebinning.
_SAMPLE_BUDGET = 16384

# --------------------------------------------------------------------------- #
# Clustering and verification
# --------------------------------------------------------------------------- #

#: Two rung peaks are the same line if they agree to within the wider of the two
#: rungs' FWHM, or this fraction of the frequency (whichever is larger).
_CLUSTER_FRACTIONAL_TOLERANCE = 0.02

#: Slow monotonic decays fitted alongside every candidate.  Spanning
#: 0 (constant) to 10 µs⁻¹ in half-decade-ish steps, this dictionary absorbs the
#: relaxing background — including stretched and multi-component decays, which
#: are completely monotone and therefore exponential mixtures — so Δχ² measures
#: the *oscillatory* content the background cannot explain.
_SLOW_DECAY_RATES_PER_US = (0.0, 0.3, 1.0, 3.0, 10.0)

#: Expected number of false lines tolerated per record.
_FALSE_LINE_RATE = 0.01

#: The rung trial count measures the cells the *scan* resolves, but every
#: shortlisted candidate is afterwards maximised continuously over a ±3 %
#: frequency neighbourhood and a wide λ span (:func:`refine_line`) — a further
#: look-elsewhere the cell count does not contain.  Inflating the trials by
#: three raises the gate by ``2·ln 3 ≈ 2.2`` and keeps the realised false rate
#: below the nominal one: over 200 pure-noise records (8 000 points, µSR-like
#: exploding errors) the largest refined Δχ² was 28.2 against a gate that this
#: factor puts at 30.3.
_TRIALS_REFINEMENT_FACTOR = 3.0

#: Clustered candidates carried into the Δχ² stage, strongest scan SNR first.
_MAX_CANDIDATES = 6

#: Lifetimes of record used to verify a candidate: past 12/λ the line has
#: decayed to ``e^{-12}`` and the remaining points only add nuisance freedom.
_VERIFY_LIFETIMES = 12.0

#: Minimum samples per cycle kept when rebinning the verification record.  Four
#: is comfortably above Nyquist; the residual bin-averaging attenuation
#: (Dirichlet kernel, ~0.90 at this rate) is corrected analytically on the
#: reported amplitude.
_VERIFY_OVERSAMPLE = 4.0

#: Never verify on fewer than this many samples.
_MIN_VERIFY_POINTS = 512

#: Accepted lines must complete at least this many cycles within their own 1/e
#: envelope lifetime (see module docstring, "Significance").
_MIN_CYCLES_PER_LIFETIME = 1.0

#: Singular values below this fraction of the largest are dropped when
#: orthonormalising the nuisance dictionary — the slow decays become nearly
#: collinear on a short verification window, and a rank-deficient basis must
#: not turn into numerical noise in the Δχ².
_RANK_TOLERANCE = 1e-10

#: Refinement rounds as ``(fractional frequency half-width, λ range factor)``.
#: The first round searches λ over a 16× span because the seed ``1/τ*`` is only
#: as good as the rung spacing, and a strongly damped line is often first found
#: on a rung longer than its own lifetime (where the guard band is loose enough
#: to admit it).
_REFINE_ROUNDS = ((0.03, 4.0), (0.012, 2.0), (0.005, 1.25))

#: Grid points per axis and per round.
_REFINE_GRID = 5

#: Minimum record length worth scanning at all.
_MIN_RECORD_POINTS = 64


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ScanPeak:
    """One local maximum in a single rung's apodised spectrum.

    ``snr`` is the magnitude's excess over the running noise floor of that
    rung's guarded band, in MAD units of that excess; it is measured against
    that rung's own noise floor and is therefore comparable
    *between rungs of the same scan* (all rungs use the same estimator on the
    same record) but not with a :class:`~asymmetry.core.fitting.peak_detection.DetectedPeak`
    SNR, which is a magnitude ratio rather than a MAD excess.
    """

    frequency_mhz: float
    snr: float
    magnitude: float


@dataclass(frozen=True)
class ScanRung:
    """The outcome of one matched-apodisation rung.

    Attributes
    ----------
    tau_us
        Apodisation lifetime of this rung, in µs.
    bin_width_us, rebin_factor, n_samples
        The record this rung actually transformed: ``n_samples`` points of
        width ``bin_width_us`` after cropping to ``~10τ`` and rebinning by
        ``rebin_factor`` (see module docstring, "Cost control").
    resolution_mhz
        Padded frequency-bin spacing, ``1/(n_fft·bin_width_us)``.
    fwhm_mhz
        Lorentzian FWHM ``1/(πτ)`` of a line seen through this apodisation —
        the peak-separation distance and the trial-counting cell width.
    band_lo_mhz, band_hi_mhz
        Guarded search band.  Empty (``band_hi_mhz <= band_lo_mhz``) when the
        rung has no usable band, in which case it contributes no trials.
    noise_floor, noise_scale
        Median of the running in-band noise floor, and the MAD-derived σ of the
        magnitudes' excess over it.
    peaks
        Up to ``max_peaks_per_rung`` peaks, strongest SNR first.
    """

    tau_us: float
    bin_width_us: float
    rebin_factor: int
    n_samples: int
    resolution_mhz: float
    fwhm_mhz: float
    band_lo_mhz: float
    band_hi_mhz: float
    noise_floor: float
    noise_scale: float
    peaks: tuple[ScanPeak, ...]

    @property
    def n_trials(self) -> float:
        """Independent (f, λ) cells this rung searched: band width / FWHM."""
        width = self.band_hi_mhz - self.band_lo_mhz
        if width <= 0.0 or self.fwhm_mhz <= 0.0:
            return 0.0
        return float(width / self.fwhm_mhz)


@dataclass(frozen=True)
class LineCandidate:
    """A scan peak clustered across rungs, before Δχ² verification.

    ``tau_us`` is the rung of maximum SNR — the matched lifetime, so
    ``λ₀ = 1/tau_us`` seeds the refinement.
    """

    frequency_mhz: float
    tau_us: float
    snr: float
    n_rungs: int


@dataclass(frozen=True)
class DampedLine:
    """One accepted damped oscillation.

    Attributes
    ----------
    frequency_mhz
        Refined line frequency in MHz.
    damping_rate_per_us
        Refined envelope rate λ in µs⁻¹ (``exp(-λt)``, 1/e lifetime ``1/λ``).
    amplitude_percent
        Amplitude ``A`` of ``A·exp(-λt)·cos(2πft + φ)`` referred to the start
        of the analysis window, **on the scale of the input asymmetry** (the
        library convention is percent), corrected for the bin-averaging
        attenuation of this module's own rebinning.
    phase_rad
        Phase φ in radians, in ``(-π, π]``, referred to the same origin.
    delta_chi_squared
        Weighted χ² improvement from adding this line to the nuisance basis
        (and to every line accepted before it), measured on the verification
        record described in the module docstring.
    tau_us
        Matched apodisation lifetime of the rung that found the line.
    snr
        That rung's scan SNR (MAD units), for provenance and ranking within the
        scan; the acceptance decision is ``delta_chi_squared``.
    """

    frequency_mhz: float
    damping_rate_per_us: float
    amplitude_percent: float
    phase_rad: float
    delta_chi_squared: float
    tau_us: float
    snr: float


@dataclass(frozen=True)
class DampedLineAnalysis:
    """The outcome of a damped-line scan over one record.

    ``lines`` is ordered by ``delta_chi_squared`` descending.  Lines are
    verified sequentially (peeling), so a later line's Δχ² is measured with the
    earlier ones already in the basis.
    """

    lines: tuple[DampedLine, ...]
    threshold_delta_chi_squared: float
    n_trials: float
    taus_us: tuple[float, ...]
    window_end_index: int
    sample_budget: int
    false_rate: float


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #


def serialize_damped_line(line: DampedLine) -> dict[str, object]:
    """Return a JSON-safe dict snapshot of a :class:`DampedLine`."""
    return {
        "frequency_mhz": float(line.frequency_mhz),
        "damping_rate_per_us": float(line.damping_rate_per_us),
        "amplitude_percent": float(line.amplitude_percent),
        "phase_rad": float(line.phase_rad),
        "delta_chi_squared": float(line.delta_chi_squared),
        "tau_us": float(line.tau_us),
        "snr": float(line.snr),
    }


def deserialize_damped_line(payload: object) -> DampedLine | None:
    """Rebuild a :class:`DampedLine` from a persisted dict, tolerating gaps."""
    if not isinstance(payload, dict):
        return None
    return DampedLine(
        frequency_mhz=float(payload.get("frequency_mhz", 0.0)),
        damping_rate_per_us=float(payload.get("damping_rate_per_us", 0.0)),
        amplitude_percent=float(payload.get("amplitude_percent", 0.0)),
        phase_rad=float(payload.get("phase_rad", 0.0)),
        delta_chi_squared=float(payload.get("delta_chi_squared", 0.0)),
        tau_us=float(payload.get("tau_us", 0.0)),
        snr=float(payload.get("snr", 0.0)),
    )


def serialize_damped_line_analysis(analysis: DampedLineAnalysis) -> dict[str, object]:
    """Return a JSON-safe dict snapshot of a :class:`DampedLineAnalysis`."""
    return {
        "lines": [serialize_damped_line(line) for line in analysis.lines],
        "threshold_delta_chi_squared": float(analysis.threshold_delta_chi_squared),
        "n_trials": float(analysis.n_trials),
        "taus_us": [float(tau) for tau in analysis.taus_us],
        "window_end_index": int(analysis.window_end_index),
        "sample_budget": int(analysis.sample_budget),
        "false_rate": float(analysis.false_rate),
    }


def deserialize_damped_line_analysis(payload: object) -> DampedLineAnalysis | None:
    """Rebuild a :class:`DampedLineAnalysis` from a persisted dict."""
    if not isinstance(payload, dict):
        return None
    lines = tuple(
        line
        for entry in payload.get("lines", [])
        if (line := deserialize_damped_line(entry)) is not None
    )
    taus = payload.get("taus_us", [])
    return DampedLineAnalysis(
        lines=lines,
        threshold_delta_chi_squared=float(payload.get("threshold_delta_chi_squared", 0.0)),
        n_trials=float(payload.get("n_trials", 0.0)),
        taus_us=tuple(float(tau) for tau in taus) if isinstance(taus, (list, tuple)) else (),
        window_end_index=int(payload.get("window_end_index", 0)),
        sample_budget=int(payload.get("sample_budget", _SAMPLE_BUDGET)),
        false_rate=float(payload.get("false_rate", _FALSE_LINE_RATE)),
    )


# --------------------------------------------------------------------------- #
# Threshold
# --------------------------------------------------------------------------- #


def look_elsewhere_threshold(n_trials: float, false_rate: float = _FALSE_LINE_RATE) -> float:
    """Δχ² an accepted line must clear, corrected for the size of the search.

    Adding a damped cosine to a linear basis costs two degrees of freedom, so
    under the null ``P(Δχ² > x) = exp(-x/2)``.  Searching ``n_trials``
    independent cells and tolerating ``false_rate`` false lines per record
    gives ``n_trials·exp(-x/2) = false_rate``, i.e. ``x = 2·ln(n_trials /
    false_rate)``.  Monotonically increasing in ``n_trials`` and in
    ``1/false_rate``.

    A single trial at the default rate already demands Δχ² ≈ 9.2; the floor of
    one trial keeps the threshold defined for a degenerate scan.
    """
    trials = max(1.0, float(n_trials))
    rate = float(false_rate)
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("false_rate must be a positive fraction")
    return float(2.0 * np.log(trials / rate))


# --------------------------------------------------------------------------- #
# τ ladder
# --------------------------------------------------------------------------- #


def tau_ladder(
    bin_width_us: float,
    duration_us: float,
    *,
    per_decade: int = _TAU_RUNGS_PER_DECADE,
) -> NDArray[np.float64]:
    """Geometric ladder of matched apodisation lifetimes, in µs.

    Spans ``max(20·dt, 1 ns)`` to half the informative duration: shorter than
    twenty bins there is nothing inside the guard band to resolve, and longer
    than half the record the apodisation stops being a filter at all.  Roughly
    ``per_decade`` rungs per decade (see ``_TAU_RUNGS_PER_DECADE`` for why four
    is enough).  Degenerate inputs return a single rung.
    """
    dt = float(bin_width_us)
    duration = float(duration_us)
    if not np.isfinite(dt) or dt <= 0.0 or not np.isfinite(duration) or duration <= 0.0:
        return np.asarray([max(dt, _TAU_MIN_US)], dtype=np.float64)
    low = max(_TAU_MIN_BINS * dt, _TAU_MIN_US)
    high = 0.5 * duration
    if high <= low:
        return np.asarray([low], dtype=np.float64)
    n_rungs = int(np.ceil(np.log10(high / low) * max(1, int(per_decade)))) + 1
    return np.geomspace(low, high, max(2, n_rungs)).astype(np.float64)


# --------------------------------------------------------------------------- #
# Scan
# --------------------------------------------------------------------------- #


def _bin_width(time: NDArray[np.float64]) -> float:
    """Mean sample spacing of a uniformly binned record."""
    if time.size < 2:
        return 0.0
    return float((time[-1] - time[0]) / (time.size - 1))


def _mad_scale(values: NDArray[np.float64]) -> float:
    """MAD-derived Gaussian σ of ``values`` about their own median."""
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def _band_noise_floor(magnitude: NDArray[np.float64], bins_per_fwhm: int) -> NDArray[np.float64]:
    """Running-median noise floor across one rung's band.

    A single number would not do here.  The relaxing background the record
    always carries transforms into a broad low-frequency skirt that stands far
    above the white part of the band, and its truncation ripples are genuine
    local maxima: against a flat median floor they score SNR in the tens and
    crowd real lines out of the shortlist.  A floor that follows the skirt
    removes them without touching a narrow line.

    The window spans sixteen line widths (``bins_per_fwhm`` accounts for the
    zero-padding oversampling), so the Lorentzian a rung is matched to sits at
    well under a percent of its peak over most of its own window and cannot
    suppress its own SNR — the failure mode that made
    ``peak_detection._global_noise_floor`` prefer a flat floor on its far
    narrower early-window crops.  One sigma-clip pass then keeps strong lines
    out of the refined median.
    """
    from scipy.ndimage import median_filter

    values = np.asarray(magnitude, dtype=np.float64)
    if values.size == 0:
        return values.copy()
    window = max(9, 16 * max(1, int(bins_per_fwhm)) + 1)
    window = min(window, values.size if values.size % 2 == 1 else values.size - 1)
    if window <= 1:
        return np.full_like(values, float(np.median(values)))
    floor = median_filter(values, size=window, mode="nearest")
    residual = values - floor
    scale = _mad_scale(residual)
    if scale > _EPS:
        clipped = np.where(residual > 3.0 * scale, floor, values)
        floor = median_filter(clipped, size=window, mode="nearest")
    return floor


def _rung_record(
    time: NDArray[np.float64],
    asymmetry: NDArray[np.float64],
    error: NDArray[np.float64],
    tau_us: float,
    sample_budget: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], int]:
    """Crop to ``~10τ`` and rebin to ``sample_budget``; return the rung record."""
    dt = _bin_width(time)
    n = time.size
    if dt <= 0.0:
        return time, asymmetry, error, 1
    span = int(np.ceil(_RUNG_LIFETIMES * tau_us / dt))
    crop = int(min(n, max(span, _MIN_RUNG_POINTS)))
    factor = max(1, int(np.ceil(crop / max(1, int(sample_budget)))))
    if factor == 1:
        return time[:crop], asymmetry[:crop], error[:crop], 1
    t_r, y_r, e_r = rebin(time[:crop], asymmetry[:crop], error[:crop], factor)
    return t_r, y_r, e_r, factor


def matched_apodisation_scan(
    time: NDArray[np.float64],
    asymmetry: NDArray[np.float64],
    error: NDArray[np.float64],
    taus_us: NDArray[np.float64],
    *,
    sample_budget: int = _SAMPLE_BUDGET,
    pad_factor: int = _SCAN_PAD_FACTOR,
    min_cycles: float = _SCAN_MIN_CYCLES,
    f_max_fraction: float = _SCAN_F_MAX_FRACTION,
    max_peaks_per_rung: int = _SCAN_MAX_PEAKS_PER_RUNG,
) -> tuple[ScanRung, ...]:
    """Transform ``(y − tail)·exp(-t/τ)`` unwindowed, once per rung.

    The tail estimate (mean of the rung's last ~20 %) removes the DC pedestal
    that would otherwise leak across the low end of the band.  No taper is
    applied beyond the matched exponential: a taper is exactly what deletes the
    early nanoseconds this scan exists to see.

    Each rung reports its guarded band, its running noise floor over that band
    (:func:`_band_noise_floor`), and up to ``max_peaks_per_rung`` peaks in the
    excess over that floor, separated by at least one FWHM and standing
    ``_SCAN_PROMINENCE_MADS`` MADs above their surroundings.

    ``error`` is not used to weight the transform — it enters through the
    rebinning (which must combine errors in quadrature) and through the Δχ²
    stage that judges the peaks this returns.
    """
    from scipy.signal import find_peaks

    t = np.asarray(time, dtype=np.float64)
    y = np.asarray(asymmetry, dtype=np.float64)
    e = np.asarray(error, dtype=np.float64)
    rungs: list[ScanRung] = []
    pad = max(1, int(pad_factor))

    for tau in np.asarray(taus_us, dtype=np.float64):
        tau_us = float(tau)
        if not np.isfinite(tau_us) or tau_us <= 0.0:
            continue
        t_r, y_r, _e_r, factor = _rung_record(t, y, e, tau_us, sample_budget)
        n_r = t_r.size
        dt_r = _bin_width(t_r)
        if n_r < _MIN_RUNG_POINTS or dt_r <= 0.0:
            continue

        late = min(n_r, max(5, n_r // 5))
        centred = y_r - float(np.mean(y_r[-late:]))
        signal = centred * np.exp(-(t_r - t_r[0]) / tau_us)

        n_fft = 1 << int(np.ceil(np.log2(max(2, n_r * pad))))
        magnitude = np.abs(np.fft.rfft(signal, n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, d=dt_r)
        df = float(freqs[1])
        fwhm = 1.0 / (np.pi * tau_us)
        band_lo = max(min_cycles / tau_us, 3.0 * df)
        band_hi = float(f_max_fraction) * 0.5 / dt_r
        in_band = (freqs > band_lo) & (freqs < band_hi)

        if int(np.count_nonzero(in_band)) < _SCAN_MIN_BAND_BINS:
            rungs.append(
                ScanRung(
                    tau_us=tau_us,
                    bin_width_us=dt_r,
                    rebin_factor=factor,
                    n_samples=n_r,
                    resolution_mhz=df,
                    fwhm_mhz=fwhm,
                    band_lo_mhz=band_lo,
                    band_hi_mhz=band_lo,
                    noise_floor=0.0,
                    noise_scale=0.0,
                    peaks=(),
                )
            )
            continue

        band_freqs = freqs[in_band]
        band_mag = magnitude[in_band]
        distance = max(1, int(fwhm / df))
        floor_curve = _band_noise_floor(band_mag, distance)
        excess = band_mag - floor_curve
        scale = max(_mad_scale(excess), _EPS)
        indices, _properties = find_peaks(
            excess, distance=distance, prominence=_SCAN_PROMINENCE_MADS * scale
        )
        snr = excess[indices] / scale if indices.size else np.empty(0)
        order = np.argsort(-snr)[: max(0, int(max_peaks_per_rung))]
        peaks = tuple(
            ScanPeak(
                frequency_mhz=float(band_freqs[indices[k]]),
                snr=float(snr[k]),
                magnitude=float(band_mag[indices[k]]),
            )
            for k in order
        )
        rungs.append(
            ScanRung(
                tau_us=tau_us,
                bin_width_us=dt_r,
                rebin_factor=factor,
                n_samples=n_r,
                resolution_mhz=df,
                fwhm_mhz=fwhm,
                band_lo_mhz=band_lo,
                band_hi_mhz=float(band_freqs[-1]),
                noise_floor=float(np.median(floor_curve)),
                noise_scale=scale,
                peaks=peaks,
            )
        )
    return tuple(rungs)


def cluster_scan_peaks(
    rungs: tuple[ScanRung, ...],
    *,
    min_snr: float = _SCAN_MIN_SNR,
    fractional_tolerance: float = _CLUSTER_FRACTIONAL_TOLERANCE,
) -> tuple[LineCandidate, ...]:
    """Group rung peaks into candidate lines, strongest scan SNR first.

    A real line appears on several neighbouring rungs (the matched-filter SNR
    maximum is broad), at frequencies agreeing to within the wider of the two
    rungs' FWHM — or ``fractional_tolerance`` of the frequency, which takes
    over when both rungs are long and their FWHM is far below the accuracy of
    a peak sitting on a noisy spectrum.  The rung of maximum SNR is kept as the
    cluster's matched lifetime ``τ*``.
    """
    clusters: list[dict[str, float]] = []
    for rung in rungs:
        for peak in rung.peaks:
            if peak.snr < float(min_snr):
                continue
            for cluster in clusters:
                tolerance = max(
                    rung.fwhm_mhz,
                    1.0 / (np.pi * cluster["tau_us"]),
                    float(fractional_tolerance) * peak.frequency_mhz,
                )
                if abs(cluster["frequency_mhz"] - peak.frequency_mhz) < tolerance:
                    cluster["n_rungs"] += 1
                    if peak.snr > cluster["snr"]:
                        cluster["frequency_mhz"] = peak.frequency_mhz
                        cluster["tau_us"] = rung.tau_us
                        cluster["snr"] = peak.snr
                    break
            else:
                clusters.append(
                    {
                        "frequency_mhz": peak.frequency_mhz,
                        "tau_us": rung.tau_us,
                        "snr": peak.snr,
                        "n_rungs": 1,
                    }
                )
    clusters.sort(key=lambda cluster: -cluster["snr"])
    return tuple(
        LineCandidate(
            frequency_mhz=float(cluster["frequency_mhz"]),
            tau_us=float(cluster["tau_us"]),
            snr=float(cluster["snr"]),
            n_rungs=int(cluster["n_rungs"]),
        )
        for cluster in clusters
    )


# --------------------------------------------------------------------------- #
# Weighted linear Δχ²
# --------------------------------------------------------------------------- #


def _weights(error: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return ``1/σ`` with non-finite or non-positive σ replaced by the median."""
    err = np.asarray(error, dtype=np.float64)
    usable = np.isfinite(err) & (err > 0.0)
    if not np.any(usable):
        return np.ones(err.size, dtype=np.float64)
    if not np.all(usable):
        err = np.where(usable, err, float(np.median(err[usable])))
    return 1.0 / err


def _line_columns(
    elapsed: NDArray[np.float64], frequency_mhz: float, damping_rate_per_us: float
) -> NDArray[np.float64]:
    """``[e^{-λt}cos(2πft), e^{-λt}sin(2πft)]`` as an ``(n, 2)`` array."""
    envelope = np.exp(-float(damping_rate_per_us) * elapsed)
    angle = 2.0 * np.pi * float(frequency_mhz) * elapsed
    return np.column_stack((envelope * np.cos(angle), envelope * np.sin(angle)))


class _NuisanceBasis:
    """Orthonormalised nuisance model, cached across a candidate's grid sweep.

    Holds the weighted residual ``r₀`` of the record against the slow-decay
    dictionary (plus any already-accepted lines).  Adding two columns then
    costs one projection instead of a fresh least-squares solve, which is what
    makes a few hundred (f, λ) evaluations per candidate affordable — the
    Frisch-Waugh-Lovell identity: the joint-fit coefficients of the added pair
    are those of regressing ``r₀`` on the pair's residualised columns, and the
    χ² improvement is the squared length of that projection.
    """

    def __init__(
        self,
        elapsed: NDArray[np.float64],
        asymmetry: NDArray[np.float64],
        weights: NDArray[np.float64],
        *,
        basis_rates: tuple[float, ...],
        extra_lines: tuple[tuple[float, float], ...],
    ) -> None:
        self._elapsed = elapsed
        self._weights = weights
        columns = [np.exp(-float(rate) * elapsed) for rate in basis_rates]
        for frequency_mhz, damping in extra_lines:
            pair = _line_columns(elapsed, frequency_mhz, damping)
            columns.extend((pair[:, 0], pair[:, 1]))
        design = np.column_stack(columns) * weights[:, None]
        left, singular, _ = np.linalg.svd(design, full_matrices=False)
        keep = singular > max(float(singular[0]), _EPS) * _RANK_TOLERANCE
        self._q = left[:, keep]
        weighted = asymmetry * weights
        self.residual = weighted - self._q @ (self._q.T @ weighted)
        self.chi_squared = float(self.residual @ self.residual)

    def evaluate(
        self, frequency_mhz: float, damping_rate_per_us: float
    ) -> tuple[float, float, float]:
        """Return ``(delta_chi_squared, amplitude, phase_rad)`` for one line."""
        pair = (
            _line_columns(self._elapsed, frequency_mhz, damping_rate_per_us)
            * self._weights[:, None]
        )
        residualised = pair - self._q @ (self._q.T @ pair)
        gram = residualised.T @ residualised
        projection = residualised.T @ self.residual
        coefficients, *_ = np.linalg.lstsq(gram, projection, rcond=None)
        delta = float(coefficients @ projection)
        cosine, sine = float(coefficients[0]), float(coefficients[1])
        amplitude = float(np.hypot(cosine, sine))
        phase = float(np.arctan2(-sine, cosine))
        return max(delta, 0.0), amplitude, phase


def damped_line_delta_chi2(
    time: NDArray[np.float64],
    asymmetry: NDArray[np.float64],
    error: NDArray[np.float64],
    frequency_mhz: float,
    damping_rate_per_us: float,
    *,
    basis_rates: tuple[float, ...] = _SLOW_DECAY_RATES_PER_US,
    extra_lines: tuple[tuple[float, float], ...] = (),
) -> tuple[float, float, float]:
    """Weighted χ² improvement from one damped cosine, with its amplitude/phase.

    The comparison is between the nuisance model — the slow-decay dictionary
    ``exp(-λ_k t)`` for ``λ_k`` in ``basis_rates``, plus an ``e^{-λt}cos`` /
    ``e^{-λt}sin`` pair for every ``(frequency_mhz, damping_rate_per_us)`` in
    ``extra_lines`` — and that model with the candidate's own pair added.  Both
    fits are weighted linear least squares with weights ``1/error``, so Δχ² is
    exact rather than the outcome of an iterative fit.

    Envelopes and phases are referred to ``time[0]``.  Returns
    ``(delta_chi_squared, amplitude, phase_rad)``; the amplitude is on the
    scale of ``asymmetry``.
    """
    t = np.asarray(time, dtype=np.float64)
    y = np.asarray(asymmetry, dtype=np.float64)
    elapsed = t - t[0] if t.size else t
    weights = _weights(error)
    basis = _NuisanceBasis(
        elapsed, y, weights, basis_rates=tuple(basis_rates), extra_lines=tuple(extra_lines)
    )
    return basis.evaluate(float(frequency_mhz), float(damping_rate_per_us))


def refine_line(
    time: NDArray[np.float64],
    asymmetry: NDArray[np.float64],
    error: NDArray[np.float64],
    frequency_mhz: float,
    damping_rate_per_us: float,
    *,
    basis_rates: tuple[float, ...] = _SLOW_DECAY_RATES_PER_US,
    extra_lines: tuple[tuple[float, float], ...] = (),
    frequency_span_mhz: float | None = None,
) -> tuple[float, float, float]:
    """Maximise Δχ² over a shrinking local (f, λ) grid.

    Three rounds of a ``5 × 5`` grid, each centred on the previous round's best
    point and narrower than it (``_REFINE_ROUNDS``).  The frequency half-width
    is the larger of a few percent and ``frequency_span_mhz`` (pass the finding
    rung's FWHM: a peak located on a broad, heavily apodised line is only
    accurate to a fraction of that width).  λ is searched geometrically over a
    wide span in the first round because the seed ``1/τ*`` inherits the rung
    spacing and may be several times off.

    The search is deliberately *unconstrained* in λ — a candidate whose Δχ² is
    maximised by an envelope shorter than its own period is telling the caller
    it is a relaxation shape rather than an oscillation, and that verdict is
    only visible if the refinement is allowed to go there (see the
    cycles-per-lifetime test in :func:`detect_damped_lines`).

    Returns ``(delta_chi_squared, frequency_mhz, damping_rate_per_us)``.
    """
    t = np.asarray(time, dtype=np.float64)
    elapsed = t - t[0] if t.size else t
    basis = _NuisanceBasis(
        elapsed,
        np.asarray(asymmetry, dtype=np.float64),
        _weights(error),
        basis_rates=tuple(basis_rates),
        extra_lines=tuple(extra_lines),
    )
    best_f = float(frequency_mhz)
    best_lambda = float(damping_rate_per_us)
    best_delta = basis.evaluate(best_f, best_lambda)[0]
    for fractional_span, rate_factor in _REFINE_ROUNDS:
        half_width = max(fractional_span * best_f, 0.5 * float(frequency_span_mhz or 0.0))
        frequencies = np.linspace(best_f - half_width, best_f + half_width, _REFINE_GRID)
        rates = np.geomspace(best_lambda / rate_factor, best_lambda * rate_factor, _REFINE_GRID)
        for candidate_f in frequencies:
            if candidate_f <= 0.0:
                continue
            for candidate_lambda in rates:
                delta = basis.evaluate(float(candidate_f), float(candidate_lambda))[0]
                if delta > best_delta:
                    best_delta = delta
                    best_f = float(candidate_f)
                    best_lambda = float(candidate_lambda)
    return best_delta, best_f, best_lambda


# --------------------------------------------------------------------------- #
# Verification record
# --------------------------------------------------------------------------- #


def _rebin_attenuation(frequency_mhz: float, bin_width_us: float, factor: int) -> float:
    """Amplitude attenuation a line suffers from rebinning by ``factor``.

    Averaging ``factor`` consecutive samples of ``cos(2πft)`` scales its
    amplitude by the Dirichlet kernel ``sin(πf·factor·dt) / (factor·sin(πf·dt))``
    — the *extra* smoothing this module applies, on top of whatever binning the
    input already carried.  Reported amplitudes divide it out.
    """
    if factor <= 1:
        return 1.0
    inner = np.pi * float(frequency_mhz) * float(bin_width_us)
    if abs(np.sin(inner)) < _EPS:
        return 1.0
    value = np.sin(inner * factor) / (factor * np.sin(inner))
    return float(value) if abs(value) > 1e-3 else 1.0


def _verification_record(
    time: NDArray[np.float64],
    asymmetry: NDArray[np.float64],
    error: NDArray[np.float64],
    *,
    frequency_mhz: float,
    tau_us: float,
    sample_budget: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float]:
    """Crop and rebin the record for one candidate's Δχ² sweep.

    The crop keeps ``_VERIFY_LIFETIMES`` matched lifetimes — past that the line
    is gone and the extra points only feed the nuisance basis.  The rebinning
    factor is the smallest that meets ``sample_budget`` but never coarse enough
    to drop below ``_VERIFY_OVERSAMPLE`` samples per cycle of the candidate; a
    candidate whose frequency forbids the rebinning the budget wants is instead
    verified on the leading ``sample_budget`` samples, which for a line fast
    enough to be in that position is far more record than it needs.

    Returns the record plus the amplitude attenuation to divide out.
    """
    dt = _bin_width(time)
    n = time.size
    if dt <= 0.0 or n == 0:
        return time, asymmetry, error, 1.0
    span = int(np.ceil(_VERIFY_LIFETIMES * tau_us / dt))
    crop = int(min(n, max(span, min(n, _MIN_VERIFY_POINTS))))
    budget = max(1, int(sample_budget))
    factor_budget = max(1, int(np.ceil(crop / budget)))
    if frequency_mhz > 0.0:
        factor_alias = max(1, int(1.0 / (_VERIFY_OVERSAMPLE * frequency_mhz * dt)))
    else:
        factor_alias = factor_budget
    factor = max(1, min(factor_budget, factor_alias))
    if factor == 1:
        t_v, y_v, e_v = time[:crop], asymmetry[:crop], error[:crop]
    else:
        t_v, y_v, e_v = rebin(time[:crop], asymmetry[:crop], error[:crop], factor)
    if t_v.size > budget:
        t_v, y_v, e_v = t_v[:budget], y_v[:budget], e_v[:budget]
    return t_v, y_v, e_v, _rebin_attenuation(frequency_mhz, dt, factor)


# --------------------------------------------------------------------------- #
# Dataset-level entry point
# --------------------------------------------------------------------------- #


def detect_damped_lines(
    time: NDArray[np.float64],
    asymmetry: NDArray[np.float64],
    error: NDArray[np.float64],
    *,
    max_lines: int = 3,
    false_rate: float = _FALSE_LINE_RATE,
    sample_budget: int = _SAMPLE_BUDGET,
    min_snr: float = _SCAN_MIN_SNR,
    max_candidates: int = _MAX_CANDIDATES,
) -> DampedLineAnalysis:
    """Find heavily damped oscillations in an asymmetry record.

    The record is truncated to its statistically informative window
    (:func:`~asymmetry.core.fitting.peak_detection.effective_analysis_window`),
    scanned over a matched apodisation ladder, and each clustered candidate is
    verified by a look-elsewhere-corrected Δχ² test with peeling: an accepted
    line joins the nuisance basis before the next candidate is judged, so a
    strong line's leakage cannot manufacture a second one.  No time crop is
    asked of the caller — the ladder replaces it.

    ``asymmetry`` and ``error`` may be on either asymmetry scale; every
    amplitude in the result is on the scale supplied (percent, by the library's
    convention).  Lines are returned Δχ²-descending.
    """
    t_full = np.asarray(time, dtype=np.float64)
    y_full = np.asarray(asymmetry, dtype=np.float64)
    e_full = np.asarray(error, dtype=np.float64)
    n_full = t_full.size
    empty = DampedLineAnalysis(
        lines=(),
        threshold_delta_chi_squared=look_elsewhere_threshold(1.0, false_rate),
        n_trials=0.0,
        taus_us=(),
        window_end_index=n_full,
        sample_budget=int(sample_budget),
        false_rate=float(false_rate),
    )
    if n_full < _MIN_RECORD_POINTS or y_full.size != n_full or e_full.size != n_full:
        return empty

    end = int(effective_analysis_window(t_full, e_full))
    t = t_full[:end]
    y = y_full[:end]
    e = e_full[:end]
    dt = _bin_width(t)
    duration = float(t[-1] - t[0]) if t.size else 0.0
    if dt <= 0.0 or duration <= 0.0:
        return empty

    taus = tau_ladder(dt, duration)
    rungs = matched_apodisation_scan(t, y, e, taus, sample_budget=sample_budget)
    n_trials = _TRIALS_REFINEMENT_FACTOR * float(sum(rung.n_trials for rung in rungs))
    threshold = look_elsewhere_threshold(n_trials, false_rate)
    candidates = cluster_scan_peaks(rungs, min_snr=min_snr)[: max(0, int(max_candidates))]
    fwhm_by_tau = {rung.tau_us: rung.fwhm_mhz for rung in rungs}

    accepted: list[DampedLine] = []
    peeled: list[tuple[float, float]] = []
    for candidate in candidates:
        if len(accepted) >= int(max_lines):
            break
        t_v, y_v, e_v, attenuation = _verification_record(
            t,
            y,
            e,
            frequency_mhz=candidate.frequency_mhz,
            tau_us=candidate.tau_us,
            sample_budget=sample_budget,
        )
        if t_v.size < _MIN_RECORD_POINTS:
            continue
        extra = tuple(peeled)
        delta, frequency, damping = refine_line(
            t_v,
            y_v,
            e_v,
            candidate.frequency_mhz,
            1.0 / candidate.tau_us,
            extra_lines=extra,
            frequency_span_mhz=fwhm_by_tau.get(candidate.tau_us),
        )
        if delta < threshold:
            continue
        if frequency < _MIN_CYCLES_PER_LIFETIME * damping:
            # Fewer than one cycle per envelope lifetime: a single dip-and-
            # recover excursion (a Gaussian Kubo-Toyabe minimum is the textbook
            # case), which the monotonic dictionary cannot fit and Δχ² would
            # therefore accept as a "line".
            continue
        _delta, amplitude, phase = damped_line_delta_chi2(
            t_v, y_v, e_v, frequency, damping, extra_lines=extra
        )
        accepted.append(
            DampedLine(
                frequency_mhz=float(frequency),
                damping_rate_per_us=float(damping),
                amplitude_percent=float(amplitude / attenuation),
                phase_rad=float(phase),
                delta_chi_squared=float(delta),
                tau_us=float(candidate.tau_us),
                snr=float(candidate.snr),
            )
        )
        peeled.append((float(frequency), float(damping)))

    accepted.sort(key=lambda line: -line.delta_chi_squared)
    return DampedLineAnalysis(
        lines=tuple(accepted),
        threshold_delta_chi_squared=threshold,
        n_trials=n_trials,
        taus_us=tuple(float(tau) for tau in taus),
        window_end_index=end,
        sample_budget=int(sample_budget),
        false_rate=float(false_rate),
    )
