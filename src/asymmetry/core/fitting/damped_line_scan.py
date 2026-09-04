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

Peeling happens in the scan, not only in the test
-------------------------------------------------
A cluster of lines is its own worst enemy: neighbouring lines lift the local
noise floor of the rung that should show them, and the strongest line's
skirt hides the rest.  So once a line is accepted its fitted model — amplitude
and phase from the weighted linear fit on the whole informative record — is
subtracted from a working copy of the record and **the ladder is scanned
again** on the residual.  The next candidate comes from that rescan, where the
line just removed no longer contributes to either the floor or the peak list.
One rescan per accepted line is cheap (a scan is a few tens of milliseconds on
10⁵ points), and it is what lets the shortlist gate stay low enough
(``_SCAN_MIN_SNR``) for a second, marginal line to be examined at all: the
scan only shortlists, the Δχ² test below decides.

Peeling one line at a time has a bias of its own — the first line is fitted
while its neighbours are still in the residual — so once no further line can be
accepted, each line is re-fitted with all the others held in the nuisance basis
(:func:`_polish_lines`).

Significance
------------
A scan peak is only a candidate.  Acceptance is decided by a **weighted linear**
Δχ² test (:func:`damped_line_delta_chi2`): the record is fitted, with weights
``1/σ``, against a small dictionary of slow monotonic decays ``exp(-λ_k t)``
(plus every already-accepted line) with and without the pair
``e^{-λt}cos(2πft)``, ``e^{-λt}sin(2πft)``.  Because the pair enters linearly,
Δχ² is available in closed form and a local (f, λ) grid can be swept cheaply
(:func:`refine_line`).

Two records are involved, deliberately.  The (f, λ) *search* runs on a short
verification record (``≈12/λ*`` of signal, rebinned) because a hundred grid
evaluations on 10⁵ points would not be affordable; that record is built **once
per candidate from its seed** and held fixed across every refinement round.
Rebuilding it around the current λ — the obvious implementation — makes the
objective itself a function of λ, and since a shorter crop carries fewer
baseline-mismatch points it rewards ever larger λ: the search then walks off to
an overdamped blob whose Δχ² is bigger only because the record got shorter.
The *decision* and every reported number are then taken on the full informative
record with the same nuisance basis, so Δχ² means the same thing for a 40 MHz
line and a 240 MHz one and the values of two accepted lines can be compared.

For pure Gaussian noise the improvement from two extra linear degrees of
freedom is ``χ²₂``-distributed, so ``P(Δχ² > x) = exp(-x/2)``.  Searching
``N`` independent (f, λ) cells and tolerating an expected ``α`` false lines per
record gives ``N·exp(-x/2) = α``, i.e.

    x = 2·ln(N / α)                      (:func:`look_elsewhere_threshold`)

with ``α = 0.01``, the same false-rate philosophy as
``peak_detection._FALSE_PEAK_RATE``.  ``N`` is counted as the scan actually
searches: summed over rungs, the width of that rung's guarded search band
divided by the Lorentzian FWHM ``1/(πτ)`` of its own matched envelope, times
``_TRIALS_REFINEMENT_FACTOR`` for the searching the cell count cannot see —
the continuous refinement of every shortlisted candidate, the length of the
shortlist, and the rescan after each acceptance.

Δχ² is a *statistical* gate; it does not know that a shape is unphysical.  A
static Gaussian Kubo-Toyabe minimum, for instance, is a single dip-and-recover
excursion that no sum of monotonic exponentials can reproduce, so a damped
cosine laid over it shows a Δχ² in the thousands and is accepted every time.
An accepted line must therefore also pass :func:`_is_oscillation`, which asks
either for at least ``_MIN_CYCLES_PER_LIFETIME`` cycles within the fitted
envelope's own lifetime or for an envelope faster than any decay in the
nuisance dictionary.  One clause alone will not do, and the reason is worth
recording: a Kubo-Toyabe dip is fitted at ``f/λ ≈ 1.2``, and so is a real
45 MHz line damped at 40 µs⁻¹.  The two are the same *shape*; what separates
them is scale — the dip lives at the relaxation's own time scale, the line
three hundred times faster than it.

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
passes percent.  They are measured on the full informative record at its own
binning, alongside Δχ² and the phase, so no rebinning attenuation of this
module's making enters them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

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

#: Minimum scan SNR for a peak to be shortlisted for the Δχ² test.  This is a
#: shortlist gate and nothing more: acceptance is the look-elsewhere-corrected
#: Δχ², which is the statistic that actually knows how improbable a line is.
#: A *second* line on a real record is routinely marginal on the scan — the
#: first line's skirt sits under it and the peeling rescan is what uncovers it
#: — so the gate is set where the null's own best noise peaks live (the ``3·MAD``
#: prominence requirement puts a null record's strongest scan peak at 4.5-5.5)
#: rather than above them.  Everything from there upwards is handed to Δχ²,
#: which rejects the noise peaks: over 400 synthetic null records (noise,
#: exponential, stretched exponential and Kubo-Toyabe) not one line is accepted
#: at this gate.
_SCAN_MIN_SNR = 5.0

#: Noise-floor window, in line widths (see :func:`_band_noise_floor`).
_FLOOR_WINDOW_FWHMS = 64

#: Places the running median is evaluated per window; between them the floor is
#: interpolated (see :func:`_running_median`).
_FLOOR_MEDIAN_STRIDES_PER_WINDOW = 4

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

#: The rung trial count measures the cells the *scan* resolves, but the search
#: is larger than that count in three ways it cannot see: every shortlisted
#: candidate is maximised continuously over a local (f, λ) box
#: (:func:`refine_line`), ``_MAX_CANDIDATES`` of them are tried per pass, and
#: the ladder is rescanned after every acceptance.  Inflating the trials by ten
#: raises the gate by ``2·ln 10 ≈ 4.6`` and brings the realised false rate back
#: under the nominal one: over 400 synthetic null records (noise, exponential,
#: stretched exponential and Kubo-Toyabe, 4 000 points, µSR-like exploding
#: errors) no line is accepted, and with the gate disabled the largest Δχ² any
#: candidate surviving :func:`_is_oscillation` claims is 28.7, against a gate
#: this factor puts at 31.2.
_TRIALS_REFINEMENT_FACTOR = 10.0

#: Clustered candidates carried into the Δχ² stage per scan, strongest scan SNR
#: first.  Verification is cheap (a hundred closed-form evaluations on a short
#: record, then one on the full one), and with the gate at ``_SCAN_MIN_SNR`` the
#: shortlist necessarily contains noise peaks, so it must be long enough that a
#: handful of them cannot displace a real line.
_MAX_CANDIDATES = 12

#: A shortlisted candidate is refined only if its *seed* already reaches this
#: fraction of the threshold on the full record.  Refinement searches a box
#: about one line width wide, which cannot manufacture significance from
#: nothing: across the synthetic corpus here the refined Δχ² of an accepted
#: line is at most ~2× its seed value, so a quarter is a wide margin, and it
#: keeps a shortlist full of null peaks from costing a hundred evaluations each.
_SEED_DELTA_CHI2_FRACTION = 0.25

#: Lifetimes of record used to verify a candidate: past 12/λ the line has
#: decayed to ``e^{-12}`` and the remaining points only add nuisance freedom.
_VERIFY_LIFETIMES = 12.0

#: Minimum samples per cycle kept when rebinning the record the (f, λ) grid is
#: swept on.  Eight leaves the bin-averaging attenuation at 0.97 and, more to
#: the point, nearly flat across the refinement box, so it cannot tilt the
#: maximum; nothing measured on that record is reported, so the attenuation
#: itself never reaches the caller.  Rebinning to this rate rather than only as
#: far as ``sample_budget`` demands is what keeps the refinement affordable: it
#: is a 5-10× shorter record for a typical candidate.
_VERIFY_OVERSAMPLE = 8.0

#: Never verify on fewer than this many samples.
_MIN_VERIFY_POINTS = 512

#: Cycles per 1/e envelope lifetime above which a fitted damped cosine counts as
#: an oscillation whatever its time scale — the same three cycles the scan band
#: demands within a rung's lifetime, now asked of the line's own envelope.  See
#: :func:`_is_oscillation`, and note that this clause alone would reject the
#: low-Q lines this module exists to find; the rate clause below is what keeps
#: them.
_MIN_CYCLES_PER_LIFETIME = 3.0

#: ...or the line's envelope is at least this multiple of the fastest decay in
#: the nuisance dictionary, which makes it faster than any relaxation this
#: module models and so not one.  Two is a compromise measured on both sides: a
#: Kubo-Toyabe dip at δ = 0.8 µs⁻¹ is fitted at λ ≈ 0.7, thirty times below the
#: resulting 20 µs⁻¹, while a cluster line whose true λ is 40 is sometimes
#: fitted as low as 25 — and at three, those lines start to be thrown away.
_FAST_LINE_RATE_FACTOR = 2.0

#: Coordinate-ascent sweeps over the accepted lines once the last one is in
#: (:func:`_polish_lines`).  Two is enough for the sweep to stop changing
#: anything on every synthetic record here; the loop exits early when a sweep
#: moves nothing.
_POLISH_SWEEPS = 2

#: Singular values below this fraction of the largest are dropped when
#: orthonormalising the nuisance dictionary — the slow decays become nearly
#: collinear on a short verification window, and a rank-deficient basis must
#: not turn into numerical noise in the Δχ².
_RANK_TOLERANCE = 1e-10

#: Half-width of :func:`refine_line`'s frequency box when the caller does not
#: pass one, in line widths (FWHM ``λ/π``) of the seed.  Callers that know how
#: good their seed is should say so: :func:`detect_damped_lines` passes one
#: rung FWHM, which is far tighter (see :func:`refine_line`).
_REFINE_FREQUENCY_FWHMS = 3.0

#: Two lines must be at least this many widths apart to be two lines.  Closer
#: than that they are one feature fitted twice, and because the pair of columns
#: is then nearly collinear with the pair already in the basis, the least
#: squares answer is two enormous amplitudes that cancel.
_MIN_LINE_SEPARATION_FWHMS = 0.5

#: Half-span of the refinement's λ box, as a factor on the seed ``1/τ*``.  The
#: ladder's four rungs per decade put the matched rung of an isolated line
#: within ~1.8× of its true lifetime (measured: 1.71× on the fast synthetic
#: line, 1.38× on the slow one), so three covers the seeding error with margin.
#:
#: It is also a regulariser, and the more important of its two jobs.  Δχ² does
#: not fall away on the far side of the true λ the way a well-posed likelihood
#: should: a shorter envelope leaves fewer points at which the nuisance model
#: can be wrong, and — decisively, on a cluster — a wider envelope covers a
#: neighbouring line as well as its own, so a single-line fit inside an
#: unresolved cluster is biased towards merging and an unbounded search finds
#: the merged blob every time.  Bounding λ is what stops that, at the price of
#: a bias of its own: a clustered line whose matched rung is biased long (the
#: neighbours lift the shorter rungs' local noise floor) reports λ pinned near
#: the box edge, some 10-15 % below the truth.  Measured on the three-line
#: synthetic, a factor of 3 recovers all three lines on every seed; at 2.5 the
#: reported λ is ~25 % low, and at 4 the merged blob comes back.
_REFINE_DAMPING_FACTOR = 3.0

#: Refinement rounds, as the shrink factor applied to the box each round.  With
#: ``_REFINE_GRID`` points per axis the grid spacing is half the current
#: half-width, so a unimodal maximum is bracketed within a quarter of it and
#: shrinking fourfold per round is exactly self-consistent.  Four rounds take
#: the frequency box to 1/64 of its width — for a 240 MHz line, ~0.6 MHz.
_REFINE_ROUNDS = 4
_REFINE_SHRINK = 4.0

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
        Median of the running in-band noise floor (:func:`_band_noise_floor`),
        and the MAD-derived σ of the magnitudes' excess over it.
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
        Weighted χ² improvement from adding this line to a nuisance basis that
        already contains the slow-decay dictionary **and every other accepted
        line**, measured on the full informative record.  Leave-one-out and on
        one common record, so two lines' values mean the same thing and both
        can be compared with
        :attr:`DampedLineAnalysis.threshold_delta_chi_squared`.
    tau_us
        Matched apodisation lifetime of the rung that found the line.  Note it
        is an estimate of ``1/λ`` only for an isolated line: inside a cluster
        the shorter rungs' local noise floor is lifted by the neighbours, which
        biases the rung of maximum SNR long.
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

    ``lines`` is ordered by ``delta_chi_squared`` descending.  Lines are found
    sequentially (peeling), but the Δχ² each one carries is measured after the
    last acceptance with all the others in the nuisance basis, so the ordering
    is a ranking of comparable numbers rather than of the order they were
    found in.
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


def _running_median(values: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """Median of a ``window``-wide window, evaluated on a stride and interpolated.

    A median filter costs ``O(n·window)``, and the windows this module wants are
    tens of line widths across a padded spectrum — enough for that product to
    dominate the whole scan.  The result is a *floor*, though, which by
    construction cannot vary faster than its own window, so it is enough to
    place ``_FLOOR_MEDIAN_STRIDES_PER_WINDOW`` full-window medians per window
    and interpolate between them.  Each estimate still sees every bin of its
    own window — decimating the window's *contents* instead would be cheaper
    still, but a median over a subsample is a noisier quantile, and that noise
    lands straight in the MAD every peak's SNR is divided by.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    n = int(values.size)
    width = max(1, min(int(window), n))
    if width >= n:
        return np.full(n, float(np.median(values)))
    step = max(1, width // _FLOOR_MEDIAN_STRIDES_PER_WINDOW)
    windows = sliding_window_view(values, width)[::step]
    coarse = np.median(windows, axis=1)
    centres = np.arange(coarse.size, dtype=np.float64) * float(step) + 0.5 * (width - 1)
    return np.interp(np.arange(n, dtype=np.float64), centres, coarse)


def _band_noise_floor(magnitude: NDArray[np.float64], bins_per_fwhm: int) -> NDArray[np.float64]:
    """Wide running-median noise floor across one rung's band.

    Two failure modes bracket the choice of window.

    Too *narrow* and a group of lines becomes its own floor.  Real records
    carry clusters — three lines inside a factor 2.5 in frequency is an
    ordinary µSR spectrum — and a window of a few line widths slides from one
    line to the next without ever seeing the white part of the band: the
    cluster's own peaks score a few MAD above a floor they themselves lifted,
    and the record reads as empty.  Measured on the synthetic three-line
    cluster in the tests, a sixteen-width window scores those lines 5-6 where a
    band-wide floor scores them 11-16.

    Too *wide* — a single median over the whole band — and whatever the
    detrending in :func:`matched_apodisation_scan` could not remove reappears
    as structure at the bottom of the band: on the longest rungs a residual
    low-frequency skirt and its truncation ripples then score 12-31 against a
    flat floor where a following floor scores them 5-6.

    The window used is therefore ``_FLOOR_WINDOW_FWHMS`` line widths
    (``bins_per_fwhm`` accounts for the zero-padding oversampling).  Measured
    across the synthetic corpus, sixty-four widths puts every real line above
    the shortlist gate — including the weakest member of the three-line
    cluster, which sixteen widths scores at 4.8 and this scores at 5.8 — while
    holding every null record's strongest peak near 5.  One sigma-clip pass
    then keeps strong lines out of the refined median.
    """
    values = np.asarray(magnitude, dtype=np.float64)
    if values.size == 0:
        return values.copy()
    window = max(9, _FLOOR_WINDOW_FWHMS * max(1, int(bins_per_fwhm)) + 1)
    window = min(window, values.size if values.size % 2 == 1 else values.size - 1)
    if window <= 1:
        return np.full_like(values, float(np.median(values)))
    floor = _running_median(values, window)
    residual = values - floor
    scale = _mad_scale(residual)
    if scale > _EPS:
        clipped = np.where(residual > 3.0 * scale, floor, values)
        floor = _running_median(clipped, window)
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


def _remove_slow_background(
    elapsed: NDArray[np.float64],
    asymmetry: NDArray[np.float64],
    error: NDArray[np.float64],
    *,
    basis_rates: tuple[float, ...] = _SLOW_DECAY_RATES_PER_US,
) -> NDArray[np.float64]:
    """Subtract the weighted least-squares fit of the slow-decay dictionary."""
    weights = _weights(error)
    columns = np.column_stack([np.exp(-float(rate) * elapsed) for rate in basis_rates])
    coefficients, *_ = np.linalg.lstsq(columns * weights[:, None], asymmetry * weights, rcond=None)
    return np.asarray(asymmetry - columns @ coefficients, dtype=np.float64)


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
    """Transform ``(y − background)·exp(-t/τ)`` unwindowed, once per rung.

    The background removed is the weighted least-squares fit of the same slow
    monotonic dictionary the Δχ² stage uses as its nuisance model
    (``_SLOW_DECAY_RATES_PER_US``), which includes the constant, so this
    subsumes removing a DC pedestal.  It matters more than it sounds: a
    relaxing background transforms into a low-frequency skirt whose truncation
    ripples are genuine local maxima, and on the longest rungs — where the
    guard band starts at a frequency the rung's own resolution barely
    distinguishes from zero — those ripples score in the tens of MADs and fill
    the shortlist.  Removing the background rather than trying to model its
    spectrum drops them to the level of the noise (measured on the synthetic
    corpus: 27.6 → 6.5 and 36.3 → 5.3 on the two records with the strongest
    backgrounds).  The dictionary cannot eat a line: the guard band only
    reports frequencies completing at least ``min_cycles`` cycles within τ, and
    a signal oscillating three times over the crop is orthogonal to smooth
    decays.

    No taper is applied beyond the matched exponential: a taper is exactly what
    deletes the early nanoseconds this scan exists to see.

    Each rung reports its guarded band, its running noise floor over that band
    (:func:`_band_noise_floor`), and up to ``max_peaks_per_rung`` peaks in the
    excess over that floor, separated by at least one FWHM and standing
    ``_SCAN_PROMINENCE_MADS`` MADs above their surroundings.

    ``error`` does not weight the transform itself; it enters through the
    rebinning (which must combine errors in quadrature), through the background
    fit, and through the Δχ² stage that judges the peaks this returns.
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
        t_r, y_r, e_r, factor = _rung_record(t, y, e, tau_us, sample_budget)
        n_r = t_r.size
        dt_r = _bin_width(t_r)
        if n_r < _MIN_RUNG_POINTS or dt_r <= 0.0:
            continue

        elapsed_r = t_r - t_r[0]
        signal = _remove_slow_background(elapsed_r, y_r, e_r) * np.exp(-elapsed_r / tau_us)

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
    """Maximise Δχ² over a shrinking local (f, λ) grid, inside a fixed box.

    The box is set by the seed and never moves: ``λ`` within
    ``_REFINE_DAMPING_FACTOR`` either way of ``damping_rate_per_us``, and ``f``
    within ``frequency_span_mhz`` of ``frequency_mhz`` — or, when the caller
    does not say how good its seed is, within ``_REFINE_FREQUENCY_FWHMS`` line
    widths (``λ/π``) of it.  ``_REFINE_ROUNDS`` rounds of a ``_REFINE_GRID ×
    _REFINE_GRID`` grid then walk in, each centred on the best point so far and
    ``_REFINE_SHRINK`` times narrower, clipped back into the box.

    The frequency box wants to be *narrow*, and narrower than the line: what it
    has to cover is the error on the seed's position, not the width of the
    thing being fitted.  :func:`detect_damped_lines` passes one FWHM of the rung
    that found the peak, which is many times the position error of a peak that
    cleared the shortlist gate — and small enough that the fit cannot walk onto
    a neighbouring line, which in a cluster it otherwise does: a single envelope
    centred between two lines, wide enough to cover both, beats either line
    alone on Δχ² while being neither.

    Bounding λ is not a nicety.  Δχ² is not flat in λ once the line is fitted:
    shortening the envelope keeps discarding points at which the nuisance model
    can be wrong, so an unbounded search drifts to several times the true rate
    for a fraction of a unit of Δχ², and on a cluster it merges neighbouring
    lines into one overdamped blob.  The ladder pins the seed to within ~1.8×,
    so a fourfold box contains the answer with room to spare.

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
    seed_f = float(frequency_mhz)
    seed_lambda = float(damping_rate_per_us)
    given_span = float(frequency_span_mhz or 0.0)
    half_width = given_span if given_span > 0.0 else _REFINE_FREQUENCY_FWHMS * seed_lambda / np.pi
    f_lo = max(seed_f - half_width, _EPS)
    f_hi = seed_f + half_width
    lambda_lo = seed_lambda / _REFINE_DAMPING_FACTOR
    lambda_hi = seed_lambda * _REFINE_DAMPING_FACTOR
    best_f, best_lambda = seed_f, seed_lambda
    best_delta = basis.evaluate(best_f, best_lambda)[0]
    span = half_width
    factor = _REFINE_DAMPING_FACTOR
    for _round in range(int(_REFINE_ROUNDS)):
        frequencies = np.linspace(max(best_f - span, f_lo), min(best_f + span, f_hi), _REFINE_GRID)
        rates = np.geomspace(
            max(best_lambda / factor, lambda_lo), min(best_lambda * factor, lambda_hi), _REFINE_GRID
        )
        for candidate_f in frequencies:
            for candidate_lambda in rates:
                delta = basis.evaluate(float(candidate_f), float(candidate_lambda))[0]
                if delta > best_delta:
                    best_delta = delta
                    best_f = float(candidate_f)
                    best_lambda = float(candidate_lambda)
        span /= _REFINE_SHRINK
        factor **= 0.5
    return best_delta, best_f, best_lambda


# --------------------------------------------------------------------------- #
# Verification record
# --------------------------------------------------------------------------- #


def _verification_record(
    time: NDArray[np.float64],
    asymmetry: NDArray[np.float64],
    error: NDArray[np.float64],
    *,
    frequency_mhz: float,
    tau_us: float,
    sample_budget: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Crop and rebin the record on which one candidate's (f, λ) grid is swept.

    The crop keeps ``_VERIFY_LIFETIMES`` matched lifetimes — past that the line
    is gone and the extra points only feed the nuisance basis.  The record is
    then rebinned as far as the candidate's own frequency allows —
    ``_VERIFY_OVERSAMPLE`` samples per cycle — but never below
    ``_MIN_VERIFY_POINTS`` samples, and always at least as far as
    ``sample_budget`` demands.  A candidate whose frequency forbids the
    rebinning the budget wants is instead swept on the leading ``sample_budget``
    samples, which for a line fast enough to be in that position is far more
    record than it needs.

    Both the crop and the rebinning are fixed by the *seed*, not by the point
    the search has reached, and the record is built once per candidate: a record
    that shrank as λ grew would make Δχ² incomparable between grid points and
    reward large λ for the wrong reason (see the module docstring).  Nothing
    read off this record survives the search — amplitude, phase and the reported
    Δχ² are re-measured on the full informative record — so the bin-averaging
    attenuation of the rebinning never reaches the caller.
    """
    dt = _bin_width(time)
    n = time.size
    if dt <= 0.0 or n == 0:
        return time, asymmetry, error
    span = int(np.ceil(_VERIFY_LIFETIMES * tau_us / dt))
    crop = int(min(n, max(span, min(n, _MIN_VERIFY_POINTS))))
    budget = max(1, int(sample_budget))
    factor_budget = max(1, int(np.ceil(crop / budget)))
    if frequency_mhz > 0.0:
        factor_alias = max(1, int(1.0 / (_VERIFY_OVERSAMPLE * frequency_mhz * dt)))
    else:
        factor_alias = factor_budget
    factor_floor = max(1, crop // _MIN_VERIFY_POINTS)
    factor = max(factor_budget, min(factor_alias, factor_floor))
    if factor == 1:
        t_v, y_v, e_v = time[:crop], asymmetry[:crop], error[:crop]
    else:
        t_v, y_v, e_v = rebin(time[:crop], asymmetry[:crop], error[:crop], factor)
    if t_v.size > budget:
        t_v, y_v, e_v = t_v[:budget], y_v[:budget], e_v[:budget]
    return t_v, y_v, e_v


@dataclass(frozen=True)
class _Accepted:
    """An accepted line together with the scan seed it came from.

    The seed is kept because every refinement of that line — the first one and
    every polish sweep — is anchored on it (see :func:`_polish_lines`).  Its
    damping half is ``1 / line.tau_us``, which the line already carries.
    """

    line: DampedLine
    seed_frequency_mhz: float


def _is_oscillation(
    frequency_mhz: float,
    damping_rate_per_us: float,
    *,
    basis_rates: tuple[float, ...] = _SLOW_DECAY_RATES_PER_US,
) -> bool:
    """Is this (f, λ) an oscillation, or a relaxation shape Δχ² has mistaken for one?

    Δχ² is a statistical gate and cannot answer this: a static Gaussian
    Kubo-Toyabe minimum is a single dip-and-recover excursion that no sum of
    monotonic decays can reproduce, so a damped cosine laid over it scores a
    Δχ² in the thousands.  Two properties, either of which suffices, separate
    the cases:

    * **It oscillates.** ``f ≥ _MIN_CYCLES_PER_LIFETIME · λ`` — the fitted
      envelope contains at least a couple of full cycles, which a single
      excursion does not.  A Kubo-Toyabe dip is fitted at ``f/λ ≈ 1.2``.
    * **It is faster than any background.** ``λ ≥ _FAST_LINE_RATE_FACTOR ×``
      the top of the slow-decay dictionary means the feature dies inside a time
      over which the relaxation this module models cannot change shape, so it
      is not that relaxation.  This clause is what keeps genuinely low-Q lines:
      a cluster of 40-70 µs⁻¹ lines at 45-115 MHz sits at ``f/λ ≈ 1.2`` too,
      and is exactly what the module exists to find.

    The pair is a compromise with a known edge: a Kubo-Toyabe whose own ``δ`` is
    faster than the dictionary's top rate has the shape of a low-Q line at a
    scale the first clause cannot judge, and is reported as one.  Telling those
    apart needs the relaxation shape in the *model*, which is the fitting
    stage's job, not the detector's.
    """
    if damping_rate_per_us <= _EPS:
        return True
    if frequency_mhz >= _MIN_CYCLES_PER_LIFETIME * damping_rate_per_us:
        return True
    return damping_rate_per_us >= _FAST_LINE_RATE_FACTOR * max(basis_rates)


def _collides(frequency_mhz: float, damping_rate_per_us: float, lines: list[DampedLine]) -> bool:
    """Is this (f, λ) the same feature as one of ``lines``?

    "Same" means closer than ``_MIN_LINE_SEPARATION_FWHMS`` of the wider of the
    two lines' widths.  A second line inside the first one's width is not a
    second line: its column pair is nearly collinear with the pair already in
    the basis, and the least-squares solution to that is a pair of amplitudes
    an order of magnitude too large which cancel each other.
    """
    for line in lines:
        width = max(damping_rate_per_us, line.damping_rate_per_us) / np.pi
        if abs(frequency_mhz - line.frequency_mhz) < _MIN_LINE_SEPARATION_FWHMS * width:
            return True
    return False


# --------------------------------------------------------------------------- #
# Joint polish
# --------------------------------------------------------------------------- #


def _polish_lines(
    time: NDArray[np.float64],
    asymmetry: NDArray[np.float64],
    error: NDArray[np.float64],
    accepted: list[_Accepted],
    *,
    sample_budget: int,
    fwhm_by_tau: dict[float, float],
) -> list[_Accepted]:
    """Re-refine each line with all the *other* lines held in the basis.

    Matching pursuit takes the lines one at a time, so the first one is fitted
    while its neighbours are still in the residual.  On a cluster that biases it
    — a single envelope widens to cover the neighbours it has not been told
    about — and the bias is exactly what the bounded refinement box stops from
    becoming a runaway.  A coordinate-ascent sweep afterwards removes it: with
    every other accepted line in the nuisance basis, each line is re-refined and
    the move is kept only if it raises that line's Δχ² and still looks like an
    oscillation.

    The re-refinement restarts **from the scan's own seed**, in the same box the
    first pass used, rather than continuing from where the line currently sits.
    That is not a detail: a box re-anchored on the current point turns repeated
    sweeps into a random walk with a drift, and the drift has a direction — a
    wider envelope covers a neighbour as well as its own line — so a line
    polished a few times marches off to exactly the merged blob the bounded box
    exists to prevent (measured: three sweeps take a 75 MHz line at λ = 65 to
    68 MHz at λ = 120, swallowing the 45 MHz line whole).  Anchored to the seed,
    the search is the same bounded problem every time, only with a better
    nuisance model; sweeping it again can change the answer but cannot walk it
    anywhere.

    The Δχ², amplitude and phase each line carries afterwards are therefore
    *leave-one-out* values on the full informative record: the improvement from
    that line given all the others.  They are directly comparable between lines,
    and comparable with the acceptance threshold, which is what the caller
    ranks and reports.
    """
    current = list(accepted)
    if not current:
        return current
    elapsed = time - time[0]
    weights = _weights(error)
    for _sweep in range(_POLISH_SWEEPS):
        moved = False
        for index, entry in enumerate(current):
            line = entry.line
            others = tuple(
                (other.line.frequency_mhz, other.line.damping_rate_per_us)
                for position, other in enumerate(current)
                if position != index
            )
            basis = _NuisanceBasis(
                elapsed,
                asymmetry,
                weights,
                basis_rates=_SLOW_DECAY_RATES_PER_US,
                extra_lines=others,
            )
            frequency, damping = line.frequency_mhz, line.damping_rate_per_us
            delta, amplitude, phase = basis.evaluate(frequency, damping)
            t_v, y_v, e_v = _verification_record(
                time,
                asymmetry,
                error,
                frequency_mhz=entry.seed_frequency_mhz,
                tau_us=line.tau_us,
                sample_budget=sample_budget,
            )
            if t_v.size >= _MIN_RECORD_POINTS:
                _local, moved_f, moved_lambda = refine_line(
                    t_v,
                    y_v,
                    e_v,
                    entry.seed_frequency_mhz,
                    1.0 / line.tau_us,
                    extra_lines=others,
                    frequency_span_mhz=fwhm_by_tau.get(line.tau_us),
                )
                candidate = basis.evaluate(moved_f, moved_lambda)
                if (
                    candidate[0] > delta
                    and _is_oscillation(moved_f, moved_lambda)
                    and (moved_f, moved_lambda) != (frequency, damping)
                ):
                    frequency, damping = moved_f, moved_lambda
                    delta, amplitude, phase = candidate
                    moved = True
            current[index] = replace(
                entry,
                line=replace(
                    line,
                    frequency_mhz=float(frequency),
                    damping_rate_per_us=float(damping),
                    amplitude_percent=float(amplitude),
                    phase_rad=float(phase),
                    delta_chi_squared=float(delta),
                ),
            )
        if not moved:
            break
    return current


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
    (:func:`~asymmetry.core.fitting.peak_detection.effective_analysis_window`)
    and scanned over a matched apodisation ladder.  Clustered scan peaks are
    shortlisted, strongest first, and each is verified by a
    look-elsewhere-corrected Δχ² test; the first candidate to pass is accepted,
    **subtracted from the record, and the ladder is scanned again** on the
    residual, so the next line is looked for in a spectrum the accepted one no
    longer shapes — neither as a peak nor as a lifted local noise floor.  An
    accepted line also joins the nuisance basis, so a strong line's leakage
    cannot manufacture a second one.  No time crop is asked of the caller: the
    ladder replaces it.

    When no further line can be accepted, every line is re-fitted with all the
    others in the nuisance basis (:func:`_polish_lines`), and the Δχ², amplitude
    and phase each one reports come from that fit — on the full informative
    record, leave-one-out — so two lines' numbers are comparable and the
    threshold means the same thing for each.  The (f, λ) search itself runs on a
    short per-candidate record purely for speed
    (:func:`_verification_record`).

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
    fwhm_by_tau = {rung.tau_us: rung.fwhm_mhz for rung in rungs}
    shortlist = max(0, int(max_candidates))

    elapsed = t - t[0]
    weights = _weights(e)
    accepted: list[_Accepted] = []
    peeled: list[tuple[float, float]] = []
    while len(accepted) < int(max_lines):
        full_basis = _NuisanceBasis(
            elapsed,
            y,
            weights,
            basis_rates=_SLOW_DECAY_RATES_PER_US,
            extra_lines=tuple(peeled),
        )
        found: _Accepted | None = None
        lines_so_far = [entry.line for entry in accepted]
        for candidate in cluster_scan_peaks(rungs, min_snr=min_snr)[:shortlist]:
            seed_lambda = 1.0 / candidate.tau_us
            if _collides(candidate.frequency_mhz, seed_lambda, lines_so_far):
                continue
            if (
                full_basis.evaluate(candidate.frequency_mhz, seed_lambda)[0]
                < _SEED_DELTA_CHI2_FRACTION * threshold
            ):
                # The seed sits inside the refinement box, which spans about one
                # line width: a seed this far below the gate cannot be walked up
                # to it, and skipping it keeps a shortlist of null peaks cheap.
                continue
            t_v, y_v, e_v = _verification_record(
                t,
                y,
                e,
                frequency_mhz=candidate.frequency_mhz,
                tau_us=candidate.tau_us,
                sample_budget=sample_budget,
            )
            if t_v.size < _MIN_RECORD_POINTS:
                continue
            _local_delta, frequency, damping = refine_line(
                t_v,
                y_v,
                e_v,
                candidate.frequency_mhz,
                seed_lambda,
                extra_lines=tuple(peeled),
                frequency_span_mhz=fwhm_by_tau.get(candidate.tau_us),
            )
            delta, amplitude, phase = full_basis.evaluate(frequency, damping)
            if delta < threshold or _collides(frequency, damping, lines_so_far):
                continue
            if not _is_oscillation(frequency, damping):
                continue
            found = _Accepted(
                line=DampedLine(
                    frequency_mhz=float(frequency),
                    damping_rate_per_us=float(damping),
                    amplitude_percent=float(amplitude),
                    phase_rad=float(phase),
                    delta_chi_squared=float(delta),
                    tau_us=float(candidate.tau_us),
                    snr=float(candidate.snr),
                ),
                seed_frequency_mhz=float(candidate.frequency_mhz),
            )
            break
        if found is None:
            break
        accepted.append(found)
        accepted = _polish_lines(
            t, y, e, accepted, sample_budget=sample_budget, fwhm_by_tau=fwhm_by_tau
        )
        peeled = [(entry.line.frequency_mhz, entry.line.damping_rate_per_us) for entry in accepted]
        if len(accepted) >= int(max_lines):
            break
        working = y - sum(
            (
                entry.line.amplitude_percent
                * np.exp(-entry.line.damping_rate_per_us * elapsed)
                * np.cos(2.0 * np.pi * entry.line.frequency_mhz * elapsed + entry.line.phase_rad)
                for entry in accepted
            ),
            start=np.zeros_like(y),
        )
        rungs = matched_apodisation_scan(t, working, e, taus, sample_budget=sample_budget)

    while accepted:
        weakest = min(accepted, key=lambda entry: entry.line.delta_chi_squared)
        if weakest.line.delta_chi_squared >= threshold:
            break
        # Significant while it stood alone, insignificant once the lines
        # accepted after it are in the basis: two envelopes covering one
        # feature.  Drop the weaker and re-fit what is left, which changes the
        # survivors' numbers and so has to be re-checked.
        accepted = [entry for entry in accepted if entry is not weakest]
        accepted = _polish_lines(
            t, y, e, accepted, sample_budget=sample_budget, fwhm_by_tau=fwhm_by_tau
        )

    lines = sorted((entry.line for entry in accepted), key=lambda line: -line.delta_chi_squared)
    return DampedLineAnalysis(
        lines=tuple(lines),
        threshold_delta_chi_squared=threshold,
        n_trials=n_trials,
        taus_us=tuple(float(tau) for tau in taus),
        window_end_index=end,
        sample_budget=int(sample_budget),
        false_rate=float(false_rate),
    )


def measure_line_at_frequency(
    time: NDArray[np.float64],
    asymmetry: NDArray[np.float64],
    error: NDArray[np.float64],
    frequency_mhz: float,
    *,
    false_rate: float = _FALSE_LINE_RATE,
    sample_budget: int = _SAMPLE_BUDGET,
) -> DampedLine | None:
    """Measure the envelope of a line the caller has already named.

    :func:`detect_damped_lines` searches for the frequency as well as the
    envelope; this runs only the second half of that machinery at a frequency
    supplied from outside — a user's click on the wizard's FFT plot, or any
    other trusted seed.  It is what lets a hand-seeded frequency carry the one
    piece of information the click itself cannot: how fast the line decays.

    The λ search is the same one the scan uses.  The Δχ² of the damped pair
    against the slow-decay dictionary is evaluated at every ladder rung whose
    guard band admits ``frequency_mhz`` (``f ≥ _SCAN_MIN_CYCLES/τ``), the best
    rung seeds :func:`refine_line`, and the refinement box is bounded in λ
    exactly as it is there — an unbounded λ drifts, for the reasons set out in
    that function.  The frequency box is one seed linewidth, wide enough to
    absorb the pointing error of a click and no wider.  Because the frequency
    is *given*, the look-elsewhere correction shrinks to the ladder: the search
    is over the admissible rungs (inflated by ``_TRIALS_REFINEMENT_FACTOR`` for
    the continuous refinement inside each), not over a whole frequency band, so
    the gate here is much lower than the scan's — which is the point of naming
    a frequency.

    Unlike the scan's per-candidate record, the λ sweep runs on **one** record —
    the whole informative window, rebinned only as far as ``frequency_mhz``
    allows (:func:`_verification_record`) — so Δχ² is comparable across the
    ladder.  The reported Δχ², amplitude and phase are then re-measured on the
    full informative record, so they mean what a :class:`DampedLine` from
    :func:`detect_damped_lines` means; ``snr`` is 0, since no scan rung found
    this line and no scan SNR was measured for it.

    Returns ``None`` when the record is too short, the frequency is outside the
    scan's usable band, no rung admits it, or the best (f, λ) does not clear
    the threshold or does not look like an oscillation.
    """
    frequency = float(frequency_mhz)
    t_full = np.asarray(time, dtype=np.float64)
    y_full = np.asarray(asymmetry, dtype=np.float64)
    e_full = np.asarray(error, dtype=np.float64)
    n_full = t_full.size
    if n_full < _MIN_RECORD_POINTS or y_full.size != n_full or e_full.size != n_full:
        return None
    if not np.isfinite(frequency) or frequency <= 0.0:
        return None

    end = int(effective_analysis_window(t_full, e_full))
    t = t_full[:end]
    y = y_full[:end]
    e = e_full[:end]
    dt = _bin_width(t)
    duration = float(t[-1] - t[0]) if t.size else 0.0
    if dt <= 0.0 or duration <= 0.0 or t.size < _MIN_RECORD_POINTS:
        return None
    if frequency >= _SCAN_F_MAX_FRACTION * 0.5 / dt:
        return None

    taus = [
        float(tau)
        for tau in tau_ladder(dt, duration)
        if frequency * float(tau) >= _SCAN_MIN_CYCLES
        and _is_oscillation(frequency, 1.0 / float(tau))
    ]
    if not taus:
        return None
    threshold = look_elsewhere_threshold(_TRIALS_REFINEMENT_FACTOR * len(taus), false_rate)

    t_w, y_w, e_w = _verification_record(
        t, y, e, frequency_mhz=frequency, tau_us=duration, sample_budget=sample_budget
    )
    if t_w.size < _MIN_RECORD_POINTS:
        return None
    sweep_basis = _NuisanceBasis(
        t_w - t_w[0],
        y_w,
        _weights(e_w),
        basis_rates=_SLOW_DECAY_RATES_PER_US,
        extra_lines=(),
    )
    best_tau = max(taus, key=lambda tau: sweep_basis.evaluate(frequency, 1.0 / tau)[0])
    _local, refined_f, refined_lambda = refine_line(
        t_w,
        y_w,
        e_w,
        frequency,
        1.0 / best_tau,
        frequency_span_mhz=1.0 / (np.pi * best_tau),
    )
    if not _is_oscillation(refined_f, refined_lambda):
        return None

    full_basis = _NuisanceBasis(
        t - t[0], y, _weights(e), basis_rates=_SLOW_DECAY_RATES_PER_US, extra_lines=()
    )
    delta, amplitude, phase = full_basis.evaluate(refined_f, refined_lambda)
    if delta < threshold:
        return None
    return DampedLine(
        frequency_mhz=float(refined_f),
        damping_rate_per_us=float(refined_lambda),
        amplitude_percent=float(amplitude),
        phase_rad=float(phase),
        delta_chi_squared=float(delta),
        tau_us=float(best_tau),
        snr=0.0,
    )
