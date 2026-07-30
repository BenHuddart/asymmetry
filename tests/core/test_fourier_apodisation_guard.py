"""The early-signal apodisation guard, and the synthetic study that calibrates it.

Every signal here is synthetic and built from clearly invented round numbers
(a 300 MHz line, damping rates of 60 or 0.05 µs⁻¹); nothing in this file
depends on measured data.

The study
---------
The guard applies the same two conditions over the leading
``_EARLY_WINDOW_FRACTION`` (15 %) of the record, on the error-weighted power
``|signal/σ|²``:

1. ``early_power_fraction ≥ 0.75`` — the power is concentrated early;
2. ``early_retained_fraction ≤ 0.2`` — the window removes that early power.

**It runs them twice**: once on the signal's own power (the ``"signal"`` pass)
and once on the power left after an error-weighted moving mean is subtracted
(the ``"oscillatory"`` pass), firing if either says the window deleted a
front-loaded signal.

The second pass exists because a real ordered-state μSR curve is never a bare
damped cosine — it carries a slowly relaxing tail whose amplitude can exceed the
oscillation's. No constant or low-order baseline removal takes out that tail's
residual curvature, so its power spreads across the whole record and *dilutes*
the concentration statistic: on the headline case below the signal pass reads
0.41, far under the trigger, even though the window has kept only 0.04 % of the
oscillatory content. High-passing first asks the question the guard means to
ask, and the same case reads 0.998.

Measured (noiseless unless stated); "pass" is the one that decided:

=========================================================  =====  ========  ===========  =====
case                                                       early  retained  pass         fires
=========================================================  =====  ========  ===========  =====
**MUST FIRE**
damped cosine + tail(20, 2.0 µs⁻¹), ``hann``               0.998  3.6e-04   oscillatory  yes
damped cosine + tail(20, 2.0 µs⁻¹), ``cosine``             0.998  8.6e-03   oscillatory  yes
damped cosine + tail(50, 0.5 µs⁻¹), ``hann``               0.998  3.6e-04   oscillatory  yes
damped cosine + tail(6, 2.0 µs⁻¹), ``hann``                0.998  3.6e-04   oscillatory  yes
damped cosine + tail(2.5, 0.5 µs⁻¹), ``hann``              0.978  8.1e-04   signal       yes
damped cosine, no tail, ``hann``                           0.996  7.6e-04   signal       yes
damped cosine, no tail, ``cosine``                         0.996  1.3e-02   signal       yes
fast pure relaxation, no oscillation, ``hann``             0.906  4.1e-04   signal       yes
**MUST NOT — safe apodisations on the headline signal**
``none``                                                   0.998  1.00      oscillatory  no
matched ``lorentzian``, τ = 1/λ                            0.998  0.538     oscillatory  no
``lorentzian``, τ = 2/λ                                    0.998  0.707     oscillatory  no
``lorentzian``, τ = 0.5/λ                                  0.998  0.354     oscillatory  no
``gaussian``, τ = 1/λ                                      0.998  0.720     oscillatory  no
**MUST NOT — line alive across a long record**
λ = 0, 8 µs, ``hann``                                      0.150  8.7e-03   signal       no
λ = 0.05 µs⁻¹, 8 µs, ``hann``                              0.206  8.4e-03   signal       no
λ = 0, 8 µs, σ ∝ e^{t/2τ_μ}, ``hann``                      0.433  7.2e-03   signal       no
λ = 0.1 µs⁻¹, 8 µs, σ ∝ e^{t/2τ_μ}, ``hann``               0.549  6.6e-03   signal       no
λ = 0.2 µs⁻¹, 8 µs, σ ∝ e^{t/2τ_μ}, ``hann``               0.644  6.0e-03   signal       no
λ = 0.4 µs⁻¹, 8 µs, σ ∝ e^{t/2τ_μ}, ``hann``               0.780  5.0e-03   signal       yes
**MUST NOT — no oscillation to delete**
straight line, ``hann``                                    0.329  6.9e-03   signal       no
quadratic, ``hann``                                        0.032  1.0e-03   signal       no
gentle exponential λ = 0.05 µs⁻¹, ``hann``                 0.361  6.6e-03   signal       no
exponential λ = 0.3 µs⁻¹, ``hann``                         0.524  5.4e-03   signal       no
pure noise, ``hann``                                       0.155  9.3e-03   signal       no
=========================================================  =====  ========  ===========  =====

Each threshold is set by the rows that must *not* fire while the must-fire rows
do.

**Retained-power threshold (0.2).** The binding constraint is the exponential
filter, because it is what the warning *recommends*: at the matched
τ = 1/λ it keeps 0.538 of the early power (weight ``e^{-λt}`` against power
``e^{-2λt}`` gives ``∫e^{-4λt}/∫e^{-2λt} → ½``), and even at τ = 0.5/λ — twice
as aggressive as matched — it keeps 0.354, still 1.8× clear of the threshold. A
taper on the same signal keeps 4e-04. Below about τ = 0.35/λ the guard does
start firing on the exponential filter, and that is intended: filtering three
times faster than the signal decays *is* deleting it, and the message's advice
(τ ≈ 1/λ) is the fix.

**Concentration threshold (0.75).** Set by the **error weighting**, not by the
flat case: real μSR errors grow with time as the counts decay, and 1/σ²
weighting tilts the statistic early on its own. A flat-error record alive across
the whole window reads 0.15; the *same* record with realistic ``σ ∝ e^{t/2τ_μ}``
growth already reads 0.43 with no damping at all, and 0.64 at λ = 0.2 µs⁻¹. 0.75
clears that tilt with margin while a lifetime well inside the record
(λ = 0.4 µs⁻¹ over 8 µs, dead to 4 % by the end) still trips it — a deliberate
yes, since a taper's zero at t = 0 does throw away the strongest part of such a
record. The oscillatory pass leaves every one of those benign rows within a few percent
of its signal-pass value and far below the trigger (an undamped 8 µs record
reads 0.150 on the signal pass and 0.184 on the oscillatory one — the
moving-mean kernel spans ~32 cycles of a 27 MHz line, so it passes the
oscillation essentially untouched), which is why one threshold serves both
passes.

**Odd padding is load-bearing.** The slow baseline is estimated with the record
odd-padded at both ends rather than the kernel truncated there. With truncation
the moving mean at the first sample is the mean of the *following* half-kernel,
a systematic ``slope · k/4`` error that lands inside the early window: a plain
straight line then leaves a residual concentrated 80 % early and trips the
guard. Odd padding reproduces a linear continuation exactly, cutting the
straight-line residual by ~150× and dropping every smooth-baseline row above
below the trigger.

Sensitivity limit (measured, and deliberately accepted): the statistic is a
*time-domain* power concentration, so a per-bin noise pedestal flattens it. The
guard fires down to a per-bin peak SNR of ~8 and stands down below it (measured:
fires at 8, silent at 6) — and, because the slow baseline is now removed first,
that limit is the **same with and without the tail**, where the whole-signal
statistic used to need ~16. False negatives are the safe direction for an
advisory warning — the documentation steering is the primary defence, and the
guard is the safety net for the unambiguous case.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fourier.apodisation import (
    ApodisationEarlySignalWarning,
    early_signal_apodisation_loss,
)
from asymmetry.core.fourier.fft import (
    fft_arrays,
    fft_asymmetry,
    prepare_fft_arrays,
    prepare_fft_time_signal,
)
from asymmetry.core.fourier.window import apply_fft_filter, apply_window

# The heavily damped case: a 300 MHz line whose 1/60 µs lifetime is a sixtieth
# of the 0.3 µs record, so the whole signal lives where a symmetric taper is
# ~zero.
_FAST_LAMBDA_PER_US = 60.0
_FAST_FREQUENCY_MHZ = 300.0
_FAST_RECORD_US = 0.3
_FAST_POINTS = 600

# The conventional case: a 27 MHz transverse-field line alive across an 8 µs
# record.
_TF_LAMBDA_PER_US = 0.05
_TF_FREQUENCY_MHZ = 27.0
_TF_RECORD_US = 8.0
_TF_POINTS = 1024


def _damped(lam: float, record_us: float, n: int, frequency_mhz: float, noise: float = 0.0):
    time = np.linspace(0.0, record_us, n)
    signal = 5.0 * np.exp(-lam * time) * np.cos(2.0 * np.pi * frequency_mhz * time)
    if noise:
        signal = signal + np.random.default_rng(20260730).normal(0.0, noise, n)
    return time, signal, np.full(n, 0.5)


def _heavily_damped():
    return _damped(_FAST_LAMBDA_PER_US, _FAST_RECORD_US, _FAST_POINTS, _FAST_FREQUENCY_MHZ)


# The physically standard shape: the damped line rides on a slowly relaxing
# tail of larger amplitude (the powder tail an ordered state always carries).
# Invented round numbers, chosen so the tail's residual curvature survives the
# constant baseline removal and dilutes the whole-signal statistic to 0.41.
_TAIL_AMPLITUDE = 20.0
_TAIL_LAMBDA_PER_US = 2.0


def _heavily_damped_on_a_slow_tail(
    amplitude: float = _TAIL_AMPLITUDE, lam: float = _TAIL_LAMBDA_PER_US
):
    time, signal, error = _heavily_damped()
    return time, signal + amplitude * np.exp(-lam * time), error


def _transverse_field(lam: float = _TF_LAMBDA_PER_US):
    return _damped(lam, _TF_RECORD_US, _TF_POINTS, _TF_FREQUENCY_MHZ)


def _fires(time, signal, error, **kwargs) -> bool:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prepare_fft_arrays(time, signal, error, **kwargs)
    return any(isinstance(entry.message, ApodisationEarlySignalWarning) for entry in caught)


# --- the study's MUST-trigger rows ------------------------------------------


@pytest.mark.parametrize("window", ["hann", "cosine"])
def test_symmetric_taper_on_a_heavily_damped_signal_warns(window: str) -> None:
    time, signal, error = _heavily_damped()

    with pytest.warns(ApodisationEarlySignalWarning, match="deleted the early-time signal"):
        prepare_fft_arrays(time, signal, error, window=window)


def test_the_warning_names_the_numbers_and_both_alternatives() -> None:
    time, signal, error = _heavily_damped()

    with pytest.warns(ApodisationEarlySignalWarning) as caught:
        prepare_fft_arrays(time, signal, error, window="hann")

    message = str(caught[0].message)
    assert "'hann'" in message
    assert "first 15%" in message
    assert 'window="none"' in message
    assert 'window="lorentzian"' in message
    assert "filter_time_constant_us" in message


# --- the study's MUST-NOT rows ----------------------------------------------


def test_no_window_does_not_warn_on_the_same_signal() -> None:
    time, signal, error = _heavily_damped()

    assert not _fires(time, signal, error, window="none")


def test_the_matched_exponential_filter_does_not_warn() -> None:
    """The apodisation the warning recommends must not trip it."""
    time, signal, error = _heavily_damped()

    assert not _fires(
        time,
        signal,
        error,
        window="lorentzian",
        filter_time_constant_us=1.0 / _FAST_LAMBDA_PER_US,
    )


@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
def test_the_exponential_filter_does_not_warn_at_any_reasonable_tau(scale: float) -> None:
    time, signal, error = _heavily_damped()

    assert not _fires(
        time,
        signal,
        error,
        window="lorentzian",
        filter_time_constant_us=scale / _FAST_LAMBDA_PER_US,
    )


def test_the_gaussian_filter_does_not_warn() -> None:
    """The WiMDA gaussian *filter* is weight-1 at t = 0, unlike a taper."""
    time, signal, error = _heavily_damped()

    assert not _fires(
        time,
        signal,
        error,
        window="gaussian",
        filter_time_constant_us=1.0 / _FAST_LAMBDA_PER_US,
    )


@pytest.mark.parametrize("window", ["hann", "cosine", "none"])
def test_a_conventional_transverse_field_record_never_warns(window: str) -> None:
    time, signal, error = _transverse_field()

    assert not _fires(time, signal, error, window=window)


def test_an_undamped_record_never_warns() -> None:
    time, signal, error = _transverse_field(lam=0.0)

    assert not _fires(time, signal, error, window="hann")


#: Muon lifetime in µs, used only to give the synthetic errors the realistic
#: ``σ ∝ e^{t/2τ_μ}`` growth that a decaying count rate produces.
_MUON_LIFETIME_US = 2.19703


@pytest.mark.parametrize("lam", [0.0, 0.1, 0.2])
def test_realistic_error_growth_does_not_by_itself_warn(lam: float) -> None:
    """The constraint that sets the concentration threshold.

    1/σ² weighting tilts the early-power statistic on its own once the errors
    grow with time, as real counting errors do. An undamped or weakly damped
    line must still not warn under that tilt.
    """
    time, signal, _flat = _transverse_field(lam=lam)
    growing = 0.05 * np.exp(time / (2.0 * _MUON_LIFETIME_US))

    assert not _fires(time, signal, growing, window="hann")


def test_pure_noise_does_not_warn() -> None:
    rng = np.random.default_rng(4242)
    time = np.linspace(0.0, _TF_RECORD_US, _TF_POINTS)
    noise = rng.normal(0.0, 1.0, _TF_POINTS)

    assert not _fires(time, noise, np.ones(_TF_POINTS), window="hann")


# --- the metric itself ------------------------------------------------------


def test_the_measured_numbers_match_the_documented_study() -> None:
    time, signal, error = _heavily_damped()
    weights = apply_window(np.ones(_FAST_POINTS), "hann")

    loss = early_signal_apodisation_loss(time, signal, weights, error)

    assert loss is not None
    assert loss.early_window_fraction == pytest.approx(0.15)
    assert loss.early_power_fraction > 0.99
    assert loss.early_retained_fraction < 0.01
    assert loss.triggered


def test_the_matched_filter_keeps_about_half_the_early_power() -> None:
    """The value that forces the retained-power threshold below 0.5."""
    time, signal, error = _heavily_damped()
    weights = apply_fft_filter(
        np.ones(_FAST_POINTS),
        time,
        mode="lorentzian",
        time_constant_us=1.0 / _FAST_LAMBDA_PER_US,
    )

    loss = early_signal_apodisation_loss(time, signal, weights, error)

    assert loss is not None
    assert loss.early_retained_fraction == pytest.approx(0.5, abs=0.1)
    assert not loss.triggered


def test_an_alive_record_puts_the_window_fraction_of_its_power_early() -> None:
    """The value that forces the concentration threshold above 0.15."""
    time, signal, error = _transverse_field(lam=0.0)
    weights = apply_window(np.ones(_TF_POINTS), "hann")

    loss = early_signal_apodisation_loss(time, signal, weights, error)

    assert loss is not None
    assert loss.early_power_fraction == pytest.approx(0.15, abs=0.02)
    assert not loss.triggered


def test_the_metric_stands_down_when_it_cannot_measure() -> None:
    time = np.linspace(0.0, 1.0, 8)
    assert early_signal_apodisation_loss(time, np.ones(8), np.zeros(8)) is None

    time = np.linspace(0.0, 1.0, 64)
    # A window that barely touches the early samples: nothing to report.
    assert early_signal_apodisation_loss(time, np.ones(64), np.ones(64)) is None
    # No power at all.
    tapered = apply_window(np.ones(64), "hann")
    assert early_signal_apodisation_loss(time, np.zeros(64), tapered) is None
    # Mismatched shapes.
    assert early_signal_apodisation_loss(time, np.ones(64), np.ones(32)) is None


def test_the_metric_falls_back_to_unweighted_power_without_errors() -> None:
    time, signal, _error = _heavily_damped()
    weights = apply_window(np.ones(_FAST_POINTS), "hann")

    loss = early_signal_apodisation_loss(time, signal, weights, None)

    assert loss is not None
    assert loss.triggered


# --- reach: both front doors, and no change to any returned value -----------


def test_the_guard_reaches_the_dataset_front_door_too() -> None:
    time, signal, error = _heavily_damped()
    dataset = MuonDataset(time=time, asymmetry=signal, error=error, metadata={})

    with pytest.warns(ApodisationEarlySignalWarning):
        prepare_fft_time_signal(dataset, window="hann")
    with pytest.warns(ApodisationEarlySignalWarning):
        fft_asymmetry(dataset, window="hann")


def test_the_guard_changes_no_returned_value() -> None:
    time, signal, error = _heavily_damped()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ApodisationEarlySignalWarning)
        warned_freqs, warned_real, warned_magnitude = fft_arrays(time, signal, error, window="hann")
        prepared = prepare_fft_arrays(time, signal, error, window="hann")

    expected = np.fft.rfft(prepared.signal)
    assert np.array_equal(warned_magnitude, np.abs(expected))
    assert np.array_equal(warned_real, expected.real)
    assert warned_freqs.size == warned_magnitude.size


def test_the_guard_sees_the_cropped_window() -> None:
    """A crop that keeps only the live part still warns under a taper.

    Cropping is not the cure on its own — the taper is still zero at the start
    of whatever window survives.
    """
    time, signal, error = _heavily_damped()

    assert _fires(time, signal, error, window="hann", t_max=0.1)


@pytest.mark.parametrize(("peak_snr", "expected"), [(8.0, True), (6.0, False)])
def test_the_documented_noise_sensitivity_limit(peak_snr: float, expected: bool) -> None:
    """Pins the sensitivity claim in this module's docstring.

    The statistic is a time-domain power concentration, so a per-bin noise
    pedestal flattens it. The guard fires down to a peak signal-to-noise of ~8
    on the heavily damped case and stands down below that — a false negative,
    which is the safe direction for an advisory warning.
    """
    sigma = 5.0 / peak_snr
    time, clean, _error = _heavily_damped()
    noisy = clean + np.random.default_rng(20260730).normal(0.0, sigma, _FAST_POINTS)

    assert _fires(time, noisy, np.full(_FAST_POINTS, sigma), window="hann") is expected


# --- the slow tail: the shape a real ordered state actually has --------------


@pytest.mark.parametrize("window", ["hann", "cosine"])
def test_a_damped_line_on_a_slow_tail_warns(window: str) -> None:
    """The headline must-fire case.

    A bare damped cosine is not what an ordered-state μSR record looks like: the
    oscillation rides on a slowly relaxing tail, often of larger amplitude. The
    tail's residual curvature survives any constant baseline removal and spreads
    its power across the whole record, so a whole-signal concentration statistic
    reads far below the trigger even though the taper has deleted essentially
    all of the oscillatory content.
    """
    time, signal, error = _heavily_damped_on_a_slow_tail()

    with pytest.warns(ApodisationEarlySignalWarning, match="deleted the early-time signal"):
        prepare_fft_arrays(time, signal, error, window=window)


@pytest.mark.parametrize(
    ("amplitude", "lam"),
    [(20.0, 2.0), (50.0, 0.5), (6.0, 2.0), (6.0, 0.5), (2.5, 0.5), (50.0, 2.0)],
)
def test_the_tail_case_warns_across_tail_shapes(amplitude: float, lam: float) -> None:
    time, signal, error = _heavily_damped_on_a_slow_tail(amplitude, lam)

    assert _fires(time, signal, error, window="hann")


def test_the_tail_dilutes_the_whole_signal_statistic_but_not_the_oscillatory_one() -> None:
    """The mechanism, measured: this is *why* the second pass exists."""
    time, signal, error = _heavily_damped_on_a_slow_tail()
    weights = apply_window(np.ones(_FAST_POINTS), "hann")
    baseline_free = signal - np.sum(signal / error) / np.sum(1.0 / error)

    loss = early_signal_apodisation_loss(time, baseline_free, weights, error)

    assert loss is not None
    # The pass that fires is the high-passed one, and it is not marginal.
    assert loss.measured_on == "oscillatory"
    assert loss.early_power_fraction > 0.99
    assert loss.early_retained_fraction < 0.01
    assert loss.triggered


def test_the_tail_case_is_safe_under_every_recommended_apodisation() -> None:
    """The warning's own advice must not trip the warning."""
    time, signal, error = _heavily_damped_on_a_slow_tail()

    assert not _fires(time, signal, error, window="none")
    for scale in (0.5, 1.0, 2.0):
        assert not _fires(
            time,
            signal,
            error,
            window="lorentzian",
            filter_time_constant_us=scale / _FAST_LAMBDA_PER_US,
        )
    assert not _fires(
        time,
        signal,
        error,
        window="gaussian",
        filter_time_constant_us=1.0 / _FAST_LAMBDA_PER_US,
    )


def test_a_heavily_over_filtered_exponential_does_warn() -> None:
    """Documented, and intended: τ well below 1/λ really is deleting signal."""
    time, signal, error = _heavily_damped_on_a_slow_tail()

    assert _fires(
        time,
        signal,
        error,
        window="lorentzian",
        filter_time_constant_us=0.25 / _FAST_LAMBDA_PER_US,
    )


def test_the_message_names_the_oscillatory_pass_when_that_is_what_fired() -> None:
    time, signal, error = _heavily_damped_on_a_slow_tail()

    with pytest.warns(ApodisationEarlySignalWarning) as caught:
        prepare_fft_arrays(time, signal, error, window="hann")

    message = str(caught[0].message)
    assert "oscillatory (slow-baseline-removed) power" in message
    assert 'window="none"' in message
    assert 'window="lorentzian"' in message


# --- the slow-baseline estimator, and the edge artefact it must not create ---


@pytest.mark.parametrize(
    ("label", "builder"),
    [
        ("straight line", lambda t: 20.0 - 2.0 * t),
        ("quadratic", lambda t: 20.0 - 2.0 * t + 0.3 * t**2),
        ("gentle exponential", lambda t: 20.0 * np.exp(-0.05 * t)),
        ("exponential", lambda t: 20.0 * np.exp(-0.3 * t)),
    ],
)
def test_a_smooth_baseline_with_no_oscillation_does_not_warn(label: str, builder) -> None:
    """Regression: the truncated-kernel edge artefact fired on a straight line.

    A moving mean whose window is truncated at the record's ends is biased by
    ``slope · k/4`` there, and that bias lands inside the early window — so a
    plain straight line, with no curvature and nothing a baseline removal could
    not take out, produced a large "oscillatory" residual concentrated at the
    start and tripped the guard. Odd padding removes it.
    """
    time = np.linspace(0.0, _TF_RECORD_US, _TF_POINTS)

    assert not _fires(time, builder(time), np.full(_TF_POINTS, 0.5), window="hann")


def test_odd_padding_leaves_a_straight_line_essentially_untouched() -> None:
    """The property that kills the edge artefact, pinned directly."""
    from asymmetry.core.fourier.apodisation import _slow_baseline

    time = np.linspace(0.0, _TF_RECORD_US, _TF_POINTS)
    ramp = 20.0 - 2.0 * time
    weights = np.ones(_TF_POINTS)
    kernel = max(1, int(round(0.15 * _TF_POINTS)))

    residual = ramp - _slow_baseline(ramp, weights, kernel)

    # A linear continuation is reproduced exactly; only rounding survives.
    assert np.max(np.abs(residual)) < 1e-2
    assert np.max(np.abs(residual)) < 1e-3 * float(np.ptp(ramp))


def test_the_slow_baseline_removes_the_tail_and_keeps_the_oscillation() -> None:
    from asymmetry.core.fourier.apodisation import _slow_baseline

    time, signal, error = _heavily_damped_on_a_slow_tail()
    tail = _TAIL_AMPLITUDE * np.exp(-_TAIL_LAMBDA_PER_US * time)
    oscillation = signal - tail
    inverse_variance = 1.0 / np.square(error)
    kernel = max(1, int(round(0.15 * _FAST_POINTS)))

    def high_pass(values: np.ndarray) -> np.ndarray:
        return values - _slow_baseline(values, inverse_variance, kernel)

    # The tail on its own is almost entirely absorbed into the baseline...
    assert np.sum(high_pass(tail) ** 2) < 0.01 * np.sum((tail - np.mean(tail)) ** 2)
    # ...while the oscillation survives essentially intact.
    assert np.sum(high_pass(oscillation) ** 2) > 0.9 * np.sum(oscillation**2)


def test_the_oscillatory_pass_is_skipped_when_there_is_nothing_oscillatory() -> None:
    """A signal that is pure slow baseline must decide on the signal pass."""
    time = np.linspace(0.0, _TF_RECORD_US, _TF_POINTS)
    ramp = 20.0 - 2.0 * time
    error = np.full(_TF_POINTS, 0.5)
    weights = apply_window(np.ones(_TF_POINTS), "hann")
    baseline_free = ramp - np.sum(ramp / error) / np.sum(1.0 / error)

    loss = early_signal_apodisation_loss(time, baseline_free, weights, error)

    assert loss is not None
    assert loss.measured_on == "signal"


def test_a_long_record_line_reads_the_same_on_both_passes() -> None:
    """Why one concentration threshold serves both passes."""
    from asymmetry.core.fourier.apodisation import _measure_early_loss, _slow_baseline

    time, signal, error = _transverse_field(lam=0.0)
    weights = apply_window(np.ones(_TF_POINTS), "hann")
    n_early = max(1, int(round(0.15 * _TF_POINTS)))
    inverse_variance = 1.0 / np.square(error)

    power = np.square(signal) * inverse_variance
    high_passed = signal - _slow_baseline(signal, inverse_variance, n_early)
    oscillatory_power = np.square(high_passed) * inverse_variance

    on_signal = _measure_early_loss(power, weights[:n_early], n_early, "signal")
    on_oscillatory = _measure_early_loss(
        oscillatory_power, weights[:n_early], n_early, "oscillatory"
    )

    assert on_signal is not None and on_oscillatory is not None
    # Both read the same to within a few percent, and both sit far below the
    # 0.75 trigger — which is why one threshold serves the two passes.
    assert on_oscillatory.early_power_fraction == pytest.approx(
        on_signal.early_power_fraction, abs=0.05
    )
    assert on_oscillatory.early_power_fraction < 0.5
    assert not on_signal.triggered and not on_oscillatory.triggered


def test_the_guard_still_changes_no_returned_value_on_the_tail_case() -> None:
    time, signal, error = _heavily_damped_on_a_slow_tail()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ApodisationEarlySignalWarning)
        _freqs, real, magnitude = fft_arrays(time, signal, error, window="hann")
        prepared = prepare_fft_arrays(time, signal, error, window="hann")

    expected = np.fft.rfft(prepared.signal)
    assert np.array_equal(magnitude, np.abs(expected))
    assert np.array_equal(real, expected.real)
