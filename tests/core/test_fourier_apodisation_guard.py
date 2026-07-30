"""The early-signal apodisation guard, and the synthetic study that calibrates it.

Every signal here is synthetic and built from clearly invented round numbers
(a 300 MHz line, damping rates of 60 or 0.05 µs⁻¹); nothing in this file
depends on measured data.

The study
---------
The guard fires on two conditions measured over the leading
``_EARLY_WINDOW_FRACTION`` (15 %) of the record, on the error-weighted power
``|signal/σ|²`` of the pre-window signal:

1. ``early_power_fraction ≥ 0.75`` — the signal's power is concentrated early;
2. ``early_retained_fraction ≤ 0.2`` — the window removes that early power.

Measured on the synthetic cases below (noiseless unless stated):

====================================================  ===========  ========  =====
case                                                  early power  retained  fires
====================================================  ===========  ========  =====
λ = 60 µs⁻¹, 0.3 µs record, ``hann``                        0.996  7.6e-04   yes
λ = 60 µs⁻¹, 0.3 µs record, ``cosine``                      0.996  1.3e-02   yes
λ = 60 µs⁻¹, 0.3 µs record, ``none``                        0.996  1.00      no
λ = 60 µs⁻¹, 0.3 µs record, matched ``lorentzian``          0.996  0.533     no
λ = 0.05 µs⁻¹, 8 µs record (alive), ``hann``                0.206  8.4e-03   no
λ = 0, 8 µs record (undamped), ``hann``                     0.150  8.7e-03   no
λ = 0, 8 µs record, σ ∝ e^{t/2τ_μ}, ``hann``                0.433  7.2e-03   no
λ = 0.2 µs⁻¹, 8 µs record, σ ∝ e^{t/2τ_μ}, ``hann``         0.644  6.0e-03   no
λ = 0.4 µs⁻¹, 8 µs record, σ ∝ e^{t/2τ_μ}, ``hann``         0.780  5.0e-03   yes
====================================================  ===========  ========  =====

Each threshold is set by the rows that must *not* fire while the first two must.

The matched exponential filter (``window="lorentzian"``,
``filter_time_constant_us = 1/λ``) is the binding constraint on the
retained-power threshold: it keeps ~½ of the early power (weight ``e^{-λt}``
against power ``e^{-2λt}`` gives ``∫e^{-4λt}/∫e^{-2λt} → ½``), and it is the
apodisation the warning *recommends*, so the threshold must sit well below 0.5 —
0.2 leaves a factor of 2.7 of margin.

The concentration threshold is set by the **error weighting**, not by the flat
case: real μSR errors grow with time as the counts decay, and 1/σ² weighting
tilts the statistic early on its own. A flat-error record alive across the whole
window reads 0.15; the *same* record with realistic ``σ ∝ e^{t/2τ_μ}`` growth
already reads 0.43 with no damping at all, and 0.64 at λ = 0.2 µs⁻¹. 0.75 clears
that pure-weighting tilt with margin while a lifetime well inside the record
(λ = 0.4 µs⁻¹ over 8 µs, i.e. dead to 4 % by the end) still trips it. The last
row is a deliberate yes: a taper's zero at t = 0 does throw away the strongest
part of such a record, and the advice is sound there.

Sensitivity limit (measured, and deliberately accepted): the statistic is a
*time-domain* power concentration, so a per-bin noise pedestal flattens it. On
the λ = 60 µs⁻¹ row above the guard fires down to a per-bin peak SNR of ~16 and
stands down below it (measured: fires at 16, silent at 12). False negatives are
the safe direction for an advisory warning — the documentation steering is the
primary defence, and the guard is the safety net for the unambiguous case.
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


@pytest.mark.parametrize(("peak_snr", "expected"), [(16.0, True), (12.0, False)])
def test_the_documented_noise_sensitivity_limit(peak_snr: float, expected: bool) -> None:
    """Pins the sensitivity claim in this module's docstring.

    The statistic is a time-domain power concentration, so a per-bin noise
    pedestal flattens it. The guard fires down to a peak signal-to-noise of ~16
    on the heavily damped case and stands down below that — a false negative,
    which is the safe direction for an advisory warning.
    """
    sigma = 5.0 / peak_snr
    time, clean, _error = _heavily_damped()
    noisy = clean + np.random.default_rng(20260730).normal(0.0, sigma, _FAST_POINTS)

    assert _fires(time, noisy, np.full(_FAST_POINTS, sigma), window="hann") is expected
