"""Baseline removal for the FFT prepare path: ``detrend=``.

Every signal here is synthetic and built from invented round numbers; nothing
in this file depends on measured data.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fourier.fft import (
    fft_arrays,
    fft_asymmetry,
    prepare_fft_arrays,
    prepare_fft_time_signal,
)

# A 25 MHz line damped at 1.5 µs⁻¹ riding on a strong exponential baseline —
# the shape a relaxing zero-field signal has, where the mean subtraction leaves
# a trend behind and the low-frequency bins swamp the line.
_FREQUENCY_MHZ = 25.0
_LAMBDA_PER_US = 1.5
_BASELINE_RATE_PER_US = 0.5
_RECORD_US = 6.0
_N_POINTS = 1024


def _relaxing_signal() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = np.linspace(0.0, _RECORD_US, _N_POINTS)
    baseline = 40.0 * np.exp(-_BASELINE_RATE_PER_US * time)
    oscillation = 3.0 * np.exp(-_LAMBDA_PER_US * time) * np.cos(2.0 * np.pi * _FREQUENCY_MHZ * time)
    error = np.full(_N_POINTS, 0.2)
    return time, baseline + oscillation, error


def _peak_snr(freqs: np.ndarray, magnitude: np.ndarray) -> float:
    """Peak height at the line over the median magnitude away from it."""
    line = np.abs(freqs - _FREQUENCY_MHZ) < 3.0
    away = (freqs > 5.0) & ~line
    peak = float(np.max(magnitude[line]))
    floor = float(np.median(magnitude[away]))
    return peak / floor


def test_polynomial_detrend_improves_the_peak_snr_on_a_relaxing_baseline() -> None:
    time, signal, error = _relaxing_signal()

    plain_freqs, _, plain_magnitude = fft_arrays(time, signal, error)
    detrended_freqs, _, detrended_magnitude = fft_arrays(time, signal, error, detrend=2)

    assert np.array_equal(detrended_freqs, plain_freqs)
    assert _peak_snr(detrended_freqs, detrended_magnitude) > 3.0 * _peak_snr(
        plain_freqs, plain_magnitude
    )

    # The mechanism: the low-frequency bins the residual trend was dumping
    # power into collapse by two orders of magnitude.
    low = (plain_freqs > 0.0) & (plain_freqs < 5.0)
    assert np.sum(detrended_magnitude[low] ** 2) < 0.01 * np.sum(plain_magnitude[low] ** 2)


def test_a_pure_quadratic_detrends_to_zero() -> None:
    time = np.linspace(0.0, 4.0, 256)
    quadratic = 1.5 - 2.0 * time + 3.0 * time**2
    error = np.full(time.size, 0.1)

    prepared = prepare_fft_arrays(time, quadratic, error, detrend=2)

    assert np.max(np.abs(prepared.signal)) < 1e-9 * float(np.max(np.abs(quadratic)))


@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_a_polynomial_of_its_own_order_detrends_to_zero(order: int) -> None:
    time = np.linspace(0.0, 4.0, 256)
    coefficients = [2.0, -1.0, 0.5, 0.25][: order + 1]
    signal = sum(c * time**k for k, c in enumerate(coefficients))

    prepared = prepare_fft_arrays(time, np.asarray(signal), detrend=order)

    assert np.max(np.abs(prepared.signal)) < 1e-9 * float(np.max(np.abs(signal)))


def test_detrend_order_zero_matches_a_weighted_constant_removal() -> None:
    """Order 0 is a constant removal — the signal is left mean-free."""
    time, signal, error = _relaxing_signal()

    prepared = prepare_fft_arrays(time, signal, error, detrend=0)

    assert abs(float(np.mean(prepared.signal))) < 1e-9 * float(np.max(np.abs(signal)))


def test_callable_detrend_subtracts_the_supplied_baseline_verbatim() -> None:
    time, signal, error = _relaxing_signal()

    def slow_model(t: np.ndarray) -> np.ndarray:
        return 40.0 * np.exp(-_BASELINE_RATE_PER_US * t)

    prepared = prepare_fft_arrays(time, signal, error, detrend=slow_model)

    expected = signal - slow_model(time)
    assert np.allclose(prepared.signal, expected, rtol=0.0, atol=1e-12)


def test_callable_detrend_receives_the_cropped_time_axis() -> None:
    time, signal, error = _relaxing_signal()
    seen: list[np.ndarray] = []

    def record(t: np.ndarray) -> np.ndarray:
        seen.append(np.asarray(t))
        return np.zeros_like(t)

    prepare_fft_arrays(time, signal, error, detrend=record, t_min=1.0, t_max=4.0)

    assert len(seen) == 1
    assert float(seen[0].min()) >= 1.0
    assert float(seen[0].max()) <= 4.0


def test_detrend_reaches_the_dataset_front_door_identically() -> None:
    time, signal, error = _relaxing_signal()
    dataset = MuonDataset(time=time, asymmetry=signal, error=error, metadata={})

    ds_freqs, ds_real, ds_magnitude = fft_asymmetry(dataset, detrend=2)
    arr_freqs, arr_real, arr_magnitude = fft_arrays(time, signal, error, detrend=2)

    assert np.array_equal(arr_freqs, ds_freqs)
    assert np.array_equal(arr_real, ds_real)
    assert np.array_equal(arr_magnitude, ds_magnitude)


def test_detrend_replaces_rather_than_follows_the_average_subtraction() -> None:
    time, signal, error = _relaxing_signal()

    implicit = prepare_fft_arrays(time, signal, error, detrend=1)
    explicit = prepare_fft_arrays(time, signal, error, detrend=1, subtract_average_signal=False)

    assert np.array_equal(implicit.signal, explicit.signal)


def test_detrend_with_an_explicit_average_subtraction_is_rejected() -> None:
    time, signal, error = _relaxing_signal()

    with pytest.raises(ValueError, match="alternatives, not a sequence"):
        prepare_fft_arrays(time, signal, error, detrend=1, subtract_average_signal=True)


def test_default_baseline_removal_is_unchanged_without_detrend() -> None:
    """Leaving both unset keeps the historical error-weighted mean subtraction."""
    time, signal, error = _relaxing_signal()

    unset = prepare_fft_arrays(time, signal, error)
    explicit = prepare_fft_arrays(time, signal, error, subtract_average_signal=True)

    assert np.array_equal(unset.signal, explicit.signal)


@pytest.mark.parametrize("order", [-1, 4, 12])
def test_polynomial_order_is_capped(order: int) -> None:
    time, signal, error = _relaxing_signal()

    with pytest.raises(ValueError, match="outside the supported polynomial orders"):
        prepare_fft_arrays(time, signal, error, detrend=order)


@pytest.mark.parametrize("value", [2.5, "2", True])
def test_detrend_rejects_a_non_integer_non_callable(value: object) -> None:
    time, signal, error = _relaxing_signal()

    with pytest.raises(ValueError, match="detrend must be None"):
        prepare_fft_arrays(time, signal, error, detrend=value)


def test_detrend_callable_returning_the_wrong_shape_is_rejected() -> None:
    time, signal, error = _relaxing_signal()

    with pytest.raises(ValueError, match="returned shape"):
        prepare_fft_arrays(time, signal, error, detrend=lambda t: np.zeros(3))


def test_detrend_callable_returning_non_finite_values_is_rejected() -> None:
    time, signal, error = _relaxing_signal()

    with pytest.raises(ValueError, match="non-finite"):
        prepare_fft_arrays(time, signal, error, detrend=lambda t: np.full(t.size, np.nan))


def test_detrend_needs_more_points_than_its_order() -> None:
    time = np.linspace(0.0, 1.0, 3)

    with pytest.raises(ValueError, match="needs more than 3 usable point"):
        prepare_fft_arrays(time, np.zeros(3), detrend=3)


def test_detrend_can_delete_the_oscillation_if_the_order_chases_it() -> None:
    """The documented caveat, pinned: a cubic over half a cycle eats the line.

    This is why the order is capped and why the docstring says to keep it low —
    a polynomial that can follow the oscillation removes it rather than the
    trend it rides on.
    """
    time = np.linspace(0.0, 1.0, 256)
    slow_oscillation = np.cos(2.0 * np.pi * 0.5 * time)

    kept = prepare_fft_arrays(time, slow_oscillation, detrend=0)
    chased = prepare_fft_arrays(time, slow_oscillation, detrend=3)

    assert np.max(np.abs(chased.signal)) < 0.05 * np.max(np.abs(kept.signal))


def test_burg_prepare_path_still_accepts_a_plain_bool() -> None:
    """Existing callers passing ``subtract_average_signal=False`` are unaffected."""
    time, signal, error = _relaxing_signal()
    dataset = MuonDataset(time=time, asymmetry=signal, error=error, metadata={})

    prepared = prepare_fft_time_signal(dataset, subtract_average_signal=False)

    assert np.allclose(prepared.signal, signal)
