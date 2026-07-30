"""Array front doors for the FFT layer: parity with the dataset path.

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
    fft_complex_arrays,
    fft_complex_asymmetry,
    prepare_fft_arrays,
    prepare_fft_time_signal,
)

# One synthetic curve for every parity check: a 12 MHz cosine damped at
# 0.4 µs⁻¹ on a 4 µs record, plus a slow baseline so average subtraction and
# the fractional footing have something to bite on.
_FREQUENCY_MHZ = 12.0
_LAMBDA_PER_US = 0.4
_N_POINTS = 512
_RECORD_US = 4.0


def _synthetic_dataset() -> MuonDataset:
    time = np.linspace(0.0, _RECORD_US, _N_POINTS)
    envelope = np.exp(-_LAMBDA_PER_US * time)
    asymmetry = 20.0 + 8.0 * envelope * np.cos(2.0 * np.pi * _FREQUENCY_MHZ * time + 0.3)
    error = 0.05 + 0.01 * time
    return MuonDataset(
        time=time,
        asymmetry=asymmetry,
        error=error,
        metadata={"run_number": 1},
    )


#: Every kwarg combination worth pinning: windows (both filter modes and both
#: symmetric tapers), padding, crops, phase/t0 rotation, fractional footing and
#: the amplitude calibration.
_KWARG_CASES: list[dict] = [
    {},
    {"window": "hann"},
    {"window": "cosine"},
    {"window": "gaussian", "filter_time_constant_us": 1.5},
    {"window": "lorentzian", "filter_time_constant_us": 0.8, "filter_start_us": 0.5},
    {"padding_factor": 4},
    {"window": "hann", "padding_factor": 8},
    {"t_min": 0.5, "t_max": 3.0},
    {"t_min": 0.5, "t_max": 3.0, "window": "hann", "padding_factor": 2},
    {"phase_degrees": 37.0},
    {"phase_degrees": -110.0, "t0_offset_us": 0.02},
    {"subtract_average_signal": False},
    {"fractional": True},
    {"fractional": True, "amplitude_calibration": True},
    {"fractional": True, "amplitude_calibration": True, "window": "hann"},
    {"amplitude_calibration": True, "padding_factor": 2, "t_min": 0.25},
]


@pytest.mark.parametrize("kwargs", _KWARG_CASES, ids=lambda k: repr(sorted(k)))
def test_fft_arrays_matches_dataset_path_exactly(kwargs: dict) -> None:
    dataset = _synthetic_dataset()

    ds_freqs, ds_real, ds_magnitude = fft_asymmetry(dataset, **kwargs)
    arr_freqs, arr_real, arr_magnitude = fft_arrays(
        dataset.time, dataset.asymmetry, dataset.error, **kwargs
    )

    # No tolerance: the two front doors share one implementation.
    assert np.array_equal(arr_freqs, ds_freqs)
    assert np.array_equal(arr_real, ds_real)
    assert np.array_equal(arr_magnitude, ds_magnitude)


@pytest.mark.parametrize("kwargs", _KWARG_CASES, ids=lambda k: repr(sorted(k)))
def test_fft_complex_arrays_matches_dataset_path_exactly(kwargs: dict) -> None:
    dataset = _synthetic_dataset()

    ds_freqs, ds_spectrum = fft_complex_asymmetry(dataset, **kwargs)
    arr_freqs, arr_spectrum = fft_complex_arrays(
        dataset.time, dataset.asymmetry, dataset.error, **kwargs
    )

    assert np.array_equal(arr_freqs, ds_freqs)
    assert np.array_equal(arr_spectrum, ds_spectrum)


@pytest.mark.parametrize("kwargs", _KWARG_CASES, ids=lambda k: repr(sorted(k)))
def test_prepare_fft_arrays_matches_dataset_path_exactly(kwargs: dict) -> None:
    dataset = _synthetic_dataset()
    prepare_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key not in {"padding_factor", "phase_degrees", "t0_offset_us", "amplitude_calibration"}
    }

    from_dataset = prepare_fft_time_signal(dataset, **prepare_kwargs)
    from_arrays = prepare_fft_arrays(
        dataset.time, dataset.asymmetry, dataset.error, **prepare_kwargs
    )

    assert np.array_equal(from_arrays.signal, from_dataset.signal)
    assert from_arrays.dt == from_dataset.dt
    assert from_arrays.window_sum == from_dataset.window_sum
    assert from_arrays.fractional_baseline == from_dataset.fractional_baseline
    assert from_arrays.fractional_applied == from_dataset.fractional_applied


def test_fft_complex_arrays_stamps_diagnostics_like_the_dataset_path() -> None:
    dataset = _synthetic_dataset()
    ds_diagnostics: dict = {}
    arr_diagnostics: dict = {}

    fft_complex_asymmetry(
        dataset, fractional=True, amplitude_calibration=True, diagnostics=ds_diagnostics
    )
    fft_complex_arrays(
        dataset.time,
        dataset.asymmetry,
        dataset.error,
        fractional=True,
        amplitude_calibration=True,
        diagnostics=arr_diagnostics,
    )

    assert arr_diagnostics == ds_diagnostics


def test_fft_arrays_accepts_lists_and_omitted_error() -> None:
    dataset = _synthetic_dataset()

    freqs, real, magnitude = fft_arrays(list(dataset.time), list(dataset.asymmetry), window="hann")

    assert freqs.shape == real.shape == magnitude.shape
    assert np.all(np.isfinite(magnitude))
    peak = float(freqs[int(np.argmax(magnitude[1:])) + 1])
    assert abs(peak - _FREQUENCY_MHZ) < 1.0


def test_fft_arrays_is_scale_linear() -> None:
    """Percent in, percent-scaled spectrum out: doubling the input doubles it."""
    dataset = _synthetic_dataset()

    _, _, magnitude = fft_arrays(dataset.time, dataset.asymmetry, dataset.error)
    _, _, doubled = fft_arrays(dataset.time, 2.0 * dataset.asymmetry, dataset.error)

    assert np.allclose(doubled, 2.0 * magnitude, rtol=1e-12, atol=0.0)


def test_fft_arrays_rejects_two_dimensional_input() -> None:
    with pytest.raises(ValueError, match="asymmetry must be a one-dimensional array"):
        fft_arrays(np.zeros(8), np.zeros((2, 4)))


def test_fft_arrays_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        fft_arrays(np.zeros(8), np.zeros(7))


def test_fft_arrays_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="nothing to transform"):
        fft_arrays(np.zeros(0), np.zeros(0))


def test_fft_arrays_rejects_non_finite_values() -> None:
    time = np.linspace(0.0, 1.0, 16)
    asymmetry = np.zeros(16)
    asymmetry[3] = np.nan
    with pytest.raises(ValueError, match="asymmetry contains 1 non-finite"):
        fft_arrays(time, asymmetry)


def test_fft_arrays_rejects_an_empty_window_after_the_crop() -> None:
    time = np.linspace(0.0, 1.0, 16)
    with pytest.raises(ValueError, match="no data points remain"):
        fft_arrays(time, np.zeros(16), t_min=5.0)


def test_guard_messages_name_the_public_entry_point() -> None:
    with pytest.raises(ValueError, match="^prepare_fft_arrays:"):
        prepare_fft_arrays(np.zeros(0), np.zeros(0))
    with pytest.raises(ValueError, match="^fft_complex_arrays:"):
        fft_complex_arrays(np.zeros(0), np.zeros(0))
    with pytest.raises(ValueError, match="^fft_arrays:"):
        fft_arrays(np.zeros(0), np.zeros(0))


def test_dataset_path_still_accepts_unvalidated_input() -> None:
    """The dataset front door keeps its behaviour: no new guards there."""
    time = np.linspace(0.0, 1.0, 16)
    asymmetry = np.zeros(16)
    asymmetry[2] = np.nan
    dataset = MuonDataset(time=time, asymmetry=asymmetry, error=np.full(16, 0.1), metadata={})

    freqs, _real, magnitude = fft_asymmetry(dataset)

    assert freqs.size == magnitude.size
