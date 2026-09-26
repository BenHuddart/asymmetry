"""Tests for the agent-facing quantitative Fourier workflow."""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.workflow.fourier import FourierSettings, fourier_spectrum


def test_fourier_spectrum_reports_a_measured_peak() -> None:
    time = np.arange(0.0, 8.0, 0.01)
    frequency = 2.5
    dataset = MuonDataset(
        time=time,
        asymmetry=12.0 * np.cos(2.0 * np.pi * frequency * time),
        error=np.full(time.size, 0.05),
        metadata={"run_number": 7},
    )

    outcome = fourier_spectrum(
        dataset,
        FourierSettings(window="none", padding_factor=4, f_min=0.2, f_max=6.0),
    )

    assert outcome.frequency[0] >= 0.2
    assert outcome.frequency[-1] <= 6.0
    assert outcome.resolution_mhz == pytest.approx(1.0 / time[-1])
    peaks = outcome.peaks["peaks"]
    assert peaks
    assert peaks[0]["frequency_mhz"] == pytest.approx(frequency, abs=0.03)


def test_fourier_settings_reject_empty_windows() -> None:
    with pytest.raises(ValueError, match="f_min must be below f_max"):
        FourierSettings(f_min=4.0, f_max=2.0)


def _line(frequency: float, amplitude: float, *, seed: int = 0) -> MuonDataset:
    time = np.arange(0.0, 8.0, 0.01)
    noise = np.random.default_rng(seed).normal(0.0, 0.5, time.size)
    return MuonDataset(
        time=time,
        asymmetry=amplitude * np.cos(2.0 * np.pi * frequency * time) + noise,
        error=np.full(time.size, 0.5),
        metadata={"run_number": 7},
    )


def test_a_zoom_around_a_line_still_detects_it() -> None:
    # The noise floor is the whole spectrum's: a band barely wider than the
    # line must not mistake the line for the noise.
    outcome = fourier_spectrum(
        _line(2.5, 4.0), FourierSettings(window="none", f_min=2.3, f_max=2.7)
    )
    peaks = outcome.peaks["peaks"]
    assert [round(peak["frequency_mhz"], 1) for peak in peaks] == [2.5]
    assert outcome.candidate_maxima == []


def test_with_nothing_detected_the_strongest_maxima_are_offered_as_candidates() -> None:
    outcome = fourier_spectrum(
        _line(2.5, 0.0), FourierSettings(window="none", f_min=0.5, f_max=6.0)
    )
    assert outcome.peaks["peaks"] == []
    candidates = outcome.candidate_maxima
    assert 1 <= len(candidates) <= 3
    heights = [entry["height_over_noise"] for entry in candidates]
    assert heights == sorted(heights, reverse=True)
    assert all(0.5 <= entry["frequency_mhz"] <= 6.0 for entry in candidates)
    assert outcome.to_dict()["candidate_maxima"] == candidates


def test_a_radical_line_pair_correlates_to_a_peak_at_the_sum_of_its_lines() -> None:
    from asymmetry.core.fourier.correlation import breit_rabi_pair
    from asymmetry.core.workflow.fourier import CorrelationSettings, correlation_spectrum
    from tests.core.test_fourier_correlation import _radical_tf_run

    field, coupling = 2900.0, 514.4
    run = _radical_tf_run(field_gauss=field, couplings_mhz=(coupling,))
    time = np.arange(2048) * 0.001

    outcome = correlation_spectrum(
        run, time, FourierSettings(), CorrelationSettings(field_gauss=field), group_ids=[1, 2]
    )

    strongest = max(outcome.peaks["peaks"], key=lambda peak: peak["amplitude"])
    assert strongest["frequency_mhz"] == pytest.approx(
        sum(breit_rabi_pair(field, coupling)), abs=2.0
    )
    assert outcome.to_dict()["axis"] == "hyperfine_coupling"
    assert outcome.to_dict()["correlation"] == {"field_gauss": field, "order": 2}


def test_a_plain_spectrum_is_on_the_frequency_axis() -> None:
    outcome = fourier_spectrum(_line(2.5, 4.0), FourierSettings(window="none"))
    assert outcome.to_dict()["axis"] == "frequency"
    assert outcome.to_dict()["correlation"] is None


def test_the_correlation_needs_a_transverse_field() -> None:
    from asymmetry.core.workflow.fourier import CorrelationSettings

    with pytest.raises(ValueError, match="transverse field"):
        CorrelationSettings(field_gauss=0.0)
