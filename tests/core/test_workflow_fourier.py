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
