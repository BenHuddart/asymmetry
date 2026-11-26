"""Check the CdS figure's grid maxima against analytic precession signals."""

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.utils.constants import MUON_LIFETIME_US
from docs.screenshots.scenarios.corpus.cds_shallow_donor import (
    _cds_grouped_fft,
    _cds_maxent_reconstruction,
    _comparison_grid_maxima,
)

pytestmark = pytest.mark.gui


@pytest.mark.parametrize(
    ("frequencies", "amplitudes"),
    [
        pytest.param((1.37,), (0.20,), id="single-line"),
        pytest.param((1.23, 1.53), (0.12, 0.12), id="doublet"),
        pytest.param((1.23, 1.37, 1.53), (0.08, 0.15, 0.08), id="triplet"),
    ],
)
def test_cds_grid_maxima_follow_injected_lines(frequencies, amplitudes):
    dt_us = 0.016
    time_us = np.arange(1980) * dt_us
    oscillation = np.cos(2 * np.pi * np.outer(frequencies, time_us))
    polarisation = np.asarray(amplitudes) @ oscillation
    decay = 200_000 * np.exp(-time_us / MUON_LIFETIME_US)
    histograms = [
        Histogram(counts=decay * (1 + sign * polarisation) + 12, bin_width=dt_us, t0_bin=0)
        for sign in (1, -1)
    ]
    run = Run(
        run_number=1,
        histograms=histograms,
        metadata={"field": 100.0},
        grouping={
            "groups": {1: [1], 2: [2]},
            "group_names": {1: "Forward", 2: "Backward"},
            "first_good_bin": 0,
            "last_good_bin": time_us.size - 1,
            "deadtime_correction": False,
        },
    )

    fft = _cds_grouped_fft(run)
    maxent, _ = _cds_maxent_reconstruction(run)
    for grid, values in ((fft.time, fft.asymmetry), (maxent.frequencies_mhz, maxent.spectrum)):
        sample_step = grid[1] - grid[0]
        measured = grid[_comparison_grid_maxima(grid, values)]
        assert measured == pytest.approx(frequencies, abs=2 * sample_step)
        if len(frequencies) > 1:
            assert measured[-1] - measured[0] == pytest.approx(
                frequencies[-1] - frequencies[0], abs=4 * sample_step
            )
