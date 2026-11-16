"""Optional pinned external-source execution of Mantid MaxentTools START.

This executes a standalone source module, not installed MuonMaxent. The checkout,
module hash, pulse constants, and mean-arrival origin are pinned in pulse-source.json.
Independent pulse mathematics lives in the ordinary analytic pulse tests.
Unconfigured checks skip; configured identity/import/command failures fail.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

from asymmetry.core.maxent import MaxEntConfig, build_maxent_input_from_counts, opus, pulse_response
from asymmetry.core.utils.constants import MUON_LIFETIME_US, PION_LIFETIME_US
from tests.mantid_pulse import run_start


@pytest.fixture
def mantid_source():
    source = os.environ.get("ASYMMETRY_MANTID_SOURCE")
    if not source:
        pytest.skip("set ASYMMETRY_MANTID_SOURCE for pinned external START execution")
    return Path(source)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # start.py's own DC div/0 (it patches gw[0])
@pytest.mark.parametrize("n_pulses", [1, 2])
def test_pulse_kernel_matches_pinned_external_start(n_pulses: int, mantid_source: Path) -> None:
    res = 0.016  # µs per time channel
    npts = 1000
    n_spectrum = 256
    # Mantid's frequency grid: ν_k = k / (res·npts·2).
    fperchan = 1.0 / (res * float(npts) * 2.0)
    frequencies = np.arange(n_spectrum) * fperchan

    # START's zero fine-offset uses the pion mean-arrival origin. Convert
    # explicitly to the proton-centred clock before comparing fixed phases.
    response = np.asarray(run_start(mantid_source, n_pulses))
    mean_arrival_response = response[:, 0] + 1j * response[:, 1]
    convol = mean_arrival_response * np.exp(-1j * 2 * np.pi * frequencies * PION_LIFETIME_US)
    # START hardcodes ISIS half-width 0.05 µs and separation 0.324 µs.
    p_cos, p_sin = pulse_response(
        frequencies, half_width_us=0.05, separation_us=0.324, n_pulses=n_pulses
    )

    # P(ω) = convolr + i·convoli = P_cos − i·P_sin (our sign convention).
    tol = dict(rtol=1.0e-3, atol=1.0e-6) if n_pulses == 2 else dict(rtol=1.0e-9, atol=1.0e-12)
    np.testing.assert_allclose(p_cos, np.real(convol), **tol)
    np.testing.assert_allclose(p_sin, -np.imag(convol), **tol)
    # DC point is exact either way.
    assert p_cos[0] == pytest.approx(1.0)
    assert p_sin[0] == pytest.approx(0.0)
    assert math.isclose(float(np.real(convol)[0]), 1.0, abs_tol=1e-12)

    config = MaxEntConfig(
        n_spectrum_points=n_spectrum,
        f_min_mhz=0,
        f_max_mhz=float(frequencies[-1]),
        auto_window=False,
        auto_steer=False,
        auto_phase_seed=False,
        fit_phases=False,
        group_phase_degrees={1: 23.0},
        pulse_mode="single" if n_pulses == 1 else "double",
    )
    counts = 5000 * np.exp(-np.arange(npts) * res / MUON_LIFETIME_US)
    prepared = build_maxent_input_from_counts({1: counts}, config, bin_width_us=res)
    group = prepared.groups[0]
    spectrum = np.linspace(0.01, 0.1, n_spectrum)
    expected = (
        np.real(
            np.exp(1j * (2 * np.pi * np.outer(group.time_us, frequencies) + np.deg2rad(23)))
            * convol
        )
        @ spectrum
    )
    np.testing.assert_allclose(
        opus(spectrum, prepared)[1], expected, rtol=0, atol=2e-5 if n_pulses == 2 else 1e-9
    )
