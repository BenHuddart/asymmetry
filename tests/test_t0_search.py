"""Automatic t0 search: prompt peak (continuous) and pulse edge (pulsed)."""

from __future__ import annotations

import os

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram
from asymmetry.core.transform import find_t0, find_t0_for_run, source_is_pulsed
from asymmetry.core.utils.constants import MUON_LIFETIME_US


def _continuous_histogram(t0_bin: int, n: int = 600, seed: int = 0) -> np.ndarray:
    """Sharp prompt peak at t0 over a decaying spectrum."""
    rng = np.random.default_rng(seed)
    t = (np.arange(n) - t0_bin) * 0.00125
    counts = 200.0 * np.exp(-np.clip(t, 0.0, None) / MUON_LIFETIME_US) * (t >= 0)
    counts[t0_bin] = 5000.0
    counts[:t0_bin] = 5.0  # uncorrelated pre-t0 background
    return rng.poisson(counts).astype(np.float64)


def _pulsed_histogram(centre_bin: int, half_width: int, n: int = 2000, seed: int = 0):
    """ISIS-like pulse: linear rise over 2·half_width, then muon decay."""
    rng = np.random.default_rng(seed)
    counts = np.zeros(n)
    rise = slice(centre_bin - half_width, centre_bin + half_width + 1)
    counts[rise] = np.linspace(0.0, 4000.0, 2 * half_width + 1)
    t = (np.arange(n) - centre_bin - half_width) * 0.016
    after = t > 0
    counts[after] = 4000.0 * np.exp(-t[after] / MUON_LIFETIME_US)
    return rng.poisson(counts).astype(np.float64)


def test_continuous_finds_prompt_peak_exactly():
    estimate = find_t0(_continuous_histogram(t0_bin=42), pulsed=False)
    assert estimate.ok
    assert estimate.strategy == "prompt_peak"
    assert estimate.t0_bin == 42


def test_continuous_ties_resolve_to_earliest_bin():
    """WiMDA parity: the descending strict-comparison scan keeps the earliest
    maximal bin; np.argmax does the same."""
    counts = np.zeros(100)
    counts[[30, 60]] = 500.0
    assert find_t0(counts, pulsed=False).t0_bin == 30


def test_pulsed_finds_rising_edge_midpoint_not_peak():
    """Divergence D9: pulse-centre (half-maximum) convention, where WiMDA
    returns the pulse peak."""
    centre, half_width = 40, 10
    estimate = find_t0(_pulsed_histogram(centre, half_width), pulsed=True)
    assert estimate.ok
    assert estimate.strategy == "pulse_edge"
    assert abs(estimate.t0_bin - centre) <= 1
    assert estimate.peak_bin >= centre + half_width - 1  # peak is later


def test_pulsed_failure_when_no_leading_edge():
    counts = np.full(100, 1000.0)  # starts above half-maximum
    estimate = find_t0(counts, pulsed=True)
    assert not estimate.ok
    assert "edge" in estimate.message.lower()


def test_empty_histogram_fails_cleanly():
    assert not find_t0(np.zeros(50), pulsed=False).ok
    assert not find_t0(np.array([]), pulsed=True).ok


def test_run_consensus_and_spread():
    histograms = [
        Histogram(counts=_continuous_histogram(40 + offset, seed=offset), bin_width=0.00125)
        for offset in (0, 1, 0, 2)
    ]
    search = find_t0_for_run(histograms, {"facility": "PSI"})
    assert search.ok
    assert search.strategy == "prompt_peak"
    assert search.consensus_t0_bin in (40, 41)
    assert search.spread_bins == 2
    assert len(search.estimates) == 4


def test_run_search_recovers_known_t0_on_pulsed_data():
    histograms = [
        Histogram(counts=_pulsed_histogram(40, 10, seed=seed), bin_width=0.016) for seed in range(4)
    ]
    search = find_t0_for_run(histograms, {"facility": "ISIS"})
    assert search.ok
    assert search.strategy == "pulse_edge"
    assert abs(search.consensus_t0_bin - 40) <= 1


def test_source_inference():
    assert source_is_pulsed({"facility": "ISIS"})
    assert source_is_pulsed({"instrument": "EMU", "facility": "Rutherford"})
    assert not source_is_pulsed({"facility": "PSI"})
    assert not source_is_pulsed({"instrument": "LEM"})
    assert source_is_pulsed({})  # unknown defaults to pulsed


def test_run_search_handles_dead_detectors():
    good = Histogram(counts=_continuous_histogram(40), bin_width=0.00125)
    dead = Histogram(counts=np.zeros(600), bin_width=0.00125)
    search = find_t0_for_run([good, dead, good], {"facility": "PSI"})
    assert search.ok
    assert search.consensus_t0_bin == 40
    assert not search.estimates[1].ok


# --- corpus -----------------------------------------------------------------------

NICKEL_FILE = os.path.expanduser(
    "~/Documents/WiMDA muon school/Magnetism/Ferromagnetic nickel/Data/emu00124254.nxs"
)
EUO_FILE = os.path.expanduser(
    "~/Documents/WiMDA muon school/Magnetism/Magnetic ordering in EuO/data/deltat_pta_gps_2966.bin"
)


@pytest.mark.skipif(not os.path.exists(NICKEL_FILE), reason="nickel corpus not available")
def test_t0_search_recovers_loader_t0_on_pulsed_corpus():
    from asymmetry.core.io import load

    dataset = load(NICKEL_FILE)
    dataset = dataset[0] if isinstance(dataset, list) else dataset
    run = dataset.run
    search = find_t0_for_run(run.histograms, run.metadata)
    assert search.ok
    assert search.strategy == "pulse_edge"
    file_t0 = int(run.histograms[0].t0_bin)
    assert abs(search.consensus_t0_bin - file_t0) <= 2


@pytest.mark.skipif(not os.path.exists(EUO_FILE), reason="EuO corpus not available")
def test_t0_search_recovers_loader_t0_on_continuous_corpus():
    from asymmetry.core.io import load

    dataset = load(EUO_FILE)
    dataset = dataset[0] if isinstance(dataset, list) else dataset
    run = dataset.run
    search = find_t0_for_run(run.histograms, run.metadata)
    assert search.ok
    assert search.strategy == "prompt_peak"
    file_t0_values = [int(h.t0_bin) for h in run.histograms]
    assert min(file_t0_values) - 1 <= search.consensus_t0_bin <= max(file_t0_values) + 1
