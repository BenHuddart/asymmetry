"""Automatic t0 search: prompt peak (continuous) and pulse edge (pulsed)."""

from __future__ import annotations

import os

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram
from asymmetry.core.transform import (
    detector_t0_shifts,
    find_t0,
    find_t0_for_run,
    run_t0_time_us,
    shift_statistics,
    source_is_pulsed,
    tolerance_bins,
)
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


def test_pulsed_edge_ignores_early_noise_spike():
    """Review fix: a prompt flash/noise spike above half-maximum far before
    the pulse must not capture the edge — the crossing adjacent to the peak
    (after the last sub-half bin) is the edge."""
    counts = _pulsed_histogram(200, 10, seed=4)
    counts[3] = counts.max() * 0.9  # early spike above half-maximum
    estimate = find_t0(counts, pulsed=True)
    assert estimate.ok
    assert abs(estimate.t0_bin - 200) <= 1


# --- the run's exact t0 over per-detector values (D4) ------------------------


def _hist(t0_bin: int, t0_time_us: float | None) -> Histogram:
    return Histogram(counts=np.ones(10), bin_width=0.016, t0_bin=t0_bin, t0_time_us=t0_time_us)


def test_run_t0_time_us_averages_the_detectors_on_the_common_bin():
    """Detectors aligned onto the common bin set the run's exact t0; others don't."""
    histograms = [_hist(10, 0.166), _hist(10, 0.168), _hist(12, 0.200)]
    assert run_t0_time_us(histograms, 10) == pytest.approx(0.167)
    assert run_t0_time_us(histograms, 12) == pytest.approx(0.200)


def test_run_t0_time_us_is_none_without_any_exact_value():
    """PSI-style runs carry integer bins only: the fact is absent, not a guess."""
    histograms = [_hist(10, None), _hist(10, None)]
    assert run_t0_time_us(histograms, 10) is None
    # ...and so is a common bin no detector with an exact value sits on.
    assert run_t0_time_us([_hist(10, 0.166), _hist(12, None)], 12) is None


def test_run_t0_time_us_of_identical_detectors_is_that_value():
    histograms = [_hist(40, 0.648), _hist(40, 0.648)]
    assert run_t0_time_us(histograms, 40) == pytest.approx(0.648)


# --- per-detector shift statistics (phase 7) ---------------------------------


def _run_histograms(t0_bins: list[int], peaks: list[int]) -> list[Histogram]:
    """Continuous histograms whose file t0 and prompt peak are set per detector."""
    return [
        Histogram(
            counts=_continuous_histogram(t0_bin=peak, n=600, seed=peak),
            bin_width=0.00125,
            t0_bin=t0_bin,
        )
        for t0_bin, peak in zip(t0_bins, peaks, strict=True)
    ]


def test_shift_statistics_measure_each_detector_against_its_own_header():
    """A staggered run: the shifts are small even though the estimates are not.

    Detector 4 sits 170 bins before the rest and its header says so, so the raw
    estimates span 172 bins while every detector moved by +2.
    """
    histograms = _run_histograms([220, 216, 220, 50], [222, 218, 222, 52])

    search = find_t0_for_run(histograms, {"facility": "PSI"})

    assert search.ok
    assert search.spread_bins == 170  # the detectors' real stagger
    assert search.shift_median_bins == 2
    assert search.shift_spread_bins == 0


def test_shift_statistics_take_the_median_so_one_stray_detector_cannot_drag_them():
    histograms = _run_histograms([220, 220, 220, 220], [222, 222, 222, 260])

    search = find_t0_for_run(histograms, {"facility": "PSI"})

    assert search.shift_median_bins == 2
    assert search.shift_spread_bins == 38


def test_detector_t0_shifts_report_none_for_a_detector_that_did_not_resolve():
    histograms = _run_histograms([220, 220], [222, 222])
    histograms[1] = Histogram(counts=np.zeros(600), bin_width=0.00125, t0_bin=220)

    search = find_t0_for_run(histograms, {"facility": "PSI"})

    assert detector_t0_shifts(histograms, search.estimates) == [2, None]
    # A detector with no counts contributes nothing to either statistic.
    assert search.shift_median_bins == 2
    assert search.shift_spread_bins == 0


def test_shift_statistics_of_a_run_with_no_resolved_detector_are_zero():
    assert shift_statistics([None, None]) == (0, 0)


# --- how wide the feature t0 is read off is (phase 7) ------------------------


def test_prompt_peak_width_is_the_peak_fwhm_in_bins():
    """A narrow spike is one bin wide; a broadened one spans its own FWHM."""
    counts = np.full(400, 10.0)
    counts[200] = 5000.0
    assert find_t0(counts, pulsed=False).width_bins == 1

    # The same peak at finer binning: six bins at or above half maximum.
    counts = np.full(400, 10.0)
    counts[197:203] = [2600.0, 4000.0, 5000.0, 4800.0, 3600.0, 2600.0]
    estimate = find_t0(counts, pulsed=False)
    assert estimate.t0_bin == 199
    assert estimate.width_bins == 6


def test_pulse_edge_width_is_the_ten_to_ninety_rise_in_bins():
    """A 21-bin linear rise measures ~16 bins between the 10 % and 90 % levels."""
    estimate = find_t0(_pulsed_histogram(200, 10, seed=7), pulsed=True)
    assert estimate.ok
    assert 14 <= estimate.width_bins <= 18


def test_run_width_is_the_median_of_the_detectors_that_resolved():
    counts = np.full(400, 10.0)
    counts[197:203] = [2600.0, 4000.0, 5000.0, 4800.0, 3600.0, 2600.0]
    histograms = [Histogram(counts=counts.copy(), bin_width=0.000098, t0_bin=199) for _ in range(3)]
    histograms[2] = Histogram(counts=np.zeros(400), bin_width=0.000098, t0_bin=199)

    search = find_t0_for_run(histograms, {"facility": "PSI"})

    assert search.width_bins == 6
    # …which is the tolerance every verdict on this run is judged against.
    assert tolerance_bins(search.strategy, search.width_bins) == 6
