"""Tests for simulate_capture_run in core/simulate.py (WP1.4).

Verification-plan §5: round-trip, provenance, bit-for-bit seed reproducibility.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.negmu.model import CaptureComponent
from asymmetry.core.simulate import simulate_capture_run

# ---------------------------------------------------------------------------
# Template helper
# ---------------------------------------------------------------------------


def _make_template(n_dets=2, n_bins=1024, bin_width=0.016, t0_bin=0):
    """Minimal 2-detector, single-group template matching verification-plan §2."""
    histograms = [
        Histogram(
            counts=np.zeros(n_bins, dtype=float),
            bin_width=bin_width,
            t0_bin=t0_bin,
            good_bin_start=t0_bin,
            good_bin_end=n_bins - 1,
        )
        for _ in range(n_dets)
    ]
    grouping = {
        "groups": {1: list(range(1, n_dets + 1))},  # 1-based detector numbers
        "group_names": {1: "Group 1"},
        "forward_group": 1,
        "backward_group": 1,
        "alpha": 1.0,
        "t0_bin": t0_bin,
        "t_good_offset": 0,
        "first_good_bin": t0_bin,
        "last_good_bin": n_bins - 1,
        "bin_index_base": 1,
        "bunching_factor": 1,
        "good_frames": 1.0,
        "deadtime_correction": False,
        "dead_time_us": [0.0] * n_dets,
        "included_groups": {1: True},
    }
    return Run(
        run_number=0,
        histograms=histograms,
        metadata={"title": "Capture test template"},
        grouping=grouping,
        source_file="",
    )


# ---------------------------------------------------------------------------
# Standard test fixture (case 2a parameters from verification-plan)
# ---------------------------------------------------------------------------

N_BINS = 1024
BIN_WIDTH = 0.016  # μs
TOTAL_EVENTS = 2.0e7
BACKGROUND = 5.0

COMPONENTS_2A = [
    CaptureComponent(label="C", tau_us=2.030),
    CaptureComponent(label="O", tau_us=1.795),
    CaptureComponent(label="decayBG", tau_us=2.1969811),
]
WEIGHTS_2A = {"C": 5.0, "O": 3.0, "decayBG": 2.0}


@pytest.fixture
def template():
    return _make_template(n_dets=2, n_bins=N_BINS, bin_width=BIN_WIDTH, t0_bin=0)


@pytest.fixture
def run_2a(template):
    return simulate_capture_run(
        template,
        COMPONENTS_2A,
        WEIGHTS_2A,
        total_events=TOTAL_EVENTS,
        seed=0,
        background_per_bin=BACKGROUND,
    )


# ---------------------------------------------------------------------------
# Provenance checks
# ---------------------------------------------------------------------------


def test_synthetic_flag(run_2a):
    assert run_2a.metadata.get("synthetic") is True


def test_capture_mode_flag(run_2a):
    sim = run_2a.metadata["simulation"]
    assert sim["capture_mode"] is True


def test_components_recorded(run_2a):
    sim = run_2a.metadata["simulation"]
    labels = [c["label"] for c in sim["components"]]
    assert "C" in labels
    assert "O" in labels
    assert "decayBG" in labels


def test_seed_recorded(run_2a):
    assert run_2a.metadata["simulation"]["seed"] == 0


def test_deadtimes_zeroed(run_2a):
    dts = run_2a.grouping.get("dead_time_us", [])
    assert all(dt == 0.0 for dt in dts)


# ---------------------------------------------------------------------------
# Bit-for-bit seed reproducibility
# ---------------------------------------------------------------------------


def test_seed_reproducibility(template):
    run_a = simulate_capture_run(
        template,
        COMPONENTS_2A,
        WEIGHTS_2A,
        total_events=TOTAL_EVENTS,
        seed=42,
        background_per_bin=BACKGROUND,
    )
    run_b = simulate_capture_run(
        template,
        COMPONENTS_2A,
        WEIGHTS_2A,
        total_events=TOTAL_EVENTS,
        seed=42,
        background_per_bin=BACKGROUND,
    )
    for ha, hb in zip(run_a.histograms, run_b.histograms):
        np.testing.assert_array_equal(ha.counts, hb.counts)


def test_different_seeds_differ(template):
    run_0 = simulate_capture_run(
        template,
        COMPONENTS_2A,
        WEIGHTS_2A,
        total_events=TOTAL_EVENTS,
        seed=0,
    )
    run_1 = simulate_capture_run(
        template,
        COMPONENTS_2A,
        WEIGHTS_2A,
        total_events=TOTAL_EVENTS,
        seed=1,
    )
    # At least one detector histogram should differ
    diffs = [
        not np.array_equal(h0.counts, h1.counts)
        for h0, h1 in zip(run_0.histograms, run_1.histograms)
    ]
    assert any(diffs)


# ---------------------------------------------------------------------------
# Window sum ≈ total_events (within Poisson tolerance)
# ---------------------------------------------------------------------------


def test_window_sum_near_total_events(run_2a):
    # Sum post-t0 counts across all detectors
    window_sum = sum(float(np.sum(h.counts)) for h in run_2a.histograms)
    # Subtract background contribution: n_dets * n_bins * background_per_bin
    n_dets = len(run_2a.histograms)
    bg_total = n_dets * N_BINS * BACKGROUND
    signal_sum = window_sum - bg_total
    # Signal should be within 10*sqrt(total_events) of total_events (very loose)
    tolerance = 10.0 * np.sqrt(TOTAL_EVENTS)
    assert abs(signal_sum - TOTAL_EVENTS) < tolerance, (
        f"signal_sum={signal_sum:.1f} not within {tolerance:.1f} of {TOTAL_EVENTS}"
    )


# ---------------------------------------------------------------------------
# Expected counts match direct numpy evaluation
# ---------------------------------------------------------------------------


def test_expected_matches_numpy(template):
    """Noise-free expected envelope matches direct Σ amp·exp(-t/τ)."""
    comps = [
        CaptureComponent(label="C", tau_us=2.030),
        CaptureComponent(label="decayBG", tau_us=2.1969811),
    ]
    weights = {"C": 0.7, "decayBG": 0.3}
    total_events = 1.0e8  # large so Poisson noise is relatively small
    seed = 7

    run = simulate_capture_run(
        template,
        comps,
        weights,
        total_events=total_events,
        seed=seed,
        background_per_bin=0.0,
    )

    # Compute expected noise-free counts from first principles
    n_dets = len(run.histograms)
    total_w = sum(weights.values())
    weights_norm = {k: v / total_w for k, v in weights.items()}

    n_bins = N_BINS
    bin_width = BIN_WIDTH
    t_post = np.arange(n_bins, dtype=float) * bin_width

    n_per_det = total_events / n_dets
    expected_per_det = np.zeros(n_bins)
    for comp in comps:
        w = weights_norm[comp.label]
        n_i = n_per_det * w
        n0_i = n_i * (1.0 - np.exp(-bin_width / comp.tau_us))
        expected_per_det += n0_i * np.exp(-t_post / comp.tau_us)

    # Both detectors should be roughly equal to expected (within ~5 Poisson σ)
    for hist in run.histograms:
        counts = hist.counts
        # Relative error ≤ 5/sqrt(N) element-wise at most bins
        # Use a loose global check: total sum within 5σ
        diff = float(np.sum(np.abs(counts - expected_per_det)))
        # Maximum expected absolute diff ≈ sqrt(total) per detector
        assert diff < 20.0 * np.sqrt(total_events), diff


# ---------------------------------------------------------------------------
# group_id filtering
# ---------------------------------------------------------------------------


def test_group_id_limits_signal():
    """When group_id is set, only that group's detectors carry the signal."""
    template_2grp = _make_template(n_dets=4, n_bins=512, bin_width=0.016, t0_bin=0)
    # Modify grouping so detectors 1-2 are group 1, detectors 3-4 are group 2
    template_2grp.grouping["groups"] = {1: [1, 2], 2: [3, 4]}
    template_2grp.grouping["group_names"] = {1: "Forward", 2: "Backward"}
    template_2grp.grouping["included_groups"] = {1: True, 2: True}
    template_2grp.grouping["dead_time_us"] = [0.0] * 4

    comps = [CaptureComponent(label="C", tau_us=2.030)]
    weights = {"C": 1.0}
    run = simulate_capture_run(
        template_2grp,
        comps,
        weights,
        total_events=1.0e5,
        group_id=1,
        seed=0,
        background_per_bin=0.0,
    )
    # Group 1 detectors (indices 0, 1) should have signal; group 2 (2, 3) should not
    signal_total_g1 = float(np.sum(run.histograms[0].counts) + np.sum(run.histograms[1].counts))
    signal_total_g2 = float(np.sum(run.histograms[2].counts) + np.sum(run.histograms[3].counts))
    assert signal_total_g1 > signal_total_g2 * 10, (
        f"Group 1 ({signal_total_g1:.0f}) should dominate Group 2 ({signal_total_g2:.0f})"
    )


# ---------------------------------------------------------------------------
# Validation guards
# ---------------------------------------------------------------------------


def test_invalid_total_events(template):
    comps = [CaptureComponent(label="C", tau_us=2.030)]
    with pytest.raises(ValueError, match="total_events"):
        simulate_capture_run(template, comps, {"C": 1.0}, total_events=-1.0)


def test_invalid_background(template):
    comps = [CaptureComponent(label="C", tau_us=2.030)]
    with pytest.raises(ValueError, match="background"):
        simulate_capture_run(template, comps, {"C": 1.0}, total_events=1e5, background_per_bin=-1.0)
