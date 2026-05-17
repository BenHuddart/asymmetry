"""Tests for file-provided deadtime correction."""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram
from asymmetry.core.transform.deadtime import (
    calibrate_deadtime_from_histograms,
    estimate_deadtime_from_histograms,
    has_file_deadtime,
    has_resolved_deadtime,
    parse_deadtime_calibration_text,
    prepare_histograms_with_deadtime,
)


def test_prepare_deadtime_uses_file_values_only() -> None:
    histogram = Histogram(np.array([100.0, 200.0]), bin_width=0.02)
    grouping = {"dead_time_us": [0.01], "good_frames": 1000.0}

    corrected, applied = prepare_histograms_with_deadtime(
        [histogram],
        grouping,
        use_deadtime=True,
    )

    assert applied is True
    assert grouping["deadtime_method"] == "file"
    assert corrected[0].counts[0] == pytest.approx(100.0 / (1.0 - (100.0 * 0.01 / (0.02 * 1000.0))))
    assert has_file_deadtime(grouping, 1) is True


def test_prepare_deadtime_without_file_values_is_noop() -> None:
    histogram = Histogram(np.array([100.0, 200.0]), bin_width=0.02)
    grouping = {}

    corrected, applied = prepare_histograms_with_deadtime(
        [histogram],
        grouping,
        use_deadtime=True,
    )

    assert applied is False
    assert "deadtime_method" not in grouping
    np.testing.assert_allclose(corrected[0].counts, histogram.counts)
    assert has_file_deadtime(grouping, 1) is False


def test_prepare_deadtime_preserves_non_file_method_for_resolved_payload() -> None:
    histogram = Histogram(np.array([100.0, 200.0]), bin_width=0.02)
    grouping = {
        "dead_time_us": [0.01],
        "good_frames": 1000.0,
        "deadtime_method": "estimate",
    }

    corrected, applied = prepare_histograms_with_deadtime(
        [histogram],
        grouping,
        use_deadtime=True,
    )

    assert applied is True
    assert grouping["deadtime_method"] == "estimate"
    assert has_resolved_deadtime(grouping, 1) is True
    assert has_file_deadtime(grouping, 1) is False
    assert corrected[0].counts[0] > histogram.counts[0]


def test_estimate_deadtime_from_histograms_recovers_uniform_tau() -> None:
    amplitude = 120.0
    tau_us = 0.02
    bin_width = 0.01
    num_good_frames = 1000.0
    times = (np.arange(12, dtype=float) + 1.0) * bin_width
    frame_scale = num_good_frames * bin_width
    true_counts = amplitude * np.exp(-times / 2.1969811)
    observed = true_counts * (
        1.0
        - (true_counts / frame_scale)
        * 2.1969811
        * (1.0 - np.exp(-tau_us / 2.1969811))
    )
    histograms = [Histogram(observed.copy(), bin_width=bin_width) for _ in range(4)]

    estimated = estimate_deadtime_from_histograms(
        histograms,
        t_good_offset=0,
        last_good_bin=11,
        num_good_frames=num_good_frames,
        max_bins=12,
    )

    assert estimated is not None
    assert estimated == pytest.approx(tau_us, rel=1e-2, abs=5e-4)


def test_calibrate_deadtime_from_histograms_recovers_per_detector_tau() -> None:
    bin_width = 0.01
    num_good_frames = 1000.0
    lifetime_us = 2.1969811
    times = (np.arange(12, dtype=float) + 1.0) * bin_width
    taus = [0.01, 0.02, 0.03]
    histograms: list[Histogram] = []
    for amplitude, tau_us in zip((120.0, 105.0, 90.0), taus, strict=True):
        frame_scale = num_good_frames * bin_width
        true_counts = amplitude * np.exp(-times / lifetime_us)
        observed = true_counts * (
            1.0
            - (true_counts / frame_scale)
            * lifetime_us
            * (1.0 - np.exp(-tau_us / lifetime_us))
        )
        histograms.append(Histogram(observed.copy(), bin_width=bin_width))

    calibrated = calibrate_deadtime_from_histograms(
        histograms,
        t_good_offset=0,
        last_good_bin=11,
        num_good_frames=num_good_frames,
        max_bins=12,
    )

    assert calibrated is not None
    assert calibrated == pytest.approx(taus, rel=1e-2, abs=5e-4)


def test_parse_deadtime_calibration_text_reads_wimda_style_file() -> None:
    text = "\n".join(
        [
            "2",
            "1 0.0274",
            "2 0.0301",
            "Run 1234",
        ]
    )

    values = parse_deadtime_calibration_text(text, n_histograms=2)

    assert values == pytest.approx([0.0274, 0.0301])
