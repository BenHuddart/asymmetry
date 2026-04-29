"""Tests for file-provided deadtime correction."""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram
from asymmetry.core.transform.deadtime import has_file_deadtime, prepare_histograms_with_deadtime


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
    assert corrected[0].counts[0] == pytest.approx(
        100.0 / (1.0 - (100.0 * 0.01 / (0.02 * 1000.0)))
    )
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
