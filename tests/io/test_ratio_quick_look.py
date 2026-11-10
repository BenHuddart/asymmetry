"""Loader quick looks retain supplied variances for signed and corrected-zero ratios."""

from dataclasses import replace

import numpy as np

from asymmetry.core.io.quick_asymmetry import quick_asymmetry
from tests.core.test_binning_entry_points import _run


def test_quick_ratio_uses_retained_variance_and_preserves_source():
    run = _run([-5.0, 0.0, 0.0], [10.0, 10.0, 0.0])
    run.histograms = [
        replace(h, count_variance=np.array(v))
        for h, v in zip(run.histograms, [[15.0, 5.0, 0.0], [10.0, 10.0, 0.0]], strict=True)
    ]
    result = quick_asymmetry(run.histograms, run.grouping)
    assert result.unavailable_reason is None
    np.testing.assert_array_equal(result.ratio_validity.reasons, [0, 0, 1])
    np.testing.assert_allclose(result.asymmetry, [-300.0, -100.0, 0.0])
    np.testing.assert_allclose(result.error[:2] ** 2, [112000.0, 2000.0])
    np.testing.assert_array_equal(run.histograms[0].counts, [-5.0, 0.0, 0.0])


def test_signed_counts_without_variance_keep_raw_run_with_unavailable_quick_look():
    run = _run([-5.0], [10.0])
    result = quick_asymmetry(run.histograms, run.grouping)
    assert "explicit variance" in result.unavailable_reason
    assert result.ratio_validity is None
    assert result.time.size == 0
    assert run.histograms[0].counts[0] == -5.0
