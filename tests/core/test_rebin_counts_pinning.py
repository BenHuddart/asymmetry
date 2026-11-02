"""Count-preserving bin merging against arithmetic-series sums."""

import numpy as np
import pytest

from asymmetry.core.transform.rebin import rebin_counts


@pytest.mark.parametrize("factor", [1, 2, 3, 20])
@pytest.mark.parametrize("size", [0, 1, 11, 100])
def test_rebin_counts_matches_arithmetic_series(factor, size):
    time = np.arange(size, dtype=float) * 0.1
    counts = np.arange(1, size + 1, dtype=float)
    actual_time, actual_counts = rebin_counts(time, counts, factor)
    starts = np.arange(size // factor) * factor
    expected_counts = factor * (2 * starts + factor + 1) / 2
    expected_time = (starts + (factor - 1) / 2) * 0.1
    np.testing.assert_allclose(actual_time, expected_time, rtol=1e-14)
    np.testing.assert_array_equal(actual_counts, expected_counts)
    assert actual_counts.sum() == counts[: size // factor * factor].sum()


@pytest.mark.parametrize("factor", [0, -1, 2.7, True, "2"])
def test_rebin_counts_rejects_invalid_factor(factor):
    with pytest.raises(ValueError, match="integer >= 1"):
        rebin_counts(np.arange(4.0), np.ones(4), factor)
