"""Behaviour-pinning for the unified ``rebin_counts`` helper (reconciliation N1).

The count-preserving bunching helper previously lived as two byte-identical
``_rebin_group_counts`` copies in ``core/fitting/grouped_time_domain.py`` and
``core/fourier/grouped.py``. This pins the shared
``core.transform.rebin.rebin_counts`` against a verbatim copy of the removed
body across a matrix of factors and sizes, so the unification stays a pure
de-duplication (zero behaviour change by construction).
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.transform.rebin import rebin_counts


def _legacy_rebin_group_counts(
    time: np.ndarray,
    counts: np.ndarray,
    factor: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Verbatim copy of the removed ``_rebin_group_counts`` body."""
    bunch_factor = max(1, int(factor))
    if bunch_factor <= 1 or counts.size < bunch_factor:
        return np.asarray(time, dtype=np.float64), np.asarray(counts, dtype=np.float64)

    n_new = counts.size // bunch_factor
    trimmed = n_new * bunch_factor
    rebinned_time = (
        np.asarray(time[:trimmed], dtype=np.float64).reshape(n_new, bunch_factor).mean(axis=1)
    )
    rebinned_counts = (
        np.asarray(counts[:trimmed], dtype=np.float64).reshape(n_new, bunch_factor).sum(axis=1)
    )
    return rebinned_time, rebinned_counts


@pytest.mark.parametrize("factor", [-3, 0, 1, 2, 3, 4, 7, 8, 50, 1000])
@pytest.mark.parametrize("size", [0, 1, 5, 16, 17, 100])
def test_rebin_counts_matches_legacy(factor: int, size: int) -> None:
    rng = np.random.default_rng(size * 100 + abs(factor))
    time = np.linspace(0.0, 8.0, size) if size else np.empty(0)
    counts = rng.poisson(1000.0, size=size).astype(np.float64)

    new_time, new_counts = rebin_counts(time, counts, factor)
    ref_time, ref_counts = _legacy_rebin_group_counts(time, counts, factor)

    np.testing.assert_array_equal(new_time, ref_time)
    np.testing.assert_array_equal(new_counts, ref_counts)
    assert new_time.dtype == ref_time.dtype == np.float64
    assert new_counts.dtype == ref_counts.dtype == np.float64


def test_rebin_counts_is_count_preserving() -> None:
    """Counts sum (not mean) — the value-domain ``rebin`` would average them."""
    counts = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float64)
    time = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    out_time, out_counts = rebin_counts(time, counts, 2)
    np.testing.assert_array_equal(out_counts, [6.0, 14.0])
    np.testing.assert_array_equal(out_time, [0.5, 2.5])
    assert out_counts.sum() == counts.sum()
