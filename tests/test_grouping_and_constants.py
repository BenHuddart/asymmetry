"""Tests for grouping utilities and physical constants."""

from __future__ import annotations

import numpy as np

from asymmetry.core.data.dataset import Histogram
from asymmetry.core.transform.grouping import apply_grouping
from asymmetry.core.utils.constants import (
    GAUSS_TO_TESLA,
    MUON_GYROMAGNETIC_RATIO_MHZ_PER_T,
    MUON_LIFETIME_US,
)


def test_apply_grouping_sums_and_truncates_to_shortest() -> None:
    h0 = Histogram(counts=np.array([1, 2, 3, 4], dtype=float), bin_width=0.01)
    h1 = Histogram(counts=np.array([10, 20, 30], dtype=float), bin_width=0.01)
    h2 = Histogram(counts=np.array([100, 200, 300, 400, 500], dtype=float), bin_width=0.01)

    grouped = apply_grouping([h0, h1, h2], [0, 1, 2])
    np.testing.assert_allclose(grouped, [111.0, 222.0, 333.0])


def test_physical_constants_reasonable_ranges() -> None:
    assert MUON_GYROMAGNETIC_RATIO_MHZ_PER_T > 100
    assert MUON_LIFETIME_US > 2.0
    assert GAUSS_TO_TESLA == 1.0e-4
