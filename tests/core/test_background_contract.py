"""Fixed numerical contracts for background count uncertainties and ranges."""

import numpy as np
import pytest

from asymmetry.core.transform import background as bg
from asymmetry.core.transform.fb_counts import BinWindow


@pytest.mark.parametrize("values", [[0.0, 0.0, 9.0], [4.0, 5.0, 0.0]])
def test_count_noise_stays_independent_of_the_exact_fixed_level(values):
    counts = np.array(values)
    result = bg.apply_grouped_background_correction(
        counts,
        counts,
        grouping={"background_mode": "fixed", "background_fixed_values": [1.0, 2.0]},
        t0_bin=0,
        bin_width_us=0.01,
    )
    np.testing.assert_array_equal(result.forward.variance, counts)
    np.testing.assert_array_equal(result.backward.variance, counts)
    assert result.forward.shared == result.backward.shared == ()


@pytest.mark.parametrize(
    "window,width,facility,expected",
    [
        ((0, 100), 0.005, "PSI", (0, 98)),
        ((0, 1), 0.005, "PSI", (0, 1)),
        ((10, 110), 0.005, "TRIUMF", (10, 105)),
        ((0, 100), 0.0, "PSI", (0, 100)),
        ((0, 100), 0.005, "ISIS", (0, 100)),
        ((0, 100), 0.005, "unknown", (0, 100)),
    ],
)
def test_period_endpoints(window, width, facility, expected):
    assert bg._adjust_range_for_beam_period(
        BinWindow(*window), bin_width_us=width, facility=facility
    ) == BinWindow(*expected)


@pytest.mark.parametrize(
    "grouping,t0,expected",
    [
        ({}, 10, ((1, 6), (1, 6))),
        ({"background_ranges": {"forward": [1, 4], "backward": [2, 3]}}, 10, ((1, 4), (2, 3))),
        ({"background_ranges": [2, 5]}, 10, ((2, 5), (2, 5))),
    ],
)
def test_range_input_values(grouping, t0, expected):
    assert bg._background_ranges(grouping, t0) == tuple(BinWindow(*window) for window in expected)


@pytest.mark.parametrize(
    "counts", [[np.nan], [1.0, np.nan], [np.inf], [2.0, np.inf], [-np.inf], [2.0, -np.inf]]
)
def test_background_boundary_rejects_nonfinite_counts(counts):
    with pytest.raises(ValueError, match="finite"):
        bg.apply_grouped_background_correction(
            np.array(counts),
            np.array(counts),
            grouping={"background_mode": "fixed", "background_fixed_values": [0.0, 0.0]},
            t0_bin=0,
            bin_width_us=0.01,
        )
