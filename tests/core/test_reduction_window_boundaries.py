"""Declared reduction windows are validated without reordering or defaults."""

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.project.profiles import (
    GoodWindowPolicy,
    resolve_good_window,
    run_file_good_window,
)
from asymmetry.core.transform.background import apply_grouped_background_correction


@pytest.mark.parametrize("facility", ["", "PSI"])
@pytest.mark.parametrize("window", [[4, 1], [-1, 2], [0, 9], [0, 1.5], ["bad", 1], [1]])
def test_declared_background_window_is_rejected(window, facility):
    with pytest.raises(ValueError, match="window|range|index"):
        apply_grouped_background_correction(
            np.ones(8),
            np.ones(8),
            grouping={"background_range": window},
            t0_bin=6,
            bin_width_us=0.01,
            facility=facility,
        )


def test_scalar_file_limits_are_not_reordered_before_manual_repair():
    run = Run(run_number=1, histograms=[Histogram(np.ones(20), bin_width=0.01)], metadata={})
    assert run_file_good_window(
        run, {"first_good_bin": 7, "last_good_bin": 3}, common_t0_bin=0
    ) == (7, 3)
    with pytest.raises(ValueError, match="window"):
        resolve_good_window(
            GoodWindowPolicy(), t0_bin=0, file_first_good=7, file_last_good=3, n_bins=20
        )


@pytest.mark.parametrize("window", [("bad", 3), (0, 1.5), (False, 3)])
def test_scalar_file_limits_reject_noninteger_indices(window):
    with pytest.raises(ValueError, match="window"):
        resolve_good_window(
            GoodWindowPolicy(),
            t0_bin=0,
            file_first_good=window[0],
            file_last_good=window[1],
            n_bins=20,
        )


def test_grouped_fourier_rejects_a_reversed_background_window():
    from asymmetry.core.fourier.group_background import _apply_group_background_correction

    with pytest.raises(ValueError, match="background range.*reversed"):
        _apply_group_background_correction(
            np.ones(8),
            grouping={
                "background_correction": True,
                "background_mode": "range",
                "background_range": [4, 1],
            },
            group_id=1,
            t0_bin=6,
            bin_width=0.01,
            metadata={"facility": "PSI"},
            source_file="synthetic.bin",
        )
