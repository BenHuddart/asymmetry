"""Declared reduction windows are validated without reordering or defaults."""

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.project.profiles import run_file_good_window
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


@pytest.mark.parametrize("window", [(7, 3), (0, 20), (-1, 3), ("bad", 3)])
def test_profile_file_window_rejects_invalid_declared_bounds(window):
    run = Run(run_number=1, histograms=[Histogram(np.ones(20), bin_width=0.01)], metadata={})
    grouping = dict(zip(("first_good_bin", "last_good_bin"), window))
    with pytest.raises(ValueError, match="window|index"):
        run_file_good_window(run, grouping, common_t0_bin=0)


def test_grouped_fourier_rejects_a_reversed_background_window():
    from asymmetry.core.fourier.grouped import _apply_group_background_correction

    with pytest.raises(ValueError, match="window.*reversed"):
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
