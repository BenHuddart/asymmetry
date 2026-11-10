"""The existing preview worker consumes acquired constituent preparation."""

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram
from tests.gui.test_grouping_preview_pane import _histogram_dataset, _reduce_preview

pytestmark = pytest.mark.gui


def test_mapped_preview_uses_constituent_correction_before_pooling(qapp):
    from asymmetry.core.io.periods import combine_mapped_periods

    sources = [
        _histogram_dataset(
            forward=np.full(16, f),
            backward=np.full(16, b),
            grouping_extra={
                "good_frames": 100000,
                "dead_time_us": [0.05, 0.05],
                "deadtime_correction": True,
                "deadtime_mode": "file",
            },
        )
        for f, b in [(80000, 20000), (10000, 90000)]
    ]
    for source in sources:
        source.run.histograms = [Histogram(h.counts, 0.1) for h in source.run.histograms]
    mapped = combine_mapped_periods(sources, {1: "red", 2: "red"})
    result = _reduce_preview(mapped, run=mapped.run)
    assert result.time.size == 16
    np.testing.assert_allclose(result.asymmetry, -12.737826507578193)
    assert result.corrections[0].deadtime.status == "applied"
