"""Manual beta remains independent of calibration staging."""

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.gui.windows.grouping.beta_section import BetaSectionWidget

pytestmark = pytest.mark.gui


def _dataset(*, beta: float | None = None, run_number: int = 4101) -> MuonDataset:
    h1 = Histogram(counts=np.array([100.0, 100.0, 100.0, 100.0]), bin_width=0.01)
    h2 = Histogram(counts=np.array([50.0, 50.0, 50.0, 50.0]), bin_width=0.01)
    grouping: dict = {
        "groups": {1: [1], 2: [2]},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.0,
        "first_good_bin": 0,
        "last_good_bin": 3,
    }
    if beta is not None:
        grouping["beta"] = beta
    run = Run(
        run_number=run_number,
        histograms=[h1, h2],
        metadata={"run_number": run_number, "title": "Beta Test"},
        grouping=grouping,
    )
    t = np.array([0.0, 0.01, 0.02, 0.03])
    return MuonDataset(
        time=t,
        asymmetry=np.zeros_like(t),
        error=np.full_like(t, 0.01),
        metadata={"run_number": run_number},
        run=run,
    )


def test_manual_value_round_trip(qapp):
    section = BetaSectionWidget()
    section.set_value(1.13)
    assert section.is_active() and section.value() == 1.13
    section.set_value(1.0)
    assert not section.is_active()
