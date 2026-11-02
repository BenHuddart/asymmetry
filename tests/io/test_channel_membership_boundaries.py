"""Raw-channel exports and simulations cannot trim authored group membership."""

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.io.nexus_writer import write_nexus_v1
from asymmetry.core.simulate import simulate_run

pytestmark = pytest.mark.io


@pytest.mark.parametrize("consumer", ["writer", "simulation"])
def test_raw_channel_consumer_rejects_unknown_authored_channel(tmp_path, consumer):
    run = Run(
        histograms=[Histogram(np.full(20, level), 0.1) for level in (10.0, 30.0, 50.0)],
        grouping={"groups": {1: [1, 4], 2: [3]}, "first_good_bin": 0, "last_good_bin": 19},
    )
    with pytest.raises(ValueError, match="4"):
        if consumer == "writer":
            write_nexus_v1(run, tmp_path / "invalid.nxs")
        else:
            simulate_run(run, lambda t: np.zeros_like(t), total_events=1000, seed=1)
