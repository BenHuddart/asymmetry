"""Verified synthetic persistence rejects representations that would lose acquisition facts."""

from dataclasses import replace

import numpy as np
import pytest

from asymmetry.core.io.persist_run import persist_histogram_run
from asymmetry.core.simulate import build_builtin_template

pytestmark = pytest.mark.io


@pytest.mark.parametrize(
    "invalid", ["fraction", "negative", "overflow", "nan", "variance", "period"]
)
def test_lossy_acquisition_never_replaces_destination(tmp_path, invalid):
    run = build_builtin_template("ideal_pulsed_fb")
    hist = run.histograms[0]
    if invalid == "variance":
        run.histograms[0] = replace(hist, count_variance=np.ones(hist.n_bins))
    elif invalid == "period":
        run.grouping["period_histograms"] = [run.histograms, run.histograms]
    else:
        hist.counts[0] = {"fraction": 0.5, "negative": -1, "overflow": 2**54, "nan": np.nan}[
            invalid
        ]
    path = tmp_path / "original.nxs"
    path.write_bytes(b"previous file")
    with pytest.raises(ValueError, match="single-period|integer counts"):
        persist_histogram_run(run, path)
    assert path.read_bytes() == b"previous file"
    assert list(tmp_path.iterdir()) == [path]
