"""Invalid timing requests retain raw inspection and project repair."""

import numpy as np
import pytest

from asymmetry.core.io.quick_asymmetry import quick_asymmetry
from asymmetry.core.project.schema import CURRENT_SCHEMA_VERSION, load_project, save_project
from tests.core.test_selected_alignment import tagged_histograms

pytestmark = pytest.mark.io


@pytest.mark.parametrize("override", [[1], [True, 3], [1, 2.5], [None, 3], "1,3", None])
def test_invalid_timing_request_survives_save_reopen_and_repair(tmp_path, override):
    grouping = {"groups": {"1": [1], "2": [2]}, "effective_detector_t0_bins": override}
    path = tmp_path / "timing.asymp"
    state = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "datasets": [
            {
                "run_number": 1,
                "source_file": "sample.nxs",
                "grouping_overrides": grouping,
            }
        ],
    }
    save_project(state, path)
    reopened = load_project(path)["datasets"][0]["grouping_overrides"]
    assert reopened == grouping
    histograms = tagged_histograms()
    unavailable = quick_asymmetry(histograms, reopened)
    assert unavailable.unavailable_reason is not None
    assert unavailable.asymmetry.size == unavailable.time.size == 0
    assert [hist.n_bins for hist in histograms] == [5, 6]
    reopened["effective_detector_t0_bins"] = [1, 3]
    repaired = quick_asymmetry(histograms, reopened)
    assert repaired.unavailable_reason is None
    np.testing.assert_allclose(repaired.time, [-0.12, -0.02, 0.08, 0.18], atol=1e-15)
    assert repaired.metadata["grouped_alignment"]["first_bin"] == 2


@pytest.mark.parametrize("kind", ["root", "nexus"])
def test_loader_keeps_raw_histograms_when_default_pair_is_unavailable(tmp_path, monkeypatch, kind):
    from asymmetry.core.io.nexus import NexusLoader, _GroupingSelection
    from asymmetry.core.io.root import RootLoader
    from tests.io.test_nexus_loader import _write_v2_file
    from tests.io.test_root_loader import _write_musrroot_directory

    if kind == "root":
        path = tmp_path / "shared.root"
        _write_musrroot_directory(path)
        loader = RootLoader()
        monkeypatch.setattr(
            loader, "_default_groups", lambda *a, **kw: ({1: [1], 2: [1]}, {}, 1, 2)
        )
    else:
        path = tmp_path / "shared.nxs"
        _write_v2_file(path)
        loader = NexusLoader()
        monkeypatch.setattr(
            loader,
            "_resolve_grouping",
            lambda *a: _GroupingSelection([0], [0], {1: [0], 2: [0]}, 1, 2),
        )
    loaded = loader.load(str(path))
    assert loaded.time.size == loaded.asymmetry.size == 0
    assert "shared acquired channels" in loaded.metadata["correction_error"]
    assert len(loaded.run.histograms) == 2
    assert all(hist.n_bins > 0 for hist in loaded.run.histograms)
    repaired = quick_asymmetry(
        loaded.run.histograms, dict(loaded.run.grouping, groups={1: [1], 2: [2]})
    )
    assert repaired.unavailable_reason is None
