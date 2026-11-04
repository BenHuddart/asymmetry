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


@pytest.mark.parametrize("kind", ["psi", "root", "nexus"])
def test_empty_file_good_support_retains_raw_data_for_window_repair(tmp_path, kind):
    if kind == "psi":
        from asymmetry.core.io.psi import PsiLoader
        from tests.io.test_psi_loader import _write_psi_bin

        path = tmp_path / "empty.bin"
        _write_psi_bin(path)
        loader = PsiLoader()
        raw = loader._read_bin(path)
        raw.t0_bins = [0, 3]
        raw.first_good_bins = [0, 0]
        raw.last_good_bins = [1, 1]
        loaded = loader._build_dataset(raw)
    elif kind == "root":
        import uproot

        from asymmetry.core.io.root import RootLoader
        from tests.io.test_root_loader import _write_musrroot_directory

        path = tmp_path / "empty.root"
        _write_musrroot_directory(path)
        with uproot.update(path) as file:
            for i, anchor in ((1, 0), (2, 4)):
                base = f"RunHeader/DetectorInfo/Detector{i:03d}/"
                file[base + "Time Zero Bin"] = str(anchor)
                file[base + "First Good Bin"] = "0"
                file[base + "Last Good Bin"] = "1"
        loaded = RootLoader().load(str(path))
    else:
        import h5py

        from asymmetry.core.io.nexus import NexusLoader
        from tests.io.test_nexus_loader import _write_v2_file

        path = tmp_path / "empty.nxs"
        _write_v2_file(path, first_good_bin_attr=1, last_good_bin_attr=2)
        with h5py.File(path, "r+") as file:
            file["raw_data_1/instrument/detector_1/counts"].attrs["t0_bin"] = np.array([1, 4])
        loaded = NexusLoader().load(str(path))
    assert loaded.time.size == loaded.asymmetry.size == loaded.error.size == 0
    assert "empty good-bin support" in loaded.metadata["correction_error"]
    histograms = loaded.run.histograms
    assert len(histograms) == 2
    assert all(hist.n_bins > 0 for hist in histograms)
    assert [hist.good_bin_end for hist in histograms] == [1, 1]
    assert all(np.any(hist.counts) for hist in histograms)
    # Authored repair replaces the unavailable file intersection without mutating raw facts.
    from asymmetry.core.transform.grouping import group_forward_backward

    repaired = group_forward_backward(histograms, {"groups": {1: [1], 2: [2]}})
    assert repaired.forward.size > 0


def test_quick_file_window_retains_a_genuine_one_bin_intersection():
    from asymmetry.core.data.dataset import Histogram

    histograms = [
        Histogram(np.arange(8.0) + 10.0, 0.1, 0, good_bin_end=0),
        Histogram(np.arange(8.0) + 20.0, 0.1, 3, good_bin_end=3),
    ]
    quick = quick_asymmetry(histograms, {"groups": {1: [1], 2: [2]}})
    assert quick.unavailable_reason is None
    assert quick.window.first == quick.window.last == 0
    np.testing.assert_allclose(quick.time[quick.window.slice], [0.0], atol=1e-15)
    assert [hist.good_bin_end for hist in histograms] == [0, 3]
