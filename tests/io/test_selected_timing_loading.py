"""Missing calibration keeps acquired data while restricting calibrated requests."""

import numpy as np
import pytest

from asymmetry.core.io.nexus import NexusLoader
from asymmetry.core.simulate import reduce_run_to_dataset
from asymmetry.core.transform.t0 import find_t0_for_run
from tests.core.test_selected_timing_policy import resolve
from tests.io.test_nexus_good_times import write_good_times

h5py = pytest.importorskip("h5py")
pytestmark = pytest.mark.io


@pytest.mark.parametrize("known", [False, True])
@pytest.mark.parametrize("failed", [False, True])
def test_missing_file_timing_load_and_rereduction(tmp_path, known, failed):
    path = tmp_path / "absent.nxs"
    write_good_times(path)
    raw = np.zeros((2, 32))
    raw[:, 10] = 10000
    raw[:, 11:] = 20
    if failed:
        raw[1] = 0
    with h5py.File(path, "a") as f:
        entry = f["raw_data_1"]
        del entry["definition"]
        entry.create_dataset("definition", data=b"pulsedTD" if known else b"muonTD")
        detector = entry["instrument/detector_1"]
        del detector["time_zero"]
        del detector["counts"]
        detector.create_dataset("counts", data=raw)
    loaded = NexusLoader().load(path)
    assert loaded.run.grouping["t0_source"] == "missing"
    np.testing.assert_array_equal(loaded.run.histograms[1].counts, raw[1])
    if known and not failed:
        assert loaded.n_points == 4
        assert loaded.run.grouping["effective_t0_sources"] == ["estimated"] * 2
        np.testing.assert_allclose(reduce_run_to_dataset(loaded.run).time, loaded.time)
    else:
        assert loaded.n_points == 0
        assert "t0" in loaded.metadata["correction_error"]
        with pytest.raises(ValueError, match="t0"):
            reduce_run_to_dataset(loaded.run)


def test_display_rebin_and_crop_do_not_move_raw_feature():
    from tests.core.test_selected_timing_policy import timing_run

    run = timing_run()
    baseline = resolve(run, pair=(1, 3))
    run.grouping.update(bunching_factor=4, first_good_bin=20, last_good_bin=25)
    changed = resolve(run, pair=(1, 3))
    assert baseline["effective_detector_t0_bins"] == changed["effective_detector_t0_bins"]


def test_failed_pulsed_search_retains_requested_strategy():
    from asymmetry.core.data.dataset import Histogram

    search = find_t0_for_run([Histogram(np.zeros(32), bin_width=0.1)], pulsed=True)
    assert not search.ok
    assert search.strategy == search.estimates[0].strategy == "pulse_edge"


@pytest.mark.parametrize("failed", [False, True])
def test_root_mixed_timing_retains_usable_file_anchor(tmp_path, failed):
    uproot = pytest.importorskip("uproot")
    from asymmetry.core.io.root import RootLoader
    from tests.io.test_root_loader import _write_musrroot_directory

    path = tmp_path / "mixed.root"
    _write_musrroot_directory(path)
    with uproot.open(path) as f:
        fields = {
            key.split(";")[0]: str(f[key])
            for key, kind in f.classnames().items()
            if kind == "TObjString" and "Detector002/Time Zero Bin" not in key
        }
        first = f["histos/DecayAnaModule/hDecay001"].to_numpy()
    with uproot.recreate(path) as f:
        for key, value in fields.items():
            f[key] = value
        counts = np.zeros(7) if failed else np.array([0, 10, 1000, 20, 20, 20, 20])
        f["histos/DecayAnaModule/hDecay001"] = first
        f["histos/DecayAnaModule/hDecay002"] = (counts, np.arange(8, dtype=float))
    loaded = RootLoader().load(path)
    assert loaded.run.grouping["detector_t0_sources"] == ["file", "missing"]
    assert loaded.run.histograms[0].t0_bin == 2
    assert loaded.run.histograms[1].t0_bin == 0
    if failed:
        assert loaded.n_points == 0
        assert "2" in loaded.metadata["correction_error"]
    else:
        assert loaded.n_points > 0
        assert loaded.run.grouping["effective_detector_t0_bins"] == [2, 2]
        assert loaded.run.grouping["effective_t0_sources"] == ["file", "estimated"]
        np.testing.assert_allclose(reduce_run_to_dataset(loaded.run).time, loaded.time)
