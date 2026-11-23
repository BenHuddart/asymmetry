"""File grouping presence controls defaults without changing acquired detector facts."""

import h5py
import numpy as np
import pytest

from asymmetry.core.io import load
from asymmetry.core.transform.grouping import group_counts_aligned
from asymmetry.core.workflow import apply_grouping_preset

pytestmark = pytest.mark.io


def grouping_file(path, version, *, table=None, orientation="L", instrument="MuSR", count=64):
    raw = np.vstack([np.full(24, 1000.0)] * (count // 2) + [np.full(24, 3000.0)] * (count // 2))
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("run" if version == "v1" else "raw_data_1")
        entry.create_dataset("analysis" if version == "v1" else "definition", data=b"muonTD")
        entry.create_dataset("number" if version == "v1" else "run_number", data=42)
        instrument_node = entry.create_group("instrument")
        instrument_node.create_dataset("name", data=instrument.encode())
        if version == "v1":
            detector = entry.create_group("histogram_data_1")
            orientation_node = instrument_node.create_group("detector")
        else:
            detector = instrument_node.create_group("detector_1")
            orientation_node = detector
        if orientation is not None:
            orientation_node.create_dataset("orientation", data=orientation.encode())
        entry.create_group("sample").create_dataset("magnetic_field_state", data=b"LF")
        entry.create_dataset("good_frames", data=100000)
        counts = detector.create_dataset("counts", data=raw)
        if version == "v1":
            counts.attrs["first_good_bin"] = 5
            counts.attrs["last_good_bin"] = 20
        if table is not None:
            detector.create_dataset("grouping", data=table)
        detector.create_dataset("resolution", data=20).attrs["units"] = "ns"
        detector.create_dataset("raw_time", data=np.arange(25) * 0.02).attrs["units"] = "us"
        detector.create_dataset("time_zero", data=0.04).attrs["units"] = "us"
        detector.create_dataset("first_good_time", data=0.08).attrs["units"] = "us"
        detector.create_dataset("last_good_time", data=0.4).attrs["units"] = "us"
        detector.create_dataset("dead_time", data=np.full(count, 0.001))
    return raw


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_absent_grouping_uses_only_the_documented_longitudinal_preset(tmp_path, version):
    path = tmp_path / "run.nxs"
    grouping_file(path, version)
    dataset = load(path)
    assert dataset.run.grouping["groups"] == {1: list(range(1, 33)), 2: list(range(33, 65))}
    assert (dataset.run.grouping["forward_group"], dataset.run.grouping["backward_group"]) == (2, 1)
    assert dataset.run.grouping["grouping_preset"] == "Longitudinal"
    provenance = {
        "origin": "profile",
        "profile_key": "isis.musr",
        "profile_revision": "2",
        "acquisition": "nexus64",
    }
    assert dataset.run.grouping["grouping_provenance"] == provenance
    assert dataset.effective_recipe.to_dict()["grouping_provenance"] == provenance
    # Manual longitudinal memberships give F96000 and B32000 counts per bin.
    np.testing.assert_allclose(dataset.asymmetry, 50.0, rtol=0, atol=1e-12)


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_present_opposite_pair_preserves_authored_sign(tmp_path, version):
    path = tmp_path / "run.nxs"
    grouping_file(path, version, table=[1] * 32 + [2] * 32)
    dataset = load(path)
    assert (dataset.run.grouping["forward_group"], dataset.run.grouping["backward_group"]) == (1, 2)
    np.testing.assert_allclose(dataset.asymmetry, -50.0, rtol=0, atol=1e-12)
    assert "grouping_preset" not in dataset.run.grouping


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_present_single_group_keeps_raw_windows_and_supports_explicit_preset_repair(
    tmp_path, version
):
    path = tmp_path / "run.nxs"
    raw = grouping_file(path, version, table=[7] * 64)
    dataset = load(path)
    assert dataset.run.grouping["groups"] == {7: list(range(1, 65))}
    assert dataset.n_points == 0
    assert "correction_error" in dataset.metadata
    assert (
        dataset.run.grouping.get("forward_group") != dataset.run.grouping.get("backward_group")
        or "forward_group" not in dataset.run.grouping
    )
    for histogram, counts in zip(dataset.run.histograms, raw, strict=True):
        np.testing.assert_array_equal(histogram.counts, counts)
        assert (histogram.t0_bin, histogram.good_bin_start, histogram.good_bin_end) == (2, 4, 19)
    assert dataset.run.grouping["deadtime_file_us"] == [0.001] * 64
    assert dataset.run.grouping["good_frames"] == 100000
    assert dataset.run.grouping["reference_exposure"]["value"] == 100000
    summed = group_counts_aligned(dataset.run.histograms, list(range(64)))
    np.testing.assert_array_equal(summed.counts, np.full(24, 128000.0))
    repaired = apply_grouping_preset(dataset, "Longitudinal")
    np.testing.assert_allclose(repaired.asymmetry, 50.0, rtol=0, atol=1e-12)
    assert (repaired.run.grouping["first_good_bin"], repaired.run.grouping["last_good_bin"]) == (
        4,
        19,
    )
    assert dataset.run.grouping["groups"] == {7: list(range(1, 65))}


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("table", [[], [1, 2], [0] * 64])
def test_present_invalid_table_is_not_reclassified_as_absence(tmp_path, version, table):
    path = tmp_path / "run.nxs"
    grouping_file(path, version, table=table)
    with pytest.raises(ValueError, match="[Gg]rouping"):
        load(path)


@pytest.mark.parametrize("period", [None, 2])
def test_two_period_single_group_preserves_window_through_explicit_repair(tmp_path, period):
    path = tmp_path / "run.nxs"
    raw = grouping_file(path, "v2", table=[7] * 64)
    with h5py.File(path, "r+") as handle:
        detector = handle["raw_data_1/instrument/detector_1"]
        del detector["counts"]
        detector.create_dataset("counts", data=np.stack([raw, raw]))
    dataset = load(path, period=period)
    assert dataset.n_points == 0
    assert (dataset.run.grouping["first_good_bin"], dataset.run.grouping["last_good_bin"]) == (
        4,
        19,
    )
    assert dataset.run.grouping["deadtime_file_us"] == [0.001] * 64
    for histogram, counts in zip(dataset.run.histograms, raw, strict=True):
        np.testing.assert_array_equal(histogram.counts, counts)
        assert (histogram.good_bin_start, histogram.good_bin_end) == (4, 19)
    repaired = apply_grouping_preset(dataset, "Longitudinal")
    assert repaired.n_points == 16
    np.testing.assert_allclose(repaired.asymmetry, 50.0, rtol=0, atol=1e-12)
    assert (repaired.run.grouping["first_good_bin"], repaired.run.grouping["last_good_bin"]) == (
        4,
        19,
    )


def test_single_group_disjoint_acquired_windows_remain_loadable(tmp_path):
    path = tmp_path / "run.nxs"
    raw = grouping_file(path, "v2", table=[7] * 64)
    with h5py.File(path, "r+") as handle:
        detector = handle["raw_data_1/instrument/detector_1"]
        for name, values in (
            ("first_good_time", [0.08] * 32 + [0.4] * 32),
            ("last_good_time", [0.2] * 32 + [0.46] * 32),
        ):
            del detector[name]
            detector.create_dataset(name, data=values).attrs["units"] = "us"
    dataset = load(path)
    assert dataset.n_points == 0
    assert dataset.run.grouping["groups"] == {7: list(range(1, 65))}
    for index, (histogram, counts) in enumerate(zip(dataset.run.histograms, raw, strict=True)):
        np.testing.assert_array_equal(histogram.counts, counts)
        assert (histogram.good_bin_start, histogram.good_bin_end) == (
            (4, 9) if index < 32 else (20, 22)
        )
    assert dataset.run.grouping["first_good_bin"] > dataset.run.grouping["last_good_bin"]
    assert "correction_error" in dataset.metadata


@pytest.mark.parametrize(
    "orientation,instrument,count",
    [(None, "MuSR", 64), ("T", "MuSR", 64), ("L", "Unknown", 64), ("L", "MuSR", 96)],
)
def test_field_state_and_count_do_not_authorize_longitudinal_default(
    tmp_path, orientation, instrument, count
):
    path = tmp_path / "run.nxs"
    grouping_file(path, "v2", orientation=orientation, instrument=instrument, count=count)
    dataset = load(path)
    assert (dataset.run.grouping["forward_group"], dataset.run.grouping["backward_group"]) == (1, 2)
    assert "grouping_preset" not in dataset.run.grouping
    np.testing.assert_allclose(dataset.asymmetry, -50.0, rtol=0, atol=1e-12)


@pytest.mark.parametrize("owner", ["ordinary", "period", "historical", "combined"])
def test_default_attribution_follows_unchanged_request_through_actual_reduction(tmp_path, owner):
    from dataclasses import replace

    from asymmetry.core.project.profiles import (
        profile_fingerprint_for_run,
        profile_from_payload,
        resolve_effective_grouping,
    )
    from asymmetry.core.workflow.reduction import ReductionSettings, reduce_run

    path = tmp_path / "run.nxs"
    grouping_file(path, "v2")
    if owner == "period":
        with h5py.File(path, "r+") as handle:
            detector = handle["raw_data_1/instrument/detector_1"]
            raw = detector["counts"][()]
            del detector["counts"]
            detector.create_dataset("counts", data=np.stack([raw, raw]))
    dataset = load(path, period=2 if owner == "period" else None)
    if owner == "historical":
        dataset.run.grouping["grouping_provenance"]["profile_revision"] = "historical"
    elif owner == "combined":
        from asymmetry.core.data.combine import combine_runs

        other = tmp_path / "other.nxs"
        grouping_file(other, "v2")
        with h5py.File(other, "r+") as handle:
            handle["raw_data_1/run_number"][()] = 43
        dataset.run = combine_runs([dataset.run, load(other).run])
    provenance = dataset.run.grouping["grouping_provenance"]
    reduced = reduce_run(dataset.run, ReductionSettings())
    period = "green" if owner == "period" else "red"
    assert (
        reduced.effective_recipe.to_dict()["periods"][period]["grouping_provenance"] == provenance
    )
    np.testing.assert_allclose(reduced.asymmetry, 50.0, rtol=0, atol=1e-12)
    profile = profile_from_payload(
        dataset.run.grouping, "default", profile_fingerprint_for_run(dataset.run)
    )
    changed = replace(profile, forward_group=1, backward_group=2)
    assert "grouping_provenance" not in resolve_effective_grouping(changed, dataset.run)
    changed = replace(profile, groups={1: [1], 2: [64]})
    assert "grouping_provenance" not in resolve_effective_grouping(changed, dataset.run)


@pytest.mark.parametrize("json_output", [True, False])
def test_actual_cli_info_keeps_single_group_inspectable(tmp_path, capsys, json_output):
    import json

    from asymmetry import cli

    path = tmp_path / "run.nxs"
    grouping_file(path, "v2", table=[7] * 64)
    cli.main(["info", str(path), *(["--json"] if json_output else [])])
    output = capsys.readouterr().out
    if json_output:
        report = json.loads(output)
        assert report["groups"] == {"7": "Group 7"}
        assert report["default_pair"] is None
        assert report["n_points"] == 0
    else:
        assert "7 Group 7" in output
        assert "independent pair unavailable" in output
