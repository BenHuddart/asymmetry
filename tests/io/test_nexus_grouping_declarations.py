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
            declaration = detector.create_dataset("grouping", data=table)
            declaration.attrs["available"] = max(1, np.count_nonzero(np.unique(table) > 0))
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


@pytest.mark.parametrize("layout", ["flat", "table"])
@pytest.mark.parametrize("same", [True, False])
def test_v1_period_grouping_indexes_histograms_without_reusing_first_period(tmp_path, layout, same):
    from asymmetry.core.io import select_period

    path = tmp_path / "periods.nxs"
    raw = grouping_file(path, "v1", table=[1] * 32 + [2] * 32)
    tables = np.array([[1] * 32 + [2] * 32, ([1] * 32 + [2] * 32) if same else [2] * 32 + [1] * 32])
    with h5py.File(path, "r+") as handle:
        entry = handle["run"]
        entry.create_dataset("switching_states", data=2)
        del entry["good_frames"]
        entry.create_dataset("good_frames", data=[100000, 200000])
        detector = entry["histogram_data_1"]
        del detector["counts"], detector["grouping"]
        counts = detector.create_dataset("counts", data=np.concatenate([raw, raw * 2]))
        counts.attrs.update(first_good_bin=5, last_good_bin=20)
        detector.create_dataset("grouping", data=tables.ravel() if layout == "flat" else tables)
    loaded = load(path)
    assert isinstance(loaded, list) is (not same)
    for index in range(2):
        dataset = select_period(loaded, index + 1)
        expected = {gid: (np.flatnonzero(tables[index] == gid) + 1).tolist() for gid in (1, 2)}
        assert dataset.run.grouping["groups"] == expected
        np.testing.assert_allclose(dataset.asymmetry, -50 if same or index == 0 else 50)
        for histogram, counts in zip(dataset.run.histograms, raw * (index + 1), strict=True):
            np.testing.assert_array_equal(histogram.counts, counts)
            assert (histogram.t0_bin, histogram.good_bin_start, histogram.good_bin_end) == (
                2,
                4,
                19,
            )
        assert dataset.run.grouping["deadtime_file_us"] == [0.001] * 64
        assert dataset.run.grouping["reference_exposure"]["value"] == 100000 * (index + 1)
        assert "grouping_preset" not in dataset.run.grouping
        assert load(path, period=index + 1).run.grouping["groups"] == expected


@pytest.mark.parametrize("fault", ["memberships", "legs"])
def test_mapping_requires_common_authored_grouping_and_ignores_unselected_periods(tmp_path, fault):
    from dataclasses import replace

    from asymmetry.core.io.periods import combine_mapped_periods

    path = tmp_path / "run.nxs"
    grouping_file(path, "v2", table=[1] * 32 + [2] * 32)
    first = load(path)
    other = replace(first, run=replace(first.run, grouping=dict(first.run.grouping)))
    if fault == "memberships":
        other.run.grouping["groups"] = {1: list(range(33, 65)), 2: list(range(1, 33))}
    else:
        other.run.grouping.update(forward_group=2, backward_group=1)
    with pytest.raises(ValueError, match="Author a common grouping"):
        combine_mapped_periods([first, other], {1: "red", 2: "green"})
    assert combine_mapped_periods([first, other], {1: "red", 2: "ignore"}).n_points > 0
    other.run.grouping = dict(first.run.grouping)
    other.run.grouping["groups"] = {
        gid: list(reversed(ids)) for gid, ids in first.run.grouping["groups"].items()
    }
    combined = combine_mapped_periods([first, other], {1: "red", 2: "green"})
    np.testing.assert_allclose(combined.asymmetry, -50)


@pytest.mark.parametrize(
    "table",
    [
        [],
        [1, 2, 1],
        [[1, 2], [0, 0]],
        [[1, 2], [-1, -2]],
        [[True, False], [True, False]],
        [[1.0, 2.0], [1.0, 2.0]],
        [[[1, 2]], [[1, 2]]],
    ],
)
def test_invalid_period_table_is_never_broadcast_truncated_or_defaulted(tmp_path, table):
    path = tmp_path / "invalid-periods.nxs"
    raw = grouping_file(path, "v2", table=[1, 2], count=2)
    with h5py.File(path, "r+") as handle:
        detector = handle["raw_data_1/instrument/detector_1"]
        del detector["counts"], detector["grouping"]
        detector.create_dataset("counts", data=np.stack([raw, raw]))
        detector.create_dataset("grouping", data=table)
    with pytest.raises(ValueError, match="Grouping"):
        load(path)


def legacy_zero_grouping_file(path, *, table=(0,) * 64, **options):
    raw = grouping_file(path, "v1", table=table, **options)
    with h5py.File(path, "r+") as handle:
        entry = handle["run"]
        entry.create_dataset("switching_states", data=1)
        entry.create_dataset("instrument/beam/period_type", data=[1])
        entry.create_dataset("instrument/detector/number", data=len(raw))
    return raw


@pytest.mark.parametrize(
    "instrument,count,profile_key,acquisition",
    [("EMU", 96, "isis.emu", "nexus96"), ("HiFi", 64, "isis.hifi", "nexus64")],
)
def test_v1_zero_declaration_applies_verified_longitudinal_preset(
    tmp_path, instrument, count, profile_key, acquisition
):
    path = tmp_path / "legacy.nxs"
    raw = legacy_zero_grouping_file(path, instrument=instrument, count=count, table=[0] * count)
    with h5py.File(path, "r+") as handle:
        handle["run/sample/magnetic_field_state"][()] = b"TF"
    with pytest.warns(
        UserWarning,
        match=f"The file’s detector-grouping table contains only zeros. Applied the standard {instrument} forward/backward grouping.",
    ):
        dataset = load(path)
    assert dataset.metadata["field_direction"] == "Transverse"
    assert dataset.run.grouping["groups"] == {
        1: list(range(1, count // 2 + 1)),
        2: list(range(count // 2 + 1, count + 1)),
    }
    assert (dataset.run.grouping["forward_group"], dataset.run.grouping["backward_group"]) == (1, 2)
    assert dataset.run.grouping["grouping_preset"] == "Longitudinal"
    declaration = dataset.run.grouping["nexus_grouping_declaration"]
    assert declaration == {"values": [0] * count, "available": 1}
    provenance = dataset.run.grouping["grouping_provenance"]
    assert provenance == {
        "origin": "profile",
        "profile_key": profile_key,
        "profile_revision": "2",
        "acquisition": acquisition,
        "fallback_reason": "nexus_v1_all_zero",
    }
    assert dataset.effective_recipe.to_dict()["grouping_provenance"] == provenance
    np.testing.assert_array_equal([h.counts for h in dataset.run.histograms], raw)


@pytest.mark.parametrize(
    "instrument,orientation,count,filename",
    [
        ("Unknown", "L", 96, "emu00000042.nxs"),
        ("DEMUX", "L", 96, "run.nxs"),
        ("EMU", "L", 96, "musr00000042.nxs"),
        ("", "L", 96, "run.nxs"),
        ("EMU", None, 96, "run.nxs"),
        ("EMU", "T", 96, "run.nxs"),
        ("EMU", "LF", 96, "run.nxs"),
        ("EMU", "L", 64, "run.nxs"),
    ],
)
def test_v1_zero_declaration_requires_recognised_compatible_layout(
    tmp_path, instrument, orientation, count, filename
):
    path = tmp_path / filename
    legacy_zero_grouping_file(
        path, instrument=instrument, orientation=orientation, count=count, table=[0] * count
    )
    with pytest.raises(ValueError, match="Grouping declaration contains no positive"):
        load(path)


@pytest.mark.parametrize(
    "version,table,error",
    [
        ("v1", [0] * 63, "Grouping declaration must contain one integer"),
        ("v1", [0.0] * 64, "Grouping declaration must contain one integer"),
        ("v1", [False] * 64, "Grouping declaration must contain one integer"),
        ("v1", [[0] * 64], "Grouping declaration contains no positive"),
        ("v1", [-1] + [0] * 63, "Grouping declaration contains no positive"),
        ("v2", [0] * 64, "Grouping declaration contains no positive"),
    ],
)
def test_only_exact_v1_zero_vector_can_use_legacy_default(tmp_path, version, table, error):
    path = tmp_path / "legacy.nxs"
    if version == "v1":
        legacy_zero_grouping_file(path, instrument="HiFi", table=table)
    else:
        grouping_file(path, version, instrument="HiFi", table=table)
    with pytest.raises(ValueError, match=error):
        load(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("switching_states", None),
        ("switching_states", 3),
        ("switching_states", 1.0),
        ("instrument/beam/period_type", None),
        ("instrument/beam/period_type", [1, 1]),
        ("instrument/detector/number", None),
        ("instrument/detector/number", 65),
        ("instrument/name", [b"HiFi", b"MuSR"]),
        ("instrument/detector/orientation", [b"L", b"T"]),
        ("histogram_data_1/histogram_number", np.arange(1.0, 65.0)),
        ("histogram_data_1/histogram_number", np.arange(64, 0, -1)),
    ],
)
def test_v1_zero_declaration_cannot_override_inconsistent_layout_fields(tmp_path, field, value):
    path = tmp_path / "legacy.nxs"
    legacy_zero_grouping_file(path, instrument="HiFi")
    with h5py.File(path, "r+") as handle:
        entry = handle["run"]
        if field in entry:
            del entry[field]
        if value is not None:
            entry.create_dataset(field, data=value)
    with pytest.raises(ValueError, match="Grouping declaration contains no positive"):
        load(path)


@pytest.mark.parametrize("available", [None, 0, 2, 1.0])
def test_v1_zero_declaration_requires_legacy_availability_attribute(tmp_path, available):
    path = tmp_path / "legacy.nxs"
    legacy_zero_grouping_file(path, instrument="HiFi")
    with h5py.File(path, "r+") as handle:
        attrs = handle["run/histogram_data_1/grouping"].attrs
        del attrs["available"]
        if available is not None:
            attrs["available"] = available
    with pytest.raises(ValueError, match="Grouping declaration contains no positive"):
        load(path)


@pytest.mark.parametrize("period", [None, 1, 2])
def test_v1_zero_declaration_is_rejected_for_a_genuine_two_period_run(tmp_path, period):
    path = tmp_path / "legacy.nxs"
    raw = legacy_zero_grouping_file(path, instrument="HiFi")
    with h5py.File(path, "r+") as handle:
        entry = handle["run"]
        entry["switching_states"][()] = 2
        detector = entry["histogram_data_1"]
        del detector["counts"]
        counts = detector.create_dataset("counts", data=np.concatenate([raw, raw]))
        counts.attrs.update(first_good_bin=5, last_good_bin=20)
    with pytest.raises(ValueError, match="Grouping declaration contains no positive"):
        load(path, period=period)


@pytest.mark.parametrize("table", [None, [2] * 32 + [3] * 32])
def test_v1_legacy_instruments_keep_missing_and_valid_declarations_authoritative(tmp_path, table):
    path = tmp_path / "legacy.nxs"
    grouping_file(path, "v1", instrument="HiFi", table=table)
    dataset = load(path)
    grouping = dataset.run.grouping
    pair = (1, 2) if table is None else (2, 3)
    assert (grouping["forward_group"], grouping["backward_group"]) == pair
    assert grouping["groups"] == {pair[0]: list(range(1, 33)), pair[1]: list(range(33, 65))}
    assert "grouping_preset" not in grouping
    assert "grouping_provenance" not in grouping
    assert "nexus_grouping_warning" not in dataset.metadata


def test_actual_cli_info_warns_on_legacy_zero_grouping(tmp_path, capsys, monkeypatch):
    import json
    import warnings

    from asymmetry import cli

    path = tmp_path / "legacy.nxs"
    legacy_zero_grouping_file(path, instrument="HiFi")
    monkeypatch.setattr(warnings, "showwarning", warnings.showwarning)
    cli.main(["info", str(path), "--json"])
    output = capsys.readouterr()
    assert json.loads(output.out)["default_pair"] == [1, 2]
    assert output.err.count("Applied the standard HiFi forward/backward grouping.") == 1


def test_replay_with_replacement_source_does_not_borrow_old_acquired_declaration(tmp_path):
    from asymmetry.core.project.grouping_request import (
        merge_grouping_request,
        persistent_grouping_for_run,
    )

    saved_source = tmp_path / "legacy.nxs"
    legacy_zero_grouping_file(saved_source, instrument="HiFi")
    with pytest.warns(UserWarning, match="Applied the standard HiFi"):
        saved = load(saved_source).run
    replacement_source = tmp_path / "replacement.nxs"
    grouping_file(replacement_source, "v1", instrument="HiFi", table=[2] * 32 + [3] * 32)
    replacement = load(replacement_source).run
    replayed = merge_grouping_request(replacement.grouping, persistent_grouping_for_run(saved))
    assert "nexus_grouping_declaration" not in replayed
    assert replayed["groups"] == saved.grouping["groups"]
    assert replayed["grouping_provenance"] == saved.grouping["grouping_provenance"]
