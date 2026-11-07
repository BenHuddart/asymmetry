"""Source-bound reference timing survives live edits, disk caches and project replay."""

import copy

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.io import load, resolve_background_reference
from asymmetry.core.io.nexus_writer import write_nexus_v1
from asymmetry.core.project.schema import (
    CURRENT_SCHEMA_VERSION,
    load_project,
    migrate_to_current,
    save_project,
)
from tests.core.test_reference_timing import reduce_reference, timing_runs


def reference_dataset(run):
    return MuonDataset(np.array([]), np.array([]), np.array([]), run=run)


@pytest.mark.parametrize("mode", ["from_file", "manual", "auto_detect", "absolute"])
@pytest.mark.parametrize("consumer", ["fb", "spectral"])
def test_live_disk_and_saved_reference_timing_agree(mode, consumer, tmp_path, monkeypatch):
    sample, authored = timing_runs()
    source = tmp_path / "reference.nxs"
    authored.metadata["instrument"] = "ISIS EMU"
    for hist in authored.histograms:
        hist.counts[6] = 1000
    write_nexus_v1(authored, source)
    live = load(str(source)).run
    if mode == "absolute":
        live.grouping["effective_detector_t0_bins"] = [4, 4]
    else:
        live.grouping["t0_policy"] = {"mode": mode, "offset_bins": 2}
    payload = {"run_number": 2, "source_file": str(source), "period_number": 1}
    loaded = resolve_background_reference(
        payload, datasets=[reference_dataset(live)], allow_file=False
    )
    expected = reduce_reference(sample, loaded, consumer, monkeypatch)
    monkeypatch.setattr(
        "asymmetry.core.io.resolve_background_reference", resolve_background_reference
    )
    project = tmp_path / "timing.asymp"
    save_project(
        {"schema_version": CURRENT_SCHEMA_VERSION, "datasets": [], "background_run": payload},
        project,
    )
    replay = load_project(project)["background_run"]
    cache = {}
    disk = resolve_background_reference(replay, cache=cache)
    np.testing.assert_array_equal(reduce_reference(sample, disk, consumer, monkeypatch), expected)
    assert loaded.timing.request() == disk.timing.request() == payload["timing"]
    assert len(cache) == 1


def test_live_policy_edits_update_request_and_disk_replay_ignores_stale_cache(monkeypatch):
    sample, live = timing_runs()
    raw = copy.deepcopy(live)
    live.grouping["t0_policy"] = {"mode": "manual", "offset_bins": 2}
    payload = {"run_number": 2, "source_file": live.source_file, "period_number": 1}
    monkeypatch.setattr("asymmetry.core.io.load_background_run", lambda _: reference_dataset(raw))
    cache = {}
    resolve_background_reference(payload, cache=cache)
    first = resolve_background_reference(payload, datasets=[reference_dataset(live)], cache=cache)
    assert first.timing.offset_bins == 2
    live.grouping["t0_policy"]["offset_bins"] = 4
    second = resolve_background_reference(payload, datasets=[reference_dataset(live)], cache=cache)
    assert second.timing.offset_bins == 4
    assert payload["timing"] == {"t0_policy": {"mode": "manual", "offset_bins": 4}}
    disk = resolve_background_reference(payload, cache=cache)
    assert disk.timing.request() == second.timing.request()
    np.testing.assert_array_equal(
        reduce_reference(sample, disk, "fb", monkeypatch), 184.0 - np.arange(32)
    )
    assert first.timing.offset_bins == 2
    assert raw.grouping == {}


def test_schema32_reference_never_inherits_sample_timing():
    old = {
        "schema_version": 32,
        "datasets": [],
        "grouping": {
            "t0_policy": {"mode": "manual", "offset_bins": 9},
            "effective_detector_t0_bins": [11, 11],
            "background_run": {"run_number": 2, "source_file": "ref.nxs", "period_number": 1},
        },
    }
    migrated = migrate_to_current(old)
    assert migrated["schema_version"] == CURRENT_SCHEMA_VERSION
    assert migrated["grouping"]["background_run"]["timing"] is None
    assert "timing" not in old["grouping"]["background_run"]
    assert migrate_to_current(migrated) == migrated


@pytest.mark.parametrize(
    "timing_request",
    [
        [],
        {},
        {"t0_policy": {"mode": "invalid"}},
        {"t0_policy": {"mode": "manual"}},
        {"t0_policy": None},
    ],
)
def test_invalid_saved_reference_request_cannot_become_file_timing(timing_request, monkeypatch):
    _, raw = timing_runs()
    monkeypatch.setattr("asymmetry.core.io.load_background_run", lambda _: reference_dataset(raw))
    with pytest.raises(ValueError, match="Reference.*timing|Reference Manual"):
        resolve_background_reference(
            {"run_number": 2, "source_file": raw.source_file, "timing": timing_request}
        )


@pytest.mark.parametrize("identity_mapping", [False, True])
def test_selected_green_reference_keeps_own_manual_timing(identity_mapping, monkeypatch):
    from asymmetry.core.io import select_period
    from tests.io.test_period_selection import _combined_two_period

    combined = _combined_two_period()
    if identity_mapping:
        combined.run.grouping["period_mapping"] = {"1": "red", "2": "green"}
    green = select_period(combined, 2)
    green.run.grouping["t0_policy"] = {"mode": "manual", "offset_bins": 2}
    payload = {"source_file": combined.run.source_file, "run_number": 12345, "period_number": 2}
    live = resolve_background_reference(
        payload, datasets=[green], allow_file=False, sample_good_frames=40
    )
    monkeypatch.setattr("asymmetry.core.io.load", lambda _: combined)
    disk = resolve_background_reference(payload, sample_good_frames=40)
    assert (
        live.timing.request()
        == disk.timing.request()
        == {"t0_policy": {"mode": "manual", "offset_bins": 2}}
    )
    assert live.period_number == disk.period_number == 2
    assert live.good_frames_reference == disk.good_frames_reference == 20
    np.testing.assert_array_equal(live.histograms[0].counts, disk.histograms[0].counts)


def test_advisory_live_timing_refresh_keeps_inactive_and_ambiguous_requests_editable():
    from asymmetry.core.io.reference_selection import refresh_reference_timing_requests

    sample, reference = timing_runs()
    sample.grouping.update(
        background_correction=False,
        background_run={
            "source_file": reference.source_file,
            "run_number": 2,
            "period_number": 1,
            "period_mapping": {"1": "red", "2": "green", "3": "red"},
        },
    )
    datasets = [reference_dataset(run) for run in (sample, reference)]
    assert refresh_reference_timing_requests(datasets) == []
    sample.grouping["background_correction"] = True
    assert refresh_reference_timing_requests(datasets) == []
    sample.grouping["background_run"].pop("period_mapping")
    datasets.append(reference_dataset(copy.deepcopy(reference)))
    assert refresh_reference_timing_requests(datasets) == []
    assert "timing" not in sample.grouping["background_run"]


def test_number_only_live_request_refreshes_timing_without_choosing_ambiguity():
    from asymmetry.core.io.reference_selection import refresh_reference_timing_requests

    sample, reference = timing_runs()
    sample.grouping["background_run"] = {"run_number": 2}
    reference.grouping["t0_policy"] = {"mode": "manual", "offset_bins": 4}
    datasets = [reference_dataset(run) for run in (sample, reference)]
    assert refresh_reference_timing_requests(datasets) == [1]
    assert sample.grouping["background_run"]["timing"] == {
        "t0_policy": {"mode": "manual", "offset_bins": 4}
    }
    reference.grouping["t0_policy"]["offset_bins"] = 6
    datasets.append(reference_dataset(copy.deepcopy(reference)))
    assert refresh_reference_timing_requests(datasets) == []
    assert sample.grouping["background_run"]["timing"]["t0_policy"]["offset_bins"] == 4


def test_number_only_multi_period_reference_stays_advisory_when_period_is_ambiguous():
    from asymmetry.core.io.reference_selection import refresh_reference_timing_requests
    from tests.io.test_period_selection import _combined_two_period

    sample, _ = timing_runs()
    sample.grouping["background_run"] = {"run_number": 12345}
    assert (
        refresh_reference_timing_requests([reference_dataset(sample), _combined_two_period()]) == []
    )


def test_advisory_refresh_preserves_an_invalid_live_timing_request_for_editing():
    from asymmetry.core.io.reference_selection import refresh_reference_timing_requests

    sample, reference = timing_runs()
    sample.grouping["background_run"] = {"run_number": 2}
    reference.grouping["t0_policy"] = {"mode": "manual", "offset_bins": None}
    datasets = [reference_dataset(run) for run in (sample, reference)]
    assert refresh_reference_timing_requests(datasets) == [1]
    assert sample.grouping["background_run"]["timing"] == {
        "t0_policy": reference.grouping["t0_policy"]
    }
    with pytest.raises(ValueError, match="Manual timing"):
        resolve_background_reference(sample.grouping["background_run"], datasets=datasets)


def test_empty_reference_declaration_cannot_capture_the_samples_own_timing():
    from asymmetry.core.io.reference_selection import refresh_reference_timing_requests

    sample, _ = timing_runs()
    sample.grouping["t0_policy"] = {"mode": "manual", "offset_bins": 9}
    sample.grouping["background_run"] = {"source_file": ""}
    assert refresh_reference_timing_requests([reference_dataset(sample)]) == []
    assert sample.grouping["background_run"] == {"source_file": ""}
