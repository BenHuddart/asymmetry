"""Captured reference source and legacy omission stay distinct through real replay."""

import copy

import pytest

from asymmetry.core.io import resolve_background_reference
from asymmetry.core.io.reference_selection import refresh_reference_timing_requests
from asymmetry.core.transform.reference_timing import ReferenceTiming
from tests.core.test_reference_timing import timing_runs
from tests.io.test_reference_timing_replay import reference_dataset


@pytest.mark.parametrize("pulsed", [True, False, None])
def test_reference_source_capture_roundtrips_despite_contrary_metadata(pulsed):
    _, reference = timing_runs()
    metadata = {} if pulsed is None else {"source_pulsed": pulsed}
    timing = ReferenceTiming.from_grouping(reference.histograms, reference.grouping, metadata)
    request = timing.request()
    assert request["source"]["pulsed"] is pulsed
    assert request["source"]["origin"]
    replay = ReferenceTiming.from_grouping(
        reference.histograms, request, {"source_pulsed": pulsed is not True}
    )
    assert replay.request() == request
    assert replay.pulsed is pulsed


def test_saved_unknown_reference_source_does_not_enable_automatic_search(monkeypatch):
    _, reference = timing_runs()
    reference.grouping["t0_policy"] = {"mode": "auto_detect"}
    captured = ReferenceTiming.from_grouping(reference.histograms, reference.grouping, {}).request()
    reference.metadata["source_pulsed"] = True
    payload = {"run_number": 2, "source_file": reference.source_file, "timing": captured}
    monkeypatch.setattr(
        "asymmetry.core.io.load_background_run", lambda _: reference_dataset(reference)
    )
    resolved = resolve_background_reference(payload)
    assert resolved.timing.pulsed is None
    assert resolved.timing.request() == captured
    with pytest.raises(ValueError, match="source|strategy"):
        resolved.timing.resolve(reference.histograms, [[0], [1]])


@pytest.mark.parametrize("pulsed", [True, False, None])
def test_legacy_reference_source_omission_uses_only_supported_metadata(pulsed, monkeypatch):
    _, reference = timing_runs()
    reference.metadata = {} if pulsed is None else {"source_pulsed": pulsed}
    payload = {
        "run_number": 2,
        "source_file": reference.source_file,
        "timing": {"t0_policy": {"mode": "from_file"}},
    }
    monkeypatch.setattr(
        "asymmetry.core.io.load_background_run", lambda _: reference_dataset(reference)
    )
    resolved = resolve_background_reference(payload)
    assert resolved.timing.pulsed is pulsed
    assert resolved.timing.request()["source"]["origin"].startswith("legacy")
    assert resolved.timing.request()["source"]["pulsed"] is pulsed


def test_unchanged_live_refresh_preserves_unknown_but_explicit_edit_recaptures_source():
    sample, reference = timing_runs()
    captured = ReferenceTiming.from_grouping(reference.histograms, reference.grouping, {}).request()
    sample.grouping["background_run"] = {"run_number": 2, "timing": copy.deepcopy(captured)}
    datasets = [reference_dataset(run) for run in (sample, reference)]
    reference.metadata["source_pulsed"] = True
    assert refresh_reference_timing_requests(datasets) == []
    assert sample.grouping["background_run"]["timing"] == captured
    assert refresh_reference_timing_requests(datasets, changed_run_numbers=[2]) == [1]
    updated = sample.grouping["background_run"]["timing"]
    assert updated["source"]["pulsed"] is True
    assert updated["t0_policy"] == captured["t0_policy"]


def test_live_timing_edit_recaptures_source_through_explicit_refresh():
    sample, reference = timing_runs()
    captured = ReferenceTiming.from_grouping(reference.histograms, reference.grouping, {}).request()
    payload = {"run_number": 2, "timing": copy.deepcopy(captured)}
    sample.grouping["background_run"] = payload
    reference.grouping["t0_policy"] = {"mode": "manual", "offset_bins": 2}
    reference.metadata["source_pulsed"] = False
    datasets = [reference_dataset(run) for run in (sample, reference)]
    assert refresh_reference_timing_requests(datasets, changed_run_numbers=[2]) == [1]
    resolved = resolve_background_reference(payload, datasets=datasets, allow_file=False)
    assert resolved.timing.offset_bins == 2
    assert resolved.timing.request()["source"]["pulsed"] is False
    assert captured["source"]["pulsed"] is None
    assert captured["t0_policy"] == {"mode": "from_file"}


@pytest.mark.parametrize("pulsed", [True, False])
def test_captured_profile_source_revision_survives_current_metadata(pulsed):
    from asymmetry.core.instruments.source import SourceAssessment

    _, reference = timing_runs()
    captured = SourceAssessment(pulsed, "known", "profile", "test.instrument", "authored")
    timing = ReferenceTiming.from_grouping(
        reference.histograms, reference.grouping, {}, source=captured
    )
    request = timing.request()
    replay = ReferenceTiming.from_grouping(
        reference.histograms, request, {"source_pulsed": not pulsed}
    )
    assert replay.source == captured
    assert replay.request() == request


@pytest.mark.parametrize(
    "invalid", [None, False, {}, {"pulsed": None, "reason": "known", "origin": "metadata"}]
)
def test_malformed_captured_source_never_becomes_legacy_inference(invalid, monkeypatch):
    _, reference = timing_runs()
    reference.metadata["source_pulsed"] = True
    payload = {
        "run_number": 2,
        "source_file": reference.source_file,
        "timing": {"t0_policy": {"mode": "from_file"}, "source": invalid},
    }
    monkeypatch.setattr(
        "asymmetry.core.io.load_background_run", lambda _: reference_dataset(reference)
    )
    with pytest.raises(ValueError, match="source|Source"):
        resolve_background_reference(payload)
    assert payload["timing"]["source"] == invalid


def test_project_request_roundtrip_keeps_captured_unknown_source(tmp_path, monkeypatch):
    from asymmetry.core.project.schema import CURRENT_SCHEMA_VERSION, load_project, save_project

    _, reference = timing_runs()
    captured = ReferenceTiming.from_grouping(reference.histograms, reference.grouping, {}).request()
    payload = {"run_number": 2, "source_file": reference.source_file, "timing": captured}
    project = tmp_path / "reference-source.asymp"
    save_project(
        {"schema_version": CURRENT_SCHEMA_VERSION, "datasets": [], "background_run": payload},
        project,
    )
    restored = load_project(project)["background_run"]
    reference.metadata["source_pulsed"] = True
    monkeypatch.setattr(
        "asymmetry.core.io.load_background_run", lambda _: reference_dataset(reference)
    )
    assert resolve_background_reference(restored).timing.request() == captured
    assert restored["timing"] == captured


def test_live_reference_resolution_preserves_source_during_ordinary_policy_refresh():
    _, reference = timing_runs()
    captured = ReferenceTiming.from_grouping(reference.histograms, reference.grouping, {}).request()
    reference.grouping["t0_policy"] = {"mode": "manual", "offset_bins": 2}
    reference.metadata["source_pulsed"] = True
    payload = {"run_number": 2, "timing": captured}
    resolved = resolve_background_reference(
        payload, datasets=[reference_dataset(reference)], allow_file=False
    )
    assert resolved.timing.offset_bins == 2
    assert resolved.timing.pulsed is None
    assert resolved.timing.request()["source"] == captured["source"]


def test_explicit_live_refresh_uses_recorded_choice_not_new_raw_hints():
    from asymmetry.core.instruments.source import SourceAssessment

    sample, reference = timing_runs()
    captured = ReferenceTiming.from_grouping(reference.histograms, reference.grouping, {}).request()
    sample.grouping["background_run"] = {"run_number": 2, "timing": captured}
    choice = SourceAssessment(None, "unsupported", "manual")
    reference.source_assessment = choice
    reference.metadata["source_pulsed"] = True
    datasets = [reference_dataset(run) for run in (sample, reference)]
    assert refresh_reference_timing_requests(datasets, changed_run_numbers=[2]) == [1]
    assert sample.grouping["background_run"]["timing"]["source"] == choice.to_dict()


@pytest.mark.parametrize("raw_source", [False, None])
def test_legacy_omission_ignores_current_profile_source_on_disk_and_refresh(
    raw_source, monkeypatch
):
    from asymmetry.core.instruments.source import SourceAssessment

    sample, reference = timing_runs()
    reference.source_assessment = SourceAssessment(
        True, "known", "profile", "test.instrument", "new"
    )
    reference.metadata = {} if raw_source is None else {"source_pulsed": raw_source}
    payload = {
        "run_number": 2,
        "source_file": reference.source_file,
        "timing": {"t0_policy": {"mode": "from_file"}},
    }
    monkeypatch.setattr(
        "asymmetry.core.io.load_background_run", lambda _: reference_dataset(reference)
    )
    resolved = resolve_background_reference(copy.deepcopy(payload))
    assert resolved.timing.pulsed is raw_source
    assert resolved.timing.source.origin.startswith("legacy:")
    sample.grouping["background_run"] = payload
    datasets = [reference_dataset(run) for run in (sample, reference)]
    assert refresh_reference_timing_requests(datasets) == [1]
    assert payload["timing"]["source"]["pulsed"] is raw_source
    assert payload["timing"]["source"]["origin"].startswith("legacy:")
