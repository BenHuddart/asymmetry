"""Grouping timing workers consume captured source records without raw re-inference."""

import threading

import h5py
import numpy as np
import pytest

from asymmetry.core.instruments.source import SourceAssessment
from asymmetry.core.io import load
from asymmetry.core.io.nexus import NexusLoader
from asymmetry.core.io.reference_selection import resolve_loaded_reference
from asymmetry.core.io.run_preparation import ReplayState
from asymmetry.gui.windows.grouping import timing_section
from asymmetry.gui.windows.grouping.dialog import GroupingDialog
from tests.gui.test_grouping_preview_pane import _wait_until
from tests.io.test_load_source_defaults import source_file

pytestmark = pytest.mark.gui


@pytest.mark.parametrize("captured", [None, SourceAssessment(None, "absent", "project")])
def test_actual_grouping_timing_worker_uses_effective_source(qapp, tmp_path, monkeypatch, captured):
    path = tmp_path / "run.nxs"
    source_file(path, file_anchor=2)
    dataset = load(path, replay=ReplayState(source=captured) if captured is not None else None)
    expected = dataset.run.source_assessment
    dataset.run.metadata["facility"] = "ISIS"
    observed = []
    compute = timing_section.find_t0_for_run

    def search(histograms, metadata, **options):
        observed.append((metadata.get("source_assessment"), threading.current_thread()))
        return compute(histograms, metadata, **options)

    monkeypatch.setattr(timing_section, "find_t0_for_run", search)
    dialog = GroupingDialog([dataset])
    try:
        _wait_until(lambda: dialog._t0_cache_key() in dialog._t0_search_cache)
        assert observed == [(expected, observed[0][1])]
        assert observed[0][1] is not threading.current_thread()
        result = dialog._t0_search_cache[dialog._t0_cache_key()]
        assert result.ok is (captured is None)
        key = dialog._t0_cache_key()
        dataset.run.source_assessment = SourceAssessment(
            expected.pulsed, expected.reason, "changed"
        )
        assert dialog._t0_cache_key() != key
    finally:
        dialog._clear_dirty()
        dialog.close()


def test_green_timing_preview_and_live_reference_use_the_acquired_owner(qapp, tmp_path):
    path = tmp_path / "run.nxs"
    raw = source_file(path, file_anchor=2)
    with h5py.File(path, "r+") as handle:
        detector = handle["raw_data_1/instrument/detector_1"]
        del detector["counts"]
        detector.create_dataset("counts", data=np.stack([raw, raw]))
    captured = SourceAssessment(False, "known", "project")
    dataset = NexusLoader().load(
        path, replay=ReplayState(source=captured, run_number=42, period_number=2)
    )
    assert dataset.run.source_assessment != captured
    selected = resolve_loaded_reference(
        [dataset.run], {"source_file": str(path), "run_number": 42, "period_number": 2}
    )
    assert selected.source_assessment == captured
    dialog = GroupingDialog([dataset])
    try:
        assert dialog._t0_preview_run("green").source_assessment == captured
        assert dialog._t0_preview_run("red").source_assessment != captured
        assert dialog._t0_cache_key("green") != dialog._t0_cache_key("red")
    finally:
        dialog._clear_dirty()
        dialog.close()


def test_single_group_editor_does_not_author_a_pair_until_explicit_repair(qapp, tmp_path):
    from tests.io.test_nexus_grouping_declarations import grouping_file

    path = tmp_path / "run.nxs"
    grouping_file(path, "v2", table=[7] * 64)
    dataset = load(path)
    dialog = GroupingDialog([dataset])
    try:
        assert dialog._forward_combo.currentData() is None
        assert dialog._backward_combo.currentData() is None
        payload = dialog._current_grouping_payload()
        assert "forward_group" not in payload
        assert "backward_group" not in payload
        assert not dialog._apply_btn.isEnabled()
        index = dialog._preset_combo.findData("Longitudinal")
        dialog._on_preset_combo_activated(index)
        payload = dialog._current_grouping_payload()
        assert (payload["forward_group"], payload["backward_group"]) == (2, 1)
        assert dataset.run.grouping["groups"] == {7: list(range(1, 65))}
    finally:
        dialog._clear_dirty()
        dialog.close()


@pytest.mark.parametrize("raw", [True, False])
@pytest.mark.parametrize("change", ["unchanged", "pair", "memberships"])
def test_actual_apply_keeps_default_attribution_only_for_unchanged_request(
    qapp, tmp_path, raw, change
):
    from dataclasses import replace

    from asymmetry.core.project.calibration import PreparedGrouping
    from asymmetry.core.project.profiles import (
        profile_fingerprint_for_run,
        profile_from_payload,
        resolve_effective_grouping,
    )
    from asymmetry.gui.mainwindow import MainWindow
    from tests.io.test_nexus_grouping_declarations import grouping_file

    path = tmp_path / "run.nxs"
    grouping_file(path, "v2")
    dataset = load(path)
    provenance = dict(dataset.run.grouping["grouping_provenance"], profile_revision="historical")
    dataset.run.grouping["grouping_provenance"] = provenance
    profile = profile_from_payload(
        dataset.run.grouping, "default", profile_fingerprint_for_run(dataset.run)
    )
    if change == "pair":
        profile = replace(profile, forward_group=1, backward_group=2)
    elif change == "memberships":
        profile = replace(profile, groups={1: [1], 2: [64]})
    payload = resolve_effective_grouping(profile, dataset.run)
    acquired = [hist.counts.copy() for hist in dataset.run.histograms]
    if not raw:
        dataset.run.histograms = []
        dataset.run.instrument_choice = None
        dataset.run.metadata = {}
    window = MainWindow()
    try:
        assert window._apply_grouping_settings_to_dataset(dataset, PreparedGrouping(payload))[0]
        if change == "unchanged":
            assert dataset.run.grouping["grouping_provenance"] == provenance
        else:
            assert "grouping_provenance" not in dataset.run.grouping
        if raw:
            np.testing.assert_allclose(
                dataset.asymmetry, -50.0 if change == "pair" else 50.0, atol=1e-12, rtol=0
            )
            recipe = dataset.effective_recipe.to_dict()["periods"]["red"]
            assert ("grouping_provenance" in recipe) is (change == "unchanged")
            for hist, counts in zip(dataset.run.histograms, acquired, strict=True):
                np.testing.assert_array_equal(hist.counts, counts)
    finally:
        window._clear_dirty()
        window.close()
