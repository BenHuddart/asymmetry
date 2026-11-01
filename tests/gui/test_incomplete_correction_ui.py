"""Incomplete correction requests remain visible and cannot become final results."""

from copy import deepcopy

import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from asymmetry.core.io import resolve_background_reference
from asymmetry.core.transform.corrections import CorrectionUnavailableError
from asymmetry.gui.mainwindow import MainWindow
from asymmetry.gui.panels.data_browser import _BADGE_ROLE
from asymmetry.gui.windows.grouping.dialog import GroupingDialog
from tests._qt_helpers import wait_for
from tests.gui.test_grouping_preview_pane import _facts, _histogram_dataset
from tests.gui.test_mainwindow_additional import _make_two_period_vector_dataset

pytestmark = pytest.mark.gui


def test_missing_preview_reference_blocks_apply(qapp, monkeypatch):
    dataset = _histogram_dataset()
    dialog = GroupingDialog([dataset])
    pane = dialog._preview_pane
    grouping = dict(
        dataset.run.grouping,
        background_correction=True,
        background_mode="reference_run",
        background_run={"run_number": 99},
    )
    pane.request_preview(histograms=dataset.run.histograms, grouping=grouping, facts=_facts())
    pane.flush()
    wait_for(lambda: pane._last_result is not None and not pane.corrections_complete, qapp)
    assert pane._incomplete_banner.text() == (
        "Incomplete preview — not applied: background (reference run is not loaded)"
    )
    assert not pane._incomplete_banner.isHidden()
    assert not dialog._apply_btn.isEnabled()
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[1]))
    dialog._on_apply()
    assert warnings == ["Corrections incomplete"]
    dialog._clear_dirty()
    dialog.close()


def test_preview_uses_loaded_reference_and_live_exposure(qapp, monkeypatch):
    import asymmetry.core.io as io
    from asymmetry.gui.windows.grouping.preview_pane import GroupingPreviewPane

    sample = _histogram_dataset(grouping_extra={"good_frames": 20})
    reference = _histogram_dataset(
        run_number=99,
        forward=np.full(6, 2.0),
        backward=np.full(6, 3.0),
        grouping_extra={"good_frames": 10},
    )
    payload = {"run_number": 99, "source_file": "unreadable.nxs"}
    monkeypatch.setattr(io, "load_background_run", lambda _: pytest.fail("preview read disk"))
    assert resolve_background_reference(payload, allow_file=False) is None
    pane = GroupingPreviewPane()
    try:
        grouping = dict(
            sample.run.grouping,
            background_correction=True,
            background_mode="reference_run",
            background_run=payload,
        )
        for frames in (10, 5):
            reference.run.grouping["good_frames"] = frames
            resolved = resolve_background_reference(
                payload, datasets=[reference], sample_good_frames=20, allow_file=False
            )
            old = pane._last_result
            pane.request_preview(
                histograms=sample.run.histograms,
                grouping=grouping,
                facts=_facts(),
                resolved_reference=resolved,
            )
            pane.flush()
            wait_for(lambda: pane._last_result is not None and pane._last_result is not old, qapp)
            result = pane._last_result
            assert result.corrections.complete
            effective = result.corrections.background.effective
            assert effective["details"]["scale"] == 20 / frames
            np.testing.assert_allclose(
                result.counts.forward, sample.run.histograms[0].counts - 40 / frames
            )
    finally:
        pane.shutdown()


def test_missing_reference_restore_keeps_request_and_badge(qapp, tmp_path, monkeypatch):
    dataset = _histogram_dataset()
    (tmp_path / "sample.nxs").touch()
    dataset.run.source_file = str(tmp_path / "sample.nxs")
    dataset.metadata["source_file"] = dataset.run.source_file
    dataset.run.grouping.update(
        background_correction=True,
        background_mode="reference_run",
        background_run={"run_number": 99},
    )
    dataset.metadata["grouping_overrides"] = True
    window = MainWindow()
    window._data_browser.add_dataset(dataset)
    state = window.collect_project_state()
    fresh = deepcopy(dataset)
    restored = MainWindow()
    monkeypatch.setattr(restored, "_load_file", lambda _: fresh)
    restored.restore_project_state(state, str(tmp_path / "project.asymp"))
    assert fresh.run.grouping["background_run"] == {"run_number": 99}
    assert "reference" in fresh.metadata["correction_error"]
    table = restored._data_browser._table
    assert any(
        table.item(row, 1).data(_BADGE_ROLE) == "Corrections incomplete"
        for row in range(table.rowCount())
    )
    assert restored._get_fit_dataset(fresh) is None
    applied, outcomes = restored._apply_grouping_settings_to_dataset(
        fresh, dict(fresh.run.grouping, background_correction=False, background_mode="none")
    )
    assert applied and all(outcome.complete for outcome in outcomes.values())
    assert "correction_error" not in fresh.metadata


def test_period_outcomes_keep_distinct_background_levels(qapp):
    dataset = _make_two_period_vector_dataset(42)
    dataset.metadata["facility"] = "PSI"
    dataset.run.metadata["facility"] = "PSI"
    window = MainWindow()
    payload = dict(
        dataset.run.grouping,
        period_mode="green_plus_red",
        background_correction=True,
        background_mode="range",
        background_range=[0, 1],
    )
    applied, outcomes = window._apply_grouping_settings_to_dataset(dataset, payload)
    assert applied
    assert set(outcomes) == {"red", "green"}
    assert outcomes["red"].background.effective != outcomes["green"].background.effective
    assert set(dataset.run.grouping["background_details"]["periods"]) == {"red", "green"}
    assert "background_values" not in dataset.run.grouping


def test_one_incomplete_period_refuses_apply_without_mutation(qapp):
    dataset = _make_two_period_vector_dataset(42)
    dataset.run.grouping.update(
        period_good_frames=[1000, None], period_dead_time_us=[[0.0001] * 6, [0.0001] * 6]
    )
    before = deepcopy(dataset.run.grouping)
    window = MainWindow()
    with pytest.raises(CorrectionUnavailableError, match="good frames"):
        window._apply_grouping_settings_to_dataset(
            dataset,
            dict(
                before,
                period_mode="green_plus_red",
                deadtime_correction=True,
                deadtime_mode="file",
                deadtime_method="file",
                dead_time_us=[0.0001] * 6,
            ),
        )
    assert dataset.run.grouping is not before
    assert dataset.run.grouping["period_good_frames"] == before["period_good_frames"]
    assert dataset.run.grouping["deadtime_correction"] is False
    np.testing.assert_array_equal(dataset.asymmetry, np.zeros(4))


def test_missing_reference_apply_leaves_run_unchanged(qapp):
    dataset = _histogram_dataset()
    grouping = dataset.run.grouping
    arrays = [value.copy() for value in (dataset.time, dataset.asymmetry, dataset.error)]
    window = MainWindow()
    with pytest.raises(ValueError, match="reference is not loaded"):
        window._apply_grouping_settings_to_dataset(
            dataset,
            dict(
                grouping,
                background_correction=True,
                background_mode="reference_run",
                background_run={"run_number": 99},
            ),
        )
    assert dataset.run.grouping is grouping
    for actual, expected in zip(
        (dataset.time, dataset.asymmetry, dataset.error), arrays, strict=True
    ):
        np.testing.assert_array_equal(actual, expected)


def test_incomplete_run_blocks_fit_report_before_file_dialog(qapp, monkeypatch):
    dataset = _histogram_dataset()
    dataset.metadata["correction_error"] = "reference run is not loaded"
    window = MainWindow()
    window._data_browser.add_dataset(dataset)
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[1]))
    window._on_export_fit_report()
    assert warnings == ["Corrections incomplete"]
