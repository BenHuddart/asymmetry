"""Incomplete correction requests remain visible and cannot become final results."""

from copy import deepcopy

import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from asymmetry.core.io import resolve_background_reference
from asymmetry.core.transform.corrections import CorrectionUnavailableError
from asymmetry.core.transform.reference_scale import ReferenceExposure
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

    sample = _histogram_dataset(
        grouping_extra={
            "good_frames": 20,
            "reference_exposure": ReferenceExposure(20, "user").to_dict(),
        }
    )
    reference = _histogram_dataset(
        run_number=99,
        forward=np.full(6, 2.0),
        backward=np.full(6, 3.0),
        grouping_extra={"good_frames": 10},
    )
    reference.run.source_file = "unreadable.nxs"
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
            reference.run.grouping["reference_exposure"] = ReferenceExposure(
                frames, "user"
            ).to_dict()
            resolved = resolve_background_reference(
                payload,
                datasets=[reference],
                sample_exposure=ReferenceExposure(20, "user"),
                allow_file=False,
            )
            old = pane._last_result
            pane.request_preview(
                histograms=sample.run.histograms,
                grouping=grouping,
                facts=_facts(),
                resolved_reference=resolved,
                run=sample.run,
            )
            pane.flush()
            wait_for(lambda: pane._last_result is not None and pane._last_result is not old, qapp)
            result = pane._last_result
            assert all(outcome.complete for outcome in result.corrections)
            effective = result.corrections[0].background.effective
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


def test_form_error_cannot_be_cleared_by_an_older_worker_result(qapp):
    from asymmetry.gui.windows.grouping.preview_pane import GroupingPreviewPane
    from tests.gui.test_grouping_preview_pane import _draw_result

    pane = GroupingPreviewPane()
    try:
        result = _draw_result(pane, generation=pane._generation)
        pane.show_error("invalid draft reference")
        pane._on_finished(result)
        assert not pane.corrections_complete
        assert "invalid draft reference" in pane._incomplete_banner.text()
    finally:
        pane.shutdown()


def test_obsolete_worker_error_cannot_replace_a_newer_preview(qapp):
    from asymmetry.gui.windows.grouping.preview_pane import GroupingPreviewPane
    from tests.gui.test_grouping_preview_pane import _draw_result

    pane = GroupingPreviewPane()
    try:
        pane._generation = 2
        result = _draw_result(pane, generation=2)
        pane._on_finished(result)
        pane._on_error("older reduction failed", generation=1)
        assert pane._last_result is result
        assert pane.corrections_complete
    finally:
        pane.shutdown()


@pytest.mark.parametrize("table", [[], [0.01], [0.0, 0.0]])
def test_file_calibration_preview_is_usable_and_apply_preserves_policy(qapp, table):
    dataset = _histogram_dataset()
    dataset.run.grouping.update(
        dead_time_us=table,
        deadtime_file_us=table,
        deadtime_correction=True,
        deadtime_mode="file",
        deadtime_method="file",
    )
    dialog = GroupingDialog([dataset])
    try:
        pane = dialog._preview_pane
        pane.flush()
        wait_for(lambda: pane._last_result is not None, qapp)
        assert pane.corrections_complete
        assert dialog._apply_btn.isEnabled()
        assert ("File deadtime correction not applied" in pane._incomplete_banner.text()) == (
            len(table) < 2
        )
        window = MainWindow()
        applied, outcomes = window._apply_grouping_settings_to_dataset(
            dataset, dict(dataset.run.grouping)
        )
        assert applied and all(item.complete for item in outcomes.values())
        assert dataset.run.grouping["deadtime_mode"] == "file"
        assert "correction_error" not in dataset.metadata
        assert list(dataset.metadata["correction_outcomes"].values())[0]["deadtime"]["status"] == (
            "skipped" if len(table) < 2 else "applied"
        )
        # Changing the supplied file fact replaces the skipped decision on replay.
        dataset.run.grouping["deadtime_file_us"] = [0.0, 0.0]
        _, outcomes = window._apply_grouping_settings_to_dataset(
            dataset, dict(dataset.run.grouping)
        )
        assert all(item.deadtime.status == "applied" for item in outcomes.values())
    finally:
        dialog._clear_dirty()
        dialog.close()


def test_released_project_zero_override_cannot_supply_absent_file_table(
    qapp, tmp_path, monkeypatch
):
    from asymmetry.core.io import load
    from asymmetry.core.project.schema import load_project, save_project
    from tests.io.test_nexus_loader import _write_v2_file

    h5py = pytest.importorskip("h5py")
    path = tmp_path / "sample.nxs"
    _write_v2_file(path)
    with h5py.File(path, "r+") as handle:
        del handle["raw_data_1/instrument/detector_1/dead_time"]
    dataset = load(path)
    window = MainWindow()
    window._data_browser.add_dataset(dataset)
    state = window.collect_project_state()
    state["schema_version"] = 31
    state["datasets"][0]["grouping_overrides"] = dict(
        dataset.run.grouping,
        deadtime_correction=True,
        deadtime_mode="file",
        dead_time_us=[0.0, 0.0],
    )
    state["datasets"][0]["grouping_overrides"].pop("deadtime_file_us", None)
    project = tmp_path / "released.asymp"
    save_project(state, project)
    fresh = load(path)
    restored = MainWindow()
    monkeypatch.setattr(restored, "_load_file", lambda _: fresh)
    restored.restore_project_state(load_project(project), str(project))
    assert fresh.run.grouping["deadtime_file_us"] == []
    assert "correction_error" not in fresh.metadata
    assert all(
        record["deadtime"]["status"] == "skipped"
        for record in fresh.metadata["correction_outcomes"].values()
    )


def test_green_apply_uses_green_file_table_and_retains_both_acquisitions(qapp):
    dataset = _make_two_period_vector_dataset(42)
    red, green = [0.0001] * 6, [-0.0002] * 6
    dataset.run.grouping.update(
        period_good_frames=[1000.0, 2000.0],
        period_dead_time_us=[red, green],
        deadtime_file_us=red,
        dead_time_us=red,
    )
    window = MainWindow()
    applied, outcomes = window._apply_grouping_settings_to_dataset(
        dataset,
        dict(
            dataset.run.grouping,
            period_mode="green",
            deadtime_correction=True,
            deadtime_mode="file",
            deadtime_method="file",
        ),
    )
    assert applied
    assert outcomes["green"].deadtime.effective["tau_us"] == tuple(green)
    assert outcomes["green"].deadtime.effective["good_frames"] == 2000.0
    assert dataset.run.grouping["period_dead_time_us"] == [red, green]


@pytest.mark.parametrize("sample_period", [1, 2])
def test_held_period_preview_enforces_reference_independence(qapp, sample_period):
    from tests.gui.test_grouping_preview_pane import _reduce_preview
    from tests.io.test_period_selection import _combined_two_period

    sample = _combined_two_period()
    sample.run.metadata["source_file"] = sample.run.source_file
    sample.run.grouping.update(
        period_mode="red" if sample_period == 1 else "green",
        background_correction=True,
        background_mode="reference_run",
        background_run={"source_file": "synthetic.nxs", "run_number": 12345, "period_number": 2},
    )
    reference = resolve_background_reference(
        sample.run.grouping["background_run"],
        datasets=[sample],
        sample_exposure=ReferenceExposure(20, "user"),
        allow_file=False,
    )
    if sample_period == 2:
        with pytest.raises(ValueError, match="identical observations"):
            _reduce_preview(sample, run=sample.run, resolved_reference=reference)
    else:
        result = _reduce_preview(sample, run=sample.run, resolved_reference=reference)
        assert all(outcome.complete for outcome in result.corrections)
        assert result.corrections[0].background.effective["reference"]["period_number"] == 2


@pytest.mark.parametrize("reference_period", [1, 2])
def test_alpha_request_retains_sample_acquisition_identity(qapp, reference_period):
    from copy import deepcopy

    from asymmetry.core.workflow.calibration import CalibrationOptions, calibrate_source
    from tests.io.test_period_selection import _combined_two_period

    sample = _combined_two_period()
    source = deepcopy(sample)
    grouping = dict(
        sample.run.grouping,
        background_correction=True,
        background_mode="reference_run",
        period_mode="green",
    )
    reference = resolve_background_reference(
        {"source_file": "synthetic.nxs", "run_number": 12345, "period_number": reference_period},
        datasets=[source],
        sample_exposure=ReferenceExposure(20, "user"),
    )
    from dataclasses import replace

    sample = replace(sample, run=replace(sample.run, grouping=grouping))
    if reference_period == 2:
        with pytest.raises(ValueError, match="identical observations"):
            calibrate_source(
                sample, CalibrationOptions("ratio"), reference_resolver=lambda _: reference
            )
    else:
        result = calibrate_source(
            sample, CalibrationOptions("ratio"), reference_resolver=lambda _: reference
        )
        assert result.source.corrections.complete


def test_reference_preview_requires_sample_identity_before_copying_periods(qapp):
    from tests.gui.test_grouping_preview_pane import _reduce_preview
    from tests.io.test_period_selection import _combined_two_period

    sample = _combined_two_period()
    sample.run.grouping.update(
        period_mode="green", background_mode="reference_run", background_correction=True
    )
    reference = resolve_background_reference(
        {"source_file": "synthetic.nxs", "run_number": 12345, "period_number": 2}, datasets=[sample]
    )
    with pytest.raises(ValueError, match="sample Run"):
        _reduce_preview(sample, resolved_reference=reference)


@pytest.mark.parametrize("consumer", ["apply", "rebunch"])
def test_gui_final_reduction_uses_canonical_run_identity(qapp, tmp_path, consumer):
    from asymmetry.core.io.nexus_writer import write_nexus_v1

    dataset = _histogram_dataset(run_number=42)
    dataset.run.metadata = {}
    dataset.run.source_file = str(tmp_path / "sample.nxs")
    write_nexus_v1(dataset.run, dataset.run.source_file, effective=True)
    dataset.run.grouping.update(
        background_correction=True,
        background_mode="reference_run",
        background_run={
            "source_file": dataset.run.source_file,
            "run_number": 42,
            "period_number": 1,
        },
    )
    window = MainWindow()
    with pytest.raises(ValueError, match="identical observations"):
        if consumer == "apply":
            window._apply_grouping_settings_to_dataset(dataset, dict(dataset.run.grouping))
        else:
            window._counts_first_rebunched_arrays(dataset, 2)


def test_inactive_reference_does_not_require_run_identity(qapp):
    from tests.gui.test_grouping_preview_pane import _reduce_preview
    from tests.io.test_period_selection import _combined_two_period

    sample = _combined_two_period()
    sample.run.grouping.update(background_correction=False, background_mode="reference_run")
    reference = resolve_background_reference(
        {"source_file": "synthetic.nxs", "run_number": 12345, "period_number": 2}, datasets=[sample]
    )
    result = _reduce_preview(sample, resolved_reference=reference)
    assert all(outcome.complete for outcome in result.corrections)
    assert result.corrections[0].background.status == "off"
