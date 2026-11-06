"""A usable automatic-tail skip stays visible through preview, apply and replay."""

import numpy as np
import pytest

from asymmetry.core.transform.corrections import correction_warnings
from asymmetry.gui.mainwindow import MainWindow
from asymmetry.gui.windows.grouping.dialog import GroupingDialog
from tests._qt_helpers import wait_for
from tests.gui.test_grouping_preview_pane import _histogram_dataset

pytestmark = pytest.mark.gui


def test_tail_skip_preview_apply_cache_and_success_transition(qapp):
    dataset = _histogram_dataset(
        forward=np.full(200, 100.0), backward=np.r_[np.full(99, 80.0), np.zeros(101)]
    )
    dataset.run.grouping.update(background_correction=True, background_mode="tail_fit", alpha=1.17)
    dialog = GroupingDialog([dataset])
    window = MainWindow()
    try:
        pane = dialog._preview_pane
        pane.flush()
        wait_for(
            lambda: (
                pane._last_result is not None and pane._last_result.generation == pane._generation
            ),
            qapp,
        )
        assert pane.corrections_complete
        assert dialog._apply_btn.isEnabled()
        assert "background subtraction skipped" in pane._incomplete_banner.text()
        assert pane._last_result.counts.background_level is None
        dataset.metadata["correction_error"] = "stale failure"
        applied, outcomes = window._apply_grouping_settings_to_dataset(
            dataset, dict(dataset.run.grouping)
        )
        assert applied and all(record.complete for record in outcomes.values())
        assert dataset.run.grouping["alpha"] == 1.17
        assert "correction_error" not in dataset.metadata
        assert "background_values" not in dataset.run.grouping
        assert correction_warnings(dataset.metadata)
        cold = window._grouped_time_domain_display_datasets(dataset, lifetime_corrected=False)
        warm = window._grouped_time_domain_display_datasets(dataset, lifetime_corrected=False)
        assert correction_warnings(cold[0].metadata) == correction_warnings(warm[0].metadata)
        hits = window._reduction_cache.hits
        dataset.run.histograms[1].counts = np.full(200, 80.0)
        changed = window._grouped_time_domain_display_datasets(dataset, lifetime_corrected=False)
        assert window._reduction_cache.hits == hits
        assert not correction_warnings(changed[0].metadata)
        window._apply_grouping_settings_to_dataset(dataset, dict(dataset.run.grouping))
        assert not correction_warnings(dataset.metadata)
        assert "background_values" in dataset.run.grouping
        dataset.run.histograms[1].counts = np.r_[np.full(99, 80.0), np.zeros(101)]
        window._apply_grouping_settings_to_dataset(dataset, dict(dataset.run.grouping))
        assert correction_warnings(dataset.metadata)
        assert "background_values" not in dataset.run.grouping
        assert "background_details" not in dataset.run.grouping
    finally:
        dialog._clear_dirty()
        dialog.close()
        window.close()


def test_tail_skip_project_restore_retains_request_and_outcome(qapp, tmp_path, monkeypatch):
    from asymmetry.core.project.schema import load_project, save_project

    dataset = _histogram_dataset()
    source = tmp_path / "sample.nxs"
    source.touch()
    dataset.run.source_file = str(source)
    dataset.metadata["source_file"] = str(source)
    dataset.run.grouping.update(background_correction=True, background_mode="tail_fit")
    window, restored = MainWindow(), MainWindow()
    try:
        window._apply_grouping_settings_to_dataset(dataset, dict(dataset.run.grouping))
        dataset.metadata["grouping_overrides"] = True
        window._data_browser.add_dataset(dataset)
        project = tmp_path / "skip.asymp"
        save_project(window.collect_project_state(), project)
        fresh = _histogram_dataset()
        fresh.run.source_file = str(source)
        fresh.metadata["source_file"] = str(source)
        monkeypatch.setattr(restored, "_load_file", lambda _: fresh)
        restored.restore_project_state(load_project(project), str(project))
        assert fresh.run.grouping["background_mode"] == "tail_fit"
        assert correction_warnings(fresh.metadata)
        assert "correction_error" not in fresh.metadata
    finally:
        window.close()
        restored.close()


def test_integral_scan_series_and_restored_provenance_keep_skip(qapp):
    from asymmetry.core.representation.series import FitSeries
    from tests.gui.test_integral_scan_gui import _ds

    window, restored = MainWindow(), MainWindow()
    try:
        datasets = [_ds(11, 110, 90, 100), _ds(12, 120, 80, 200)]
        for dataset in datasets:
            dataset.run.grouping.update(background_correction=True, background_mode="tail_fit")
            window._data_browser.add_dataset(dataset)
            restored._data_browser.add_dataset(dataset)
        window._plot_workspace.set_active_view("integral_scan")
        window._fit_panel.set_datasets(datasets)
        window._on_scan_requested()
        series = window._project_model.batch(window._alc_scan_series_id)
        for summary in series.results_by_run.values():
            assert summary["correction_outcomes"]["red"]["background"]["status"] == "skipped"
        saved = FitSeries.from_dict(series.to_dict())
        restored._project_model.add_batch(saved)
        restored._alc_scan_points = restored._alc_points_from_series(saved)
        scan = restored._alc_display_scan("run")
        assert all(
            record.background.status == "skipped"
            for periods in scan.correction_outcomes.values()
            for record in periods.values()
        )
        restored._render_alc_scan()
        assert "Correction warning" in restored._alc_scan_view._provenance_label.text()
        assert (
            "background subtraction skipped" in restored._alc_scan_view._provenance_label.toolTip()
        )
    finally:
        window.close()
        restored.close()


def test_maxent_completion_and_cached_display_show_actual_skip(qapp):
    from asymmetry.core.data.dataset import MuonDataset
    from asymmetry.core.maxent import MaxEntConfig, maxent
    from tests.core.test_tail_skip_consumers import tail_run

    run = tail_run()
    dataset = MuonDataset(
        np.arange(200) * 0.1, np.zeros(200), np.ones(200), metadata=run.metadata, run=run
    )
    config = MaxEntConfig(
        n_spectrum_points=64, inner_iterations=2, auto_window=False, f_min_mhz=0.1, f_max_mhz=2
    )
    with pytest.warns(UserWarning, match="subtraction skipped"):
        result = maxent(run, config, cycles=1)
    window = MainWindow()
    try:
        window._data_browser.add_dataset(dataset)
        window._on_dataset_selected(42)
        window._maxent_active_run_number = 42
        window._maxent_active_run = run
        window._maxent_active_config = config
        window._on_maxent_worker_finished(result)
        assert (
            "background subtraction skipped"
            in window._maxent_panel._action_footer._status_label.text()
        )
        window._sync_frequency_plot_for_run(42)
        assert "background subtraction skipped" in window.statusBar().currentMessage()
    finally:
        window.close()


def test_in_place_count_edit_refreshes_grouped_data_and_tail_warning(qapp):
    dataset = _histogram_dataset(forward=np.full(200, 100.0), backward=np.full(200, 80.0))
    dataset.run.grouping.update(background_correction=True, background_mode="tail_fit")
    window = MainWindow()
    try:
        first = window._grouped_time_domain_display_datasets(dataset, lifetime_corrected=False)
        assert not correction_warnings(first[0].metadata)
        counts = dataset.run.histograms[1].counts
        counts[99:] = 0
        skipped = window._grouped_time_domain_display_datasets(dataset, lifetime_corrected=False)
        assert dataset.run.histograms[1].counts is counts
        assert correction_warnings(skipped[0].metadata)
        np.testing.assert_array_equal(skipped[1].asymmetry, counts)
        counts[99:] = 80
        recovered = window._grouped_time_domain_display_datasets(dataset, lifetime_corrected=False)
        assert not correction_warnings(recovered[0].metadata)
    finally:
        window.close()


def test_in_place_supplied_variance_edit_refreshes_grouped_errors(qapp):
    dataset = _histogram_dataset(forward=np.full(200, 100.0), backward=np.full(200, 80.0))
    variance = np.full(200, 4.0)
    dataset.run.histograms[0].count_variance = variance
    window = MainWindow()
    try:
        first = window._grouped_time_domain_display_datasets(dataset, lifetime_corrected=False)
        variance[:] = 9
        changed = window._grouped_time_domain_display_datasets(dataset, lifetime_corrected=False)
        np.testing.assert_array_equal(first[0].error, 2)
        np.testing.assert_array_equal(changed[0].error, 3)
    finally:
        window.close()


@pytest.mark.parametrize("has_warning", [False, True])
def test_maxent_batch_completion_separates_warning_without_trailing_space(
    qapp, monkeypatch, has_warning
):
    from asymmetry.core.transform.corrections import CorrectionOutcome, CorrectionOutcomes

    dataset = _histogram_dataset()
    if has_warning:
        outcome = CorrectionOutcome("background", "tail_fit", "skipped", "empty tail")
        dataset.metadata["correction_outcomes"] = {
            "red": CorrectionOutcomes(background=outcome).to_dict()
        }
    window = MainWindow()
    try:
        monkeypatch.setattr(window._data_browser, "get_selected_datasets", lambda: [dataset])
        monkeypatch.setattr(window, "_frequency_spectra_from_cache", lambda *args: [dataset])
        window._maxent_batch_done = 1
        window._advance_maxent_batch()
        message = window.statusBar().currentMessage()
        completion = "MaxEnt batch: reconstructed 1 run(s); moments sent to trend."
        assert message == completion + (" " + outcome.warning if has_warning else "")
    finally:
        window.close()
