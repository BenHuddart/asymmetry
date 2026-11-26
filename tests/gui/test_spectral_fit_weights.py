"""FFT observations flow through both GUI fit entry points."""

import numpy as np
import pytest
from PySide6.QtWidgets import QFileDialog

from asymmetry.core.fitting.likelihood import dataset_observations
from asymmetry.core.fitting.result_summary import fit_result_summary
from asymmetry.core.project.schema import load_project, save_project
from asymmetry.core.representation.base import RepresentationType
from asymmetry.core.representation.project_model import ProjectModel
from asymmetry.gui.mainwindow import MainWindow
from tests._qt_helpers import wait_for
from tests.gui.test_fourier_staleness_gui import _computed_window

pytestmark = pytest.mark.gui
FFT = RepresentationType.FREQ_FFT


@pytest.fixture
def computed_fft(qapp):
    window, first_source = _computed_window(MainWindow())
    _, second_source = _computed_window(window, run_number=78)
    run = int(first_source.run_number)
    window._current_dataset = first_source
    window._plot_workspace.set_active_view("frequency")
    spectrum = window._cached_frequency_spectra(run, FFT)[0]
    return window, run, int(second_source.run_number), spectrum


def test_gui_default_fft_uses_unit_weights_in_single_and_batch(
    computed_fft, qapp, monkeypatch, tmp_path
):
    window, run, other_run, spectrum = computed_fft
    assert not np.any(spectrum.error > 0)
    fit_range = (2.2, 3.15)
    expected_time = spectrum.time[(spectrum.time >= fit_range[0]) & (spectrum.time <= fit_range[1])]
    window._frequency_plot_panel._set_fit_range(*fit_range, emit_signal=True, redraw=False)
    single = window._fit_panel._single_tab
    assert np.array_equal(single._current_dataset.time, expected_time)
    assert np.array_equal(single._current_dataset.error, np.zeros_like(expected_time))

    solved = []
    single.fit_completed.connect(lambda result, *_: solved.append(result))
    single._run_fit()
    wait_for(lambda: bool(solved), qapp)
    single_result = solved[0]
    assert single_result.success
    assert single_result.parameters["nu0"].value == pytest.approx(2.7, abs=0.03)
    np.testing.assert_array_equal(single_result.residual_time, expected_time)
    assert single_result.residual_sigma is None
    assert not single_result.uncertainties
    assert single_result.covariance is None
    summary = fit_result_summary(single_result)
    assert summary["quality"] is None and summary["reduced_chi_squared"] is None
    assert single_result.chi_squared == pytest.approx(
        np.dot(single_result.residuals, single_result.residuals)
    )
    assert "Unit weights" in single._results_card.content_html()
    assert "Residual sum of squares" in single._results_card.content_html()

    window._fit_panel.restore_ui_state({"active_tab_index": 1})
    batch = window._fit_panel._global_tab
    window._data_browser.select_runs([run, other_run])
    assert {dataset.run_number for dataset in window._fit_panel.batch_datasets()} == {
        run,
        other_run,
    }
    monkeypatch.setattr("asymmetry.gui.panels.fit.global_tab.os.cpu_count", lambda: 1)
    batches = []
    batch.global_fit_completed.connect(lambda results, _: batches.append(results))
    batch._run_global_fit()
    wait_for(lambda: bool(batches), qapp)
    batch_result = batches[0][run][0]
    assert batch_result.success
    np.testing.assert_array_equal(batch_result.residual_time, expected_time)
    assert batch_result.residual_sigma is None
    assert not batch_result.uncertainties
    assert batch_result.covariance is None
    assert batch_result.chi_squared == pytest.approx(
        np.dot(batch_result.residuals, batch_result.residuals)
    )
    series_id = window._project_model.active_series_id(FFT)
    assert series_id is not None
    window._open_series_in_batch_tab(series_id)
    assert "Unit weights" in batch._results_card._meta.text()

    project = tmp_path / "fits.asymp"
    save_project(window.collect_project_state(), project)
    window._project_model = ProjectModel.from_project_state(load_project(project))
    slot = window._project_model.representation(run, FFT).fit
    series = window._project_model.batch(series_id)
    for record in (slot.result, *series.results_by_run.values()):
        assert record["input_attribution"]["spectral_weighting"]["policy"] == "unit"
        assert record["quality"] is None and record["reduced_chi_squared"] is None
    report = tmp_path / "fit-report.txt"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_: (str(report), "Text files"))
    window._on_export_fit_report()
    content = report.read_text()
    assert content.count("Unit weights on") == 3
    assert content.count("Residual sum of squares =") == 3
    assert "target" not in content


def test_gui_fft_weights_follow_positive_bins_when_window_changes(computed_fft, qapp):
    window, run, _, _ = computed_fft
    window._fourier_panel._estimate_average_error_check.setChecked(True)
    window._on_compute_fourier()
    wait_for(lambda: not window._fourier_compute_active, qapp)
    spectrum = window._cached_frequency_spectra(run, FFT)[0]
    selected = (spectrum.time >= 2.2) & (spectrum.time <= 3.15)
    positive = spectrum.error.copy()
    assert (positive[selected] > 0).all()
    single = window._fit_panel._single_tab
    solved = []
    single.fit_completed.connect(lambda result, *_: solved.append(result))
    prior_weights = None
    for index, (errors, bounds, expected_policy) in enumerate(
        [
            (positive, (2.2, 3.15), "diagonal"),
            (np.where(np.arange(positive.size) % 4, positive, 0.0), (2.2, 3.15), "positive_only"),
            (np.where(np.arange(positive.size) % 4, positive, 0.0), (2.4, 2.95), "positive_only"),
        ],
        start=1,
    ):
        spectrum.error = errors
        window._frequency_plot_panel._set_fit_range(*bounds, emit_signal=True, redraw=False)
        single._run_fit()
        wait_for(lambda: len(solved) == index, qapp)
        fit = solved[-1]
        wanted = (spectrum.time >= bounds[0]) & (spectrum.time <= bounds[1]) & (errors > 0)
        np.testing.assert_array_equal(fit.residual_time, spectrum.time[wanted])
        np.testing.assert_array_equal(fit.residual_sigma, positive[wanted])
        expected = np.sum((fit.residuals / positive[wanted]) ** 2)
        assert fit.chi_squared == pytest.approx(expected / spectrum.metadata["fourier_padding"])
        assert fit.input_attribution["spectral_weighting"]["policy"] == expected_policy
        assert fit_result_summary(fit)["quality"] is None
        assert fit.uncertainties["nu0"] > 0
        used_weights = dict(zip(fit.residual_time, fit.residual_sigma, strict=True))
        if prior_weights is not None:
            assert all(
                weight == prior_weights[frequency] for frequency, weight in used_weights.items()
            )
        prior_weights = used_weights


def test_gui_batch_records_a_successful_fft_when_another_lacks_enough_weighted_bins(
    computed_fft, qapp, monkeypatch
):
    window, run, other_run, _ = computed_fft
    other_spectrum = window._cached_frequency_spectra(other_run, FFT)[0]
    other_spectrum.error[np.argmin(abs(other_spectrum.time - 2.7))] = 0.1
    window._fit_panel.restore_ui_state({"active_tab_index": 1})
    window._data_browser.select_runs([run, other_run])
    monkeypatch.setattr("asymmetry.gui.panels.fit.global_tab.os.cpu_count", lambda: 1)

    batch = window._fit_panel._global_tab
    published = []
    batch.global_fit_completed.connect(lambda results, _: published.append(results))
    batch._run_global_fit()
    wait_for(lambda: bool(published), qapp)

    assert set(published[0]) == {run}
    series = window._project_model.batch(window._project_model.active_series_id(FFT))
    assert list(series.results_by_run) == [run]


@pytest.mark.parametrize("historical_error", [{}, {"error": [0.0, 0.0, 0.0]}])
def test_old_spectrum_without_errors_restores_unit_fit_input(qapp, historical_error):
    window = MainWindow()
    window._restore_frequency_spectra_state(
        {"77": [{"time": [1.0, 2.0, 3.0], "asymmetry": [0.1, 1.4, 0.3], **historical_error}]}
    )
    spectrum = window._cached_frequency_spectra(77, FFT)[0]
    assert spectrum.metadata["plot_domain"] == "frequency"
    np.testing.assert_array_equal(spectrum.error, np.zeros(3))
    observations = dataset_observations(spectrum)
    assert observations.uses_unit_weights
    np.testing.assert_array_equal(observations.sigma, np.ones(3))
