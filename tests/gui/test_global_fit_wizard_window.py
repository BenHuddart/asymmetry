"""Tests for the Global Fit Wizard window: its five steps, runs and landing rules."""

from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pytest

pytestmark = [pytest.mark.gui, pytest.mark.slow, pytest.mark.integration]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication, QMessageBox, QPushButton

import asymmetry.gui.windows.global_fit_wizard_window as wizard_window_module
from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.component_tags import FieldGeometry, PhysicsClass
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.fit_wizard import (
    CandidateAssessment,
    CandidateTemplate,
    ConfidenceTier,
    FitTiming,
    FitWizardRecommendation,
    RecommendationVerdict,
    SelectionMetric,
    SpectrumFingerprint,
)
from asymmetry.core.fitting.global_fit_wizard import (
    GlobalCandidateAssessment,
    GlobalFitWizardRecommendation,
    GlobalFitWizardScreeningTable,
    GlobalParameterRecommendation,
    RunResidualDiagnostic,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.fitting.wizard_scope import WizardScope
from asymmetry.gui.panels.log_panel import LogPanel
from asymmetry.gui.utils.fit_times import shared_fit_time_store
from asymmetry.gui.widgets.panel_section import PanelSection
from asymmetry.gui.widgets.wizard_stepper import StepState
from asymmetry.gui.windows.global_fit_wizard_window import GlobalFitWizardWindow
from tests._qt_helpers import wait_for

LF_DYNAMICS = WizardScope(physics=frozenset({PhysicsClass.DYNAMICS, PhysicsClass.MAGNETISM}))


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def datasets() -> list[MuonDataset]:
    time_axis = np.linspace(0.0, 8.0, 120)
    error = np.full_like(time_axis, 0.01)
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    items: list[MuonDataset] = []
    for idx, lam in enumerate((0.2, 0.3, 0.5), start=1):
        asymmetry = model.function(time_axis, A_1=0.2, Lambda=lam, A_bg=0.01)
        items.append(
            MuonDataset(
                time=time_axis,
                asymmetry=asymmetry,
                error=error,
                metadata={
                    "run_number": 700 + idx,
                    "run_label": str(700 + idx),
                    "field": 100.0 * idx,
                    "temperature": 5.0,
                },
            )
        )
    return items


def _fake_recommendation(datasets: list[MuonDataset]) -> GlobalFitWizardRecommendation:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    template = CandidateTemplate(
        key="exp_constant",
        title="Exponential + Constant",
        category="General",
        rationale="Baseline candidate",
        model=model,
    )

    fit_results: dict[int, FitResult] = {}
    fitted_curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    component_curves: dict[int, tuple[tuple[str, np.ndarray], ...]] = {}
    fingerprints: dict[int, SpectrumFingerprint] = {}
    run_diagnostics: list[RunResidualDiagnostic] = []
    for idx, dataset in enumerate(datasets, start=1):
        lam = 0.1 + 0.1 * idx
        params = ParameterSet(
            [
                Parameter("A_1", value=0.2, min=0.0, max=1.0),
                Parameter("Lambda", value=lam, min=0.0, max=5.0),
                Parameter("A_bg", value=0.01, min=-0.2, max=0.2),
            ]
        )
        curve = model.function(dataset.time, A_1=0.2, Lambda=lam, A_bg=0.01)
        run_number = int(dataset.run_number)
        fit_results[run_number] = FitResult(
            success=True,
            chi_squared=3.0 + idx,
            reduced_chi_squared=0.05 + 0.01 * idx,
            parameters=params,
            uncertainties={"A_1": 0.01, "Lambda": 0.02, "A_bg": 0.001},
            residuals=np.asarray(dataset.asymmetry - curve, dtype=float),
            message="ok",
        )
        fitted_curves[run_number] = (
            np.asarray(dataset.time, dtype=float).copy(),
            np.asarray(curve, dtype=float),
        )
        component_curves[run_number] = tuple(
            model.evaluate_components(
                dataset.time,
                additive_only=True,
                A_1=0.2,
                Lambda=lam,
                A_bg=0.01,
            )
        )
        fingerprints[run_number] = SpectrumFingerprint(
            tail_estimate=0.01,
            initial_amplitude_estimate=0.2,
            zero_crossings=0,
            smoothed_zero_crossings=0,
            smoothed_turning_points=0,
            dominant_fft_frequency_mhz=0.0,
            dominant_fft_snr=0.0,
            dominant_fft_cycles_in_window=0.0,
            monotonic_decay_fraction=1.0,
            early_time_curvature=-0.1,
            semilog_slope_ratio=1.0,
            late_time_dip_recovery_score=0.0,
            oscillatory_hint=False,
            kt_like_hint=False,
            multi_rate_hint=True,
        )
        run_diagnostics.append(
            RunResidualDiagnostic(
                run_number=run_number,
                run_label=dataset.run_label,
                axis_value=float(dataset.metadata["field"]),
                residual_rms=0.8,
                runs_z_score=0.1,
                max_abs_autocorrelation=0.1,
                residual_fft_peak_snr=1.0,
                gate_passed=True,
                gate_reasons=(),
            )
        )

    assessment = GlobalCandidateAssessment(
        template=template,
        fit_results_by_run=fit_results,
        global_parameters=ParameterSet(
            [
                Parameter("A_1", value=0.2, min=0.0, max=1.0),
                Parameter("A_bg", value=0.01, min=-0.2, max=0.2),
            ]
        ),
        global_param_names=("A_1", "A_bg"),
        local_param_names=("Lambda",),
        fixed_param_names=(),
        parameter_recommendations=(
            GlobalParameterRecommendation(
                name="A_1",
                recommended_role="Global",
                global_score=10.0,
                local_score=12.0,
                score_delta=2.0,
                rationale="Shared amplitude is adequate.",
            ),
            GlobalParameterRecommendation(
                name="Lambda",
                recommended_role="Local",
                global_score=15.0,
                local_score=9.0,
                score_delta=6.0,
                rationale="Rate variation is strongly supported.",
            ),
            GlobalParameterRecommendation(
                name="A_bg",
                recommended_role="Global",
                global_score=10.0,
                local_score=11.0,
                score_delta=1.0,
                rationale="Background remains stable.",
            ),
        ),
        run_diagnostics=tuple(run_diagnostics),
        series_warnings=(),
        aic=10.0,
        aicc=10.2,
        bic=12.0,
        selected_score=10.2,
        fitted_curves_by_run=fitted_curves,
        component_curves_by_run=component_curves,
    )
    return GlobalFitWizardRecommendation(
        series_axis_key="field",
        series_axis_label="Field (G)",
        mixed_axes_warning=None,
        fingerprints_by_run=fingerprints,
        dataset_order=tuple(int(dataset.run_number) for dataset in datasets),
        templates=(template,),
        assessments=(assessment,),
        metric=SelectionMetric.AICC,
        recommended_key="exp_constant",
        comparable_keys=(),
        summary="Recommended: Exponential + Constant by AICc.",
    )


def _fake_screening_recommendation(datasets: list[MuonDataset]) -> GlobalFitWizardRecommendation:
    optimized = _fake_recommendation(datasets)
    assessment = optimized.assessments[0]
    screening_assessment = replace(
        assessment,
        parameter_recommendations=(),
        prescreen_only=True,
        global_parameters=ParameterSet(),
        global_param_names=("A_1", "Lambda", "A_bg"),
        local_param_names=(),
        fixed_param_names=(),
    )
    return replace(
        optimized,
        assessments=(screening_assessment,),
        recommended_key=None,
        comparable_keys=(),
        summary=(
            "Single-fit screening complete. These scores come from independent per-dataset fits only "
            "and have not yet been optimized for coupled global fitting. Select one or more candidates to continue."
        ),
    )


def _fake_multi_variant_recommendation(
    datasets: list[MuonDataset],
) -> GlobalFitWizardRecommendation:
    base = _fake_recommendation(datasets)
    best = replace(
        base.assessments[0],
        assessment_key="exp_constant|g=A_1,A_bg|l=Lambda",
    )
    shared = replace(
        base.assessments[0],
        global_param_names=("A_1", "Lambda", "A_bg"),
        local_param_names=(),
        global_parameters=ParameterSet(
            [
                Parameter("A_1", value=0.2, min=0.0, max=1.0),
                Parameter("Lambda", value=0.35, min=0.0, max=5.0),
                Parameter("A_bg", value=0.01, min=-0.2, max=0.2),
            ]
        ),
        parameter_recommendations=(),
        aic=11.0,
        aicc=11.2,
        bic=13.0,
        selected_score=11.2,
        assessment_key="exp_constant|g=A_1,Lambda,A_bg|l=none",
    )
    return replace(
        base,
        assessments=(best, shared),
        recommended_key=best.selection_key,
        comparable_keys=(best.selection_key, shared.selection_key),
        summary="Recommended globally optimized candidate: Exponential + Constant by AICc, with a similarly scoring alternative to inspect.",
    )


def _fake_single_fit_recommendation(dataset: MuonDataset) -> FitWizardRecommendation:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    template = CandidateTemplate(
        key="exp_constant",
        title="Exponential + Constant",
        category="General",
        rationale="Baseline candidate",
        model=model,
    )
    curve = model.function(dataset.time, A_1=0.2, Lambda=0.3, A_bg=0.01)
    assessment = CandidateAssessment(
        template=template,
        fit_result=FitResult(
            success=True,
            chi_squared=4.0,
            reduced_chi_squared=0.08,
            parameters=ParameterSet(
                [
                    Parameter("A_1", value=0.2, min=0.0, max=1.0),
                    Parameter("Lambda", value=0.3, min=0.0, max=5.0),
                    Parameter("A_bg", value=0.01, min=-0.2, max=0.2),
                ]
            ),
            uncertainties={"A_1": 0.01, "Lambda": 0.02, "A_bg": 0.001},
            residuals=np.asarray(dataset.asymmetry - curve, dtype=float),
            message="ok",
        ),
        aic=8.0,
        aicc=8.2,
        bic=9.0,
        selected_score=8.2,
        residual_rms=0.8,
        runs_z_score=0.1,
        max_abs_autocorrelation=0.1,
        residual_fft_peak_snr=1.0,
        residual_gate_passed=True,
        residual_gate_reasons=(),
        bound_hits=(),
        fitted_time=np.asarray(dataset.time, dtype=float).copy(),
        fitted_curve=np.asarray(curve, dtype=float),
        component_curves=tuple(
            model.evaluate_components(
                dataset.time,
                additive_only=True,
                A_1=0.2,
                Lambda=0.3,
                A_bg=0.01,
            )
        ),
        timing=FitTiming(0.02, dataset.n_points),
    )
    return FitWizardRecommendation(
        fingerprint=SpectrumFingerprint(
            tail_estimate=0.01,
            initial_amplitude_estimate=0.2,
            zero_crossings=0,
            smoothed_zero_crossings=0,
            smoothed_turning_points=0,
            dominant_fft_frequency_mhz=0.0,
            dominant_fft_snr=0.0,
            dominant_fft_cycles_in_window=0.0,
            monotonic_decay_fraction=1.0,
            early_time_curvature=-0.1,
            semilog_slope_ratio=1.0,
            late_time_dip_recovery_score=0.0,
            oscillatory_hint=False,
            kt_like_hint=False,
            multi_rate_hint=False,
        ),
        templates=(template,),
        assessments=(assessment,),
        metric=SelectionMetric.AICC,
        recommended_key="exp_constant",
        comparable_keys=(),
        summary="Recommended: Exponential + Constant by AICc.",
    )


def _analysis_complete(window: GlobalFitWizardWindow) -> bool:
    return window._recommendation is not None and window._tasks.active_count == 0


def _fake_screening_table(
    datasets_arg: list[MuonDataset],
    completed_by_run: dict[int, object],
    *,
    sources_by_run: dict[int, object] | None = None,
    store: dict[int, object] | None = None,
    generated: tuple[int, ...] = (),
) -> GlobalFitWizardScreeningTable:
    """Stand in for phase 1: a portfolio plus a completed per-run score table.

    ``store`` is the caller's cache, which the real helper replaces in place with
    the runs' own single-run analyses; the window emits what changed there.
    """
    sources = sources_by_run if sources_by_run is not None else completed_by_run
    if store is not None:
        store.clear()
        store.update(sources)
    return GlobalFitWizardScreeningTable(
        portfolio=wizard_window_module.build_global_fit_wizard_candidate_portfolio(datasets_arg),
        recommendations_by_run=completed_by_run,
        single_fit_recommendations_by_run=sources,
        generated_run_numbers=generated,
        series_rebin_factor=1,
    )


def _step(window: GlobalFitWizardWindow, key: str) -> tuple[StepState, str]:
    """A step's state and its one-line summary (the button's tooltip)."""
    return window._stepper.state(key), window._stepper._buttons[key].toolTip()


def _screen(window: GlobalFitWizardWindow, qapp: QApplication) -> None:
    window._start_analysis()
    wait_for(lambda: _analysis_complete(window), qapp)


def _optimise(window: GlobalFitWizardWindow, qapp: QApplication, rows: int) -> None:
    window._optimise_btn.click()
    wait_for(
        lambda: _analysis_complete(window) and window._optimised_table.rowCount() == rows, qapp
    )


def _run_progress_cancel(window: GlobalFitWizardWindow) -> QPushButton:
    return next(
        button
        for button in window._run_progress.findChildren(QPushButton)
        if button.text() == "Cancel"
    )


def test_global_fit_wizard_window_populates_tables(
    qapp: QApplication,
    datasets: list[MuonDataset],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        wizard_window_module,
        "build_global_fit_wizard_screening_recommendation",
        lambda datasets_arg, **_kwargs: _fake_screening_recommendation(datasets_arg),
    )
    window = GlobalFitWizardWindow()
    window.set_analysis_context(datasets)

    _screen(window, qapp)

    assert "Field" in window._overview_banner.text()
    # No tab scaffolding on the _build_central override path; screening lands
    # the window on the Screen step.
    assert window._tabs is None
    assert window._stepper.current_key() == "screen"
    assert window._overview_table.rowCount() == len(datasets)
    assert window._portfolio_table.rowCount() == 1
    assert window._leaderboard.selected_key() == "exp_constant"
    assert window._screening_table.rowCount() == 1
    assert window._optimised_table.rowCount() == 0
    assert window._compare_body.isHidden()
    assert not window._compare_empty.isHidden()


def test_global_fit_wizard_window_continue_then_apply_emits_the_assessment(
    qapp: QApplication,
    datasets: list[MuonDataset],
) -> None:
    window = GlobalFitWizardWindow()
    window.set_analysis_context(datasets)
    window.set_cached_recommendation(_fake_recommendation(datasets))

    emitted: dict[str, object] = {}
    window.apply_assessment_requested.connect(
        lambda assessment, recommendation: emitted.update(
            {"assessment": assessment, "recommendation": recommendation}
        )
    )

    # Compare's A defaults to the recommended row; Continue opens Apply.
    assert window._compare_panel.a_key() == "exp_constant"
    window._compare_panel._continue.click()
    assert window._stepper.current_key() == "apply"
    assert window._apply_btn.text() == "Apply to the global fit tab"
    window._apply_btn.click()

    assert emitted["assessment"].template.key == "exp_constant"
    assert emitted["recommendation"].recommended_key == "exp_constant"


def test_global_fit_wizard_window_shows_progress_log(
    qapp: QApplication,
    datasets: list[MuonDataset],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_build(datasets_arg, **kwargs):
        progress_callback = kwargs.get("progress_callback")
        if callable(progress_callback):
            progress_callback("Preparing missing single-fit wizard tables for global screening.")
            progress_callback("Single-fit table 701: evaluating shared candidate portfolio.")
        return _fake_screening_recommendation(datasets_arg)

    monkeypatch.setattr(
        wizard_window_module,
        "build_global_fit_wizard_screening_recommendation",
        _fake_build,
    )
    window = GlobalFitWizardWindow()
    window.set_analysis_context(datasets)

    _screen(window, qapp)

    # Progress messages stream into the Screen step's progress block, which
    # collapses to a Run log, and are mirrored into current_log_text() (the
    # analysis_cached payload).
    log_text = window.current_log_text()
    assert "Starting screening for 3 datasets." in log_text
    assert "Preparing missing single-fit wizard tables for global screening." in log_text
    assert "Single-fit table 701: evaluating shared candidate portfolio." in log_text
    panel_text = window._run_progress.findChild(LogPanel).to_plain_text()
    assert "Starting screening for 3 datasets." in panel_text
    assert window._run_progress.parentWidget() is window._progress_hosts["screen"]
    assert window._run_progress.findChild(PanelSection).title() == "Run log"
    assert not window._run_progress.isHidden()


def test_global_fit_wizard_window_optimizes_selected_candidates(
    qapp: QApplication,
    datasets: list[MuonDataset],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        wizard_window_module,
        "build_global_fit_wizard_screening_recommendation",
        lambda datasets_arg, **_kwargs: _fake_screening_recommendation(datasets_arg),
    )

    def _fake_build(datasets_arg, **kwargs):
        captured["datasets"] = datasets_arg
        captured["selected_template_keys"] = kwargs.get("selected_template_keys")
        progress_callback = kwargs.get("progress_callback")
        if callable(progress_callback):
            progress_callback("Coupled optimisation 1/1: Exponential + Constant.")
            progress_callback("Completed coupled optimisation for Exponential + Constant.")
        return _fake_multi_variant_recommendation(datasets_arg)

    monkeypatch.setattr(
        wizard_window_module,
        "build_global_fit_wizard_recommendation",
        _fake_build,
    )
    window = GlobalFitWizardWindow()
    window.set_analysis_context(datasets)
    _screen(window, qapp)

    # The one screened family is within Δ ≤ 10 of the best, so it is pre-ticked (D7).
    assert window._leaderboard.ticked() == ("exp_constant",)
    _optimise(window, qapp, rows=2)

    assert captured["datasets"] == datasets
    assert captured["selected_template_keys"] == ("exp_constant",)
    assert window._stepper.current_key() == "compare"
    assert window._run_progress.parentWidget() is window._progress_hosts["compare"]
    assert window._screening_table.item(0, 5).text() == "Optimized"
    assert window._optimised_table.item(0, 6).text() == "A_1, A_bg"
    assert window._optimised_table.item(0, 7).text() == "Lambda"
    assert window._optimised_table.item(1, 6).text() == "A_1, Lambda, A_bg"
    assert window._optimised_table.item(1, 7).text() == "None"
    # An optimise keeps the user's ticks.
    assert window._leaderboard.ticked() == ("exp_constant",)
    log_text = window.current_log_text()
    assert "Starting coupled global optimisation for: Exponential + Constant." in log_text
    assert "Coupled optimisation 1/1: Exponential + Constant." in log_text
    assert "Completed coupled optimisation for Exponential + Constant." in log_text


def test_global_fit_wizard_window_emits_generated_single_fit_analyses(
    qapp: QApplication,
    datasets: list[MuonDataset],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fit tabs get the runs' own analyses; the screen gets the derived table.

    Phase 1 produces both, and they are not interchangeable: the per-run
    single-run Fit Wizard analysis is what a fit tab shows and what a later
    series analysis can reuse, while the completed score table is one resolution
    and one alphabet, derived. Only the former is written back.
    """
    sources = {
        int(dataset.run_number): _fake_single_fit_recommendation(dataset) for dataset in datasets
    }
    completed = {
        int(dataset.run_number): replace(
            _fake_single_fit_recommendation(dataset), summary="Completed score table."
        )
        for dataset in datasets
    }
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        wizard_window_module,
        "build_or_complete_single_fit_wizard_recommendations_for_global_portfolio",
        lambda datasets_arg, **kwargs: _fake_screening_table(
            datasets_arg,
            completed,
            sources_by_run=sources,
            store=kwargs.get("existing_recommendations_by_run"),
            generated=tuple(sources),
        ),
    )

    def _fake_build(datasets_arg, **kwargs):
        captured["single_fit"] = kwargs.get("single_fit_recommendations_by_run")
        captured["portfolio"] = kwargs.get("portfolio")
        return _fake_screening_recommendation(datasets_arg)

    monkeypatch.setattr(
        wizard_window_module,
        "build_global_fit_wizard_screening_recommendation",
        _fake_build,
    )

    window = GlobalFitWizardWindow()
    emitted: dict[int, FitWizardRecommendation] = {}
    window.single_fit_recommendations_generated.connect(lambda payload: emitted.update(payload))
    window.set_analysis_context(datasets)

    window._start_analysis()
    wait_for(lambda: _analysis_complete(window), qapp)

    assert set(emitted) == {int(dataset.run_number) for dataset in datasets}
    for run_number, recommendation in emitted.items():
        assert recommendation is sources[run_number]
    # The screening builder is handed the completed table and the portfolio it
    # was completed against, so it never re-runs phase 1.
    assert captured["single_fit"] == completed
    assert captured["portfolio"] is not None


def test_global_fit_wizard_window_records_the_fits_it_ran_and_shows_the_estimates(
    qapp: QApplication,
    datasets: list[MuonDataset],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the analyses phase 1 generated are timed; a reused one is not recorded again."""
    sources = {
        int(dataset.run_number): _fake_single_fit_recommendation(dataset) for dataset in datasets
    }
    generated = tuple(sources)[1:]
    captured: dict[str, object] = {}

    def _fake_phase_one(datasets_arg, **kwargs):
        captured["phase_one"] = kwargs["fit_times"]
        return _fake_screening_table(
            datasets_arg,
            sources,
            store=kwargs["existing_recommendations_by_run"],
            generated=generated,
        )

    def _fake_build(datasets_arg, **kwargs):
        captured["screening"] = kwargs["fit_times"]
        return _fake_screening_recommendation(datasets_arg)

    monkeypatch.setattr(
        wizard_window_module,
        "build_or_complete_single_fit_wizard_recommendations_for_global_portfolio",
        _fake_phase_one,
    )
    monkeypatch.setattr(
        wizard_window_module, "build_global_fit_wizard_screening_recommendation", _fake_build
    )
    window = GlobalFitWizardWindow()
    window.set_analysis_context(datasets)
    window._start_analysis()
    wait_for(lambda: _analysis_complete(window), qapp)

    # Each generated run's one Exponential fit took 0.02 s on its points.
    points = datasets[0].n_points
    assert shared_fit_time_store().samples == {"Exponential": (pytest.approx(20.0 / points),)}
    assert captured["phase_one"] == captured["screening"]
    window._picker._show_details("Exponential")
    assert window._picker._details.facts["Fitting cost"].text() == (
        "Quick — ≈ 0.02 s per run on this computer"
    )


def test_global_fit_wizard_window_warning_info_dialog_contains_expected_text(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}

    def _fake_information(_parent, title: str, text: str) -> None:
        captured["title"] = title
        captured["text"] = text

    monkeypatch.setattr(QMessageBox, "information", _fake_information)
    window = GlobalFitWizardWindow()
    window._show_warning_info()

    assert captured["title"] == "Global Fit Wizard Warnings"
    assert "abrupt change in the spectra" in captured["text"]


def _expectation_rows_by_name(window: GlobalFitWizardWindow) -> dict[str, int]:
    table = window._expectations_table
    return {table.item(row, 0).text(): row for row in range(table.rowCount())}


def test_global_fit_wizard_expectations_default_amplitudes_global_rates_local(
    qapp: QApplication,
    datasets: list[MuonDataset],
) -> None:
    """The embedded expectations table carries the old dialog's defaults."""
    window = GlobalFitWizardWindow()
    window.set_analysis_context(datasets)

    table = window._expectations_table
    row_by_name = _expectation_rows_by_name(window)
    amplitude_row = row_by_name["A_1"]
    lambda_row = row_by_name["Lambda"]
    background_row = row_by_name["A_bg"]

    amplitude_role = table.cellWidget(amplitude_row, 1)
    lambda_role = table.cellWidget(lambda_row, 1)
    background_role = table.cellWidget(background_row, 1)

    assert isinstance(amplitude_role, wizard_window_module.QComboBox)
    assert isinstance(lambda_role, wizard_window_module.QComboBox)
    assert isinstance(background_role, wizard_window_module.QComboBox)
    assert amplitude_role.currentText() == "Global"
    assert lambda_role.currentText() == "Local"
    assert background_role.currentText() == "Global"
    assert table.item(amplitude_row, 2).text() == "0, inf"
    assert table.item(lambda_row, 2).text() == "0, inf"
    assert table.item(background_row, 2).text() == "-inf, inf"


def test_global_fit_wizard_window_passes_expectation_types_and_bounds(
    qapp: QApplication,
    datasets: list[MuonDataset],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Edits to the embedded table flow to the builder and the applied-config signal."""
    captured: dict[str, object] = {}

    def _fake_build(datasets_arg, **kwargs):
        captured["types"] = kwargs.get("current_parameter_types")
        captured["bounds"] = kwargs.get("parameter_bounds")
        return _fake_screening_recommendation(datasets_arg)

    monkeypatch.setattr(
        wizard_window_module,
        "build_global_fit_wizard_screening_recommendation",
        _fake_build,
    )

    window = GlobalFitWizardWindow()
    emitted_config: dict[str, object] = {}
    window.parameter_setup_applied.connect(lambda config: emitted_config.update(config))
    window.set_analysis_context(datasets)

    row_by_name = _expectation_rows_by_name(window)
    window._expectations_table.item(row_by_name["A_bg"], 2).setText("-0.5, 0.5")

    window._start_analysis()
    wait_for(lambda: _analysis_complete(window), qapp)

    assert captured["types"] is not None
    assert captured["bounds"] is not None
    assert captured["types"]["A_1"] == "Global"
    assert captured["types"]["Lambda"] == "Local"
    assert captured["bounds"]["A_bg"] == (-0.5, 0.5)
    assert emitted_config["types"]["Lambda"] == "Local"
    assert emitted_config["bounds"]["A_bg"] == (-0.5, 0.5)


def test_global_fit_wizard_window_invalid_expectation_bounds_block_screening(
    qapp: QApplication,
    datasets: list[MuonDataset],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unparseable bounds stop the run inline instead of via a modal warning."""
    monkeypatch.setattr(
        wizard_window_module,
        "build_global_fit_wizard_screening_recommendation",
        lambda *_args, **_kwargs: pytest.fail("screening must not start with invalid bounds"),
    )
    window = GlobalFitWizardWindow()
    window.set_analysis_context(datasets)

    row_by_name = _expectation_rows_by_name(window)
    window._expectations_table.item(row_by_name["Lambda"], 2).setText("nonsense")

    window._start_analysis()

    assert window._analysis_in_progress is False
    assert window._tasks.active_count == 0
    assert window._stepper.current_key() == "scope"
    # The section expands to surface the inline error naming the parameter.
    assert window._expectations_section.isExpanded() is True
    assert "Lambda" in window._expectations_error_label.text()
    assert "parameter expectations" in window._status_label.text()


# ── Scope step: presence and builder wiring ───────────────────────────────────────────────────────────────────


def test_global_fit_wizard_window_opens_on_the_scope_step(
    qapp: QApplication,
    datasets: list[MuonDataset],
) -> None:
    window = GlobalFitWizardWindow()
    window.set_analysis_context(datasets)
    assert window._stepper.current_key() == "scope"
    assert _step(window, "scope")[0] is StepState.READY
    for key in ("screen", "compare", "phases"):
        assert _step(window, key) == (StepState.PENDING, "Not screened yet")
    assert _step(window, "apply") == (StepState.PENDING, "Pick a model first")


def test_global_fit_wizard_window_forwards_scope_to_screening(
    qapp: QApplication,
    datasets: list[MuonDataset],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_build(datasets_arg, **kwargs):
        captured.update(kwargs)
        return _fake_screening_recommendation(datasets_arg)

    monkeypatch.setattr(
        wizard_window_module,
        "build_global_fit_wizard_screening_recommendation",
        _fake_build,
    )
    window = GlobalFitWizardWindow()
    window.set_analysis_context(datasets)
    window._picker.set_scope(LF_DYNAMICS)

    window._start_analysis()
    wait_for(lambda: _analysis_complete(window), qapp)

    scope = captured.get("scope")
    assert scope == LF_DYNAMICS


def test_global_fit_wizard_window_forwards_scope_to_optimize(
    qapp: QApplication,
    datasets: list[MuonDataset],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        wizard_window_module,
        "build_global_fit_wizard_screening_recommendation",
        lambda datasets_arg, **_kwargs: _fake_screening_recommendation(datasets_arg),
    )

    def _fake_build(datasets_arg, **kwargs):
        captured.update(kwargs)
        return _fake_multi_variant_recommendation(datasets_arg)

    monkeypatch.setattr(
        wizard_window_module,
        "build_global_fit_wizard_recommendation",
        _fake_build,
    )
    window = GlobalFitWizardWindow()
    window.set_analysis_context(datasets)
    window._picker.set_scope(LF_DYNAMICS)
    _screen(window, qapp)
    _optimise(window, qapp, rows=2)

    scope = captured.get("scope")
    assert scope == LF_DYNAMICS


def test_global_fit_wizard_window_scope_in_analysis_signature(
    qapp: QApplication,
    datasets: list[MuonDataset],
) -> None:
    """A scope change invalidates the same-signature cache short-circuit."""
    window = GlobalFitWizardWindow()
    window.set_analysis_context(datasets)

    baseline = window._analysis_signature()
    assert baseline["scope"] == WizardScope().to_payload()

    window._picker.set_scope(LF_DYNAMICS)
    changed = window._analysis_signature()
    assert changed["scope"] == LF_DYNAMICS.to_payload()
    assert changed != baseline


# ── Staleness after a scope change ───────────────────────────────────────────


def test_global_fit_wizard_window_scope_change_marks_the_results_stale(
    qapp: QApplication,
    datasets: list[MuonDataset],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        wizard_window_module,
        "build_global_fit_wizard_screening_recommendation",
        lambda datasets_arg, **_kwargs: _fake_screening_recommendation(datasets_arg),
    )
    window = GlobalFitWizardWindow()
    window.set_analysis_context(datasets)
    _screen(window, qapp)

    assert window._optimise_btn.isEnabled() is True
    assert window._stale_banner.isHidden() is True
    assert _step(window, "screen")[0] is StepState.DONE

    previous = window.current_recommendation()
    window._show_step("scope")

    # A picker edit: look for spin dynamics.
    window._picker._chips[PhysicsClass.DYNAMICS].click()

    assert window._analysis_stale is True
    assert window._stale_banner.isHidden() is False
    assert window._optimise_btn.isEnabled() is False
    # Screen and Compare read stale; the results stay on show and reachable.
    assert _step(window, "screen")[0] is StepState.STALE
    assert _step(window, "compare")[0] is StepState.STALE
    assert window.current_recommendation() is previous
    window._stepper._buttons["screen"].click()
    assert window._stepper.current_key() == "screen"


# ── Field-direction answer ───────────────────────────────────────────────────


def test_global_fit_wizard_window_direction_answer_saves_on_unrecorded_runs(
    qapp: QApplication,
    datasets: list[MuonDataset],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        wizard_window_module,
        "build_global_fit_wizard_screening_recommendation",
        lambda datasets_arg, **_kwargs: _fake_screening_recommendation(datasets_arg),
    )
    datasets[0].metadata["field_direction"] = "Zero field"
    window = GlobalFitWizardWindow()
    window.set_analysis_context(datasets)
    _screen(window, qapp)
    answers: list[tuple] = []
    window.field_direction_answered.connect(lambda runs, geometry: answers.append((runs, geometry)))
    scopes: list = []
    window._picker.scope_changed.connect(scopes.append)

    window._picker._direction_group.button(1).click()  # Longitudinal

    assert datasets[0].metadata["field_direction"] == "Zero field"
    assert "field_direction_source" not in datasets[0].metadata
    assert [d.metadata["field_direction"] for d in datasets[1:]] == ["Longitudinal"] * 2
    assert [d.metadata["field_direction_source"] for d in datasets[1:]] == ["user"] * 2
    assert answers == [(frozenset({701, 702, 703}), FieldGeometry.LF)]
    assert scopes == []
    assert window._analysis_stale is True
    assert window._stale_banner.isHidden() is False
    assert window._optimise_btn.isEnabled() is False
    assert _step(window, "screen")[0] is StepState.STALE
    # The picker re-describes the runs: the answer is checked and noted.
    assert window._picker._direction_group.checkedButton().text() == "Longitudinal"
    assert window._picker._direction_note.text() == (
        "Set by you — saved on the 2 runs that record none; the files record the other 1."
    )
    assert _step(window, "scope")[1].startswith("Zero field / Longitudinal · ")

    window._picker._direction_group.button(3).click()  # Not recorded withdraws it
    assert all("field_direction" not in d.metadata for d in datasets[1:])
    assert answers[-1] == (frozenset({701, 702, 703}), None)


# ── Build Screening disabled when scope is invalid ───────────────────────────


def test_global_fit_wizard_window_build_disabled_when_scope_invalid(
    qapp: QApplication,
    datasets: list[MuonDataset],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = GlobalFitWizardWindow()
    window.set_analysis_context(datasets)
    assert window._refresh_btn.isEnabled() is True

    monkeypatch.setattr(window._picker, "is_valid", lambda: False)
    window._on_scope_validity_changed(False)

    assert window._refresh_btn.isEnabled() is False
    assert "at least one candidate family" in window._status_label.text()


# ── Cached restore with scope + legacy fallback ──────────────────────────────


def test_global_fit_wizard_window_cached_restore_with_scope(
    qapp: QApplication,
    datasets: list[MuonDataset],
) -> None:
    window = GlobalFitWizardWindow()
    window.set_analysis_context(datasets)

    window.set_cached_recommendation(
        _fake_recommendation(datasets),
        signature={
            "run_numbers": [int(dataset.run_number) for dataset in datasets],
            "model": None,
            "scope": LF_DYNAMICS.to_payload(),
        },
        log_text="cached",
    )

    assert window._picker.scope() == LF_DYNAMICS
    assert window._picker._chips[PhysicsClass.DYNAMICS].isChecked()
    assert window._analysis_stale is False
    assert window._stale_banner.isHidden() is True


def test_global_fit_wizard_window_cached_restore_legacy_signature_is_auto(
    qapp: QApplication,
    datasets: list[MuonDataset],
) -> None:
    window = GlobalFitWizardWindow()
    window.set_analysis_context(datasets)
    window.set_cached_recommendation(_fake_recommendation(datasets))

    # A signature without a scope keeps the default scope and is not stale.
    assert window._picker.scope() == WizardScope()
    assert window._analysis_stale is False
    assert window._stale_banner.isHidden() is True


# ── Single "Optimize" mode (PR 5 rework) ─────────────────────────────────────
# The four-position effort slider collapsed to one honest exact mode: every
# EffortTier now runs the exact engine, so the visible control is a single
# disabled item and the payload always records the exact tier.


def test_global_fit_wizard_window_effort_defaults_to_exhaustive(
    qapp: QApplication,
    datasets: list[MuonDataset],
) -> None:
    from asymmetry.core.fitting.wizard_scope import EffortTier

    window = GlobalFitWizardWindow()
    window.set_analysis_context(datasets)

    assert window.current_effort_tier() is EffortTier.EXHAUSTIVE


def test_global_fit_wizard_window_effort_control_is_single_disabled_optimize_item(
    qapp: QApplication,
    datasets: list[MuonDataset],
) -> None:
    """A 4-way slider where every position did the same work would be misleading.

    The visible control is one disabled item labelled as the exact optimize mode.
    """
    window = GlobalFitWizardWindow()
    window.set_analysis_context(datasets)

    assert window._effort_combo.count() == 1
    assert window._effort_combo.isEnabled() is False
    assert "optimize" in window._effort_combo.itemText(0).lower()


def test_global_fit_wizard_window_forwards_effort_tier_to_optimize(
    qapp: QApplication,
    datasets: list[MuonDataset],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from asymmetry.core.fitting.wizard_scope import EffortTier

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        wizard_window_module,
        "build_global_fit_wizard_screening_recommendation",
        lambda datasets_arg, **_kwargs: _fake_screening_recommendation(datasets_arg),
    )

    def _fake_build(datasets_arg, **kwargs):
        captured.update(kwargs)
        return _fake_multi_variant_recommendation(datasets_arg)

    monkeypatch.setattr(
        wizard_window_module,
        "build_global_fit_wizard_recommendation",
        _fake_build,
    )
    window = GlobalFitWizardWindow()
    window.set_analysis_context(datasets)
    _screen(window, qapp)
    _optimise(window, qapp, rows=2)

    # The single visible mode always forwards the exact tier.
    assert captured.get("effort_tier") is EffortTier.EXHAUSTIVE


def test_global_fit_wizard_window_effort_tier_in_analysis_signature(
    qapp: QApplication,
    datasets: list[MuonDataset],
) -> None:
    """The analysis signature records the exact tier (the single Optimize mode)."""
    window = GlobalFitWizardWindow()
    window.set_analysis_context(datasets)

    assert window._analysis_signature()["effort_tier"] == "exhaustive"


def test_global_fit_wizard_window_cached_restore_legacy_tier_stays_exact(
    qapp: QApplication,
    datasets: list[MuonDataset],
) -> None:
    """Restoring a legacy Low/Balanced payload stays on the exact Optimize mode.

    The one-item control cannot represent the retired heuristic tiers, and every
    tier runs the exact engine anyway, so a legacy payload degrades to the exact
    tier without being marked stale.
    """
    from asymmetry.core.fitting.wizard_scope import EffortTier

    window = GlobalFitWizardWindow()
    window.set_analysis_context(datasets)

    window.set_cached_recommendation(
        _fake_recommendation(datasets),
        signature={
            "run_numbers": [int(dataset.run_number) for dataset in datasets],
            "model": None,
            "effort_tier": "low",
        },
        log_text="cached",
    )

    assert window.current_effort_tier() is EffortTier.EXHAUSTIVE
    assert window._analysis_stale is False
    assert window._stale_banner.isHidden() is True


def test_global_fit_wizard_window_cached_restore_legacy_signature_is_exhaustive(
    qapp: QApplication,
    datasets: list[MuonDataset],
) -> None:
    from asymmetry.core.fitting.wizard_scope import EffortTier

    window = GlobalFitWizardWindow()
    window.set_analysis_context(datasets)
    window.set_cached_recommendation(_fake_recommendation(datasets))

    # Legacy signature (no effort_tier key) restores the exact tier, not stale.
    assert window.current_effort_tier() is EffortTier.EXHAUSTIVE
    assert window._analysis_stale is False
    assert window._stale_banner.isHidden() is True


# ── Cooperative cancel ───────────────────────────────────────────────────────


def test_cancel_in_the_progress_block_cancels_and_lands_on_the_origin(
    qapp: QApplication,
    datasets: list[MuonDataset],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The progress block's Cancel, shown while running, cooperatively cancels the run.

    Phase-one blocks on a threading.Event so the GUI thread can confirm the
    worker is live, click Cancel, then release phase-one. The worker's next
    cancel checkpoint raises FitCancelledError (declared in _cancel_exceptions);
    the base's cancelled slot clears busy, the progress block collapses to its
    Run log, and the window lands on the step the run started from. The
    screening builder must never run.
    """
    import threading

    released = threading.Event()

    def _blocking_build(datasets_arg, **kwargs):
        raise AssertionError("builder must observe cancellation first")

    def _phase_one(datasets_arg, **kwargs):
        # Block on the worker thread until the GUI thread has clicked Cancel.
        released.wait(timeout=5.0)
        return _fake_screening_table(datasets_arg, {})

    monkeypatch.setattr(
        wizard_window_module,
        "build_or_complete_single_fit_wizard_recommendations_for_global_portfolio",
        _phase_one,
    )
    monkeypatch.setattr(
        wizard_window_module,
        "build_global_fit_wizard_screening_recommendation",
        _blocking_build,
    )

    window = GlobalFitWizardWindow()
    window.set_analysis_context(datasets)
    assert window._run_progress.isHidden() is True

    window._start_analysis()
    # Busy: the progress block, with Cancel, runs in the Screen step.
    wait_for(lambda: window._tasks.active_count == 1, qapp)
    cancel = _run_progress_cancel(window)
    assert window._stepper.current_key() == "screen"
    assert _step(window, "screen") == (StepState.RUNNING, "Screening…")
    assert cancel.isVisibleTo(window) is True

    cancel.click()
    released.set()
    wait_for(lambda: window._tasks.active_count == 0, qapp)

    assert window._analysis_in_progress is False
    assert cancel.isVisibleTo(window) is False
    assert "cancelled" in window._status_label.text().lower()
    assert window._stepper.current_key() == "scope"
    assert _step(window, "screen") == (StepState.PENDING, "Not screened yet")
    window.close()


def test_worker_task_cancel_between_phases_raises_and_skips_builder(
    qapp: QApplication,
    datasets: list[MuonDataset],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancel before the phase-one boundary skips the builders entirely.

    Drives the worker task closure through a real TaskWorker.run() (no thread):
    a pre-cancelled worker makes _run_global_fit_wizard_analysis raise
    FitCancelledError at its first checkpoint, so TaskWorker emits cancelled and
    the screening builder is never called.
    """
    from asymmetry.core.fitting.engine import FitCancelledError
    from asymmetry.gui.tasks import TaskWorker

    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("builder must not run after cancellation")

    monkeypatch.setattr(
        wizard_window_module,
        "build_global_fit_wizard_screening_recommendation",
        _fail_if_called,
    )
    monkeypatch.setattr(
        wizard_window_module,
        "build_or_complete_single_fit_wizard_recommendations_for_global_portfolio",
        _fail_if_called,
    )

    window = GlobalFitWizardWindow()
    window.set_analysis_context(datasets)
    task = window._create_worker_task(window._analysis_request_id)

    worker = TaskWorker(task, cancel_exceptions=(FitCancelledError,))
    cancelled: list[bool] = []
    errored: list[str] = []
    worker.cancelled.connect(lambda: cancelled.append(True))
    worker.error.connect(lambda message: errored.append(message))
    worker.cancel()
    worker.run()

    assert cancelled == [True]
    assert errored == []


# ── Confidence / verdict display ─────────────────────────────────────────────


def _single_fit_with(
    dataset: MuonDataset,
    *,
    confidence: ConfidenceTier,
    verdict: RecommendationVerdict,
    caveat: str = "",
) -> FitWizardRecommendation:
    base = _fake_single_fit_recommendation(dataset)
    return replace(base, confidence=confidence, verdict=verdict, caveat=caveat)


def test_global_fit_wizard_window_overview_shows_confidence_and_caveat(
    qapp: QApplication,
    datasets: list[MuonDataset],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    per_run = {
        int(dataset.run_number): _single_fit_with(
            dataset,
            confidence=ConfidenceTier.MEDIUM,
            verdict=RecommendationVerdict.STRUCTURED,
            caveat="Residuals fail the whiteness gate.",
        )
        for dataset in datasets
    }
    monkeypatch.setattr(
        wizard_window_module,
        "build_or_complete_single_fit_wizard_recommendations_for_global_portfolio",
        lambda datasets_arg, **kwargs: _fake_screening_table(
            datasets_arg,
            per_run,
            store=kwargs.get("existing_recommendations_by_run"),
            generated=tuple(int(dataset.run_number) for dataset in datasets_arg),
        ),
    )
    monkeypatch.setattr(
        wizard_window_module,
        "build_global_fit_wizard_screening_recommendation",
        lambda datasets_arg, **_kwargs: _fake_screening_recommendation(datasets_arg),
    )

    window = GlobalFitWizardWindow()
    window.set_analysis_context(datasets)
    window._start_analysis()
    wait_for(lambda: _analysis_complete(window), qapp)

    assert window._overview_table.columnCount() == 8
    assert window._overview_table.horizontalHeaderItem(6).text() == "Confidence"
    confidence_cell = window._overview_table.item(0, 6)
    assert confidence_cell.text() == "Medium"
    assert "whiteness" in confidence_cell.toolTip()
    # No null-structure runs → the series verdict banner stays hidden.
    assert window._verdict_banner.isHidden() is True


def test_global_fit_wizard_window_marks_no_significant_structure_run(
    qapp: QApplication,
    datasets: list[MuonDataset],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flagged_run = int(datasets[0].run_number)
    per_run = {}
    for dataset in datasets:
        run_number = int(dataset.run_number)
        if run_number == flagged_run:
            per_run[run_number] = _single_fit_with(
                dataset,
                confidence=ConfidenceTier.NONE,
                verdict=RecommendationVerdict.NO_SIGNIFICANT_STRUCTURE,
                caveat="Best template does not beat a flat baseline.",
            )
        else:
            per_run[run_number] = _single_fit_with(
                dataset,
                confidence=ConfidenceTier.HIGH,
                verdict=RecommendationVerdict.STRUCTURED,
            )
    monkeypatch.setattr(
        wizard_window_module,
        "build_or_complete_single_fit_wizard_recommendations_for_global_portfolio",
        lambda datasets_arg, **kwargs: _fake_screening_table(
            datasets_arg,
            per_run,
            store=kwargs.get("existing_recommendations_by_run"),
            generated=tuple(int(dataset.run_number) for dataset in datasets_arg),
        ),
    )
    monkeypatch.setattr(
        wizard_window_module,
        "build_global_fit_wizard_screening_recommendation",
        lambda datasets_arg, **_kwargs: _fake_screening_recommendation(datasets_arg),
    )

    window = GlobalFitWizardWindow()
    window.set_analysis_context(datasets)
    window._start_analysis()
    wait_for(lambda: _analysis_complete(window), qapp)

    # Series-level banner is unmissable and names the flagged run.
    assert window._verdict_banner.isHidden() is False
    banner_text = window._verdict_banner.text()
    assert "No significant structure" in banner_text
    assert datasets[0].run_label in banner_text

    # The flagged run's row is marked (rows follow dataset_order == input order).
    flagged_row = next(
        row
        for row in range(window._overview_table.rowCount())
        if window._overview_table.item(row, 0).text() == datasets[0].run_label
    )
    recommendation_cell = window._overview_table.item(flagged_row, 7)
    assert recommendation_cell.text() == "No significant structure"
    from asymmetry.gui.styles import tokens

    assert recommendation_cell.foreground().color().name() == QColor(tokens.ERROR).name()


def test_global_fit_wizard_window_confidence_survives_cache_restore(
    qapp: QApplication,
    datasets: list[MuonDataset],
) -> None:
    """On a cached reopen, the overview confidence/verdict cells still render.

    Confidence/verdict live only on the per-run single-fit recommendations. The
    tab feeds those into set_analysis_context (existing_single_fit_recommendations_by_run)
    before set_cached_recommendation, so a restored recommendation still shows
    per-run confidence and the null-structure banner.
    """
    per_run = {
        int(datasets[0].run_number): _single_fit_with(
            datasets[0],
            confidence=ConfidenceTier.NONE,
            verdict=RecommendationVerdict.NO_SIGNIFICANT_STRUCTURE,
            caveat="Flat baseline wins.",
        ),
    }
    for dataset in datasets[1:]:
        per_run[int(dataset.run_number)] = _single_fit_with(
            dataset,
            confidence=ConfidenceTier.HIGH,
            verdict=RecommendationVerdict.STRUCTURED,
        )

    window = GlobalFitWizardWindow()
    window.set_analysis_context(
        datasets,
        existing_single_fit_recommendations_by_run=per_run,
    )
    window.set_cached_recommendation(
        _fake_recommendation(datasets),
        signature={
            "run_numbers": [int(dataset.run_number) for dataset in datasets],
            "model": None,
            "scope": WizardScope().to_payload(),
        },
        log_text="cached",
    )

    # Null-structure banner fires and the flagged run's confidence cell renders.
    assert window._verdict_banner.isHidden() is False
    assert datasets[0].run_label in window._verdict_banner.text()
    flagged_row = next(
        row
        for row in range(window._overview_table.rowCount())
        if window._overview_table.item(row, 0).text() == datasets[0].run_label
    )
    assert window._overview_table.item(flagged_row, 7).text() == "No significant structure"


# ── Screen and Compare ────────────────────────────────────────────────────────────────────


def test_global_fit_wizard_window_cached_optimised_result_lands_on_compare(
    qapp: QApplication,
    datasets: list[MuonDataset],
) -> None:
    """A cached optimised recommendation lands on Compare with A drawn and trended."""
    window = GlobalFitWizardWindow()
    window.set_analysis_context(datasets)
    window.set_cached_recommendation(_fake_recommendation(datasets), log_text="Cached run.\nDone.")

    assert window._stepper.current_key() == "compare"
    panel = window._compare_panel
    assert panel.a_key() == "exp_constant"
    a = panel._canvas._a
    assert [run.run_number for run in a.runs] == [701, 702, 703]
    assert [run.axis_value for run in a.runs] == [100.0, 200.0, 300.0]
    # The first local parameter (Lambda) trends across the series axis.
    assert panel._trend == "Lambda"
    assert panel._continue.isEnabled() is True
    assert window._roles_table.rowCount() == 3
    # The cached log is restored in the landing step's Run log.
    assert window.current_log_text() == "Cached run.\nDone."
    assert window._run_progress.parentWidget() is window._progress_hosts["compare"]
    assert window._run_progress.findChild(PanelSection).title() == "Run log"
    assert not window._run_progress.isHidden()


def test_global_fit_wizard_window_screen_previews_the_selected_family(
    qapp: QApplication,
    datasets: list[MuonDataset],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Screen previews the selected family's per-run fits; Compare waits for an optimise."""
    monkeypatch.setattr(
        wizard_window_module,
        "build_global_fit_wizard_screening_recommendation",
        lambda datasets_arg, **_kwargs: _fake_screening_recommendation(datasets_arg),
    )
    window = GlobalFitWizardWindow()
    window.set_analysis_context(datasets)
    _screen(window, qapp)

    preview = window._screen_canvas._a
    assert preview.key == "exp_constant"
    assert preview.prescreen is True
    assert len(preview.runs) == len(datasets)
    assert window._compare_panel.a_key() is None
    assert window._compare_body.isHidden()
    window._compare_empty.findChild(QPushButton).click()
    assert window._stepper.current_key() == "screen"


def test_global_fit_wizard_window_optimise_button_label_tracks_the_ticks(
    qapp: QApplication,
    datasets: list[MuonDataset],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        wizard_window_module,
        "build_global_fit_wizard_screening_recommendation",
        lambda datasets_arg, **_kwargs: _fake_screening_recommendation(datasets_arg),
    )
    window = GlobalFitWizardWindow()
    window.set_analysis_context(datasets)
    _screen(window, qapp)

    assert window._optimise_btn.text() == "Optimise 1 family →"
    assert window._optimise_btn.isEnabled() is True

    title = window._leaderboard._table.item(0, 0)
    title.setCheckState(Qt.CheckState.Unchecked)

    assert window._optimise_btn.text() == "Optimise 0 families →"
    assert window._optimise_btn.isEnabled() is False

    title.setCheckState(Qt.CheckState.Checked)

    assert window._optimise_btn.text() == "Optimise 1 family →"
    assert window._optimise_btn.isEnabled() is True


# ── Navigation and landing ────────────────────────────────────────────────────────────────────────────────────────


def test_global_fit_wizard_window_stepper_navigates_both_ways(
    qapp: QApplication,
    datasets: list[MuonDataset],
) -> None:
    window = GlobalFitWizardWindow()
    window.set_analysis_context(datasets)
    window.set_cached_recommendation(_fake_recommendation(datasets))
    assert window._stepper.current_key() == "compare"

    window._stepper._buttons["scope"].click()
    assert window._stepper.current_key() == "scope"
    assert window._stack.currentWidget() is window._views["scope"]
    assert window._refresh_btn.isEnabled() is True

    window._stepper._buttons["compare"].click()
    assert window._stepper.current_key() == "compare"
    assert window._stack.currentWidget() is window._views["compare"]
    # A pending step is not a way in.
    assert window._stepper._buttons["apply"].isEnabled() is False


def test_global_fit_wizard_window_metric_rerank_stays_on_the_current_step(
    qapp: QApplication,
    datasets: list[MuonDataset],
) -> None:
    recommendation = _fake_multi_variant_recommendation(datasets)
    _best, shared = recommendation.assessments
    window = GlobalFitWizardWindow()
    window.set_analysis_context(datasets)
    window.set_cached_recommendation(recommendation)
    window._compare_panel.set_a(shared.selection_key)

    # On Compare: the re-rank stays there and keeps the picked A.
    window._metric_combo.setCurrentText(SelectionMetric.BIC.value)
    assert window.current_recommendation().metric is SelectionMetric.BIC
    assert window._stepper.current_key() == "compare"
    assert window._compare_panel.a_key() == shared.selection_key
    assert window._compare_panel._caption.text() == "Role splits · ΔBIC from best"

    window._stepper._buttons["scope"].click()
    window._metric_combo.setCurrentText(SelectionMetric.AIC.value)
    assert window.current_recommendation().metric is SelectionMetric.AIC
    assert window._stepper.current_key() == "scope"


def test_global_fit_wizard_window_failed_rescreen_lands_on_scope_with_runs(
    qapp: QApplication,
    datasets: list[MuonDataset],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail(*_args, **_kwargs):
        raise RuntimeError("screening exploded")

    monkeypatch.setattr(
        wizard_window_module, "build_global_fit_wizard_screening_recommendation", _fail
    )
    window = GlobalFitWizardWindow()
    window.set_analysis_context(datasets)
    window.set_cached_recommendation(_fake_recommendation(datasets))
    window._stepper._buttons["scope"].click()
    # Change the scope so Run screening recomputes instead of serving the cache.
    window._picker._chips[PhysicsClass.DYNAMICS].click()

    window._refresh_btn.click()
    wait_for(lambda: window._tasks.active_count == 0, qapp)

    assert "screening exploded" in window._status_label.text()
    assert window.current_recommendation() is None
    assert window._stepper.current_key() == "scope"
    assert window._overview_table.rowCount() == len(datasets)
    assert _step(window, "screen") == (StepState.PENDING, "Not screened yet")
    assert window._stepper._buttons["compare"].isEnabled() is False


# ── Picking A, continuing and applying ─────────────────────────────────────────────────────────────────


def test_global_fit_wizard_window_picking_a_redraws_and_apply_applies_it(
    qapp: QApplication,
    datasets: list[MuonDataset],
) -> None:
    recommendation = _fake_multi_variant_recommendation(datasets)
    best, shared = recommendation.assessments
    window = GlobalFitWizardWindow()
    window.set_analysis_context(datasets)
    window.set_cached_recommendation(recommendation)
    panel = window._compare_panel
    # The recommended variant trends its local Lambda.
    assert panel.a_key() == best.selection_key
    assert panel._trend == "Lambda"

    applied: list[GlobalCandidateAssessment] = []
    window.apply_assessment_requested.connect(
        lambda assessment, _recommendation: applied.append(assessment)
    )

    # A → the all-global variant: no local parameter left to trend, and
    # Continue → Apply hands back the variant Compare now draws.
    panel.set_a(shared.selection_key)
    assert panel._canvas._a.key == shared.selection_key
    assert panel._trend is None
    panel._continue.click()
    assert window._apply_roles_section.title() == "Parameter roles"
    window._apply_btn.click()
    assert [assessment.selection_key for assessment in applied] == [shared.selection_key]

    panel.set_a(best.selection_key)
    assert panel._canvas._a.key == best.selection_key
    assert panel._trend == "Lambda"


def test_global_fit_wizard_window_cached_screening_lands_on_screen(
    qapp: QApplication,
    datasets: list[MuonDataset],
) -> None:
    window = GlobalFitWizardWindow()
    window.set_analysis_context(datasets)
    window.set_cached_recommendation(_fake_screening_recommendation(datasets))

    assert window._stepper.current_key() == "screen"
    # No cached log, so no Run log either.
    assert window._run_progress.isHidden()
    assert window._leaderboard.ticked() == ("exp_constant",)


def test_global_fit_wizard_window_cached_restore_needs_the_context_runs(
    qapp: QApplication,
    datasets: list[MuonDataset],
) -> None:
    window = GlobalFitWizardWindow()
    window.set_analysis_context(datasets[:2])
    with pytest.raises(ValueError, match=r"runs \[703\] are missing"):
        window.set_cached_recommendation(_fake_recommendation(datasets))


def test_global_fit_wizard_window_failed_optimise_lands_on_screen(
    qapp: QApplication,
    datasets: list[MuonDataset],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail(*_args, **_kwargs):
        raise RuntimeError("optimise exploded")

    monkeypatch.setattr(
        wizard_window_module,
        "build_global_fit_wizard_screening_recommendation",
        lambda datasets_arg, **_kwargs: _fake_screening_recommendation(datasets_arg),
    )
    monkeypatch.setattr(wizard_window_module, "build_global_fit_wizard_recommendation", _fail)
    window = GlobalFitWizardWindow()
    window.set_analysis_context(datasets)
    _screen(window, qapp)

    window._optimise_btn.click()
    assert window._stepper.current_key() == "compare"
    assert window._leaderboard._table.item(0, 5).text() == "Running"
    wait_for(lambda: window._tasks.active_count == 0, qapp)

    assert "optimise exploded" in window._status_label.text()
    assert window._stepper.current_key() == "screen"
    assert window._leaderboard._table.item(0, 5).text() == "Not optimised"
    assert _step(window, "compare") == (StepState.READY, "Next: optimise the shortlist")


def test_the_stepper_follows_screening_optimise_and_apply(
    qapp: QApplication,
    datasets: list[MuonDataset],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        wizard_window_module,
        "build_global_fit_wizard_screening_recommendation",
        lambda datasets_arg, **_kwargs: _fake_screening_recommendation(datasets_arg),
    )
    monkeypatch.setattr(
        wizard_window_module,
        "build_global_fit_wizard_recommendation",
        lambda datasets_arg, **_kwargs: _fake_multi_variant_recommendation(datasets_arg),
    )
    window = GlobalFitWizardWindow()
    window.set_analysis_context(datasets)
    _screen(window, qapp)

    assert _step(window, "scope")[0] is StepState.DONE
    assert _step(window, "scope")[1] == window._picker.summary()
    assert _step(window, "screen") == (StepState.DONE, "Exponential + Constant leads")
    assert _step(window, "compare") == (StepState.READY, "Next: optimise the shortlist")
    assert _step(window, "phases") == (StepState.SKIPPED, "No transition found")
    assert _step(window, "apply") == (StepState.PENDING, "Pick a model first")

    _optimise(window, qapp, rows=2)
    assert _step(window, "compare") == (StepState.DONE, "2 role splits optimised")

    window._compare_panel._continue.click()
    assert window._stepper.current_key() == "apply"
    assert _step(window, "apply") == (StepState.READY, "Ready: Exponential + Constant")
    assert window._apply_note.text() == (
        "Review what will be handed to the global fit tab, then apply it."
    )
    window._apply_btn.click()
    assert _step(window, "apply") == (StepState.DONE, "Applied: Exponential + Constant")
    assert window._apply_note.text() == "Applied to the global fit tab."


def test_the_apply_step_reviews_roles_values_and_rationale(
    qapp: QApplication,
    datasets: list[MuonDataset],
) -> None:
    window = GlobalFitWizardWindow()
    window.set_analysis_context(datasets)
    window.set_cached_recommendation(_fake_recommendation(datasets))
    window._compare_panel._continue.click()

    assert window._apply_title.text() == "Exponential + Constant"
    rows = window._apply_roles._grid
    assert [rows.itemAtPosition(row, 1).widget().text() for row in range(3)] == [
        "A_1, A_bg",
        "λ",
        "none",
    ]
    table = window._apply_values_table
    # Starting values come from the first run's fit, as the global fit tab takes them.
    assert [table.item(row, 1).text() for row in range(table.rowCount())] == [
        "0.2",
        "0.2",
        "0.01",
    ]
    assert "run 701" in window._apply_values_section._hint_label.text()
    assert window._apply_why.isExpanded() is False
    assert "Rate variation is strongly supported." in window._apply_rationale.text()
    assert window._apply_warnings.isHidden()


def test_a_rerank_keeps_the_users_ticks(
    qapp: QApplication,
    datasets: list[MuonDataset],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        wizard_window_module,
        "build_global_fit_wizard_screening_recommendation",
        lambda datasets_arg, **_kwargs: _fake_screening_recommendation(datasets_arg),
    )
    window = GlobalFitWizardWindow()
    window.set_analysis_context(datasets)
    _screen(window, qapp)
    window._leaderboard._table.item(0, 0).setCheckState(Qt.CheckState.Unchecked)

    window._metric_combo.setCurrentText(SelectionMetric.BIC.value)

    # The shortlist pre-ticks only a new screening, never a re-rank of it.
    assert window._leaderboard.ticked() == ()
    assert window._stepper.current_key() == "screen"
