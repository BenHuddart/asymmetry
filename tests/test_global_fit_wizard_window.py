"""Tests for the global fit wizard window UI."""

from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pytest

pytestmark = [pytest.mark.gui, pytest.mark.slow, pytest.mark.integration]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QMessageBox

import asymmetry.gui.windows.global_fit_wizard_window as wizard_window_module
from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.fit_wizard import (
    CandidateAssessment,
    CandidateTemplate,
    FitWizardRecommendation,
    SelectionMetric,
    SpectrumFingerprint,
)
from asymmetry.core.fitting.global_fit_wizard import (
    GlobalCandidateAssessment,
    GlobalFitWizardRecommendation,
    GlobalParameterRecommendation,
    RunResidualDiagnostic,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.gui.windows.global_fit_wizard_window import (
    GlobalFitWizardParameterSetupDialog,
    GlobalFitWizardWindow,
)
from tests._qt_helpers import wait_for


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


@pytest.fixture(autouse=True)
def auto_parameter_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    def _config(self: GlobalFitWizardWindow, portfolio) -> dict[str, object]:
        names = []
        seen = set()
        for template in portfolio.templates:
            for name in template.model.param_names:
                if name in seen:
                    continue
                seen.add(name)
                names.append(name)
        return {
            "types": {
                name: (
                    "Global" if name.startswith("A") else ("Local" if name != "A_bg" else "Global")
                )
                for name in names
            },
            "bounds": {
                name: ((-float("inf"), float("inf")) if name == "A_bg" else (0.0, float("inf")))
                for name in names
            },
        }

    monkeypatch.setattr(GlobalFitWizardWindow, "_prompt_parameter_setup", _config)


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
                total_variation=0.0,
                roughness=0.0,
                rationale="Shared amplitude is adequate.",
            ),
            GlobalParameterRecommendation(
                name="Lambda",
                recommended_role="Local",
                global_score=15.0,
                local_score=9.0,
                score_delta=6.0,
                total_variation=1.8,
                roughness=0.2,
                rationale="Rate variation is strongly supported.",
            ),
            GlobalParameterRecommendation(
                name="A_bg",
                recommended_role="Global",
                global_score=10.0,
                local_score=11.0,
                score_delta=1.0,
                total_variation=0.0,
                roughness=0.0,
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

    window._start_analysis()
    wait_for(lambda: _analysis_complete(window), qapp)

    assert "Field" in window._overview_banner.text()
    assert window._tabs.count() == 6
    assert window._tabs.tabText(1) == "2. Candidate Portfolio"
    assert window._tabs.tabText(2) == "3. Single-Fit Screening"
    assert window._overview_table.rowCount() == len(datasets)
    assert window._portfolio_table.rowCount() == 1
    assert window._compare_table.rowCount() == 1
    assert window._optimized_table.rowCount() == 0
    assert window._roles_table.rowCount() == 0
    assert window._portfolio_table.columnWidth(0) >= 420
    assert window._compare_table.columnWidth(0) >= 420


def test_global_fit_wizard_window_apply_recommended_emits_assessment(
    qapp: QApplication,
    datasets: list[MuonDataset],
) -> None:
    window = GlobalFitWizardWindow()
    window.set_cached_recommendation(_fake_recommendation(datasets))

    emitted: dict[str, object] = {}
    window.apply_assessment_requested.connect(
        lambda assessment, recommendation: emitted.update(
            {"assessment": assessment, "recommendation": recommendation}
        )
    )

    window._apply_recommended_fit()

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

    window._start_analysis()
    wait_for(lambda: _analysis_complete(window), qapp)

    assert window._log_window is not None
    assert window._log_window.isVisible()
    log_text = window._log_window.to_plain_text()
    assert "Starting screening for 3 datasets." in log_text
    assert "Preparing missing single-fit wizard tables for global screening." in log_text
    assert "Single-fit table 701: evaluating shared candidate portfolio." in log_text


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
    window._start_analysis()
    wait_for(lambda: _analysis_complete(window), qapp)
    assert window._log_window is not None
    window._log_window.hide()
    qapp.processEvents()

    window._compare_table.selectRow(0)
    window._on_compare_selection_changed()
    window._start_selected_optimisation()
    wait_for(lambda: _analysis_complete(window) and window._optimized_table.rowCount() == 2, qapp)

    assert captured["datasets"] == datasets
    assert captured["selected_template_keys"] == ("exp_constant",)
    assert window._compare_table.item(0, 5).text() == "Optimized"
    assert window._optimized_table.columnWidth(0) >= 420
    assert window._optimized_table.item(0, 6).text() == "A_1, A_bg"
    assert window._optimized_table.item(0, 7).text() == "Lambda"
    assert window._optimized_table.item(1, 6).text() == "A_1, Lambda, A_bg"
    assert window._optimized_table.item(1, 7).text() == "None"
    assert window._log_window is not None
    assert window._log_window.isVisible()
    log_text = window._log_window.to_plain_text()
    assert "Starting coupled global optimisation for: Exponential + Constant." in log_text
    assert "Coupled optimisation 1/1: Exponential + Constant." in log_text
    assert "Completed coupled optimisation for Exponential + Constant." in log_text


def test_global_fit_wizard_window_emits_generated_single_fit_tables(
    qapp: QApplication,
    datasets: list[MuonDataset],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated_run = int(datasets[0].run_number)
    phase_one_recommendations = {
        int(dataset.run_number): _fake_single_fit_recommendation(dataset) for dataset in datasets
    }
    original_generated_recommendation = phase_one_recommendations[generated_run]
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        wizard_window_module,
        "build_or_complete_single_fit_wizard_recommendations_for_global_portfolio",
        lambda datasets_arg, **_kwargs: (
            wizard_window_module.build_global_fit_wizard_candidate_portfolio(datasets_arg),
            phase_one_recommendations,
            (generated_run,),
        ),
    )

    def _fake_build(datasets_arg, **kwargs):
        captured["single_fit"] = kwargs.get("single_fit_recommendations_by_run")
        single_fit = kwargs.get("single_fit_recommendations_by_run")
        if isinstance(single_fit, dict):
            single_fit[generated_run] = replace(
                single_fit[generated_run],
                summary="Repaired by screening.",
            )
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
    assert emitted[generated_run] is not original_generated_recommendation
    assert emitted[generated_run].summary == "Repaired by screening."
    assert captured["single_fit"] == phase_one_recommendations


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
    assert "continuity diagnostics" in captured["text"]


def test_global_fit_wizard_parameter_setup_dialog_defaults_amplitudes_global_rates_local(
    qapp: QApplication,
    datasets: list[MuonDataset],
) -> None:
    portfolio = wizard_window_module.build_global_fit_wizard_candidate_portfolio(datasets)
    dialog = GlobalFitWizardParameterSetupDialog(
        portfolio,
        current_parameter_types={},
        current_parameter_bounds={},
    )

    row_by_name = {
        dialog._table.item(row, 0).text(): row  # type: ignore[attr-defined]
        for row in range(dialog._table.rowCount())  # type: ignore[attr-defined]
    }
    amplitude_row = row_by_name["A_1"]
    lambda_row = row_by_name["Lambda"]
    background_row = row_by_name["A_bg"]

    amplitude_role = dialog._table.cellWidget(amplitude_row, 1)  # type: ignore[attr-defined]
    lambda_role = dialog._table.cellWidget(lambda_row, 1)  # type: ignore[attr-defined]
    background_role = dialog._table.cellWidget(background_row, 1)  # type: ignore[attr-defined]

    assert isinstance(amplitude_role, wizard_window_module.QComboBox)
    assert isinstance(lambda_role, wizard_window_module.QComboBox)
    assert isinstance(background_role, wizard_window_module.QComboBox)
    assert amplitude_role.currentText() == "Global"
    assert lambda_role.currentText() == "Local"
    assert background_role.currentText() == "Global"
    assert dialog._table.item(amplitude_row, 2).text() == "0, inf"  # type: ignore[attr-defined]
    assert dialog._table.item(lambda_row, 2).text() == "0, inf"  # type: ignore[attr-defined]
    assert dialog._table.item(background_row, 2).text() == "-inf, inf"  # type: ignore[attr-defined]


def test_global_fit_wizard_window_passes_dialog_adjusted_types_and_bounds(
    qapp: QApplication,
    datasets: list[MuonDataset],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    monkeypatch.setattr(
        GlobalFitWizardWindow,
        "_prompt_parameter_setup",
        lambda self, _portfolio: {
            "types": {"A_1": "Global", "Lambda": "Local", "A_bg": "Global"},
            "bounds": {"A_1": (0.0, 1.0), "Lambda": (0.0, 5.0), "A_bg": (-0.5, 0.5)},
        },
    )

    window = GlobalFitWizardWindow()
    window.set_analysis_context(datasets)
    window._start_analysis()
    wait_for(lambda: _analysis_complete(window), qapp)

    assert captured["types"] is not None
    assert captured["bounds"] is not None
    assert captured["types"]["A_1"] == "Global"
    assert captured["types"]["Lambda"] == "Local"
    assert captured["bounds"]["A_bg"] == (-0.5, 0.5)
