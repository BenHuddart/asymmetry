"""Tests for the fit wizard window UI."""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QMessageBox

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
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.gui.windows.fit_wizard_window import FitWizardWindow
import asymmetry.gui.windows.fit_wizard_window as wizard_window_module


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def dataset() -> MuonDataset:
    t = np.linspace(0.0, 8.0, 120)
    y = 0.2 * np.exp(-0.4 * t) + 0.01
    e = np.full_like(t, 0.01)
    return MuonDataset(time=t, asymmetry=y, error=e, metadata={"run_number": 101})


def _fake_recommendation(dataset: MuonDataset) -> FitWizardRecommendation:
    exp_model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    gauss_model = CompositeModel(["Gaussian", "Constant"], operators=["+"])

    exp_params = ParameterSet(
        [
            Parameter("A_1", value=0.2, min=0.0, max=1.0),
            Parameter("Lambda", value=0.4, min=0.0, max=5.0),
            Parameter("A_bg", value=0.01, min=0.0, max=0.5),
        ]
    )
    exp_curve = exp_model.function(dataset.time, A_1=0.2, Lambda=0.4, A_bg=0.01)
    exp_result = FitResult(
        success=True,
        chi_squared=5.0,
        reduced_chi_squared=0.05,
        parameters=exp_params,
        uncertainties={"A_1": 0.01, "Lambda": 0.02, "A_bg": 0.001},
        residuals=np.asarray(dataset.asymmetry - exp_curve, dtype=float),
        message="ok",
    )

    gauss_params = ParameterSet(
        [
            Parameter("A_1", value=0.18, min=0.0, max=1.0),
            Parameter("sigma", value=0.6, min=0.0, max=5.0),
            Parameter("A_bg", value=0.02, min=0.0, max=0.5),
        ]
    )
    gauss_curve = gauss_model.function(dataset.time, A_1=0.18, sigma=0.6, A_bg=0.02)
    gauss_result = FitResult(
        success=True,
        chi_squared=9.0,
        reduced_chi_squared=0.09,
        parameters=gauss_params,
        uncertainties={"A_1": 0.02, "sigma": 0.03, "A_bg": 0.002},
        residuals=np.asarray(dataset.asymmetry - gauss_curve, dtype=float),
        message="ok",
    )

    fingerprint = SpectrumFingerprint(
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
    )

    exp_template = CandidateTemplate(
        key="exp_constant",
        title="Exponential + Constant",
        category="General",
        rationale="Baseline single-relaxation model.",
        model=exp_model,
    )
    gauss_template = CandidateTemplate(
        key="gaussian_constant",
        title="Gaussian + Constant",
        category="General",
        rationale="Alternative Gaussian envelope.",
        model=gauss_model,
    )

    exp_assessment = CandidateAssessment(
        template=exp_template,
        fit_result=exp_result,
        aic=8.0,
        aicc=8.2,
        bic=10.0,
        selected_score=8.2,
        residual_rms=0.9,
        runs_z_score=0.2,
        max_abs_autocorrelation=0.1,
        residual_fft_peak_snr=1.2,
        residual_gate_passed=True,
        residual_gate_reasons=(),
        bound_hits=(),
        fitted_time=np.asarray(dataset.time, dtype=float).copy(),
        fitted_curve=np.asarray(exp_curve, dtype=float),
        component_curves=tuple(exp_model.evaluate_components(dataset.time, additive_only=True, A_1=0.2, Lambda=0.4, A_bg=0.01)),
    )
    gauss_assessment = CandidateAssessment(
        template=gauss_template,
        fit_result=gauss_result,
        aic=12.0,
        aicc=12.2,
        bic=14.0,
        selected_score=12.2,
        residual_rms=1.5,
        runs_z_score=2.5,
        max_abs_autocorrelation=0.4,
        residual_fft_peak_snr=7.0,
        residual_gate_passed=False,
        residual_gate_reasons=("runs-test z score suggests structure (2.50)",),
        bound_hits=(),
        fitted_time=np.asarray(dataset.time, dtype=float).copy(),
        fitted_curve=np.asarray(gauss_curve, dtype=float),
        component_curves=tuple(gauss_model.evaluate_components(dataset.time, additive_only=True, A_1=0.18, sigma=0.6, A_bg=0.02)),
    )
    return FitWizardRecommendation(
        fingerprint=fingerprint,
        templates=(exp_template, gauss_template),
        assessments=(exp_assessment, gauss_assessment),
        metric=SelectionMetric.AICC,
        recommended_key="exp_constant",
        comparable_keys=(),
        summary="Recommended: Exponential + Constant by AICc.",
    )


def _wait_for(predicate, qapp: QApplication, timeout_s: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Timed out waiting for fit wizard UI state")


def _analysis_complete(window: FitWizardWindow) -> bool:
    return window._recommendation is not None and window._analysis_thread is None


def test_fit_wizard_window_populates_banners_and_tables(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        wizard_window_module,
        "build_fit_wizard_recommendation",
        lambda dataset, current_model=None, metric=SelectionMetric.AICC: _fake_recommendation(dataset),
    )
    window = FitWizardWindow()

    window.set_analysis_context(dataset)
    assert "Click Start Analysis" in window._status_label.text()
    assert window._portfolio_table.rowCount() == 0

    window._start_analysis()
    _wait_for(lambda: _analysis_complete(window), qapp)

    assert window._fingerprint_banner.text()
    assert window._portfolio_banner.text()
    assert window._compare_banner.text()
    assert window._apply_banner.text()
    assert window._portfolio_table.rowCount() == 2
    assert window._compare_table.rowCount() == 2
    assert window._apply_parameters_table.rowCount() == 3


def test_fit_wizard_window_metric_info_dialog_contains_expected_text(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}

    def _fake_information(_parent, title: str, text: str) -> None:
        captured["title"] = title
        captured["text"] = text

    monkeypatch.setattr(QMessageBox, "information", _fake_information)
    window = FitWizardWindow()
    window._show_metric_info()

    assert captured["title"] == "Fit Wizard Metrics"
    assert "AICc" in captured["text"]
    assert "BIC" in captured["text"]


def test_fit_wizard_window_selection_updates_apply_page(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        wizard_window_module,
        "build_fit_wizard_recommendation",
        lambda dataset, current_model=None, metric=SelectionMetric.AICC: _fake_recommendation(dataset),
    )
    window = FitWizardWindow()
    window.set_analysis_context(dataset)
    window._start_analysis()
    _wait_for(lambda: _analysis_complete(window), qapp)

    window._compare_table.selectRow(1)
    qapp.processEvents()

    assert "Gaussian + Constant" in window._apply_selection_label.text()
    assert "Residual warnings" in window._apply_warning_text.toPlainText()


def test_fit_wizard_window_apply_recommended_emits_assessment(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        wizard_window_module,
        "build_fit_wizard_recommendation",
        lambda dataset, current_model=None, metric=SelectionMetric.AICC: _fake_recommendation(dataset),
    )
    window = FitWizardWindow()
    window.set_analysis_context(dataset)
    window._start_analysis()
    _wait_for(lambda: _analysis_complete(window), qapp)

    emitted: dict[str, object] = {}
    window.apply_assessment_requested.connect(
        lambda assessment, recommendation: emitted.update(
            {"assessment": assessment, "recommendation": recommendation}
        )
    )

    window._apply_recommended_fit()

    assert emitted["assessment"].template.key == "exp_constant"
    assert emitted["recommendation"].recommended_key == "exp_constant"


def test_fit_wizard_window_shows_progress_while_analysis_runs(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _slow_recommendation(dataset, current_model=None, metric=SelectionMetric.AICC):
        time.sleep(0.05)
        return _fake_recommendation(dataset)

    monkeypatch.setattr(
        wizard_window_module,
        "build_fit_wizard_recommendation",
        _slow_recommendation,
    )
    window = FitWizardWindow()
    window.set_analysis_context(dataset)

    window._start_analysis()

    assert window._analysis_in_progress is True
    assert window._progress_bar.isHidden() is False
    _wait_for(lambda: window._analysis_in_progress is False and window._analysis_thread is None, qapp)
    assert window._progress_bar.isHidden() is True


def test_fit_wizard_window_emits_cached_analysis_payload(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        wizard_window_module,
        "build_fit_wizard_recommendation",
        lambda dataset, current_model=None, metric=SelectionMetric.AICC: _fake_recommendation(dataset),
    )
    window = FitWizardWindow()
    payload: dict[str, object] = {}
    window.analysis_cached.connect(
        lambda recommendation, log_text, signature: payload.update(
            {
                "recommendation": recommendation,
                "log_text": log_text,
                "signature": signature,
            }
        )
    )

    window.set_analysis_context(dataset)
    window._start_analysis()
    _wait_for(lambda: _analysis_complete(window), qapp)

    assert isinstance(payload.get("recommendation"), FitWizardRecommendation)
    assert payload.get("log_text") == ""
    assert payload.get("signature") == {
        "run_number": int(dataset.run_number),
        "model": None,
    }


def test_fit_wizard_window_accepts_cached_recommendation(
    qapp: QApplication,
    dataset: MuonDataset,
) -> None:
    window = FitWizardWindow()
    recommendation = _fake_recommendation(dataset)

    window.set_analysis_context(dataset)
    window.set_cached_recommendation(
        recommendation,
        signature={"run_number": int(dataset.run_number), "model": None},
        log_text="cached log",
    )

    assert window.current_recommendation() is recommendation
    assert window.current_log_text() == "cached log"
    assert window._compare_table.rowCount() == 2
    assert window._apply_parameters_table.rowCount() == 3
