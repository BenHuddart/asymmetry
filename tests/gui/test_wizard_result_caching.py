"""Characterization: fit-wizard result caching for Single and Global fit tabs.

``SingleFitTab``/``GlobalFitTab`` each hold a "same signature -> reuse the
cached recommendation, changed signature -> invalidate" cache in front of the
(non-modal) wizard window (``fit_panel.py``: ``_cached_wizard_signature`` /
``_wizard_base_signature_matches`` / ``_cache_wizard_analysis``). Opening the
wizard never itself spawns the analysis worker — the user clicks "Start
Analysis"/"Rebuild Screening" inside the wizard window — so the observable
cache-hit/miss contract is entirely in what the tab hands the wizard window on
open: ``set_cached_recommendation(...)`` is called (cache hit) or is not
(cache miss).

These tests seed the tab's cache the same way production code does: by
driving the real ``analysis_cached`` signal through the tab's
``_on_fit_wizard_analysis_cached`` slot (the same connection
``_open_fit_wizard`` wires up), then re-open with a matching vs. a changed
signature and assert on a fake wizard window's captured calls. No threads,
no real wizard window — mirrors the ``_FakeWizard`` pattern already used in
``test_fit_panel_tabs.py`` for opening the Single Fit Wizard.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

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
from asymmetry.gui.panels.fit import global_tab as global_tab_module
from asymmetry.gui.panels.fit import single_tab as single_tab_module
from asymmetry.gui.panels.fit_panel import GlobalFitTab, SingleFitTab


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def dataset() -> MuonDataset:
    t = np.linspace(0.0, 4.0, 80)
    a = 0.2 * np.exp(-0.4 * t)
    e = np.full_like(t, 0.01)
    return MuonDataset(time=t, asymmetry=a, error=e, metadata={"run_number": 101})


class _FakeSingleWizard:
    """Stand-in for FitWizardWindow: records calls, spawns no threads."""

    def __init__(self, _parent) -> None:
        self.apply_assessment_requested = SimpleNamespace(connect=lambda _cb: None)
        self.analysis_cached = SimpleNamespace(connect=lambda _cb: None)
        self.set_cached_recommendation_calls: list[tuple] = []

    def set_analysis_context(self, dataset_arg, current_model=None) -> None:
        pass

    def set_cached_recommendation(self, recommendation, *, signature=None, log_text="") -> None:
        self.set_cached_recommendation_calls.append((recommendation, signature, log_text))

    def show(self) -> None:
        pass

    def raise_(self) -> None:
        pass

    def activateWindow(self) -> None:
        pass


def _single_wizard_payload(
    dataset: MuonDataset,
) -> tuple[CandidateAssessment, FitWizardRecommendation]:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    parameters = ParameterSet(
        [
            Parameter("A_1", 0.2, min=0.0, max=1.0),
            Parameter("Lambda", 0.4, min=0.0, max=5.0),
            Parameter("A_bg", 0.01, min=-0.5, max=0.5),
        ]
    )
    curve = model.function(dataset.time, A_1=0.2, Lambda=0.4, A_bg=0.01)
    result = FitResult(
        success=True,
        chi_squared=5.0,
        reduced_chi_squared=0.1,
        parameters=parameters,
        uncertainties={"A_1": 0.01, "Lambda": 0.02, "A_bg": 0.001},
        residuals=np.asarray(dataset.asymmetry - curve, dtype=float),
    )
    template = CandidateTemplate(
        key="exp_constant",
        title="Exponential + Constant",
        category="General",
        rationale="Baseline candidate",
        model=model,
    )
    assessment = CandidateAssessment(
        template=template,
        fit_result=result,
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
        fitted_curve=np.asarray(curve, dtype=float),
        component_curves=tuple(
            model.evaluate_components(
                dataset.time, additive_only=True, A_1=0.2, Lambda=0.4, A_bg=0.01
            )
        ),
    )
    recommendation = FitWizardRecommendation(
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
    return assessment, recommendation


def test_single_fit_wizard_reopen_with_matching_signature_serves_cache(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-opening the wizard with an unchanged run/model signature reuses the cache."""
    monkeypatch.setattr(single_tab_module, "FitWizardWindow", _FakeSingleWizard)
    tab = SingleFitTab()
    tab.set_dataset(dataset)
    tab._set_composite_model(CompositeModel(["Exponential", "Constant"], operators=["+"]))

    # First open constructs the (fake) wizard and wires analysis_cached; seed
    # the cache the same way production code does when analysis finishes.
    tab._open_fit_wizard()
    _assessment, recommendation = _single_wizard_payload(dataset)
    signature = tab._wizard_context_signature()
    tab._on_fit_wizard_analysis_cached(recommendation, "log text", signature)

    fake = tab._fit_wizard_window
    assert fake.set_cached_recommendation_calls == []  # nothing cached at first open

    # Re-open without changing dataset/model: the signature still matches.
    tab._open_fit_wizard()

    assert len(fake.set_cached_recommendation_calls) == 1
    cached_recommendation, cached_signature, cached_log_text = fake.set_cached_recommendation_calls[
        0
    ]
    assert cached_recommendation is recommendation
    assert cached_signature == signature
    assert cached_log_text == "log text"


def test_single_fit_wizard_reopen_with_changed_model_does_not_serve_cache(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing the composite model invalidates the cached wizard recommendation."""
    monkeypatch.setattr(single_tab_module, "FitWizardWindow", _FakeSingleWizard)
    tab = SingleFitTab()
    tab.set_dataset(dataset)
    tab._set_composite_model(CompositeModel(["Exponential", "Constant"], operators=["+"]))

    tab._open_fit_wizard()
    _assessment, recommendation = _single_wizard_payload(dataset)
    signature = tab._wizard_context_signature()
    tab._on_fit_wizard_analysis_cached(recommendation, "log text", signature)

    # Change the model -> signature no longer matches the cached one.
    tab._set_composite_model(CompositeModel(["Gaussian", "Constant"], operators=["+"]))
    tab._open_fit_wizard()

    fake = tab._fit_wizard_window
    assert fake.set_cached_recommendation_calls == [], (
        "a changed model must invalidate the cached recommendation"
    )


def test_single_fit_wizard_reopen_with_changed_run_does_not_serve_cache(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Switching to a different run invalidates the cached wizard recommendation."""
    monkeypatch.setattr(single_tab_module, "FitWizardWindow", _FakeSingleWizard)
    tab = SingleFitTab()
    tab.set_dataset(dataset)
    tab._set_composite_model(CompositeModel(["Exponential", "Constant"], operators=["+"]))

    tab._open_fit_wizard()
    _assessment, recommendation = _single_wizard_payload(dataset)
    signature = tab._wizard_context_signature()
    tab._on_fit_wizard_analysis_cached(recommendation, "log text", signature)

    other_dataset = MuonDataset(dataset.time, dataset.asymmetry, dataset.error, {"run_number": 999})
    tab.set_dataset(other_dataset)
    tab._open_fit_wizard()

    fake = tab._fit_wizard_window
    assert fake.set_cached_recommendation_calls == [], (
        "a changed run must invalidate the cached recommendation"
    )


# ── GlobalFitTab (Batch) wizard cache, keyed per run-set ──────────────────────


class _FakeGlobalWizard:
    """Stand-in for GlobalFitWizardWindow: records calls, spawns no threads."""

    def __init__(self, _parent) -> None:
        self.apply_assessment_requested = SimpleNamespace(connect=lambda _cb: None)
        self.analysis_cached = SimpleNamespace(connect=lambda _cb: None)
        self.single_fit_recommendations_generated = SimpleNamespace(connect=lambda _cb: None)
        self.parameter_setup_applied = SimpleNamespace(connect=lambda _cb: None)
        self.set_cached_recommendation_calls: list[tuple] = []

    def set_analysis_context(self, *_args, **_kwargs) -> None:
        pass

    def set_cached_recommendation(
        self, recommendation, *, signature=None, log_text="", status_text=None
    ) -> None:
        self.set_cached_recommendation_calls.append((recommendation, signature, log_text))

    def show(self) -> None:
        pass

    def raise_(self) -> None:
        pass

    def activateWindow(self) -> None:
        pass


def _two_dataset_batch(dataset: MuonDataset) -> list[MuonDataset]:
    d2 = MuonDataset(dataset.time, dataset.asymmetry, dataset.error, {"run_number": 102})
    return [dataset, d2]


def test_global_fit_wizard_reopen_with_matching_signature_serves_cache(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(global_tab_module, "GlobalFitWizardWindow", _FakeGlobalWizard)
    tab = GlobalFitTab()
    tab.set_datasets(_two_dataset_batch(dataset))

    tab._open_fit_wizard()
    parsed = tab._parse_parameter_configuration()
    signature = tab._wizard_context_signature(parsed)
    recommendation = _global_wizard_recommendation_for_dataset(dataset)
    tab._on_fit_wizard_analysis_cached(recommendation, "batch log", signature)

    fake = tab._fit_wizard_window
    assert fake.set_cached_recommendation_calls == []

    tab._open_fit_wizard()

    assert len(fake.set_cached_recommendation_calls) == 1
    cached_recommendation, cached_signature, cached_log_text = fake.set_cached_recommendation_calls[
        0
    ]
    assert cached_recommendation is recommendation
    assert cached_log_text == "batch log"


def test_global_fit_wizard_reopen_with_changed_run_set_does_not_serve_cache(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(global_tab_module, "GlobalFitWizardWindow", _FakeGlobalWizard)
    tab = GlobalFitTab()
    tab.set_datasets(_two_dataset_batch(dataset))

    tab._open_fit_wizard()
    parsed = tab._parse_parameter_configuration()
    signature = tab._wizard_context_signature(parsed)
    recommendation = _global_wizard_recommendation_for_dataset(dataset)
    tab._on_fit_wizard_analysis_cached(recommendation, "batch log", signature)

    # Swap in a different run-set entirely -> no cache entry for these runs.
    d3 = MuonDataset(dataset.time, dataset.asymmetry, dataset.error, {"run_number": 201})
    d4 = MuonDataset(dataset.time, dataset.asymmetry, dataset.error, {"run_number": 202})
    tab.set_datasets([d3, d4])
    tab._open_fit_wizard()

    fake = tab._fit_wizard_window
    assert fake.set_cached_recommendation_calls == [], (
        "a changed run set must not serve another run-set's cached recommendation"
    )


def _global_wizard_recommendation_for_dataset(
    dataset: MuonDataset,
) -> GlobalFitWizardRecommendation:
    """Minimal real GlobalFitWizardRecommendation, ported from test_fit_panel_tabs.py."""
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    fitted_curves = {
        int(dataset.run_number): (
            np.asarray(dataset.time, dtype=float),
            np.asarray(model.function(dataset.time, A_1=0.2, Lambda=0.4, A_bg=0.01), dtype=float),
        ),
        102: (
            np.asarray(dataset.time, dtype=float),
            np.asarray(model.function(dataset.time, A_1=0.2, Lambda=0.6, A_bg=0.01), dtype=float),
        ),
    }
    component_curves = {
        run_number: tuple(
            model.evaluate_components(
                dataset.time,
                additive_only=True,
                A_1=0.2,
                Lambda=(0.4 if run_number == int(dataset.run_number) else 0.6),
                A_bg=0.01,
            )
        )
        for run_number in fitted_curves
    }
    fit_results = {}
    for run_number, lambda_value in ((int(dataset.run_number), 0.4), (102, 0.6)):
        fit_results[run_number] = FitResult(
            success=True,
            chi_squared=5.0,
            reduced_chi_squared=0.1,
            parameters=ParameterSet(
                [
                    Parameter("A_1", 0.2, min=0.0, max=1.0),
                    Parameter("Lambda", lambda_value, min=0.0, max=5.0),
                    Parameter("A_bg", 0.01, min=-0.5, max=0.5),
                ]
            ),
            uncertainties={"A_1": 0.01, "Lambda": 0.02, "A_bg": 0.001},
        )

    template = CandidateTemplate(
        key="exp_constant",
        title="Exponential + Constant",
        category="General",
        rationale="Baseline candidate",
        model=model,
    )
    assessment = GlobalCandidateAssessment(
        template=template,
        fit_results_by_run=fit_results,
        global_parameters=ParameterSet([Parameter("A_1", 0.2), Parameter("A_bg", 0.01)]),
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
                rationale="Shared amplitude is sufficient.",
            ),
            GlobalParameterRecommendation(
                name="Lambda",
                recommended_role="Local",
                global_score=15.0,
                local_score=10.0,
                score_delta=5.0,
                total_variation=0.2,
                roughness=0.1,
                rationale="Local relaxation rates improve the score.",
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
        run_diagnostics=(
            RunResidualDiagnostic(
                run_number=int(dataset.run_number),
                run_label=dataset.run_label,
                axis_value=0.0,
                residual_rms=0.8,
                runs_z_score=0.1,
                max_abs_autocorrelation=0.1,
                residual_fft_peak_snr=1.0,
                gate_passed=True,
                gate_reasons=(),
            ),
            RunResidualDiagnostic(
                run_number=102,
                run_label="102",
                axis_value=100.0,
                residual_rms=0.8,
                runs_z_score=0.1,
                max_abs_autocorrelation=0.1,
                residual_fft_peak_snr=1.0,
                gate_passed=True,
                gate_reasons=(),
            ),
        ),
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
        fingerprints_by_run={},
        dataset_order=(int(dataset.run_number), 102),
        templates=(assessment.template,),
        assessments=(assessment,),
        metric=SelectionMetric.AICC,
        recommended_key="exp_constant",
        comparable_keys=(),
        summary="Recommended: Exponential + Constant by AICc.",
    )
