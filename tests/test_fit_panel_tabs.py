"""Focused tests for SingleFitTab and GlobalFitTab logic."""

from __future__ import annotations

import os
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QComboBox, QMessageBox, QSizePolicy

from asymmetry.core.data.dataset import MuonDataset, Run
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
from asymmetry.core.utils.constants import (
    GAUSS_TO_TESLA,
    MUON_GYROMAGNETIC_RATIO_MHZ_PER_T,
    MUON_LIFETIME_US,
)
from asymmetry.gui.panels import fit_panel as fit_panel_module
from asymmetry.gui.panels.fit_panel import FitPanel, GlobalFitTab, SingleFitTab


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


def _wizard_payload_for_dataset(
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


def _global_wizard_recommendation_for_dataset(
    dataset: MuonDataset,
) -> GlobalFitWizardRecommendation:
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


def test_single_fit_requires_dataset(qapp: QApplication) -> None:
    tab = SingleFitTab()
    tab._current_dataset = None
    tab._run_fit()
    assert "No dataset selected" in tab._result_label.text()


def test_single_fit_fit_wizard_button_tracks_dataset_and_block_state(
    qapp: QApplication, dataset: MuonDataset
) -> None:
    tab = SingleFitTab()
    assert tab._fit_wizard_btn.isEnabled() is False

    tab.set_dataset(dataset)
    assert tab._fit_wizard_btn.isEnabled() is True

    tab.set_fit_blocked(True, "blocked")
    assert tab._fit_wizard_btn.isEnabled() is False


def test_single_fit_open_fit_wizard_without_dataset_shows_message(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tab = SingleFitTab()
    captured = {}

    monkeypatch.setattr(
        fit_panel_module.QMessageBox,
        "information",
        lambda *_args: captured.setdefault("shown", True),
    )

    tab._open_fit_wizard()

    assert captured["shown"] is True


def test_single_fit_open_fit_wizard_uses_active_dataset_and_model(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received = {}

    class _FakeWizard:
        def __init__(self, _parent) -> None:
            self.apply_assessment_requested = SimpleNamespace(connect=lambda _cb: None)
            self.analysis_cached = SimpleNamespace(connect=lambda _cb: None)

        def set_analysis_context(self, dataset_arg, current_model=None) -> None:
            received["dataset"] = dataset_arg
            received["model"] = current_model

        def show(self) -> None:
            received["show"] = True

        def raise_(self) -> None:
            received["raise"] = True

        def activateWindow(self) -> None:
            received["activate"] = True

    monkeypatch.setattr(fit_panel_module, "FitWizardWindow", _FakeWizard)
    tab = SingleFitTab()
    tab.set_dataset(dataset)
    tab._set_composite_model(CompositeModel(["Gaussian", "Constant"], operators=["+"]))

    tab._open_fit_wizard()

    assert received["dataset"] is dataset
    assert received["model"].component_names == ["Gaussian", "Constant"]


def test_single_fit_apply_fit_wizard_assessment_emits_fit_completed(
    qapp: QApplication,
    dataset: MuonDataset,
) -> None:
    tab = SingleFitTab()
    tab.set_dataset(dataset)
    assessment, recommendation = _wizard_payload_for_dataset(dataset)

    emitted = {}
    tab.fit_completed.connect(
        lambda result, curve, components: emitted.update(
            {"result": result, "curve": curve, "components": components}
        )
    )

    tab._apply_fit_wizard_assessment(assessment, recommendation)

    assert "Fit Wizard" in tab._result_label.text()
    assert emitted["result"].success is True
    assert tab._composite_model.component_names == ["Exponential", "Constant"]


def test_fit_panel_forwards_fit_wizard_apply_via_normal_fit_completed_signal(
    qapp: QApplication,
    dataset: MuonDataset,
) -> None:
    panel = FitPanel()
    panel.set_dataset(dataset)
    assessment, recommendation = _wizard_payload_for_dataset(dataset)

    emitted = {}
    panel.fit_completed.connect(
        lambda result, curve, components: emitted.update(
            {"result": result, "curve": curve, "components": components}
        )
    )

    panel._single_tab._apply_fit_wizard_assessment(assessment, recommendation)

    assert emitted["result"].success is True
    assert len(emitted["curve"][0]) >= 500


def test_fit_panel_forwards_single_tab_preview(qapp: QApplication, dataset: MuonDataset) -> None:
    panel = FitPanel()
    panel.set_dataset(dataset)

    emitted = {}
    panel.preview_requested.connect(
        lambda result, curve, components: emitted.update(
            {"result": result, "curve": curve, "components": components}
        )
    )

    panel._single_tab._on_preview()

    assert "curve" in emitted
    assert len(emitted["curve"][0]) == 500


def test_single_fit_invalid_value_shows_error(qapp: QApplication, dataset: MuonDataset) -> None:
    tab = SingleFitTab()
    tab.set_dataset(dataset)

    # Corrupt first value cell.
    tab._param_table.item(0, 1).setText("not-a-number")
    tab._run_fit()

    assert "Invalid value" in tab._result_label.text()


def test_single_fit_success_emits_and_updates_table(
    qapp: QApplication, dataset: MuonDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    tab = SingleFitTab()
    tab.set_dataset(dataset)

    model = tab._composite_model

    fitted = ParameterSet(
        [Parameter(name=p, value=float(i + 1)) for i, p in enumerate(model.param_names)]
    )
    result = FitResult(
        success=True,
        chi_squared=10.0,
        reduced_chi_squared=0.5,
        parameters=fitted,
        uncertainties={p: 0.01 for p in model.param_names},
    )

    tab._fit_engine = SimpleNamespace(fit=lambda *_args, **_kwargs: result)

    emitted = {}
    tab.fit_completed.connect(lambda res, curve: emitted.update({"res": res, "curve": curve}))

    tab._run_fit()

    assert "Fit failed" not in tab._result_label.text()
    assert "χ²" in tab._result_label.text()
    assert emitted["res"].success is True
    assert len(emitted["curve"][0]) == 500


def test_single_fit_preview_upsamples_high_frequency_models(
    qapp: QApplication,
    dataset: MuonDataset,
) -> None:
    tab = SingleFitTab()
    tab.set_dataset(dataset)
    tab._set_composite_model(CompositeModel(["Oscillatory"], operators=[]))

    row_by_name: dict[str, int] = {}
    for row in range(tab._param_table.rowCount()):
        name_item = tab._param_table.item(row, 0)
        assert name_item is not None
        pname = name_item.data(Qt.ItemDataRole.UserRole)
        row_by_name[str(pname)] = row

    tab._param_table.item(row_by_name["A_1"], 1).setText("1.0")
    tab._param_table.item(row_by_name["frequency"], 1).setText("50.0")
    tab._param_table.item(row_by_name["phase"], 1).setText("0.0")

    emitted: dict[str, object] = {}
    tab.preview_requested.connect(lambda _r, curve, _c: emitted.update({"curve": curve}))
    tab._on_preview()

    assert "curve" in emitted
    curve = emitted["curve"]
    assert isinstance(curve, tuple)
    assert len(curve[0]) > 500


def test_single_fit_uses_dataset_object_it_was_given(
    qapp: QApplication, dataset: MuonDataset
) -> None:
    rebinned = MuonDataset(
        time=dataset.time[::4],
        asymmetry=dataset.asymmetry[::4],
        error=dataset.error[::4],
        metadata=dict(dataset.metadata),
    )

    tab = SingleFitTab()
    tab.set_dataset(rebinned)

    model = tab._composite_model

    captured = {}

    def _fit(captured_dataset, model_fn, parameters):
        captured["dataset"] = captured_dataset
        captured["model_fn"] = model_fn
        captured["n_points"] = len(captured_dataset.time)
        return FitResult(
            success=True,
            chi_squared=1.0,
            reduced_chi_squared=0.1,
            parameters=ParameterSet(
                [Parameter(name=p, value=float(i + 1)) for i, p in enumerate(model.param_names)]
            ),
            uncertainties={p: 0.01 for p in model.param_names},
        )

    tab._fit_engine = SimpleNamespace(fit=_fit)

    tab._run_fit()

    assert captured["dataset"] is rebinned
    assert captured["n_points"] == len(rebinned.time)


def test_global_tab_set_datasets_states(qapp: QApplication, dataset: MuonDataset) -> None:
    tab = GlobalFitTab()

    tab.set_datasets([])
    assert tab._fit_btn.isEnabled() is False
    assert "No datasets selected" in tab._result_text.toPlainText()

    tab.set_datasets([dataset])
    assert tab._fit_btn.isEnabled() is False
    assert "requires at least 2 datasets" in tab._result_text.toPlainText()

    d2 = MuonDataset(dataset.time, dataset.asymmetry, dataset.error, {"run_number": 102})
    tab.set_datasets([dataset, d2])
    assert tab._fit_btn.isEnabled() is True


def test_global_fit_rejects_non_finite_value(qapp: QApplication, dataset: MuonDataset) -> None:
    tab = GlobalFitTab()
    d2 = MuonDataset(dataset.time, dataset.asymmetry, dataset.error, {"run_number": 102})
    tab.set_datasets([dataset, d2])

    # Make first parameter non-finite.
    tab._param_table.item(0, 1).setText("nan")
    tab._run_global_fit()

    assert "must be finite" in tab._result_text.toPlainText()


def test_global_fit_rejects_invalid_bounds(qapp: QApplication, dataset: MuonDataset) -> None:
    tab = GlobalFitTab()
    d2 = MuonDataset(dataset.time, dataset.asymmetry, dataset.error, {"run_number": 102})
    tab.set_datasets([dataset, d2])

    tab._param_table.item(0, 3).setText("2, 1")
    tab._run_global_fit()

    assert "invalid bounds" in tab._result_text.toPlainText()


def test_global_fit_finished_success_emits(qapp: QApplication, dataset: MuonDataset) -> None:
    tab = GlobalFitTab()
    d2 = MuonDataset(dataset.time, dataset.asymmetry, dataset.error, {"run_number": 102})
    tab._datasets = [dataset, d2]

    model = tab._composite_model
    tab._current_model = model
    tab._current_global_params = [model.param_names[0]]

    pset = ParameterSet([Parameter(name=p, value=1.0) for p in model.param_names])
    result = FitResult(
        success=True,
        chi_squared=5.0,
        reduced_chi_squared=0.5,
        parameters=pset,
        uncertainties={model.param_names[0]: 0.1},
    )
    fitted_global = ParameterSet([Parameter(name=model.param_names[0], value=1.0)])

    emitted = {}
    tab.global_fit_completed.connect(lambda res, glob: emitted.update({"res": res, "glob": glob}))

    tab._on_fit_finished({101: result, 102: result}, fitted_global)

    assert "Batch fit converged" in tab._result_text.toHtml()
    assert set(emitted["res"]) == {101, 102}


def test_global_fit_finished_failure_lists_failed_runs(
    qapp: QApplication, dataset: MuonDataset
) -> None:
    tab = GlobalFitTab()
    tab._current_model = tab._composite_model
    tab._current_global_params = []
    fail = FitResult(success=False, message="x")

    tab._on_fit_finished({101: fail}, ParameterSet())
    assert "Batch fit failed" in tab._result_text.toPlainText()


def test_global_fit_error_sets_message(qapp: QApplication, dataset: MuonDataset) -> None:
    tab = GlobalFitTab()
    tab.set_datasets(
        [
            dataset,
            replace(dataset, metadata={**dataset.metadata, "run_number": 102}),
        ]
    )
    tab._fit_btn.setEnabled(False)
    tab._on_fit_error("boom")
    assert tab._fit_btn.isEnabled() is True
    assert "Error during global fit" in tab._result_text.toPlainText()
    assert "boom" in tab._result_text.toPlainText()


def test_grouped_fit_error_formats_keyerror_message(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tab = GlobalFitTab(member_kind="groups")
    grouped_groups = [
        SimpleNamespace(group_id=1, group_name="Forward", counts=np.array([120.0, 118.0])),
        SimpleNamespace(group_id=2, group_name="Backward", counts=np.array([80.0, 79.0])),
    ]
    monkeypatch.setattr(tab, "_grouped_mode_context", lambda: (grouped_groups, [], "ready"))
    tab.set_current_dataset(dataset)

    tab._on_fit_error(fit_panel_module._format_fit_worker_exception(KeyError(1651)))

    assert "Error during grouped fit" in tab._result_text.toPlainText()
    assert "Missing fit parameter mapping" in tab._result_text.toPlainText()
    assert "1651" in tab._result_text.toPlainText()


def test_grouped_fit_finished_updates_grouped_tables(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    time = np.linspace(0.0, 4.0, 80)
    field = 150.0
    dataset = MuonDataset(
        time=time,
        asymmetry=np.zeros_like(time),
        error=np.full_like(time, 0.01),
        metadata={"run_number": 101, "field": field},
        run=Run(run_number=101, metadata={"field": field}),
    )
    grouped_groups = [
        SimpleNamespace(
            group_id=1,
            group_name="Forward",
            time=time,
            counts=np.full_like(time, 120.0),
            error=np.full_like(time, 1.0),
            metadata={"field": field},
        ),
        SimpleNamespace(
            group_id=2,
            group_name="Backward",
            time=time,
            counts=np.full_like(time, 80.0),
            error=np.full_like(time, 1.0),
            metadata={"field": field},
        ),
    ]
    grouped_datasets = [
        MuonDataset(
            time=np.array([0.0, 0.5, 1.0]),
            asymmetry=np.array([120.0, 110.0, 95.0]),
            error=np.array([1.0, 1.0, 1.0]),
            metadata={
                "group_id": 1,
                "grouped_time_domain": True,
                "run_number": 9001,
                "run_label": "Forward",
            },
            run=dataset.run,
        ),
        MuonDataset(
            time=np.array([0.0, 0.5, 1.0]),
            asymmetry=np.array([80.0, 75.0, 70.0]),
            error=np.array([1.0, 1.0, 1.0]),
            metadata={
                "group_id": 2,
                "grouped_time_domain": True,
                "run_number": 9002,
                "run_label": "Backward",
            },
            run=dataset.run,
        ),
    ]

    tab = GlobalFitTab(member_kind="groups")
    monkeypatch.setattr(
        tab, "_grouped_mode_context", lambda: (grouped_groups, grouped_datasets, "ready")
    )
    tab.set_current_dataset(dataset)

    forward_params = ParameterSet(
        [
            Parameter("N0", 101.0, min=0.0, max=500.0),
            Parameter("background", 6.0, min=0.0, max=50.0),
            Parameter("amplitude", 0.21, min=-1.0, max=1.0),
            Parameter("relative_phase", 0.33, min=-np.pi, max=np.pi),
            Parameter("field", 151.0, min=0.0, max=1000.0),
            Parameter("phase", 0.42, min=-np.pi, max=np.pi),
        ]
    )
    backward_params = ParameterSet(
        [
            Parameter("N0", 202.0, min=0.0, max=500.0),
            Parameter("background", 7.0, min=0.0, max=50.0),
            Parameter("amplitude", 0.27, min=-1.0, max=1.0),
            Parameter("relative_phase", -0.41, min=-np.pi, max=np.pi),
            Parameter("field", 151.0, min=0.0, max=1000.0),
            Parameter("phase", 0.42, min=-np.pi, max=np.pi),
        ]
    )
    grouped_result = SimpleNamespace(
        shared_parameters=ParameterSet(
            [
                Parameter("field", 151.0, min=0.0, max=1000.0),
                Parameter("phase", 0.42, min=-np.pi, max=np.pi),
            ]
        ),
        group_results={
            1: FitResult(
                success=True, chi_squared=1.0, reduced_chi_squared=0.1, parameters=forward_params
            ),
            2: FitResult(
                success=True, chi_squared=1.0, reduced_chi_squared=0.1, parameters=backward_params
            ),
        },
    )

    tab._current_model = tab._composite_model
    tab._on_grouped_fit_finished(grouped_datasets, grouped_result)

    group_param_rows = {
        tab._group_param_table.item(row, 0).data(Qt.ItemDataRole.UserRole): row
        for row in range(tab._group_param_table.rowCount())
    }
    group_model_rows = {
        tab._group_model_table.item(row, 0).data(Qt.ItemDataRole.UserRole): row
        for row in range(tab._group_model_table.rowCount())
    }

    assert float(tab._group_param_table.item(group_param_rows["N0"], 1).text()) == pytest.approx(
        101.0
    )
    assert float(tab._group_param_table.item(group_param_rows["N0"], 2).text()) == pytest.approx(
        202.0
    )
    assert float(
        tab._group_param_table.item(group_param_rows["relative_phase"], 1).text()
    ) == pytest.approx(0.33)
    assert float(
        tab._group_param_table.item(group_param_rows["relative_phase"], 2).text()
    ) == pytest.approx(-0.41)
    assert float(tab._group_model_table.item(group_model_rows["field"], 1).text()) == pytest.approx(
        151.0
    )
    assert float(tab._group_model_table.item(group_model_rows["phase"], 1).text()) == pytest.approx(
        0.42
    )
    assert "Grouped fit converged" in tab._result_text.toPlainText()


def test_global_fit_parses_type_combo_defaults(qapp: QApplication) -> None:
    tab = GlobalFitTab()
    # First row defaults to Global, subsequent rows Local.
    c0 = tab._param_table.cellWidget(0, 2)
    c1 = tab._param_table.cellWidget(1, 2) if tab._param_table.rowCount() > 1 else None
    assert isinstance(c0, QComboBox)
    assert c0.currentText() == "Global"
    if isinstance(c1, QComboBox):
        assert c1.currentText() == "Local"


def test_global_fit_type_combo_includes_file_for_bl_parameters(
    qapp: QApplication, dataset: MuonDataset
) -> None:
    """Parameters like B_L should have File as an option in the Type combo."""
    from asymmetry.core.fitting.composite import CompositeModel

    tab = GlobalFitTab()
    d2 = MuonDataset(dataset.time, dataset.asymmetry, dataset.error, {"run_number": 102})
    tab.set_datasets([dataset, d2])

    # Set an LF-KT model that includes B_L parameter
    lf_model = CompositeModel(["LongitudinalFieldKT", "Constant"], operators=["+"])
    tab._set_composite_model(lf_model)

    # Find the B_L row
    bl_row = None
    for i in range(tab._param_table.rowCount()):
        item = tab._param_table.item(i, 0)
        if item and item.data(Qt.ItemDataRole.UserRole) == "B_L":
            bl_row = i
            break

    assert bl_row is not None, "B_L parameter not found in table"

    # The Type combo for B_L should have File as an option
    type_combo = tab._param_table.cellWidget(bl_row, 2)
    assert isinstance(type_combo, QComboBox)
    items = [type_combo.itemText(i) for i in range(type_combo.count())]
    assert "File" in items


def test_global_fit_parameter_help_button_opens_dialog(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    tab = GlobalFitTab()
    captured: dict[str, object] = {}

    def _fake_information(parent, title, text):
        captured["parent"] = parent
        captured["title"] = title
        captured["text"] = text
        return QMessageBox.StandardButton.Ok

    monkeypatch.setattr(fit_panel_module.QMessageBox, "information", _fake_information)

    assert tab._param_help_btn.text() == "?"
    tab._param_help_btn.click()

    assert captured["parent"] is tab
    assert captured["title"] == "Parameter Classification Help"
    assert "Global: Same value for all datasets" in str(captured["text"])
    assert "Local: Different value for each dataset" in str(captured["text"])
    assert "Fixed: Held constant at the specified value" in str(captured["text"])
    assert "File: Use the value from dataset metadata" in str(captured["text"])


def test_single_tab_default_model_includes_background(qapp: QApplication) -> None:
    tab = SingleFitTab()
    assert tab._composite_model.component_names == ["Exponential", "Constant"]
    assert "A_bg" in tab._composite_model.param_names


def test_global_tab_default_model_includes_background(qapp: QApplication) -> None:
    tab = GlobalFitTab()
    assert tab._composite_model.component_names == ["Exponential", "Constant"]
    assert "A_bg" in tab._composite_model.param_names


def test_grouped_tab_shows_one_value_column_per_group(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tab = GlobalFitTab(member_kind="groups")
    grouped_groups = [
        SimpleNamespace(group_id=1, group_name="Forward", counts=np.array([120.0, 118.0])),
        SimpleNamespace(group_id=2, group_name="Backward", counts=np.array([80.0, 79.0])),
    ]
    monkeypatch.setattr(tab, "_grouped_mode_context", lambda: (grouped_groups, [], "ready"))

    tab.set_current_dataset(dataset)

    headers = [
        tab._group_param_table.horizontalHeaderItem(column).text()
        for column in range(tab._group_param_table.columnCount())
    ]
    row_by_name = {
        tab._group_param_table.item(row, 0).data(Qt.ItemDataRole.UserRole): row
        for row in range(tab._group_param_table.rowCount())
    }
    forward_background, forward_n0, _forward_amplitude = (
        fit_panel_module._seed_group_background_and_n0(np.array([120.0, 118.0]))
    )
    backward_background, backward_n0, _backward_amplitude = (
        fit_panel_module._seed_group_background_and_n0(np.array([80.0, 79.0]))
    )

    assert headers == ["Parameter", "Forward", "Backward", "Type", "Bounds"]
    assert float(tab._group_param_table.item(row_by_name["N0"], 1).text()) == pytest.approx(
        forward_n0
    )
    assert float(tab._group_param_table.item(row_by_name["N0"], 2).text()) == pytest.approx(
        backward_n0
    )
    assert float(tab._group_param_table.item(row_by_name["background"], 1).text()) == pytest.approx(
        forward_background
    )
    assert float(tab._group_param_table.item(row_by_name["background"], 2).text()) == pytest.approx(
        backward_background
    )
    assert tab._group_param_table.cellWidget(row_by_name["N0"], 2) is None
    assert isinstance(tab._group_param_table.cellWidget(row_by_name["N0"], 3), QComboBox)

    group_model_row_by_name = {
        tab._group_model_table.item(row, 0).data(Qt.ItemDataRole.UserRole): row
        for row in range(tab._group_model_table.rowCount())
    }
    assert "A_1" not in group_model_row_by_name
    assert "field" in group_model_row_by_name


def test_grouped_tab_defaults_to_oscillatory_field_model(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = MuonDataset(
        time=np.linspace(0.0, 4.0, 80),
        asymmetry=np.zeros(80),
        error=np.full(80, 0.01),
        metadata={"run_number": 101},
        run=Run(run_number=101, metadata={"field": 150.0}),
    )
    tab = GlobalFitTab(member_kind="groups")
    grouped_groups = [
        SimpleNamespace(group_id=1, group_name="Forward", counts=np.array([120.0, 118.0])),
        SimpleNamespace(group_id=2, group_name="Backward", counts=np.array([80.0, 79.0])),
    ]
    monkeypatch.setattr(tab, "_grouped_mode_context", lambda: (grouped_groups, [], "ready"))

    tab.set_current_dataset(dataset)

    assert tab._composite_model.component_names == ["OscillatoryField"]
    assert "field*t" in tab._formula_label.toolTip()
    row_by_name = {
        tab._group_model_table.item(row, 0).data(Qt.ItemDataRole.UserRole): row
        for row in range(tab._group_model_table.rowCount())
    }
    assert set(row_by_name) == {"field", "phase"}
    assert tab._group_model_table.item(row_by_name["field"], 1).text() == "150"


def test_grouped_tab_clearing_current_dataset_does_not_crash(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tab = GlobalFitTab(member_kind="groups")
    grouped_groups = [
        SimpleNamespace(group_id=1, group_name="Forward", counts=np.array([120.0, 118.0])),
        SimpleNamespace(group_id=2, group_name="Backward", counts=np.array([80.0, 79.0])),
    ]
    monkeypatch.setattr(tab, "_grouped_mode_context", lambda: (grouped_groups, [], "ready"))

    tab.set_current_dataset(dataset)
    tab.set_current_dataset(None)

    assert tab._current_dataset is None


def test_grouped_tab_fractionizes_additive_fit_function(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tab = GlobalFitTab(member_kind="groups")
    grouped_groups = [
        SimpleNamespace(group_id=1, group_name="Forward", counts=np.array([120.0, 118.0])),
        SimpleNamespace(group_id=2, group_name="Backward", counts=np.array([80.0, 79.0])),
    ]
    monkeypatch.setattr(tab, "_grouped_mode_context", lambda: (grouped_groups, [], "ready"))

    tab.set_current_dataset(dataset)
    tab._set_composite_model(CompositeModel(["Exponential", "Constant"], operators=["+"]))

    assert "fraction_1" in tab._formula_label.text()
    assert "A_1" not in tab._formula_label.text()
    row_by_name = {
        tab._group_model_table.item(row, 0).data(Qt.ItemDataRole.UserRole): row
        for row in range(tab._group_model_table.rowCount())
    }
    assert set(row_by_name) == {"Lambda", "fraction_1", "fraction_2"}


def test_grouped_tab_synchronizes_fraction_rows(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tab = GlobalFitTab(member_kind="groups")
    grouped_groups = [
        SimpleNamespace(group_id=1, group_name="Forward", counts=np.array([120.0, 118.0])),
        SimpleNamespace(group_id=2, group_name="Backward", counts=np.array([80.0, 79.0])),
    ]
    monkeypatch.setattr(tab, "_grouped_mode_context", lambda: (grouped_groups, [], "ready"))
    tab.set_current_dataset(dataset)
    tab._set_composite_model(CompositeModel(["Exponential", "Constant"], operators=["+"]))

    row_by_name = {
        tab._group_model_table.item(row, 0).data(Qt.ItemDataRole.UserRole): row
        for row in range(tab._group_model_table.rowCount())
    }
    tab._group_model_table.item(row_by_name["fraction_1"], 1).setText("0.2")
    final_type_combo = tab._group_model_table.cellWidget(row_by_name["fraction_2"], 2)

    assert tab._group_model_table.item(row_by_name["fraction_1"], 1).text() == "0.2"
    assert tab._group_model_table.item(row_by_name["fraction_2"], 1).text() == "0.8"
    assert isinstance(final_type_combo, QComboBox)
    assert final_type_combo.currentText() == "Fixed"
    assert not final_type_combo.isEnabled()


def test_grouped_fit_uses_per_group_seed_values_from_group_columns(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tab = GlobalFitTab(member_kind="groups")
    grouped_groups = [
        SimpleNamespace(group_id=1, group_name="Forward", counts=np.array([120.0, 118.0])),
        SimpleNamespace(group_id=2, group_name="Backward", counts=np.array([80.0, 79.0])),
    ]
    monkeypatch.setattr(tab, "_grouped_mode_context", lambda: (grouped_groups, [], "ready"))
    tab.set_current_dataset(dataset)

    row_by_name = {
        tab._group_param_table.item(row, 0).data(Qt.ItemDataRole.UserRole): row
        for row in range(tab._group_param_table.rowCount())
    }
    forward_background, forward_n0, forward_amplitude = (
        fit_panel_module._seed_group_background_and_n0(np.array([120.0, 118.0]))
    )
    backward_background, backward_n0, backward_amplitude = (
        fit_panel_module._seed_group_background_and_n0(np.array([80.0, 79.0]))
    )
    tab._group_param_table.item(row_by_name["N0"], 1).setText("101")
    tab._group_param_table.item(row_by_name["N0"], 2).setText("202")
    tab._group_param_table.item(row_by_name["amplitude"], 1).setText("0.1")
    tab._group_param_table.item(row_by_name["amplitude"], 2).setText("0.3")

    captured: dict[str, object] = {}

    class _DummySignal:
        def connect(self, *_args, **_kwargs):
            return None

    class _FakeThread:
        def __init__(self):
            self.started = _DummySignal()
            self.finished = _DummySignal()

        def start(self):
            return None

        def quit(self):
            return None

        def wait(self):
            return None

        def deleteLater(self):
            return None

    class _FakeWorker:
        def __init__(
            self,
            _grouped_groups,
            _grouped_datasets,
            _model_fn,
            _global_params,
            _local_params,
            initial_params,
        ):
            captured["initial_params"] = initial_params
            self.finished = _DummySignal()
            self.error = _DummySignal()

        def moveToThread(self, _thread):
            return None

        def run(self):
            return None

        def deleteLater(self):
            return None

    monkeypatch.setattr(fit_panel_module, "QThread", _FakeThread)
    monkeypatch.setattr(fit_panel_module, "GroupedTimeDomainFitWorker", _FakeWorker)

    tab._run_global_fit()

    initial_params = captured["initial_params"]
    assert initial_params[1]["N0"].value == pytest.approx(101.0)
    assert initial_params[2]["N0"].value == pytest.approx(202.0)
    assert initial_params[1]["amplitude"].value == pytest.approx(0.1)
    assert initial_params[2]["amplitude"].value == pytest.approx(0.3)


def test_grouped_phase_seed_uses_fft_peak_estimate_relative_to_first_group() -> None:
    time = np.linspace(0.0, 8.0, 801)
    frequency = 3.2
    background = 8.0
    n0 = 100.0
    amplitude = 0.18
    phase_a = np.deg2rad(25.0)
    phase_b = np.deg2rad(-55.0)

    def _counts(phase: float) -> np.ndarray:
        return n0 * (1.0 + amplitude * np.cos(2.0 * np.pi * frequency * time + phase)) + (
            background * np.exp(time / float(MUON_LIFETIME_US))
        )

    grouped_groups = [
        SimpleNamespace(
            group_id=1,
            group_name="Forward",
            time=time,
            counts=_counts(phase_a),
            error=np.full_like(time, 0.05),
            metadata={"field": frequency / (MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * GAUSS_TO_TESLA)},
        ),
        SimpleNamespace(
            group_id=2,
            group_name="Backward",
            time=time,
            counts=_counts(phase_b),
            error=np.full_like(time, 0.05),
            metadata={"field": frequency / (MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * GAUSS_TO_TESLA)},
        ),
    ]

    phases = fit_panel_module._seed_group_relative_phases(grouped_groups)

    expected_delta = float(np.angle(np.exp(1j * (phase_b - phase_a))))
    assert phases["1"] == pytest.approx(0.0, abs=0.2)
    assert phases["2"] == pytest.approx(expected_delta, abs=0.3)


def test_grouped_model_phase_seed_uses_first_group_estimate(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    time = np.linspace(0.0, 8.0, 801)
    frequency = 3.2
    field = frequency / (MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * GAUSS_TO_TESLA)
    n0 = 100.0
    background = 8.0
    amplitude = 0.18
    phase_a = np.deg2rad(25.0)
    phase_b = np.deg2rad(-55.0)

    def _counts(phase: float) -> np.ndarray:
        return n0 * (1.0 + amplitude * np.cos(2.0 * np.pi * frequency * time + phase)) + (
            background * np.exp(time / float(MUON_LIFETIME_US))
        )

    dataset = MuonDataset(
        time=time,
        asymmetry=np.zeros_like(time),
        error=np.full_like(time, 0.05),
        metadata={"run_number": 101, "field": field},
        run=Run(run_number=101, metadata={"field": field}),
    )
    grouped_groups = [
        SimpleNamespace(
            group_id=1,
            group_name="Forward",
            time=time,
            counts=_counts(phase_a),
            error=np.full_like(time, 0.05),
            metadata={"field": field},
        ),
        SimpleNamespace(
            group_id=2,
            group_name="Backward",
            time=time,
            counts=_counts(phase_b),
            error=np.full_like(time, 0.05),
            metadata={"field": field},
        ),
    ]

    tab = GlobalFitTab(member_kind="groups")
    monkeypatch.setattr(tab, "_grouped_mode_context", lambda: (grouped_groups, [], "ready"))
    tab.set_current_dataset(dataset)

    row_by_name = {
        tab._group_model_table.item(row, 0).data(Qt.ItemDataRole.UserRole): row
        for row in range(tab._group_model_table.rowCount())
    }
    reference_phase, _relative_phases = fit_panel_module._seed_group_phase_estimates(grouped_groups)

    assert float(tab._group_model_table.item(row_by_name["phase"], 1).text()) == pytest.approx(
        reference_phase,
        abs=0.2,
    )


def test_grouped_tab_reset_button_restores_estimated_values(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tab = GlobalFitTab(member_kind="groups")
    grouped_groups = [
        SimpleNamespace(group_id=1, group_name="Forward", counts=np.array([120.0, 118.0])),
        SimpleNamespace(group_id=2, group_name="Backward", counts=np.array([80.0, 79.0])),
    ]
    monkeypatch.setattr(tab, "_grouped_mode_context", lambda: (grouped_groups, [], "ready"))
    tab.set_current_dataset(dataset)

    row_by_name = {
        tab._group_param_table.item(row, 0).data(Qt.ItemDataRole.UserRole): row
        for row in range(tab._group_param_table.rowCount())
    }
    forward_background, forward_n0, forward_amplitude = (
        fit_panel_module._seed_group_background_and_n0(np.array([120.0, 118.0]))
    )
    backward_background, backward_n0, backward_amplitude = (
        fit_panel_module._seed_group_background_and_n0(np.array([80.0, 79.0]))
    )
    relative_phase_defaults = fit_panel_module._seed_group_relative_phases(grouped_groups)
    tab._group_param_table.item(row_by_name["N0"], 1).setText("101")
    tab._group_param_table.item(row_by_name["N0"], 2).setText("202")
    tab._group_param_table.item(row_by_name["background"], 1).setText("1.5")
    tab._group_param_table.item(row_by_name["background"], 2).setText("2.5")
    tab._group_param_table.item(row_by_name["amplitude"], 1).setText("0.1")
    tab._group_param_table.item(row_by_name["amplitude"], 2).setText("0.3")
    tab._group_param_table.item(row_by_name["relative_phase"], 1).setText("0.4")
    tab._group_param_table.item(row_by_name["relative_phase"], 2).setText("-0.2")
    n0_type = tab._group_param_table.cellWidget(row_by_name["N0"], 3)
    assert isinstance(n0_type, QComboBox)
    assert n0_type.currentText() == "Local"

    tab._group_param_reset_btn.click()

    assert float(tab._group_param_table.item(row_by_name["N0"], 1).text()) == pytest.approx(
        forward_n0
    )
    assert float(tab._group_param_table.item(row_by_name["N0"], 2).text()) == pytest.approx(
        backward_n0
    )
    assert float(tab._group_param_table.item(row_by_name["background"], 1).text()) == pytest.approx(
        forward_background
    )
    assert float(tab._group_param_table.item(row_by_name["background"], 2).text()) == pytest.approx(
        backward_background
    )
    assert float(tab._group_param_table.item(row_by_name["amplitude"], 1).text()) == pytest.approx(
        forward_amplitude
    )
    assert float(tab._group_param_table.item(row_by_name["amplitude"], 2).text()) == pytest.approx(
        backward_amplitude
    )
    assert float(
        tab._group_param_table.item(row_by_name["relative_phase"], 1).text()
    ) == pytest.approx(relative_phase_defaults["1"])
    assert float(
        tab._group_param_table.item(row_by_name["relative_phase"], 2).text()
    ) == pytest.approx(relative_phase_defaults["2"])
    reset_n0_type = tab._group_param_table.cellWidget(row_by_name["N0"], 3)
    assert isinstance(reset_n0_type, QComboBox)
    assert reset_n0_type.currentText() == "Local"


def test_grouped_tab_state_roundtrip_preserves_per_group_parameter_columns(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grouped_groups = [
        SimpleNamespace(group_id=1, group_name="Forward", counts=np.array([120.0, 118.0])),
        SimpleNamespace(group_id=2, group_name="Backward", counts=np.array([80.0, 79.0])),
    ]

    tab = GlobalFitTab(member_kind="groups")
    monkeypatch.setattr(tab, "_grouped_mode_context", lambda: (grouped_groups, [], "ready"))
    tab.set_current_dataset(dataset)
    row_by_name = {
        tab._group_param_table.item(row, 0).data(Qt.ItemDataRole.UserRole): row
        for row in range(tab._group_param_table.rowCount())
    }
    tab._group_param_table.item(row_by_name["background"], 1).setText("1.5")
    tab._group_param_table.item(row_by_name["background"], 2).setText("2.5")

    saved = tab.get_state()

    restored = GlobalFitTab(member_kind="groups")
    monkeypatch.setattr(restored, "_grouped_mode_context", lambda: (grouped_groups, [], "ready"))
    restored.set_current_dataset(dataset)
    restored.restore_state(saved)

    restored_row_by_name = {
        restored._group_param_table.item(row, 0).data(Qt.ItemDataRole.UserRole): row
        for row in range(restored._group_param_table.rowCount())
    }
    assert restored._group_param_table.item(restored_row_by_name["background"], 1).text() == "1.5"
    assert restored._group_param_table.item(restored_row_by_name["background"], 2).text() == "2.5"


def test_grouped_tab_preview_emits_curves_for_each_group(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grouped_groups = [
        SimpleNamespace(group_id=1, group_name="Forward", counts=np.array([120.0, 118.0])),
        SimpleNamespace(group_id=2, group_name="Backward", counts=np.array([80.0, 79.0])),
    ]
    grouped_datasets = [
        MuonDataset(
            time=np.array([0.0, 0.5, 1.0]),
            asymmetry=np.array([120.0, 110.0, 95.0]),
            error=np.array([1.0, 1.0, 1.0]),
            metadata={
                "group_id": 1,
                "grouped_time_domain": True,
                "run_number": 9001,
                "run_label": "Forward",
            },
            run=dataset.run,
        ),
        MuonDataset(
            time=np.array([0.0, 0.5, 1.0]),
            asymmetry=np.array([80.0, 75.0, 70.0]),
            error=np.array([1.0, 1.0, 1.0]),
            metadata={
                "group_id": 2,
                "grouped_time_domain": True,
                "run_number": 9002,
                "run_label": "Backward",
            },
            run=dataset.run,
        ),
    ]

    tab = GlobalFitTab(member_kind="groups")
    monkeypatch.setattr(
        tab,
        "_grouped_mode_context",
        lambda: (grouped_groups, grouped_datasets, "ready"),
    )
    tab.set_current_dataset(dataset)

    row_by_name = {
        tab._group_param_table.item(row, 0).data(Qt.ItemDataRole.UserRole): row
        for row in range(tab._group_param_table.rowCount())
    }
    tab._group_param_table.item(row_by_name["N0"], 1).setText("101")
    tab._group_param_table.item(row_by_name["N0"], 2).setText("202")

    emitted: dict[str, object] = {}
    tab.grouped_preview_requested.connect(
        lambda datasets, curves: emitted.update({"datasets": datasets, "curves": curves})
    )

    tab._on_preview_requested()

    assert emitted["datasets"] == grouped_datasets
    curves = emitted["curves"]
    assert set(curves) == {9001, 9002}
    assert curves[9001][0] is not None
    assert len(curves[9001][1][0]) >= len(grouped_datasets[0].time)


def test_grouped_mode_context_uses_current_fit_window(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = MuonDataset(
        time=np.array([1.0, 2.0, 3.0]),
        asymmetry=np.array([0.0, 0.0, 0.0]),
        error=np.array([1.0, 1.0, 1.0]),
        metadata={"run_number": 101},
        run=Run(run_number=101),
    )
    captured: dict[str, tuple[float | None, float | None]] = {}

    def _fake_groups(_dataset, *, t_min=None, t_max=None):
        captured["groups"] = (t_min, t_max)
        return [
            SimpleNamespace(group_id=1, group_name="Forward"),
            SimpleNamespace(group_id=2, group_name="Backward"),
        ]

    def _fake_datasets(_dataset, *, t_min=None, t_max=None):
        captured["datasets"] = (t_min, t_max)
        return [
            MuonDataset(
                time=np.array([1.0, 2.0, 3.0]),
                asymmetry=np.array([1.0, 1.0, 1.0]),
                error=np.array([1.0, 1.0, 1.0]),
                metadata={"run_number": -101001, "group_id": 1},
            ),
            MuonDataset(
                time=np.array([1.0, 2.0, 3.0]),
                asymmetry=np.array([1.0, 1.0, 1.0]),
                error=np.array([1.0, 1.0, 1.0]),
                metadata={"run_number": -101002, "group_id": 2},
            ),
        ]

    monkeypatch.setattr(fit_panel_module, "build_grouped_time_domain_groups", _fake_groups)
    monkeypatch.setattr(fit_panel_module, "build_grouped_time_domain_datasets", _fake_datasets)

    tab = GlobalFitTab(member_kind="groups")
    tab.set_current_dataset(dataset)

    grouped_groups, grouped_datasets, _message = tab._grouped_mode_context()

    assert grouped_groups is not None
    assert grouped_datasets is not None
    assert captured["groups"] == pytest.approx((1.0, 3.0))
    assert captured["datasets"] == (None, None)


def test_grouped_context_builds_members_for_multiple_runs(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _ds(run: int) -> MuonDataset:
        return MuonDataset(
            time=np.array([1.0, 2.0, 3.0]),
            asymmetry=np.array([0.0, 0.0, 0.0]),
            error=np.array([1.0, 1.0, 1.0]),
            metadata={"run_number": run},
            run=Run(run_number=run),
        )

    def _fake_groups(_dataset, *, t_min=None, t_max=None):
        return [
            SimpleNamespace(group_id=1, group_name="Forward"),
            SimpleNamespace(group_id=2, group_name="Backward"),
        ]

    def _fake_datasets(dataset, *, t_min=None, t_max=None):
        run = int(dataset.run_number)
        return [
            MuonDataset(
                time=np.array([1.0, 2.0]),
                asymmetry=np.array([1.0, 1.0]),
                error=np.array([1.0, 1.0]),
                metadata={"run_number": -(run * 1000 + i), "group_id": i, "source_run_number": run},
            )
            for i in (1, 2)
        ]

    monkeypatch.setattr(fit_panel_module, "build_grouped_time_domain_groups", _fake_groups)
    monkeypatch.setattr(fit_panel_module, "build_grouped_time_domain_datasets", _fake_datasets)

    tab = GlobalFitTab(member_kind="groups")
    tab.set_member_datasets([_ds(50), _ds(51)])

    groups, datasets, _message = tab._grouped_mode_context()

    assert groups is not None
    assert set(tab._grouped_members) == {50, 51}
    assert len(tab._grouped_members[50]) == 2
    assert len(datasets) == 4  # 2 runs × 2 groups


def test_grouped_series_fit_dispatches_for_multiple_members(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tab = GlobalFitTab(member_kind="groups")
    # Pretend the context found two member runs (the series-branch trigger).
    tab._grouped_members = {
        50: [SimpleNamespace(group_id=1)],
        51: [SimpleNamespace(group_id=1)],
    }
    monkeypatch.setattr(
        tab,
        "_grouped_mode_context",
        lambda: ([SimpleNamespace(group_id=1)], [SimpleNamespace()], "ready"),
    )
    monkeypatch.setattr(
        tab,
        "_parse_grouped_parameter_configuration",
        lambda: {
            "global": [],
            "local": [],
            "fixed": [],
            "model_values": {},
            "group_values": {},
            "bounds": {},
        },
    )
    monkeypatch.setattr(fit_panel_module, "validate_grouped_model_contract", lambda *a, **k: None)

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        tab,
        "_run_grouped_series_fit",
        lambda *args, **kwargs: captured.setdefault("called", True),
    )

    tab._run_grouped_time_domain_fit()

    assert captured.get("called") is True


def test_derive_grouped_relationship_maps_roles_and_rejects_mixing() -> None:
    derive = GlobalFitTab._derive_grouped_relationship
    # One member is always "individual" regardless of roles.
    assert derive({"lambda": "global"}, 1) == ("individual", None)
    # Multi-member: any Global physics → global; otherwise batch.
    assert derive({"lambda": "global"}, 3) == ("global", None)
    assert derive({"lambda": "local"}, 3) == ("batch", None)
    assert derive({}, 3) == ("batch", None)
    # Mixing Global + Local physics is rejected (A1 engine limit).
    relationship, error = derive({"a": "global", "b": "local"}, 3)
    assert relationship is None
    assert error is not None and "mix" in error.lower()


def test_send_single_model_to_batch_copies_model(qapp: QApplication) -> None:
    panel = FitPanel()
    panel._single_tab._set_composite_model(
        CompositeModel(["Gaussian", "Constant"], operators=["+"])
    )

    assert panel.send_single_model_to_batch() is True
    assert (
        panel._global_tab._composite_model.component_names
        == panel._single_tab._composite_model.component_names
    )


def test_send_to_batch_button_copies_model_and_switches_tab(qapp: QApplication) -> None:
    panel = FitPanel()
    panel._single_tab._set_composite_model(
        CompositeModel(["StretchedExponential", "Constant"], operators=["+"])
    )
    panel._tabs.setCurrentWidget(panel._single_tab)

    panel._single_tab._send_to_batch_btn.click()

    assert (
        panel._global_tab._composite_model.component_names
        == panel._single_tab._composite_model.component_names
    )
    assert panel._tabs.currentWidget() is panel._global_tab


def test_initial_values_dialog_edits_local_and_protects_shared(qapp: QApplication) -> None:
    from asymmetry.gui.panels.initial_values_dialog import InitialValuesDialog

    members = [(10, "10"), (11, "11")]
    params = [("A", "A", "local"), ("L", "L", "global"), ("bg", "bg", "fixed")]
    values = {10: {"A": 0.2, "L": 0.5, "bg": 0.0}, 11: {"A": 0.3, "L": 0.5, "bg": 0.0}}
    dlg = InitialValuesDialog(members, params, values)

    assert dlg._table.item(0, 0).flags() & Qt.ItemFlag.ItemIsEditable  # Local editable
    assert not (dlg._table.item(0, 1).flags() & Qt.ItemFlag.ItemIsEditable)  # Global read-only
    assert not (dlg._table.item(0, 2).flags() & Qt.ItemFlag.ItemIsEditable)  # Fixed read-only

    dlg._table.item(1, 0).setText("0.99")
    edited = dlg.edited_values()
    assert edited[11]["A"] == pytest.approx(0.99)
    # Only Local parameters are returned.
    assert "L" not in edited.get(10, {})
    assert "bg" not in edited.get(10, {})


def test_batch_initial_values_user_override_takes_precedence(qapp: QApplication) -> None:
    def _ds(run: int) -> MuonDataset:
        return MuonDataset(
            time=np.array([0.0, 0.1, 0.2]),
            asymmetry=np.array([0.1, 0.1, 0.1]),
            error=np.array([0.01, 0.01, 0.01]),
            metadata={"run_number": run},
            run=Run(run_number=run),
        )

    tab = GlobalFitTab(member_kind="runs")
    tab.set_datasets([_ds(10), _ds(11)])
    tab._set_composite_model(CompositeModel(["Exponential", "Constant"], operators=["+"]))
    pname = tab._composite_model.param_names[0]

    # _set_composite_model clears overrides, so set them afterwards.
    tab._user_initial_values_by_run = {10: {pname: 7.0}}
    parsed = tab._parse_parameter_configuration()
    effective = tab._effective_initial_values_by_run(parsed)

    assert effective[10][pname] == pytest.approx(7.0)  # user override wins
    assert 11 in effective  # other runs still present


def test_grouped_initial_values_user_override_per_run_group(qapp: QApplication) -> None:
    nuisance = ["N0", "background", "amplitude", "relative_phase"]
    tab = GlobalFitTab(member_kind="groups")
    groups = [
        SimpleNamespace(group_id=1, group_name="Forward"),
        SimpleNamespace(group_id=2, group_name="Backward"),
    ]
    tab._grouped_members = {42: groups}

    specs = tab._grouped_member_specs()
    assert [key for key, _label, _run, _gid in specs] == [-42001, -42002]

    # Override N0 for run 42's second group (synthetic key -42002).
    tab._user_grouped_initial_values = {-42002: {"N0": 123.0}}
    config = {
        "group_values": {name: {1: 1.0, 2: 2.0} for name in nuisance},
        "model_values": {},
        "bounds": {name: (-float("inf"), float("inf")) for name in nuisance},
        "fixed": set(),
    }

    initial = tab._build_grouped_initial_params(groups, config, run_number=42)

    assert initial[2]["N0"].value == pytest.approx(123.0)  # override applied
    assert initial[1]["N0"].value == pytest.approx(1.0)  # other group from the table


def test_single_tab_annotates_and_round_trips_batch_roles(qapp: QApplication) -> None:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    names = list(model.param_names)
    tab = SingleFitTab()
    roles = {names[0]: "global"}
    if len(names) >= 2:
        roles[names[1]] = "local"
    state = {
        "composite_model": model.to_dict(),
        "parameters": [
            {
                "name": n,
                "value": 0.1,
                "fixed": False,
                "min": "-inf",
                "max": "inf",
                "role": roles.get(n),
            }
            for n in names
        ],
        "result_html": "",
    }
    tab.restore_state(state)

    row0 = next(
        i
        for i in range(tab._param_table.rowCount())
        if tab._param_table.item(i, 0).data(Qt.ItemDataRole.UserRole) == names[0]
    )
    role_item = tab._param_table.item(row0, fit_panel_module._SINGLE_PARAM_BATCH_COLUMN)
    assert role_item.data(fit_panel_module._PARAM_BATCH_ROLE_DATA) == "global"
    assert role_item.text() == "Global"

    # The role round-trips through get_state (survives selection switches/save).
    round_tripped = {p["name"]: p.get("role") for p in tab.get_state()["parameters"]}
    assert round_tripped.get(names[0]) == "global"


def test_grouped_tab_preview_rejects_conflicting_fit_function_amplitude(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grouped_groups = [
        SimpleNamespace(group_id=1, group_name="Forward", counts=np.array([120.0, 118.0])),
        SimpleNamespace(group_id=2, group_name="Backward", counts=np.array([80.0, 79.0])),
    ]
    grouped_datasets = [
        MuonDataset(
            time=np.array([0.0, 0.5, 1.0]),
            asymmetry=np.array([120.0, 110.0, 95.0]),
            error=np.array([1.0, 1.0, 1.0]),
            metadata={"group_id": 1, "grouped_time_domain": True, "run_number": 9001},
            run=dataset.run,
        ),
        MuonDataset(
            time=np.array([0.0, 0.5, 1.0]),
            asymmetry=np.array([80.0, 75.0, 70.0]),
            error=np.array([1.0, 1.0, 1.0]),
            metadata={"group_id": 2, "grouped_time_domain": True, "run_number": 9002},
            run=dataset.run,
        ),
    ]

    tab = GlobalFitTab(member_kind="groups")
    monkeypatch.setattr(
        tab,
        "_grouped_mode_context",
        lambda: (grouped_groups, grouped_datasets, "ready"),
    )
    tab.set_current_dataset(dataset)

    row_by_name = {
        tab._group_model_table.item(row, 0).data(Qt.ItemDataRole.UserRole): row
        for row in range(tab._group_model_table.rowCount())
    }
    assert "A_1" not in row_by_name

    emitted: dict[str, object] = {}
    tab.grouped_preview_requested.connect(
        lambda datasets, curves: emitted.update({"datasets": datasets, "curves": curves})
    )

    tab._on_preview_requested()

    assert "datasets" in emitted
    assert "curves" in emitted


def test_grouped_mode_ui_refresh_rebuilds_group_value_columns_when_groups_appear(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tab = GlobalFitTab(member_kind="groups")
    grouped_groups = [
        SimpleNamespace(group_id=1, group_name="Forward", counts=np.array([120.0, 118.0])),
        SimpleNamespace(group_id=2, group_name="Backward", counts=np.array([80.0, 79.0])),
    ]
    state = {"ready": False}

    def _context():
        if not state["ready"]:
            return None, None, "not ready"
        return grouped_groups, [], "ready"

    monkeypatch.setattr(tab, "_grouped_mode_context", _context)

    tab.set_current_dataset(dataset)
    assert tab._group_param_table.columnCount() == 4

    state["ready"] = True
    tab._update_mode_ui(preserve_result=True)

    headers = [
        tab._group_param_table.horizontalHeaderItem(column).text()
        for column in range(tab._group_param_table.columnCount())
    ]
    row_by_name = {
        tab._group_param_table.item(row, 0).data(Qt.ItemDataRole.UserRole): row
        for row in range(tab._group_param_table.rowCount())
    }
    _background_seed, backward_n0_seed, _amplitude_seed = (
        fit_panel_module._seed_group_background_and_n0(np.array([80.0, 79.0]))
    )
    assert headers == ["Parameter", "Forward", "Backward", "Type", "Bounds"]
    assert float(tab._group_param_table.item(row_by_name["N0"], 2).text()) == pytest.approx(
        backward_n0_seed
    )
    assert tab._group_param_table.cellWidget(row_by_name["N0"], 2) is None
    assert isinstance(tab._group_param_table.cellWidget(row_by_name["N0"], 3), QComboBox)


def test_grouped_seed_estimates_background_n0_and_amplitude_from_counts_definition() -> None:
    time = np.linspace(0.0, 8.0, 801)
    n0 = 100.0
    background = 12.0
    amplitude = 0.18
    polarization = amplitude * np.cos(2.0 * np.pi * 1.2 * time)
    fast_component = 80.0 * np.exp(-time / 0.05)
    counts = (
        n0 * (1.0 + polarization)
        + background * np.exp(time / float(MUON_LIFETIME_US))
        + fast_component
    )

    estimated_background, estimated_n0, estimated_amplitude = (
        fit_panel_module._seed_group_background_and_n0(
            counts,
            time=time,
        )
    )

    assert estimated_background == pytest.approx(background, rel=0.2)
    assert estimated_n0 == pytest.approx(n0, rel=0.2)
    assert estimated_amplitude == pytest.approx(amplitude, rel=0.25)


def test_single_tab_keeps_model_tools_inline(qapp: QApplication) -> None:
    tab = SingleFitTab()

    assert not hasattr(tab, "_model_tools_section")


def test_global_tab_keeps_model_tools_inline(qapp: QApplication) -> None:
    tab = GlobalFitTab()

    assert not hasattr(tab, "_model_tools_section")


def test_formula_label_shows_raw_formula_and_tooltip(qapp: QApplication) -> None:
    tab = GlobalFitTab()
    model = CompositeModel(["StretchedExponential", "Constant"], operators=["+"])

    tab._set_composite_model(model)

    assert tab._formula_label.text() == model.formula_string()
    assert tab._formula_label.toolTip() == model.formula_string()


def test_single_edit_function_updates_parameter_rows(qapp: QApplication) -> None:
    tab = SingleFitTab()
    tab._set_composite_model(CompositeModel(["Exponential", "Constant"], operators=["+"]))
    assert tab._param_table.rowCount() == 3


def test_global_edit_function_updates_parameter_rows(qapp: QApplication) -> None:
    tab = GlobalFitTab()
    tab._set_composite_model(CompositeModel(["Gaussian", "Constant"], operators=["+"]))
    assert tab._param_table.rowCount() == 3


def test_global_tab_inherits_model_and_average_values_from_single_fits(
    qapp: QApplication,
    dataset: MuonDataset,
) -> None:
    tab = GlobalFitTab()
    d2 = MuonDataset(dataset.time, dataset.asymmetry, dataset.error, {"run_number": 102})
    tab.set_datasets([dataset, d2])

    model = CompositeModel(["Gaussian", "Constant"], operators=["+"])
    p1 = ParameterSet([Parameter("A_1", 0.20), Parameter("sigma", 1.1), Parameter("A_bg", 0.01)])
    p2 = ParameterSet([Parameter("A_1", 0.30), Parameter("sigma", 1.5), Parameter("A_bg", 0.03)])
    r1 = FitResult(success=True, parameters=p1)
    r2 = FitResult(success=True, parameters=p2)

    tab.register_single_fit_seed(101, model, r1)
    tab.register_single_fit_seed(102, model, r2)

    assert tab._composite_model.to_dict() == model.to_dict()

    value_by_name = {}
    for row in range(tab._param_table.rowCount()):
        name_item = tab._param_table.item(row, 0)
        value_item = tab._param_table.item(row, 1)
        assert name_item is not None
        assert value_item is not None
        pname = name_item.data(Qt.ItemDataRole.UserRole)
        value_by_name[pname] = float(value_item.text())

    assert value_by_name["A_1"] == pytest.approx(0.25)
    assert value_by_name["sigma"] == pytest.approx(1.3)
    assert value_by_name["A_bg"] == pytest.approx(0.02)


def test_global_fit_uses_inherited_local_values_per_run(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tab = GlobalFitTab()
    d2 = MuonDataset(dataset.time, dataset.asymmetry, dataset.error, {"run_number": 102})
    tab.set_datasets([dataset, d2])

    model = tab._composite_model
    p1 = ParameterSet([Parameter("A_1", 0.22), Parameter("Lambda", 0.40), Parameter("A_bg", 0.01)])
    p2 = ParameterSet([Parameter("A_1", 0.30), Parameter("Lambda", 0.85), Parameter("A_bg", 0.02)])
    tab.register_single_fit_seed(101, model, FitResult(success=True, parameters=p1))
    tab.register_single_fit_seed(102, model, FitResult(success=True, parameters=p2))

    # Enforce classification for this test case.
    row_by_name = {}
    for row in range(tab._param_table.rowCount()):
        name_item = tab._param_table.item(row, 0)
        assert name_item is not None
        row_by_name[name_item.data(Qt.ItemDataRole.UserRole)] = row

    global_combo = tab._param_table.cellWidget(row_by_name["A_1"], 2)
    local_combo = tab._param_table.cellWidget(row_by_name["Lambda"], 2)
    fixed_combo = tab._param_table.cellWidget(row_by_name["A_bg"], 2)
    assert isinstance(global_combo, QComboBox)
    assert isinstance(local_combo, QComboBox)
    assert isinstance(fixed_combo, QComboBox)
    global_combo.setCurrentText("Global")
    local_combo.setCurrentText("Local")
    fixed_combo.setCurrentText("Fixed")

    captured: dict[str, object] = {}

    class _DummySignal:
        def connect(self, *_args, **_kwargs):
            return None

    class _FakeThread:
        def __init__(self):
            self.started = _DummySignal()
            self.finished = _DummySignal()

        def start(self):
            return None

        def quit(self):
            return None

        def wait(self):
            return None

        def deleteLater(self):
            return None

    class _FakeWorker:
        def __init__(
            self,
            _fit_engine,
            _datasets,
            _model_fn,
            _global_params,
            _local_params,
            initial_params,
        ):
            captured["initial_params"] = initial_params
            self.finished = _DummySignal()
            self.error = _DummySignal()

        def moveToThread(self, _thread):
            return None

        def run(self):
            return None

        def deleteLater(self):
            return None

    monkeypatch.setattr(fit_panel_module, "QThread", _FakeThread)
    monkeypatch.setattr(fit_panel_module, "GlobalFitWorker", _FakeWorker)

    tab._run_global_fit()

    initial_params = captured["initial_params"]
    pset_101 = initial_params[101]
    pset_102 = initial_params[102]

    assert pset_101["Lambda"].value == pytest.approx(0.40)
    assert pset_102["Lambda"].value == pytest.approx(0.85)
    # Global/fixed parameters are seeded from per-run averages.
    assert pset_101["A_1"].value == pytest.approx(0.26)
    assert pset_102["A_1"].value == pytest.approx(0.26)
    assert pset_101["A_bg"].value == pytest.approx(0.015)
    assert pset_102["A_bg"].value == pytest.approx(0.015)


def test_fit_panel_restores_single_fit_state_per_dataset(
    qapp: QApplication, dataset: MuonDataset
) -> None:
    panel = FitPanel()
    d1 = dataset
    d2 = MuonDataset(dataset.time, dataset.asymmetry, dataset.error, {"run_number": 102})

    panel.set_dataset(d1)
    panel._single_tab._result_label.setText("fit for run 101")
    panel._single_tab._param_table.item(0, 1).setText("0.123")

    panel.set_dataset(d2)
    panel._single_tab._result_label.setText("fit for run 102")
    panel._single_tab._param_table.item(0, 1).setText("0.456")

    panel.set_dataset(d1)
    assert "fit for run 101" in panel._single_tab._result_label.text()
    assert float(panel._single_tab._param_table.item(0, 1).text()) == pytest.approx(0.123)

    panel.set_dataset(d2)
    assert "fit for run 102" in panel._single_tab._result_label.text()
    assert float(panel._single_tab._param_table.item(0, 1).text()) == pytest.approx(0.456)


def test_fit_panel_single_state_roundtrip_preserves_per_run_states(
    qapp: QApplication,
    dataset: MuonDataset,
) -> None:
    panel = FitPanel()
    d1 = dataset
    d2 = MuonDataset(dataset.time, dataset.asymmetry, dataset.error, {"run_number": 102})

    panel.set_dataset(d1)
    panel._single_tab._result_label.setText("saved fit 101")
    panel.set_dataset(d2)
    panel._single_tab._result_label.setText("saved fit 102")

    saved = panel.get_single_state()
    assert isinstance(saved.get("states_by_run"), dict)
    assert "101" in saved["states_by_run"]
    assert "102" in saved["states_by_run"]

    restored = FitPanel()
    restored.set_dataset(d1)
    restored.restore_single_state(saved)
    assert "saved fit 101" in restored._single_tab._result_label.text()

    restored.set_dataset(d2)
    assert "saved fit 102" in restored._single_tab._result_label.text()


def test_single_tab_state_roundtrip_preserves_cached_wizard_results(
    qapp: QApplication,
    dataset: MuonDataset,
) -> None:
    tab = SingleFitTab()
    tab.set_dataset(dataset)
    _assessment, recommendation = _wizard_payload_for_dataset(dataset)
    signature = {
        "run_number": int(dataset.run_number),
        "model": tab._composite_model.to_dict(),
    }

    tab._cache_wizard_analysis(recommendation, signature=signature, log_text="cached log")
    saved = tab.get_state()

    restored = SingleFitTab()
    restored.set_dataset(dataset)
    restored.restore_state(saved)

    assert restored._cached_wizard_recommendation is not None
    assert restored._cached_wizard_recommendation.summary == recommendation.summary
    assert restored._cached_wizard_signature == signature
    assert restored._cached_wizard_log_text == "cached log"


def test_fit_panel_single_state_roundtrip_preserves_per_run_wizard_state(
    qapp: QApplication,
    dataset: MuonDataset,
) -> None:
    panel = FitPanel()
    d1 = dataset
    d2 = MuonDataset(dataset.time, dataset.asymmetry, dataset.error, {"run_number": 102})
    _assessment_1, recommendation_1 = _wizard_payload_for_dataset(d1)
    _assessment_2, recommendation_2 = _wizard_payload_for_dataset(d2)

    panel.set_dataset(d1)
    panel._single_tab._cache_wizard_analysis(
        recommendation_1,
        signature={
            "run_number": int(d1.run_number),
            "model": panel._single_tab._composite_model.to_dict(),
        },
        log_text="run 101 log",
    )
    panel.set_dataset(d2)
    panel._single_tab._cache_wizard_analysis(
        recommendation_2,
        signature={
            "run_number": int(d2.run_number),
            "model": panel._single_tab._composite_model.to_dict(),
        },
        log_text="run 102 log",
    )

    saved = panel.get_single_state()

    restored = FitPanel()
    restored.set_dataset(d1)
    restored.restore_single_state(saved)
    assert restored._single_tab._cached_wizard_recommendation is not None
    assert restored._single_tab._cached_wizard_log_text == "run 101 log"

    restored.set_dataset(d2)
    assert restored._single_tab._cached_wizard_recommendation is not None
    assert restored._single_tab._cached_wizard_log_text == "run 102 log"


def test_fit_panel_persist_single_fit_wizard_cache_restores_unopened_run(
    qapp: QApplication,
    dataset: MuonDataset,
) -> None:
    panel = FitPanel()
    d1 = dataset
    d2 = MuonDataset(dataset.time, dataset.asymmetry, dataset.error, {"run_number": 102})
    _assessment, recommendation = _wizard_payload_for_dataset(d2)

    panel.set_dataset(d1)
    panel.persist_single_fit_wizard_cache_for_run(
        102,
        recommendation,
        signature={"run_number": 102, "model": None},
        log_text="phase 1",
    )

    panel.set_dataset(d2)

    assert panel._single_tab._cached_wizard_recommendation is not None
    assert panel._single_tab._cached_wizard_recommendation.summary == recommendation.summary
    assert panel._single_tab._cached_wizard_log_text == "phase 1"
    assert panel._single_tab._cached_wizard_signature == {"run_number": 102, "model": None}


def test_fit_panel_global_fit_results_seed_single_state_per_run(
    qapp: QApplication,
    dataset: MuonDataset,
) -> None:
    panel = FitPanel()
    d1 = dataset
    d2 = MuonDataset(dataset.time, dataset.asymmetry, dataset.error, {"run_number": 102})

    panel.set_dataset(d1)

    model = panel._global_tab._composite_model
    pnames = model.param_names

    def _fit_result(values: list[float]) -> FitResult:
        params = ParameterSet(
            [Parameter(name=name, value=value) for name, value in zip(pnames, values, strict=False)]
        )
        return FitResult(
            success=True,
            chi_squared=2.0,
            reduced_chi_squared=1.0,
            parameters=params,
            uncertainties={name: 0.01 for name in pnames},
        )

    results = {
        101: (_fit_result([0.11, 0.22, 0.33]), (np.array([0.0, 1.0]), np.array([0.2, 0.1])), []),
        102: (_fit_result([0.44, 0.55, 0.66]), (np.array([0.0, 1.0]), np.array([0.2, 0.1])), []),
    }
    panel.register_global_fit_results(results)

    assert "Batch fit" in panel._single_tab._result_label.text()
    assert float(panel._single_tab._param_table.item(0, 1).text()) == pytest.approx(0.11)

    panel.set_dataset(d2)
    assert "Batch fit" in panel._single_tab._result_label.text()
    assert float(panel._single_tab._param_table.item(0, 1).text()) == pytest.approx(0.44)

    saved = panel.get_single_state()
    assert "101" in saved.get("states_by_run", {})
    assert "102" in saved.get("states_by_run", {})


def test_global_fit_results_preserve_parameter_type_selection(
    qapp: QApplication,
    dataset: MuonDataset,
) -> None:
    panel = FitPanel()
    d1 = dataset
    d2 = MuonDataset(dataset.time, dataset.asymmetry, dataset.error, {"run_number": 102})
    panel.set_datasets([d1, d2])

    first_type = panel._global_tab._param_table.cellWidget(0, 2)
    second_type = panel._global_tab._param_table.cellWidget(1, 2)
    assert isinstance(first_type, QComboBox)
    assert isinstance(second_type, QComboBox)
    first_type.setCurrentText("Local")
    second_type.setCurrentText("Global")

    model = panel._global_tab._composite_model
    pnames = model.param_names

    def _fit_result(values: list[float]) -> FitResult:
        params = ParameterSet(
            [Parameter(name=name, value=value) for name, value in zip(pnames, values, strict=False)]
        )
        return FitResult(
            success=True, parameters=params, uncertainties={name: 0.01 for name in pnames}
        )

    results = {
        101: (_fit_result([0.11, 0.22, 0.33]), (np.array([0.0, 1.0]), np.array([0.2, 0.1])), []),
        102: (_fit_result([0.44, 0.55, 0.66]), (np.array([0.0, 1.0]), np.array([0.2, 0.1])), []),
    }

    panel.register_global_fit_results(results)

    first_type_after = panel._global_tab._param_table.cellWidget(0, 2)
    second_type_after = panel._global_tab._param_table.cellWidget(1, 2)
    assert isinstance(first_type_after, QComboBox)
    assert isinstance(second_type_after, QComboBox)
    assert first_type_after.currentText() == "Local"
    assert second_type_after.currentText() == "Global"


def test_fit_panel_share_single_function_state_to_other_runs(
    qapp: QApplication,
    dataset: MuonDataset,
) -> None:
    panel = FitPanel()
    d1 = dataset
    d2 = MuonDataset(dataset.time, dataset.asymmetry, dataset.error, {"run_number": 102})
    d3 = MuonDataset(dataset.time, dataset.asymmetry, dataset.error, {"run_number": 103})

    panel.set_dataset(d1)
    panel._single_tab._param_table.item(0, 1).setText("0.777")
    panel._single_tab._result_label.setText("source fit result")

    datasets_by_run = {101: d1, 102: d2, 103: d3}
    copied = panel.share_single_function_state(101, [102, 103], datasets_by_run=datasets_by_run)
    assert copied == 2

    panel.set_dataset(d2)
    assert float(panel._single_tab._param_table.item(0, 1).text()) == pytest.approx(0.777)
    assert panel._single_tab._result_label.text() == "No fit performed yet"

    panel.set_dataset(d3)
    assert float(panel._single_tab._param_table.item(0, 1).text()) == pytest.approx(0.777)
    assert panel._single_tab._result_label.text() == "No fit performed yet"


def test_fit_panel_share_single_function_seeds_bl_from_target_field(
    qapp: QApplication,
    dataset: MuonDataset,
) -> None:
    """B_L parameter should be seeded from each target dataset's applied field."""
    from asymmetry.core.fitting.composite import CompositeModel

    panel = FitPanel()
    d1 = MuonDataset(
        dataset.time,
        dataset.asymmetry,
        dataset.error,
        {"run_number": 101, "field": 50.0},
    )
    d2 = MuonDataset(
        dataset.time,
        dataset.asymmetry,
        dataset.error,
        {"run_number": 102, "field": 150.0},
    )
    d3 = MuonDataset(
        dataset.time,
        dataset.asymmetry,
        dataset.error,
        {"run_number": 103, "field": 300.0},
    )

    # Set an LF-KT model on d1 and adjust B_L to 50 G (the source field).
    panel.set_dataset(d1)
    lf_model = CompositeModel(["LongitudinalFieldKT", "Constant"], operators=["+"])
    panel._single_tab._set_composite_model(lf_model)
    # Find the B_L row and set it to the source field.
    for row in range(panel._single_tab._param_table.rowCount()):
        item = panel._single_tab._param_table.item(row, 0)
        if item and item.data(Qt.ItemDataRole.UserRole) == "B_L":
            panel._single_tab._param_table.item(row, 1).setText("50.0")
    panel._single_tab._result_label.setText("source fit result")

    datasets_by_run = {101: d1, 102: d2, 103: d3}
    copied = panel.share_single_function_state(101, [102, 103], datasets_by_run=datasets_by_run)
    assert copied == 2

    # d2 has field=150 G — B_L in shared state should be overridden to 150.
    panel.set_dataset(d2)
    bl_value_d2 = None
    for row in range(panel._single_tab._param_table.rowCount()):
        item = panel._single_tab._param_table.item(row, 0)
        if item and item.data(Qt.ItemDataRole.UserRole) == "B_L":
            bl_value_d2 = float(panel._single_tab._param_table.item(row, 1).text())
    assert bl_value_d2 == pytest.approx(150.0)

    # d3 has field=300 G.
    panel.set_dataset(d3)
    bl_value_d3 = None
    for row in range(panel._single_tab._param_table.rowCount()):
        item = panel._single_tab._param_table.item(row, 0)
        if item and item.data(Qt.ItemDataRole.UserRole) == "B_L":
            bl_value_d3 = float(panel._single_tab._param_table.item(row, 1).text())
    assert bl_value_d3 == pytest.approx(300.0)


def test_fit_panel_clear_fits_for_runs_removes_cached_fit_state(
    qapp: QApplication,
    dataset: MuonDataset,
) -> None:
    panel = FitPanel()
    d1 = dataset
    d2 = MuonDataset(dataset.time, dataset.asymmetry, dataset.error, {"run_number": 102})

    panel.set_dataset(d1)
    panel._single_tab._result_label.setText("fit for run 101")
    panel._single_state_by_run[101] = panel._single_tab.get_state()

    panel.set_dataset(d2)
    panel._single_tab._result_label.setText("fit for run 102")
    panel._single_state_by_run[102] = panel._single_tab.get_state()

    panel._global_tab._single_fit_seed_by_run[101] = {"model": {}, "values": {"A": 0.1}}
    panel._global_tab._single_fit_seed_by_run[102] = {"model": {}, "values": {"A": 0.2}}

    cleared = panel.clear_fits_for_runs([101])

    assert cleared == 1
    assert 101 not in panel._single_state_by_run
    assert 101 not in panel._global_tab._single_fit_seed_by_run
    assert 102 in panel._single_state_by_run
    assert 102 in panel._global_tab._single_fit_seed_by_run


def test_global_fit_wizard_button_tracks_dataset_and_block_state(
    qapp: QApplication, dataset: MuonDataset
) -> None:
    tab = GlobalFitTab()
    assert tab._fit_wizard_btn.isEnabled() is False

    d2 = MuonDataset(dataset.time, dataset.asymmetry, dataset.error, {"run_number": 102})
    tab.set_datasets([dataset, d2])
    assert tab._fit_wizard_btn.isEnabled() is True

    tab.set_fit_blocked(True, "blocked")
    assert tab._fit_wizard_btn.isEnabled() is False


def test_global_fit_apply_fit_wizard_assessment_updates_roles_and_emits(
    qapp: QApplication,
    dataset: MuonDataset,
) -> None:
    tab = GlobalFitTab()
    d2 = MuonDataset(
        dataset.time,
        0.2 * np.exp(-0.6 * dataset.time) + 0.01,
        dataset.error,
        {"run_number": 102, "run_label": "102", "field": 100.0},
    )
    tab.set_datasets([dataset, d2])

    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    fit_results: dict[int, FitResult] = {}
    fitted_curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    component_curves: dict[int, tuple[tuple[str, np.ndarray], ...]] = {}
    for run_number, lam, ds in ((int(dataset.run_number), 0.25, dataset), (102, 0.6, d2)):
        params = ParameterSet(
            [
                Parameter("A_1", value=0.2, min=0.0, max=1.0),
                Parameter("Lambda", value=lam, min=0.0, max=5.0),
                Parameter("A_bg", value=0.01, min=-0.5, max=0.5),
            ]
        )
        curve = model.function(ds.time, A_1=0.2, Lambda=lam, A_bg=0.01)
        fit_results[run_number] = FitResult(
            success=True,
            chi_squared=5.0,
            reduced_chi_squared=0.1,
            parameters=params,
            uncertainties={"A_1": 0.01, "Lambda": 0.02, "A_bg": 0.001},
            residuals=np.asarray(ds.asymmetry - curve, dtype=float),
        )
        fitted_curves[run_number] = (
            np.asarray(ds.time, dtype=float).copy(),
            np.asarray(curve, dtype=float),
        )
        component_curves[run_number] = tuple(
            model.evaluate_components(ds.time, additive_only=True, A_1=0.2, Lambda=lam, A_bg=0.01)
        )

    assessment = GlobalCandidateAssessment(
        template=CandidateTemplate(
            key="exp_constant",
            title="Exponential + Constant",
            category="General",
            rationale="Baseline candidate",
            model=model,
        ),
        fit_results_by_run=fit_results,
        global_parameters=ParameterSet(
            [
                Parameter("A_1", value=0.2, min=0.0, max=1.0),
                Parameter("A_bg", value=0.01, min=-0.5, max=0.5),
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
                global_score=14.0,
                local_score=8.0,
                score_delta=6.0,
                total_variation=1.5,
                roughness=0.1,
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
    recommendation = GlobalFitWizardRecommendation(
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

    emitted: dict[str, object] = {}
    tab.global_fit_completed.connect(
        lambda results, global_params: emitted.update({"results": results, "global": global_params})
    )

    tab._apply_fit_wizard_assessment(assessment, recommendation)

    lambda_row = 1
    lambda_combo = tab._param_table.cellWidget(lambda_row, 2)
    assert isinstance(lambda_combo, QComboBox)
    assert lambda_combo.currentText() == "Local"
    assert "Global Fit Wizard" in tab._result_text.toPlainText()
    assert "results" in emitted


def test_global_tab_state_roundtrip_preserves_cached_wizard_results(
    qapp: QApplication,
    dataset: MuonDataset,
) -> None:
    tab = GlobalFitTab()
    dataset_102 = MuonDataset(dataset.time, dataset.asymmetry, dataset.error, {"run_number": 102})
    tab.set_datasets([dataset, dataset_102])

    recommendation = _global_wizard_recommendation_for_dataset(dataset)
    signature = {
        "run_numbers": [int(dataset.run_number), 102],
        "model": tab._composite_model.to_dict(),
        "types": {"A_1": "Global", "Lambda": "Local", "A_bg": "Global"},
        "values": {"A_1": 0.2, "Lambda": 0.4, "A_bg": 0.01},
        "bounds": {"A_1": [0.0, 1.0], "Lambda": [0.0, 5.0], "A_bg": [-0.5, 0.5]},
    }
    tab._cache_wizard_analysis(recommendation, signature=signature, log_text="cached log")

    saved = tab.get_state()
    assert len(saved["wizard_state_by_run_set"]) == 1

    restored = GlobalFitTab()
    restored.set_datasets([dataset, dataset_102])
    restored.restore_state(saved)

    assert restored._cached_wizard_recommendation is not None
    assert restored._cached_wizard_recommendation.summary == recommendation.summary
    assert restored._cached_wizard_signature == signature
    assert restored._cached_wizard_log_text == "cached log"


def test_single_fit_fraction_rows_auto_complete_final_fraction(qapp: QApplication) -> None:
    tab = SingleFitTab()
    tab._set_composite_model(
        CompositeModel.from_expression("( Exponential + Gaussian + Constant ){frac}")
    )

    row_by_name = {
        tab._param_table.item(row, 0).data(Qt.ItemDataRole.UserRole): row
        for row in range(tab._param_table.rowCount())
    }
    final_item = tab._param_table.item(row_by_name["fraction_3"], 1)
    assert final_item is not None
    assert not bool(final_item.flags() & Qt.ItemFlag.ItemIsEditable)

    tab._param_table.item(row_by_name["fraction_1"], 1).setText("0.2")
    tab._param_table.item(row_by_name["fraction_2"], 1).setText("0.3")

    assert tab._param_table.item(row_by_name["fraction_3"], 1).text() == "0.5"


def test_global_fit_fraction_rows_auto_complete_final_fraction(qapp: QApplication) -> None:
    tab = GlobalFitTab()
    tab._set_composite_model(
        CompositeModel.from_expression("( Exponential + Gaussian + Constant ){frac}")
    )

    row_by_name = {
        tab._param_table.item(row, 0).data(Qt.ItemDataRole.UserRole): row
        for row in range(tab._param_table.rowCount())
    }
    final_item = tab._param_table.item(row_by_name["fraction_3"], 1)
    bounds_item = tab._param_table.item(row_by_name["fraction_3"], 3)
    assert final_item is not None
    assert not bool(final_item.flags() & Qt.ItemFlag.ItemIsEditable)
    assert bounds_item is not None and bounds_item.text() == "0, 1"

    tab._param_table.item(row_by_name["fraction_1"], 1).setText("0.25")
    tab._param_table.item(row_by_name["fraction_2"], 1).setText("0.5")

    assert tab._param_table.item(row_by_name["fraction_3"], 1).text() == "0.25"


def test_global_tab_reopens_cached_results_for_prior_run_set_after_switching_groups(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tab = GlobalFitTab()
    dataset_102 = MuonDataset(dataset.time, dataset.asymmetry, dataset.error, {"run_number": 102})
    dataset_103 = MuonDataset(dataset.time, dataset.asymmetry, dataset.error, {"run_number": 103})

    tab.set_datasets([dataset, dataset_102])
    signature_12 = tab._wizard_context_signature(tab._parse_parameter_configuration())
    recommendation_12 = replace(
        _global_wizard_recommendation_for_dataset(dataset),
        summary="cached 101/102",
    )
    tab._cache_wizard_analysis(recommendation_12, signature=signature_12, log_text="group 12")

    tab.set_datasets([dataset, dataset_103])
    signature_13 = tab._wizard_context_signature(tab._parse_parameter_configuration())
    recommendation_13 = replace(
        _global_wizard_recommendation_for_dataset(dataset),
        summary="cached 101/103",
    )
    tab._cache_wizard_analysis(recommendation_13, signature=signature_13, log_text="group 13")

    captured: dict[str, object] = {}

    class _FakeGlobalWizard:
        def __init__(self, _parent) -> None:
            self.apply_assessment_requested = SimpleNamespace(connect=lambda _cb: None)
            self.analysis_cached = SimpleNamespace(connect=lambda _cb: None)
            self.single_fit_recommendations_generated = SimpleNamespace(connect=lambda _cb: None)
            self.parameter_setup_applied = SimpleNamespace(connect=lambda _cb: None)

        def set_analysis_context(self, datasets_arg, **kwargs) -> None:
            captured["datasets"] = datasets_arg
            captured["single_fit"] = kwargs.get("existing_single_fit_recommendations_by_run")

        def set_cached_recommendation(self, recommendation, **kwargs) -> None:
            captured["recommendation"] = recommendation
            captured["status_text"] = kwargs.get("status_text")
            captured["signature"] = kwargs.get("signature")

        def show(self) -> None:
            captured["show"] = True

        def raise_(self) -> None:
            captured["raise"] = True

        def activateWindow(self) -> None:
            captured["activate"] = True

    monkeypatch.setattr(fit_panel_module, "GlobalFitWizardWindow", _FakeGlobalWizard)

    tab.set_datasets([dataset, dataset_102])
    tab._fit_wizard_window = None
    tab._open_fit_wizard()

    assert captured["datasets"] == [dataset, dataset_102]
    assert captured["show"] is True
    assert captured["recommendation"].summary == "cached 101/102"
    assert captured["status_text"] is None
    assert captured["signature"] == signature_12


def test_global_tab_reopens_historical_results_for_same_run_set_when_signature_changes(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tab = GlobalFitTab()
    dataset_102 = MuonDataset(dataset.time, dataset.asymmetry, dataset.error, {"run_number": 102})
    tab.set_datasets([dataset, dataset_102])

    signature = tab._wizard_context_signature(tab._parse_parameter_configuration())
    recommendation = replace(
        _global_wizard_recommendation_for_dataset(dataset),
        summary="cached same-group result",
    )
    tab._cache_wizard_analysis(recommendation, signature=signature, log_text="old setup")

    lambda_value_item = tab._param_table.item(1, 1)
    assert lambda_value_item is not None
    lambda_value_item.setText("0.55")

    captured: dict[str, object] = {}

    class _FakeGlobalWizard:
        def __init__(self, _parent) -> None:
            self.apply_assessment_requested = SimpleNamespace(connect=lambda _cb: None)
            self.analysis_cached = SimpleNamespace(connect=lambda _cb: None)
            self.single_fit_recommendations_generated = SimpleNamespace(connect=lambda _cb: None)
            self.parameter_setup_applied = SimpleNamespace(connect=lambda _cb: None)

        def set_analysis_context(self, _datasets_arg, **_kwargs) -> None:
            return None

        def set_cached_recommendation(self, recommendation, **kwargs) -> None:
            captured["recommendation"] = recommendation
            captured["status_text"] = kwargs.get("status_text")

        def show(self) -> None:
            return None

        def raise_(self) -> None:
            return None

        def activateWindow(self) -> None:
            return None

    monkeypatch.setattr(fit_panel_module, "GlobalFitWizardWindow", _FakeGlobalWizard)

    tab._fit_wizard_window = None
    tab._open_fit_wizard()

    assert captured["recommendation"].summary == "cached same-group result"
    assert isinstance(captured["status_text"], str)
    assert "previously cached Global Fit Wizard results" in captured["status_text"]


def test_global_tab_state_roundtrip_preserves_multiple_run_set_wizard_caches(
    qapp: QApplication,
    dataset: MuonDataset,
) -> None:
    tab = GlobalFitTab()
    dataset_102 = MuonDataset(dataset.time, dataset.asymmetry, dataset.error, {"run_number": 102})
    dataset_103 = MuonDataset(dataset.time, dataset.asymmetry, dataset.error, {"run_number": 103})

    tab.set_datasets([dataset, dataset_102])
    signature_12 = tab._wizard_context_signature(tab._parse_parameter_configuration())
    recommendation_12 = replace(
        _global_wizard_recommendation_for_dataset(dataset),
        summary="cached 101/102",
    )
    tab._cache_wizard_analysis(recommendation_12, signature=signature_12, log_text="group 12")

    tab.set_datasets([dataset, dataset_103])
    signature_13 = tab._wizard_context_signature(tab._parse_parameter_configuration())
    recommendation_13 = replace(
        _global_wizard_recommendation_for_dataset(dataset),
        summary="cached 101/103",
    )
    tab._cache_wizard_analysis(recommendation_13, signature=signature_13, log_text="group 13")

    saved = tab.get_state()

    assert len(saved["wizard_state_by_run_set"]) == 2

    restored = GlobalFitTab()
    restored.set_datasets([dataset, dataset_102])
    restored.restore_state(saved)

    assert len(restored._wizard_cache_by_run_set) == 2
    assert restored._cached_wizard_recommendation is not None
    assert restored._cached_wizard_recommendation.summary == "cached 101/102"

    restored.set_datasets([dataset, dataset_103])

    assert restored._cached_wizard_recommendation is not None
    assert restored._cached_wizard_recommendation.summary == "cached 101/103"


def test_global_tab_open_fit_wizard_passes_cached_single_fit_recommendations(
    qapp: QApplication,
    dataset: MuonDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = FitPanel()
    d1 = dataset
    d2 = MuonDataset(dataset.time, dataset.asymmetry, dataset.error, {"run_number": 102})
    _assessment_1, recommendation_1 = _wizard_payload_for_dataset(d1)
    _assessment_2, recommendation_2 = _wizard_payload_for_dataset(d2)

    panel.set_dataset(d1)
    panel._single_tab._cache_wizard_analysis(
        recommendation_1,
        signature={
            "run_number": int(d1.run_number),
            "model": panel._single_tab._composite_model.to_dict(),
        },
        log_text="run 101",
    )
    panel.set_dataset(d2)
    panel._single_tab._cache_wizard_analysis(
        recommendation_2,
        signature={
            "run_number": int(d2.run_number),
            "model": panel._single_tab._composite_model.to_dict(),
        },
        log_text="run 102",
    )
    panel.set_datasets([d1, d2])

    captured: dict[str, object] = {}

    class _FakeGlobalWizard:
        def __init__(self, _parent) -> None:
            self.apply_assessment_requested = SimpleNamespace(connect=lambda _cb: None)
            self.analysis_cached = SimpleNamespace(connect=lambda _cb: None)
            self.single_fit_recommendations_generated = SimpleNamespace(connect=lambda _cb: None)
            self.parameter_setup_applied = SimpleNamespace(connect=lambda _cb: None)

        def set_analysis_context(self, datasets_arg, **kwargs) -> None:
            captured["datasets"] = datasets_arg
            captured["single_fit"] = kwargs.get("existing_single_fit_recommendations_by_run")

        def show(self) -> None:
            captured["show"] = True

        def raise_(self) -> None:
            captured["raise"] = True

        def activateWindow(self) -> None:
            captured["activate"] = True

    monkeypatch.setattr(fit_panel_module, "GlobalFitWizardWindow", _FakeGlobalWizard)

    panel._global_tab._open_fit_wizard()

    assert captured["datasets"] == [d1, d2]
    assert captured["show"] is True
    assert set(captured["single_fit"]) == {int(d1.run_number), int(d2.run_number)}
    assert captured["single_fit"][int(d1.run_number)].summary == recommendation_1.summary
    assert captured["single_fit"][int(d2.run_number)].summary == recommendation_2.summary


def test_global_tab_apply_fit_wizard_assessment_uses_assignment_roles_without_detailed_retests(
    qapp: QApplication,
    dataset: MuonDataset,
) -> None:
    tab = GlobalFitTab()
    dataset_102 = MuonDataset(dataset.time, dataset.asymmetry, dataset.error, {"run_number": 102})
    tab.set_datasets([dataset, dataset_102])

    recommendation = _global_wizard_recommendation_for_dataset(dataset)
    assessment = recommendation.recommended_assessment
    assert assessment is not None
    assessment = replace(assessment, parameter_recommendations=())
    recommendation = replace(recommendation, assessments=(assessment,))

    tab._apply_fit_wizard_assessment(assessment, recommendation)

    lambda_row = 1
    lambda_combo = tab._param_table.cellWidget(lambda_row, 2)
    assert isinstance(lambda_combo, QComboBox)
    assert lambda_combo.currentText() == "Local"


def test_global_tab_get_state_persists_active_window_recommendation_without_cached_signature(
    qapp: QApplication,
    dataset: MuonDataset,
) -> None:
    tab = GlobalFitTab()
    dataset_102 = MuonDataset(dataset.time, dataset.asymmetry, dataset.error, {"run_number": 102})
    tab.set_datasets([dataset, dataset_102])

    recommendation = _global_wizard_recommendation_for_dataset(dataset)
    tab._fit_wizard_window = SimpleNamespace(
        current_recommendation=lambda: recommendation,
        current_log_text=lambda: "window log",
    )
    tab._cached_wizard_signature = None
    tab._cached_wizard_recommendation = None

    saved = tab.get_state()

    assert "wizard_state" in saved
    assert saved["wizard_state"]["log_text"] == "window log"
    assert saved["wizard_state"]["signature"]["run_numbers"] == [int(dataset.run_number), 102]
    assert tab._cached_wizard_recommendation is not None
    assert tab._cached_wizard_signature is not None


def test_bounded_phase_seed_padding_caps_large_signals() -> None:
    """Phase-seed FFT padding is capped so huge histograms stay cheap."""
    from asymmetry.gui.panels.fit_panel import (
        _MAX_PHASE_SEED_FFT_POINTS,
        _bounded_phase_seed_padding,
    )

    # Small signals keep the full desired padding.
    assert _bounded_phase_seed_padding(1024, desired=8) == 8
    # Very large signals are capped to padding 1 (zero-padding only
    # interpolates, so this does not alias the phase seed).
    huge = _MAX_PHASE_SEED_FFT_POINTS * 4
    assert _bounded_phase_seed_padding(huge, desired=8) == 1
    # For a signal within the bound, padding never pushes the padded length
    # past the cap.
    n = _MAX_PHASE_SEED_FFT_POINTS // 2
    assert _bounded_phase_seed_padding(n, desired=8) * n <= _MAX_PHASE_SEED_FFT_POINTS
    assert _bounded_phase_seed_padding(0) == 1


def test_many_parameter_model_keeps_every_row_reachable(qapp: QApplication) -> None:
    """A 13-parameter model must expose every parameter row for editing.

    Regression: the single-fit Parameters table used to collapse to a handful
    of rows with no scrollbar, so the lower parameters of the CdS three-line
    model (A_5/frequency_5/phase_5/Lambda_6/A_bg) were unreachable — they could
    not be seen, seeded, or fixed.
    """
    tab = SingleFitTab()
    cds_model = CompositeModel.from_expression(
        "Oscillatory * Exponential + Oscillatory * Exponential "
        "+ Oscillatory * Exponential + Constant"
    )
    tab._set_composite_model(cds_model)

    # Every model parameter has its own row.
    assert tab._param_table.rowCount() == len(cds_model.param_names) == 13

    # Each row carries an editable Value cell, including the very last one.
    row_by_name: dict[str, int] = {}
    for row in range(tab._param_table.rowCount()):
        name_item = tab._param_table.item(row, 0)
        assert name_item is not None
        row_by_name[str(name_item.data(Qt.ItemDataRole.UserRole))] = row
        value_item = tab._param_table.item(row, 1)
        assert value_item is not None
        assert value_item.flags() & Qt.ItemFlag.ItemIsEditable

    assert set(row_by_name) == set(cds_model.param_names)

    # The last parameter (A_bg) is reachable and round-trips an edited value.
    last_row = row_by_name["A_bg"]
    assert last_row == tab._param_table.rowCount() - 1
    tab._param_table.item(last_row, 1).setText("3.5")
    assert float(tab._param_table.item(last_row, 1).text()) == pytest.approx(3.5)

    # The table is configured to scroll rather than clip: it grows with the
    # dock and shows a scrollbar on demand instead of suppressing it.
    assert (
        tab._param_table.verticalScrollBarPolicy()
        != Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    )
    assert (
        tab._param_table.sizePolicy().verticalPolicy()
        == QSizePolicy.Policy.Expanding
    )
