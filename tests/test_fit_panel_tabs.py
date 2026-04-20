"""Focused tests for SingleFitTab and GlobalFitTab logic."""

from __future__ import annotations

import os
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QComboBox

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.global_fit_wizard import (
    GlobalCandidateAssessment,
    GlobalFitWizardRecommendation,
    GlobalParameterRecommendation,
    RunResidualDiagnostic,
)
from asymmetry.core.fitting.fit_wizard import (
    CandidateAssessment,
    CandidateTemplate,
    FitWizardRecommendation,
    SelectionMetric,
    SpectrumFingerprint,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
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


def _wizard_payload_for_dataset(dataset: MuonDataset) -> tuple[CandidateAssessment, FitWizardRecommendation]:
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
            model.evaluate_components(dataset.time, additive_only=True, A_1=0.2, Lambda=0.4, A_bg=0.01)
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


def _global_wizard_recommendation_for_dataset(dataset: MuonDataset) -> GlobalFitWizardRecommendation:
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


def test_fit_panel_forwards_single_tab_preview(
    qapp: QApplication, dataset: MuonDataset
) -> None:
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

    assert "Global Fit Successful" in tab._result_text.toHtml()
    assert set(emitted["res"]) == {101, 102}


def test_global_fit_finished_failure_lists_failed_runs(qapp: QApplication, dataset: MuonDataset) -> None:
    tab = GlobalFitTab()
    tab._current_model = tab._composite_model
    tab._current_global_params = []
    fail = FitResult(success=False, message="x")

    tab._on_fit_finished({101: fail}, ParameterSet())
    assert "Global fit failed" in tab._result_text.toPlainText()


def test_global_fit_error_sets_message(qapp: QApplication) -> None:
    tab = GlobalFitTab()
    tab._fit_btn.setEnabled(False)
    tab._on_fit_error("boom")
    assert tab._fit_btn.isEnabled() is True
    assert "boom" in tab._result_text.toPlainText()


def test_global_fit_parses_type_combo_defaults(qapp: QApplication) -> None:
    tab = GlobalFitTab()
    # First row defaults to Global, subsequent rows Local.
    c0 = tab._param_table.cellWidget(0, 2)
    c1 = tab._param_table.cellWidget(1, 2) if tab._param_table.rowCount() > 1 else None
    assert isinstance(c0, QComboBox)
    assert c0.currentText() == "Global"
    if isinstance(c1, QComboBox):
        assert c1.currentText() == "Local"


def test_single_tab_default_model_includes_background(qapp: QApplication) -> None:
    tab = SingleFitTab()
    assert tab._composite_model.component_names == ["Exponential", "Constant"]
    assert "A_bg" in tab._composite_model.param_names


def test_global_tab_default_model_includes_background(qapp: QApplication) -> None:
    tab = GlobalFitTab()
    assert tab._composite_model.component_names == ["Exponential", "Constant"]
    assert "A_bg" in tab._composite_model.param_names


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


def test_fit_panel_restores_single_fit_state_per_dataset(qapp: QApplication, dataset: MuonDataset) -> None:
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
        signature={"run_number": int(d1.run_number), "model": panel._single_tab._composite_model.to_dict()},
        log_text="run 101 log",
    )
    panel.set_dataset(d2)
    panel._single_tab._cache_wizard_analysis(
        recommendation_2,
        signature={"run_number": int(d2.run_number), "model": panel._single_tab._composite_model.to_dict()},
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

    assert "Global fit" in panel._single_tab._result_label.text()
    assert float(panel._single_tab._param_table.item(0, 1).text()) == pytest.approx(0.11)

    panel.set_dataset(d2)
    assert "Global fit" in panel._single_tab._result_label.text()
    assert float(panel._single_tab._param_table.item(0, 1).text()) == pytest.approx(0.44)

    saved = panel.get_single_state()
    assert "101" in saved.get("states_by_run", {})
    assert "102" in saved.get("states_by_run", {})


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

    copied = panel.share_single_function_state(101, [102, 103])
    assert copied == 2

    panel.set_dataset(d2)
    assert float(panel._single_tab._param_table.item(0, 1).text()) == pytest.approx(0.777)
    assert panel._single_tab._result_label.text() == "No fit performed yet"

    panel.set_dataset(d3)
    assert float(panel._single_tab._param_table.item(0, 1).text()) == pytest.approx(0.777)
    assert panel._single_tab._result_label.text() == "No fit performed yet"


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


def test_global_fit_wizard_button_tracks_dataset_and_block_state(qapp: QApplication, dataset: MuonDataset) -> None:
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
        fitted_curves[run_number] = (np.asarray(ds.time, dtype=float).copy(), np.asarray(curve, dtype=float))
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
        signature={"run_number": int(d1.run_number), "model": panel._single_tab._composite_model.to_dict()},
        log_text="run 101",
    )
    panel.set_dataset(d2)
    panel._single_tab._cache_wizard_analysis(
        recommendation_2,
        signature={"run_number": int(d2.run_number), "model": panel._single_tab._composite_model.to_dict()},
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
