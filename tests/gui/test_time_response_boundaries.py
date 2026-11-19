"""Response context survives authoring, candidate injection, and curve sampling."""

import numpy as np
import pytest

from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.time_response import TimeResponse
from tests.core.test_time_response_consumers import causal_dataset


@pytest.mark.gui
def test_supported_centres_allow_preview_with_earlier_literal_window_edge(qapp):
    from asymmetry.gui.panels.fit.single_tab import SingleFitTab

    tab = SingleFitTab()
    try:
        data = causal_dataset()
        data.metadata["fit_range"] = [-0.3, 4.0]
        tab.set_dataset(data)
        model = CompositeModel(
            ["Exponential", "Constant"], time_response=TimeResponse("double_delta", 0.324)
        )
        tab._set_composite_model(model)
        previews = []
        tab.preview_requested.connect(lambda *args: previews.append(args))
        tab._on_preview()
        assert previews
        time = previews[0][1][0]
        assert time.min() >= -0.162
    finally:
        tab.close()


@pytest.mark.gui
def test_response_change_invalidates_last_fit_diagnostic(qapp):
    from asymmetry.core.data.dataset import Histogram, Run
    from asymmetry.gui.panels.fit.single_tab import SingleFitTab

    tab = SingleFitTab()
    try:
        data = causal_dataset()
        tab.set_dataset(data)
        model = CompositeModel(
            ["Exponential", "Constant"], time_response=TimeResponse("double_delta", 0.324)
        )
        tab._set_composite_model(model)
        tab._run_fit()
        assert tab.wait_for_fit()
        assert tab._last_fit_result.success
        # The diagnostic eligibility check only needs an actual histogram-bearing run.
        data.run = Run(run_number=1, histograms=[Histogram(np.ones(120), 0.04)])
        assert tab._last_fit_snapshot is not None
        tab._set_composite_model(model.with_time_response(TimeResponse("double_delta", 0.4)))
        assert not tab._can_run_pull_diagnostic()
        assert tab._last_fit_result is None
        assert tab._last_fit_snapshot is None
    finally:
        tab.shutdown_workers()
        tab.close()


@pytest.mark.gui
def test_grouped_count_builder_does_not_offer_ordinary_response(qapp, monkeypatch):
    from asymmetry.gui.panels.fit.global_tab import GlobalFitTab
    from asymmetry.gui.panels.fit_function_builder import FitFunctionBuilderDialog

    tab = GlobalFitTab(member_kind="groups")
    opened = []

    def inspect(dialog):
        opened.append(dialog)
        return 0

    monkeypatch.setattr(FitFunctionBuilderDialog, "exec", inspect)
    try:
        tab._edit_function()
        assert opened
        editor = opened[0]._response_editor
        assert editor.isHidden() or not editor.isEnabled()
    finally:
        for dialog in opened:
            dialog.close()
        tab.close()


@pytest.mark.gui
def test_response_change_in_text_mode_keeps_invalid_text_invalid(qapp):
    from PySide6.QtWidgets import QDialogButtonBox

    from asymmetry.gui.panels.fit_function_builder import FitFunctionBuilderDialog

    dialog = FitFunctionBuilderDialog()
    try:
        dialog._toggle_text_mode()
        dialog._text_edit.setPlainText("Exponential +")
        ok = dialog._buttons.button(QDialogButtonBox.StandardButton.Ok)
        dialog._response_editor.mode.setCurrentIndex(1)
        assert not ok.isEnabled()
    finally:
        dialog.close()


@pytest.mark.gui
def test_unchanged_authored_separation_is_not_rounded_by_builder(qapp):
    from asymmetry.gui.panels.fit_function_builder import FitFunctionBuilderDialog

    model = CompositeModel(
        ["Exponential"], time_response=TimeResponse("double_delta", 0.324123456789)
    )
    dialog = FitFunctionBuilderDialog(initial_model=model)
    try:
        dialog._on_accept()
        assert dialog.get_composite_model().time_response == model.time_response
    finally:
        dialog.close()


@pytest.mark.gui
def test_missing_model_legacy_cache_cannot_override_active_response(qapp):
    from asymmetry.gui.panels.fit.single_tab import SingleFitTab

    tab = SingleFitTab()
    try:
        data = causal_dataset()
        tab.set_dataset(data)
        off = tab._wizard_context_signature(data)
        legacy = {k: v for k, v in off.items() if k != "model"}
        assert tab._wizard_base_signature_matches(legacy, off)
        tab._set_composite_model(
            CompositeModel(
                ["Exponential", "Constant"], time_response=TimeResponse("double_delta", 0.324)
            )
        )
        active = tab._wizard_context_signature(data)
        assert not tab._wizard_base_signature_matches(legacy, active)
    finally:
        tab.close()


@pytest.mark.gui
def test_stale_single_wizard_apply_cannot_reset_response(qapp):
    from asymmetry.gui.panels.fit.single_tab import SingleFitTab
    from tests.gui.test_wizard_result_caching import _single_wizard_payload

    data = causal_dataset()
    assessment, recommendation = _single_wizard_payload(data)
    tab = SingleFitTab()
    try:
        tab.set_dataset(data)
        active = CompositeModel(
            ["Exponential", "Constant"], time_response=TimeResponse("double_delta", 0.324)
        )
        tab._set_composite_model(active)
        tab._apply_fit_wizard_assessment(assessment, recommendation)
        assert tab._composite_model.time_response == active.time_response
    finally:
        tab.close()


@pytest.mark.gui
def test_stale_global_wizard_apply_cannot_reset_response(qapp):
    from asymmetry.gui.panels.fit.global_tab import GlobalFitTab
    from tests.gui.test_wizard_result_caching import _global_wizard_recommendation_for_dataset

    data = causal_dataset(101)
    recommendation = _global_wizard_recommendation_for_dataset(data)
    tab = GlobalFitTab(member_kind="runs")
    try:
        tab.set_datasets([data, causal_dataset(102)])
        active = CompositeModel(
            ["Exponential", "Constant"], time_response=TimeResponse("double_delta", 0.324)
        )
        tab._set_composite_model(active)
        tab._apply_fit_wizard_assessment(recommendation.assessments[0], recommendation)
        assert tab._composite_model.time_response == active.time_response
    finally:
        tab.close()


@pytest.mark.gui
def test_stale_phase_apply_is_not_forwarded_under_changed_response(qapp):
    from asymmetry.gui.panels.fit.global_tab import GlobalFitTab
    from tests.gui.test_wizard_result_caching import _global_wizard_recommendation_for_dataset

    data = causal_dataset(101)
    recommendation = _global_wizard_recommendation_for_dataset(data)
    tab = GlobalFitTab(member_kind="runs")
    forwarded = []
    tab.apply_wizard_phases_requested.connect(lambda *args: forwarded.append(args))
    try:
        tab.set_datasets([data, causal_dataset(102)])
        active = CompositeModel(
            ["Exponential", "Constant"], time_response=TimeResponse("double_delta", 0.324)
        )
        tab._set_composite_model(active)
        tab._on_fit_wizard_apply_phases_requested(recommendation, 2)
        assert not forwarded
    finally:
        tab.close()


@pytest.mark.gui
def test_stale_direct_phase_application_keeps_bound_group_and_response(qapp):
    from asymmetry.gui.panels.fit.global_tab import GlobalFitTab
    from tests.gui.test_wizard_result_caching import _global_wizard_recommendation_for_dataset

    data = causal_dataset(101)
    recommendation = _global_wizard_recommendation_for_dataset(data)
    tab = GlobalFitTab(member_kind="runs")
    try:
        tab.set_datasets([data, causal_dataset(102)])
        active = CompositeModel(
            ["Exponential", "Constant"], time_response=TimeResponse("double_delta", 0.324)
        )
        tab._set_composite_model(active)
        tab.apply_wizard_phase_assessments(
            recommendation, [("phase", "Phase", recommendation.assessments[0])]
        )
        assert tab._bound_group_id is None
        assert tab._composite_model.time_response == active.time_response
    finally:
        tab.close()


@pytest.mark.gui
def test_single_preview_displays_model_work_limit(qapp):
    from asymmetry.gui.panels.fit.single_tab import SingleFitTab

    tab = SingleFitTab()
    try:
        tab.set_dataset(causal_dataset())
        tab._set_composite_model(CompositeModel(["DynamicGaussianKT"]))
        for row in range(tab._param_table.rowCount()):
            name = tab._param_table.item(row, 0).data(256)
            if name == "B_L":
                tab._param_table.item(row, 1).setText("1000000")
        tab._on_preview()
        assert tab._results_card.tag_text() == "Error"
        assert "524,288" in tab._results_card.content_html()
        assert "Shorten" in tab._results_card.content_html()
        for row in range(tab._param_table.rowCount()):
            if tab._param_table.item(row, 0).data(256) == "B_L":
                tab._param_table.item(row, 1).setText("100")
        tab._on_preview()
        assert tab._results_card.tag_text() == "Preview"
        assert "work budget" not in tab._results_card.content_html()
    finally:
        tab.close()


@pytest.mark.gui
def test_grouped_tab_preview_emits_curves_for_each_group(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from PySide6.QtCore import Qt

    from asymmetry.core.data.dataset import MuonDataset
    from asymmetry.core.fitting.grouped_time_domain import GroupedTimeDomainGroup
    from asymmetry.gui.panels.fit.global_tab import GlobalFitTab

    dataset = causal_dataset()
    grouped_groups = [
        GroupedTimeDomainGroup(1, "Forward", np.arange(2.0), np.array([120.0, 118.0]), np.ones(2)),
        GroupedTimeDomainGroup(2, "Backward", np.arange(2.0), np.array([80.0, 79.0]), np.ones(2)),
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

    from asymmetry.core.fitting.strong_collision import EvaluationWorkLimitError

    def oversized_preview(**kwargs):
        raise EvaluationWorkLimitError("Required samples exceed the work budget.")

    monkeypatch.setattr(tab, "_build_grouped_preview_curves", oversized_preview)
    emitted.clear()
    tab._on_preview_requested()
    assert not emitted
    assert tab._results_card.tag_text() == "Error"
    assert "work budget" in tab._results_card.content_html()
