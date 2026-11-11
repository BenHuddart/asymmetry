"""Response context survives authoring, candidate injection, and curve sampling."""

import numpy as np
import pytest

from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.time_response import TimeResponse
from tests.core.test_time_response_consumers import causal_dataset


def test_model_fraction_copy_preserves_response():
    model = CompositeModel(
        ["Exponential", "Constant"], time_response=TimeResponse("double_delta", 0.324)
    )
    assert model.with_default_fraction_groups().time_response == model.time_response


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


def test_peak_seeded_extra_receives_current_response(monkeypatch):
    from dataclasses import replace

    import asymmetry.core.fitting.fit_wizard as wizard

    data = causal_dataset()
    model = CompositeModel(
        ["Oscillatory", "Exponential", "Constant"],
        operators=["*", "+"],
        time_response=TimeResponse("double_delta", 0.324),
    )
    original_families = wizard.build_wizard_families

    def families(fingerprint, current_model=None, **kwargs):
        return tuple(
            replace(f, stage2_members=())
            for f in original_families(fingerprint, current_model, **kwargs)
            if f.key == "oscillatory"
        )

    monkeypatch.setattr(wizard, "build_wizard_families", families)
    monkeypatch.setattr(
        wizard,
        "_decide_family_promotions",
        lambda fs, rs, *args, **kwargs: [(f, r, True, "forced promotion") for f, r in zip(fs, rs)],
    )
    extra = wizard.CandidateTemplate(
        key="extra_candidate",
        title="Extra",
        category="Oscillatory",
        rationale="",
        model=CompositeModel(["Gaussian", "Constant"]),
    )
    monkeypatch.setattr(
        wizard, "build_oscillatory_multiplet_templates", lambda *args, **kwargs: (extra,)
    )
    seen = []
    original_run = wizard._run_template_assessments

    def run(tasks, **kwargs):
        seen.extend(task.template for task in tasks)
        return original_run(tasks, **kwargs)

    monkeypatch.setattr(wizard, "_run_template_assessments", run)
    wizard.build_fit_wizard_recommendation(
        data, current_model=model, max_workers=1, refine_top_candidates=0
    )
    extras = [t for t in seen if t.key == "extra_candidate"]
    assert extras
    assert all(t.model.time_response == model.time_response for t in extras)
