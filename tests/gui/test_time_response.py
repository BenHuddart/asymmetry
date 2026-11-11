"""Ordinary-fit authoring, saved setup, wizard identity and launch snapshots."""

import numpy as np
import pytest

from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.time_response import TimeResponse
from asymmetry.gui.panels.fit.global_tab import GlobalFitTab
from asymmetry.gui.panels.fit.single_tab import SingleFitTab
from asymmetry.gui.panels.fit_function_builder import FitFunctionBuilderDialog
from tests.core.test_time_response_consumers import causal_dataset

pytestmark = pytest.mark.gui


def test_shared_builder_authors_response_without_mutating_open_model(qapp):
    model = CompositeModel(["Exponential", "Constant"])
    dialog = FitFunctionBuilderDialog(initial_model=model)
    editor = dialog._response_editor
    assert editor.value().mode == "off"
    editor.mode.setCurrentIndex(1)
    editor.separation.setValue(0.4)
    dialog._on_accept()
    active = dialog.get_composite_model()
    assert active.time_response == TimeResponse("double_delta", 0.4)
    assert model.time_response.mode == "off"
    assert active.param_names == model.param_names
    reopened = FitFunctionBuilderDialog(initial_model=active)
    reopened._on_accept()
    assert reopened.get_composite_model().to_dict() == active.to_dict()
    dialog.close()
    reopened.close()


@pytest.mark.parametrize("kind", ["single", "batch"])
def test_actual_tab_state_and_wizard_identity_restore_response(qapp, kind):
    tab = SingleFitTab() if kind == "single" else GlobalFitTab(member_kind="runs")
    restored = SingleFitTab() if kind == "single" else GlobalFitTab(member_kind="runs")
    model = CompositeModel(["Exponential", "Constant"]).with_time_response(
        TimeResponse("double_delta", 0.324)
    )
    data = causal_dataset()
    if kind == "single":
        tab.set_dataset(data)
        off_signature = tab._wizard_context_signature(data)
    else:
        tab.set_datasets([data, causal_dataset(2)])
    tab._set_composite_model(model)
    if kind == "single":
        assert tab._wizard_context_signature(data) != off_signature
    restored.restore_state(tab.get_state())
    assert restored._composite_model.time_response == model.time_response
    parameters = model.param_defaults
    np.testing.assert_array_equal(
        restored._composite_model.function(data.time, **parameters),
        model.function(data.time, **parameters),
    )
    tab.close()
    restored.close()


def test_live_single_worker_fits_restored_response_and_ignores_changed_launch(qapp):
    import threading
    from types import SimpleNamespace

    from asymmetry.core.fitting.engine import FitEngine

    tab = SingleFitTab()
    release = threading.Event()
    try:
        tab.set_dataset(causal_dataset())
        model = CompositeModel(
            ["Exponential", "Constant"], time_response=TimeResponse("double_delta", 0.324)
        )
        tab._set_composite_model(model)
        tab._run_fit()
        assert tab.wait_for_fit()
        assert tab._last_fit_result.success
        assert tab._last_fit_result.parameters["Lambda"].value == pytest.approx(0.7, abs=1e-4)
        recorded = tab._last_fit_result

        def delayed(data, source, parameters, **kwargs):
            assert release.wait(10)
            return FitEngine().fit(data, source, parameters, **kwargs)

        tab._fit_engine = SimpleNamespace(fit=delayed)
        tab._run_fit()
        replacement = model.with_time_response(TimeResponse("double_delta", 0.4))
        tab._set_composite_model(replacement)
        release.set()
        assert tab.wait_for_fit()
        assert tab._last_fit_result is recorded
        assert model.time_response.separation_us == 0.324
    finally:
        release.set()
        tab.shutdown_workers()
        tab.close()


@pytest.mark.parametrize("tab_class", [SingleFitTab, GlobalFitTab])
def test_invalid_authored_response_is_not_silently_restored_off(qapp, tab_class):
    tab = tab_class()
    try:
        state = tab.get_state()
        state["composite_model"]["time_response"] = {"mode": "double_delta", "separation_us": -1}
        with pytest.raises(ValueError, match="finite nonnegative"):
            tab.restore_state(state)
    finally:
        tab.close()
