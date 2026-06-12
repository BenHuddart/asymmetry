"""GUI behaviour around user functions: placeholder degrade in the fit tabs.

The named-placeholder contract (W1): a saved model referencing a user
component that is not registered must open with its original expression,
never be silently replaced, and refuse to fit with a message naming the
missing components.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.gui.panels.fit_panel import GlobalFitTab, SingleFitTab


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _state_with_missing_component() -> dict:
    return {
        "composite_model": {
            "component_names": ["UserGoneDecay", "Constant"],
            "operators": ["+"],
            "open_parentheses": [0, 0],
            "close_parentheses": [0, 0],
            "fraction_groups": [],
        },
        "parameters": [],
    }


def test_single_fit_restore_preserves_model_with_missing_user_component(qapp):
    tab = SingleFitTab()
    tab.restore_state(_state_with_missing_component())

    model = tab._composite_model
    assert model.component_names == ["UserGoneDecay", "Constant"]
    assert model.missing_component_names == ("UserGoneDecay",)
    assert "UserGoneDecay" in tab._result_label.text()
    # Re-saving emits the original names — nothing silently dropped.
    assert tab.get_state()["composite_model"]["component_names"] == [
        "UserGoneDecay",
        "Constant",
    ]


def test_single_fit_blocks_fitting_with_missing_user_component(qapp):
    tab = SingleFitTab()
    tab.restore_state(_state_with_missing_component())
    t = np.linspace(0.0, 4.0, 50)
    tab._current_dataset = MuonDataset(
        time=t,
        asymmetry=0.2 * np.exp(-0.4 * t),
        error=np.full_like(t, 0.01),
        metadata={"run_number": 1},
    )

    tab._run_fit()

    text = tab._result_label.text()
    assert "missing user function" in text
    assert "UserGoneDecay" in text


def test_single_fit_restore_still_defaults_on_malformed_model(qapp):
    tab = SingleFitTab()
    tab.restore_state({"composite_model": {"component_names": "not-a-list"}, "parameters": []})
    assert tab._composite_model.component_names == ["Exponential", "Constant"]


def test_global_fit_restore_and_fit_block_with_missing_user_component(qapp):
    tab = GlobalFitTab()
    tab.restore_state(_state_with_missing_component())

    model = tab._composite_model
    assert model.component_names == ["UserGoneDecay", "Constant"]
    assert model.missing_component_names == ("UserGoneDecay",)

    tab._run_global_fit()
    text = tab._result_text.toPlainText()
    assert "missing user function" in text
    assert "UserGoneDecay" in text
