"""GUI: affine-tie editing, persistence, and auxiliary-parameter preservation.

Covers the fit-panel surface of the affine-ties feature: the per-row Tie editor,
the dialog, the get_state/restore_state round-trip of the ``tie`` field, and the
preservation of auxiliary (non-model) parameters an API-authored tie references
(so a GUI save/re-fit does not silently drop them). See docs/porting/link-groups/.
"""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.parameters import AffineTie
from asymmetry.gui.panels.fit_panel import (
    _SINGLE_PARAM_TIE_COLUMN,
    AffineTieDialog,
    SingleFitTab,
    _set_tie_button_value,
    _tie_button_value,
)

_TRIPLET = (
    "Oscillatory * Exponential + Oscillatory * Exponential + Oscillatory * Exponential + Constant"
)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _row_of(tab: SingleFitTab, param_name: str) -> int:
    for i in range(tab._param_table.rowCount()):
        item = tab._param_table.item(i, 0)
        if item is not None and item.data(0x0100) == param_name:  # Qt.UserRole
            return i
    raise AssertionError(f"{param_name} not in table")


def _tie_button(tab: SingleFitTab, param_name: str):
    return tab._param_table.cellWidget(_row_of(tab, param_name), _SINGLE_PARAM_TIE_COLUMN)


def test_tie_column_present(qapp: QApplication) -> None:
    tab = SingleFitTab()
    tab._set_composite_model(CompositeModel.from_expression(_TRIPLET))
    assert tab._param_table.columnCount() == 8
    header = tab._param_table.horizontalHeaderItem(_SINGLE_PARAM_TIE_COLUMN)
    assert header is not None and header.text() == "Tie"
    # Every row carries a tie button, untied by default.
    assert _tie_button_value(_tie_button(tab, "frequency_3")) is None


def test_tie_round_trips_through_state(qapp: QApplication) -> None:
    tab = SingleFitTab()
    tab._set_composite_model(CompositeModel.from_expression(_TRIPLET))
    tie = AffineTie(main="frequency_1", const=-0.12)
    _set_tie_button_value(_tie_button(tab, "frequency_3"), tie)

    # The fit-facing ParameterSet carries the tie.
    ps = tab._parameter_set_from_table()
    assert ps["frequency_3"].tie == tie
    assert "frequency_3" in ps.tie_followers()

    # Save → restore into a fresh tab recovers the tie.
    state = tab.get_state()
    restored = SingleFitTab()
    restored.restore_state(state)
    assert _tie_button_value(_tie_button(restored, "frequency_3")) == tie


def test_auxiliary_parameter_is_preserved(qapp: QApplication) -> None:
    """An API-authored tie referencing a non-model `delta` survives a GUI round-trip."""
    model = CompositeModel.from_expression(_TRIPLET)
    state = {
        "model_name": "Composite",
        "composite_model": model.to_dict(),
        "parameters": [
            {"name": name, "value": 1.0, "fixed": False, "min": "-inf", "max": "inf"}
            for name in model.param_names
        ]
        + [
            {
                "name": "frequency_3",
                "value": 1.27,
                "tie": AffineTie(main="frequency_1", offset="delta", offset_scale=-1.0).to_dict(),
            },
            # Auxiliary, non-model parameter the model never consumes:
            {"name": "delta", "value": 0.12, "fixed": False, "min": "0.0", "max": "inf"},
        ],
    }
    # The duplicate frequency_3 entry (with the tie) overrides the plain one.
    state["parameters"] = [p for p in state["parameters"] if p["name"] != "frequency_3"] + [
        p for p in state["parameters"] if p["name"] == "frequency_3"
    ]

    tab = SingleFitTab()
    tab.restore_state(state)

    # The tie on the model-param row is restored…
    assert _tie_button_value(_tie_button(tab, "frequency_3")) == AffineTie(
        main="frequency_1", offset="delta", offset_scale=-1.0
    )
    # …the auxiliary `delta` (no table row) is preserved for re-save…
    assert tab.get_state()["parameters"][-1]["name"] == "delta" or any(
        p["name"] == "delta" for p in tab.get_state()["parameters"]
    )
    # …and a re-fit ParameterSet includes it, so the tie resolves.
    ps = tab._parameter_set_from_table()
    assert "delta" in ps
    assert ps["frequency_3"].tie is not None
    assert ps["frequency_3"].tie.offset == "delta"


def test_affine_tie_dialog_builds_and_clears(qapp: QApplication) -> None:
    candidates = ["frequency_1", "frequency_5"]
    dialog = AffineTieDialog("frequency_3", candidates, None, None)
    # Disabled by default → no tie.
    assert dialog.tie() is None
    # Enable and configure an equal-spacing-style tie f3 = 2*f1 - f5.
    dialog._enable.setChecked(True)
    dialog._main.setCurrentText("frequency_1")
    dialog._scale.setValue(2.0)
    dialog._offset.setCurrentIndex(dialog._offset.findData("frequency_5"))
    dialog._offset_scale.setValue(-1.0)
    tie = dialog.tie()
    assert tie == AffineTie(main="frequency_1", scale=2.0, offset="frequency_5", offset_scale=-1.0)
    # A pre-seeded tie is reflected back.
    seeded = AffineTieDialog("frequency_3", candidates, tie, None)
    assert seeded.tie() == tie
