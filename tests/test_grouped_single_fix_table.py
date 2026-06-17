"""The single grouped (individual-groups) fit uses a Fix-tickbox physics table.

All detector groups of one dataset share the fit-function, so its physics
parameters take the single-fit-style Fix checkbox (no Global/Local/Fixed combo);
per-group quantities live in the nuisance block. Verifies the physics table is a
FitParameterTable and that the parsed roles are all global/fixed (never local).
"""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QCheckBox

from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.gui.panels.fit_panel import FitParameterTable, GlobalFitTab


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def _grouped_single_tab(qapp) -> GlobalFitTab:
    tab = GlobalFitTab(member_kind="groups", grouped_single=True)
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    # Stub the grouped context so the physics table can be rebuilt without a
    # live dataset/grouping.
    tab._composite_model = model
    tab._grouped_fit_model = lambda: model  # type: ignore[method-assign]
    tab._grouped_mode_context = lambda: ([], [], "")  # type: ignore[method-assign]
    return tab


def test_single_grouped_physics_table_is_fix_tickbox(qapp):
    tab = _grouped_single_tab(qapp)
    assert isinstance(tab._group_model_table, FitParameterTable)
    tab._rebuild_grouped_model_table({})

    table = tab._group_model_table
    # Non-amplitude physics params get a row with a Fix checkbox (no Type combo).
    names = [
        table.item(r, FitParameterTable.COL_NAME).data(Qt.ItemDataRole.UserRole)
        for r in range(table.rowCount())
    ]
    assert names  # at least the relaxation params
    assert all(
        isinstance(table.cellWidget(r, FitParameterTable.COL_FIX).findChild(QCheckBox), QCheckBox)
        for r in range(table.rowCount())
    )
    # Link/Tie/Batch columns hidden (the grouped engine has no tie support).
    assert table.isColumnHidden(FitParameterTable.COL_LINK)
    assert table.isColumnHidden(FitParameterTable.COL_TIE)


def test_single_grouped_parse_roles_are_global_or_fixed(qapp):
    tab = _grouped_single_tab(qapp)
    tab._rebuild_grouped_model_table({})
    table = tab._group_model_table

    # Tick Fix on the first physics row; leave the rest free.
    first = table.cellWidget(0, FitParameterTable.COL_FIX).findChild(QCheckBox)
    first.setChecked(True)
    first_name = table.item(0, FitParameterTable.COL_NAME).data(Qt.ItemDataRole.UserRole)

    config = tab._parse_grouped_parameter_configuration()
    roles = config["physics_roles"]
    assert roles  # physics params classified
    assert "local" not in roles.values()  # single fit → never per-run local
    assert roles[first_name] == "fixed"
    assert all(role in ("global", "fixed") for role in roles.values())

    # get_grouped_state mirrors the parsed physics roles.
    state = tab.get_grouped_state()
    assert state["param_roles"][first_name] == "fixed"
    assert "local" not in state["param_roles"].values()


def test_single_grouped_physics_state_round_trips(qapp):
    # The Fix-tickbox physics table serialises in the shared {value, type, bounds}
    # shape so project save/restore preserves value, Fix and bounds.
    tab = _grouped_single_tab(qapp)
    tab._rebuild_grouped_model_table({})
    table = tab._group_model_table
    free_name = table.item(0, FitParameterTable.COL_NAME).data(Qt.ItemDataRole.UserRole)
    table.item(0, FitParameterTable.COL_VALUE).setText("0.42")
    table.item(0, FitParameterTable.COL_MIN).setText("0")
    table.item(0, FitParameterTable.COL_MAX).setText("5")
    table.cellWidget(0, FitParameterTable.COL_FIX).findChild(QCheckBox).setChecked(True)

    saved = tab._table_state_for(table)

    other = _grouped_single_tab(qapp)
    other._rebuild_grouped_model_table({})
    other._restore_table_state(other._group_model_table, saved)

    restored = {p.name: p for p in other._group_model_table.read_parameter_set()}
    assert restored[free_name].value == pytest.approx(0.42)
    assert restored[free_name].min == pytest.approx(0.0)
    assert restored[free_name].max == pytest.approx(5.0)
    assert restored[free_name].fixed is True
