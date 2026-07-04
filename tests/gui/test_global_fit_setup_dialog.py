"""Tests for the explicit global-fit setup dialog (Phase 4).

The dialog is built from a plain-data :class:`GlobalFitSetupData` snapshot, so
these tests exercise it directly without a fully wired FitParametersPanel.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.gui]

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QDialogButtonBox,
    QDoubleSpinBox,
)

from asymmetry.gui.panels.global_fit_setup_dialog import (  # noqa: E402
    GlobalFitSetupData,
    GlobalFitSetupDialog,
    GlobalFitSetupSeries,
)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def _data() -> GlobalFitSetupData:
    series = (
        GlobalFitSetupSeries(
            group_id="g0",
            group_name="G0",
            member_count=5,
            param_names=("Lambda", "Amplitude"),
        ),
        GlobalFitSetupSeries(
            group_id="g1",
            group_name="G1",
            member_count=4,
            param_names=("Lambda", "Beta"),  # no Amplitude
        ),
    )
    x_options = (
        ("B (G)", "field"),
        ("T (K)", "temperature"),
        ("Run", "run"),
        ("Current (A)", "custom:cur"),
    )
    gv_options = (
        ("Temperature", "temperature"),
        ("Field", "field"),
        ("Run", "run"),
        ("Current (A)", "custom:cur"),
    )

    def default_gv(x_key: str) -> str:
        return {"field": "temperature", "temperature": "field"}.get(x_key, "run")

    # Per-group medians: temperature by group, field by group, custom column.
    gv_values = {
        ("g0", "temperature"): 10.0,
        ("g1", "temperature"): 20.0,
        ("g0", "field"): 100.0,
        ("g1", "field"): 200.0,
        ("g0", "custom:cur"): 1.5,
        ("g1", "custom:cur"): 2.5,
    }

    def gv_value(group_id: str, gv_key: str) -> float:
        return gv_values.get((group_id, gv_key), 0.0)

    def gv_label(gv_key: str) -> str:
        return {
            "temperature": "T (K)",
            "field": "B (G)",
            "run": "Run",
            "custom:cur": "Current (A)",
        }.get(gv_key, gv_key)

    return GlobalFitSetupData(
        series=series,
        x_key_options=x_options,
        group_variable_options=gv_options,
        default_group_variable_key=default_gv,
        group_variable_value=gv_value,
        group_variable_label=gv_label,
    )


def _ok(dlg: GlobalFitSetupDialog):
    return dlg._buttons.button(QDialogButtonBox.StandardButton.Ok)


def test_series_checklist_gates_ok(qapp: QApplication) -> None:
    # No preselection → all checked → OK enabled.
    dlg = GlobalFitSetupDialog(_data())
    assert _ok(dlg).isEnabled()

    # Uncheck one series → only one checked → OK disabled.
    dlg._series_list.item(0).setCheckState(Qt.CheckState.Unchecked)
    assert not _ok(dlg).isEnabled()

    # Re-check → enabled again.
    dlg._series_list.item(0).setCheckState(Qt.CheckState.Checked)
    assert _ok(dlg).isEnabled()


def test_parameter_combo_is_intersection_and_updates_on_uncheck(qapp: QApplication) -> None:
    dlg = GlobalFitSetupDialog(_data())
    # Both series checked → only "Lambda" is common (Amplitude/Beta differ).
    params = [dlg._param_combo.itemText(i) for i in range(dlg._param_combo.count())]
    assert params == ["Lambda"]

    # Uncheck g1 → g0 alone contributes Amplitude + Lambda.
    dlg._series_list.item(1).setCheckState(Qt.CheckState.Unchecked)
    params = sorted(dlg._param_combo.itemText(i) for i in range(dlg._param_combo.count()))
    assert params == ["Amplitude", "Lambda"]


def test_group_variable_default_for_field_is_temperature(qapp: QApplication) -> None:
    dlg = GlobalFitSetupDialog(_data(), preselected_x_key="field")
    assert dlg._current_gv_key() == "temperature"
    assert dlg._gv_label_edit.text() == "T (K)"

    # Table prefilled with per-group temperature medians.
    values = dlg._current_table_values()
    assert values == {"g0": 10.0, "g1": 20.0}


def test_group_variable_custom_column_median_prefill(qapp: QApplication) -> None:
    dlg = GlobalFitSetupDialog(_data(), preselected_x_key="field")
    # Switch the group variable to the custom column.
    idx = dlg._gv_combo.findData("custom:cur")
    dlg._gv_combo.setCurrentIndex(idx)
    assert dlg._gv_label_edit.text() == "Current (A)"
    assert dlg._current_table_values() == {"g0": 1.5, "g1": 2.5}


def test_editable_override_lands_in_result(qapp: QApplication) -> None:
    dlg = GlobalFitSetupDialog(_data(), preselected_x_key="field", preselected_parameter="Lambda")
    # Edit g0's group-variable value directly in the spin box.
    spin = dlg._gv_table.cellWidget(0, 1)
    assert isinstance(spin, QDoubleSpinBox)
    spin.setValue(42.0)

    result = dlg.result_data()
    assert result is not None
    assert result.parameter_name == "Lambda"
    assert result.x_key == "field"
    assert result.x_label == "B (G)"
    assert result.group_variable_key == "temperature"
    assert result.group_variable_label == "T (K)"
    assert result.group_ids == ["g0", "g1"]
    assert result.group_variable_values["g0"] == 42.0
    assert result.group_variable_values["g1"] == 20.0


def test_x_label_prefill_for_custom_column(qapp: QApplication) -> None:
    dlg = GlobalFitSetupDialog(_data(), preselected_x_key="custom:cur")
    # Custom x-axis has no T↔B complement → group variable defaults to run.
    assert dlg._current_gv_key() == "run"
    result = dlg.result_data()
    assert result is not None
    assert result.x_key == "custom:cur"
    assert result.x_label == "Current (A)"


def test_result_none_when_fewer_than_two_series(qapp: QApplication) -> None:
    dlg = GlobalFitSetupDialog(_data())
    dlg._series_list.item(0).setCheckState(Qt.CheckState.Unchecked)
    assert dlg.result_data() is None
