"""Tests for cross-group fit dialog UI parity with model-fit labels."""

from __future__ import annotations

import threading
import time

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from asymmetry.core.fitting.parameter_models import (
    CrossGroupFitResult,
    ParameterCompositeModel,
    ParameterGroupData,
)
from asymmetry.gui.panels.cross_group_fit_dialog import CrossGroupFitDialog


def _groups() -> list[ParameterGroupData]:
    x = np.array([100.0, 200.0, 300.0], dtype=float)
    g0 = ParameterGroupData(
        group_id="g0",
        group_name="G0",
        x=x,
        y=np.array([0.1, 0.2, 0.3], dtype=float),
        yerr=np.array([0.01, 0.01, 0.01], dtype=float),
        group_variable_value=0.0,
    )
    g1 = ParameterGroupData(
        group_id="g1",
        group_name="G1",
        x=x,
        y=np.array([0.12, 0.22, 0.32], dtype=float),
        yerr=np.array([0.01, 0.01, 0.01], dtype=float),
        group_variable_value=1.0,
    )
    return [g0, g1]


def test_cross_group_dialog_parameter_labels_include_units() -> None:
    app = QApplication.instance() or QApplication([])
    dlg = CrossGroupFitDialog(
        parameter_name="Lambda",
        x_key="field",
        groups=_groups(),
        parent=None,
    )

    # Default model is Linear -> params m and b.
    labels = [dlg._param_table.item(row, 0).text() for row in range(dlg._param_table.rowCount())]
    assert any("[us^-1 / G]" in label for label in labels)
    assert any("[us^-1]" in label for label in labels)

    # Canonical parameter names used by fitting are preserved in UserRole.
    canonical = [
        dlg._param_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        for row in range(dlg._param_table.rowCount())
    ]
    assert "m" in canonical
    assert "b" in canonical

    # Keep app referenced so Qt objects are valid through assertions.
    assert app is not None


def test_use_fit_without_result_shows_feedback(monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    dlg = CrossGroupFitDialog(
        parameter_name="Lambda",
        x_key="field",
        groups=_groups(),
        parent=None,
    )

    called = {"info": 0}

    def _fake_info(*_args, **_kwargs):
        called["info"] += 1
        return None

    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "information", _fake_info)

    dlg._on_use_fit()

    assert called["info"] == 1
    assert dlg.result() != dlg.DialogCode.Accepted
    assert app is not None


def test_use_fit_failed_result_requires_confirmation(monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    dlg = CrossGroupFitDialog(
        parameter_name="Lambda",
        x_key="field",
        groups=_groups(),
        parent=None,
    )
    dlg._result = CrossGroupFitResult(
        success=False,
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        message="Fit failed",
    )

    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.No,
    )
    dlg._on_use_fit()
    assert dlg.result() != dlg.DialogCode.Accepted

    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )
    dlg._on_use_fit()
    assert dlg.result() == dlg.DialogCode.Accepted
    assert app is not None


def test_cross_group_dialog_redfield_m_is_unitless() -> None:
    app = QApplication.instance() or QApplication([])
    dlg = CrossGroupFitDialog(
        parameter_name="Lambda",
        x_key="field",
        groups=_groups(),
        parent=None,
    )

    dlg._model = ParameterCompositeModel(["Redfield"], [])
    dlg._rebuild_param_table()

    labels = [dlg._param_table.item(row, 0).text() for row in range(dlg._param_table.rowCount())]
    assert "m" in labels
    assert not any(label.startswith("m [") for label in labels)
    assert dlg._fit.ranges[0].parameters["m"].value == 2.0
    assert app is not None


def test_cross_group_dialog_sc_shape_factor_a_defaults_to_fixed() -> None:
    app = QApplication.instance() or QApplication([])
    dlg = CrossGroupFitDialog(
        parameter_name="Lambda",
        x_key="temperature",
        groups=_groups(),
        parent=None,
    )

    dlg._model = ParameterCompositeModel(["SC_PWaveAxial"], [])
    dlg._rebuild_param_table()

    row_by_name = {
        dlg._param_table.item(row, 0).data(Qt.ItemDataRole.UserRole): row
        for row in range(dlg._param_table.rowCount())
    }
    shape_row = row_by_name["shape_factor_a"]
    type_combo = dlg._param_table.cellWidget(shape_row, 4)

    assert type_combo is not None
    assert type_combo.currentText() == "Fixed"
    assert dlg._fit.ranges[0].parameters["shape_factor_a"].fixed is True
    assert dlg._fit.ranges[0].parameters["shape_factor_a"].value == 0.0
    assert app is not None


def test_cross_group_dialog_shows_inherited_source_in_banner() -> None:
    app = QApplication.instance() or QApplication([])
    dlg = CrossGroupFitDialog(
        parameter_name="Lambda",
        x_key="field",
        groups=_groups(),
        existing_config={
            "source_group_name": "G0",
            "source_reduced_chi_squared": 0.8123,
        },
        parent=None,
    )

    banner = dlg.layout().itemAt(0).widget() if dlg.layout() is not None else None
    assert banner is not None
    text = banner.text()
    assert "Inherited from" in text
    assert "G0" in text
    assert "chi2_r=" in text
    assert app is not None


def test_cross_group_run_fit_sets_in_progress_state(monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    dlg = CrossGroupFitDialog(
        parameter_name="Lambda",
        x_key="field",
        groups=_groups(),
        parent=None,
    )

    gate = threading.Event()

    def _fake_global_fit(**_kwargs):
        gate.wait(timeout=1.0)
        return CrossGroupFitResult(success=True, chi_squared=1.0, reduced_chi_squared=1.0)

    monkeypatch.setattr(
        "asymmetry.gui.panels.cross_group_fit_dialog.global_fit_parameter_model",
        _fake_global_fit,
    )

    try:
        dlg._run_fit(0)

        assert dlg._fit_in_progress is True
        assert "in progress" in dlg._fit_progress_label.text().lower()
    finally:
        gate.set()
        deadline = time.time() + 2.0
        while dlg._fit_in_progress and time.time() < deadline:
            app.processEvents()
            time.sleep(0.01)

    assert dlg._fit_in_progress is False
    assert app is not None
