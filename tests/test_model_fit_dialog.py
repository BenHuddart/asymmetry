"""Tests for ModelFitDialog range-parameter labels and bounds normalization."""

from __future__ import annotations

import os
import threading
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QDialog

from asymmetry.core.fitting.parameter_models import ModelFitRange, ParameterCompositeModel, ParameterModelFit
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.gui.panels.model_fit_dialog import (
    ModelFitDialog,
    ParameterModelBuilderDialog,
    _ComponentSelectorButton,
    _SC_COMPONENT_MENU_TITLE,
    _component_pool_for_context,
    _format_model_param_label,
)
from asymmetry.gui.widgets.component_info_dialog import build_component_info_html


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_range_parameter_name_displays_units(qapp: QApplication) -> None:
    x = np.linspace(10.0, 100.0, 10)
    y = np.linspace(0.1, 0.2, 10)
    yerr = np.full_like(x, 0.01)

    model = ParameterCompositeModel(["Redfield"], [])
    params = ParameterSet(
        [
            Parameter("D", 10.0),
            Parameter("nu", 8.0),
            Parameter("m", 2.0),
        ]
    )
    fit = ParameterModelFit(
        parameter_name="Lambda",
        x_key="field",
        ranges=[ModelFitRange(x_min=10.0, x_max=100.0, model=model, parameters=params)],
    )

    dlg = ModelFitDialog(
        parameter_name="Lambda",
        x_key="field",
        x_values=x,
        y_values=y,
        y_errors=yerr,
        existing_fit=fit,
    )

    labels = [dlg._param_table.item(row, 0).text() for row in range(dlg._param_table.rowCount())]
    assert any("D [MHz]" in text for text in labels)
    assert any("nu [MHz]" in text for text in labels)
    assert "m" in labels


def test_component_pool_filters_constant_vs_lambda_bg() -> None:
    lambda_pool = _component_pool_for_context("field", "Lambda")
    sigma_pool = _component_pool_for_context("field", "sigma")

    assert "Lambda_bg" in lambda_pool
    assert "Constant" not in lambda_pool

    assert "Constant" in sigma_pool
    assert "Lambda_bg" not in sigma_pool


def test_format_model_param_label_redfield_m_is_unitless() -> None:
    redfield = ParameterCompositeModel(["Redfield"], [])
    linear = ParameterCompositeModel(["Linear"], [])

    redfield_label = _format_model_param_label(redfield, "m", "field", "Lambda")
    linear_label = _format_model_param_label(linear, "m", "field", "Lambda")

    assert redfield_label == "m"
    assert linear_label == "m [us^-1 / G]"


def test_linear_model_slope_uses_x_and_y_units(qapp: QApplication) -> None:
    x = np.linspace(5.0, 25.0, 8)
    y = np.linspace(0.1, 0.3, 8)
    yerr = np.full_like(x, 0.01)

    model = ParameterCompositeModel(["Linear"], [])
    params = ParameterSet([Parameter("m", 0.01), Parameter("b", 0.2)])
    fit = ParameterModelFit(
        parameter_name="Lambda",
        x_key="temperature",
        ranges=[ModelFitRange(x_min=5.0, x_max=25.0, model=model, parameters=params)],
    )

    dlg = ModelFitDialog(
        parameter_name="Lambda",
        x_key="temperature",
        x_values=x,
        y_values=y,
        y_errors=yerr,
        existing_fit=fit,
    )

    labels = [dlg._param_table.item(row, 0).text() for row in range(dlg._param_table.rowCount())]
    assert any("m [us^-1 / K]" in text for text in labels)
    assert any("b [us^-1]" in text for text in labels)


def test_commit_parameter_table_normalizes_domain_limits(qapp: QApplication) -> None:
    x = np.linspace(10.0, 100.0, 12)
    y = np.linspace(0.2, 0.4, 12)
    yerr = np.full_like(x, 0.02)

    model = ParameterCompositeModel(["Redfield", "DiffusionLF_2D", "Lambda_bg"], operators=["+", "+"])
    params = ParameterSet(
        [
            Parameter("D", 1.0, min=-10.0, max=10.0),
            Parameter("nu", 0.0, min=-5.0, max=-1.0),
            Parameter("m", 2.0, min=-3.0, max=-2.0),
            Parameter("A", 1.0, min=-10.0, max=10.0),
            Parameter("D_2D", -3.0, min=-2.0, max=-1.0),
            Parameter("D_perp", -4.0, min=-2.0, max=-1.0),
            Parameter("lambda_BG", -2.0, min=-1.0, max=-0.5),
        ]
    )
    fit = ParameterModelFit(
        parameter_name="Lambda",
        x_key="field",
        ranges=[ModelFitRange(x_min=10.0, x_max=100.0, model=model, parameters=params)],
    )

    dlg = ModelFitDialog(
        parameter_name="Lambda",
        x_key="field",
        x_values=x,
        y_values=y,
        y_errors=yerr,
        existing_fit=fit,
    )

    labels = [dlg._param_table.item(row, 0).text() for row in range(dlg._param_table.rowCount())]
    assert any("A [MHz]" in text for text in labels)

    dlg._commit_param_table()
    normalized = {p.name: p for p in dlg.get_model_fit().ranges[0].parameters}

    assert normalized["nu"].min > 0.0
    assert normalized["nu"].max >= normalized["nu"].min
    assert normalized["nu"].value >= normalized["nu"].min

    assert normalized["m"].min > 0.0

    assert normalized["D_2D"].min >= 0.0
    assert normalized["D_2D"].max >= normalized["D_2D"].min
    assert normalized["D_2D"].value >= normalized["D_2D"].min

    assert normalized["D_perp"].min >= 0.0
    assert normalized["lambda_BG"].min >= 0.0


def test_run_fit_sets_in_progress_state_immediately(qapp: QApplication, monkeypatch) -> None:
    x = np.linspace(1.0, 10.0, 20)
    y = 2.0 * x + 1.0
    yerr = np.full_like(x, 0.1)

    dlg = ModelFitDialog(
        parameter_name="Lambda",
        x_key="field",
        x_values=x,
        y_values=y,
        y_errors=yerr,
        existing_fit=None,
    )

    gate = threading.Event()

    from asymmetry.core.fitting.parameter_models import ParameterModelFitResult

    def _fake_fit(**_kwargs):
        gate.wait(timeout=1.0)
        return ParameterModelFitResult(success=True, reduced_chi_squared=1.0)

    monkeypatch.setattr("asymmetry.gui.panels.model_fit_dialog.fit_parameter_model", _fake_fit)

    try:
        dlg._run_fit(0)

        assert dlg._fit_in_progress is True
        assert "in progress" in dlg._fit_progress_label.text().lower()
    finally:
        gate.set()
        deadline = time.time() + 2.0
        while dlg._fit_in_progress and time.time() < deadline:
            qapp.processEvents()
            time.sleep(0.01)

    assert dlg._fit_in_progress is False


def test_edit_model_to_redfield_resets_m_to_default(qapp: QApplication, monkeypatch) -> None:
    x = np.linspace(1.0, 10.0, 20)
    y = 0.01 * x + 0.2
    yerr = np.full_like(x, 0.01)

    fit = ParameterModelFit(
        parameter_name="Lambda",
        x_key="field",
        ranges=[
            ModelFitRange(
                x_min=1.0,
                x_max=10.0,
                model=ParameterCompositeModel(["Linear"], []),
                parameters=ParameterSet([Parameter("m", 0.01), Parameter("b", 0.2)]),
            )
        ],
    )
    dlg = ModelFitDialog(
        parameter_name="Lambda",
        x_key="field",
        x_values=x,
        y_values=y,
        y_errors=yerr,
        existing_fit=fit,
    )

    class _FakeBuilder:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def exec(self):
            return QDialog.DialogCode.Accepted

        def get_model(self):
            return ParameterCompositeModel(["Redfield"], [])

    monkeypatch.setattr("asymmetry.gui.panels.model_fit_dialog.ParameterModelBuilderDialog", _FakeBuilder)
    dlg._edit_model(0)

    params = dlg.get_model_fit().ranges[0].parameters
    assert params["m"].value == pytest.approx(2.0)


def test_parameter_model_builder_has_info_column(qapp: QApplication) -> None:
    dialog = ParameterModelBuilderDialog(component_pool=["Linear", "Arrhenius"])

    headers = [dialog._table.horizontalHeaderItem(i).text() for i in range(dialog._table.columnCount())]
    assert headers == ["Op", "Component", "Info", "Remove"]

    info_btn = dialog._table.cellWidget(0, 2)
    assert info_btn is not None
    assert info_btn.text() == "Info"


def test_parameter_model_builder_groups_sc_models_in_submenu(qapp: QApplication) -> None:
    dialog = ParameterModelBuilderDialog(component_pool=["Linear", "SC_SWave", "SC_DWave"])

    selector = dialog._table.cellWidget(0, 1)
    assert isinstance(selector, _ComponentSelectorButton)

    menu = selector._build_component_menu()
    assert menu is not None

    top_actions = menu.actions()
    assert any(action.text() == "Linear" and action.menu() is None for action in top_actions)

    sc_action = next(
        (action for action in top_actions if action.menu() is not None and action.text() == _SC_COMPONENT_MENU_TITLE),
        None,
    )
    assert sc_action is not None
    sc_items = [action.text() for action in sc_action.menu().actions()]
    assert sc_items == ["SC_DWave", "SC_SWave"]


def test_component_info_html_contains_equation_and_parameters() -> None:
    from asymmetry.core.fitting.parameter_models import PARAMETER_MODEL_COMPONENTS

    html_doc = build_component_info_html(PARAMETER_MODEL_COMPONENTS["Arrhenius"])

    assert "Model Expression" in html_doc
    assert "Applicability" in html_doc
    assert "Ea" in html_doc
    assert "Activation energy" in html_doc
    assert "Availability" not in html_doc
    assert "<i>" in html_doc


def test_component_info_html_contains_sc_kernel_and_gap_function() -> None:
    from asymmetry.core.fitting.parameter_models import PARAMETER_MODEL_COMPONENTS

    html_doc = build_component_info_html(PARAMETER_MODEL_COMPONENTS["SC_AlphaModel"])

    assert "Superfluid-Density Kernel" in html_doc
    assert "Gap Function / Model Form" in html_doc
    assert "Measured Linewidth Convention" in html_doc
    assert "normalized superfluid density" in html_doc
    assert "Single-gap BCS shape" in html_doc
