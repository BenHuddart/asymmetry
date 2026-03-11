"""Tests for ModelFitDialog range-parameter labels and bounds normalization."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from asymmetry.core.fitting.parameter_models import ModelFitRange, ParameterCompositeModel, ParameterModelFit
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.gui.panels.model_fit_dialog import ModelFitDialog


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
