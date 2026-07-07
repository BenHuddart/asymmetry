"""B7: a non-numeric / mangled seed edit must NOT silently reset the seed to 0.

Live-testing finding (Round-3): in the parameter-trend "Model Fit" dialog, typing
into a seed cell occasionally produced a mangled string (the Windows TextInputHost
focus issue rendered ``-10`` as ``--0``). ``_commit_param_table`` caught the
``ValueError`` and **silently reset the seed to 0.0**, destroying the user's value
with no feedback. The committed value should instead fall back to the parameter's
previous value (and ideally flag the rejected edit), never silently 0.
"""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from asymmetry.core.fitting.parameter_models import (
    ModelFitRange,
    ParameterCompositeModel,
    ParameterModelFit,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.gui.panels.model_fit_dialog import ModelFitDialog

pytestmark = [pytest.mark.gui]


def _ea_row(dlg: ModelFitDialog) -> int:
    for r in range(dlg._param_table.rowCount()):
        item = dlg._param_table.item(r, 0)
        if item is not None and item.data(Qt.ItemDataRole.UserRole) == "Ea":
            return r
    raise AssertionError("Ea parameter row not found")


def test_unparseable_seed_edit_preserves_previous_value(qapp: QApplication) -> None:
    x = np.linspace(100.0, 400.0, 12)
    y = np.linspace(0.7, 0.06, 12)
    err = np.full_like(x, 0.01)

    model = ParameterCompositeModel(["Arrhenius", "Constant"], ["+"])
    params = ParameterSet([Parameter("a", 0.001), Parameter("Ea", -12.0), Parameter("c", 0.065)])
    fit = ParameterModelFit(
        parameter_name="Lambda",
        x_key="temperature",
        ranges=[ModelFitRange(x_min=100.0, x_max=400.0, model=model, parameters=params)],
    )
    dlg = ModelFitDialog(
        parameter_name="Lambda",
        x_key="temperature",
        x_values=x,
        y_values=y,
        y_errors=err,
        existing_fit=fit,
    )
    dlg._select_range(0)
    row = _ea_row(dlg)

    # A mangled / non-numeric entry (what the live input layer produced).
    dlg._param_table.item(row, 1).setText("--0")
    dlg._commit_param_table()

    committed = dlg.get_model_fit().ranges[0].parameters["Ea"].value
    # Must keep the previous seed (-12), NOT silently reset to 0.
    assert committed == pytest.approx(-12.0), (
        f"unparseable seed edit silently reset Ea to {committed} instead of "
        "preserving the previous -12"
    )
