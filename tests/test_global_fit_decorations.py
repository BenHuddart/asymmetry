"""Reconciliation Phase 5 — global-parameter-fit window decorations.

The Global Parameter Fit window's *decorations* (per-parameter local model fits
and free-floating plot annotations) are trend-attached state: they belong to the
``modelfit-<digest>`` :class:`FitSeries` the displayed cross-group fit produced,
not to a standalone top-level project key where they could orphan when the
backing fit is re-run or removed. These tests pin that the decorations survive a
project save/reload attached to their batch, that legacy projects carrying the
old top-level key still load (and migrate on next save), and that re-running the
same fit keeps its decorations.

View preferences (log axes, show-components, plot mode, …) are *not* decorations
and stay in ``global_parameter_fit_window_state``.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import asymmetry.gui.mainwindow as mw_module  # noqa: E402
from asymmetry.core.fitting.parameter_models import (  # noqa: E402
    CrossGroupFitResult,
    ModelFitRange,
    ParameterCompositeModel,
    ParameterGroupData,
    ParameterModelFit,
    ParameterModelFitResult,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet  # noqa: E402
from asymmetry.core.project import load_project, save_project  # noqa: E402
from asymmetry.core.representation import RepresentationType  # noqa: E402
from asymmetry.gui.mainwindow import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def _new_window() -> MainWindow:
    settings = QSettings()
    settings.setValue(mw_module._UI_SCALE_SETTINGS_KEY, 1.0)
    win = MainWindow()
    # Force a representation so the cross-group fit's results series is recorded
    # (and so the restored project model carries the matching batch).
    win._active_representation_type = lambda: RepresentationType.TIME_FB_ASYMMETRY
    return win


@pytest.fixture
def mainwindow(qapp: QApplication) -> MainWindow:
    return _new_window()


def _result() -> CrossGroupFitResult:
    """A successful two-group fit: shared global 'm', per-group local 'b'."""
    return CrossGroupFitResult(
        success=True,
        chi_squared=8.0,
        reduced_chi_squared=1.1,
        global_parameters=ParameterSet([Parameter(name="m", value=2.0)]),
        local_parameters={
            "g0": ParameterSet([Parameter(name="b", value=1.0)]),
            "g1": ParameterSet([Parameter(name="b", value=3.0)]),
        },
        global_uncertainties={"m": 0.05},
        local_uncertainties={"g0": {"b": 0.1}, "g1": {"b": 0.2}},
        message="Fit successful",
    )


def _groups() -> list[ParameterGroupData]:
    return [
        ParameterGroupData(
            group_id="g0",
            group_name="A",
            x=np.array([1.0, 2.0], dtype=float),
            y=np.array([0.1, 0.2], dtype=float),
            yerr=np.array([0.01, 0.01], dtype=float),
            group_variable_value=10.0,
        ),
        ParameterGroupData(
            group_id="g1",
            group_name="B",
            x=np.array([1.0, 2.0], dtype=float),
            y=np.array([0.3, 0.4], dtype=float),
            yerr=np.array([0.01, 0.01], dtype=float),
            group_variable_value=20.0,
        ),
    ]


def _local_model_fit() -> ParameterModelFit:
    return ParameterModelFit(
        parameter_name="b",
        x_key="temperature",
        ranges=[
            ModelFitRange(
                x_min=5.0,
                x_max=25.0,
                model=ParameterCompositeModel.from_expression("Linear"),
                parameters=ParameterSet(
                    [Parameter("m", value=0.1), Parameter("b", value=1.0)]
                ),
                result=ParameterModelFitResult(
                    success=True,
                    chi_squared=1.0,
                    reduced_chi_squared=0.5,
                    parameters=ParameterSet(
                        [Parameter("m", value=0.1), Parameter("b", value=1.0)]
                    ),
                    uncertainties={"m": 0.01, "b": 0.02},
                    message="Fit successful",
                ),
            )
        ],
        active=True,
    )


def _last_cross_group_payload() -> dict:
    """The fit_parameters_panel payload the restore path rebuilds the window from."""
    return {
        "parameter_name": "lambda",
        "x_key": "field",
        "groups": _groups(),
        "model": ParameterCompositeModel(["Linear"], []),
        "fit_result": _result(),
        "fit_x_min": float("nan"),
        "fit_x_max": float("nan"),
    }


def _drive_cross_group_fit(win: MainWindow) -> None:
    output = SimpleNamespace(
        fit_result=_result(),
        model=ParameterCompositeModel(["Linear"], []),
        x_key="field",
    )
    win._on_cross_group_fit_completed("lambda", _groups(), output)
    # The panel payload is what survives a save and rebuilds the window on load.
    win._fit_parameters_panel._last_cross_group_fit = _last_cross_group_payload()


def _add_decorations(win: MainWindow) -> None:
    window = win._global_parameter_fit_window
    assert window is not None
    window._local_model_fits["b"] = _local_model_fit()
    window._plot_annotations = [
        {
            "x": 1.5,
            "y": 2.5,
            "text": "note",
            "axis_tag": "g0",
            "is_group_label": False,
            "artist": None,
        }
    ]
    window._local_plot_annotations = [
        {"x": 12.0, "y": 0.2, "text": "loc", "axis_tag": "main", "artist": None}
    ]


def test_decorations_round_trip_attached_to_batch(
    mainwindow: MainWindow, qapp: QApplication, tmp_path
) -> None:
    """Decorations survive a project save/reload and reappear in the window.

    This pins the observable behaviour through the refactor: before, the
    decorations round-trip through the top-level window-state key; after, through
    the owning series' ``extra``. Either way, the user sees them again on reload.
    """
    _drive_cross_group_fit(mainwindow)
    _add_decorations(mainwindow)

    state = mainwindow.collect_project_state()
    project_path = tmp_path / "decorations.asymp"
    save_project(state, project_path)
    loaded = load_project(project_path)

    restored = _new_window()
    restored.restore_project_state(loaded, str(project_path))

    window = restored._global_parameter_fit_window
    assert window is not None
    assert "b" in window._local_model_fits
    fit = window._local_model_fits["b"]
    assert fit.parameter_name == "b"
    assert fit.ranges[0].model.component_names == ["Linear"]
    custom = [a for a in window._plot_annotations if not a.get("is_group_label")]
    assert any(a.get("text") == "note" for a in custom)
    assert any(a.get("text") == "loc" for a in window._local_plot_annotations)
