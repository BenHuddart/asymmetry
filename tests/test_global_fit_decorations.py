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

import copy
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
                parameters=ParameterSet([Parameter("m", value=0.1), Parameter("b", value=1.0)]),
                result=ParameterModelFitResult(
                    success=True,
                    chi_squared=1.0,
                    reduced_chi_squared=0.5,
                    parameters=ParameterSet([Parameter("m", value=0.1), Parameter("b", value=1.0)]),
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


def _modelfit_batch(state: dict) -> dict:
    """Return the per-fit ``modelfit-<digest>`` batch dict in a project state."""
    batches = [
        b
        for b in state.get("batches", [])
        if isinstance(b, dict)
        and str(b.get("batch_id", "")).startswith("modelfit-")
        and not str(b.get("batch_id", "")).startswith("modelfit-globals")
    ]
    assert len(batches) == 1, f"expected one modelfit series, got {len(batches)}"
    return batches[0]


def test_decorations_persist_in_series_extra_not_window_key(
    mainwindow: MainWindow,
) -> None:
    """The new home is the owning series' ``extra``; the window-state key holds
    view preferences only (no decoration keys)."""
    from asymmetry.gui.mainwindow import _GLOBAL_FIT_DECORATIONS_EXTRA_KEY
    from asymmetry.gui.windows.global_parameter_fit_window import _DECORATION_STATE_KEYS

    _drive_cross_group_fit(mainwindow)
    _add_decorations(mainwindow)

    state = mainwindow.collect_project_state()

    extra = _modelfit_batch(state).get("extra", {})
    assert _GLOBAL_FIT_DECORATIONS_EXTRA_KEY in extra
    deco = extra[_GLOBAL_FIT_DECORATIONS_EXTRA_KEY]
    assert "b" in deco["local_model_fits"]

    window_key = state["global_parameter_fit_window_state"]
    assert isinstance(window_key, dict)
    # View prefs survive; no decoration key leaks into the window-state key.
    assert "local_log_x" in window_key
    for key in _DECORATION_STATE_KEYS:
        assert key not in window_key


def test_legacy_window_state_migrates_to_series_extra(mainwindow: MainWindow, tmp_path) -> None:
    """A legacy project (decorations inline in the window-state key, none on the
    series) loads with decorations visible, then migrates to the series' extra on
    the next save."""
    from asymmetry.gui.mainwindow import _GLOBAL_FIT_DECORATIONS_EXTRA_KEY

    _drive_cross_group_fit(mainwindow)
    _add_decorations(mainwindow)
    new_state = mainwindow.collect_project_state()

    # Rewrite into the legacy shape: decorations live inline under the window
    # key, and the series carries none (older app versions never wrote them).
    legacy = copy.deepcopy(new_state)
    batch = _modelfit_batch(legacy)
    decorations = batch["extra"].pop(_GLOBAL_FIT_DECORATIONS_EXTRA_KEY)
    legacy["global_parameter_fit_window_state"].update(decorations)
    assert _GLOBAL_FIT_DECORATIONS_EXTRA_KEY not in _modelfit_batch(legacy)["extra"]

    legacy_path = tmp_path / "legacy.asymp"
    save_project(legacy, legacy_path)
    loaded = load_project(legacy_path)

    restored = _new_window()
    restored.restore_project_state(loaded, str(legacy_path))

    # Decorations are visible after loading the legacy project.
    window = restored._global_parameter_fit_window
    assert window is not None
    assert "b" in window._local_model_fits
    assert any(a.get("text") == "note" for a in window._plot_annotations)

    # On the next save they migrate to the series' extra and leave the key.
    resaved = restored.collect_project_state()
    assert _GLOBAL_FIT_DECORATIONS_EXTRA_KEY in _modelfit_batch(resaved)["extra"]
    assert "local_model_fits" not in resaved["global_parameter_fit_window_state"]


def test_rerun_fit_preserves_decorations(mainwindow: MainWindow) -> None:
    """Re-running the same fit (its series replaced in place) keeps decorations —
    in the live window and stamped on the (re-recorded) series' extra."""
    from asymmetry.gui.mainwindow import _GLOBAL_FIT_DECORATIONS_EXTRA_KEY

    _drive_cross_group_fit(mainwindow)
    _add_decorations(mainwindow)
    # Stamp decorations onto the series (as a save would).
    mainwindow._sync_global_fit_decorations_to_series()
    batch_id = mainwindow._global_parameter_fit_window.batch_id()
    assert batch_id is not None
    assert _GLOBAL_FIT_DECORATIONS_EXTRA_KEY in mainwindow._project_model.batch(batch_id).extra

    # Re-run the identical fit: same logical key → same batch id, replaced in place.
    output = SimpleNamespace(
        fit_result=_result(),
        model=ParameterCompositeModel(["Linear"], []),
        x_key="field",
    )
    mainwindow._on_cross_group_fit_completed("lambda", _groups(), output)

    window = mainwindow._global_parameter_fit_window
    assert window.batch_id() == batch_id  # same fit, not a new one
    # Live decorations follow the replacement.
    assert "b" in window._local_model_fits
    assert any(a.get("text") == "note" for a in window._plot_annotations)
    # And the re-recorded series still carries them (extra preserved across add_batch).
    extra = mainwindow._project_model.batch(batch_id).extra
    assert _GLOBAL_FIT_DECORATIONS_EXTRA_KEY in extra


def test_switching_fits_does_not_orphan_or_bleed_decorations(
    mainwindow: MainWindow,
) -> None:
    """Decorations belong to their fit: a new fit starts clean, and switching
    back to the first restores its decorations from its series."""
    _drive_cross_group_fit(mainwindow)  # fit A: lambda vs field
    _add_decorations(mainwindow)
    window = mainwindow._global_parameter_fit_window
    first_id = window.batch_id()

    # Fit B: a different parameter → different batch id.
    output_b = SimpleNamespace(
        fit_result=_result(),
        model=ParameterCompositeModel(["Linear"], []),
        x_key="field",
    )
    mainwindow._on_cross_group_fit_completed("nu", _groups(), output_b)
    second_id = window.batch_id()
    assert second_id != first_id
    # B starts with no decorations (A's did not bleed across).
    assert window._local_model_fits == {}
    assert window._plot_annotations == []

    # Switch back to A: its decorations are restored from its series.
    output_a = SimpleNamespace(
        fit_result=_result(),
        model=ParameterCompositeModel(["Linear"], []),
        x_key="field",
    )
    mainwindow._on_cross_group_fit_completed("lambda", _groups(), output_a)
    assert window.batch_id() == first_id
    assert "b" in window._local_model_fits


def test_window_view_state_and_decorations_split(qapp: QApplication) -> None:
    """get_view_state holds prefs only; get_decorations holds the decoration
    subset; set_results clears decorations when the batch changes."""
    from asymmetry.gui.windows.global_parameter_fit_window import (
        _DECORATION_STATE_KEYS,
        GlobalParameterFitWindow,
    )

    window = GlobalParameterFitWindow()
    window._local_log_x_check.setChecked(True)
    window._local_model_fits["b"] = _local_model_fit()
    window._plot_annotations = [
        {"x": 1.0, "y": 2.0, "text": "n", "axis_tag": "g0", "is_group_label": False}
    ]

    view = window.get_view_state()
    deco = window.get_decorations()
    assert view.get("local_log_x") is True
    assert all(key not in view for key in _DECORATION_STATE_KEYS)
    assert set(deco).issubset(set(_DECORATION_STATE_KEYS))
    assert "b" in deco["local_model_fits"]
    assert window.has_decorations() is True

    # A batch change clears decorations; a matching batch keeps them.
    window.set_results(
        parameter_name="lambda",
        x_key="field",
        groups=_groups(),
        model=ParameterCompositeModel(["Linear"], []),
        result=_result(),
        batch_id="modelfit-aaaa",
    )
    window._local_model_fits["b"] = _local_model_fit()
    window.set_results(
        parameter_name="lambda",
        x_key="field",
        groups=_groups(),
        model=ParameterCompositeModel(["Linear"], []),
        result=_result(),
        batch_id="modelfit-aaaa",  # same batch → kept
    )
    assert "b" in window._local_model_fits
    window.set_results(
        parameter_name="lambda",
        x_key="field",
        groups=_groups(),
        model=ParameterCompositeModel(["Linear"], []),
        result=_result(),
        batch_id="modelfit-bbbb",  # different batch → cleared
    )
    assert window._local_model_fits == {}
