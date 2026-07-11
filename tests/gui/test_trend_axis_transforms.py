"""Per-axis trend transforms + multi-series overlay in FitParametersPanel."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from asymmetry.core.fitting.axis_transforms import AxisTransform
from asymmetry.gui.panels.fit_parameters_panel import (
    FitParametersPanel,
    _FitRow,
    _GroupFitData,
)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def _row(run_number: int, field: float, lam: float, lam_err: float = 0.02) -> _FitRow:
    return _FitRow(
        run_number=run_number,
        run_label=str(run_number),
        field=field,
        temperature=10.0,
        values={"Lambda": lam},
        errors={"Lambda": lam_err},
    )


def _lambda_series(fields: list[float], lambdas: list[float]) -> list[_FitRow]:
    return [_row(i + 1, f, lam) for i, (f, lam) in enumerate(zip(fields, lambdas, strict=True))]


def _panel_with_rows(rows: list[_FitRow]) -> FitParametersPanel:
    panel = FitParametersPanel()
    panel._rows = rows
    panel._varying_params = ["Lambda"]
    panel._inferred_x_key = "field"
    panel._selected_y_param_names = ["Lambda"]
    panel._rebuild_y_controls(preferred_selected=["Lambda"])
    return panel


# ---------------------------------------------------------------------------
# Data-assembly boundary
# ---------------------------------------------------------------------------


def test_x_transform_applied_to_arrays(qapp):
    panel = _panel_with_rows(_lambda_series([1.0, 2.0], [4.0, 2.0]))
    panel._x_transform = AxisTransform.preset("square")
    x_vals, _ = panel._apply_x_transform(np.array([1.0, 2.0]), None)
    np.testing.assert_allclose(x_vals, [1.0, 4.0])


def test_y_transform_reciprocal_with_error_propagation(qapp):
    rows = _lambda_series([1.0, 2.0], [4.0, 2.0])
    panel = _panel_with_rows(rows)
    panel._y_transform = AxisTransform.preset("reciprocal")
    y_vals, y_err = panel._series_y_arrays(rows, "Lambda")
    np.testing.assert_allclose(y_vals, [0.25, 0.5])
    # sigma_(1/λ) = sigma_λ / λ^2
    np.testing.assert_allclose(y_err, [0.02 / 16.0, 0.02 / 4.0])


def test_identity_transform_is_passthrough(qapp):
    rows = _lambda_series([1.0, 2.0], [4.0, 2.0])
    panel = _panel_with_rows(rows)
    y_vals, y_err = panel._series_y_arrays(rows, "Lambda")
    np.testing.assert_allclose(y_vals, [4.0, 2.0])
    np.testing.assert_allclose(y_err, [0.02, 0.02])


# ---------------------------------------------------------------------------
# Labels & header chip
# ---------------------------------------------------------------------------


def test_transformed_axis_labels(qapp):
    panel = _panel_with_rows(_lambda_series([1.0], [4.0]))
    panel._x_transform = AxisTransform.preset("square")
    panel._y_transform = AxisTransform.preset("reciprocal")
    # Symbol-only: unit is stripped so the transform wraps a bare symbol.
    assert panel._transformed_x_axis_label("field") == "B²"
    assert panel._transformed_y_axis_label("Lambda").startswith("1/")


def test_transform_suffix_reflects_active_transforms(qapp):
    panel = _panel_with_rows(_lambda_series([1.0], [4.0]))
    panel._set_axis_transform("x", AxisTransform.preset("reciprocal"))
    panel._set_axis_transform("y", AxisTransform.preset("log"))
    # Both transforms surface in the collapsed section chip.
    panel._update_transform_suffix()
    # (No direct getter; assert the describe() pieces the suffix is built from.)
    assert panel._x_transform.describe("x") == "1/x"
    assert panel._y_transform.describe("y") == "ln y"


# ---------------------------------------------------------------------------
# Log-scale guard
# ---------------------------------------------------------------------------


def test_log_transform_disables_log_axis_checkbox(qapp):
    panel = _panel_with_rows(_lambda_series([1.0, 2.0], [4.0, 2.0]))
    panel._log_x_check.setChecked(True)
    panel._set_axis_transform("x", AxisTransform.preset("log"))
    assert not panel._log_x_check.isEnabled()
    assert not panel._log_x_check.isChecked()
    # Clearing the transform restores the checkbox.
    panel._set_axis_transform("x", AxisTransform.identity())
    assert panel._log_x_check.isEnabled()


def test_log_y_transform_disables_per_param_log(qapp):
    panel = _panel_with_rows(_lambda_series([1.0, 2.0], [4.0, 2.0]))
    controls = panel._y_controls["Lambda"]
    controls.log.setChecked(True)
    panel._set_axis_transform("y", AxisTransform.preset("log"))
    assert not controls.log.isEnabled()
    assert not controls.log.isChecked()


# ---------------------------------------------------------------------------
# Overlay staleness guard
# ---------------------------------------------------------------------------


def test_overlay_suppressed_when_transform_changes(qapp):
    panel = _panel_with_rows(_lambda_series([1.0, 2.0], [4.0, 2.0]))
    # A fit recorded under identity is not suppressed under identity...
    panel._model_fit_transform_sig["Lambda"] = panel._transform_signature()
    assert not panel._overlay_suppressed_for_transform("Lambda")
    # ...but is once a transform is active it was not fit under.
    panel._x_transform = AxisTransform.preset("square")
    assert panel._overlay_suppressed_for_transform("Lambda")


# ---------------------------------------------------------------------------
# State round-trip
# ---------------------------------------------------------------------------


def test_transform_state_round_trip(qapp):
    panel = _panel_with_rows(_lambda_series([1.0, 2.0], [4.0, 2.0]))
    panel._set_axis_transform("x", AxisTransform.preset("square"))
    panel._set_axis_transform("y", AxisTransform.custom("1/x"))
    panel._axis_transform_custom_memory["y"] = "1/x"
    state = panel.get_state()

    restored = FitParametersPanel()
    restored.restore_state(state)
    assert restored._x_transform == AxisTransform.preset("square")
    assert restored._y_transform == AxisTransform.custom("1/x")
    assert restored._axis_transform_custom_memory.get("y") == "1/x"


def test_identity_transform_not_persisted(qapp):
    panel = _panel_with_rows(_lambda_series([1.0], [4.0]))
    state = panel.get_state()
    assert state["x_transform"] == {}
    assert state["y_transform"] == {}


# ---------------------------------------------------------------------------
# Multi-series overlay
# ---------------------------------------------------------------------------


def _group(gid: str, name: str, rows: list[_FitRow]) -> _GroupFitData:
    return _GroupFitData(
        group_id=gid,
        group_name=name,
        rows=rows,
        global_params=None,
        varying_params=["Lambda"],
        inferred_x_key="field",
        model_fits={},
        plot_annotations=[],
    )


def _panel_with_two_series(qapp) -> FitParametersPanel:
    rows_a = _lambda_series([1.0, 2.0, 3.0], [4.0, 2.0, 1.5])
    rows_b = _lambda_series([1.0, 2.0, 3.0], [3.0, 1.6, 1.2])
    g_a = _group("A", "400 G", rows_a)
    g_b = _group("B", "200 G", rows_b)
    panel = FitParametersPanel()
    panel._group_fit_results = {"A": g_a, "B": g_b}
    panel._rebuild_group_buttons()
    panel._active_group_id = "A"
    panel._rows = list(rows_a)
    panel._varying_params = ["Lambda"]
    panel._inferred_x_key = "field"
    panel._selected_y_param_names = ["Lambda"]
    panel._rebuild_y_controls(preferred_selected=["Lambda"])
    return panel


def test_series_to_plot_single_when_one_selected(qapp):
    panel = _panel_with_two_series(qapp)
    panel._set_selected_group_ids(["A"], emit=False)
    series = panel._series_to_plot()
    assert len(series) == 1
    assert series[0].is_active


def test_series_to_plot_overlay_when_multiple_selected(qapp):
    panel = _panel_with_two_series(qapp)
    panel._set_selected_group_ids(["A", "B"], emit=False)
    series = panel._series_to_plot()
    assert len(series) == 2
    names = {s.name for s in series}
    assert names == {"400 G", "200 G"}
    # Distinct, stable colours per series.
    assert len({s.color for s in series}) == 2
    assert sum(s.is_active for s in series) == 1


def test_multi_series_draw_smoke(qapp):
    panel = _panel_with_two_series(qapp)
    panel._set_selected_group_ids(["A", "B"], emit=False)
    panel._x_transform = AxisTransform.preset("square")
    panel._y_transform = AxisTransform.preset("reciprocal")
    # Should render two overlaid, transformed series without raising.
    panel._draw_plot()
    ax = panel._figure.axes[0]
    assert ax.get_xlabel() == "B²"
    # One scatter collection per series.
    assert len(ax.collections) >= 2
