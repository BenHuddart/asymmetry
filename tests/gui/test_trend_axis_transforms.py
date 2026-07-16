"""Per-axis trend transforms + multi-series overlay in FitParametersPanel."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

import asymmetry.gui.panels.fit_parameters_panel as fpp
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
    # Unit-aware: the field unit (G) is squared, and the reciprocal Y keeps its
    # transformed unit.
    assert panel._transformed_x_axis_label("field") == "B² (G²)"
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


def _read_tsv(path):
    lines = path.read_text().splitlines()
    comments = [ln for ln in lines if ln.startswith("#")]
    data = [ln for ln in lines if not ln.startswith("#")]
    header = data[0].split("\t")
    body = [ln.split("\t") for ln in data[1:]]
    return comments, header, body


def test_tsv_export_transformed_columns_and_provenance(qapp, tmp_path, monkeypatch):
    panel = _panel_with_rows(_lambda_series([1.0, 2.0, 3.0], [4.0, 2.0, 1.5]))
    panel._set_axis_transform("x", AxisTransform.preset("square"))
    panel._set_axis_transform("y", AxisTransform.preset("reciprocal"))
    out = tmp_path / "t.tsv"
    monkeypatch.setattr(
        fpp.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out), ""))
    )
    panel._export_tsv()
    comments, header, body = _read_tsv(out)

    assert "# X transform: x**2" in comments
    assert "# Y transform: 1/x" in comments
    # Raw columns stay; transformed columns are appended (unit-aware headers).
    assert "B (G)" in header and "reduced_chi2" in header
    assert "B² (G²)" in header
    assert any(h.startswith("1/") for h in header)
    assert "Series" not in header  # single series: no Series column
    # The transformed B² column equals field² for the first data row.
    b2_col = header.index("B² (G²)")
    assert float(body[0][b2_col]) == pytest.approx(1.0)


def test_tsv_export_overlay_has_series_column(qapp, tmp_path, monkeypatch):
    panel = _panel_with_two_series(qapp)
    panel._set_selected_group_ids(["A", "B"], emit=False)
    out = tmp_path / "t.tsv"
    monkeypatch.setattr(
        fpp.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out), ""))
    )
    panel._export_tsv()
    _comments, header, body = _read_tsv(out)
    assert header[0] == "Series"
    series_names = {r[0] for r in body}
    assert series_names == {"400 G", "200 G"}


def test_tsv_export_plain_single_series_no_extra_columns(qapp, tmp_path, monkeypatch):
    panel = _panel_with_rows(_lambda_series([1.0, 2.0], [4.0, 2.0]))
    out = tmp_path / "t.tsv"
    monkeypatch.setattr(
        fpp.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out), ""))
    )
    panel._export_tsv()
    _comments, header, _body = _read_tsv(out)
    assert "Series" not in header
    assert header[:3] == ["Run", "B (G)", "T (K)"]


def test_gle_data_file_appends_transformed_columns(qapp, tmp_path):
    panel = _panel_with_rows(_lambda_series([1.0, 2.0, 3.0], [4.0, 2.0, 1.5]))
    panel._set_axis_transform("x", AxisTransform.preset("square"))
    panel._set_axis_transform("y", AxisTransform.preset("reciprocal"))
    data_path = tmp_path / "fit.dat"
    panel._write_gle_data_file(data_path)
    text = data_path.read_text()

    assert "! X transform: x**2" in text
    assert "! Y transform: 1/x" in text
    # Transformed columns are appended after the χ² pair, so raw column indices
    # (B_field at c2) are unchanged and the transformed x lands past them.
    x_col = panel._gle_transformed_x_column()
    assert x_col is not None and x_col > panel._gle_base_column_count()
    # First data row: raw B = 1.0 at c2, transformed B² = 1.0 at the new column.
    data_rows = [ln for ln in text.splitlines() if not ln.startswith("!") and ln.strip()]
    first = data_rows[0].split()
    assert float(first[1]) == pytest.approx(1.0)  # raw B (G)
    assert float(first[x_col - 1]) == pytest.approx(1.0)  # transformed B²


def test_gle_effective_columns_point_at_transformed(qapp):
    panel = _panel_with_rows(_lambda_series([1.0, 2.0], [4.0, 2.0]))
    # Untransformed: effective == raw.
    assert panel._gle_effective_x_column("field") == panel._gle_x_column("field")
    panel._set_axis_transform("x", AxisTransform.preset("square"))
    panel._set_axis_transform("y", AxisTransform.preset("reciprocal"))
    assert panel._gle_effective_x_column("field") == panel._gle_transformed_x_column()
    assert panel._gle_effective_columns_for_param("Lambda") == (
        panel._gle_transformed_columns_for_param("Lambda")
    )


def test_gle_iter_skips_stale_transform_fits(qapp):
    panel = _panel_with_rows(_lambda_series([1.0, 2.0], [4.0, 2.0]))
    panel._model_fits["Lambda"] = _StubFit()
    # Recorded under identity → visible under identity.
    panel._model_fit_transform_sig["Lambda"] = panel._transform_signature()
    assert [p for p, _i, _r in panel._iter_active_fit_ranges("field")] == ["Lambda"]
    # A transform the fit was NOT computed under → the .fit sidecar is skipped.
    panel._set_axis_transform("x", AxisTransform.preset("square"))
    assert list(panel._iter_active_fit_ranges("field")) == []


def test_select_series_public_api_arms_overlay(qapp):
    panel = _panel_with_two_series(qapp)
    panel.select_series(["A", "B"])
    series = panel._series_to_plot()
    assert {s.name for s in series} == {"400 G", "200 G"}
    # Unknown ids are ignored; all-unknown is a no-op (selection unchanged).
    panel.select_series(["nope"])
    assert len(panel._series_to_plot()) == 2


def test_active_series_flagged_in_legend(qapp):
    panel = _panel_with_two_series(qapp)
    panel.select_series(["A", "B"])
    panel._draw_plot()
    ax = panel._figure.axes[0]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any(lbl.endswith("(active)") for lbl in labels)
    assert sum(lbl.endswith("(active)") for lbl in labels) == 1


class _StubResult:
    success = True


class _StubRange:
    result = _StubResult()


class _StubFit:
    active = True
    x_key = "field"
    ranges = [_StubRange()]


def test_stale_fit_button_shows_warning_state(qapp):
    panel = _panel_with_rows(_lambda_series([1.0, 2.0], [4.0, 2.0]))
    panel._model_fits["Lambda"] = _StubFit()
    panel._model_fit_transform_sig["Lambda"] = panel._transform_signature()
    panel._refresh_model_fit_button_labels()
    assert panel._y_controls["Lambda"].fit_button.text() == "Model Fit*"
    # A transform the fit was not computed under → stale ⚠ state, not a bare star.
    panel._set_axis_transform("x", AxisTransform.preset("square"))
    assert panel._y_controls["Lambda"].fit_button.text() == "Model Fit ⚠"
    # Returning to the fit's transform clears the warning.
    panel._set_axis_transform("x", AxisTransform.identity())
    assert panel._y_controls["Lambda"].fit_button.text() == "Model Fit*"


def test_transform_dropped_count(qapp):
    rows = _lambda_series([0.0, 1.0, 2.0], [4.0, -1.0, 2.0])
    panel = _panel_with_rows(rows)
    # No transform → nothing dropped.
    assert panel._transform_dropped_count(rows, "field", ["Lambda"]) == 0
    # 1/x drops the zero-field point.
    panel._x_transform = AxisTransform.preset("reciprocal")
    assert panel._transform_dropped_count(rows, "field", ["Lambda"]) == 1
    # ln y additionally drops the negative-λ point (distinct row) → 2 total.
    panel._y_transform = AxisTransform.preset("log")
    assert panel._transform_dropped_count(rows, "field", ["Lambda"]) == 2


def test_provenance_reports_transform_drops(qapp):
    rows = _lambda_series([0.0, 1.0], [4.0, 2.0])
    panel = _panel_with_rows(rows)
    panel._update_trend_provenance(rows, transform_dropped=1)
    label = panel._trend_provenance_label
    # The panel isn't shown in a headless test, so check the explicit-hidden flag
    # (cleared by label.show()) rather than isVisible(), which needs a shown parent.
    assert not label.isHidden()
    assert "dropped by transform" in label.text()


def test_component_sort_places_polynomials_by_degree():
    from asymmetry.gui.panels.model_fit_dialog import _component_sort_key

    ordered = sorted(["Cubic", "Arrhenius", "Linear", "Quadratic"], key=_component_sort_key)
    assert ordered == ["Arrhenius", "Linear", "Quadratic", "Cubic"]


def test_multi_series_draw_smoke(qapp):
    panel = _panel_with_two_series(qapp)
    panel.select_series(["A", "B"])
    panel._x_transform = AxisTransform.preset("square")
    panel._y_transform = AxisTransform.preset("reciprocal")
    # Should render two overlaid, transformed series without raising.
    panel._draw_plot()
    ax = panel._figure.axes[0]
    assert ax.get_xlabel() == "B² (G²)"
    # One scatter collection per series.
    assert len(ax.collections) >= 2
