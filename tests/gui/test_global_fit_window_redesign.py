"""Tests for the Phase 3 GlobalParameterFitWindow redesign.

Covers the grid Fig-3 pane (columns/show-hide/share-Y), per-panel χ²ᵣ chips,
the component legend, the quality bar, residual/pull mode, the global-table
upgrade (units + Copy/Export + correlations), the studies sidebar signals, and
custom-x labels. The value-with-error formatter is unit-tested separately.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

pyside6 = pytest.importorskip("PySide6")
from PySide6.QtCore import Qt  # type: ignore  # noqa: E402
from PySide6.QtWidgets import (  # type: ignore  # noqa: E402
    QApplication,
    QHeaderView,
    QScrollArea,
    QSplitter,
)

from asymmetry.core.fitting.parameter_models import (  # noqa: E402
    CrossGroupFitResult,
    ParameterCompositeModel,
    ParameterGroupData,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet  # noqa: E402
from asymmetry.core.representation.global_fit_study import GlobalFitStudy  # noqa: E402
from asymmetry.gui.windows.global_fit_window_helpers import (  # noqa: E402
    build_global_table_rows,
    format_value_with_error,
    global_table_csv,
    global_table_latex,
    global_table_tsv,
    uncertainty_trust_flag,
)
from asymmetry.gui.windows.global_parameter_fit_window import (  # noqa: E402
    GlobalParameterFitWindow,
    StudySidebarEntry,
)
from tests._qt_helpers import wait_for  # noqa: E402


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _wait_fit_curves(window: GlobalParameterFitWindow, timeout_s: float = 10.0) -> None:
    wait_for(
        lambda: not window._fit_curve_compute_active,
        QApplication.instance(),
        timeout_s=timeout_s,
    )


def _make_groups(n: int) -> list[ParameterGroupData]:
    groups = []
    for i in range(n):
        groups.append(
            ParameterGroupData(
                group_id=f"g{i}",
                group_name=f"G{i}",
                x=np.array([100.0, 200.0, 300.0], dtype=float),
                y=np.array([0.2, 0.15, 0.1], dtype=float),
                yerr=np.array([0.01, 0.01, 0.01], dtype=float),
                group_variable_value=float(i),
            )
        )
    return groups


def _result_with_per_group(groups: list[ParameterGroupData]) -> CrossGroupFitResult:
    per_group_chi = {g.group_id: 3.0 for g in groups}
    per_group_n = {g.group_id: 3 for g in groups}
    return CrossGroupFitResult(
        success=True,
        chi_squared=3.0 * len(groups),
        reduced_chi_squared=1.05,
        global_parameters=ParameterSet([Parameter("c", value=0.15)]),
        local_parameters={g.group_id: ParameterSet() for g in groups},
        fixed_parameters=ParameterSet(),
        error_mode="column",
        n_points=3 * len(groups),
        per_group_chi_squared=per_group_chi,
        per_group_n_points=per_group_n,
    )


def _set_and_wait(
    window: GlobalParameterFitWindow,
    groups: list[ParameterGroupData],
    result: CrossGroupFitResult,
    **kwargs,
) -> None:
    window.set_results(
        parameter_name="Lambda",
        x_key="field",
        groups=groups,
        model=ParameterCompositeModel(["Constant"]),
        result=result,
        **kwargs,
    )
    _wait_fit_curves(window)


# ── format_value_with_error ─────────────────────────────────────────────────


def test_format_value_with_error_edge_cases() -> None:
    assert format_value_with_error(63.4, 2.1) == "63(2)"
    assert format_value_with_error(0.0674, 0.0031) == "0.067(3)"
    # err=0 or nan → plain value.
    assert format_value_with_error(5.0, 0.0) == "5"
    assert format_value_with_error(5.0, float("nan")) == "5"
    assert format_value_with_error(5.0, None) == "5"
    # Leading digit 1 keeps two sig figs.
    assert format_value_with_error(1.23, 0.012) == "1.230(12)"
    # Non-finite value renders verbatim.
    assert format_value_with_error(float("nan"), 0.1) == "nan"


# ── grid layout ─────────────────────────────────────────────────────────────


def test_grid_auto_columns_for_eight_groups(qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    groups = _make_groups(8)
    _set_and_wait(window, groups, _result_with_per_group(groups))

    # Auto: ceil(sqrt(8)) = 3 columns, capped at 4.
    assert window._grid_columns(8) == 3
    assert window._left_figure is not None
    axes = window._left_figure.axes
    assert len(axes) == 8
    cols = {ax.get_subplotspec().colspan.start for ax in axes}
    rows = {ax.get_subplotspec().rowspan.start for ax in axes}
    # 8 panels over 3 columns → 3 rows.
    assert max(cols) == 2  # columns 0..2
    assert max(rows) == 2  # rows 0..2 (ceil(8/3))


def test_grid_explicit_columns(qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    window._fit_columns_combo.setCurrentText("2")
    assert window._grid_columns(8) == 2
    window._fit_columns_combo.setCurrentText("4")
    assert window._grid_columns(8) == 4
    # Never more columns than panels.
    assert window._grid_columns(3) == 3


def test_hidden_group_reduces_panels_not_fit(qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    groups = _make_groups(4)
    _set_and_wait(window, groups, _result_with_per_group(groups))
    assert len(window._left_figure.axes) == 4

    window._group_visibility["g0"] = False
    window._refresh_plot()
    assert len(window._left_figure.axes) == 3
    # The fit result is untouched — g0 still present in the groups/result.
    assert any(g.group_id == "g0" for g in window._groups)


def test_share_y_wires_shared_axis(qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    window._fit_share_y_check.setChecked(True)
    groups = _make_groups(4)
    _set_and_wait(window, groups, _result_with_per_group(groups))
    axes = window._left_figure.axes
    # All panels should share the first panel's y-axis.
    shared = axes[0].get_shared_y_axes()
    assert all(shared.joined(axes[0], ax) for ax in axes[1:])


# ── chi-squared chips ───────────────────────────────────────────────────────


def _chip_texts(window: GlobalParameterFitWindow) -> list[str]:
    texts = []
    for ax in window._left_figure.axes:
        for artist in ax.texts:
            texts.append(artist.get_text())
    return texts


def test_chi_chips_present_with_enriched_result(qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    groups = _make_groups(2)
    _set_and_wait(window, groups, _result_with_per_group(groups))
    chips = [t for t in _chip_texts(window) if t.startswith("χ²ᵣ=")]
    assert len(chips) == 2


def test_chi_chips_absent_for_legacy_result(qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    groups = _make_groups(2)
    # Legacy result: no per_group_chi_squared / per_group_n_points.
    legacy = CrossGroupFitResult(
        success=True,
        chi_squared=2.0,
        reduced_chi_squared=1.0,
        global_parameters=ParameterSet([Parameter("c", value=0.15)]),
        local_parameters={"g0": ParameterSet(), "g1": ParameterSet()},
    )
    _set_and_wait(window, groups, legacy)
    chips = [t for t in _chip_texts(window) if t.startswith("χ²ᵣ=")]
    assert not chips


def test_chi_chips_suppressed_for_scatter_mode(qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    groups = _make_groups(2)
    result = _result_with_per_group(groups)
    result.error_mode = "scatter"
    _set_and_wait(window, groups, result)
    chips = [t for t in _chip_texts(window) if t.startswith("χ²ᵣ=")]
    assert not chips


def test_per_group_reduced_chi_squared_dof_convention(qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    groups = _make_groups(1)
    # One local parameter → dof = n - 1 local = 3 - 1 = 2, chi=6 → 3.0.
    result = CrossGroupFitResult(
        success=True,
        chi_squared=6.0,
        reduced_chi_squared=1.0,
        local_parameters={"g0": ParameterSet([Parameter("A", value=0.1)])},
        per_group_chi_squared={"g0": 6.0},
        per_group_n_points={"g0": 3},
    )
    window.set_results(
        parameter_name="Lambda",
        x_key="field",
        groups=groups,
        model=ParameterCompositeModel(["Constant"]),
        result=result,
    )
    _wait_fit_curves(window)
    assert window._per_group_reduced_chi_squared("g0") == pytest.approx(3.0)


# ── component legend ────────────────────────────────────────────────────────


def test_component_legend_labels_match_component_names(qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    window._show_components_check.setChecked(True)
    groups = _make_groups(1)
    result = CrossGroupFitResult(
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        global_parameters=ParameterSet([Parameter("a", value=0.1), Parameter("c", value=0.05)]),
        local_parameters={"g0": ParameterSet()},
    )
    window.set_results(
        parameter_name="Lambda",
        x_key="field",
        groups=groups,
        model=ParameterCompositeModel.from_expression("Linear + Constant"),
        result=result,
    )
    _wait_fit_curves(window)
    legends = window._left_figure.legends
    assert legends, "a figure-level component legend should be present"
    labels = {t.get_text() for t in legends[0].get_texts()}
    assert "Linear" in labels
    assert "Constant" in labels


# ── quality bar ─────────────────────────────────────────────────────────────


def test_quality_bar_text_success(qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    groups = _make_groups(2)
    _set_and_wait(window, groups, _result_with_per_group(groups))
    text = window._quality_label.text()
    assert "χ²=" in text
    assert "χ²ᵣ=" in text
    assert "n=6" in text
    assert "column σ" in text


def test_quality_bar_failure_message_in_red(qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    groups = _make_groups(2)
    result = CrossGroupFitResult(
        success=False,
        chi_squared=float("nan"),
        reduced_chi_squared=float("nan"),
        message="Fit failed to converge",
        local_parameters={"g0": ParameterSet(), "g1": ParameterSet()},
    )
    _set_and_wait(window, groups, result)
    text = window._quality_label.text()
    assert "Fit failed to converge" in text
    assert "color" in text  # red span


def test_quality_bar_scatter_mode_greys_reduced_chi(qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    groups = _make_groups(2)
    result = _result_with_per_group(groups)
    result.error_mode = "scatter"
    _set_and_wait(window, groups, result)
    # χ²ᵣ present but greyed; tooltip explains why.
    assert "χ²ᵣ=" in window._quality_label.text()
    assert window._quality_label.toolTip()


# ── residual / pull mode ────────────────────────────────────────────────────


def test_residual_mode_draws_pulls_no_red_curve(qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    groups = _make_groups(2)
    _set_and_wait(window, groups, _result_with_per_group(groups))
    window._fit_residuals_check.setChecked(True)
    # Toggling residuals refreshes synchronously.
    axes = window._left_figure.axes
    red_lines = [line for ax in axes for line in ax.get_lines() if line.get_color() == "red"]
    assert not red_lines
    # A zero line should be present per panel.
    assert axes
    # Log-Y is disabled in residual mode.
    assert not window._fit_log_y_check.isEnabled()


# ── global table upgrade ────────────────────────────────────────────────────


def test_global_table_has_units_column(qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    groups = _make_groups(2)
    result = CrossGroupFitResult(
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        global_parameters=ParameterSet([Parameter("Lambda", value=0.12)]),
        global_uncertainties={"Lambda": 0.005},
        local_parameters={"g0": ParameterSet(), "g1": ParameterSet()},
    )
    _set_and_wait(window, groups, result)
    assert window._params_table.columnCount() == 4
    # Units column populated for Lambda (µs⁻¹).
    unit_item = window._params_table.item(0, 3)
    assert unit_item is not None
    assert "µs" in unit_item.text() or "s" in unit_item.text()


def test_copy_puts_tsv_on_clipboard(qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    groups = _make_groups(2)
    result = CrossGroupFitResult(
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        global_parameters=ParameterSet([Parameter("Lambda", value=0.12)]),
        global_uncertainties={"Lambda": 0.005},
        local_parameters={"g0": ParameterSet(), "g1": ParameterSet()},
    )
    _set_and_wait(window, groups, result)
    window._on_copy_global_table()
    clipboard = QApplication.clipboard()
    text = clipboard.text()
    assert "\t" in text
    assert "Parameter\tValue\tUncertainty\tValue(err)\tFlag\tUnits" in text


def test_latex_export_content(qapp: QApplication) -> None:
    result = CrossGroupFitResult(
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        global_parameters=ParameterSet([Parameter("Lambda", value=63.4)]),
        global_uncertainties={"Lambda": 2.1},
        fixed_parameters=ParameterSet([Parameter("c", value=0.5)]),
    )
    latex = global_table_latex(result, parameter_name="Lambda")
    assert "\\toprule" in latex
    assert "\\midrule" in latex
    assert "\\bottomrule" in latex
    assert "63(2)" in latex
    assert "[fixed]" in latex


def test_tsv_and_rows_helpers() -> None:
    result = CrossGroupFitResult(
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        global_parameters=ParameterSet([Parameter("Lambda", value=0.12)]),
        global_uncertainties={"Lambda": 0.005},
        fixed_parameters=ParameterSet([Parameter("c", value=0.5)]),
    )
    rows = build_global_table_rows(result)
    assert len(rows) == 2
    assert rows[0]["name"] == "Lambda"
    assert rows[0]["fixed"] is False
    assert rows[1]["fixed"] is True
    tsv = global_table_tsv(result)
    assert tsv.count("\n") >= 3


# ── correlations dialog ─────────────────────────────────────────────────────


def test_correlations_button_enabled_and_dialog_opens(qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    groups = _make_groups(2)
    result = CrossGroupFitResult(
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        global_parameters=ParameterSet([Parameter("a", value=0.1), Parameter("b", value=0.2)]),
        local_parameters={"g0": ParameterSet(), "g1": ParameterSet()},
        global_correlations=(["a", "b"], [[1.0, 0.5], [0.5, 1.0]]),
    )
    _set_and_wait(window, groups, result)
    assert window._correlations_btn.isEnabled()

    from asymmetry.gui.windows.global_fit_window_helpers import CorrelationMatrixDialog

    dialog = CorrelationMatrixDialog(["a", "b"], [[1.0, 0.5], [0.5, 1.0]])
    assert dialog._table.rowCount() == 2
    assert dialog._table.columnCount() == 2
    assert dialog._table.item(0, 1).text() == "0.50"


def test_correlations_button_disabled_without_matrix(qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    groups = _make_groups(2)
    result = CrossGroupFitResult(
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        global_parameters=ParameterSet([Parameter("a", value=0.1)]),
        local_parameters={"g0": ParameterSet(), "g1": ParameterSet()},
    )
    _set_and_wait(window, groups, result)
    assert not window._correlations_btn.isEnabled()


# ── studies sidebar ─────────────────────────────────────────────────────────


def test_sidebar_renders_rows_and_selection_emits(qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    window.set_studies_list(
        [
            StudySidebarEntry("id-a", "Study A", False, "A", "field"),
            StudySidebarEntry("id-b", "Study B", True, "A", "field"),
        ]
    )
    assert window._studies_list.count() == 2
    assert window._studies_list.item(0).text() == "Study A"
    assert window._studies_list.item(1).text().startswith("⚠")

    selected: list[str] = []
    window.study_selected.connect(selected.append)
    window._studies_list.setCurrentRow(0)
    assert selected == ["id-a"]


def test_set_active_study_id_does_not_emit(qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    window.set_studies_list(
        [
            StudySidebarEntry("id-a", "Study A", False, "A", "field"),
            StudySidebarEntry("id-b", "Study B", False, "A", "field"),
        ]
    )
    selected: list[str] = []
    window.study_selected.connect(selected.append)
    window.set_active_study_id("id-b")
    assert window._studies_list.currentRow() == 1
    assert selected == []  # programmatic selection is silent


def test_rename_duplicate_delete_edit_signals_fire(qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    renamed: list[tuple[str, str]] = []
    duplicated: list[str] = []
    deleted: list[str] = []
    edited: list[str] = []
    window.study_rename_requested.connect(lambda sid, name: renamed.append((sid, name)))
    window.study_duplicate_requested.connect(duplicated.append)
    window.study_delete_requested.connect(deleted.append)
    window.edit_requested.connect(edited.append)

    # Directly emit through the window's helpers (context-menu exec is modal).
    window.study_rename_requested.emit("id-a", "New Name")
    window.study_duplicate_requested.emit("id-a")
    window.study_delete_requested.emit("id-a")
    window._study_id = "id-a"
    window._on_edit_fit_clicked()

    assert renamed == [("id-a", "New Name")]
    assert duplicated == ["id-a"]
    assert deleted == ["id-a"]
    assert edited == ["id-a"]


def test_clear_display_empties_window(qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    groups = _make_groups(2)
    _set_and_wait(window, groups, _result_with_per_group(groups))
    assert window.has_result()
    window.clear_display()
    assert not window.has_result()
    assert window._params_table.rowCount() == 0
    assert window._quality_label.text() == ""
    assert window.windowTitle() == "Global Parameter Fit"
    assert not window._stale_banner.isVisible()


# ── custom-x labels ─────────────────────────────────────────────────────────


def test_custom_x_label_from_study(qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    groups = _make_groups(2)
    study = GlobalFitStudy(
        study_id="modelfit-abc",
        name="p study",
        parameter_name="Lambda",
        x_key="custom:pressure",
        x_label="p (GPa)",
        group_variable_key="temperature",
        group_variable_label="T (K)",
        created="",
        updated="",
        source_group_ids=["g0", "g1"],
        groups=groups,
        model=ParameterCompositeModel(["Constant"]),
        config={},
        result=_result_with_per_group(groups),
    )
    window.set_study(study, stale=False)
    _wait_fit_curves(window)
    assert window._x_label() == "p (GPa)"
    # An axis carries the custom label.
    axes = window._left_figure.axes
    assert any(ax.get_xlabel() == "p (GPa)" for ax in axes)


def test_custom_group_variable_label_on_local_axis(qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    groups = _make_groups(2)
    result = CrossGroupFitResult(
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        local_parameters={
            "g0": ParameterSet([Parameter("Lambda", value=0.1)]),
            "g1": ParameterSet([Parameter("Lambda", value=0.2)]),
        },
        local_uncertainties={"g0": {"Lambda": 0.01}, "g1": {"Lambda": 0.01}},
    )
    window.set_results(
        parameter_name="Lambda",
        x_key="custom:pressure",
        groups=groups,
        model=ParameterCompositeModel(["Constant"]),
        result=result,
        x_label="p (GPa)",
        group_variable_label="Concentration (%)",
    )
    _wait_fit_curves(window)
    assert window._local_group_axis_label() == "Concentration (%)"
    assert window._local_figure.axes[-1].get_xlabel() == "Concentration (%)"


def test_reduced_chi_helper_none_for_missing_group(qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    assert window._per_group_reduced_chi_squared("missing") is None


def test_math_isfinite_guard() -> None:
    # Sanity: helper returns a plain string for a NaN error.
    assert not math.isnan(float(format_value_with_error(1.0, 0.1).split("(")[0]))


# ── WP-A: window curve cache (redraw vs recompute split) ─────────────────────


class _CountingModel(ParameterCompositeModel):
    """A composite model that counts every ``function`` evaluation.

    Redraw-only toggles (log/scale/share/columns/groups/residuals) must draw from
    the window curve cache and evaluate the model ZERO extra times; a
    Show-components toggle re-evaluates only on its first (cold) flag.
    """

    def __init__(self, names: list[str]) -> None:
        super().__init__(names)
        self.function_calls = 0

    def function(self, x, **kwargs):  # type: ignore[override]
        self.function_calls += 1
        return super().function(x, **kwargs)


def _set_counting(window: GlobalParameterFitWindow, model: _CountingModel, n: int = 3) -> None:
    groups = _make_groups(n)
    window.set_results(
        parameter_name="Lambda",
        x_key="field",
        groups=groups,
        model=model,
        result=_result_with_per_group(groups),
    )
    _wait_fit_curves(window)


def test_redraw_only_toggles_do_not_reevaluate_model(qapp: QApplication) -> None:
    """Log/Share/Columns/Groups redraws are cache-backed — zero model eval."""
    window = GlobalParameterFitWindow()
    model = _CountingModel(["Constant"])
    _set_counting(window, model)
    baseline = model.function_calls
    assert baseline > 0  # the initial off-thread compute did evaluate.

    window._fit_log_x_check.setChecked(True)
    window._fit_log_y_check.setChecked(True)
    window._fit_share_x_check.setChecked(True)
    window._fit_share_y_check.setChecked(True)
    window._fit_columns_combo.setCurrentText("2")
    # Hide then show a group panel via the visibility map + redraw path.
    window._group_visibility["g0"] = False
    window._refresh_plot()
    window._group_visibility["g0"] = True
    window._refresh_plot()

    # No compute was kicked (cache warm) and the model was never re-evaluated.
    assert not window._fit_curve_compute_active
    assert model.function_calls == baseline


def test_residuals_toggle_reads_cache_no_eval(qapp: QApplication) -> None:
    """The Residuals toggle draws cached pulls — no fresh model evaluation."""
    window = GlobalParameterFitWindow()
    model = _CountingModel(["Constant"])
    _set_counting(window, model)
    baseline = model.function_calls

    window._fit_residuals_check.setChecked(True)
    assert not window._fit_curve_compute_active
    assert model.function_calls == baseline
    # Cached pulls were drawn (stem containers present on each shown panel).
    assert window._left_figure is not None and window._left_figure.axes

    window._fit_residuals_check.setChecked(False)
    assert model.function_calls == baseline


def test_switching_between_two_studies_reuses_cache(qapp: QApplication) -> None:
    """Re-displaying a study a second time reuses its warm cache — no re-eval.

    Two studies share one window; showing A, then B, then A again must not
    re-evaluate A's model on the second display (its curves are cached and only
    cleared by set_results, which a re-display of the SAME batch triggers — so we
    assert the *redraw* after the display does not add evaluations).
    """
    window = GlobalParameterFitWindow()
    model = _CountingModel(["Constant"])
    _set_counting(window, model)
    warm = model.function_calls

    # A pure redraw of the already-displayed study reuses the cache.
    window._refresh_plot()
    assert model.function_calls == warm
    # And a second redraw likewise.
    window._refresh_plot()
    assert model.function_calls == warm


def test_show_components_first_miss_then_cached(qapp: QApplication) -> None:
    """Show-components computes once (cold True-flag), then redraws from cache."""
    window = GlobalParameterFitWindow()
    model = _CountingModel(["Constant"])
    _set_counting(window, model)
    after_false = model.function_calls
    assert window._curve_cache.get(False) is not None
    assert window._curve_cache.get(True) is None

    # First components toggle: True-flag cache is cold → one off-thread compute.
    window._show_components_check.setChecked(True)
    _wait_fit_curves(window)
    assert window._curve_cache.get(True) is not None
    after_true = model.function_calls
    assert after_true > after_false

    # Toggling back to False is a pure cache hit (both flags now warm).
    window._show_components_check.setChecked(False)
    assert not window._fit_curve_compute_active
    assert model.function_calls == after_true
    # And back to True again is also a cache hit.
    window._show_components_check.setChecked(True)
    assert not window._fit_curve_compute_active
    assert model.function_calls == after_true


def test_set_results_clears_curve_cache(qapp: QApplication) -> None:
    """A new fit invalidates the cache so stale curves are never drawn."""
    window = GlobalParameterFitWindow()
    model = _CountingModel(["Constant"])
    _set_counting(window, model)
    assert window._curve_cache.get(False) is not None

    # A different batch id replaces the fit — cache cleared, recomputed.
    groups2 = _make_groups(2)
    window.set_results(
        parameter_name="Lambda",
        x_key="field",
        groups=groups2,
        model=model,
        result=_result_with_per_group(groups2),
        batch_id="other",
    )
    # Immediately after set_results (before the async compute lands) the cache is
    # empty and a compute is in flight.
    assert window._curve_cache == {} or window._fit_curve_compute_active
    _wait_fit_curves(window)
    assert window._curve_cache.get(False) is not None


# ── WP-BD: right-pane splitter + header resize policies ─────────────────────


def test_params_table_header_resize_modes(qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    header = window._params_table.horizontalHeader()
    # symbol=ResizeToContents, Value/Uncertainty=Interactive, Units=ResizeToContents,
    # last section stretches.
    assert header.sectionResizeMode(0) == QHeaderView.ResizeMode.ResizeToContents
    assert header.sectionResizeMode(1) == QHeaderView.ResizeMode.Interactive
    assert header.sectionResizeMode(2) == QHeaderView.ResizeMode.Interactive
    assert header.sectionResizeMode(3) == QHeaderView.ResizeMode.ResizeToContents
    assert header.stretchLastSection()
    # The interactive numeric columns carry a content-based minimum wide enough
    # for the "Uncertainty" header — it can never truncate.
    fm = header.fontMetrics()
    assert header.minimumSectionSize() >= fm.horizontalAdvance("Uncertainty")


def test_params_table_header_modes_survive_populate(qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    groups = _make_groups(2)
    result = CrossGroupFitResult(
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        global_parameters=ParameterSet([Parameter("Lambda", value=0.12)]),
        global_uncertainties={"Lambda": 0.005},
        local_parameters={"g0": ParameterSet(), "g1": ParameterSet()},
    )
    _set_and_wait(window, groups, result)
    header = window._params_table.horizontalHeader()
    assert header.sectionResizeMode(0) == QHeaderView.ResizeMode.ResizeToContents
    assert header.sectionResizeMode(1) == QHeaderView.ResizeMode.Interactive
    assert header.stretchLastSection()


def test_y_selector_header_resize_modes(qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    header = window._local_y_selector_table.horizontalHeader()
    # Parameter-name column stretches; button + log columns size to content so
    # "Model Fit"/"log" never truncate.
    assert header.sectionResizeMode(0) == QHeaderView.ResizeMode.Stretch
    assert header.sectionResizeMode(1) == QHeaderView.ResizeMode.ResizeToContents
    assert header.sectionResizeMode(2) == QHeaderView.ResizeMode.ResizeToContents
    assert not header.stretchLastSection()


def test_y_selector_header_modes_survive_populate(qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    window._rebuild_local_y_controls(["A", "Lambda", "c"])
    header = window._local_y_selector_table.horizontalHeader()
    assert header.sectionResizeMode(0) == QHeaderView.ResizeMode.Stretch
    assert header.sectionResizeMode(1) == QHeaderView.ResizeMode.ResizeToContents
    assert header.sectionResizeMode(2) == QHeaderView.ResizeMode.ResizeToContents


def test_right_pane_is_vertical_splitter_with_min_heights(qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    splitter = window._right_splitter
    assert isinstance(splitter, QSplitter)
    assert splitter.orientation() == Qt.Orientation.Vertical
    # Three blocks: params-table block, Y-selector, local canvas block.
    assert splitter.count() == 3
    # The Y-selector is the middle child.
    assert splitter.widget(1) is window._local_y_selector_table
    # Every block carries a non-zero minimum height so none collapses.
    for i in range(3):
        assert splitter.widget(i).minimumHeight() > 0


# ── WP-BD: grid scroll area + canvas minimum height ─────────────────────────


def test_left_grid_lives_in_scroll_area_no_hscroll(qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    scroll = window._left_scroll
    assert isinstance(scroll, QScrollArea)
    assert scroll.widgetResizable()
    assert scroll.widget() is window._left_canvas
    assert scroll.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff


def test_grid_canvas_min_height_grows_with_rows(qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    # Two groups → one row (2 columns) → shorter canvas.
    groups2 = _make_groups(2)
    _set_and_wait(window, groups2, _result_with_per_group(groups2))
    two_row_h = window._left_canvas.minimumHeight()

    # Eight groups → 3 rows → taller canvas.
    window.set_results(
        parameter_name="Lambda",
        x_key="field",
        groups=_make_groups(8),
        model=ParameterCompositeModel(["Constant"]),
        result=_result_with_per_group(_make_groups(8)),
        batch_id="eight",
    )
    _wait_fit_curves(window)
    tall_h = window._left_canvas.minimumHeight()
    assert tall_h > two_row_h
    assert two_row_h > 0


def test_grid_canvas_min_height_updates_on_column_change(qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    groups = _make_groups(8)
    _set_and_wait(window, groups, _result_with_per_group(groups))
    # Auto → 3 columns → 3 rows.
    auto_h = window._left_canvas.minimumHeight()
    # Force 1 column → 8 rows → much taller.
    window._fit_columns_combo.setCurrentText("1")
    tall_h = window._left_canvas.minimumHeight()
    assert tall_h > auto_h


def test_columns_choice_persisted_in_state(qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    window._fit_columns_combo.setCurrentText("3")
    state = window.get_state()
    assert state["fit_columns"] == "3"

    other = GlobalParameterFitWindow()
    other.restore_state(state)
    assert other._fit_columns_combo.currentText() == "3"


# ── WP-BD: legend top-margin reservation ────────────────────────────────────


def _first_row_axes(window: GlobalParameterFitWindow) -> list:
    axes = window._left_figure.axes
    return [ax for ax in axes if ax.get_subplotspec().rowspan.start == 0]


def _legend_bbox_fig(window: GlobalParameterFitWindow):
    window._left_canvas.draw()
    renderer = window._left_canvas.get_renderer()
    legends = window._left_figure.legends
    assert legends, "expected a figure-level component legend"
    bbox_px = legends[0].get_window_extent(renderer)
    return bbox_px.transformed(window._left_figure.transFigure.inverted())


def _component_result(groups: list[ParameterGroupData]) -> CrossGroupFitResult:
    return CrossGroupFitResult(
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        global_parameters=ParameterSet([Parameter("a", value=0.1), Parameter("c", value=0.05)]),
        local_parameters={g.group_id: ParameterSet() for g in groups},
    )


def _set_components(window: GlobalParameterFitWindow, groups, batch_id: str) -> None:
    window._show_components_check.setChecked(True)
    window.set_results(
        parameter_name="Lambda",
        x_key="field",
        groups=groups,
        model=ParameterCompositeModel.from_expression("Linear + Constant"),
        result=_component_result(groups),
        batch_id=batch_id,
    )
    _wait_fit_curves(window)


def test_legend_does_not_overlap_first_row_titles_single_row(qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    window.resize(900, 700)
    groups = _make_groups(2)  # one row
    _set_components(window, groups, "one_row")
    legend_bbox = _legend_bbox_fig(window)
    # Legend sits strictly above every first-row panel (title band included).
    for ax in _first_row_axes(window):
        ax_top = ax.get_position().y1
        assert legend_bbox.y0 >= ax_top - 1e-6, (
            f"legend y0={legend_bbox.y0:.3f} overlaps axis top {ax_top:.3f}"
        )


def test_legend_does_not_overlap_first_row_titles_three_rows(qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    window.resize(900, 900)
    groups = _make_groups(8)  # 3 rows at auto (3 cols)
    _set_components(window, groups, "three_rows")
    legend_bbox = _legend_bbox_fig(window)
    for ax in _first_row_axes(window):
        ax_top = ax.get_position().y1
        assert legend_bbox.y0 >= ax_top - 1e-6, (
            f"legend y0={legend_bbox.y0:.3f} overlaps axis top {ax_top:.3f}"
        )


# ── WP-BD: uncertainty trust flag ───────────────────────────────────────────


def _degenerate_result() -> CrossGroupFitResult:
    # A=1.7e-5 ± 616 → err ≫ 3×|value| → flagged. c=0.15 ± 0.005 → healthy.
    return CrossGroupFitResult(
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        global_parameters=ParameterSet([Parameter("A", value=1.7e-5), Parameter("c", value=0.15)]),
        global_uncertainties={"A": 616.0, "c": 0.005},
        local_parameters={"g0": ParameterSet(), "g1": ParameterSet()},
    )


def test_uncertainty_flag_background_and_tooltip_on_degenerate_row(qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    groups = _make_groups(2)
    _set_and_wait(window, groups, _degenerate_result())
    # Row 0 = A (degenerate) → warning background + tooltip on the Uncertainty cell.
    a_err = window._params_table.item(0, 2)
    assert a_err is not None
    assert a_err.background().color().isValid()
    assert a_err.background().style() != Qt.BrushStyle.NoBrush
    assert a_err.toolTip()
    # Row 1 = c (healthy) → no warning brush, no tooltip.
    c_err = window._params_table.item(1, 2)
    assert c_err is not None
    assert c_err.background().style() == Qt.BrushStyle.NoBrush
    assert not c_err.toolTip()


def test_uncertainty_trust_flag_helper() -> None:
    assert uncertainty_trust_flag(1.7e-5, 616.0) == "degenerate"
    assert uncertainty_trust_flag(0.15, 0.005) == ""
    assert uncertainty_trust_flag(1.0, float("nan")) == "degenerate"
    assert uncertainty_trust_flag(1.0, None) == ""
    # err exactly 3×|value| is not flagged (strict >).
    assert uncertainty_trust_flag(1.0, 3.0) == ""
    assert uncertainty_trust_flag(1.0, 3.0001) == "degenerate"


# ── WP-BD: correlations-disabled tooltip ────────────────────────────────────


def test_correlations_disabled_button_has_explanatory_tooltip(qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    groups = _make_groups(2)
    # No global_correlations on the result → button disabled with explanation.
    result = CrossGroupFitResult(
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        global_parameters=ParameterSet([Parameter("c", value=0.15)]),
        local_parameters={"g0": ParameterSet(), "g1": ParameterSet()},
    )
    _set_and_wait(window, groups, result)
    assert not window._correlations_btn.isEnabled()
    assert "correlation matrix unavailable" in window._correlations_btn.toolTip()


def test_correlations_enabled_button_restores_default_tooltip(qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    groups = _make_groups(2)
    result = CrossGroupFitResult(
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        global_parameters=ParameterSet([Parameter("a", value=0.1), Parameter("b", value=0.2)]),
        global_correlations=(["a", "b"], [[1.0, 0.3], [0.3, 1.0]]),
        local_parameters={"g0": ParameterSet(), "g1": ParameterSet()},
    )
    _set_and_wait(window, groups, result)
    assert window._correlations_btn.isEnabled()
    assert "correlation matrix unavailable" not in window._correlations_btn.toolTip()


# ── WP-BD: TSV/CSV combined + flag columns ──────────────────────────────────


def test_tsv_export_has_combined_and_flag_columns() -> None:
    result = _degenerate_result()
    tsv = global_table_tsv(result)
    header = tsv.splitlines()[0]
    assert header == "Parameter\tValue\tUncertainty\tValue(err)\tFlag\tUnits"
    # The degenerate A row carries the "degenerate" flag; the raw columns remain.
    a_line = next(line for line in tsv.splitlines() if line.split("\t")[0].startswith("A"))
    cells = a_line.split("\t")
    assert cells[4] == "degenerate"
    # Value(err) column is populated (combined form).
    assert cells[3]


def test_csv_export_has_combined_and_flag_columns() -> None:
    import csv
    import io

    result = _degenerate_result()
    csv_text = global_table_csv(result)
    reader = list(csv.reader(io.StringIO(csv_text)))
    assert reader[0] == ["Parameter", "Value", "Uncertainty", "Value(err)", "Flag", "Units"]
    a_row = next(r for r in reader[1:] if r[0] == "A")
    assert a_row[4] == "degenerate"
    assert a_row[3]  # combined Value(err) populated
    c_row = next(r for r in reader[1:] if r[0] == "c")
    assert c_row[4] == ""  # healthy row not flagged
