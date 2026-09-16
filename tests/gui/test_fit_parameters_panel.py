"""Tests for FitParametersPanel GLE export helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

pyside6 = pytest.importorskip("PySide6")
from PySide6.QtCore import QEvent, QPoint, Qt  # type: ignore
from PySide6.QtWidgets import (  # type: ignore
    QApplication,
    QDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
)

from asymmetry.core.fitting.axis_transforms import AxisTransform
from asymmetry.core.fitting.composite_parameters import CompositeParameterDefinition
from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.parameter_models import (
    CrossGroupFitResult,
    ErrorMode,
    ModelFitRange,
    ParameterCompositeModel,
    ParameterGroupData,
    ParameterModelFit,
    ParameterModelFitResult,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.gui.export_paths import resolve_gle_export_paths
from asymmetry.gui.mainwindow import _inspector_scroll_area
from asymmetry.gui.panels.fit_parameters_panel import (
    FitParametersPanel,
    _FitRow,
    _format_gle_label,
    _format_gle_legend_label,
    _format_plot_label,
    _format_plot_legend_label,
    _GroupFitData,
)
from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.widgets import CONTEXT_CHIP_OBJECT_NAME
from asymmetry.gui.utils import gle_export
from asymmetry.gui.utils.formatting import format_param_label as _format_param_label
from asymmetry.gui.widgets.flow_layout import FlowLayout
from asymmetry.gui.windows.fit_results_window import FitResultsWindow
from tests._qt_helpers import wait_for
from tests.gui._trend_panel import axes_for, card, select_params


def _active_linear_fit(param: str, x_key: str = "field") -> ParameterModelFit:
    """An active model fit with one solved range, for trend-overlay tests."""
    model = ParameterCompositeModel(["Linear"])
    params = ParameterSet([Parameter("m", value=0.001), Parameter("b", value=0.2)])
    # χ² = 4.5 over ν = 5 − 2 → χ²ᵣ = 1.5, a "good" verdict at 95 % confidence.
    result = ParameterModelFitResult(
        success=True,
        parameters=params,
        chi_squared=4.5,
        reduced_chi_squared=1.5,
        n_points=5,
        uncertainties={"m": 0.0005, "b": 0.01},
    )
    return ParameterModelFit(
        parameter_name=param,
        x_key=x_key,
        active=True,
        ranges=[
            ModelFitRange(x_min=100.0, x_max=200.0, model=model, parameters=params, result=result)
        ],
    )


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    """Ensure a QApplication exists for widget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def panel(qapp: QApplication) -> FitParametersPanel:
    """Create a panel with predictable test data."""
    w = FitParametersPanel()
    w._rows = [
        _FitRow(
            run_number=2,
            run_label="2",
            field=200.0,
            temperature=10.0,
            values={"A0": 0.22, "Lambda": 0.12},
            errors={"A0": 0.02, "Lambda": 0.01},
        ),
        _FitRow(
            run_number=1,
            run_label="1",
            field=100.0,
            temperature=10.0,
            values={"A0": 0.20, "Lambda": 0.10},
            errors={"A0": 0.01, "Lambda": 0.01},
        ),
    ]
    w._varying_params = ["A0", "Lambda"]
    w._inferred_x_key = "field"
    w._x_combo.setCurrentText("Auto")
    w._rebuild_y_controls()
    w._update_row_dependent_controls()
    return w


def test_gle_columns_for_param(panel: FitParametersPanel) -> None:
    assert panel._gle_columns_for_param("A0") == (4, 5)
    assert panel._gle_columns_for_param("Lambda") == (6, 7)
    assert panel._gle_columns_for_param("missing") is None


def test_write_gle_data_file_contains_column_map_and_sorted_rows(
    panel: FitParametersPanel, tmp_path: Path
) -> None:
    panel._global_params = ParameterSet([Parameter("Lambda", value=0.5)])

    out = tmp_path / "fit_parameters.dat"
    panel._write_gle_data_file(out)
    text = out.read_text(encoding="utf-8")

    assert "! Column map:" in text
    assert "!   Lambda (µs⁻¹) = 0.5" in text
    assert "!   c 1 = Run" in text
    assert "!   c 2 = B_field(G)" in text
    assert "!   c 3 = Temperature(K)" in text
    assert "!   c 4 = A0 (%)" in text
    assert "!   c 5 = err_A0 (%)" in text
    assert "!   c 6 = Lambda (µs⁻¹)" in text
    assert "!   c 7 = err_Lambda (µs⁻¹)" in text

    data_lines = [ln for ln in text.splitlines() if ln and not ln.startswith("!")]
    assert len(data_lines) == 2

    # Rows should be sorted by inferred x-axis (field): 100 before 200.
    assert float(data_lines[0].split()[0]) == pytest.approx(1.0)
    assert float(data_lines[0].split()[1]) == pytest.approx(100.0)
    assert float(data_lines[1].split()[0]) == pytest.approx(2.0)
    assert float(data_lines[1].split()[1]) == pytest.approx(200.0)


def test_write_gle_data_file_includes_combined_run_mapping_comments(
    tmp_path: Path, qapp: QApplication
) -> None:
    panel = FitParametersPanel()
    panel._rows = [
        _FitRow(
            run_number=-1,
            run_label="3039 + 3040",
            field=150.0,
            temperature=10.0,
            values={"A0": 0.21},
            errors={"A0": 0.01},
            combined_from=[3039, 3040],
        )
    ]
    panel._varying_params = ["A0"]
    panel._inferred_x_key = "run"
    panel._x_combo.setCurrentText("Run")

    out = tmp_path / "fit_parameters.dat"
    panel._write_gle_data_file(out)
    text = out.read_text(encoding="utf-8")

    assert "! Combined run mapping:" in text
    assert "!   -1 = 3039 + 3040" in text

    data_lines = [ln for ln in text.splitlines() if ln and not ln.startswith("!")]
    assert len(data_lines) == 1
    assert float(data_lines[0].split()[0]) == pytest.approx(-1.0)
    assert float(data_lines[0].split()[1]) == pytest.approx(150.0)
    assert float(data_lines[0].split()[2]) == pytest.approx(10.0)


def test_export_tsv_headers_include_units(
    panel: FitParametersPanel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    panel._refresh_table()
    out = tmp_path / "fit_parameters.tsv"

    monkeypatch.setattr(
        "asymmetry.gui.panels.fit_parameters_panel.QFileDialog.getSaveFileName",
        lambda *_a, **_k: (str(out), "TSV files (*.tsv)"),
    )

    panel._export_tsv()

    first_line = out.read_text(encoding="utf-8").splitlines()[0]
    assert first_line == (
        "Run\tB (G)\tT (K)\tA0 (%)\terr_A0 (%)\tLambda (µs⁻¹)\terr_Lambda (µs⁻¹)\t"
        "reduced_chi2\tchi2"
    )


def test_export_tsv_golden_header_and_data_rows(
    panel: FitParametersPanel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full golden string for header + both data rows (Phase 0 characterization).

    Pins the exact tab-separated content — including value/uncertainty
    formatting sourced from the table's displayed text and the trailing
    reduced_chi2/chi2 columns — so Phase 1c's shared-TSV-writer extraction
    can diff its output against this literal string.
    """
    panel._refresh_table()
    out = tmp_path / "fit_parameters.tsv"

    monkeypatch.setattr(
        "asymmetry.gui.panels.fit_parameters_panel.QFileDialog.getSaveFileName",
        lambda *_a, **_k: (str(out), "TSV files (*.tsv)"),
    )

    panel._export_tsv()

    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0] == (
        "Run\tB (G)\tT (K)\tA0 (%)\terr_A0 (%)\tLambda (µs⁻¹)\terr_Lambda (µs⁻¹)\t"
        "reduced_chi2\tchi2"
    )
    # panel fixture rows are sorted by run number ascending: run 1 (field=100)
    # then run 2 (field=200); neither row carries a chi-squared value.
    data_lines = lines[1:]
    assert len(data_lines) == 2
    assert data_lines[0].split("\t")[:7] == [
        "1",
        "100",
        "10",
        "0.2",
        "0.01",
        "0.1",
        "0.01",
    ]
    assert data_lines[0].split("\t")[7:] == ["", ""]
    assert data_lines[1].split("\t")[:7] == [
        "2",
        "200",
        "10",
        "0.22",
        "0.02",
        "0.12",
        "0.01",
    ]
    assert data_lines[1].split("\t")[7:] == ["", ""]


def test_export_tsv_includes_custom_abscissa_column(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A free-text (custom/Angle) x-axis appends its own trailing TSV column.

    ``_export_abscissa_column`` is the single source of this trailing column
    for the table, TSV, and GLE exports; this pins its TSV contribution
    specifically (header label + per-row value) ahead of Phase 1c.
    """
    panel = FitParametersPanel()
    panel._rows = [
        _FitRow(
            run_number=1,
            run_label="1",
            field=100.0,
            temperature=5.0,
            values={"A0": 0.20},
            errors={"A0": 0.01},
            custom_values={"angle_deg": "45.5"},
        ),
    ]
    panel._varying_params = ["A0"]
    panel.set_angle_x_field(("Angle", "angle_deg"))
    panel._x_combo.setCurrentText("Angle")
    panel._refresh_table()

    out = tmp_path / "angle.tsv"
    monkeypatch.setattr(
        "asymmetry.gui.panels.fit_parameters_panel.QFileDialog.getSaveFileName",
        lambda *_a, **_k: (str(out), "TSV files (*.tsv)"),
    )
    panel._export_tsv()

    lines = out.read_text(encoding="utf-8").splitlines()
    header = lines[0]
    assert header.startswith("Run\tB (G)\tT (K)\tA0 (%)\terr_A0 (%)\t")
    assert header.endswith("\tAngle\treduced_chi2\tchi2")
    data_line = lines[1]
    assert data_line.split("\t")[-3] == "45.5"


def test_refresh_table_uses_run_label_for_combined_rows(qapp: QApplication) -> None:
    panel = FitParametersPanel()
    panel._rows = [
        _FitRow(
            run_number=-1,
            run_label="3039 + 3040",
            field=150.0,
            temperature=10.0,
            values={"A0": 0.21},
            errors={"A0": 0.01},
            combined_from=[3039, 3040],
        )
    ]
    panel._varying_params = ["A0"]
    panel._refresh_table()

    run_item = panel._table.item(0, 0)
    assert run_item is not None
    assert run_item.text() == "3039 + 3040"


def test_tsv_export_chi2_tracks_row_not_label(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Two runs that share a run_label must each export their OWN chi-squared:
    # a label-keyed join would collapse them and misattribute the values.
    panel = FitParametersPanel()
    panel._rows = [
        _FitRow(
            run_number=1,
            run_label="dup",
            field=100.0,
            temperature=5.0,
            values={"A0": 0.21},
            errors={"A0": 0.01},
            chi_squared=11.0,
            reduced_chi_squared=1.1,
        ),
        _FitRow(
            run_number=2,
            run_label="dup",
            field=100.0,
            temperature=5.0,
            values={"A0": 0.22},
            errors={"A0": 0.01},
            chi_squared=22.0,
            reduced_chi_squared=2.2,
        ),
    ]
    panel._varying_params = ["A0"]
    panel._x_combo.setCurrentText("Run")
    panel._refresh_table()

    out = tmp_path / "dup.tsv"
    monkeypatch.setattr(
        "asymmetry.gui.panels.fit_parameters_panel.QFileDialog.getSaveFileName",
        lambda *_a, **_k: (str(out), "TSV files (*.tsv)"),
    )
    panel._export_tsv()
    data_lines = [
        ln
        for ln in out.read_text(encoding="utf-8").splitlines()
        if not ln.startswith(("#", "Run\t"))
    ]
    assert len(data_lines) == 2
    # Sorted by run number: row 0 is run 1 (1.1 / 11), row 1 is run 2 (2.2 / 22).
    assert data_lines[0].split("\t")[-2:] == ["1.1", "11"]
    assert data_lines[1].split("\t")[-2:] == ["2.2", "22"]


def test_exports_embed_model_global_params_and_chi2(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    panel = FitParametersPanel()
    panel._rows = [
        _FitRow(
            run_number=3001,
            run_label="3001",
            field=100.0,
            temperature=5.0,
            values={"A0": 0.21, "Lambda": 0.5},
            errors={"A0": 0.01, "Lambda": 0.02},
            model_name="A0 exp(-Lambda t)",
            chi_squared=42.0,
            reduced_chi_squared=1.05,
        )
    ]
    panel._varying_params = ["Lambda"]
    panel._global_params = ParameterSet([Parameter("A0", value=0.21)])
    panel._global_param_uncertainties = {"A0": 0.004}
    panel._refresh_table()

    # TSV: provenance comment header + per-run chi-squared columns.
    tsv_out = tmp_path / "fit_parameters.tsv"
    monkeypatch.setattr(
        "asymmetry.gui.panels.fit_parameters_panel.QFileDialog.getSaveFileName",
        lambda *_a, **_k: (str(tsv_out), "TSV files (*.tsv)"),
    )
    panel._export_tsv()
    tsv_text = tsv_out.read_text(encoding="utf-8")
    assert "# Model: A0 exp(-Lambda t)" in tsv_text
    assert "# Global fitting parameters:" in tsv_text
    assert "A0" in tsv_text and "0.21" in tsv_text
    header_line = next(ln for ln in tsv_text.splitlines() if ln.startswith("Run\t"))
    assert header_line.endswith("reduced_chi2\tchi2")
    data_line = tsv_text.splitlines()[-1]
    assert "1.05" in data_line and "42" in data_line

    # GLE .dat: model line + global params + trailing chi-squared columns.
    dat_out = tmp_path / "fit_parameters.dat"
    panel._write_gle_data_file(dat_out)
    dat_text = dat_out.read_text(encoding="utf-8")
    assert "! Model: A0 exp(-Lambda t)" in dat_text
    assert "! Global fitting parameters:" in dat_text
    assert "reduced_chi2" in dat_text and "chi2" in dat_text
    data_rows = [ln for ln in dat_text.splitlines() if ln and not ln.startswith("!")]
    assert len(data_rows) == 1
    cols = data_rows[0].split()
    assert float(cols[-2]) == pytest.approx(1.05)
    assert float(cols[-1]) == pytest.approx(42.0)


def test_panel_layout_is_rail_over_plot_area(panel: FitParametersPanel) -> None:
    """The dock scrolls the panel, so it owns no scroll area or splitter itself."""
    assert panel.findChild(QScrollArea) is None
    assert panel.findChild(QSplitter) is None
    assert panel._plot_pages.count() == 2
    assert panel._plot_pages.widget(0) is panel._card_stack
    assert panel._plot_pages.widget(1).isAncestorOf(panel._canvas)
    # The Overlay canvas lives on the second page; Subplots is the default.
    assert panel._plot_mode() == "Subplots"
    assert panel._plot_pages.currentIndex() == 0


def test_more_menu_actions_call_the_export_paths(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[object] = []
    # The actions bind their slots at construction, so patch the class first.
    monkeypatch.setattr(FitParametersPanel, "_export_tsv", lambda _self: calls.append("tsv"))
    monkeypatch.setattr(FitParametersPanel, "_export_gle", lambda _self, fmt: calls.append(fmt))
    panel = FitParametersPanel()
    panel._rows = [_FitRow(1, "1", 100.0, 10.0, {"A0": 0.2}, {"A0": 0.01})]
    panel._update_row_dependent_controls()
    texts = [action.text() for action in panel._more_menu.actions() if not action.isSeparator()]
    assert texts == [
        "Export TSV…",
        "Export GLE (PDF)…",
        "Export GLE (EPS)…",
        "Show components",
        "Knight shift window…",
    ]
    for action in panel._export_actions:
        action.trigger()
    assert calls == ["tsv", "pdf", "eps"]


def test_add_menu_gates_on_what_the_series_offers(panel: FitParametersPanel) -> None:
    panel._update_row_dependent_controls()
    assert panel._derived_action.isEnabled()
    # No Knight-convertible component on this series → no rail shortcut.
    assert not panel._add_knight_action.isVisible()
    panel._knight_observables = {"Nu": "frequency"}
    panel._update_row_dependent_controls()
    assert panel._add_knight_action.isVisible()


def test_chips_build_cards_in_card_order(panel: FitParametersPanel) -> None:
    panel._rebuild_y_controls()
    assert sorted(panel._y_chips) == ["A0", "Lambda"]

    panel._card_order = ["Lambda", "A0"]
    select_params(panel, ["A0", "Lambda"])
    assert [c.name for c in panel._card_stack.cards()] == ["Lambda", "A0"]
    assert panel._selected_y_parameters() == ["Lambda", "A0"]
    # Each card keeps the fit/log controls the panel's per-parameter code drives.
    assert panel._y_controls["A0"].fit_button is card(panel, "A0").fit_button
    assert panel._y_controls["A0"].log is card(panel, "A0").log_check


def test_unchecking_a_chip_removes_its_card(panel: FitParametersPanel) -> None:
    select_params(panel, ["A0", "Lambda"])
    assert len(panel._card_stack.cards()) == 2
    panel._y_chips["Lambda"].setChecked(False)
    assert [c.name for c in panel._card_stack.cards()] == ["A0"]
    assert "Lambda" not in panel._y_controls
    # Re-checking builds it back.
    panel._y_chips["Lambda"].setChecked(True)
    assert [c.name for c in panel._card_stack.cards()] == ["A0", "Lambda"]


def test_subplots_draws_axes_per_expanded_card_and_a_sparkline_when_collapsed(
    panel: FitParametersPanel,
) -> None:
    select_params(panel, ["A0", "Lambda"])
    assert len(axes_for(panel, "A0").get_figure().axes) == 1
    assert len(axes_for(panel, "Lambda").get_figure().axes) == 1
    # The shared Overlay figure stays untouched in Subplots mode.
    assert panel._figure.axes == []

    card(panel, "Lambda").set_expanded(False)
    panel._refresh_plot()  # a collapse redraws the stack behind the debounce timer
    assert card(panel, "Lambda").figure.axes == []
    assert not card(panel, "Lambda")._sparkline_label.pixmap().isNull()
    assert panel._collapsed_params == {"Lambda"}


def test_overlay_mode_draws_into_the_shared_figure(panel: FitParametersPanel) -> None:
    panel._overlay_button.setChecked(True)
    select_params(panel, ["A0"])
    assert panel._plot_pages.currentIndex() == 1
    assert len(panel._figure.axes) == 1
    assert axes_for(panel, "A0") is panel._figure.axes[0]
    assert card(panel, "A0").figure.axes == []


def test_focus_expands_one_card_and_restores_on_exit(panel: FitParametersPanel) -> None:
    select_params(panel, ["A0", "Lambda"])
    card(panel, "Lambda").set_expanded(False)

    panel._card_stack.set_focus("A0")
    assert card(panel, "A0").is_expanded()
    assert not card(panel, "Lambda").is_expanded()
    assert card(panel, "A0").is_focused()

    panel._card_stack.set_focus(None)
    assert card(panel, "A0").is_expanded()
    assert not card(panel, "Lambda").is_expanded()
    assert panel._collapsed_params == {"Lambda"}


def test_expanded_cards_share_the_dock_height_without_scrolling(
    panel: FitParametersPanel, qapp: QApplication
) -> None:
    """Two cards must divide the dock's viewport, not overflow it.

    The stack used to pass each card's figure-sized height preference up through
    the panel's height-for-width chain, so the dock's scroll area grew the panel
    past its own viewport and the whole panel scrolled while the cards sat at
    their preferred height. 376 × 585 is the Parameters dock's viewport on a
    13-inch MacBook at UI scale 1.
    """
    area = _inspector_scroll_area(panel)
    area.resize(376, 585)
    area.show()
    select_params(panel, ["A0", "Lambda"])
    for _ in range(4):
        qapp.processEvents()

    assert not area.verticalScrollBar().isVisible()
    assert panel.height() <= area.viewport().height()
    for name in ("A0", "Lambda"):
        assert card(panel, name).height() > card(panel, name).minimumSizeHint().height()
    area.deleteLater()


def test_card_fit_button_labels_track_the_fit_state(panel: FitParametersPanel) -> None:
    select_params(panel, ["A0"])
    assert card(panel, "A0").fit_button.text() == "Fit…"

    panel._model_fits["A0"] = _active_linear_fit("A0")
    panel._model_fit_transform_sig["A0"] = panel._transform_signature("A0")
    panel._refresh_model_fit_button_labels()
    assert card(panel, "A0").fit_button.text() == "Fit ✓"
    assert card(panel, "A0").fit_button.toolTip() == "Model fit active"

    panel._set_x_transform(AxisTransform.preset("square"))
    assert card(panel, "A0").fit_button.text() == "Fit ⚠"
    assert "different axis transform" in card(panel, "A0").fit_button.toolTip()


def test_card_result_chip_appears_with_the_fit_and_goes_with_it(
    panel: FitParametersPanel,
) -> None:
    select_params(panel, ["A0"])
    assert card(panel, "A0").result_chip.text() == ""

    panel._model_fits["A0"] = _active_linear_fit("A0")
    panel._refresh_model_fit_button_labels()

    chip = card(panel, "A0").result_chip
    assert chip.text() == "χ²ᵣ 1.5"
    assert chip.isVisibleTo(card(panel, "A0"))
    # χ² of 3 over ν = 4 − 2 reads "good" at the default 95 % confidence.
    assert tokens.OK in chip.styleSheet()
    tooltip = chip.toolTip().splitlines()
    assert tooltip[0].startswith("χ²ᵣ 1.5 · good fit (band ")
    assert tooltip[0].endswith(" %)")
    assert tooltip[1:3] == ["m = 0.0010(5)", "b = 0.200(10)"]
    assert tooltip[-1] == "Click for the fit results."

    del panel._model_fits["A0"]
    panel._refresh_model_fit_button_labels()
    assert chip.text() == ""
    assert not chip.isVisibleTo(card(panel, "A0"))


def test_card_result_chip_is_neutral_without_a_verdict(panel: FitParametersPanel) -> None:
    select_params(panel, ["A0"])
    fit = _active_linear_fit("A0")
    # Scatter-estimated errors force χ²ᵣ toward 1, so it carries no verdict.
    fit.ranges[0].result.error_mode = ErrorMode.SCATTER.value
    panel._model_fits["A0"] = fit
    panel._refresh_model_fit_button_labels()

    chip = card(panel, "A0").result_chip
    assert chip.toolTip().startswith("χ²ᵣ 1.5 · no verdict")
    assert tokens.TEXT_MUTED in chip.styleSheet()


def test_card_result_chip_opens_and_reuses_one_results_window(
    panel: FitParametersPanel,
) -> None:
    select_params(panel, ["A0"])
    panel._model_fits["A0"] = _active_linear_fit("A0")
    panel._refresh_model_fit_button_labels()

    card(panel, "A0").result_chip.click()
    window = panel._fit_results_windows["A0"]

    assert isinstance(window, FitResultsWindow)
    assert window.windowTitle() == f"Fit results — {_format_param_label('A0')}"
    table = window.findChild(QTableWidget)
    assert table.rowCount() == 2
    assert [table.item(row, 0).text() for row in range(2)] == ["m", "b"]
    assert [table.item(row, 1).text() for row in range(2)] == ["0.0010(5)", "0.200(10)"]

    card(panel, "A0").result_chip.click()
    assert panel._fit_results_windows["A0"] is window


def test_results_window_copy_puts_the_table_on_the_clipboard(
    panel: FitParametersPanel, qapp: QApplication
) -> None:
    select_params(panel, ["A0"])
    panel._model_fits["A0"] = _active_linear_fit("A0")
    panel._refresh_model_fit_button_labels()
    card(panel, "A0").result_chip.click()

    panel._fit_results_windows["A0"]._copy()

    lines = qapp.clipboard().text().splitlines()
    assert lines[0] == "Model: Linear"
    assert lines[1] == "χ²ᵣ 1.5"
    assert lines[2].split("\t")[0] == "m"
    assert "±" in lines[2]


def test_results_window_closes_when_the_chip_is_unchecked(panel: FitParametersPanel) -> None:
    select_params(panel, ["A0", "Lambda"])
    panel._model_fits["A0"] = _active_linear_fit("A0")
    panel._refresh_model_fit_button_labels()
    card(panel, "A0").result_chip.click()
    assert "A0" in panel._fit_results_windows

    select_params(panel, ["Lambda"])

    assert "A0" not in panel._fit_results_windows


def test_y_transform_menu_applies_to_one_card(
    panel: FitParametersPanel, monkeypatch: pytest.MonkeyPatch
) -> None:
    select_params(panel, ["A0", "Lambda"])
    monkeypatch.setattr(
        FitParametersPanel,
        "_exec_menu",
        lambda _self, menu, _pos: next(a for a in menu.actions() if a.data() == "reciprocal"),
    )
    panel._show_y_transform_menu("A0", QPoint(0, 0))
    assert panel._y_transform_for("A0") == AxisTransform.preset("reciprocal")
    assert panel._y_transform_for("Lambda").is_identity
    assert card(panel, "A0").transform_button.text() == "1/y"


def test_plot_mode_card_order_and_collapse_round_trip(panel: FitParametersPanel) -> None:
    select_params(panel, ["A0", "Lambda"])
    panel._card_stack.move_card("Lambda", 0)
    card(panel, "A0").set_expanded(False)
    panel._overlay_button.setChecked(True)

    state = panel.get_state()
    assert state["plot_mode"] == "Overlay"
    assert state["card_order"] == ["Lambda", "A0"]
    assert state["collapsed_params"] == ["A0"]

    restored = FitParametersPanel()
    restored.restore_state(state)
    assert restored._plot_mode() == "Overlay"
    assert restored._card_order == ["Lambda", "A0"]
    assert restored._collapsed_params == {"A0"}
    assert [c.name for c in restored._card_stack.cards()] == ["Lambda", "A0"]
    assert not card(restored, "A0").is_expanded()


def test_restoring_a_collapsed_card_in_subplots_mode_draws_its_sparkline(
    panel: FitParametersPanel,
) -> None:
    select_params(panel, ["A0", "Lambda"])
    card(panel, "A0").set_expanded(False)
    state = panel.get_state()
    assert state["plot_mode"] == "Subplots"

    restored = FitParametersPanel()
    restored.restore_state(state)
    restored._refresh_plot()
    assert not card(restored, "A0").is_expanded()
    assert card(restored, "A0").figure.axes == []
    assert len(card(restored, "Lambda").figure.axes) == 1


def test_legacy_single_axes_plot_mode_reads_as_overlay(panel: FitParametersPanel) -> None:
    state = panel.get_state()
    state["plot_mode"] = "Single Axes"
    restored = FitParametersPanel()
    restored.restore_state(state)
    assert restored._plot_mode() == "Overlay"


def test_table_dialog_hosts_the_live_table(panel: FitParametersPanel) -> None:
    panel._refresh_table()
    panel._show_table_dialog()
    assert panel._table_dialog.isAncestorOf(panel._table)
    assert panel._table_dialog.windowTitle() == "Fitted parameters"

    # A live Trend checkbox in the pop-out still routes back to the panel.
    toggled: list[tuple[str, int, bool]] = []
    include_col = panel._table.columnCount() - 1
    panel._table.item(0, include_col).setData(Qt.ItemDataRole.UserRole, ("batch", 7))
    panel.member_trend_inclusion_changed.connect(lambda *a: toggled.append(a))
    panel._table.item(0, include_col).setCheckState(Qt.CheckState.Unchecked)
    assert toggled == [("batch", 7, False)]

    panel._set_x_transform(AxisTransform.preset("square"))
    panel._show_table_dialog()
    assert "raw values" in panel._table_dialog.windowTitle()


def test_table_dialog_opens_at_its_content_width(panel: FitParametersPanel) -> None:
    """The pop-out must fit its columns rather than open at the layout's hint."""
    panel._refresh_table()
    panel._show_table_dialog()

    header = panel._table.horizontalHeader()
    assert not panel._table.horizontalScrollBar().isVisible()
    assert panel._table_dialog.width() >= header.length() + panel._table.verticalHeader().width()

    # A pop-out the user has widened is never shrunk back by a later show.
    widened = panel._table_dialog.width() + 120
    panel._table_dialog.resize(widened, panel._table_dialog.height())
    panel._show_table_dialog()
    assert panel._table_dialog.width() == widened


# ── Plot right-click menu ────────────────────────────────────────────────────


def _menu_texts(panel: FitParametersPanel, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the actions one right-click offers, and dismiss the menu."""
    captured: list[str] = []

    def _capture(_self, menu, _pos):
        captured.extend(action.text() or "—" for action in menu.actions())
        return None

    monkeypatch.setattr(FitParametersPanel, "_exec_menu", _capture)
    return captured


def _press_event(panel: FitParametersPanel, name: str, x: float, y: float):
    """A right-press on *name*'s axes at the data position (*x*, *y*)."""
    axes = axes_for(panel, name)
    px, py = axes.transData.transform((x, y))
    return SimpleNamespace(
        button=3, inaxes=axes, x=px, y=py, xdata=x, ydata=y, canvas=card(panel, name).canvas
    )


def test_right_click_on_empty_plot_offers_add_label(
    panel: FitParametersPanel, monkeypatch: pytest.MonkeyPatch
) -> None:
    select_params(panel, ["A0"])
    texts = _menu_texts(panel, monkeypatch)

    panel._on_plot_button_press(_press_event(panel, "A0", 150.0, 0.30))
    assert texts == ["Add label here…"]

    # Clear labels appears only once a label exists.
    panel._plot_annotations.append(
        {"x": 150.0, "y": 0.3, "text": "note", "axis_tag": "A0", "artist": None}
    )
    texts.clear()
    panel._on_plot_button_press(_press_event(panel, "A0", 150.0, 0.30))
    assert texts == ["Add label here…", "Clear labels"]


def test_right_click_on_a_label_offers_edit_and_remove(
    panel: FitParametersPanel, monkeypatch: pytest.MonkeyPatch
) -> None:
    select_params(panel, ["A0"])
    panel._plot_annotations.append(
        {"x": 150.0, "y": 0.3, "text": "note", "axis_tag": "A0", "artist": None}
    )
    monkeypatch.setattr(FitParametersPanel, "_detect_annotation_hit", lambda _self, _event: 0)
    texts = _menu_texts(panel, monkeypatch)

    panel._on_plot_button_press(_press_event(panel, "A0", 150.0, 0.30))

    assert texts == ["Edit label…", "Remove label", "Clear labels"]


def test_right_click_near_a_trend_point_also_offers_the_trend_toggle(
    panel: FitParametersPanel, monkeypatch: pytest.MonkeyPatch
) -> None:
    panel._rows[1].batch_id = "batch"
    panel._rows[1].trend_member_key = 7
    panel._rows[1].quality_flags = ["failed"]
    select_params(panel, ["A0"])
    texts = _menu_texts(panel, monkeypatch)

    # The second row sits at field 100 G, A0 = 0.20 — right on the cursor.
    panel._on_plot_button_press(_press_event(panel, "A0", 100.0, 0.20))

    assert texts == [
        "Add label here…",
        "—",
        "Exclude from trend",
        "Flags: fit did not converge",
    ]


def test_add_label_here_places_an_annotation_on_the_clicked_axes(
    panel: FitParametersPanel, monkeypatch: pytest.MonkeyPatch
) -> None:
    select_params(panel, ["A0", "Lambda"])
    monkeypatch.setattr(
        FitParametersPanel,
        "_exec_menu",
        lambda _self, menu, _pos: menu.actions()[0],
    )
    monkeypatch.setattr(
        "asymmetry.gui.panels.fit_parameters_panel.QInputDialog.getText",
        staticmethod(lambda *_args, **_kwargs: ("T_N", True)),
    )

    panel._on_plot_button_press(_press_event(panel, "Lambda", 150.0, 0.11))

    assert len(panel._plot_annotations) == 1
    annotation = panel._plot_annotations[0]
    assert annotation["text"] == "T_N"
    assert annotation["axis_tag"] == "Lambda"
    assert annotation["x"] == pytest.approx(150.0)


def test_copy_table_tsv_writes_the_data_columns(panel: FitParametersPanel, qapp) -> None:
    panel._refresh_table()
    panel._copy_table_tsv()
    lines = qapp.clipboard().text().splitlines()
    assert lines[0].split("\t") == [
        panel._table.horizontalHeaderItem(col).text() for col in range(panel._table_data_columns)
    ]
    # The χ²ᵣ / Trend gate columns are not part of the data block.
    assert len(lines) == panel._table.rowCount() + 1
    assert "Trend" not in lines[0]


def test_long_parameter_name_does_not_widen_the_rail(panel: FitParametersPanel) -> None:
    long_name = "A_extremely_long_parameter_name_that_would_overflow_the_dock"
    panel._rows = [
        _FitRow(
            run_number=1,
            run_label="1",
            field=100.0,
            temperature=10.0,
            values={long_name: 0.2},
            errors={long_name: 0.01},
        ),
    ]
    panel._varying_params = [long_name]
    panel._rebuild_y_controls()
    # The chip carries the full label on hover, and the card's name label elides
    # rather than imposing its text width as a minimum.
    assert panel._y_chips[long_name].toolTip() == _format_param_label(long_name)
    assert card(panel, long_name)._name_label.minimumSizeHint().width() == 0


def _accept_message_box(
    monkeypatch: pytest.MonkeyPatch, *, accept: bool = True
) -> list[tuple[str, str]]:
    """Answer the series-delete confirmation without showing it.

    Returns the (text, informative text) of each box raised, so a caller can
    assert on what the prompt actually said.
    """
    shown: list[tuple[str, str]] = []

    def _exec(box: QMessageBox) -> QMessageBox.StandardButton:
        shown.append((box.text(), box.informativeText()))
        return QMessageBox.StandardButton.Ok if accept else QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(QMessageBox, "exec", _exec)
    return shown


def test_delete_group_fits_removes_only_that_series(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = FitParametersPanel()
    rows = [
        _FitRow(
            run_number=101,
            run_label="101",
            field=100.0,
            temperature=5.0,
            values={"A0": 0.2},
            errors={"A0": 0.01},
        ),
        _FitRow(
            run_number=102,
            run_label="102",
            field=120.0,
            temperature=5.0,
            values={"A0": 0.19},
            errors={"A0": 0.01},
        ),
    ]
    panel._group_fit_results = {
        "g1": _GroupFitData(
            group_id="g1",
            group_name="Group 1",
            rows=rows,
            global_params=None,
            varying_params=["A0"],
            inferred_x_key="field",
            model_fits={},
            plot_annotations=[],
        ),
        "g2": _GroupFitData(
            group_id="g2",
            group_name="Group 2",
            rows=list(rows),
            global_params=None,
            varying_params=["A0"],
            inferred_x_key="field",
            model_fits={},
            plot_annotations=[],
        ),
    }
    panel._active_group_id = "g1"
    panel._rebuild_group_buttons()
    panel._set_selected_group_ids(["g1"], emit=False)

    emitted: list[str] = []
    panel.series_delete_requested.connect(emitted.append)

    shown = _accept_message_box(monkeypatch)
    panel._delete_group_fits("g1")

    assert "g1" not in panel._group_fit_results
    assert panel._active_group_id == "g2"
    # Only the series id travels: what else to clear is the main window's call
    # (D6 — other series and the runs' single fits are kept).
    assert emitted == ["g1"]
    assert shown == [
        (
            'Delete series "Group 1"?',
            "Removes this series and its trend. Other series and single fits on "
            "these runs are kept.",
        )
    ]


def test_background_labels_use_subscript_formatting() -> None:
    assert _format_plot_label("A_bg") == "$A_{bg}$ (%)"
    assert _format_plot_legend_label("A_bg") == "$A_{bg}$"
    assert _format_gle_label("A_bg") == "{\\it A}_{bg} (%)"
    assert _format_gle_legend_label("A_bg") == "{\\it A}_{bg}"


class _FakeAxis:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.line_calls: list[dict[str, object]] = []
        self.xscale_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.yscale_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.ylabel_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def errorbar_from_file(self, *args, **kwargs) -> None:
        self.calls.append({"args": args, "kwargs": kwargs})

    def line_from_file(self, *args, **kwargs) -> None:
        self.line_calls.append({"args": args, "kwargs": kwargs})

    def set_xlabel(self, *_args, **_kwargs) -> None:
        return

    def set_ylabel(self, *_args, **_kwargs) -> None:
        self.ylabel_calls.append((_args, _kwargs))
        return

    def set_xscale(self, *_args, **_kwargs) -> None:
        self.xscale_calls.append((_args, _kwargs))
        return

    def set_yscale(self, *_args, **_kwargs) -> None:
        self.yscale_calls.append((_args, _kwargs))
        return

    def legend(self, *_args, **_kwargs) -> None:
        return


class _FakeFigure:
    def __init__(self, axis: _FakeAxis) -> None:
        self._axis = axis
        self.saved_paths: list[str] = []
        self.saved_kwargs: list[dict[str, object]] = []

    def add_subplot(self, *_args, **_kwargs) -> _FakeAxis:
        return self._axis

    def savefig(self, path: str, **kwargs) -> None:
        self.saved_kwargs.append(dict(kwargs))
        output_path = Path(path)
        if kwargs.get("folder"):
            output_path, export_dir = resolve_gle_export_paths(output_path, folder=True)
            export_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        self.saved_paths.append(str(output_path))
        output_path.write_text("! fake gle", encoding="utf-8")


def _patch_save_dialog(monkeypatch: pytest.MonkeyPatch, out_gle: Path) -> None:
    monkeypatch.setattr(
        "asymmetry.gui.panels.fit_parameters_panel.QFileDialog.getSaveFileName",
        lambda *_a, **_k: (str(out_gle), "GLE export folders (*.gleplot)"),
    )


def test_export_gle_uses_errorbar_from_file(
    panel: FitParametersPanel, qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_gle = tmp_path / "plot.gleplot"
    gle_path, _ = resolve_gle_export_paths(out_gle, folder=True)

    axis = _FakeAxis()
    fig = _FakeFigure(axis)

    fake_glp = SimpleNamespace(
        Axes=type(
            "FakeAxes",
            (),
            {
                "errorbar_from_file": staticmethod(lambda *a, **k: None),
                "line_from_file": staticmethod(lambda *a, **k: None),
            },
        ),
        figure=lambda **_kwargs: fig,
    )

    compiled: list[tuple[Path, str, Path]] = []

    monkeypatch.setitem(sys.modules, "gleplot", fake_glp)
    _patch_save_dialog(monkeypatch, out_gle)
    monkeypatch.setattr(gle_export, "get_gle_executable", lambda: "/fake/gle")
    monkeypatch.setattr(
        gle_export,
        "compile_gle",
        lambda exe, gle_file, fmt, *, cwd, **kw: compiled.append((Path(gle_file), fmt, Path(cwd))),
    )
    results: list[tuple[str, str]] = []
    monkeypatch.setattr(
        gle_export,
        "show_export_result_dialog",
        lambda p, title, summary, details: results.append((summary, details)),
    )
    monkeypatch.setattr(gle_export, "post_export_view", lambda p, gp: None)

    panel._export_gle("pdf")

    assert axis.calls, "Expected at least one errorbar_from_file call"
    first = axis.calls[0]
    assert first["args"][0] == gle_path.with_suffix(".dat").name
    assert first["kwargs"]["x_col"] == 2
    assert first["kwargs"]["y_col"] == 4
    assert first["kwargs"]["yerr_col"] == 5
    assert str(gle_path) in fig.saved_paths
    assert "folder" not in fig.saved_kwargs[-1]

    wait_for(lambda: bool(compiled), qapp)
    assert compiled[0][0] == gle_path
    assert compiled[0][1] == "pdf"
    assert compiled[0][2] == gle_path.parent
    wait_for(lambda: bool(results), qapp)


def test_export_gle_saves_to_resolved_gle_path(
    panel: FitParametersPanel, qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_gle = tmp_path / "plot.gleplot"
    gle_path, _export_dir = resolve_gle_export_paths(out_gle, folder=True)

    axis = _FakeAxis()

    class _CaptureFigure(_FakeFigure):
        def __init__(self, axis: _FakeAxis) -> None:
            super().__init__(axis)
            self.requested_paths: list[str] = []

        def savefig(self, path: str, **kwargs) -> None:
            self.requested_paths.append(path)
            super().savefig(path, **kwargs)

    fig = _CaptureFigure(axis)
    fake_glp = SimpleNamespace(
        Axes=type("FakeAxes", (), {"errorbar_from_file": staticmethod(lambda *a, **k: None)}),
        figure=lambda **_kwargs: fig,
    )

    monkeypatch.setitem(sys.modules, "gleplot", fake_glp)
    _patch_save_dialog(monkeypatch, out_gle)
    monkeypatch.setattr(gle_export, "get_gle_executable", lambda: None)
    infos: list[tuple[str, str]] = []
    monkeypatch.setattr(gle_export, "show_info", lambda p, title, msg: infos.append((title, msg)))
    monkeypatch.setattr(gle_export, "post_export_view", lambda p, gp: None)

    panel._export_gle("pdf")

    assert fig.requested_paths == [str(gle_path)]
    assert str(gle_path) in fig.saved_paths
    # Synchronous "no GLE binary" path: no task/compile involved.
    assert infos and "GLE script saved" in infos[0][1]


def test_export_gle_warns_for_old_gleplot(
    panel: FitParametersPanel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_gle = tmp_path / "plot.gleplot"

    warnings: list[tuple[str, str]] = []
    fake_glp = SimpleNamespace(Axes=type("OldAxes", (), {}))

    monkeypatch.setitem(sys.modules, "gleplot", fake_glp)
    _patch_save_dialog(monkeypatch, out_gle)
    monkeypatch.setattr(
        gle_export, "show_warning", lambda p, title, msg: warnings.append((title, msg))
    )

    panel._export_gle("pdf")

    assert warnings
    assert warnings[0][0] == "gleplot update required"
    assert "gleplot" in warnings[0][1]


def test_effective_x_key_mapping(panel: FitParametersPanel) -> None:
    panel._x_combo.setCurrentText("𝐵 (G)")
    assert panel._effective_x_key() == "field"

    panel._x_combo.setCurrentText("𝑇 (K)")
    assert panel._effective_x_key() == "temperature"

    panel._x_combo.setCurrentText("Run")
    assert panel._effective_x_key() == "run"

    panel._x_combo.setCurrentText("Auto")
    panel._inferred_x_key = "temperature"
    assert panel._effective_x_key() == "temperature"


def test_export_gle_subplots_mode_uses_black_series(
    panel: FitParametersPanel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_gle = tmp_path / "plot_subplots.gleplot"

    ax1 = _FakeAxis()
    ax2 = _FakeAxis()
    fig = _FakeFigure(ax1)

    fake_glp = SimpleNamespace(
        Axes=type("FakeAxes", (), {"errorbar_from_file": staticmethod(lambda *a, **k: None)}),
        subplots=lambda **_kwargs: (fig, [ax1, ax2]),
        figure=lambda **_kwargs: fig,
    )

    panel._subplots_button.setChecked(True)
    panel._log_x_check.setChecked(True)
    panel._log_y_params = {"A0", "Lambda"}
    monkeypatch.setattr(panel, "_selected_y_parameters", lambda: ["A0", "Lambda"])

    monkeypatch.setitem(sys.modules, "gleplot", fake_glp)
    _patch_save_dialog(monkeypatch, out_gle)
    monkeypatch.setattr(gle_export, "get_gle_executable", lambda: None)
    monkeypatch.setattr(gle_export, "show_info", lambda p, title, msg: None)
    monkeypatch.setattr(gle_export, "post_export_view", lambda p, gp: None)

    panel._export_gle("pdf")

    assert ax1.calls and ax2.calls
    assert ax1.calls[0]["kwargs"]["color"] == "black"
    assert ax2.calls[0]["kwargs"]["color"] == "black"
    assert ax1.xscale_calls and ax1.yscale_calls
    assert ax2.xscale_calls and ax2.yscale_calls


def test_export_gle_dual_axis_assigns_y_and_y2(
    panel: FitParametersPanel, qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_gle = tmp_path / "plot_dual.gleplot"

    axis = _FakeAxis()
    fig = _FakeFigure(axis)
    fake_glp = SimpleNamespace(
        Axes=type("FakeAxes", (), {"errorbar_from_file": staticmethod(lambda *a, **k: None)}),
        figure=lambda **_kwargs: fig,
    )

    panel._overlay_button.setChecked(True)
    monkeypatch.setattr(panel, "_selected_y_parameters", lambda: ["A0", "Lambda"])

    monkeypatch.setitem(sys.modules, "gleplot", fake_glp)
    _patch_save_dialog(monkeypatch, out_gle)
    monkeypatch.setattr(gle_export, "get_gle_executable", lambda: "/fake/gle")
    monkeypatch.setattr(gle_export, "compile_gle", lambda *a, **k: None)
    results: list[object] = []
    monkeypatch.setattr(
        gle_export,
        "show_export_result_dialog",
        lambda p, title, summary, details: results.append(summary),
    )
    monkeypatch.setattr(gle_export, "post_export_view", lambda p, gp: None)

    panel._export_gle("pdf")

    assert len(axis.calls) == 2
    assert axis.calls[0]["kwargs"]["x_col"] == 2
    assert axis.calls[1]["kwargs"]["x_col"] == 2
    assert axis.calls[0]["kwargs"]["yaxis"] == "y"
    assert axis.calls[1]["kwargs"]["yaxis"] == "y2"
    wait_for(lambda: bool(results), qapp)


def test_write_fit_files_restored_fit_without_bounds(
    panel: FitParametersPanel, tmp_path: Path
) -> None:
    """Loaded fits with missing range bounds should still export .fit curve files."""
    model = ParameterCompositeModel(["Constant"])
    params = ParameterSet([Parameter("c", value=0.21)])
    result = ParameterModelFitResult(
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        parameters=params,
        uncertainties={"c": 0.01},
        message="Fit successful",
    )

    panel._model_fits = {
        "A0": ParameterModelFit(
            parameter_name="A0",
            x_key="field",
            active=True,
            ranges=[
                ModelFitRange(
                    x_min=None,
                    x_max=None,
                    model=model,
                    parameters=params,
                    result=result,
                )
            ],
        )
    }

    fit_map = panel._write_fit_files(tmp_path / "fit_parameters.gle", "field", ["A0"])
    assert fit_map

    out_path = next(iter(fit_map.values()))
    assert out_path.exists()

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert not any("line_style" in ln for ln in lines)
    assert not any("line_color" in ln for ln in lines)
    assert not any("columns:" in ln for ln in lines)
    data_lines = [ln for ln in lines if ln and not ln.startswith("!")]
    assert data_lines, "Expected numeric x,y rows in exported .fit file"


def test_sample_fit_range_curve_uses_log_spacing_for_field(panel: FitParametersPanel) -> None:
    model = ParameterCompositeModel(["Constant"])
    params = ParameterSet([Parameter("c", value=0.21)])
    result = ParameterModelFitResult(
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        parameters=params,
        uncertainties={"c": 0.01},
        message="Fit successful",
    )

    fit_range = ModelFitRange(
        x_min=100.0,
        x_max=10000.0,
        model=model,
        parameters=params,
        result=result,
    )

    sampled = panel._sample_fit_range_curve(fit_range, x_key="field", num_points=5)

    assert sampled is not None
    xs, ys = sampled
    np.testing.assert_allclose(xs[[0, -1]], [100.0, 10000.0])
    np.testing.assert_allclose(ys, np.full(5, 0.21))
    ratios = xs[1:] / xs[:-1]
    np.testing.assert_allclose(ratios, np.full_like(ratios, ratios[0]))


def test_export_gle_writes_fit_files_for_active_unselected_fit_param(
    panel: FitParametersPanel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Export should still emit .fit sidecars for active fits not in current Y selection."""
    model = ParameterCompositeModel(["Constant"])
    params = ParameterSet([Parameter("c", value=0.21)])
    result = ParameterModelFitResult(
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        parameters=params,
        uncertainties={"c": 0.01},
        message="Fit successful",
    )
    panel._model_fits = {
        "A0": ParameterModelFit(
            parameter_name="A0",
            x_key="field",
            active=True,
            ranges=[
                ModelFitRange(
                    x_min=100.0,
                    x_max=200.0,
                    model=model,
                    parameters=params,
                    result=result,
                )
            ],
        )
    }

    out_gle = tmp_path / "fit_parameters.gleplot"
    resolved_gle, _ = resolve_gle_export_paths(out_gle, folder=True)
    _patch_save_dialog(monkeypatch, out_gle)

    # Simulate user selecting a different parameter than the one with active fit.
    monkeypatch.setattr(panel, "_selected_y_parameters", lambda: ["Lambda"])

    captured: dict[str, object] = {}
    original_build = panel._build_gle_export

    def _capturing_build(glp, gle_path, export_dir):
        result = original_build(glp, gle_path, export_dir)
        captured["gle_path"] = gle_path
        captured["result"] = result
        return result

    monkeypatch.setattr(panel, "_build_gle_export", _capturing_build)

    axis = _FakeAxis()
    fig = _FakeFigure(axis)
    fake_glp = SimpleNamespace(
        Axes=type(
            "FakeAxes",
            (),
            {
                "errorbar_from_file": staticmethod(lambda *a, **k: None),
                "line_from_file": staticmethod(lambda *a, **k: None),
            },
        ),
        figure=lambda **_kwargs: fig,
    )
    monkeypatch.setitem(sys.modules, "gleplot", fake_glp)
    monkeypatch.setattr(gle_export, "get_gle_executable", lambda: None)
    monkeypatch.setattr(gle_export, "show_info", lambda p, title, msg: None)
    monkeypatch.setattr(gle_export, "post_export_view", lambda p, gp: None)

    panel._export_gle("pdf")

    assert captured["gle_path"] == resolved_gle
    result = captured.get("result")
    assert result is not None, "Expected the builder to succeed, not abort"
    fit_files = [p for p in result.files if p.suffix == ".fit"]
    assert fit_files, "Expected active fit sidecar map to be passed to GLE generator"
    assert fit_files[0].exists()
    assert fit_files[0].parent == resolved_gle.parent


def test_add_gle_model_overlay_uses_formatted_labels_without_hash_one(
    panel: FitParametersPanel, tmp_path: Path
) -> None:
    model = ParameterCompositeModel(["Constant"])
    params = ParameterSet([Parameter("c", value=0.21)])
    result = ParameterModelFitResult(
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        parameters=params,
        uncertainties={"c": 0.01},
        message="Fit successful",
    )
    panel._model_fits = {
        "A_bg": ParameterModelFit(
            parameter_name="A_bg",
            x_key="field",
            active=True,
            ranges=[
                ModelFitRange(
                    x_min=100.0, x_max=150.0, model=model, parameters=params, result=result
                ),
                ModelFitRange(
                    x_min=150.0, x_max=200.0, model=model, parameters=params, result=result
                ),
            ],
        )
    }
    fit_map = {
        ("A_bg", 0): tmp_path / "a.fit",
        ("A_bg", 1): tmp_path / "b.fit",
    }
    axis = _FakeAxis()

    panel._add_gle_model_overlay(
        axis, "A_bg", color="red", include_labels=True, fit_file_map=fit_map
    )

    assert len(axis.line_calls) == 2
    assert axis.line_calls[0]["kwargs"]["label"] == "fit {\\it A}_{bg}"
    assert axis.line_calls[1]["kwargs"]["label"] == "fit {\\it A}_{bg} #2"
    assert axis.line_calls[0]["kwargs"]["color"] != axis.line_calls[1]["kwargs"]["color"]


def test_build_inherited_cross_group_config_uses_best_successful_fit(
    panel: FitParametersPanel,
) -> None:
    model = ParameterCompositeModel(["Linear"])

    best_params = ParameterSet(
        [
            Parameter("m", value=0.002, min=0.0, max=1.0, fixed=True),
            Parameter("b", value=0.21, min=-1.0, max=1.0, fixed=False),
        ]
    )
    best_result = ParameterModelFitResult(
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=0.4,
        parameters=best_params,
        uncertainties={"b": 0.01},
        message="ok",
    )
    best_range = ModelFitRange(
        x_min=100.0,
        x_max=300.0,
        model=model,
        parameters=best_params,
        result=best_result,
    )

    other_params = ParameterSet(
        [
            Parameter("m", value=0.01, min=0.0, max=1.0, fixed=False),
            Parameter("b", value=0.5, min=-1.0, max=1.0, fixed=False),
        ]
    )
    other_result = ParameterModelFitResult(
        success=True,
        chi_squared=10.0,
        reduced_chi_squared=2.0,
        parameters=other_params,
        uncertainties={"m": 0.1, "b": 0.1},
        message="ok",
    )
    other_range = ModelFitRange(
        x_min=90.0,
        x_max=310.0,
        model=model,
        parameters=other_params,
        result=other_result,
    )

    best_group = _GroupFitData(
        group_id="g_best",
        group_name="Best",
        rows=[],
        global_params=None,
        varying_params=["Lambda"],
        inferred_x_key="field",
        model_fits={
            "Lambda": ParameterModelFit(
                parameter_name="Lambda",
                x_key="field",
                ranges=[best_range],
                active=True,
            )
        },
        plot_annotations=[],
    )
    other_group = _GroupFitData(
        group_id="g_other",
        group_name="Other",
        rows=[],
        global_params=None,
        varying_params=["Lambda"],
        inferred_x_key="field",
        model_fits={
            "Lambda": ParameterModelFit(
                parameter_name="Lambda",
                x_key="field",
                ranges=[other_range],
                active=True,
            )
        },
        plot_annotations=[],
    )

    config = panel._build_inherited_cross_group_config(
        "Lambda",
        "field",
        [other_group, best_group],
    )

    assert config is not None
    # Renamed source_group_* → trend_group_* (D8) to end the collision with the
    # unrelated FitSeries.source_group_id.
    assert config["trend_group_id"] == "g_best"
    assert config["trend_group_name"] == "Best"
    assert config["fit_x_min"] == pytest.approx(100.0)
    assert config["fit_x_max"] == pytest.approx(300.0)

    rows = config.get("parameter_rows")
    assert isinstance(rows, list)
    row_by_name = {str(row["name"]): row for row in rows if isinstance(row, dict)}
    assert row_by_name["m"]["initial"] == pytest.approx(0.002)
    assert row_by_name["m"]["type"] == "Fixed"
    assert row_by_name["b"]["initial"] == pytest.approx(0.21)
    assert row_by_name["b"]["type"] == "Global"


def test_build_inherited_cross_group_config_returns_none_without_success(
    panel: FitParametersPanel,
) -> None:
    model = ParameterCompositeModel(["Linear"])
    failed_params = ParameterSet([Parameter("m", value=0.1), Parameter("b", value=0.2)])
    failed_range = ModelFitRange(
        x_min=0.0,
        x_max=1.0,
        model=model,
        parameters=failed_params,
        result=ParameterModelFitResult(success=False, message="failed"),
    )
    failed_group = _GroupFitData(
        group_id="g0",
        group_name="G0",
        rows=[],
        global_params=None,
        varying_params=["Lambda"],
        inferred_x_key="field",
        model_fits={
            "Lambda": ParameterModelFit(
                parameter_name="Lambda",
                x_key="field",
                ranges=[failed_range],
                active=True,
            )
        },
        plot_annotations=[],
    )

    config = panel._build_inherited_cross_group_config(
        "Lambda",
        "field",
        [failed_group],
    )
    assert config is None


def test_group_switch_persists_active_group_model_fits(panel: FitParametersPanel) -> None:
    fit = ParameterModelFit(parameter_name="Lambda", x_key="field", ranges=[], active=True)

    group_a = _GroupFitData(
        group_id="g_a",
        group_name="Group A",
        rows=list(panel._rows),
        global_params=None,
        varying_params=list(panel._varying_params),
        inferred_x_key="field",
        model_fits={},
        plot_annotations=[],
    )
    group_b = _GroupFitData(
        group_id="g_b",
        group_name="Group B",
        rows=list(panel._rows),
        global_params=None,
        varying_params=list(panel._varying_params),
        inferred_x_key="field",
        model_fits={},
        plot_annotations=[],
    )

    panel._group_fit_results = {"g_a": group_a, "g_b": group_b}
    panel._active_group_id = "g_a"
    panel._model_fits = {"Lambda": fit}
    panel._rebuild_group_buttons()

    panel._set_selected_group_ids(["g_b"], emit=False)
    panel._apply_group_selection_to_view()

    assert "Lambda" in panel._group_fit_results["g_a"].model_fits


def test_get_state_syncs_active_group_before_serializing(panel: FitParametersPanel) -> None:
    fit = ParameterModelFit(parameter_name="Lambda", x_key="field", ranges=[], active=True)

    panel._group_fit_results = {
        "g_a": _GroupFitData(
            group_id="g_a",
            group_name="Group A",
            rows=list(panel._rows),
            global_params=None,
            varying_params=list(panel._varying_params),
            inferred_x_key="field",
            model_fits={},
            plot_annotations=[],
        )
    }
    panel._active_group_id = "g_a"
    panel._model_fits = {"Lambda": fit}

    state = panel.get_state()
    groups_state = state.get("group_fit_results", {})
    assert isinstance(groups_state, dict)
    g_a_state = groups_state.get("g_a", {})
    assert isinstance(g_a_state, dict)
    model_fits_state = g_a_state.get("model_fits", {})
    assert isinstance(model_fits_state, dict)
    assert "Lambda" in model_fits_state


def test_multi_group_view_excludes_asymmetry_global_parameters(panel: FitParametersPanel) -> None:
    rows_a = [
        _FitRow(
            run_number=1,
            run_label="1",
            field=100.0,
            temperature=10.0,
            values={"A0": 0.20, "Lambda": 0.10},
            errors={"A0": 0.01, "Lambda": 0.01},
        )
    ]
    rows_b = [
        _FitRow(
            run_number=2,
            run_label="2",
            field=200.0,
            temperature=10.0,
            values={"A0": 0.30, "Lambda": 0.15},
            errors={"A0": 0.01, "Lambda": 0.01},
        )
    ]

    global_params = ParameterSet([Parameter("A0", value=0.25)])

    panel._group_fit_results = {
        "g_a": _GroupFitData(
            group_id="g_a",
            group_name="Group A",
            rows=rows_a,
            global_params=global_params,
            varying_params=["Lambda"],
            inferred_x_key="field",
            model_fits={},
            plot_annotations=[],
        ),
        "g_b": _GroupFitData(
            group_id="g_b",
            group_name="Group B",
            rows=rows_b,
            global_params=global_params,
            varying_params=["Lambda"],
            inferred_x_key="field",
            model_fits={},
            plot_annotations=[],
        ),
    }
    panel._active_group_id = "g_a"
    panel._rebuild_group_buttons()
    panel._set_selected_group_ids(["g_a", "g_b"], emit=False)

    panel._apply_group_selection_to_view(sync_active=False)

    assert "Lambda" in panel._varying_params
    assert "A0" not in panel._varying_params


def test_group_global_params_roundtrip_in_panel_state(panel: FitParametersPanel) -> None:
    global_params = ParameterSet([Parameter("A0", value=0.25, min=0.0, max=1.0, fixed=False)])
    panel._group_fit_results = {
        "g_a": _GroupFitData(
            group_id="g_a",
            group_name="Group A",
            rows=list(panel._rows),
            global_params=global_params,
            varying_params=list(panel._varying_params),
            inferred_x_key="field",
            model_fits={},
            plot_annotations=[],
        )
    }
    panel._active_group_id = "g_a"
    panel._global_params = global_params

    state = panel.get_state()

    restored = FitParametersPanel()
    restored.restore_state(state)

    assert "g_a" in restored._group_fit_results
    restored_global = restored._group_fit_results["g_a"].global_params
    assert restored_global is not None
    assert "A0" in restored_global
    assert restored_global["A0"].value == pytest.approx(0.25)


def test_panel_state_roundtrip_preserves_grouped_parameter_model(panel: FitParametersPanel) -> None:
    grouped_model = ParameterCompositeModel.from_expression("Linear + ( Arrhenius * Constant )")
    grouped_fit = ParameterModelFit(
        parameter_name="Lambda",
        x_key="temperature",
        ranges=[
            ModelFitRange(
                x_min=5.0,
                x_max=25.0,
                model=grouped_model,
                parameters=ParameterSet(
                    [
                        Parameter("m", value=0.01),
                        Parameter("b", value=0.2),
                        Parameter("a", value=0.3),
                        Parameter("Ea", value=2.0),
                        Parameter("c", value=0.05),
                    ]
                ),
            )
        ],
        active=True,
    )
    panel._model_fits = {"Lambda": grouped_fit}

    state = panel.get_state()

    restored = FitParametersPanel()
    restored.restore_state(state)

    restored_fit = restored._model_fits["Lambda"]
    restored_model = restored_fit.ranges[0].model
    assert restored_model.component_names == grouped_model.component_names
    assert restored_model.operators == grouped_model.operators
    assert restored_model.open_parentheses == grouped_model.open_parentheses
    assert restored_model.close_parentheses == grouped_model.close_parentheses


def test_trend_model_memory_round_trips(panel: FitParametersPanel) -> None:
    """The trend Model Fit dialog's "remember last model" memory is
    project-scoped: it round-trips through get_state/restore_state (persisted
    with the project) rather than living in QSettings."""
    panel._trend_model_memory = {"Lambda|field": "Linear"}

    state = panel.get_state()

    restored = FitParametersPanel()
    restored.restore_state(state)

    assert restored._trend_model_memory == {"Lambda|field": "Linear"}


def test_new_asymmetry_fit_overwrites_existing_group_model_fits(panel: FitParametersPanel) -> None:
    existing_fit = ParameterModelFit(parameter_name="Lambda", x_key="field", ranges=[], active=True)
    panel._group_fit_results = {
        "g1": _GroupFitData(
            group_id="g1",
            group_name="Group 1",
            rows=list(panel._rows),
            global_params=None,
            varying_params=list(panel._varying_params),
            inferred_x_key="field",
            model_fits={"Lambda": existing_fit},
            plot_annotations=[{"parameter": "Lambda"}],
        )
    }
    panel._active_group_id = "g1"
    panel._model_fits = {"Lambda": existing_fit}
    panel._plot_annotations = [{"parameter": "Lambda"}]

    result = FitResult(
        success=True,
        reduced_chi_squared=1.0,
        parameters=ParameterSet([Parameter("A0", value=0.24), Parameter("Lambda", value=0.11)]),
        uncertainties={"A0": 0.01, "Lambda": 0.01},
    )
    datasets_by_run = {
        101: SimpleNamespace(
            metadata={
                "run_label": "101",
                "field": 150.0,
                "temperature": 12.0,
            }
        )
    }

    panel.set_fit_results(
        {101: (result, (np.array([0.0, 1.0]), np.array([0.24, 0.24])))},
        datasets_by_run,
        global_params=None,
        group_id="g1",
        group_name="Group 1",
    )

    assert panel._group_fit_results["g1"].model_fits == {}
    assert panel._group_fit_results["g1"].plot_annotations == []
    assert panel._model_fits == {}
    assert panel._plot_annotations == []


def test_multi_group_view_does_not_sync_merged_rows_back_into_active_group(
    panel: FitParametersPanel,
) -> None:
    rows_a = [
        _FitRow(
            run_number=1,
            run_label="1",
            field=100.0,
            temperature=10.0,
            values={"A0": 0.20, "Lambda": 0.10},
            errors={"A0": 0.01, "Lambda": 0.01},
        )
    ]
    rows_b = [
        _FitRow(
            run_number=2,
            run_label="2",
            field=200.0,
            temperature=11.0,
            values={"A0": 0.30, "Lambda": 0.15},
            errors={"A0": 0.01, "Lambda": 0.01},
        )
    ]

    panel._group_fit_results = {
        "g_a": _GroupFitData(
            group_id="g_a",
            group_name="Group A",
            rows=rows_a,
            global_params=None,
            varying_params=["A0", "Lambda"],
            inferred_x_key="field",
            model_fits={},
            plot_annotations=[],
        ),
        "g_b": _GroupFitData(
            group_id="g_b",
            group_name="Group B",
            rows=rows_b,
            global_params=None,
            varying_params=["A0", "Lambda"],
            inferred_x_key="field",
            model_fits={},
            plot_annotations=[],
        ),
    }
    panel._active_group_id = "g_a"
    panel._rebuild_group_buttons()
    panel._set_selected_group_ids(["g_a", "g_b"], emit=False)

    panel._apply_group_selection_to_view(sync_active=False)
    assert panel._active_group_id == "g_a"
    assert len(panel._rows) == 1
    assert panel._rows[0].run_number == 1

    # Regression guard: merged rows must never be written into Group A.
    panel._sync_active_group_state()
    assert [row.run_number for row in panel._group_fit_results["g_a"].rows] == [1]


def test_group_button_styles_distinguish_single_active_from_multi_highlight(
    panel: FitParametersPanel,
) -> None:
    panel._group_fit_results = {
        "g_a": _GroupFitData(
            group_id="g_a",
            group_name="Group A",
            rows=list(panel._rows),
            global_params=None,
            varying_params=list(panel._varying_params),
            inferred_x_key="field",
            model_fits={},
            plot_annotations=[],
        ),
        "g_b": _GroupFitData(
            group_id="g_b",
            group_name="Group B",
            rows=list(panel._rows),
            global_params=None,
            varying_params=list(panel._varying_params),
            inferred_x_key="field",
            model_fits={},
            plot_annotations=[],
        ),
    }
    panel._rebuild_group_buttons()

    panel._set_selected_group_ids(["g_a"], emit=False)
    panel._apply_group_selection_to_view(sync_active=False)
    # Series buttons now use the red accent palette (ACCENT_RED = #a8332a).
    assert "2px solid #a8332a" in panel._group_button_map["g_a"].styleSheet()

    panel._set_selected_group_ids(["g_a", "g_b"], emit=False)
    panel._apply_group_selection_to_view(sync_active=False)
    assert "2px solid #a8332a" in panel._group_button_map["g_a"].styleSheet()
    assert "2px solid #a8332a" not in panel._group_button_map["g_b"].styleSheet()
    assert "1px solid #a8332a" in panel._group_button_map["g_b"].styleSheet()


def test_group_button_styles_refresh_after_scale_change(panel: FitParametersPanel) -> None:
    panel._group_fit_results = {
        "g_a": _GroupFitData(
            group_id="g_a",
            group_name="Group A",
            rows=list(panel._rows),
            global_params=None,
            varying_params=list(panel._varying_params),
            inferred_x_key="field",
            model_fits={},
            plot_annotations=[],
        )
    }
    panel._rebuild_group_buttons()
    panel._set_selected_group_ids(["g_a"], emit=False)
    panel._apply_group_selection_to_view(sync_active=False)

    panel._on_ui_scale_changed(1.0, 1.1)

    style = panel._group_button_map["g_a"].styleSheet()
    # Series buttons use the red accent palette (ACCENT_RED = #a8332a).
    assert "2px solid #a8332a" in style
    assert "border-radius: 13px;" in style


def test_shift_click_highlights_group_without_changing_active_selection(
    panel: FitParametersPanel, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows_a = [
        _FitRow(
            run_number=1,
            run_label="1",
            field=100.0,
            temperature=10.0,
            values={"Lambda": 0.10},
            errors={"Lambda": 0.01},
        )
    ]
    rows_b = [
        _FitRow(
            run_number=2,
            run_label="2",
            field=200.0,
            temperature=10.0,
            values={"Lambda": 0.14},
            errors={"Lambda": 0.01},
        )
    ]

    panel._group_fit_results = {
        "g_a": _GroupFitData(
            group_id="g_a",
            group_name="Group A",
            rows=rows_a,
            global_params=None,
            varying_params=["Lambda"],
            inferred_x_key="field",
            model_fits={},
            plot_annotations=[],
        ),
        "g_b": _GroupFitData(
            group_id="g_b",
            group_name="Group B",
            rows=rows_b,
            global_params=None,
            varying_params=["Lambda"],
            inferred_x_key="field",
            model_fits={},
            plot_annotations=[],
        ),
    }
    panel._active_group_id = "g_a"
    panel._rebuild_group_buttons()
    panel._set_selected_group_ids(["g_a"], emit=False)
    panel._apply_group_selection_to_view(sync_active=False)

    monkeypatch.setattr(
        "asymmetry.gui.panels.fit_parameters_panel.QApplication.keyboardModifiers",
        lambda: Qt.KeyboardModifier.ShiftModifier,
    )

    panel._group_button_map["g_b"].click()

    assert panel._active_group_id == "g_a"
    assert panel._group_button_map["g_a"].isChecked()
    assert panel._group_button_map["g_b"].isChecked()
    assert len(panel._rows) == 1
    assert panel._rows[0].run_number == 1


def test_cross_group_fit_success_updates_each_group_model_fit(panel: FitParametersPanel) -> None:
    rows_a = [
        _FitRow(
            run_number=1,
            run_label="1",
            field=100.0,
            temperature=10.0,
            values={"Lambda": 0.10},
            errors={"Lambda": 0.01},
        )
    ]
    rows_b = [
        _FitRow(
            run_number=2,
            run_label="2",
            field=200.0,
            temperature=10.0,
            values={"Lambda": 0.14},
            errors={"Lambda": 0.01},
        )
    ]
    group_a = _GroupFitData(
        group_id="g_a",
        group_name="Group A",
        rows=rows_a,
        global_params=None,
        varying_params=["Lambda"],
        inferred_x_key="field",
        model_fits={},
        plot_annotations=[],
    )
    group_b = _GroupFitData(
        group_id="g_b",
        group_name="Group B",
        rows=rows_b,
        global_params=None,
        varying_params=["Lambda"],
        inferred_x_key="field",
        model_fits={},
        plot_annotations=[],
    )
    panel._group_fit_results = {"g_a": group_a, "g_b": group_b}

    model = ParameterCompositeModel(["Linear"])
    global_params = ParameterSet([Parameter("m", value=0.002, min=0.0, max=1.0, fixed=False)])
    local_params = {
        "g_a": ParameterSet([Parameter("b", value=0.18, min=-1.0, max=1.0, fixed=False)]),
        "g_b": ParameterSet([Parameter("b", value=0.23, min=-1.0, max=1.0, fixed=False)]),
    }
    fixed_params = ParameterSet()
    fit_result = CrossGroupFitResult(
        success=True,
        chi_squared=1.2,
        reduced_chi_squared=0.6,
        global_parameters=global_params,
        local_parameters=local_params,
        fixed_parameters=fixed_params,
        global_uncertainties={"m": 1e-4},
        local_uncertainties={"g_a": {"b": 0.01}, "g_b": {"b": 0.02}},
        message="Fit successful",
    )

    output = SimpleNamespace(
        fit_result=fit_result,
        model=model,
        x_key="field",
        fit_x_min=90.0,
        fit_x_max=250.0,
        groups=[
            ParameterGroupData(
                "g_a", "Group A", np.array([100.0]), np.array([0.1]), np.array([0.01]), 100.0
            ),
            ParameterGroupData(
                "g_b", "Group B", np.array([200.0]), np.array([0.14]), np.array([0.01]), 200.0
            ),
        ],
    )

    panel._apply_cross_group_fit_to_groups(
        parameter_name="Lambda",
        x_key="field",
        selected_groups=[group_a, group_b],
        output=output,
    )

    fit_a = panel._group_fit_results["g_a"].model_fits.get("Lambda")
    fit_b = panel._group_fit_results["g_b"].model_fits.get("Lambda")
    assert fit_a is not None
    assert fit_b is not None
    assert fit_a.ranges[0].result is not None
    assert fit_b.ranges[0].result is not None
    assert fit_a.ranges[0].result.parameters["b"].value == pytest.approx(0.18)
    assert fit_b.ranges[0].result.parameters["b"].value == pytest.approx(0.23)
    assert fit_a.ranges[0].result.parameters["m"].value == pytest.approx(0.002)
    assert fit_b.ranges[0].result.parameters["m"].value == pytest.approx(0.002)


def test_build_cross_group_group_model_fit_defaults_shape_factor_a_to_fixed(
    panel: FitParametersPanel,
) -> None:
    model = ParameterCompositeModel(["SC_PWaveAxial"])
    fit_result = CrossGroupFitResult(
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=0.5,
        global_parameters=ParameterSet([Parameter("sigma_0", value=1.0)]),
        local_parameters={"g1": ParameterSet([Parameter("Tc", value=20.0)])},
        fixed_parameters=ParameterSet(
            [
                Parameter("gap_ratio", value=2.0, fixed=True),
                Parameter("sigma_bg", value=0.0, fixed=True),
            ]
        ),
    )

    fit = panel._build_cross_group_group_model_fit(
        parameter_name="Lambda",
        x_key="temperature",
        group_id="g1",
        model=model,
        fit_result=fit_result,
        fit_x_min=1.0,
        fit_x_max=30.0,
    )

    shape_factor = fit.ranges[0].parameters["shape_factor_a"]
    assert shape_factor.value == 0.0
    assert shape_factor.fixed is True


def test_set_fit_results_copies_global_params_per_group(panel: FitParametersPanel) -> None:
    shared_global = ParameterSet([Parameter("A0", value=0.25, min=0.0, max=1.0, fixed=False)])

    result_g1 = FitResult(
        success=True,
        reduced_chi_squared=1.0,
        parameters=ParameterSet([Parameter("A0", value=0.24), Parameter("Lambda", value=0.11)]),
        uncertainties={"A0": 0.01, "Lambda": 0.01},
    )
    datasets_g1 = {
        101: SimpleNamespace(metadata={"run_label": "101", "field": 150.0, "temperature": 12.0})
    }
    panel.set_fit_results(
        {101: (result_g1, (np.array([0.0, 1.0]), np.array([0.24, 0.24])))},
        datasets_g1,
        global_params=shared_global,
        group_id="g1",
        group_name="Group 1",
    )

    # Simulate reuse/mutation of the same ParameterSet object by later fits.
    shared_global["A0"].value = 0.42

    result_g2 = FitResult(
        success=True,
        reduced_chi_squared=1.0,
        parameters=ParameterSet([Parameter("A0", value=0.30), Parameter("Lambda", value=0.12)]),
        uncertainties={"A0": 0.01, "Lambda": 0.01},
    )
    datasets_g2 = {
        202: SimpleNamespace(metadata={"run_label": "202", "field": 300.0, "temperature": 12.0})
    }
    panel.set_fit_results(
        {202: (result_g2, (np.array([0.0, 1.0]), np.array([0.30, 0.30])))},
        datasets_g2,
        global_params=shared_global,
        group_id="g2",
        group_name="Group 2",
    )

    g1_params = panel._group_fit_results["g1"].global_params
    g2_params = panel._group_fit_results["g2"].global_params
    assert g1_params is not None
    assert g2_params is not None
    assert g1_params["A0"].value == pytest.approx(0.25)
    assert g2_params["A0"].value == pytest.approx(0.42)


def test_click_group_switch_does_not_copy_previous_group_rows(
    panel: FitParametersPanel, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows_a = [
        _FitRow(
            run_number=11,
            run_label="11",
            field=101.0,
            temperature=0.1,
            values={"Lambda": 0.11},
            errors={"Lambda": 0.01},
        )
    ]
    rows_b = [
        _FitRow(
            run_number=22,
            run_label="22",
            field=202.0,
            temperature=2.0,
            values={"Lambda": 0.22},
            errors={"Lambda": 0.02},
        )
    ]

    panel._group_fit_results = {
        "g_a": _GroupFitData(
            group_id="g_a",
            group_name="T = 0.1 K",
            rows=rows_a,
            global_params=None,
            varying_params=["Lambda"],
            inferred_x_key="field",
            model_fits={},
            plot_annotations=[],
        ),
        "g_b": _GroupFitData(
            group_id="g_b",
            group_name="T = 2 K",
            rows=rows_b,
            global_params=None,
            varying_params=["Lambda"],
            inferred_x_key="field",
            model_fits={},
            plot_annotations=[],
        ),
    }
    panel._active_group_id = "g_a"
    panel._rows = list(rows_a)
    panel._varying_params = ["Lambda"]
    panel._rebuild_group_buttons()
    panel._set_selected_group_ids(["g_a"], emit=False)

    monkeypatch.setattr(
        "asymmetry.gui.panels.fit_parameters_panel.QApplication.keyboardModifiers",
        lambda: Qt.KeyboardModifier.NoModifier,
    )

    panel._group_button_map["g_b"].click()

    # Active view now shows group B's own rows.
    assert panel._active_group_id == "g_b"
    assert len(panel._rows) == 1
    assert panel._rows[0].run_number == 22

    # Group B storage remains intact and was not overwritten by group A rows.
    assert len(panel._group_fit_results["g_b"].rows) == 1
    assert panel._group_fit_results["g_b"].rows[0].run_number == 22


def test_group_switch_keeps_selected_y_parameter_when_available(panel: FitParametersPanel) -> None:
    rows_a = [
        _FitRow(
            run_number=11,
            run_label="11",
            field=101.0,
            temperature=0.1,
            values={"Lambda": 0.11, "A0": 0.21},
            errors={"Lambda": 0.01, "A0": 0.01},
        ),
        _FitRow(
            run_number=12,
            run_label="12",
            field=102.0,
            temperature=0.1,
            values={"Lambda": 0.12, "A0": 0.22},
            errors={"Lambda": 0.01, "A0": 0.01},
        ),
    ]
    rows_b = [
        _FitRow(
            run_number=21,
            run_label="21",
            field=201.0,
            temperature=2.0,
            values={"Lambda": 0.21, "A0": 0.31},
            errors={"Lambda": 0.02, "A0": 0.02},
        ),
        _FitRow(
            run_number=22,
            run_label="22",
            field=202.0,
            temperature=2.0,
            values={"Lambda": 0.22, "A0": 0.32},
            errors={"Lambda": 0.02, "A0": 0.02},
        ),
    ]

    panel._group_fit_results = {
        "g_a": _GroupFitData(
            group_id="g_a",
            group_name="T = 0.1 K",
            rows=rows_a,
            global_params=None,
            varying_params=["A0", "Lambda"],
            inferred_x_key="field",
            model_fits={},
            plot_annotations=[],
        ),
        "g_b": _GroupFitData(
            group_id="g_b",
            group_name="T = 2 K",
            rows=rows_b,
            global_params=None,
            varying_params=["A0", "Lambda"],
            inferred_x_key="field",
            model_fits={},
            plot_annotations=[],
        ),
    }
    panel._active_group_id = "g_a"
    panel._rebuild_group_buttons()
    panel._set_selected_group_ids(["g_a"], emit=False)
    panel._apply_group_selection_to_view(sync_active=False)

    # Select Lambda in group A.
    select_params(panel, ["Lambda"])
    assert panel._selected_y_parameters() == ["Lambda"]

    # Switch to group B; Lambda exists there too and should stay selected.
    panel._set_selected_group_ids(["g_b"], emit=False)
    panel._active_group_id = "g_b"
    panel._apply_group_selection_to_view()
    assert panel._selected_y_parameters() == ["Lambda"]


def test_group_variable_value_for_rows_uses_complementary_axis(panel: FitParametersPanel) -> None:
    rows = [
        _FitRow(
            run_number=1,
            run_label="1",
            field=20.0,
            temperature=20.10,
            values={"Lambda": 0.1},
            errors={"Lambda": 0.01},
        ),
        _FitRow(
            run_number=2,
            run_label="2",
            field=20.0,
            temperature=20.13,
            values={"Lambda": 0.1},
            errors={"Lambda": 0.01},
        ),
    ]

    gv_field_fit = panel._group_variable_value_for_rows(rows, "field")
    gv_temp_fit = panel._group_variable_value_for_rows(rows, "temperature")

    # For field-based fits, group variable should be temperature.
    assert gv_field_fit == pytest.approx((20.10 + 20.13) / 2.0)
    # For temperature-based fits, group variable should be field.
    assert gv_temp_fit == pytest.approx(20.0)


def test_composite_parameter_materializes_and_gets_a_chip_and_card(
    panel: FitParametersPanel,
) -> None:
    panel._composite_parameters = [
        CompositeParameterDefinition(name="Lambda_eff", expression="sqrt(A0^2 + Lambda^2)")
    ]
    panel._apply_composite_parameters_to_rows(panel._rows, panel._composite_parameters)
    panel._rebuild_y_controls(preferred_selected=["Lambda_eff"])

    assert "Lambda_eff" in panel._display_y_parameters()
    assert np.isfinite(panel._rows[0].values["Lambda_eff"])

    assert "Lambda_eff" in panel._y_chips
    assert panel._y_chips["Lambda_eff"].isChecked()
    # A composite gets the "derived" tag on its card.
    assert card(panel, "Lambda_eff").findChild(QLabel, CONTEXT_CHIP_OBJECT_NAME) is not None


def test_open_composite_dialog_adds_definition_and_values(
    panel: FitParametersPanel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeDialog:
        def __init__(self, **kwargs):
            self._definition = CompositeParameterDefinition(
                name="A0_plus_Lambda",
                expression="A0 + Lambda",
            )

        def exec(self) -> int:
            return int(QDialog.DialogCode.Accepted)

        def composite_definition(self) -> CompositeParameterDefinition:
            return self._definition

    monkeypatch.setattr(
        "asymmetry.gui.panels.fit_parameters_panel.CompositeParameterDialog",
        _FakeDialog,
    )

    panel._open_composite_parameter_dialog()

    assert any(defn.name == "A0_plus_Lambda" for defn in panel._composite_parameters)
    assert np.isfinite(panel._rows[0].values["A0_plus_Lambda"])
    assert "A0_plus_Lambda" in panel._display_y_parameters()


def test_set_fit_results_preserves_group_composite_definitions(
    panel: FitParametersPanel,
) -> None:
    definition = CompositeParameterDefinition(name="sum_param", expression="A0 + Lambda")
    panel._group_fit_results = {
        "g1": _GroupFitData(
            group_id="g1",
            group_name="Group 1",
            rows=[],
            global_params=None,
            varying_params=["A0", "Lambda"],
            inferred_x_key="field",
            model_fits={},
            plot_annotations=[],
            composite_parameters=[definition],
        )
    }

    result = FitResult(
        success=True,
        reduced_chi_squared=1.0,
        parameters=ParameterSet([Parameter("A0", value=0.24), Parameter("Lambda", value=0.11)]),
        uncertainties={"A0": 0.01, "Lambda": 0.02},
    )
    datasets = {
        101: SimpleNamespace(
            metadata={
                "run_label": "101",
                "field": 150.0,
                "temperature": 12.0,
            }
        )
    }

    panel.set_fit_results(
        {101: (result, (np.array([0.0, 1.0]), np.array([0.24, 0.24])))},
        datasets,
        global_params=None,
        group_id="g1",
        group_name="Group 1",
    )

    stored = panel._group_fit_results["g1"]
    assert any(defn.name == "sum_param" for defn in stored.composite_parameters)
    assert np.isfinite(stored.rows[0].values["sum_param"])


def test_edit_composite_parameter_updates_definition(
    panel: FitParametersPanel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel._composite_parameters = [
        CompositeParameterDefinition(name="sum_param", expression="A0 + Lambda")
    ]
    panel._apply_composite_parameters_to_rows(panel._rows, panel._composite_parameters)
    panel._rebuild_y_controls(preferred_selected=["sum_param"])

    class _FakeDialog:
        def __init__(self, **kwargs):
            self._definition = CompositeParameterDefinition(
                name="sum_param_edited",
                expression="A0 - Lambda",
            )

        def exec(self) -> int:
            return int(QDialog.DialogCode.Accepted)

        def composite_definition(self) -> CompositeParameterDefinition:
            return self._definition

    monkeypatch.setattr(
        "asymmetry.gui.panels.fit_parameters_panel.CompositeParameterDialog",
        _FakeDialog,
    )

    panel._edit_composite_parameter("sum_param")

    names = [definition.name for definition in panel._composite_parameters]
    assert "sum_param" not in names
    assert "sum_param_edited" in names
    assert "sum_param" not in panel._rows[0].values
    assert np.isfinite(panel._rows[0].values["sum_param_edited"])


def test_remove_derived_parameter_drops_values(
    panel: FitParametersPanel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel._composite_parameters = [
        CompositeParameterDefinition(name="sum_param", expression="A0 + Lambda")
    ]
    panel._apply_composite_parameters_to_rows(panel._rows, panel._composite_parameters)
    panel._rebuild_y_controls(preferred_selected=["sum_param"])

    monkeypatch.setattr(
        "asymmetry.gui.panels.fit_parameters_panel.QMessageBox.question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )

    panel._remove_derived_parameters(["sum_param"])

    assert panel._composite_parameters == []
    assert "sum_param" not in panel._rows[0].values
    assert "sum_param" not in panel._rows[0].errors


def test_apply_cross_group_fit_keeps_existing_composite_definitions(
    panel: FitParametersPanel,
) -> None:
    composite = CompositeParameterDefinition(name="sum_param", expression="A0 + Lambda")
    model = ParameterCompositeModel(["Linear"])
    fit_result = CrossGroupFitResult(
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=0.5,
        global_parameters=ParameterSet([Parameter("m", value=0.002)]),
        local_parameters={"g1": ParameterSet([Parameter("b", value=0.18)])},
        fixed_parameters=ParameterSet(),
        global_uncertainties={"m": 0.0},
        local_uncertainties={"g1": {"b": 0.01}},
    )

    group = _GroupFitData(
        group_id="g1",
        group_name="Group 1",
        rows=list(panel._rows),
        global_params=None,
        varying_params=list(panel._varying_params),
        inferred_x_key="field",
        model_fits={},
        plot_annotations=[],
        composite_parameters=[composite],
    )
    panel._group_fit_results = {"g1": group}

    output = SimpleNamespace(
        fit_result=fit_result,
        model=model,
        fit_x_min=100.0,
        fit_x_max=200.0,
    )

    panel._apply_cross_group_fit_to_groups(
        parameter_name="Lambda",
        x_key="field",
        selected_groups=[group],
        output=output,
    )

    updated = panel._group_fit_results["g1"]
    assert any(defn.name == "sum_param" for defn in updated.composite_parameters)


# ── Series context menu and new signals ─────────────────────────────────────


def _panel_with_two_groups(qapp: QApplication) -> FitParametersPanel:
    rows = [
        _FitRow(
            run_number=101,
            run_label="101",
            field=100.0,
            temperature=5.0,
            values={"A": 0.2},
            errors={"A": 0.01},
        )
    ]
    panel = FitParametersPanel()
    panel._group_fit_results = {
        "g1": _GroupFitData(
            group_id="g1",
            group_name="Series 1",
            rows=rows,
            global_params=None,
            varying_params=["A"],
            inferred_x_key="field",
            model_fits={},
            plot_annotations=[],
        ),
        "g2": _GroupFitData(
            group_id="g2",
            group_name="Series 2",
            rows=list(rows),
            global_params=None,
            varying_params=["A"],
            inferred_x_key="field",
            model_fits={},
            plot_annotations=[],
        ),
    }
    panel._active_group_id = "g1"
    panel._rebuild_group_buttons()
    panel._set_selected_group_ids(["g1"], emit=False)
    # Refresh styles AFTER setting the checked state.
    panel._refresh_group_button_styles()
    return panel


def test_series_buttons_use_red_palette(qapp: QApplication) -> None:
    """Active series button stylesheet must include the ACCENT_RED token."""
    from asymmetry.gui.styles import tokens

    panel = _panel_with_two_groups(qapp)
    active_btn = panel._group_button_map["g1"]
    assert tokens.ACCENT_RED in active_btn.styleSheet()


def _panel_with_long_series(qapp: QApplication, count: int = 3) -> FitParametersPanel:
    """A panel holding *count* series with long full names and run-range short names."""
    panel = FitParametersPanel()
    panel._group_fit_results = {
        f"s{i}": _GroupFitData(
            group_id=f"s{i}",
            group_name=f"StretchedExponential + Constant · {390 + i}–{393 + i} · high",
            short_name=f"{390 + i}–{393 + i}",
            rows=[
                _FitRow(
                    run_number=390 + i,
                    run_label=str(390 + i),
                    field=100.0 * i,
                    temperature=5.0,
                    values={"A": 0.2},
                    errors={"A": 0.01},
                )
            ],
            global_params=None,
            varying_params=["A"],
            inferred_x_key="field",
            model_fits={},
            plot_annotations=[],
        )
        for i in range(count)
    }
    panel._active_group_id = "s0"
    panel._rebuild_group_buttons()
    panel._set_selected_group_ids(["s0"], emit=False)
    panel._refresh_group_button_styles()
    return panel


def test_series_pill_shows_short_name_with_full_name_on_tooltip(qapp: QApplication) -> None:
    panel = _panel_with_long_series(qapp)
    button = panel._group_button_map["s0"]
    assert button.text() == "390–393"
    assert button.toolTip().splitlines() == [
        "StretchedExponential + Constant · 390–393 · high",
        "Click to view this series · Shift+click to overlay it with the selected series.",
    ]
    # The pill is a short handle: it never takes the full name's width.
    long_pill = QPushButton(panel._group_fit_results["s0"].group_name)
    assert button.sizeHint().width() < long_pill.sizeHint().width() / 2


def test_series_pill_elides_a_long_short_name(qapp: QApplication) -> None:
    panel = _panel_with_long_series(qapp, count=1)
    panel._group_fit_results["s0"].short_name = "groups 100000–100031 extra"
    panel._rebuild_group_buttons()

    text = panel._group_button_map["s0"].text()
    # Elided in the middle so the tail (a default name's fit range) survives.
    assert "…" in text
    assert text.endswith("extra")
    assert len(text) < len("groups 100000–100031 extra")
    # The full name — not the truncated short one — is what hover reveals.
    assert (
        panel._group_button_map["s0"]
        .toolTip()
        .startswith("StretchedExponential + Constant · 390–393 · high")
    )


def test_stale_series_pill_keeps_glyph_and_warning_on_the_short_pill(qapp: QApplication) -> None:
    panel = _panel_with_long_series(qapp, count=1)
    panel._stale_series_ids = {"s0"}
    panel._rebuild_group_buttons()

    button = panel._group_button_map["s0"]
    assert button.text() == "390–393 ⚠"
    assert button.toolTip().splitlines()[0] == (
        "Membership changed since last fit — re-run to refresh."
    )
    assert button.toolTip().splitlines()[1] == "StretchedExponential + Constant · 390–393 · high"


def test_series_pill_falls_back_to_full_name_without_short_names(
    panel: FitParametersPanel,
) -> None:
    """Every caller that omits ``short_names_by_id`` keeps today's full-name pill."""
    row_dicts = [
        {
            "run_number": 1,
            "run_label": "1",
            "field": 100.0,
            "temperature": 10.0,
            "values": {"Lambda": 0.1},
            "errors": {"Lambda": 0.01},
        }
    ]
    # Short enough to sit under the elision cap in any font (CI's differs from macOS).
    panel.load_representation_series([("batch-1", "Fast · 1", row_dicts)])
    assert panel._group_button_map["batch-1"].text() == "Fast · 1"

    panel.load_representation_series(
        [("batch-1", "Fast damped · 1", row_dicts)],
        short_names_by_id={"batch-1": "1"},
    )
    assert panel._group_button_map["batch-1"].text() == "1"


def _row_dicts_for(run_number: int) -> list[dict]:
    return [
        {
            "run_number": run_number,
            "run_label": str(run_number),
            "field": 100.0,
            "temperature": 10.0,
            "values": {"Lambda": 0.1},
            "errors": {"Lambda": 0.01},
        }
    ]


def test_sections_group_chips_under_a_swatched_header(panel: FitParametersPanel) -> None:
    """A section header carries the group's kind swatch and name (item 1)."""
    from asymmetry.gui.styles.widgets import SECTION_HEADER_OBJECT_NAME

    panel.load_representation_series(
        [
            ("batch-1", "Series 1", _row_dicts_for(1)),
            ("batch-2", "Series 2", _row_dicts_for(2)),
            ("batch-3", "Series 3", _row_dicts_for(3)),
        ],
        sections=[
            ("T scan — EuO", tokens.GROUP_HEADER_BG, ["batch-1", "batch-2"]),
            ("Standalone", None, ["batch-3"]),
        ],
    )
    # header, chip row, header, chip row: two sections, each with its own.
    assert panel._group_tabs_layout.count() == 4

    header1 = panel._group_tabs_layout.itemAt(0).widget()
    label1 = header1.findChild(QLabel, SECTION_HEADER_OBJECT_NAME)
    assert label1.text() == "T SCAN — EUO"
    # Swatch + name label + trailing stretch: the swatch is there for a real group.
    assert header1.layout().count() == 3

    row1 = panel._group_tabs_layout.itemAt(1).widget()
    assert isinstance(row1.layout(), FlowLayout)
    chips_in_row1 = [row1.layout().itemAt(i).widget() for i in range(row1.layout().count())]
    assert chips_in_row1 == [panel._group_button_map["batch-1"], panel._group_button_map["batch-2"]]

    header2 = panel._group_tabs_layout.itemAt(2).widget()
    label2 = header2.findChild(QLabel, SECTION_HEADER_OBJECT_NAME)
    assert label2.text() == "STANDALONE"
    # No swatch for a group-less section: name label + trailing stretch only.
    assert header2.layout().count() == 2


def test_sections_header_hidden_for_a_group_less_project(panel: FitParametersPanel) -> None:
    """A single "Standalone" section (no groups at all) shows no header (item 1)."""
    panel.load_representation_series(
        [("batch-1", "Series 1", _row_dicts_for(1))],
        sections=[("Standalone", None, ["batch-1"])],
    )
    # Just the one chip row — no header widget precedes it.
    assert panel._group_tabs_layout.count() == 1
    row = panel._group_tabs_layout.itemAt(0).widget()
    assert isinstance(row.layout(), FlowLayout)


def test_series_strip_wraps_instead_of_widening(qapp: QApplication) -> None:
    panel = _panel_with_long_series(qapp, count=4)
    # A group-less project is one "Standalone" section: no header, one row
    # widget wrapping the pills in a FlowLayout (item 1).
    assert panel._group_tabs_layout.count() == 1
    row_widget = panel._group_tabs_layout.itemAt(0).widget()
    layout = row_widget.layout()
    assert isinstance(layout, FlowLayout)

    one_row = layout.heightForWidth(1000)
    assert layout.heightForWidth(180) > one_row
    # A strip of four pills asks for the width of one, not of the row.
    assert layout.minimumSize().width() < 150


def test_footer_names_the_active_series_and_counts_an_overlay(qapp: QApplication) -> None:
    panel = _panel_with_long_series(qapp)
    label = panel._active_series_label
    assert label.text() == "StretchedExponential + Constant · 390–393 · high"

    panel._set_selected_group_ids(["s0", "s1"], emit=False)
    panel._apply_group_selection_to_view(sync_active=False)
    assert label.text() == "2 series"
    assert label.toolTip().splitlines() == [
        "StretchedExponential + Constant · 390–393 · high",
        "StretchedExponential + Constant · 391–394 · high",
    ]


def test_footer_series_line_hidden_without_a_series(qapp: QApplication) -> None:
    panel = FitParametersPanel()
    assert panel._active_series_label.text() == ""
    assert panel._active_series_label.isHidden()


def test_footer_global_note_keeps_its_size_hint_at_dock_width(qapp: QApplication) -> None:
    """At a 13-inch dock width the series line elides; the Global note does not shrink."""
    panel = _panel_with_long_series(qapp)
    panel._group_fit_results[
        "s0"
    ].group_name = (
        "StretchedExponential + StretchedExponential + Constant · 394–397 · high transverse field"
    )
    panel._rebuild_group_buttons()
    panel._set_selected_group_ids(["s0"], emit=False)
    panel._refresh_group_button_styles()
    panel._global_params = ParameterSet(
        [Parameter("A_1", value=0.2), Parameter("A_bg", value=0.05)]
    )
    panel._update_global_param_hint()
    panel.show()
    panel.resize(376, 700)
    qapp.processEvents()

    note = panel._global_param_hint
    assert note.text() == "A_1 (%), A_bg (%) Global — held constant"
    assert note.width() == note.sizeHint().width()
    series_line = panel._active_series_label
    assert series_line.width() < series_line.fontMetrics().horizontalAdvance(series_line.text())
    assert series_line.toolTip() == series_line.text()
    panel.close()


def _capture_menu(captured: list) -> object:
    def _exec(menu, _pos):
        captured.append(menu)
        return None

    return _exec


def test_context_menu_action_order(qapp: QApplication) -> None:
    """The six actions appear in the documented order, with the two separators."""
    panel = _panel_with_two_groups(qapp)
    captured: list = []
    panel._exec_menu = _capture_menu(captured)  # type: ignore[method-assign]
    panel._show_group_button_context_menu("g1", panel._group_button_map["g1"], QPoint(0, 0))

    menu = captured[0]
    texts = [a.text() if not a.isSeparator() else "---" for a in menu.actions()]
    assert texts == [
        "Open in Batch tab",
        "Duplicate…",
        "Rename…",
        "---",
        "Select members in browser",
        "Show fit overlay",
        "---",
        "Delete series…",
    ]


def test_context_menu_open_emits_signal(qapp: QApplication) -> None:
    panel = _panel_with_two_groups(qapp)
    emitted: list[str] = []
    panel.series_open_requested.connect(lambda gid: emitted.append(gid))

    panel._exec_menu = lambda menu, pos: menu.actions()[0]  # type: ignore[method-assign]
    panel._show_group_button_context_menu("g1", panel._group_button_map["g1"], QPoint(0, 0))
    assert emitted == ["g1"]


def test_context_menu_duplicate_emits_signal(qapp: QApplication) -> None:
    panel = _panel_with_two_groups(qapp)
    emitted: list[str] = []
    panel.series_duplicate_requested.connect(lambda gid: emitted.append(gid))

    panel._exec_menu = lambda menu, pos: menu.actions()[1]  # type: ignore[method-assign]
    panel._show_group_button_context_menu("g1", panel._group_button_map["g1"], QPoint(0, 0))
    assert emitted == ["g1"]


def test_double_click_a_chip_emits_open_requested(qapp: QApplication) -> None:
    """A chip double-click is the same gesture as the menu's first action."""
    panel = _panel_with_two_groups(qapp)
    emitted: list[str] = []
    panel.series_open_requested.connect(lambda gid: emitted.append(gid))

    button = panel._group_button_map["g1"]
    event = QEvent(QEvent.Type.MouseButtonDblClick)
    panel.eventFilter(button, event)
    assert emitted == ["g1"]


def test_context_menu_rename_emits_signal(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QInputDialog

    panel = _panel_with_two_groups(qapp)
    emitted: list[tuple[str, str]] = []
    panel.series_rename_requested.connect(lambda gid, lbl: emitted.append((gid, lbl)))

    # Patch _exec_menu at instance level (class-level QMenu.exec patch bypassed by PySide6).
    panel._exec_menu = lambda menu, pos: menu.actions()[2]  # type: ignore[method-assign]
    monkeypatch.setattr(
        QInputDialog,
        "getText",
        lambda *_a, **_kw: ("My label", True),
    )
    panel._show_group_button_context_menu("g1", panel._group_button_map["g1"], QPoint(0, 0))

    assert emitted == [("g1", "My label")]


def test_context_menu_rename_cancel_emits_nothing(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QInputDialog

    panel = _panel_with_two_groups(qapp)
    emitted: list = []
    panel.series_rename_requested.connect(lambda *a: emitted.append(a))

    panel._exec_menu = lambda menu, pos: menu.actions()[2]  # type: ignore[method-assign]
    monkeypatch.setattr(QInputDialog, "getText", lambda *_a, **_kw: ("", False))

    panel._show_group_button_context_menu("g1", panel._group_button_map["g1"], QPoint(0, 0))
    assert emitted == []


def test_context_menu_select_members_emits_signal(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    panel = _panel_with_two_groups(qapp)
    emitted: list[str] = []
    panel.series_select_members_requested.connect(lambda gid: emitted.append(gid))

    panel._exec_menu = lambda menu, pos: menu.actions()[4]  # type: ignore[method-assign]
    panel._show_group_button_context_menu("g1", panel._group_button_map["g1"], QPoint(0, 0))
    assert emitted == ["g1"]


def test_context_menu_show_overlay_activates_series(qapp: QApplication) -> None:
    """ "Show fit overlay" on the inactive series behaves exactly like a chip press."""
    panel = _panel_with_two_groups(qapp)
    emitted: list[str] = []
    panel.series_selection_changed.connect(lambda gid: emitted.append(gid))
    assert panel._active_group_id == "g1"

    panel._exec_menu = lambda menu, pos: menu.actions()[5]  # type: ignore[method-assign]
    panel._show_group_button_context_menu("g2", panel._group_button_map["g2"], QPoint(0, 0))

    assert emitted == ["g2"]
    assert panel._active_group_id == "g2"


def test_context_menu_show_overlay_disabled_for_the_active_series(qapp: QApplication) -> None:
    panel = _panel_with_two_groups(qapp)
    captured: list = []
    panel._exec_menu = _capture_menu(captured)  # type: ignore[method-assign]
    panel._show_group_button_context_menu("g1", panel._group_button_map["g1"], QPoint(0, 0))

    overlay_action = captured[0].actions()[5]
    assert overlay_action.text() == "Show fit overlay"
    assert overlay_action.isChecked()
    assert not overlay_action.isEnabled()


def test_context_menu_delete_confirm_emits_signal(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    panel = _panel_with_two_groups(qapp)
    emitted: list[str] = []
    panel.series_delete_requested.connect(lambda gid: emitted.append(gid))

    _accept_message_box(monkeypatch)
    panel._exec_menu = lambda menu, pos: menu.actions()[7]  # type: ignore[method-assign]
    panel._show_group_button_context_menu("g1", panel._group_button_map["g1"], QPoint(0, 0))
    assert emitted == ["g1"]


def test_context_menu_delete_cancel_emits_nothing(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    panel = _panel_with_two_groups(qapp)
    emitted: list = []
    panel.series_delete_requested.connect(lambda *a: emitted.append(a))

    _accept_message_box(monkeypatch, accept=False)
    panel._exec_menu = lambda menu, pos: menu.actions()[7]  # type: ignore[method-assign]
    panel._show_group_button_context_menu("g1", panel._group_button_map["g1"], QPoint(0, 0))
    assert emitted == []


def test_set_highlight_active_reemits_series_selection_changed(qapp: QApplication) -> None:
    panel = _panel_with_two_groups(qapp)
    emitted: list[str] = []
    panel.series_selection_changed.connect(lambda gid: emitted.append(gid))

    panel.set_highlight_active(True)
    assert emitted == ["g1"]


def test_set_highlight_active_false_emits_nothing(qapp: QApplication) -> None:
    panel = _panel_with_two_groups(qapp)
    emitted: list = []
    panel.series_selection_changed.connect(lambda *a: emitted.append(a))

    panel.set_highlight_active(False)
    assert emitted == []


def test_set_highlight_active_no_active_group_emits_nothing(qapp: QApplication) -> None:
    panel = FitParametersPanel()
    emitted: list = []
    panel.series_selection_changed.connect(lambda *a: emitted.append(a))

    panel.set_highlight_active(True)
    assert emitted == []


def test_model_fit_serialization_round_trips_windows_and_error_mode(
    panel: FitParametersPanel,
) -> None:
    """Fit windows and the result's error_mode/n_points must survive a
    save/reload cycle — silently dropping them leaves a windowed or
    scatter-mode fit mislabelled as a plain column-error full-range fit."""
    import json

    model = ParameterCompositeModel(["Constant"])
    params = ParameterSet([Parameter("c", value=0.21)])
    result = ParameterModelFitResult(
        success=True,
        chi_squared=4.2,
        reduced_chi_squared=0.6,
        parameters=params,
        uncertainties={"c": 0.01},
        message="Fit successful",
        error_mode="scatter",
        n_points=9,
    )
    panel._model_fits = {
        "A0": ParameterModelFit(
            parameter_name="A0",
            x_key="field",
            active=True,
            ranges=[
                ModelFitRange(
                    x_min=0.0,
                    x_max=10.0,
                    model=model,
                    parameters=params,
                    result=result,
                    windows=[(1.0, 4.0), (7.0, 10.0)],
                )
            ],
        )
    }

    # JSON round-trip mirrors project persistence (tuples become lists).
    state = json.loads(json.dumps(panel._serialize_model_fits()))
    restored = panel._deserialize_model_fits(state)

    fit_range = restored["A0"].ranges[0]
    assert fit_range.windows == [(1.0, 4.0), (7.0, 10.0)]
    assert fit_range.result is not None
    assert fit_range.result.error_mode == "scatter"
    assert fit_range.result.n_points == 9

    # Legacy state without the new keys still loads (defaults applied).
    for range_state in state["A0"]["ranges"]:
        range_state.pop("windows", None)
        range_state["result"].pop("error_mode", None)
        range_state["result"].pop("n_points", None)
    legacy = panel._deserialize_model_fits(state)
    legacy_range = legacy["A0"].ranges[0]
    assert legacy_range.windows is None
    assert legacy_range.result.error_mode == "column"
    assert legacy_range.result.n_points == 0


def test_model_fit_serialization_round_trips_covariance_and_params_at_bound(
    panel: FitParametersPanel,
) -> None:
    """``covariance`` and ``params_at_bound`` must survive a save/reload cycle:
    both feed the ill-conditioning warning banner (BED next-point suggestion,
    docs/studies/bed-next-point-suggestion.md §3.4/§5.2), and params_at_bound
    was previously silently dropped by the serializer."""
    import json

    model = ParameterCompositeModel(["Linear"])
    params = ParameterSet([Parameter("a", value=1.0), Parameter("b", value=2.0)])
    result = ParameterModelFitResult(
        success=True,
        chi_squared=4.2,
        reduced_chi_squared=0.6,
        parameters=params,
        uncertainties={"a": 0.1, "b": 0.2},
        message="Fit successful",
        error_mode="column",
        n_points=9,
        params_at_bound=("a",),
        covariance=(["a", "b"], [[0.01, 0.002], [0.002, 0.04]]),
    )
    panel._model_fits = {
        "A0": ParameterModelFit(
            parameter_name="A0",
            x_key="field",
            active=True,
            ranges=[
                ModelFitRange(
                    x_min=0.0,
                    x_max=10.0,
                    model=model,
                    parameters=params,
                    result=result,
                )
            ],
        )
    }

    # JSON round-trip mirrors project persistence (tuples become lists).
    state = json.loads(json.dumps(panel._serialize_model_fits()))
    restored = panel._deserialize_model_fits(state)

    fit_range = restored["A0"].ranges[0]
    assert fit_range.result is not None
    assert fit_range.result.params_at_bound == ("a",)
    assert fit_range.result.covariance is not None
    cov_names, cov_matrix = fit_range.result.covariance
    assert cov_names == ["a", "b"]
    assert cov_matrix == [[0.01, 0.002], [0.002, 0.04]]

    # Legacy state without either key still loads (defaults applied).
    for range_state in state["A0"]["ranges"]:
        range_state["result"].pop("params_at_bound", None)
        range_state["result"].pop("covariance", None)
    legacy = panel._deserialize_model_fits(state)
    legacy_range = legacy["A0"].ranges[0]
    assert legacy_range.result is not None
    assert legacy_range.result.params_at_bound == ()
    assert legacy_range.result.covariance is None


def test_fit_range_curve_sampler_spans_window_envelope(panel: FitParametersPanel) -> None:
    """The panel overlay/GLE sampler must cover the window-union envelope,
    not the stale x_min/x_max, so the drawn curve matches what was fitted."""
    model = ParameterCompositeModel(["Constant"])
    params = ParameterSet([Parameter("c", value=0.21)])
    result = ParameterModelFitResult(success=True, parameters=params)
    fit_range = ModelFitRange(
        x_min=2.0,
        x_max=5.0,
        model=model,
        parameters=params,
        result=result,
        windows=[(1.0, 3.0), (7.0, 12.0)],
    )

    sampled = panel._sample_fit_range_curve(fit_range, x_key="field")
    assert sampled is not None
    xs, _ys = sampled
    assert xs[0] == pytest.approx(1.0)
    assert xs[-1] == pytest.approx(12.0)

    # Invalid (inverted) windows skip the curve instead of raising.
    fit_range.windows = [(5.0, 1.0)]
    assert panel._sample_fit_range_curve(fit_range, x_key="field") is None


def test_trend_curves_recompute_off_thread_behind_overlay(
    panel: FitParametersPanel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An active trend fit recomputes its overlay off-thread under the overlay."""
    panel._model_fits = {"A0": _active_linear_fit("A0")}
    monkeypatch.setattr(panel, "_selected_y_parameters", lambda: ["A0"])

    panel._start_trend_curve_compute()
    # Mid-flight: the overlay covers the plot and the compute is active.
    assert panel._trend_curve_compute_active
    assert panel._trend_overlay is not None and not panel._trend_overlay.isHidden()

    wait_for(
        lambda: not panel._trend_curve_compute_active,
        QApplication.instance(),
        timeout_s=10.0,
    )

    # Landed: overlay cleared, curves cached (keyed by signature) for reuse, and
    # the model curve drawn.
    assert not panel._trend_curve_compute_active
    assert panel._trend_overlay.isHidden()
    assert isinstance(panel._precomputed_trend_curves, dict)
    assert "A0" in panel._precomputed_trend_curves
    assert panel._trend_cache_sig is not None
    # Subplots is the default mode, so the curve lands on A0's own card figure.
    assert axes_for(panel, "A0").get_lines(), (
        "the trend overlay curve should be drawn after the compute"
    )


def test_trend_compute_without_active_fit_draws_synchronously(
    panel: FitParametersPanel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no active overlay to evaluate, the scatter draws synchronously."""
    panel._model_fits = {}
    monkeypatch.setattr(panel, "_selected_y_parameters", lambda: ["A0"])

    panel._start_trend_curve_compute()

    assert not panel._trend_curve_compute_active
    assert panel._tasks.active_count == 0
    assert panel._trend_overlay is not None and panel._trend_overlay.isHidden()


def test_shutdown_workers_joins_trend_compute(
    panel: FitParametersPanel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """shutdown_workers tears the trend worker down within the bounded wait."""
    panel._model_fits = {"A0": _active_linear_fit("A0")}
    monkeypatch.setattr(panel, "_selected_y_parameters", lambda: ["A0"])

    panel._start_trend_curve_compute()
    assert panel._trend_curve_compute_active

    panel.shutdown_workers()
    assert panel._tasks.active_count == 0


def test_restore_state_defers_trend_overlay_to_async(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """restore_state must not evaluate the (heavy) trend overlay synchronously.

    Regression: a project with a slow trend model (e.g. DiffusionLF_2D, which
    runs scipy quadrature per sample) froze the GUI for seconds after the file
    loader closed, because restore_state's intermediate refreshes drew the
    overlay inline. The draw must be suppressed during restore and deferred to a
    single off-thread recompute.
    """
    src = FitParametersPanel()
    src._rows = [
        _FitRow(
            run_number=1,
            run_label="1",
            field=100.0,
            temperature=10.0,
            values={"Lambda": 0.10},
            errors={"Lambda": 0.01},
        ),
        _FitRow(
            run_number=2,
            run_label="2",
            field=200.0,
            temperature=10.0,
            values={"Lambda": 0.20},
            errors={"Lambda": 0.01},
        ),
    ]
    src._varying_params = ["Lambda"]
    src._inferred_x_key = "field"
    src._model_fits = {"Lambda": _active_linear_fit("Lambda")}
    src._selected_y_param_names = ["Lambda"]
    state = src.get_state()

    target = FitParametersPanel()
    monkeypatch.setattr(target, "_selected_y_parameters", lambda: ["Lambda"])
    drew: list[int] = []
    monkeypatch.setattr(target, "_draw_model_overlay_mpl", lambda *a, **k: drew.append(1))

    target.restore_state(state)

    # The synchronous restore drew no overlay inline and re-enabled plotting…
    assert drew == []
    assert target._suspend_plot_refresh is False
    # …it deferred the heavy evaluation to an off-thread recompute.
    assert target._trend_curve_compute_active

    wait_for(
        lambda: not target._trend_curve_compute_active,
        QApplication.instance(),
        timeout_s=10.0,
    )
    assert drew, "the overlay should be drawn once the async recompute lands"
    target.shutdown_workers()


def test_restore_state_clears_suspend_flag_on_exception(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mid-restore exception must not leave the plot permanently suspended.

    Regression: _suspend_plot_refresh is set for the duration of restore_state;
    without the try/finally an exception (malformed project) would wedge it True,
    silently disabling every subsequent plot refresh for the session.
    """
    panel = FitParametersPanel()

    def _boom(*_a, **_k):
        raise RuntimeError("malformed project")

    monkeypatch.setattr(panel, "_rebuild_y_controls", _boom)

    with pytest.raises(RuntimeError):
        panel.restore_state({"rows": []})

    assert panel._suspend_plot_refresh is False, "suspend guard must clear after a failed restore"


def test_pure_render_refresh_reuses_trend_cache(
    panel: FitParametersPanel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A redraw with unchanged inputs reuses the cached curves (no new worker)."""
    panel._model_fits = {"A0": _active_linear_fit("A0")}
    monkeypatch.setattr(panel, "_selected_y_parameters", lambda: ["A0"])

    panel._refresh_plot()
    assert panel._trend_curve_compute_active
    wait_for(lambda: not panel._trend_curve_compute_active, QApplication.instance(), timeout_s=10.0)
    sig1 = panel._trend_cache_sig
    assert sig1 is not None

    # A pure-render refresh (e.g. a log-scale toggle) does not change the
    # signature, so it draws from the cache synchronously without starting a new
    # compute (a cache hit never sets the busy flag).
    panel._refresh_plot()
    assert not panel._trend_curve_compute_active
    assert panel._trend_cache_sig == sig1
    panel.shutdown_workers()


def test_show_components_toggle_recomputes_trend_overlay(
    panel: FitParametersPanel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Toggling an input (show components) invalidates the cache and recomputes."""
    panel._model_fits = {"A0": _active_linear_fit("A0")}
    monkeypatch.setattr(panel, "_selected_y_parameters", lambda: ["A0"])

    panel._refresh_plot()
    wait_for(lambda: not panel._trend_curve_compute_active, QApplication.instance(), timeout_s=10.0)
    sig1 = panel._trend_cache_sig

    # Toggling show-components fires _refresh_plot; the signature now differs, so
    # a fresh off-thread recompute is dispatched rather than reusing the cache.
    panel._show_components_action.setChecked(True)
    wait_for(lambda: not panel._trend_curve_compute_active, QApplication.instance(), timeout_s=10.0)
    assert panel._trend_cache_sig != sig1
    panel.shutdown_workers()


def test_cache_present_missing_param_does_not_sample_inline(
    panel: FitParametersPanel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cache that omits an active param must not fall back to inline sampling.

    Regression: when the off-thread worker produced no curves for a param (or the
    error path stored an empty cache), _draw_model_overlay_mpl treated the missing
    key as 'no cache' and re-evaluated the (possibly very slow) model on the GUI
    thread — defeating the off-thread design and re-running on every redraw.
    """
    panel._model_fits = {"A0": _active_linear_fit("A0")}
    monkeypatch.setattr(panel, "_selected_y_parameters", lambda: ["A0"])

    sampled_calls: list[int] = []
    orig = panel._sampled_fit_curves
    monkeypatch.setattr(
        panel,
        "_sampled_fit_curves",
        lambda *a, **k: (sampled_calls.append(1), orig(*a, **k))[1],
    )

    # Cache present (non-None) but missing A0 — as after a worker that produced
    # nothing for it, or the error path's empty cache.
    panel._precomputed_trend_curves = {}
    panel._trend_cache_sig = panel._trend_overlay_signature(panel._active_overlay_params())
    panel._draw_plot()

    assert sampled_calls == [], "must not sample the model inline when a cache is present"


def test_load_representation_series_reconstructs_global_params(panel: FitParametersPanel) -> None:
    """A reloaded global/grouped series rebuilds the 'Global fitting parameters' set.

    The shared param's value/error live in every member summary (a global fit shares
    one value); the panel must surface them rather than reporting None.
    """
    row_dicts = [
        {
            "run_number": 1658,
            "run_label": "R1658/G4",
            "field": 7800.0,
            "temperature": 150.0,
            "values": {"f_Aa": 0.0604, "Lambda_1": 0.1418},
            "errors": {"f_Aa": 0.0012, "Lambda_1": 0.111},
        },
        {
            "run_number": 1657,
            "run_label": "R1657/G4",
            "field": 7800.0,
            "temperature": 150.0,
            "values": {"f_Aa": 0.0604, "Lambda_1": 0.4352},
            "errors": {"f_Aa": 0.0012, "Lambda_1": 0.108},
        },
    ]
    panel.load_representation_series(
        [("batch-1", "Series 1", row_dicts)],
        global_params_by_id={"batch-1": {"f_Aa": {"value": 0.0604, "error": 0.0012}}},
        select_id="batch-1",
    )

    assert panel._global_params is not None
    names = {p.name: p.value for p in panel._global_params}
    assert names == pytest.approx({"f_Aa": 0.0604})
    assert panel._global_param_uncertainties.get("f_Aa") == pytest.approx(0.0012)
    # A purely-local param is not promoted to the shared header.
    assert "Lambda_1" not in {p.name for p in panel._global_params}


def test_load_representation_series_without_global_names_reports_none(
    panel: FitParametersPanel,
) -> None:
    """A pure batch fit (no shared params) keeps the header empty."""
    row_dicts = [
        {
            "run_number": 1,
            "run_label": "1",
            "field": 100.0,
            "temperature": 10.0,
            "values": {"Lambda": 0.1},
            "errors": {"Lambda": 0.01},
        }
    ]
    panel.load_representation_series([("batch-2", "Series 2", row_dicts)], select_id="batch-2")
    assert panel._global_params is None


# ── Phase 4: setup entry point, button relabel, group-variable overrides ─────


def _phase4_group(gid: str, name: str, temps: list[float], fields: list[float]) -> _GroupFitData:
    rows = [
        _FitRow(
            run_number=i + 1,
            run_label=str(i + 1),
            field=f,
            temperature=t,
            values={"Lambda": 0.10 + 0.01 * i},
            errors={"Lambda": 0.01},
        )
        for i, (t, f) in enumerate(zip(temps, fields, strict=True))
    ]
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


def test_model_fit_button_relabels_for_two_groups(qapp: QApplication) -> None:
    panel = FitParametersPanel()
    panel._group_fit_results = {
        "g_a": _phase4_group("g_a", "Group A", [10.0, 10.0], [100.0, 200.0]),
        "g_b": _phase4_group("g_b", "Group B", [20.0, 20.0], [100.0, 200.0]),
    }
    panel._active_group_id = "g_a"
    panel._rebuild_group_buttons()

    # One group selected → single-series label.
    panel._set_selected_group_ids(["g_a"], emit=False)
    panel._apply_group_selection_to_view(sync_active=False)
    select_params(panel, ["Lambda"])
    panel._refresh_model_fit_button_labels()
    assert card(panel, "Lambda").fit_button.text() == "Fit…"

    # Two groups selected → cross-group "Global fit ×N…" label.
    panel._set_selected_group_ids(["g_a", "g_b"], emit=False)
    panel._apply_group_selection_to_view(sync_active=False)
    panel._refresh_model_fit_button_labels()
    fit_button = card(panel, "Lambda").fit_button
    assert fit_button.text() == "Global fit ×2…"
    assert "jointly across the selected series" in fit_button.toolTip()


def test_setup_overrides_reach_parameter_group_data(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The setup dialog's per-group group-variable value replaces the inferred
    orthogonal coordinate in the assembled ParameterGroupData."""
    from asymmetry.gui.panels import fit_parameters_panel as panel_mod
    from asymmetry.gui.panels.global_fit_setup_dialog import GlobalFitSetupResult

    panel = FitParametersPanel()
    panel._group_fit_results = {
        "g_a": _phase4_group("g_a", "Group A", [10.0, 10.0], [100.0, 200.0]),
        "g_b": _phase4_group("g_b", "Group B", [20.0, 20.0], [100.0, 200.0]),
    }

    # Stub the setup dialog: accept with explicit overrides that differ from the
    # T-median inference (which would give 10 and 20).
    setup_result = GlobalFitSetupResult(
        parameter_name="Lambda",
        x_key="field",
        x_label="B (G)",
        group_ids=["g_a", "g_b"],
        group_variable_key="temperature",
        group_variable_label="My T (K)",
        group_variable_values={"g_a": 111.0, "g_b": 222.0},
    )

    class _SetupStub:
        def __init__(self, *a, **k):
            pass

        def exec(self):
            return QDialog.DialogCode.Accepted

        def result_data(self):
            return setup_result

    monkeypatch.setattr(panel_mod, "GlobalFitSetupDialog", _SetupStub, raising=False)
    # Also patch the lazily-imported name inside open_global_fit_setup.
    import asymmetry.gui.panels.global_fit_setup_dialog as setup_mod

    monkeypatch.setattr(setup_mod, "GlobalFitSetupDialog", _SetupStub)

    class _RecordingDialogStub:
        captured_groups: list | None = None
        captured_config: dict | None = None

        def __init__(self, *, groups, **_kwargs):
            type(self).captured_groups = groups

        def exec(self):
            return QDialog.DialogCode.Accepted

        def output(self):
            model = panel_mod.ParameterCompositeModel(["Linear"], [])
            result = panel_mod.CrossGroupFitResult(
                success=True, chi_squared=1.0, reduced_chi_squared=1.0
            )
            config: dict = {"model": model.to_dict(), "parameter_rows": []}
            type(self).captured_config = config
            return SimpleNamespace(
                model=model,
                fit_result=result,
                groups=type(self).captured_groups,
                x_key="field",
                fit_x_min=float("nan"),
                fit_x_max=float("nan"),
                config=config,
            )

    monkeypatch.setattr(panel_mod, "CrossGroupFitDialog", _RecordingDialogStub)

    recorded: list = []
    panel.cross_group_fit_completed.connect(
        lambda param, groups, output: recorded.append((param, groups, output))
    )

    payload = panel.open_global_fit_setup(
        preselected_group_ids=["g_a", "g_b"],
        preselected_parameter="Lambda",
        preselected_x_key="field",
    )

    assert payload is not None
    captured = _RecordingDialogStub.captured_groups
    assert captured is not None
    by_id = {g.group_id: g for g in captured}
    # Overrides replaced the inferred temperature medians (10 / 20).
    assert by_id["g_a"].group_variable_value == 111.0
    assert by_id["g_b"].group_variable_value == 222.0

    # The output config carries the group_variable block for the study path.
    assert recorded
    _param, _groups, output = recorded[0]
    gv = output.config["group_variable"]
    assert gv["key"] == "temperature"
    assert gv["label"] == "My T (K)"
    assert gv["values"] == {"g_a": 111.0, "g_b": 222.0}


# ── WP-A: data_revision (staleness-cache invalidation signal) ────────────────


def test_data_revision_bumps_at_trend_input_choke_points(qapp: QApplication) -> None:
    """The revision advances on ingest, clear, and a value-changing relink."""
    panel = FitParametersPanel()
    start = panel.data_revision

    # load_representation_series is the host's pull entry point (series
    # recorded/updated/deleted, representation change, membership toggle).
    panel.load_representation_series(
        [
            (
                "batch-1",
                "S1",
                [
                    {
                        "run_number": 1,
                        "run_label": "1",
                        "field": 100.0,
                        "temperature": 10.0,
                        "values": {"Lambda": 0.1},
                        "errors": {"Lambda": 0.01},
                        "custom_values": {"custom:c1": "5"},
                    }
                ],
            )
        ],
        select_id="batch-1",
    )
    after_load = panel.data_revision
    assert after_load > start

    # A relink that changes a row's custom values bumps; a no-op relink does not.
    panel.relink_custom_values({1: {"custom:c1": "9"}})
    after_relink = panel.data_revision
    assert after_relink > after_load
    panel.relink_custom_values({1: {"custom:c1": "9"}})  # unchanged → no bump
    assert panel.data_revision == after_relink

    # clear() drops all rows — a trend-input change.
    panel.clear()
    assert panel.data_revision > after_relink


def test_focus_leaves_the_persisted_collapse_alone(panel: FitParametersPanel) -> None:
    select_params(panel, ["A0", "Lambda"])

    panel._card_stack.set_focus("A0")
    assert panel._collapsed_params == set()
    assert panel.get_state()["collapsed_params"] == []

    panel._card_stack.set_focus(None)
    assert card(panel, "Lambda").is_expanded()


def test_restoring_an_empty_selection_checks_no_chip(panel: FitParametersPanel) -> None:
    state = panel.get_state()
    state["selected_y_params"] = []

    restored = FitParametersPanel()
    restored.restore_state(state)

    assert not any(chip.isChecked() for chip in restored._y_chips.values())
    assert restored._card_stack.cards() == []


def test_one_card_redraw_keeps_the_other_cards_labels_draggable(
    panel: FitParametersPanel,
) -> None:
    select_params(panel, ["A0", "Lambda"])
    panel._plot_annotations = [
        {"x": 150.0, "y": 0.2, "text": "a", "axis_tag": "A0", "artist": None},
        {"x": 150.0, "y": 0.1, "text": "l", "axis_tag": "Lambda", "artist": None},
    ]
    panel._refresh_plot()
    assert all(ann["artist"] is not None for ann in panel._plot_annotations)

    panel._draw_card("A0")

    assert all(ann["artist"] is not None for ann in panel._plot_annotations)


def test_clear_drops_the_lenses_and_labels(panel: FitParametersPanel) -> None:
    select_params(panel, ["Lambda"])
    panel._set_x_transform(AxisTransform.custom("x**2"))
    panel._set_y_transform("Lambda", AxisTransform.custom("1/x"))
    panel._plot_annotations = [{"x": 1.0, "y": 1.0, "text": "a", "axis_tag": "Lambda"}]

    panel.clear()

    assert panel._x_transform.is_identity
    assert panel._y_transforms == {}
    assert panel._plot_annotations == []
    assert panel._x_transform_button.text() == "ƒ"


def test_table_globals_follow_every_table_refresh(panel: FitParametersPanel) -> None:
    panel._global_params = ParameterSet([Parameter("A_bg", value=3.5)])
    panel._global_param_uncertainties = {"A_bg": 0.25}

    panel._refresh_table()

    assert "A_bg = 3.5 ± 0.25" in panel._table_globals_label.text()


def test_refit_completion_for_a_replaced_fit_set_is_dropped(
    panel: FitParametersPanel, monkeypatch: pytest.MonkeyPatch
) -> None:
    select_params(panel, ["Lambda"])
    panel._model_fits["Lambda"] = _active_linear_fit("Lambda")
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        panel._tasks,
        "start",
        lambda fn, on_finished, on_error: captured.update(fn=fn, done=on_finished),
    )

    panel.refit_active_model_fits()
    panel.clear()
    captured["done"](captured["fn"](None))

    assert panel._model_fits == {}
    assert not panel._refit_in_progress


def test_panel_minimum_width_does_not_follow_parameter_names(panel: FitParametersPanel) -> None:
    panel.show()
    narrow = panel.minimumSizeHint().width()
    for row in panel._rows:
        row.values["sigma_KT_fast_component_rate"] = 0.3
        row.errors["sigma_KT_fast_component_rate"] = 0.01
    panel._varying_params = ["A0", "Lambda", "sigma_KT_fast_component_rate"]

    panel._rebuild_y_controls(preferred_selected=["A0"])

    assert panel.minimumSizeHint().width() == narrow
    chip = panel._y_chips["sigma_KT_fast_component_rate"]
    assert chip.text().endswith("…")
    assert chip.toolTip() == _format_param_label("sigma_KT_fast_component_rate")


def _hover(panel: FitParametersPanel, view_row: int) -> None:
    """Rest the pointer on a table row, then let the 30 ms ring timer fire."""
    panel._table.entered.emit(panel._table.model().index(view_row, 0))
    panel._apply_hover_ring()


def _hover_panel(qapp: QApplication, runs: int) -> FitParametersPanel:
    """A Subplots panel over *runs* runs, both parameters carded and tabled."""
    w = FitParametersPanel()
    w._rows = [
        _FitRow(
            run_number=i + 1,
            run_label=str(i + 1),
            field=100.0 * (i + 1),
            temperature=10.0,
            values={"A0": 0.20 + 0.01 * i, "Lambda": 0.10 + 0.01 * i},
            errors={"A0": 0.01, "Lambda": 0.01},
        )
        for i in range(runs)
    ]
    w._varying_params = ["A0", "Lambda"]
    w._inferred_x_key = "field"
    w._x_combo.setCurrentText("Auto")
    w._rebuild_y_controls()
    w._update_row_dependent_controls()
    select_params(w, ["A0", "Lambda"])
    w._refresh_table()
    return w


class TestTableHoverRing:
    """Hovering a row in the Fitted parameters pop-out rings that run's point.

    The rings are animated artists blitted over each canvas' cached background,
    so every assertion here is about artist state plus the pin that no redraw
    happens. Offscreen ``draw_idle()`` never paints, so a test that needs a
    cached background calls ``canvas.draw()`` itself.
    """

    def test_hover_rings_the_run_on_every_card_through_the_lenses(
        self, panel: FitParametersPanel
    ) -> None:
        select_params(panel, ["A0", "Lambda"])
        # Non-identity lenses on both axes: the ring must sit on the plotted
        # point, not on the raw fitted value.
        panel._set_x_transform(AxisTransform.preset("square"))
        panel._set_y_transform("Lambda", AxisTransform.preset("reciprocal"))
        panel._refresh_table()
        for card_ in panel._card_stack.cards():
            card_.canvas.draw()

        # Rows are x-sorted, so view row 1 is run 2 (B = 200 G).
        panel._table.entered.emit(panel._table.model().index(1, 0))
        assert panel._hover_ring_timer.isActive()
        panel._apply_hover_ring()

        a0_ring = panel._hover_rings[id(axes_for(panel, "A0"))]
        lam_ring = panel._hover_rings[id(axes_for(panel, "Lambda"))]
        assert a0_ring.get_visible() and lam_ring.get_visible()
        np.testing.assert_allclose(a0_ring.get_xydata(), [[200.0**2, 0.22]])
        np.testing.assert_allclose(lam_ring.get_xydata(), [[200.0**2, 1.0 / 0.12]])

    def test_leaving_the_rows_hides_every_ring(self, panel: FitParametersPanel) -> None:
        select_params(panel, ["A0", "Lambda"])
        panel._refresh_table()
        for card_ in panel._card_stack.cards():
            card_.canvas.draw()
        _hover(panel, 0)
        assert all(ring.get_visible() for ring in panel._hover_rings.values())

        QApplication.sendEvent(panel._table.viewport(), QEvent(QEvent.Type.Leave))

        assert not any(ring.get_visible() for ring in panel._hover_rings.values())
        assert panel._hovered_row is None
        assert not panel._hover_ring_timer.isActive()

    def test_hiding_the_pop_out_hides_every_ring(self, panel: FitParametersPanel) -> None:
        select_params(panel, ["A0", "Lambda"])
        panel._refresh_table()
        _hover(panel, 0)
        assert all(ring.get_visible() for ring in panel._hover_rings.values())

        QApplication.sendEvent(panel._table_dialog, QEvent(QEvent.Type.Hide))

        assert not any(ring.get_visible() for ring in panel._hover_rings.values())

    def test_a_collapsed_card_carries_no_ring(self, panel: FitParametersPanel) -> None:
        select_params(panel, ["A0", "Lambda"])
        card(panel, "A0").set_expanded(False)
        panel._refresh_plot()
        panel._refresh_table()

        _hover(panel, 0)

        assert card(panel, "A0").figure.axes == []
        assert list(panel._hover_rings) == [id(axes_for(panel, "Lambda"))]

    def test_a_parameter_without_a_value_for_that_run_stays_hidden(
        self, panel: FitParametersPanel
    ) -> None:
        run_one = next(row for row in panel._rows if row.run_number == 1)
        run_one.values["Lambda"] = float("nan")
        select_params(panel, ["A0", "Lambda"])
        panel._refresh_table()

        _hover(panel, 0)

        assert panel._hover_rings[id(axes_for(panel, "A0"))].get_visible()
        assert not panel._hover_rings[id(axes_for(panel, "Lambda"))].get_visible()

    def test_overlay_rings_the_shared_axis_and_its_twin(self, panel: FitParametersPanel) -> None:
        panel._overlay_button.setChecked(True)
        select_params(panel, ["A0", "Lambda"])
        panel._refresh_table()
        panel._canvas.draw()

        _hover(panel, 1)

        rings = [panel._hover_rings[id(ax)] for ax in panel._figure.axes]
        assert len(rings) == 2
        np.testing.assert_allclose(
            panel._hover_rings[id(axes_for(panel, "A0"))].get_xydata(), [[200.0, 0.22]]
        )
        np.testing.assert_allclose(
            panel._hover_rings[id(axes_for(panel, "Lambda"))].get_xydata(), [[200.0, 0.12]]
        )

    def test_a_draw_that_lands_after_a_hover_defers_the_ring_to_the_timer(
        self, panel: FitParametersPanel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A hover before the first paint has no background to blit over; the
        draw that follows caches one and arms the timer, which puts the ring
        back. The draw itself must never blit: the Qt canvas runs a pending draw
        from inside its paint event, and a blit is a synchronous repaint —
        from there it recursed into the paint ("Recursive repaint detected")."""
        select_params(panel, ["A0", "Lambda"])
        panel._refresh_table()
        _hover(panel, 0)
        panel._hover_ring_timer.stop()
        assert panel._hover_backgrounds == {}

        canvas = card(panel, "A0").canvas
        blits: list[int] = []
        monkeypatch.setattr(canvas, "blit", lambda bbox=None: blits.append(1))
        canvas.draw()

        assert canvas in panel._hover_backgrounds
        assert blits == []
        assert panel._hover_ring_timer.isActive()
        panel._apply_hover_ring()
        assert blits == [1]
        assert panel._hover_rings[id(axes_for(panel, "A0"))].get_visible()

    def test_a_draw_under_no_hover_leaves_the_timer_alone(
        self, panel: FitParametersPanel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        select_params(panel, ["A0"])
        panel._refresh_table()
        canvas = card(panel, "A0").canvas
        blits: list[int] = []
        monkeypatch.setattr(canvas, "blit", lambda bbox=None: blits.append(1))

        canvas.draw()

        assert canvas in panel._hover_backgrounds
        assert blits == []
        assert not panel._hover_ring_timer.isActive()

    def test_hovering_never_redraws(
        self, qapp: QApplication, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The responsiveness pin: a hover is a blit, never a 200-500 ms redraw."""
        panel = _hover_panel(qapp, runs=5)
        canvases = [card_.canvas for card_ in panel._card_stack.cards()]
        for canvas in canvases:
            canvas.draw()

        counts = {"draw_plot": 0, "refresh_plot": 0, "draw": 0, "draw_idle": 0}

        def _count(key: str):
            def _bump(*_args, **_kwargs) -> None:
                counts[key] += 1

            return _bump

        monkeypatch.setattr(FitParametersPanel, "_draw_plot", _count("draw_plot"))
        monkeypatch.setattr(FitParametersPanel, "_refresh_plot", _count("refresh_plot"))
        for canvas in canvases:
            monkeypatch.setattr(canvas, "draw", _count("draw"))
            monkeypatch.setattr(canvas, "draw_idle", _count("draw_idle"))

        ring = panel._hover_rings[id(axes_for(panel, "Lambda"))]
        seen = []
        for view_row in range(5):
            _hover(panel, view_row)
            seen.append(ring.get_xydata().tolist())

        assert counts == {"draw_plot": 0, "refresh_plot": 0, "draw": 0, "draw_idle": 0}
        # The pin would pass vacuously if the ring never moved.
        assert seen[0] != seen[-1]
        assert len({tuple(point[0]) for point in seen}) == 5

    def test_moving_onto_blank_viewport_hides_every_ring(self, panel: FitParametersPanel) -> None:
        """``entered`` fires only over cells; the blank space below the last row
        reports through ``viewportEntered`` and is not a run either."""
        select_params(panel, ["A0", "Lambda"])
        panel._refresh_table()
        _hover(panel, 0)
        assert all(ring.get_visible() for ring in panel._hover_rings.values())

        panel._table.viewportEntered.emit()

        assert not any(ring.get_visible() for ring in panel._hover_rings.values())
        assert panel._hovered_row is None

    def test_a_full_redraw_under_a_resting_pointer_keeps_the_run_ringed(
        self, panel: FitParametersPanel
    ) -> None:
        """A redraw replaces every ring with a hidden one; the hovered run must
        come back on the next tick without the pointer moving."""
        select_params(panel, ["A0", "Lambda"])
        panel._refresh_table()
        _hover(panel, 1)
        old_ring = panel._hover_rings[id(axes_for(panel, "A0"))]

        panel._set_x_transform(AxisTransform.preset("square"))
        panel._refresh_plot()

        assert panel._hover_ring_timer.isActive()
        panel._apply_hover_ring()
        new_ring = panel._hover_rings[id(axes_for(panel, "A0"))]
        assert new_ring is not old_ring
        assert new_ring.get_visible()
        np.testing.assert_allclose(new_ring.get_xydata(), [[200.0**2, 0.22]])

    def test_a_single_card_redraw_under_a_resting_pointer_keeps_the_run_ringed(
        self, panel: FitParametersPanel
    ) -> None:
        select_params(panel, ["A0", "Lambda"])
        panel._refresh_table()
        _hover(panel, 1)
        untouched = panel._hover_rings[id(axes_for(panel, "Lambda"))]

        panel._draw_card("A0")

        assert panel._hover_ring_timer.isActive()
        panel._apply_hover_ring()
        assert panel._hover_rings[id(axes_for(panel, "Lambda"))] is untouched
        ring = panel._hover_rings[id(axes_for(panel, "A0"))]
        assert ring.get_visible()
        np.testing.assert_allclose(ring.get_xydata(), [[200.0, 0.22]])

    def test_a_redraw_with_no_pointer_on_the_table_arms_nothing(
        self, panel: FitParametersPanel
    ) -> None:
        select_params(panel, ["A0", "Lambda"])
        panel._refresh_table()
        _hover(panel, 0)
        QApplication.sendEvent(panel._table.viewport(), QEvent(QEvent.Type.Leave))

        panel._refresh_plot()

        assert not panel._hover_ring_timer.isActive()
        assert not any(ring.get_visible() for ring in panel._hover_rings.values())
