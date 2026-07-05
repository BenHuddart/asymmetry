"""Tests for cross-group fit dialog UI parity with model-fit labels."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.gui, pytest.mark.slow, pytest.mark.integration]

import threading
import time

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDialog

from asymmetry.core.fitting.parameter_models import (
    CrossGroupFitResult,
    ParameterCompositeModel,
    ParameterGroupData,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.gui.panels.cross_group_fit_dialog import CrossGroupFitDialog
from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel, _FitRow, _GroupFitData


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


def test_single_card_no_add_no_remove() -> None:
    app = QApplication.instance() or QApplication([])
    dlg = CrossGroupFitDialog(
        parameter_name="Lambda",
        x_key="field",
        groups=_groups(),
        parent=None,
    )

    # Cross-group mode pins exactly one shared range: "Add Range" is hidden
    # (adding a second range is meaningless here) and the single card's view
    # reports can_remove=False (no Remove action available).
    assert dlg._add_range_btn.isVisible() is False
    assert len(dlg._range_cards) == 1
    assert dlg._range_cards[0]._view.can_remove is False
    assert app is not None


def test_cross_group_card_always_active() -> None:
    app = QApplication.instance() or QApplication([])
    dlg = CrossGroupFitDialog(
        parameter_name="Lambda",
        x_key="field",
        groups=_groups(),
        parent=None,
    )

    assert len(dlg._range_cards) == 1
    assert dlg.active_range_index() == 0
    assert dlg._range_cards[0]._view.show_run is True
    assert app is not None


def test_cross_group_card_title() -> None:
    app = QApplication.instance() or QApplication([])
    dlg = CrossGroupFitDialog(
        parameter_name="Lambda",
        x_key="field",
        groups=_groups(),
        parent=None,
    )

    # Formula-only degradation: the title is not the base "Range 1" label.
    # This dialog's chosen alternative is the trended parameter name.
    title = dlg._range_cards[0]._view.title
    assert title != "Range 1"
    assert title == "Lambda"
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

    # The banner lives in the base dialog's named header slot (contract C6),
    # not at a hardcoded top-level layout index.
    assert dlg._header_slot.count() >= 1
    banner = dlg._header_slot.itemAt(0).widget()
    assert banner is not None
    text = banner.text()
    assert "Inherited from" in text
    assert "G0" in text
    assert "chi2_r=" in text
    assert app is not None


def test_cross_group_dialog_footer_slot_holds_suggest_roles_above_buttons() -> None:
    """The suggest-roles row + rationale panel sit in the footer slot, which the
    base dialog places directly above the OK/Cancel button box (contract C6)."""
    app = QApplication.instance() or QApplication([])
    dlg = CrossGroupFitDialog(
        parameter_name="Lambda",
        x_key="field",
        groups=_groups(),
        parent=None,
    )

    footer_widgets = [dlg._footer_slot.itemAt(i).widget() for i in range(dlg._footer_slot.count())]
    assert dlg._suggest_btn in footer_widgets or any(
        widget is not None and dlg._suggest_btn in widget.findChildren(type(dlg._suggest_btn))
        for widget in footer_widgets
    )
    assert dlg._rationale_panel in footer_widgets

    main_layout = dlg.layout()
    footer_index = None
    buttons_index = None
    for i in range(main_layout.count()):
        item = main_layout.itemAt(i)
        if item is None:
            continue
        if item.layout() is dlg._footer_slot:
            footer_index = i
        if item.widget() is dlg._buttons:
            buttons_index = i
    assert footer_index is not None
    assert buttons_index is not None
    assert footer_index < buttons_index
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


def _param_groups_with_xerr() -> list[ParameterGroupData]:
    """Groups whose abscissa is a fitted parameter (param:nu) carrying σ_x."""
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    xe = np.array([0.1, 0.1, 0.1, 0.1], dtype=float)
    g0 = ParameterGroupData(
        group_id="g0",
        group_name="G0",
        x=x,
        y=np.array([0.1, 0.2, 0.3, 0.4], dtype=float),
        yerr=np.full_like(x, 0.01),
        group_variable_value=0.0,
        xerr=xe,
    )
    g1 = ParameterGroupData(
        group_id="g1",
        group_name="G1",
        x=x,
        y=np.array([0.12, 0.22, 0.32, 0.42], dtype=float),
        yerr=np.full_like(x, 0.01),
        group_variable_value=1.0,
        xerr=xe,
    )
    return [g0, g1]


def test_cross_group_dialog_exposes_error_modes_and_windows() -> None:
    """global_fit_parameter_model now honours error modes and windows, so the
    inherited selector and 'Exclude region…' button are shown and wired. The
    effective-variance toggle is hidden for a run-level axis (field has no
    x-uncertainty), not because cross-group lacks backend support."""
    QApplication.instance() or QApplication([])
    dlg = CrossGroupFitDialog(
        parameter_name="Lambda",
        x_key="field",
        groups=_groups(),
        parent=None,
    )

    assert dlg._error_mode_combo is not None
    assert dlg._error_value_spin is not None
    from asymmetry.core.fitting.parameter_models import ErrorMode

    assert dlg._error_mode() is ErrorMode.COLUMN  # default

    # The "Exclude region…" path is a dedicated button on the range card (the
    # overflow dropdown was replaced by visible action buttons).
    card = dlg._range_cards[0]
    assert card._exclude_button.text() == "Exclude region…"

    # field is exact (no σ_x), so the effective-variance toggle is not offered.
    assert dlg._x_error_check is None

    # Cross-group's single pinned range renders as exactly one RangeCard; a
    # rebuild keeps that invariant (Step 1 of the range-cards redesign — the
    # old per-range 'active' checkbox no longer exists).
    dlg._rebuild_ranges_ui()
    assert len(dlg._range_cards) == 1


def test_cross_group_card_no_status_chip() -> None:
    """A cross-group card shows no single-fit verdict chip: CrossGroupFitResult
    has no per-range χ²ᵣ/dof, so a "good/poor" chip would mislead (same reason
    the quality line is suppressed)."""
    dlg = CrossGroupFitDialog(
        parameter_name="Lambda",
        x_key="field",
        groups=_groups(),
        parent=None,
    )
    fit_range = dlg._fit.ranges[0]
    # The chip helper is suppressed regardless of any result object.
    assert dlg._range_status_chip_html(fit_range, object()) == ""
    # And the assembled card view carries no chip html.
    view = dlg._range_card_view(0, show_run=True)
    assert view.status_chip_html == ""


def test_cross_group_dialog_offers_x_uncertainty_for_param_axis() -> None:
    """When the abscissa is a fitted parameter carrying σ_x, the cross-group
    dialog now exposes the effective-variance toggle (decision E retired); it is
    disabled under None/Scatter, whose unit y-weights have no scale for σ_x."""
    QApplication.instance() or QApplication([])
    from asymmetry.core.fitting.parameter_models import ErrorMode

    dlg = CrossGroupFitDialog(
        parameter_name="Lambda",
        x_key="param:nu",
        groups=_param_groups_with_xerr(),
        parent=None,
    )

    assert dlg._x_error_check is not None
    # Default Column mode → live and usable.
    assert dlg._x_error_check.isEnabled()
    dlg._x_error_check.setChecked(True)
    assert dlg._use_x_errors() is True

    # None/Scatter disable the toggle, so a box left checked stays inert.
    none_idx = dlg._error_mode_combo.findData(ErrorMode.NONE.value)
    dlg._error_mode_combo.setCurrentIndex(none_idx)
    assert not dlg._x_error_check.isEnabled()
    assert dlg._use_x_errors() is False


def test_cross_group_dialog_x_uncertainty_config_roundtrips() -> None:
    """use_x_errors survives _collect_config → _apply_existing_config; legacy
    config (no key) loads as off."""
    QApplication.instance() or QApplication([])
    dlg = CrossGroupFitDialog(
        parameter_name="Lambda",
        x_key="param:nu",
        groups=_param_groups_with_xerr(),
        parent=None,
    )
    assert dlg._x_error_check is not None
    dlg._x_error_check.setChecked(True)
    config = dlg._collect_config()
    assert config["use_x_errors"] is True

    reloaded = CrossGroupFitDialog(
        parameter_name="Lambda",
        x_key="param:nu",
        groups=_param_groups_with_xerr(),
        existing_config=config,
        parent=None,
    )
    assert reloaded._use_x_errors() is True

    # Legacy config without the key → off.
    legacy = dict(config)
    legacy.pop("use_x_errors")
    legacy_dlg = CrossGroupFitDialog(
        parameter_name="Lambda",
        x_key="param:nu",
        groups=_param_groups_with_xerr(),
        existing_config=legacy,
        parent=None,
    )
    assert legacy_dlg._use_x_errors() is False


def test_local_param_error_cell_shows_per_group_values_not_just_group0() -> None:
    """Regression: a Local parameter has a distinct value/error per group, so
    the Error column must not silently report only the first group's number.
    It shows 'varies' with every group's value ± error in the tooltip."""
    QApplication.instance() or QApplication([])
    dlg = CrossGroupFitDialog(
        parameter_name="Lambda",
        x_key="field",
        groups=_groups(),
        parent=None,
    )

    result = CrossGroupFitResult(
        success=True,
        chi_squared=2.0,
        reduced_chi_squared=1.0,
        global_parameters=ParameterSet([Parameter(name="m", value=0.5)]),
        local_parameters={
            "g0": ParameterSet([Parameter(name="b", value=1.0)]),
            "g1": ParameterSet([Parameter(name="b", value=3.0)]),
        },
        global_uncertainties={"m": 0.05},
        local_uncertainties={"g0": {"b": 0.1}, "g1": {"b": 0.2}},
    )

    # Global parameter: single shared uncertainty.
    global_cell = dlg._build_error_cell("m", "Global", result)
    assert global_cell.text() == f"{0.05:.4g}"

    # Local parameter: 'varies' + per-group breakdown covering BOTH groups,
    # not just g0 (the bug showed only group 0's 0.1).
    local_cell = dlg._build_error_cell("b", "Local", result)
    assert local_cell.text() == "varies"
    tip = local_cell.toolTip()
    assert "G0" in tip and "G1" in tip
    assert "1" in tip and "3" in tip  # both fitted values present
    assert "0.1" in tip and "0.2" in tip  # both per-group errors present

    # Fixed parameter: no uncertainty shown.
    assert dlg._build_error_cell("b", "Fixed", result).text() == ""
    # No result yet: blank.
    assert dlg._build_error_cell("b", "Local", None).text() == ""


# --- Phase 0: _run_cross_group_model_fit must honour include_in_trend --------


def _fit_row(
    run_number: int, field: float, value: float, *, include_in_trend: bool = True
) -> _FitRow:
    return _FitRow(
        run_number=run_number,
        run_label=str(run_number),
        field=field,
        temperature=5.0,
        values={"Lambda": value},
        errors={"Lambda": 0.01},
        include_in_trend=include_in_trend,
    )


class _RecordingDialogStub:
    """Stand-in for CrossGroupFitDialog: records the assembled groups and
    reports the dialog as cancelled, so ``_run_cross_group_model_fit`` returns
    ``None`` right after assembly without needing a full dialog/fit round trip."""

    captured_groups: list[ParameterGroupData] | None = None

    def __init__(self, *, groups: list[ParameterGroupData], **_kwargs) -> None:
        type(self).captured_groups = groups

    def exec(self) -> int:
        return QDialog.DialogCode.Rejected


def test_cross_group_fit_excludes_trend_excluded_member(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A member with include_in_trend=False must be absent from the assembled
    ParameterGroupData x/y arrays fed to the cross-group dialog/fit — mirroring
    the single-group path's ``_included_trend_rows`` gate."""
    from asymmetry.gui.panels import fit_parameters_panel as panel_mod

    monkeypatch.setattr(panel_mod, "CrossGroupFitDialog", _RecordingDialogStub)
    _RecordingDialogStub.captured_groups = None

    panel = FitParametersPanel()
    rows_a = [
        _fit_row(1, 100.0, 0.10),
        _fit_row(2, 200.0, 0.20),
        _fit_row(3, 300.0, 999.0, include_in_trend=False),  # excluded outlier
    ]
    rows_b = [
        _fit_row(4, 100.0, 0.12),
        _fit_row(5, 200.0, 0.22),
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

    payload = panel._run_cross_group_model_fit("Lambda", [group_a, group_b])

    assert payload is None  # dialog was "cancelled" by the stub
    captured = _RecordingDialogStub.captured_groups
    assert captured is not None
    assert len(captured) == 2

    by_id = {g.group_id: g for g in captured}
    # The excluded run-3 point (x=300, y=999) must not appear.
    assert list(by_id["g_a"].x) == [100.0, 200.0]
    assert list(by_id["g_a"].y) == [0.10, 0.20]
    assert list(by_id["g_b"].x) == [100.0, 200.0]
    assert list(by_id["g_b"].y) == [0.12, 0.22]


def test_cross_group_fit_drops_fully_excluded_group_and_guards_below_two(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A group whose members are all trend-excluded (or left with <2 included
    points) is dropped entirely; if that leaves fewer than two groups, the
    existing "need two groups" guard fires and the dialog is never opened."""
    from asymmetry.gui.panels import fit_parameters_panel as panel_mod

    monkeypatch.setattr(panel_mod, "CrossGroupFitDialog", _RecordingDialogStub)
    _RecordingDialogStub.captured_groups = None

    info_calls: list[tuple] = []
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda *args, **kwargs: info_calls.append(args) or None,
    )

    panel = FitParametersPanel()
    rows_a = [
        _fit_row(1, 100.0, 0.10),
        _fit_row(2, 200.0, 0.20),
    ]
    # Every member of group B is excluded from the trend.
    rows_b = [
        _fit_row(4, 100.0, 0.12, include_in_trend=False),
        _fit_row(5, 200.0, 0.22, include_in_trend=False),
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

    payload = panel._run_cross_group_model_fit("Lambda", [group_a, group_b])

    assert payload is None
    # The dialog was never constructed — the guard fired first.
    assert _RecordingDialogStub.captured_groups is None
    assert len(info_calls) == 1
    guard_text = str(info_calls[0])
    assert "trend" in guard_text.lower()


def test_cross_group_fit_all_included_matches_prior_unfiltered_behaviour(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When every member is included (the common case), the assembled arrays
    are unchanged from a plain group.rows pass-through."""
    from asymmetry.gui.panels import fit_parameters_panel as panel_mod

    monkeypatch.setattr(panel_mod, "CrossGroupFitDialog", _RecordingDialogStub)
    _RecordingDialogStub.captured_groups = None

    panel = FitParametersPanel()
    rows_a = [
        _fit_row(1, 300.0, 0.30),
        _fit_row(2, 100.0, 0.10),
        _fit_row(3, 200.0, 0.20),
    ]
    rows_b = [
        _fit_row(4, 100.0, 0.12),
        _fit_row(5, 200.0, 0.22),
        _fit_row(6, 300.0, 0.32),
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

    payload = panel._run_cross_group_model_fit("Lambda", [group_a, group_b])

    assert payload is None
    captured = _RecordingDialogStub.captured_groups
    assert captured is not None
    by_id = {g.group_id: g for g in captured}
    # Sorted ascending by x (field), all three points present per group.
    assert list(by_id["g_a"].x) == [100.0, 200.0, 300.0]
    assert list(by_id["g_a"].y) == [0.10, 0.20, 0.30]
    assert list(by_id["g_b"].x) == [100.0, 200.0, 300.0]
    assert list(by_id["g_b"].y) == [0.12, 0.22, 0.32]


# --- Phase 4: Suggest roles button --------------------------------------------


def _suggest_wait(dlg, timeout_s: float = 20.0) -> None:
    """Wait for a Suggest-roles task to leave the busy state."""
    app = QApplication.instance()
    deadline = time.time() + timeout_s
    while dlg._suggest_in_progress and time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)


def _linear_groups() -> list[ParameterGroupData]:
    """Two groups following y = m·x + b with a shared slope, distinct offset."""
    x = np.linspace(0.0, 10.0, 8)
    g0 = ParameterGroupData(
        group_id="g0",
        group_name="G0",
        x=x.copy(),
        y=0.02 * x + 0.10,
        yerr=np.full_like(x, 0.005),
        group_variable_value=10.0,
    )
    g1 = ParameterGroupData(
        group_id="g1",
        group_name="G1",
        x=x.copy(),
        y=0.02 * x + 0.40,
        yerr=np.full_like(x, 0.005),
        group_variable_value=20.0,
    )
    return [g0, g1]


def test_suggest_roles_applies_recommendation_and_shows_rationale() -> None:
    QApplication.instance() or QApplication([])
    dlg = CrossGroupFitDialog(
        parameter_name="Lambda",
        x_key="field",
        groups=_linear_groups(),
        parent=None,
    )
    # Default model is Linear (m, b). Run the real suggestion off-thread.
    dlg._on_suggest_roles_clicked()
    _suggest_wait(dlg)

    assert dlg._suggest_recommendation is not None
    # A recommendation was applied to the role combos and the rationale is shown.
    assert dlg._rationale_panel.isVisibleTo(dlg)
    text = dlg._rationale_panel.toPlainText()
    assert "Per-parameter recommendation" in text
    assert "candidates" in text.lower()

    # Every role combo now reads a concrete Global/Local (not blank).
    for row in range(dlg._param_table.rowCount()):
        combo = dlg._param_table.cellWidget(row, 4)
        assert combo is not None
        assert combo.currentText() in {"Global", "Local", "Fixed"}


def test_suggest_roles_renders_failed_candidate(monkeypatch) -> None:
    QApplication.instance() or QApplication([])
    from asymmetry.core.fitting.cross_group_roles import (
        CrossGroupCandidate,
        CrossGroupParameterRecommendation,
        CrossGroupRoleRecommendation,
    )

    good = CrossGroupCandidate(
        global_params=("m", "b"),
        local_params=(),
        fixed_params=(),
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        n_free=2,
        n_points=16,
        aic=5.0,
        aicc=6.0,
        bic=7.0,
        result=CrossGroupFitResult(success=True, chi_squared=1.0, reduced_chi_squared=1.0),
    )
    bad = CrossGroupCandidate(
        global_params=("m",),
        local_params=("b",),
        fixed_params=(),
        success=False,
        chi_squared=float("inf"),
        reduced_chi_squared=float("inf"),
        n_free=3,
        n_points=16,
        aic=float("inf"),
        aicc=float("inf"),
        bic=float("inf"),
        result=None,
    )
    rec = CrossGroupRoleRecommendation(
        candidates=[good, bad],
        recommended=good,
        parameters=[
            CrossGroupParameterRecommendation(
                name="m",
                recommended_role="global",
                score_delta=-1.0,
                total_variation=0.0,
                roughness=0.0,
                rationale="Sharing m is favoured.",
            ),
        ],
        criterion="aicc",
        message="Recommended partition: local = [] (AICc = 6.00).",
    )

    monkeypatch.setattr(
        "asymmetry.gui.panels.cross_group_fit_dialog.suggest_cross_group_roles",
        lambda *a, **k: rec,
    )

    dlg = CrossGroupFitDialog(
        parameter_name="Lambda",
        x_key="field",
        groups=_linear_groups(),
        parent=None,
    )
    dlg._on_suggest_roles_clicked()
    _suggest_wait(dlg)

    text = dlg._rationale_panel.toPlainText()
    assert "did not converge" in text
    assert "Sharing m is favoured" in text


def test_suggest_roles_cancel_restores_button(monkeypatch) -> None:
    QApplication.instance() or QApplication([])
    from asymmetry.core.fitting.cross_group_roles import CrossGroupRoleRecommendation

    gate = threading.Event()

    def _slow_suggest(*_a, cancel_callback=None, **_k):
        # Wait until the test flips cancel, then honour it like the real engine.
        gate.wait(timeout=2.0)
        return CrossGroupRoleRecommendation(criterion="aicc", message="cancelled")

    monkeypatch.setattr(
        "asymmetry.gui.panels.cross_group_fit_dialog.suggest_cross_group_roles",
        _slow_suggest,
    )

    dlg = CrossGroupFitDialog(
        parameter_name="Lambda",
        x_key="field",
        groups=_linear_groups(),
        parent=None,
    )
    dlg._on_suggest_roles_clicked()
    assert dlg._suggest_in_progress is True
    assert dlg._suggest_btn.isEnabled() is False
    assert dlg._suggest_cancel_btn.isVisibleTo(dlg) is True

    # User cancels; release the worker.
    dlg._on_suggest_cancel_clicked()
    gate.set()
    _suggest_wait(dlg)

    # Button state restored, cancellation noted, table unchanged.
    assert dlg._suggest_in_progress is False
    assert dlg._suggest_btn.isEnabled() is True
    assert dlg._suggest_cancel_btn.isVisibleTo(dlg) is False
    assert "cancel" in dlg._rationale_panel.toPlainText().lower()


# --- WP-C: model-edit invalidates the stale cross-group result ---------------


def _wait_fit_done(dlg: CrossGroupFitDialog, timeout_s: float = 5.0) -> None:
    app = QApplication.instance()
    deadline = time.time() + timeout_s
    while dlg._fit_in_progress and time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)


class _FakeModelBuilder:
    """Stand-in for ParameterModelBuilderDialog: accepts immediately with a
    fixed replacement model, mirroring the pattern used for the base
    ModelFitDialog's own _edit_model tests."""

    _next_model: ParameterCompositeModel | None = None

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def exec(self):
        return QDialog.DialogCode.Accepted

    def get_model(self):
        return type(self)._next_model


def test_edit_model_clears_stale_result_and_blocks_use_fit(monkeypatch) -> None:
    """Editing the range model after a successful run must drop the cached
    result/config so _on_use_fit refuses until the fit is re-run — otherwise a
    study could be saved with the NEW model but the OLD (now-mismatched)
    parameter_rows, which fails on the next refit."""
    app = QApplication.instance() or QApplication([])
    dlg = CrossGroupFitDialog(
        parameter_name="Lambda",
        x_key="field",
        groups=_groups(),
        parent=None,
    )

    # Default model is Linear (m, b); run it for real so _result/_last_config
    # are populated exactly as the live flow would leave them.
    dlg._run_fit(0)
    _wait_fit_done(dlg)
    assert dlg._result is not None
    assert dlg._result.success is True
    assert dlg._last_config is not None

    # Edit the model to something with a disjoint parameter set (Arrhenius:
    # a, Ea) via the base dialog's _edit_model, stubbing the builder dialog.
    _FakeModelBuilder._next_model = ParameterCompositeModel(["Arrhenius"], [])
    monkeypatch.setattr(
        "asymmetry.gui.panels.model_fit_dialog.ParameterModelBuilderDialog",
        _FakeModelBuilder,
    )
    dlg._edit_model(0)

    assert dlg._result is None
    assert dlg._last_config is None

    info_calls: list[tuple] = []
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda *args, **kwargs: info_calls.append(args) or None,
    )
    dlg._on_use_fit()

    assert dlg.result() != dlg.DialogCode.Accepted
    assert len(info_calls) == 1
    assert "Run Cross-Group Fit before using the fit" in str(info_calls[0])
    assert app is not None


def test_edit_model_then_rerun_accepts_with_new_model_config(monkeypatch) -> None:
    """After the invalidation above, re-running the (now edited) fit clears
    the block and output().config reflects the NEW model, not the stale one."""
    app = QApplication.instance() or QApplication([])
    dlg = CrossGroupFitDialog(
        parameter_name="Lambda",
        x_key="field",
        groups=_groups(),
        parent=None,
    )

    dlg._run_fit(0)
    _wait_fit_done(dlg)

    _FakeModelBuilder._next_model = ParameterCompositeModel(["Arrhenius"], [])
    monkeypatch.setattr(
        "asymmetry.gui.panels.model_fit_dialog.ParameterModelBuilderDialog",
        _FakeModelBuilder,
    )
    dlg._edit_model(0)
    assert dlg._result is None

    # Re-run against the new model.
    dlg._run_fit(0)
    _wait_fit_done(dlg)
    assert dlg._result is not None

    dlg._on_use_fit()
    assert dlg.result() == dlg.DialogCode.Accepted

    output = dlg.output()
    assert output is not None
    assert output.model.component_names == ["Arrhenius"]
    assert set(output.config["model"]["component_names"]) == {"Arrhenius"}
    row_names = {row["name"] for row in output.config["parameter_rows"]}
    assert row_names == {"a", "Ea"}
    assert app is not None


def test_preview_series_per_group() -> None:
    """The cross-group override returns one preview series per group."""
    QApplication.instance() or QApplication([])
    groups = _groups()
    dlg = CrossGroupFitDialog(
        parameter_name="Lambda",
        x_key="field",
        groups=groups,
        parent=None,
    )

    series = dlg._preview_series()
    assert len(series) == len(groups)
    assert [s.label for s in series] == [g.group_name for g in groups]
    # The primary series (index 0) mirrors the first group's data.
    np.testing.assert_allclose(series[0].x, groups[0].x)
    np.testing.assert_allclose(series[0].y, groups[0].y)


def test_exclude_region_single_range_carves() -> None:
    """A carve on the cross-group single pinned range yields a two-window union."""
    QApplication.instance() or QApplication([])
    dlg = CrossGroupFitDialog(
        parameter_name="Lambda",
        x_key="field",
        groups=_groups(),
        parent=None,
    )

    # Data x spans 100..300; carve out the interior [180, 220].
    fit_range = dlg._fit.ranges[0]
    dlg._on_preview_exclude_region(0, 180.0, 220.0)

    assert fit_range.windows is not None
    assert len(fit_range.windows) == 2
    his = sorted(hi for _lo, hi in fit_range.windows)
    los = sorted(lo for lo, _hi in fit_range.windows)
    assert 180.0 in his
    assert 220.0 in los
    # The details-pane bounds pair disables once the range carries windows.
    dlg._select_range(0)
    assert dlg._bounds_min_spin.isEnabled() is False


def test_cross_group_success_no_modal(monkeypatch) -> None:
    """A successful cross-group fit tints the result box green with NO success modal."""
    app = QApplication.instance() or QApplication([])
    from asymmetry.gui.styles.widgets import RESULT_BOX_SUCCESS_STYLE

    dlg = CrossGroupFitDialog(
        parameter_name="Lambda",
        x_key="field",
        groups=_groups(),
        parent=None,
    )

    info_calls: list[tuple] = []
    monkeypatch.setattr(
        "asymmetry.gui.panels.cross_group_fit_dialog._show_info",
        lambda *a, **k: info_calls.append(a),
    )

    def _fake_global_fit(**_kwargs):
        return CrossGroupFitResult(success=True, chi_squared=1.0, reduced_chi_squared=1.0)

    monkeypatch.setattr(
        "asymmetry.gui.panels.cross_group_fit_dialog.global_fit_parameter_model",
        _fake_global_fit,
    )

    dlg._run_fit(0)

    deadline = time.time() + 5.0
    while dlg._fit_in_progress and time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)

    assert dlg._fit_in_progress is False
    # No success modal fired.
    assert info_calls == []
    # Inline success text + green tint.
    assert "successful" in dlg._chi2_label.text().lower()
    assert dlg._result_box.styleSheet() == RESULT_BOX_SUCCESS_STYLE
    assert app is not None


def test_formula_uses_break_points() -> None:
    """The cross-group formula renders through the shared pan/zoom FormulaBox
    (set_formula -> insert_formula_break_points + re-measure), NOT the bare
    label. Deleting the subclass _set_formula_display override (Phase 5.c) made
    it inherit the base's box path."""
    from asymmetry.gui.panels.model_fit_dialog import ModelFitDialog
    from asymmetry.gui.styles.widgets import FormulaBox

    app = QApplication.instance() or QApplication([])
    dlg = CrossGroupFitDialog(
        parameter_name="Lambda",
        x_key="field",
        groups=_groups(),
        parent=None,
    )

    # The subclass override is gone -- it now inherits the base's box path.
    assert CrossGroupFitDialog._set_formula_display is ModelFitDialog._set_formula_display

    # A multi-term model whose formula carries top-level operators (m*x + b + c),
    # i.e. exactly where insert_formula_break_points inserts break opportunities.
    dlg._fit.ranges[0].model = ParameterCompositeModel(["Linear", "Constant"], ["+"])
    dlg._rebuild_param_table()  # rebuilds params + re-selects range -> _set_formula_display
    dlg._select_range(0)

    # The formula box is the shared FormulaBox.
    assert isinstance(dlg._formula_box, FormulaBox)

    # The label carries the break-marked text (a zero-width-space break point was
    # inserted at the top-level operator) and the raw formula lives in the
    # tooltip -- the set_formula path, not a bare setText of the plain string.
    label_text = dlg._formula_box.label.text()
    assert "​" in label_text  # a top-level break opportunity was inserted
    assert "m*x + b + c" in dlg._formula_box.label.toolTip()

    assert app is not None


def test_plot_add_select_inert_single_range() -> None:
    """Cross-group pins one range: the plot's add/select gestures are left
    unconnected (no phantom range, no selection change), but right-drag exclude
    on the single range still carves a window (that signal stays wired)."""
    app = QApplication.instance() or QApplication([])
    dlg = CrossGroupFitDialog(
        parameter_name="Lambda",
        x_key="field",
        groups=_groups(),
        parent=None,
    )

    assert len(dlg._fit.ranges) == 1
    assert dlg.active_range_index() == 0

    # Drag-out a new span on empty canvas: no consumer -> no new range.
    dlg._preview.range_add_requested.emit(120.0, 280.0)
    assert len(dlg._fit.ranges) == 1

    # Click a (notional) other range: no consumer -> selection unchanged.
    dlg._preview.range_select_requested.emit(5)
    assert dlg.active_range_index() == 0

    # Right-drag exclude on the single range STILL carves a window: the base's
    # exclude_region_requested wiring is untouched by the add/select override.
    assert dlg._fit.ranges[0].windows in (None, [])
    dlg._preview.exclude_region_requested.emit(0, 180.0, 220.0)
    assert dlg._fit.ranges[0].windows  # a gap was carved -> windows now populated

    assert app is not None
