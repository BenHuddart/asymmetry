"""Tests for ModelFitDialog range-parameter labels and bounds normalization."""

from __future__ import annotations

import os
import threading

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtWidgets import QApplication, QCheckBox, QDialog

from asymmetry.core.fitting.parameter_models import (
    ModelFitRange,
    ParameterCompositeModel,
    ParameterModelFit,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.gui.panels.model_fit_dialog import (
    _SC_COMPONENT_MENU_TITLE,
    ModelFitDialog,
    ParameterModelBuilderDialog,
    _component_pool_for_context,
    _ComponentSelectorButton,
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

    model = ParameterCompositeModel(
        ["Redfield", "DiffusionLF_2D", "Lambda_bg"], operators=["+", "+"]
    )
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

    # Wait for the worker→main-thread signal chain to complete by running a
    # proper nested event loop.  A manual processEvents() poll is not reliable
    # after a long test suite run because inter-thread queued signals require
    # the event loop to be *entered*, not just poked.
    _loop = QEventLoop()
    _check = QTimer()
    _check.timeout.connect(lambda: _loop.quit() if not dlg._fit_in_progress else None)
    _check.start(20)
    QTimer.singleShot(30000, _loop.quit)
    _loop.exec()
    _check.stop()

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

    monkeypatch.setattr(
        "asymmetry.gui.panels.model_fit_dialog.ParameterModelBuilderDialog", _FakeBuilder
    )
    dlg._edit_model(0)

    params = dlg.get_model_fit().ranges[0].parameters
    assert params["m"].value == pytest.approx(2.0)


def test_edit_model_to_sc_component_keeps_shape_factor_a_fixed_by_default(
    qapp: QApplication, monkeypatch
) -> None:
    x = np.linspace(1.0, 10.0, 20)
    y = 0.01 * x + 0.2
    yerr = np.full_like(x, 0.01)

    fit = ParameterModelFit(
        parameter_name="Lambda",
        x_key="temperature",
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
        x_key="temperature",
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
            return ParameterCompositeModel(["SC_PWaveAxial"], [])

    monkeypatch.setattr(
        "asymmetry.gui.panels.model_fit_dialog.ParameterModelBuilderDialog", _FakeBuilder
    )
    dlg._edit_model(0)

    params = dlg.get_model_fit().ranges[0].parameters
    assert params["shape_factor_a"].fixed is True
    assert params["shape_factor_a"].value == 0.0

    row_by_name = {
        dlg._param_table.item(row, 0).data(Qt.ItemDataRole.UserRole): row
        for row in range(dlg._param_table.rowCount())
    }
    fixed_container = dlg._param_table.cellWidget(row_by_name["shape_factor_a"], 4)
    assert fixed_container is not None
    checkboxes = fixed_container.findChildren(QCheckBox)
    assert len(checkboxes) == 1
    assert checkboxes[0].isChecked() is True


def test_parameter_model_builder_has_info_column(qapp: QApplication) -> None:
    dialog = ParameterModelBuilderDialog(component_pool=["Linear", "Arrhenius"])

    assert dialog._info_button.text() == "Info"
    assert dialog._expression_edit.text() == "Arrhenius"


def test_parameter_model_builder_accepts_parenthesized_expression(qapp: QApplication) -> None:
    dialog = ParameterModelBuilderDialog(component_pool=["Linear", "Arrhenius", "Constant"])
    dialog._expression_edit.setText("Linear + ( Arrhenius * Constant )")

    dialog._on_accept()
    model = dialog.get_model()

    assert model is not None
    assert model.component_names == ["Linear", "Arrhenius", "Constant"]
    assert model.operators == ["+", "*"]
    assert model.open_parentheses == [0, 1, 0]
    assert model.close_parentheses == [0, 0, 1]


def test_parameter_model_builder_offers_quadrature_operator(qapp: QApplication) -> None:
    """The parameter-model builder exposes a ⊕ keypad button and accepts a
    quadrature expression (the operator is parameter-grammar only)."""
    dialog = ParameterModelBuilderDialog(component_pool=["PowerLaw", "Constant"])
    assert "⊕" in dialog._extra_token_buttons

    dialog._expression_edit.setText("PowerLaw ⊕ Constant")
    dialog._on_accept()
    model = dialog.get_model()
    assert model is not None
    assert model.operators == ["⊕"]
    assert model.component_names == ["PowerLaw", "Constant"]


def test_parameter_model_builder_groups_sc_models_in_submenu(qapp: QApplication) -> None:
    dialog = ParameterModelBuilderDialog(component_pool=["Linear", "SC_SWave", "SC_DWave"])

    selector = dialog._component_selector
    assert isinstance(selector, _ComponentSelectorButton)

    menu = selector._build_component_menu()
    assert menu is not None

    top_actions = menu.actions()
    assert any(action.text() == "Linear" and action.menu() is None for action in top_actions)

    sc_action = next(
        (
            action
            for action in top_actions
            if action.menu() is not None and action.text() == _SC_COMPONENT_MENU_TITLE
        ),
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


def test_component_info_html_explains_shape_factor_a_fallbacks() -> None:
    from asymmetry.core.fitting.parameter_models import PARAMETER_MODEL_COMPONENTS

    ani_html = build_component_info_html(PARAMETER_MODEL_COMPONENTS["SC_AnisotropicS_Cos4"])
    p_html = build_component_info_html(PARAMETER_MODEL_COMPONENTS["SC_PWaveAxial"])
    ext_html = build_component_info_html(PARAMETER_MODEL_COMPONENTS["SC_ExtendedS"])

    assert "shape_factor_a" in ani_html
    assert "Carrington-Manzano" in ani_html
    assert "positive fixed value or allows it to vary" in ani_html

    assert "shape_factor_a" in p_html
    assert "Carrington-Manzano" in p_html
    assert "positive fixed or fitted value" in p_html

    assert "a = 4/3" in ext_html
    assert "Carrington-Manzano" in ext_html


# ---------------------------------------------------------------------------
# WiMDA Model-layer machinery (Phase 2): error modes, windows, quality verdict
# ---------------------------------------------------------------------------


def _make_dialog(qapp: QApplication) -> ModelFitDialog:
    x = np.linspace(1.0, 10.0, 20)
    y = 2.0 * x + 1.0
    yerr = np.full_like(x, 0.1)
    return ModelFitDialog(
        parameter_name="Lambda",
        x_key="field",
        x_values=x,
        y_values=y,
        y_errors=yerr,
        existing_fit=None,
    )


def _wait_for_fit(dlg: ModelFitDialog) -> None:
    loop = QEventLoop()
    check = QTimer()
    check.timeout.connect(lambda: loop.quit() if not dlg._fit_in_progress else None)
    check.start(20)
    QTimer.singleShot(30000, loop.quit)
    loop.exec()
    check.stop()


def test_error_mode_selector_defaults_to_column_and_toggles_value_field(
    qapp: QApplication,
) -> None:
    from asymmetry.core.fitting.parameter_models import ErrorMode

    dlg = _make_dialog(qapp)
    assert dlg._error_mode() is ErrorMode.COLUMN
    assert not dlg._error_value_spin.isEnabled()
    assert dlg._error_value() is None

    for index in range(dlg._error_mode_combo.count()):
        dlg._error_mode_combo.setCurrentIndex(index)
        mode = dlg._error_mode()
        needs_value = mode in (ErrorMode.PERCENT, ErrorMode.ABSOLUTE)
        assert dlg._error_value_spin.isEnabled() == needs_value
        assert (dlg._error_value() is not None) == needs_value


def test_run_fit_passes_error_mode_and_windows_to_core(qapp: QApplication, monkeypatch) -> None:
    from asymmetry.core.fitting.parameter_models import ErrorMode, ParameterModelFitResult

    dlg = _make_dialog(qapp)
    dlg._fit.ranges[0].windows = [(1.0, 4.0), (7.0, 10.0)]
    idx = dlg._error_mode_combo.findData(ErrorMode.PERCENT.value)
    dlg._error_mode_combo.setCurrentIndex(idx)
    dlg._error_value_spin.setValue(7.5)

    captured: dict[str, object] = {}

    def _fake_fit(**kwargs):
        captured.update(kwargs)
        return ParameterModelFitResult(success=True, reduced_chi_squared=1.0)

    monkeypatch.setattr("asymmetry.gui.panels.model_fit_dialog.fit_parameter_model", _fake_fit)
    dlg._run_fit(0)
    _wait_for_fit(dlg)

    assert captured["error_mode"] is ErrorMode.PERCENT
    assert captured["error_value"] == 7.5
    assert captured["windows"] == [(1.0, 4.0), (7.0, 10.0)]


def test_window_editor_round_trips_model_fit_range(qapp: QApplication) -> None:
    dlg = _make_dialog(qapp)
    fit_range = dlg._fit.ranges[0]
    assert fit_range.windows is None

    # First click seeds one window from the current bounds; second appends.
    dlg._add_window(0)
    assert dlg._fit.ranges[0].windows == [(1.0, 10.0)]
    dlg._add_window(0)
    assert len(dlg._fit.ranges[0].windows) == 2
    assert "∪" in dlg._range_selector.currentText()

    # Range bounds are disabled while windows drive the mask.
    assert not dlg._range_widgets[0].x_min.isEnabled()
    assert not dlg._range_widgets[0].x_max.isEnabled()

    dlg._on_window_bounds_changed(0, 0, 1, 4.0)
    assert dlg._fit.ranges[0].windows[0] == (1.0, 4.0)

    # Removing all windows restores plain min/max behaviour.
    dlg._remove_window(0, 1)
    dlg._remove_window(0, 0)
    assert dlg._fit.ranges[0].windows is None
    assert dlg._range_widgets[0].x_min.isEnabled()


def test_run_fit_rejects_inverted_window(qapp: QApplication, monkeypatch) -> None:
    dlg = _make_dialog(qapp)
    dlg._fit.ranges[0].windows = [(5.0, 2.0)]
    warnings: list[str] = []
    monkeypatch.setattr(
        "asymmetry.gui.panels.model_fit_dialog._show_warning",
        lambda _parent, _title, text: warnings.append(text),
    )
    dlg._run_fit(0)
    assert warnings and "inverted" in warnings[0]
    assert not dlg._fit_in_progress


def test_quality_verdict_shown_for_column_mode_fit(qapp: QApplication) -> None:
    from asymmetry.core.fitting.parameter_models import ParameterModelFitResult

    dlg = _make_dialog(qapp)
    fit_range = dlg._fit.ranges[0]
    fit_range.result = ParameterModelFitResult(
        success=True,
        chi_squared=9.0,
        reduced_chi_squared=0.9,
        parameters=fit_range.parameters,
        error_mode="column",
        n_points=12,
    )
    dlg._select_range(0)
    text = dlg._quality_label.text()
    assert "Quality of fit" in text
    assert "good" in text
    assert "target band" in text
    assert dlg._quality_label.toolTip()


def test_quality_verdict_suppressed_for_scatter_mode(qapp: QApplication) -> None:
    from asymmetry.core.fitting.parameter_models import ParameterModelFitResult

    dlg = _make_dialog(qapp)
    fit_range = dlg._fit.ranges[0]
    fit_range.result = ParameterModelFitResult(
        success=True,
        chi_squared=10.0,
        reduced_chi_squared=1.0,
        parameters=fit_range.parameters,
        error_mode="scatter",
        n_points=12,
    )
    dlg._select_range(0)
    text = dlg._quality_label.text()
    assert "No χ² quality verdict" in text

    fit_range.result = None
    dlg._select_range(0)
    assert dlg._quality_label.text() == ""


def test_window_bounds_edit_invalidates_stale_result(qapp: QApplication) -> None:
    from asymmetry.core.fitting.parameter_models import ParameterModelFitResult

    dlg = _make_dialog(qapp)
    fit_range = dlg._fit.ranges[0]
    fit_range.windows = [(1.0, 5.0)]
    fit_range.result = ParameterModelFitResult(
        success=True,
        chi_squared=9.0,
        reduced_chi_squared=0.9,
        parameters=fit_range.parameters,
        error_mode="column",
        n_points=12,
    )
    dlg._rebuild_ranges_ui()
    dlg._select_range(0)
    assert "Quality of fit" in dlg._quality_label.text()

    dlg._on_window_bounds_changed(0, 0, 1, 4.0)

    assert fit_range.result is None
    assert fit_range.windows == [(1.0, 4.0)]
    assert dlg._quality_label.text() == ""
    assert "not yet run" in dlg._chi2_label.text().lower()
    assert "Not run" in dlg._range_widgets[0].status_label.text()


def test_range_bounds_edit_invalidates_stale_result(qapp: QApplication) -> None:
    from asymmetry.core.fitting.parameter_models import ParameterModelFitResult

    dlg = _make_dialog(qapp)
    fit_range = dlg._fit.ranges[0]
    fit_range.result = ParameterModelFitResult(
        success=True, reduced_chi_squared=1.0, parameters=fit_range.parameters
    )
    dlg._range_widgets[0].x_max.setValue(8.0)
    assert fit_range.result is None


def test_param_edit_clears_quality_label(qapp: QApplication) -> None:
    from asymmetry.core.fitting.parameter_models import ParameterModelFitResult

    dlg = _make_dialog(qapp)
    fit_range = dlg._fit.ranges[0]
    fit_range.result = ParameterModelFitResult(
        success=True,
        chi_squared=9.0,
        reduced_chi_squared=0.9,
        parameters=fit_range.parameters,
        error_mode="column",
        n_points=12,
    )
    dlg._select_range(0)
    assert dlg._quality_label.text()

    dlg._on_param_table_edited()

    assert dlg._quality_label.text() == ""
    assert fit_range.result is None


def test_quality_verdict_silent_for_unknown_point_count(qapp: QApplication) -> None:
    """Results built outside fit_parameter_model (cross-group bridge, legacy
    saved state) have n_points=0 — say nothing rather than implying the fit
    had no degrees of freedom."""
    from asymmetry.core.fitting.parameter_models import ParameterModelFitResult

    dlg = _make_dialog(qapp)
    fit_range = dlg._fit.ranges[0]
    fit_range.result = ParameterModelFitResult(
        success=True,
        chi_squared=9.0,
        reduced_chi_squared=0.9,
        parameters=fit_range.parameters,
    )
    assert dlg._quality_text_for_range(fit_range) == ""
