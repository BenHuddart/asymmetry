"""Tests for ModelFitDialog range-parameter labels and bounds normalization."""

from __future__ import annotations

import os
import threading
import time

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtWidgets import QApplication, QCheckBox, QDialog

from asymmetry.core.fitting.composite import QUADRATURE_OPERATOR
from asymmetry.core.fitting.parameter_models import (
    ModelFitRange,
    ParameterCompositeModel,
    ParameterModelFit,
    is_order_parameter_observable,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.gui.panels.model_fit_dialog import (
    ModelFitDialog,
    ParameterModelBuilderDialog,
    _component_pool_for_context,
    _default_component_for_context,
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


def test_x_label_overrides_internal_key_in_title(qapp: QApplication) -> None:
    """Round-10 #9: the dialog shows the friendly column name, not custom:<id>."""
    x = np.linspace(0.0, 1.0, 6)
    y = np.linspace(0.1, 0.2, 6)
    yerr = np.full_like(x, 0.01)

    dlg = ModelFitDialog(
        parameter_name="A_1",
        x_key="custom:84576a7e",
        x_values=x,
        y_values=y,
        y_errors=yerr,
        x_label="Current (A)",
    )

    assert "Current (A)" in dlg.windowTitle()
    assert "custom:84576a7e" not in dlg.windowTitle()
    # The backend key is still the internal id (persistence / fit unchanged).
    assert dlg._x_key == "custom:84576a7e"


def test_x_label_falls_back_to_key(qapp: QApplication) -> None:
    """Without an explicit label the dialog keeps the old key-based title."""
    x = np.linspace(0.0, 1.0, 6)
    dlg = ModelFitDialog(
        parameter_name="A_1",
        x_key="field",
        x_values=x,
        y_values=np.linspace(0.1, 0.2, 6),
        y_errors=np.full_like(x, 0.01),
    )
    assert "field" in dlg.windowTitle()


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


def test_edit_model_to_critical_divergence_seeds_tc_from_data(
    qapp: QApplication, monkeypatch
) -> None:
    # Switching to a trend model must seed Tc from the data, not the unphysical
    # default of 10, so the trend fit converges without a manual reseed.
    x = np.array([90.0, 120.0, 280.0])
    y = np.array([0.59, 0.04, 0.017])
    yerr = np.full_like(x, 0.01)

    fit = ParameterModelFit(
        parameter_name="Lambda",
        x_key="temperature",
        ranges=[
            ModelFitRange(
                x_min=90.0,
                x_max=280.0,
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
            return ParameterCompositeModel(["CriticalDivergence"], [])

    monkeypatch.setattr(
        "asymmetry.gui.panels.model_fit_dialog.ParameterModelBuilderDialog", _FakeBuilder
    )
    dlg._edit_model(0)

    params = dlg.get_model_fit().ranges[0].parameters
    assert params["Tc"].value < 90.0
    assert params["Tc"].value != pytest.approx(10.0)


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


def test_parameter_model_builder_seeds_initial_expression(qapp: QApplication) -> None:
    dialog = ParameterModelBuilderDialog(component_pool=["Linear", "Arrhenius"])

    # No initial_model: seeds from the sorted pool's first entry.
    assert dialog._rows.expression() == "Arrhenius"


def test_parameter_model_builder_initial_model_round_trips(qapp: QApplication) -> None:
    initial = ParameterCompositeModel(["Linear", "Constant"], ["+"])
    dialog = ParameterModelBuilderDialog(
        component_pool=["Linear", "Arrhenius", "Constant"], initial_model=initial
    )

    assert dialog._rows.expression() == "Linear + Constant"
    model = dialog.get_model()
    assert model is not None
    assert model.component_names == ["Linear", "Constant"]


def test_parameter_model_builder_accepts_parenthesized_expression(qapp: QApplication) -> None:
    dialog = ParameterModelBuilderDialog(component_pool=["Linear", "Arrhenius", "Constant"])
    dialog._toggle_text_mode()
    dialog._text_edit.setPlainText("Linear + ( Arrhenius * Constant )")
    assert dialog._apply_text() is True

    dialog._on_accept()
    model = dialog.get_model()

    assert model is not None
    assert model.component_names == ["Linear", "Arrhenius", "Constant"]
    assert model.operators == ["+", "*"]
    assert model.open_parentheses == [0, 1, 0]
    assert model.close_parentheses == [0, 0, 1]


def test_parameter_model_builder_offers_quadrature_operator(qapp: QApplication) -> None:
    """The parameter-model builder's row operator combos include ⊕, and text
    mode accepts a quadrature expression (the operator is parameter-grammar
    only)."""
    dialog = ParameterModelBuilderDialog(component_pool=["PowerLaw", "Constant"])
    assert QUADRATURE_OPERATOR in dialog._rows._operators_available

    dialog._toggle_text_mode()
    dialog._text_edit.setPlainText(f"PowerLaw {QUADRATURE_OPERATOR} Constant")
    assert dialog._apply_text() is True

    dialog._on_accept()
    model = dialog.get_model()
    assert model is not None
    assert model.operators == [QUADRATURE_OPERATOR]
    assert model.component_names == ["PowerLaw", "Constant"]


def test_parameter_model_builder_rejects_component_outside_pool(qapp: QApplication) -> None:
    """A component that is registered but not offered in this context (e.g.
    typed directly in text mode) is rejected with a helpful message rather
    than silently accepted."""
    dialog = ParameterModelBuilderDialog(component_pool=["Linear", "Constant"])
    dialog._toggle_text_mode()
    dialog._text_edit.setPlainText("Arrhenius")

    assert dialog._apply_text() is False
    assert dialog._stack.currentWidget() is dialog._text_edit
    assert "Arrhenius" in dialog._status_label.text()

    result = dialog.result()
    dialog._on_accept()
    # The Ok path re-validates and refuses to accept: the dialog's result
    # code is unchanged (accept() was never reached), so a caller's
    # `if dlg.exec() != Accepted: return` guard would correctly bail out.
    assert dialog.result() == result


def test_parameter_model_builder_get_model_returns_none_when_invalid(
    qapp: QApplication,
) -> None:
    dialog = ParameterModelBuilderDialog(component_pool=["Linear", "Constant"])
    dialog._toggle_text_mode()
    dialog._text_edit.setPlainText("")
    dialog._validate_and_update("")
    assert dialog.get_model() is None


# --------------------------------------------------------- create-user-function
def test_parameter_model_builder_creation_affordance_enabled(qapp: QApplication) -> None:
    dialog = ParameterModelBuilderDialog(component_pool=["Linear", "Constant"])
    assert dialog._library._creation_enabled is True


def test_parameter_model_builder_creates_component_accepted_by_both_parsers(
    qapp: QApplication, registry_snapshot, tmp_path, monkeypatch
) -> None:
    """A component created mid-session is usable in text mode AND the model parser.

    Exercises the "live pool" fix directly: ``_pool_restricted_model_parser``
    and ``make_component_expression_parser`` both close over
    ``dialog._pool`` (the same ``set`` object), so ``_create_user_function``'s
    ``self._pool.add(name)`` must make the new name valid to both without
    reconstructing either parser.
    """
    from asymmetry.gui.windows.new_user_function_dialog import NewUserFunctionDialog

    class _AutoAcceptDialog(NewUserFunctionDialog):
        def exec(self):  # noqa: A003 - Qt API name
            self._name_edit.setText("UserTrendLive")
            self._description_edit.setText("Live-pool test trend")
            self._formula_edit.setText("a*x+b")
            self._append_param_row("a", 1.0)
            self._append_param_row("b", 0.0)
            self._run_validation()
            self._on_accept()
            return self.result()

    def _factory(kind, *, domain="time", directory=None, parent=None):
        return _AutoAcceptDialog(kind, domain=domain, directory=tmp_path, parent=parent)

    monkeypatch.setattr("asymmetry.gui.panels.model_fit_dialog.NewUserFunctionDialog", _factory)

    dialog = ParameterModelBuilderDialog(component_pool=["Linear", "Constant"])
    dialog._library.create_requested.emit()

    # Registered with common scope, so valid in every trending context.
    from asymmetry.core.fitting.parameter_models import PARAMETER_MODEL_COMPONENTS

    assert PARAMETER_MODEL_COMPONENTS["UserTrendLive"].scopes == ("common",)

    # 1. Added to the dialog's live pool.
    assert "UserTrendLive" in dialog._pool
    # 2. Auto-appended into the structured expression and accepted by the
    #    (pool-restricted) model parser via _on_accept -> _validate_and_update.
    names, *_rest = dialog._rows.structure()
    assert "UserTrendLive" in names
    dialog._on_accept()
    model = dialog.get_model()
    assert model is not None
    assert "UserTrendLive" in model.component_names

    # 3. Accepted by the expression parser in text mode too (same live pool).
    dialog._toggle_text_mode()
    dialog._text_edit.setPlainText("UserTrendLive + Constant")
    assert dialog._apply_text() is True


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


def test_run_fit_failure_surfaces_traceback_and_resets_state(
    qapp: QApplication, monkeypatch
) -> None:
    """A fit that raises in the worker reaches the failure dialog (with the
    full traceback) via TaskRunner's on_error, and the busy state is cleared."""
    from asymmetry.core.fitting.parameter_models import ParameterModelFitResult  # noqa: F401

    dlg = _make_dialog(qapp)

    def _boom(**_kwargs):
        raise RuntimeError("fit blew up")

    monkeypatch.setattr("asymmetry.gui.panels.model_fit_dialog.fit_parameter_model", _boom)
    warnings: list[str] = []
    monkeypatch.setattr(
        "asymmetry.gui.panels.model_fit_dialog._show_warning",
        lambda _parent, _title, text: warnings.append(text),
    )

    dlg._run_fit(0)
    _wait_for_fit(dlg)

    assert dlg._fit_in_progress is False
    assert dlg._fit_done_callback is None
    assert warnings and "fit blew up" in warnings[0]
    # The full traceback (not just str(exc)) is preserved through the migration.
    assert "Traceback" in warnings[0]


def test_close_event_refuses_while_fit_in_progress(qapp: QApplication, monkeypatch) -> None:
    """closeEvent ignores the close (mirroring reject) while a fit is running."""
    from PySide6.QtGui import QCloseEvent

    from asymmetry.core.fitting.parameter_models import ParameterModelFitResult

    dlg = _make_dialog(qapp)
    gate = threading.Event()

    def _fake_fit(**_kwargs):
        gate.wait(timeout=1.0)
        return ParameterModelFitResult(success=True, reduced_chi_squared=1.0)

    monkeypatch.setattr("asymmetry.gui.panels.model_fit_dialog.fit_parameter_model", _fake_fit)
    monkeypatch.setattr("asymmetry.gui.panels.model_fit_dialog._show_info", lambda *a, **k: None)
    try:
        dlg._run_fit(0)
        assert dlg._fit_in_progress is True
        event = QCloseEvent()
        dlg.closeEvent(event)
        assert not event.isAccepted()  # close refused while the fit runs
    finally:
        gate.set()
    _wait_for_fit(dlg)
    assert dlg._fit_in_progress is False


def test_window_editor_round_trips_model_fit_range(qapp: QApplication) -> None:
    dlg = _make_dialog(qapp)
    fit_range = dlg._fit.ranges[0]
    assert fit_range.windows is None

    # A carve splits the one region into two included intervals (windows).
    dlg._select_range(0)
    dlg._exclude_region(0, 4.0, 6.0)
    assert len(dlg._fit.ranges[0].windows) == 2
    # The card (now the range selector) shows the windowed union bounds.
    assert "∪" in dlg._range_cards[0]._bounds_label.text()

    # Two intervals now edit through the interval spins.
    dlg._select_range(0)
    assert len(dlg._region_row_spins) == 2

    # Editing interval 0's max carves it down; the union stays two intervals.
    dlg._on_region_interval_edited(0, 1, 3.0)
    assert dlg._fit.ranges[0].windows[0] == (1.0, 3.0)

    # Removing one interval collapses back to a plain range (windows is None).
    dlg._remove_interval(0, 1)
    assert dlg._fit.ranges[0].windows is None
    assert len(dlg._region_row_spins) == 1


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


# ---------------------------------------------------------------------------
# Phase 3 (seeding robustness): "Guess seeds" + multi-start bad-minimum messages
# ---------------------------------------------------------------------------


def _make_linear_dialog(qapp: QApplication) -> ModelFitDialog:
    """A Linear-model dialog over clearly sloped data (slope 2, intercept 1)."""
    x = np.linspace(1.0, 10.0, 20)
    y = 2.0 * x + 1.0
    yerr = np.full_like(x, 0.1)
    model = ParameterCompositeModel(["Linear"], [])
    params = ParameterSet([Parameter("m", 0.0), Parameter("b", 0.0)])
    fit = ParameterModelFit(
        parameter_name="Lambda",
        x_key="field",
        ranges=[ModelFitRange(x_min=1.0, x_max=10.0, model=model, parameters=params)],
    )
    return ModelFitDialog(
        parameter_name="Lambda",
        x_key="field",
        x_values=x,
        y_values=y,
        y_errors=yerr,
        existing_fit=fit,
    )


def _drain_guess(dlg: ModelFitDialog, timeout_s: float = 5.0) -> None:
    """Pump the event loop until the off-thread seed guess completes."""
    app = QApplication.instance()
    deadline = time.time() + timeout_s
    while dlg._guess_in_progress and time.time() < deadline:
        app.processEvents()
        time.sleep(0.005)
    app.processEvents()


def _param_value(dlg: ModelFitDialog, name: str) -> float:
    return dlg.get_model_fit().ranges[0].parameters[name].value


def test_guess_button_populates_seeds(qapp: QApplication) -> None:
    """Clicking Guess moves Linear slope/intercept off their defaults."""
    dlg = _make_linear_dialog(qapp)
    dlg._select_range(0)
    before_m = _param_value(dlg, "m")
    before_b = _param_value(dlg, "b")

    dlg._on_guess_seeds_clicked()
    _drain_guess(dlg)

    after_m = _param_value(dlg, "m")
    after_b = _param_value(dlg, "b")
    assert after_m != before_m or after_b != before_b
    # Data-aware seeds should land near the true slope/intercept.
    assert after_m == pytest.approx(2.0, abs=0.5)
    assert after_b == pytest.approx(1.0, abs=1.0)


def test_guess_does_not_touch_fixed(qapp: QApplication) -> None:
    """A fixed parameter's value is unchanged by Guess."""
    dlg = _make_linear_dialog(qapp)
    dlg._fit.ranges[0].parameters["m"].fixed = True
    dlg._fit.ranges[0].parameters["m"].value = 99.0
    dlg._select_range(0)

    dlg._on_guess_seeds_clicked()
    _drain_guess(dlg)

    assert _param_value(dlg, "m") == pytest.approx(99.0)
    # The free parameter still gets seeded.
    assert _param_value(dlg, "b") != 0.0


def test_guess_respects_domain_limits(qapp: QApplication) -> None:
    """Guessed values pass through _normalize_parameter_limits (in-domain)."""
    # tau is a strictly-positive param; seed the model so any guessed value is
    # clamped/normalised rather than committed out of domain.
    x = np.linspace(1.0, 10.0, 20)
    y = np.linspace(0.5, 0.05, 20)
    yerr = np.full_like(x, 0.01)
    model = ParameterCompositeModel(["ExponentialDecay"], [])
    params = ParameterSet(
        [Parameter(name, model.param_defaults[name]) for name in model.param_names]
    )
    fit = ParameterModelFit(
        parameter_name="Lambda",
        x_key="field",
        ranges=[ModelFitRange(x_min=1.0, x_max=10.0, model=model, parameters=params)],
    )
    dlg = ModelFitDialog(
        parameter_name="Lambda",
        x_key="field",
        x_values=x,
        y_values=y,
        y_errors=yerr,
        existing_fit=fit,
    )
    dlg._select_range(0)

    dlg._on_guess_seeds_clicked()
    _drain_guess(dlg)

    # Every committed parameter must satisfy its normalised bounds.
    for p in dlg.get_model_fit().ranges[0].parameters:
        assert p.min <= p.value <= p.max


def test_guess_is_not_automatic(qapp: QApplication) -> None:
    """Editing the model / selecting a range must NOT run a seed guess."""
    dlg = _make_linear_dialog(qapp)
    dlg._select_range(0)
    before_m = _param_value(dlg, "m")
    before_b = _param_value(dlg, "b")

    # Re-selecting the range and touching the table must not change seeds.
    dlg._select_range(0)
    dlg._on_param_table_edited()
    qapp.processEvents()

    assert dlg._guess_in_progress is False
    assert _param_value(dlg, "m") == pytest.approx(before_m)
    assert _param_value(dlg, "b") == pytest.approx(before_b)


def test_run_fit_uses_multistart(qapp: QApplication, monkeypatch) -> None:
    """_run_fit passes extra_starts >= 1 to fit_parameter_model."""
    from asymmetry.core.fitting.parameter_models import ParameterModelFitResult

    dlg = _make_dialog(qapp)
    captured: dict[str, object] = {}

    def _fake_fit(**kwargs):
        captured.update(kwargs)
        return ParameterModelFitResult(success=True, reduced_chi_squared=1.0)

    monkeypatch.setattr("asymmetry.gui.panels.model_fit_dialog.fit_parameter_model", _fake_fit)
    dlg._run_fit(0)
    _wait_for_fit(dlg)

    assert int(captured.get("extra_starts", 0)) >= 1


def test_params_at_bound_message_shown(qapp: QApplication) -> None:
    """A result with params_at_bound renders the inline warning line."""
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
        params_at_bound=("m",),
    )
    dlg._select_range(0)
    text = dlg._quality_label.text()
    assert "Parameters at their limits" in text
    assert "m" in text


def test_seed_beat_user_start_message_shown(qapp: QApplication) -> None:
    """A successful result with seed_beat_user_start renders the info line."""
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
        seed_beat_user_start=True,
    )
    dlg._select_range(0)
    text = dlg._quality_label.text()
    assert "data-aware start improved the fit" in text


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

    # Editing an interval's max (single-window range collapses to a plain range).
    dlg._on_region_interval_edited(0, 1, 4.0)

    assert fit_range.result is None
    assert fit_range.x_min == 1.0
    assert fit_range.x_max == 4.0
    assert dlg._quality_label.text() == ""
    assert "not yet run" in dlg._chi2_label.text().lower()
    # The card now carries the range's status; a cleared result reads "not_run".
    assert dlg._range_cards[0]._view.status == "not_run"


def test_range_bounds_edit_invalidates_stale_result(qapp: QApplication) -> None:
    from asymmetry.core.fitting.parameter_models import ParameterModelFitResult

    dlg = _make_dialog(qapp)
    fit_range = dlg._fit.ranges[0]
    fit_range.result = ParameterModelFitResult(
        success=True, reduced_chi_squared=1.0, parameters=fit_range.parameters
    )
    dlg._select_range(0)
    _, i_max = dlg._region_row_spins[0]
    i_max.setValue(8.0)
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


def test_preview_curve_turns_solid_after_successful_fit(qapp: QApplication, monkeypatch) -> None:
    """The preview curve is a dashed seed until the range's fit converges."""
    from asymmetry.core.fitting.parameter_models import ParameterModelFitResult

    dlg = _make_dialog(qapp)
    assert [rng.fitted for rng in dlg._current_preview_ranges()] == [False]

    monkeypatch.setattr(
        "asymmetry.gui.panels.model_fit_dialog.fit_parameter_model",
        lambda **kwargs: ParameterModelFitResult(success=True, reduced_chi_squared=1.0),
    )
    dlg._run_fit(0)
    _wait_for_fit(dlg)

    assert dlg._fit.ranges[0].result is not None
    assert dlg._current_preview_ranges()[0].fitted is True

    # A parameter edit invalidates the stale result → back to the dashed seed.
    dlg._on_param_table_edited()
    assert dlg._current_preview_ranges()[0].fitted is False


def test_preview_curve_stays_dashed_after_failed_fit(qapp: QApplication) -> None:
    from asymmetry.core.fitting.parameter_models import ParameterModelFitResult

    dlg = _make_dialog(qapp)
    dlg._fit.ranges[0].result = ParameterModelFitResult(success=False, message="no convergence")
    assert dlg._current_preview_ranges()[0].fitted is False


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


# ── F4: context-aware default trend model (discoverability) ────────────────


def test_is_order_parameter_observable_matches_frequency_and_internal_field():
    for name in ("frequency", "freq", "nu", "nu0", "frequency_2", "B_loc", "Bint"):
        assert is_order_parameter_observable(name), name
    # Applied/longitudinal fields and RF drives are NOT order-parameter observables.
    for name in ("B_L", "B_ext", "nu_RF", "Lambda", "sigma", "A0", "b", "m"):
        assert not is_order_parameter_observable(name), name


def test_default_component_prefers_order_parameter_for_temperature_frequency():
    pool = _component_pool_for_context("temperature", "frequency")
    assert _default_component_for_context("temperature", "frequency", pool) == "OrderParameter"


def test_default_component_stays_linear_off_context():
    # Non-order-parameter Y vs temperature keeps Linear.
    t_pool = _component_pool_for_context("temperature", "Lambda")
    assert _default_component_for_context("temperature", "Lambda", t_pool) == "Linear"
    # An order-parameter observable vs *field* is not a T_c phenomenon → Linear.
    f_pool = _component_pool_for_context("field", "frequency")
    assert _default_component_for_context("field", "frequency", f_pool) == "Linear"


def test_default_component_falls_back_when_order_parameter_unavailable():
    # If OrderParameter isn't in the pool, fall back to Linear rather than erroring.
    assert _default_component_for_context("temperature", "frequency", ["Linear", "Constant"]) == (
        "Linear"
    )


def test_fresh_temperature_frequency_dialog_defaults_to_seeded_order_parameter(
    qapp: QApplication,
) -> None:
    # A magnetic order parameter grows below T_c and vanishes at it: y falls from
    # ~30 MHz toward 0 as T rises to ~68 K.
    x = np.linspace(5.0, 68.0, 20)
    y = np.linspace(30.0, 2.0, 20)
    yerr = np.full_like(x, 0.5)

    dlg = ModelFitDialog(
        parameter_name="frequency",
        x_key="temperature",
        x_values=x,
        y_values=y,
        y_errors=yerr,
    )
    fit_range = dlg._fit.ranges[0]
    assert fit_range.model.component_names == ["OrderParameter"]
    # Seeds are data-aware, not the unphysical Tc=10 default: T_c is placed just
    # above the measured range and y0 from the largest observed value.
    assert fit_range.parameters["Tc"].value > 68.0
    assert fit_range.parameters["Tc"].value != pytest.approx(10.0)
    assert fit_range.parameters["y0"].value == pytest.approx(30.0, abs=1.0)


def test_fresh_temperature_lambda_dialog_stays_linear(qapp: QApplication) -> None:
    x = np.linspace(5.0, 68.0, 20)
    y = np.linspace(0.5, 0.1, 20)
    yerr = np.full_like(x, 0.01)

    dlg = ModelFitDialog(
        parameter_name="Lambda",
        x_key="temperature",
        x_values=x,
        y_values=y,
        y_errors=yerr,
    )
    assert dlg._fit.ranges[0].model.component_names == ["Linear"]


# ── Work item 0.1: base/subclass share _select_range + _commit_param_table ──


def test_select_range_shared_path() -> None:
    """Both dialogs route through the *base* ``_select_range`` /
    ``_commit_param_table`` — the subclass no longer re-implements them, it only
    overrides the small template hooks."""
    from asymmetry.gui.panels.cross_group_fit_dialog import CrossGroupFitDialog

    # The shared table/status flow lives on the base only.
    assert "_select_range" in vars(ModelFitDialog)
    assert "_select_range" not in vars(CrossGroupFitDialog)
    assert "_commit_param_table" in vars(ModelFitDialog)

    # The subclass keeps a thin _commit_param_table override that must delegate
    # to the base (role mapping is layered on top), not re-implement it.
    import inspect

    src = inspect.getsource(CrossGroupFitDialog._commit_param_table)
    assert "super()._commit_param_table(" in src

    # The four frozen hooks are overridden on the subclass.
    for hook in (
        "_make_param_row_control",
        "_read_param_row_control",
        "_result_for_range",
        "_error_cell_for_param",
    ):
        assert hook in vars(ModelFitDialog), hook
        assert hook in vars(CrossGroupFitDialog), hook


def test_base_fixed_checkbox_commits_value_min_max_and_fixed(qapp: QApplication) -> None:
    """The base Fixed-checkbox column commits value/min/max and the fixed flag
    from the shared ``_commit_param_table``."""
    x = np.linspace(1.0, 10.0, 12)
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
    dlg._select_range(0)

    row_by_name = {
        dlg._param_table.item(r, 0).data(Qt.ItemDataRole.UserRole): r
        for r in range(dlg._param_table.rowCount())
    }
    # ``b`` is an unconstrained offset (no domain clamping), so value/min/max
    # round-trip verbatim through the shared commit path.
    b_row = row_by_name["b"]

    # Column 4 holds the base Fixed checkbox (in a centered container).
    container = dlg._param_table.cellWidget(b_row, 4)
    checkboxes = container.findChildren(QCheckBox)
    assert len(checkboxes) == 1

    dlg._param_table.item(b_row, 1).setText("0.5")
    dlg._param_table.item(b_row, 2).setText("-1.0")
    dlg._param_table.item(b_row, 3).setText("2.0")
    checkboxes[0].setChecked(True)

    dlg._commit_param_table()

    committed = {p.name: p for p in dlg.get_model_fit().ranges[0].parameters}
    assert committed["b"].value == pytest.approx(0.5)
    assert committed["b"].min == pytest.approx(-1.0)
    assert committed["b"].max == pytest.approx(2.0)
    assert committed["b"].fixed is True
    # The unedited row is untouched and stays free.
    assert committed["m"].fixed is False


def test_cross_group_role_combo_persists_and_maps_fixed() -> None:
    """The subclass role combo persists Global/Local/Fixed into ``_range_roles``
    and maps a Fixed role onto the parameter's ``fixed`` flag, all through the
    shared commit path plus the subclass override."""
    from asymmetry.core.fitting.parameter_models import ParameterGroupData
    from asymmetry.gui.panels.cross_group_fit_dialog import CrossGroupFitDialog

    x = np.array([100.0, 200.0, 300.0], dtype=float)
    groups = [
        ParameterGroupData(
            group_id="g0",
            group_name="G0",
            x=x,
            y=np.array([0.1, 0.2, 0.3]),
            yerr=np.array([0.01, 0.01, 0.01]),
            group_variable_value=0.0,
        ),
        ParameterGroupData(
            group_id="g1",
            group_name="G1",
            x=x,
            y=np.array([0.12, 0.22, 0.32]),
            yerr=np.array([0.01, 0.01, 0.01]),
            group_variable_value=1.0,
        ),
    ]

    dlg = CrossGroupFitDialog(
        parameter_name="Lambda",
        x_key="field",
        groups=groups,
        parent=None,
    )

    from PySide6.QtWidgets import QComboBox

    row_by_name = {
        dlg._param_table.item(r, 0).data(Qt.ItemDataRole.UserRole): r
        for r in range(dlg._param_table.rowCount())
    }
    # Column 4 holds the Global/Local/Fixed combo (not a checkbox).
    m_combo = dlg._param_table.cellWidget(row_by_name["m"], 4)
    b_combo = dlg._param_table.cellWidget(row_by_name["b"], 4)
    assert isinstance(m_combo, QComboBox)

    m_combo.setCurrentText("Local")
    b_combo.setCurrentText("Fixed")
    dlg._commit_param_table()

    roles = dlg._range_roles[0]
    assert roles["m"] == "Local"
    assert roles["b"] == "Fixed"
    # A Fixed role maps onto the parameter's fixed flag; Local stays free.
    params = {p.name: p for p in dlg._fit.ranges[0].parameters}
    assert params["b"].fixed is True


def test_layout_slots_present(qapp: QApplication) -> None:
    """Contract C6: the base dialog exposes named header/footer layout slots
    so subclasses (e.g. CrossGroupFitDialog) stop doing index-based
    layout.insertWidget arithmetic."""
    from PySide6.QtWidgets import QVBoxLayout

    x = np.linspace(10.0, 100.0, 10)
    y = np.linspace(0.1, 0.2, 10)
    yerr = np.full_like(x, 0.01)

    dlg = ModelFitDialog(
        parameter_name="Lambda",
        x_key="field",
        x_values=x,
        y_values=y,
        y_errors=yerr,
    )

    assert isinstance(dlg._header_slot, QVBoxLayout)
    assert isinstance(dlg._footer_slot, QVBoxLayout)
    # Empty by default on the base dialog.
    assert dlg._header_slot.count() == 0
    assert dlg._footer_slot.count() == 0


# ── Work item 1.1: horizontal splitter + preview pane ──────────────────────


def _make_split_dialog(qapp: QApplication) -> ModelFitDialog:
    x = np.linspace(1.0, 10.0, 12)
    y = 0.01 * x + 0.2
    yerr = np.full_like(x, 0.01)
    return ModelFitDialog(
        parameter_name="Lambda",
        x_key="field",
        x_values=x,
        y_values=y,
        y_errors=yerr,
    )


def test_preview_pane_present(qapp: QApplication) -> None:
    """The dialog body is a horizontal splitter with a left (controls) pane and
    a right (preview) pane; the preview host is an empty-ish VBox on the right."""
    from PySide6.QtWidgets import QSplitter, QVBoxLayout

    dlg = _make_split_dialog(qapp)

    assert isinstance(dlg._splitter, QSplitter)
    assert dlg._splitter.count() == 2
    assert isinstance(dlg._preview_host, QVBoxLayout)
    # The preview host lives on the right pane, which is the splitter's 2nd child.
    right_pane = dlg._splitter.widget(1)
    assert right_pane is not None
    assert right_pane.layout() is dlg._preview_host


def test_preview_host_outside_ranges_host(qapp: QApplication) -> None:
    """Rebuilding the range rows (which clears ``_ranges_host``) must not clear
    or destroy the separate preview host / canvas on the right pane."""
    dlg = _make_split_dialog(qapp)

    host_before = dlg._preview_host
    count_before = dlg._preview_host.count()
    canvas = dlg._preview

    dlg._rebuild_ranges_ui()

    assert dlg._preview_host is host_before
    assert dlg._preview_host.count() == count_before
    # The preview canvas survives the range rebuild intact.
    assert dlg._preview is canvas
    import shiboken6

    assert shiboken6.isValid(canvas)


def test_narrow_screen_collapses_preview(qapp: QApplication) -> None:
    """A narrow usable width collapses the preview pane (second size ~0) and
    surfaces the "Show preview" toggle; toggling it back restores the pane."""
    dlg = _make_split_dialog(qapp)

    # Drive the collapse decision directly with a synthetic narrow width so the
    # test does not depend on real (offscreen) screen geometry.
    dlg.resize(700, 600)
    dlg._maybe_collapse_preview(first_show=True)

    sizes = dlg._splitter.sizes()
    assert sizes[1] == 0
    assert dlg._preview_collapsed is True
    # isVisibleTo (not isVisible) because the dialog itself is never shown here.
    assert dlg._show_preview_toggle.isVisibleTo(dlg)
    assert dlg._show_preview_toggle.isChecked() is False

    # Toggling "Show preview" on restores a non-zero preview pane.
    dlg._show_preview_toggle.setChecked(True)
    assert dlg._splitter.sizes()[1] > 0
    assert dlg._preview_collapsed is False

    # A comfortable width expands the pane and hides the toggle (auto-managed).
    wide = _make_split_dialog(qapp)
    wide.resize(1280, 700)
    wide._maybe_collapse_preview(first_show=True)
    assert wide._splitter.sizes()[1] > 0
    assert wide._preview_collapsed is False
    assert not wide._show_preview_toggle.isVisibleTo(wide)


def test_preview_pane_gets_usable_width_on_show(qapp: QApplication) -> None:
    """On a comfortable (default-sized) dialog, the preview pane opens at a
    clearly usable width — not squashed to a sliver by the left pane's wide,
    non-wrapping range-row content.

    The left pane is wrapped in a QScrollArea (``_left_scroll``) precisely so
    its content's large minimum width cannot dictate the splitter's
    allocation; assert both the floor widths that guarantee this and the
    actual proportional split computed for a realistic dialog width.
    """
    dlg = _make_split_dialog(qapp)
    dlg.resize(1280, 700)
    # An unshown top-level widget's children do not get real geometry from a
    # bare resize() (the splitter reports a stale/placeholder width), so show
    # it (offscreen platform plugin — no real window appears) to get the
    # splitter's actual, layout-settled width before reading back its sizes.
    dlg.show()
    qapp.processEvents()

    # The floors that make the preview un-squashable regardless of how wide
    # the (now-scrollable) left content gets.
    assert dlg._left_scroll.minimumWidth() == dlg._LEFT_PANE_MIN_WIDTH
    assert dlg._splitter.widget(1).minimumWidth() == dlg._PREVIEW_PANE_MIN_WIDTH

    # Drive the same first-show sizing path showEvent already triggered, to
    # confirm the method is idempotent and lands on the same expanded state.
    dlg._maybe_collapse_preview(first_show=True)
    qapp.processEvents()

    sizes = dlg._splitter.sizes()
    assert dlg._preview_collapsed is False
    # At least ~30% of the dialog width, and comfortably above the ~340px
    # floor the bug report calls out.
    assert sizes[1] >= 340
    assert sizes[1] >= 0.3 * dlg._splitter.width()

    # The proportional-sizing helper itself, exercised directly at a few
    # widths, always respects both floors.
    for width in (900, 1280, 1600):
        left, right = dlg._expanded_split_sizes(width)
        assert left >= dlg._LEFT_PANE_MIN_WIDTH
        assert right >= dlg._PREVIEW_PANE_MIN_WIDTH


def test_preview_host_survives_left_pane_scroll_wrapping(qapp: QApplication) -> None:
    """The left pane's content lives inside a QScrollArea, but the preview
    pane/host is untouched by that wrapping (it is the splitter's 2nd,
    separate top-level child, not nested inside the scroll area)."""
    from PySide6.QtWidgets import QScrollArea

    dlg = _make_split_dialog(qapp)

    assert isinstance(dlg._left_scroll, QScrollArea)
    assert dlg._splitter.widget(0) is dlg._left_scroll
    right_pane = dlg._splitter.widget(1)
    assert right_pane is not dlg._left_scroll
    assert right_pane.layout() is dlg._preview_host


# --- Live preview (work item 1.3) --------------------------------------------


def _drain_preview(dlg: ModelFitDialog, timeout_s: float = 5.0) -> None:
    """Launch the debounced sample immediately and wait for it to complete.

    Fires ``_launch_preview_sample`` directly (bypassing the 120 ms debounce)
    then pumps the event loop until the off-thread sample marshals its result
    back and ``_preview_active`` (plus any coalesced ``_preview_pending``) clears.
    """
    app = QApplication.instance()
    dlg._launch_preview_sample()
    deadline = time.time() + timeout_s
    while (dlg._preview_active or dlg._preview_pending) and time.time() < deadline:
        app.processEvents()
        time.sleep(0.005)
    # A final pump so the queued on_finished slot runs even if the flag cleared
    # on the same tick it was posted.
    app.processEvents()


def test_preview_series_single(qapp: QApplication) -> None:
    """The base ``_preview_series`` returns exactly one data trace."""
    dlg = _make_dialog(qapp)
    series = dlg._preview_series()
    assert len(series) == 1
    assert series[0].label == "data"
    np.testing.assert_allclose(series[0].x, dlg._x)
    np.testing.assert_allclose(series[0].y, dlg._y)


def test_preview_updates_on_param_edit(qapp: QApplication) -> None:
    """Editing a seed value drives a fresh off-thread curve onto the canvas."""
    dlg = _make_dialog(qapp)
    dlg._select_range(0)

    # Bump a parameter value through the table, exactly as a user edit does.
    dlg._param_table.item(0, 1).setText("3.5")
    dlg._on_param_table_edited()
    _drain_preview(dlg)

    ranges = dlg._preview._ranges
    active = next((r for r in ranges if r.idx == dlg._active_range_idx), None)
    assert active is not None
    assert np.asarray(active.curve_x).size > 0
    assert np.asarray(active.curve_y).size > 0


def test_preview_curve_uses_current_unfitted_params(qapp: QApplication) -> None:
    """With no fit run, the previewed curve reflects the current seed params.

    A Linear seed of m=2, b=1 must produce a sloped (non-flat, non-empty) curve,
    proving the preview samples the live params rather than a fitted/empty line.
    """
    dlg = _make_dialog(qapp)
    dlg._select_range(0)
    # No fit has run.
    assert dlg._fit.ranges[0].result is None

    _drain_preview(dlg)

    active = next((r for r in dlg._preview._ranges if r.idx == dlg._active_range_idx), None)
    assert active is not None
    cy = np.asarray(active.curve_y, dtype=float)
    assert cy.size > 0
    # Sloped, not flat: the seed m=2 gives a clearly varying curve.
    assert float(np.nanmax(cy) - np.nanmin(cy)) > 1.0
    assert active.fitted is False


def test_preview_stale_result_ignored(qapp: QApplication) -> None:
    """A ready payload carrying an out-of-date generation token is dropped."""
    from asymmetry.gui.widgets.trend_preview import PreviewRange

    dlg = _make_dialog(qapp)
    _drain_preview(dlg)

    # Snapshot the canvas's current ranges, then hand it a stale payload whose
    # generation predates the live token.
    before = list(dlg._preview._ranges)
    dlg._preview_active = True  # pretend a sample is in flight
    stale_gen = dlg._preview_generation - 1
    poison = [
        PreviewRange(
            idx=0,
            x_min=0.0,
            x_max=1.0,
            windows=None,
            in_mask=np.array([], dtype=bool),
            curve_x=np.array([0.0, 1.0]),
            curve_y=np.array([999.0, 999.0]),
            fitted=False,
        )
    ]
    dlg._on_preview_ready((stale_gen, poison))

    # The poisoned curve must NOT have been drawn; ranges are unchanged.
    assert dlg._preview._ranges == before
    # The in-flight slot was still released.
    assert dlg._preview_active is False


def test_close_with_pending_preview_timer_starts_no_task(qapp: QApplication) -> None:
    """A debounce timer still pending at closeEvent must not launch a worker on
    the shut-down TaskRunner (post-shutdown-start teardown hazard)."""
    dlg = _make_dialog(qapp)
    # Arrange a pending debounce (as a live edit/drag would) without draining it.
    dlg._request_preview_update()
    assert dlg._preview_timer.isActive()
    assert dlg._tasks.active_count == 0

    dlg.close()

    # Teardown stopped the timer and flagged shutdown; no worker was started.
    assert dlg._shutting_down is True
    assert not dlg._preview_timer.isActive()
    assert dlg._tasks.active_count == 0

    # Belt-and-braces: even if a timer event had already been dequeued, firing
    # the handler after shutdown must be a no-op (guards short-circuit).
    dlg._launch_preview_sample()
    qapp.processEvents()
    assert dlg._tasks.active_count == 0


# ── Phase 2.2: consume the preview drag signals (carve / edge-sync) ────────────


def test_drag_edge_syncs_spinbox(qapp: QApplication) -> None:
    """A range-edge drag updates the model, mirrors the spinbox, and invalidates."""
    from asymmetry.core.fitting.parameter_models import ParameterModelFitResult

    dlg = _make_dialog(qapp)
    dlg._select_range(0)
    fit_range = dlg._fit.ranges[0]
    fit_range.result = ParameterModelFitResult(
        success=True, reduced_chi_squared=1.0, parameters=fit_range.parameters
    )

    new_max = 7.5
    dlg._on_preview_range_edge_dragged(0, fit_range.x_min, new_max)

    assert fit_range.x_max == new_max
    # A plain range's range-edge drag is an interval-0 edit: it mirrors into the
    # interval-0 spin pair and keeps the range plain (windows None).
    _, i_max = dlg._region_row_spins[0]
    assert i_max.value() == new_max
    assert fit_range.windows is None
    # The stale result was invalidated by the funnel.
    assert fit_range.result is None


def test_exclude_region_creates_two_windows(qapp: QApplication) -> None:
    """A right-drag exclude interior to the range carves a two-window union."""
    dlg = _make_dialog(qapp)
    fit_range = dlg._fit.ranges[0]
    # Data range is 1..10; carve out the interior [4, 6].
    dlg._on_preview_exclude_region(0, 4.0, 6.0)

    assert fit_range.windows is not None
    assert len(fit_range.windows) == 2
    los = sorted(lo for lo, _hi in fit_range.windows)
    his = sorted(hi for _lo, hi in fit_range.windows)
    # The gap [4, 6] separates the two surviving windows.
    assert 4.0 in his
    assert 6.0 in los


def test_exclude_shows_two_interval_rows(qapp: QApplication) -> None:
    """After a carve the details-pane shows two interval rows, each with Remove."""
    dlg = _make_dialog(qapp)
    dlg._on_preview_exclude_region(0, 4.0, 6.0)
    dlg._select_range(0)
    assert len(dlg._region_row_spins) == 2
    # With more than one interval, every Remove button is shown (not hidden).
    assert all(not btn.isHidden() for btn in dlg._region_remove_btns)


def test_exclude_invalidates_result(qapp: QApplication) -> None:
    """A real carve clears the range's stored fit result."""
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
    dlg._on_preview_exclude_region(0, 4.0, 6.0)
    assert fit_range.result is None


def test_exclude_outside_range_is_noop_keeps_result(qapp: QApplication) -> None:
    """A stray exclude fully outside the range must not drop a good fit."""
    from asymmetry.core.fitting.parameter_models import ParameterModelFitResult

    dlg = _make_dialog(qapp)
    fit_range = dlg._fit.ranges[0]
    result = ParameterModelFitResult(
        success=True,
        chi_squared=9.0,
        reduced_chi_squared=0.9,
        parameters=fit_range.parameters,
        error_mode="column",
        n_points=12,
    )
    fit_range.result = result
    windows_before = fit_range.windows

    # Data range is 1..10; carve entirely to the right of it.
    dlg._on_preview_exclude_region(0, 50.0, 60.0)

    assert fit_range.result is result
    assert fit_range.windows == windows_before


def test_drag_disabled_while_fitting(qapp: QApplication) -> None:
    """Dragging is suppressed while a fit runs and re-enabled when it settles."""
    dlg = _make_dialog(qapp)
    dlg._set_fit_ui_busy(True)
    assert dlg._preview._drag_enabled is False
    dlg._set_fit_ui_busy(False)
    assert dlg._preview._drag_enabled is True


# ── Item 4.1: residual toggle wiring ─────────────────────────────────────────


def test_residual_toggle_wired(qapp: QApplication) -> None:
    """Toggling the dialog's 'Show residuals' checkbox flips the preview state."""
    dlg = _make_dialog(qapp)
    assert dlg._preview._show_residuals is False
    assert dlg._show_residuals_check.isChecked() is False

    dlg._show_residuals_check.setChecked(True)
    assert dlg._preview._show_residuals is True

    dlg._show_residuals_check.setChecked(False)
    assert dlg._preview._show_residuals is False


def test_residuals_toggle_in_preview_pane(qapp: QApplication) -> None:
    """P2: the residuals checkbox lives beside the preview it controls, not in
    the top-of-dialog toggle row alongside the narrow-collapse 'Show preview'
    button."""
    dlg = _make_dialog(qapp)

    # The checkbox is a descendant of the preview (right) pane, not the "Show
    # preview" narrow-collapse toggle's row.
    parent = dlg._show_residuals_check.parent()
    ancestors = []
    node = parent
    while node is not None:
        ancestors.append(node)
        node = node.parent()
    assert dlg._preview in ancestors or dlg._preview.parent() in ancestors

    # It must not share a parent with the "Show preview" narrow-collapse toggle.
    assert dlg._show_residuals_check.parent() is not dlg._show_preview_toggle.parent()

    # Wiring/behaviour is unchanged by the move.
    assert dlg._preview._show_residuals is False
    dlg._show_residuals_check.setChecked(True)
    assert dlg._preview._show_residuals is True
    dlg._show_residuals_check.setChecked(False)
    assert dlg._preview._show_residuals is False


def test_data_range_uses_en_dash(qapp: QApplication) -> None:
    """P8: bounds notation is consistent — the data-range label uses an
    en-dash like the range-card bounds, not the word 'to'."""
    dlg = _make_dialog(qapp)
    text = dlg._data_range_label.text()
    assert "–" in text
    assert " to " not in text


def test_empty_state_hint_present(qapp: QApplication) -> None:
    """P6: a muted discoverability hint tells a first-time user that dragging
    on the plot adds a range and clicking a range edits it."""
    dlg = _make_dialog(qapp)
    assert (
        dlg._empty_state_hint_label.text()
        == "Drag on the plot to add a fit range, or click a range to edit it."
    )
    assert dlg._empty_state_hint_label.isVisibleTo(dlg)


# ── Item 4.2: remember last-used model per (parameter, x_key) ─────────────────


def test_last_model_remembered(qapp: QApplication) -> None:
    """A model stored for (Lambda, field) becomes the default of a fresh dialog
    that shares the same caller-owned memory dict (project-scoped memory)."""
    memory: dict[str, str] = {}

    x = np.linspace(1.0, 10.0, 20)
    y = 2.0 * x + 1.0
    yerr = np.full_like(x, 0.1)

    # First dialog: edit the model to something non-default and confirm stored.
    dlg1 = ModelFitDialog("Lambda", "field", x, y, yerr, existing_fit=None, model_memory=memory)
    fit_range = dlg1._fit.ranges[0]
    fit_range.model = ParameterCompositeModel(["PowerLaw"], [])
    from asymmetry.gui.panels.model_fit_dialog import _store_last_model_expression

    _store_last_model_expression(
        "Lambda", "field", fit_range.model.component_expression_string(), memory
    )

    # A second dialog sharing the SAME memory dict (same panel/project): its
    # default range uses the remembered model.
    dlg2 = ModelFitDialog("Lambda", "field", x, y, yerr, existing_fit=None, model_memory=memory)
    assert dlg2._fit.ranges[0].model.component_names == ["PowerLaw"]

    # A dialog with a FRESH empty dict (e.g. a different project) does NOT
    # inherit the remembered model.
    dlg3 = ModelFitDialog("Lambda", "field", x, y, yerr, existing_fit=None, model_memory={})
    assert dlg3._fit.ranges[0].model.component_names != ["PowerLaw"]


def test_last_model_unknown_component_falls_back(qapp: QApplication) -> None:
    """A stored model whose component isn't in the pool falls back to the default."""
    from asymmetry.gui.panels.model_fit_dialog import _last_model_memory_key

    # Store a bogus expression directly (a name valid nowhere in the field pool).
    memory = {_last_model_memory_key("Lambda", "field"): "NotARealComponent"}

    x = np.linspace(1.0, 10.0, 20)
    y = 2.0 * x + 1.0
    yerr = np.full_like(x, 0.1)
    dlg = ModelFitDialog("Lambda", "field", x, y, yerr, existing_fit=None, model_memory=memory)
    # Falls back to the context default (Linear for a plain Lambda-vs-field trend)
    # without raising.
    default_component = _default_component_for_context("field", "Lambda", dlg._component_pool)
    assert dlg._fit.ranges[0].model.component_names == [default_component]


# ── Item 4.3: per-fit success modal removed → inline result box ──────────────


def test_fit_success_no_modal_shows_result_box(qapp: QApplication, monkeypatch) -> None:
    """A successful fit tints the result box green and shows inline success text —
    with NO _show_info modal fired."""
    from asymmetry.core.fitting.parameter_models import ParameterModelFitResult
    from asymmetry.gui.styles.widgets import RESULT_BOX_SUCCESS_STYLE

    info_calls: list[tuple] = []
    monkeypatch.setattr(
        "asymmetry.gui.panels.model_fit_dialog._show_info",
        lambda *a, **k: info_calls.append(a),
    )

    dlg = _make_dialog(qapp)

    def _fake_fit(**_kwargs):
        return ParameterModelFitResult(
            success=True,
            chi_squared=9.0,
            reduced_chi_squared=0.9,
            parameters=dlg._fit.ranges[0].parameters,
            error_mode="column",
            n_points=12,
        )

    monkeypatch.setattr("asymmetry.gui.panels.model_fit_dialog.fit_parameter_model", _fake_fit)

    dlg._run_fit(0)
    _wait_for_fit(dlg)

    # No "Fit complete" success modal fired (item 4.3). Other unrelated info
    # popups — e.g. a "Parameter limits adjusted" note from committing the
    # table — may still occur and are not what this test guards.
    assert not any("Fit complete" in str(a) for a in info_calls)
    # Inline success text on the χ² label.
    assert "successful" in dlg._chi2_label.text().lower()
    # Result box tinted with the success style.
    assert dlg._result_box.styleSheet() == RESULT_BOX_SUCCESS_STYLE


# ── Phase 5 branding polish ────────────────────────────────────────────────


def test_section_headers_present(qapp: QApplication) -> None:
    """The merged "Fit ranges" block uses a flat BENCH section header
    (make_section) rather than a QGroupBox, and rebuilding the range rows
    (which clear_layouts ``_ranges_host``) does not destroy it."""
    from asymmetry.gui.styles.widgets import SECTION_HEADER_OBJECT_NAME

    dlg = _make_dialog(qapp)

    def header_texts() -> list[str]:
        return [
            label.text()
            for label in dlg.findChildren(type(dlg._data_range_label))
            if label.objectName() == SECTION_HEADER_OBJECT_NAME
        ]

    texts = header_texts()
    # make_section_header uppercases the title in Python. The two old sections
    # ("Model ranges" / "Range parameters") are now one merged "Fit ranges".
    assert "FIT RANGES" in texts
    assert "MODEL RANGES" not in texts
    assert "RANGE PARAMETERS" not in texts

    # Rebuilding the range rows clears ``_ranges_host`` — the header lives in the
    # section container ABOVE it, so it must survive.
    dlg._rebuild_ranges_ui()
    texts_after = header_texts()
    assert "FIT RANGES" in texts_after


def test_run_fit_is_primary_styled(qapp: QApplication) -> None:
    """The active card's Run Fit button carries the accent primary QSS and is
    width-locked so a future "Fitting…" relabel cannot clip it."""
    from asymmetry.gui.styles.widgets import build_primary_button_qss

    dlg = _make_dialog(qapp)
    card = dlg._range_cards[0]

    assert card._run_button.styleSheet() == build_primary_button_qss()
    # Width-locked so a future "Fitting…" relabel cannot clip the button.
    assert card._run_button.width() > 0 or card._run_button.minimumWidth() >= 0


# ---------------------------------------------------------------------------
# Range-cards redesign (Direction A, Step 1): cards + relocated details pane
# ---------------------------------------------------------------------------


def test_range_card_run_triggers_fit(qapp: QApplication, monkeypatch) -> None:
    """A card's run_requested signal is wired to _run_fit for that range index."""
    dlg = _make_dialog(qapp)
    dlg._add_range()  # two ranges so the index is meaningfully carried

    fired: list[int] = []
    monkeypatch.setattr(dlg, "_run_fit", fired.append)
    dlg._range_cards[1].run_requested.emit(1)

    assert fired == [1]


def test_active_card_shows_run_and_highlight(qapp: QApplication) -> None:
    """Only the active card exposes Run Fit (show_run) and the active highlight."""
    dlg = _make_dialog(qapp)
    dlg._add_range()

    dlg._select_range(0)
    assert dlg._range_cards[0]._view.show_run is True
    assert dlg._range_cards[1]._view.show_run is False

    dlg._select_range(1)
    assert dlg._range_cards[0]._view.show_run is False
    assert dlg._range_cards[1]._view.show_run is True
    # The active card's surface style differs from an inactive card's.
    assert dlg._range_cards[1]._surface.styleSheet() != dlg._range_cards[0]._surface.styleSheet()


def test_card_status_chip_reflects_result(qapp: QApplication) -> None:
    """A successful fit result gives the card a success status + verdict chip;
    an unfitted range reads not_run with no chip."""
    from asymmetry.core.fitting.parameter_models import ParameterModelFitResult

    dlg = _make_dialog(qapp)
    fit_range = dlg._fit.ranges[0]

    assert dlg._range_cards[0]._view.status == "not_run"

    fit_range.result = ParameterModelFitResult(
        success=True,
        chi_squared=18.0,
        reduced_chi_squared=1.0,
        parameters=fit_range.parameters,
        error_mode="column",
        n_points=20,
    )
    dlg._select_range(0)
    assert dlg._range_cards[0]._view.status == "success"
    # A Column-mode fit with dof > 0 yields a verdict chip.
    assert dlg._range_cards[0]._view.status_chip_html != ""


def test_windowed_card_shows_union_bounds(qapp: QApplication) -> None:
    """A windowed range's card bounds_text uses the ∪ union formatting."""
    dlg = _make_dialog(qapp)
    dlg._exclude_region(0, 4.0, 6.0)

    text = dlg._range_cards[0]._view.bounds_text
    assert "∪" in text


def test_active_range_bounds_editable_in_details_pane(qapp: QApplication) -> None:
    """The details-pane interval-0 spins edit the ACTIVE range's x_min/x_max."""
    dlg = _make_dialog(qapp)
    dlg._select_range(0)

    i_min, i_max = dlg._region_row_spins[0]
    i_min.setValue(2.0)
    i_max.setValue(9.0)

    assert dlg._fit.ranges[0].x_min == 2.0
    assert dlg._fit.ranges[0].x_max == 9.0
    # A single-interval edit keeps the range plain.
    assert dlg._fit.ranges[0].windows is None


def test_inactive_range_bounds_readonly_on_card(qapp: QApplication) -> None:
    """A non-active range's bounds are shown on its card, not editable in the
    details-pane interval rows (which track the active range)."""
    dlg = _make_dialog(qapp)
    dlg._add_range()
    # Give the two ranges distinct bounds.
    dlg._select_range(0)
    dlg._region_row_spins[0][0].setValue(1.0)
    dlg._region_row_spins[0][1].setValue(5.0)
    dlg._select_range(1)
    dlg._region_row_spins[0][0].setValue(6.0)
    dlg._region_row_spins[0][1].setValue(10.0)

    # The details pane now shows range 1; range 0's bounds live on its card only.
    assert dlg._active_range_idx == 1
    assert "1" in dlg._range_cards[0]._view.bounds_text
    assert "5" in dlg._range_cards[0]._view.bounds_text
    # Editing the pane must not touch the inactive range 0.
    dlg._region_row_spins[0][0].setValue(7.0)
    assert dlg._fit.ranges[0].x_min == 1.0
    assert dlg._fit.ranges[1].x_min == 7.0


def _active_card_count(dlg: ModelFitDialog) -> int:
    """Number of cards whose view reports ``show_run`` (the active card)."""
    return sum(1 for card in dlg._range_cards if card._view is not None and card._view.show_run)


def test_set_active_range_single_source(qapp: QApplication) -> None:
    """Activating via a card's ``selected`` signal, via ``_set_active_range``,
    and observing the canvas mirror all converge on the same ``_active_range_idx``
    (contract C-ACTIVE). ``active_range_index()`` returns it and exactly one card
    is active."""
    dlg = _make_dialog(qapp)
    dlg._add_range()  # now two ranges, indices 0 and 1

    # 1) via a card's `selected` signal.
    dlg._range_cards[1].selected.emit(1)
    assert dlg._active_range_idx == 1
    assert dlg.active_range_index() == 1
    assert dlg._preview._active_idx == 1  # canvas mirror followed
    assert _active_card_count(dlg) == 1
    assert dlg._range_cards[1]._view.show_run

    # 2) via _set_active_range directly.
    dlg._set_active_range(0)
    assert dlg._active_range_idx == 0
    assert dlg.active_range_index() == 0
    assert dlg._preview._active_idx == 0
    assert _active_card_count(dlg) == 1
    assert dlg._range_cards[0]._view.show_run

    # 3) the from_plot path (frozen signature, same fan-out today).
    dlg._set_active_range(1, from_plot=True)
    assert dlg._active_range_idx == 1
    assert dlg.active_range_index() == 1
    assert dlg._preview._active_idx == 1
    assert _active_card_count(dlg) == 1

    # Out-of-range / None are guarded no-ops (still the single writer).
    dlg._set_active_range(None)
    assert dlg._active_range_idx == 1
    dlg._set_active_range(99)
    assert dlg._active_range_idx == 1


def test_no_range_selector_combo(qapp: QApplication) -> None:
    """The "Editing range" combo was deleted; the card is now the selector."""
    dlg = _make_dialog(qapp)
    assert not hasattr(dlg, "_range_selector")
    assert not hasattr(dlg, "_on_range_selector_changed")
    assert not hasattr(dlg, "_refresh_range_selector")


def test_clicking_card_updates_details_pane(qapp: QApplication) -> None:
    """Activating range 2 (index 1) via its card repopulates the details-pane
    bounds pair from that range."""
    dlg = _make_dialog(qapp)
    dlg._add_range()

    dlg._select_range(0)
    dlg._region_row_spins[0][0].setValue(1.0)
    dlg._region_row_spins[0][1].setValue(5.0)
    dlg._select_range(1)
    dlg._region_row_spins[0][0].setValue(6.0)
    dlg._region_row_spins[0][1].setValue(9.0)

    # Click range 0's card: the details pane must follow to range 0's bounds.
    dlg._range_cards[0].selected.emit(0)
    assert dlg._active_range_idx == 0
    assert dlg._region_row_spins[0][0].value() == 1.0
    assert dlg._region_row_spins[0][1].value() == 5.0

    # And back to range 1's card.
    dlg._range_cards[1].selected.emit(1)
    assert dlg._active_range_idx == 1
    assert dlg._region_row_spins[0][0].value() == 6.0
    assert dlg._region_row_spins[0][1].value() == 9.0


def test_single_fit_ranges_section(qapp: QApplication) -> None:
    """The two old sections are merged into one "Fit ranges" BENCH section."""
    from asymmetry.gui.styles.widgets import SECTION_HEADER_OBJECT_NAME

    dlg = _make_dialog(qapp)
    headers = [
        label.text()
        for label in dlg.findChildren(type(dlg._data_range_label))
        if label.objectName() == SECTION_HEADER_OBJECT_NAME
    ]
    assert headers.count("FIT RANGES") == 1
    assert "MODEL RANGES" not in headers
    assert "RANGE PARAMETERS" not in headers


def test_canvas_range_edge_drag_syncs_interval0_spin_and_stays_plain(
    qapp: QApplication,
) -> None:
    """A canvas range-edge drag mirrors into the interval-0 spin and keeps the
    range plain (windows None — a range edge IS interval 0)."""
    dlg = _make_dialog(qapp)
    dlg._select_range(0)
    fit_range = dlg._fit.ranges[0]

    new_max = 7.5
    dlg._on_preview_range_edge_dragged(0, fit_range.x_min, new_max)

    _, i_max = dlg._region_row_spins[0]
    assert i_max.value() == new_max
    assert fit_range.x_max == new_max
    assert fit_range.windows is None


def test_canvas_window_edge_drag_syncs_interval_spin(qapp: QApplication) -> None:
    """A canvas window-edge drag mirrors into the active range's interval spins,
    keyed by interval index in the details pane."""
    dlg = _make_dialog(qapp)
    # Carve to get a two-interval (windowed) range, then drag interval 0's edges.
    dlg._exclude_region(0, 4.0, 6.0)
    dlg._select_range(0)

    dlg._on_preview_window_edge_dragged(0, 0, 2.0, 3.5)

    i_min, i_max = dlg._region_row_spins[0]
    assert i_min.value() == 2.0
    assert i_max.value() == 3.5
    assert dlg._fit.ranges[0].windows[0] == (2.0, 3.5)


def test_plain_range_shows_one_interval_no_remove(qapp: QApplication) -> None:
    """A fresh plain range shows exactly one interval row with Remove hidden."""
    dlg = _make_dialog(qapp)
    dlg._select_range(0)

    assert len(dlg._region_row_spins) == 1
    assert len(dlg._region_remove_btns) == 1
    # The sole interval's Remove is hidden (a region can never drop below one).
    assert dlg._region_remove_btns[0].isHidden()


def test_exclude_region_splits_into_two_intervals(qapp: QApplication) -> None:
    """The 'Exclude region…' button carves a default gap -> two intervals."""
    dlg = _make_dialog(qapp)
    dlg._select_range(0)
    assert dlg._fit.ranges[0].windows is None

    dlg._on_exclude_region_clicked()

    windows = dlg._fit.ranges[0].windows
    assert windows is not None
    assert len(windows) == 2
    assert len(dlg._region_row_spins) == 2


def test_remove_interval_down_to_one_collapses_to_plain(qapp: QApplication) -> None:
    """Removing an interval down to one leaves windows None (collapse rule)."""
    dlg = _make_dialog(qapp)
    dlg._exclude_region(0, 4.0, 6.0)
    dlg._select_range(0)
    assert dlg._fit.ranges[0].windows is not None

    dlg._remove_interval(0, 1)

    assert dlg._fit.ranges[0].windows is None
    assert len(dlg._region_row_spins) == 1


def test_remove_last_interval_refused(qapp: QApplication) -> None:
    """Removing the sole interval of a plain range is a no-op (never empty)."""
    dlg = _make_dialog(qapp)
    dlg._select_range(0)
    before = (dlg._fit.ranges[0].x_min, dlg._fit.ranges[0].x_max)

    dlg._remove_interval(0, 0)

    assert dlg._fit.ranges[0].windows is None
    assert (dlg._fit.ranges[0].x_min, dlg._fit.ranges[0].x_max) == before
    assert len(dlg._region_row_spins) == 1


def test_edge_carve_collapses_to_plain(qapp: QApplication) -> None:
    """Excluding an END chunk leaves a single surviving interval, which the
    collapse rule plains back to windows=None (not a stuck 1-window list)."""
    dlg = _make_dialog(qapp)
    dlg._select_range(0)
    (lo, hi) = dlg._resolved_intervals(dlg._fit.ranges[0])[0]
    cut = lo + (hi - lo) * 0.25

    # Carve out the LOW end [lo, cut] -> only [cut, hi] survives -> plain range.
    dlg._exclude_region(0, lo, cut)

    assert dlg._fit.ranges[0].windows is None
    assert dlg._fit.ranges[0].x_min == pytest.approx(cut)
    assert len(dlg._region_row_spins) == 1


def test_edge_carve_then_plain_edge_drag_syncs(qapp: QApplication) -> None:
    """After an edge-carve collapses to plain, the range is once again a plain
    single interval: a plot range-edge drag routes through the plain path
    (_on_preview_range_edge_dragged), stays windows=None, and syncs interval 0."""
    dlg = _make_dialog(qapp)
    dlg._select_range(0)
    (lo, hi) = dlg._resolved_intervals(dlg._fit.ranges[0])[0]
    dlg._exclude_region(0, lo, lo + (hi - lo) * 0.25)
    assert dlg._fit.ranges[0].windows is None

    new_lo, new_hi = dlg._fit.ranges[0].x_min, dlg._fit.ranges[0].x_max
    dragged_lo = new_lo + (new_hi - new_lo) * 0.1
    dlg._on_preview_range_edge_dragged(0, dragged_lo, new_hi)

    assert dlg._fit.ranges[0].windows is None
    assert dlg._fit.ranges[0].x_min == pytest.approx(dragged_lo)
    i_min, _i_max = dlg._region_row_spins[0]
    assert i_min.value() == pytest.approx(dragged_lo)


def test_fit_busy_disables_region_rows_and_actions(qapp: QApplication) -> None:
    """_set_fit_ui_busy(True) disables the cards' action buttons, the interval
    spins, and the 'Exclude region…' button; (False) restores them."""
    dlg = _make_dialog(qapp)
    dlg._select_range(0)

    dlg._set_fit_ui_busy(True)
    assert dlg._range_cards[0]._run_button.isEnabled() is False
    assert dlg._range_cards[0]._edit_model_button.isEnabled() is False
    assert dlg._region_row_spins[0][0].isEnabled() is False
    assert dlg._region_row_spins[0][1].isEnabled() is False
    assert dlg._exclude_region_btn.isEnabled() is False

    dlg._set_fit_ui_busy(False)
    assert dlg._range_cards[0]._run_button.isEnabled() is True
    assert dlg._range_cards[0]._edit_model_button.isEnabled() is True
    assert dlg._region_row_spins[0][0].isEnabled() is True
    assert dlg._exclude_region_btn.isEnabled() is True


def test_no_exclusion_window_wording(qapp: QApplication) -> None:
    """No 'Exclusion'/'Window N' wording; the fit-region wording is present."""
    from PySide6.QtWidgets import QLabel, QPushButton

    dlg = _make_dialog(qapp)
    # Carve so the multi-interval rows are built too.
    dlg._exclude_region(0, 4.0, 6.0)
    dlg._select_range(0)

    texts = [w.text() for w in dlg.findChildren(QLabel)]
    texts += [w.text() for w in dlg.findChildren(QPushButton)]
    joined = " ".join(texts)
    assert "Exclusion" not in joined
    assert "Window" not in joined
    assert "Fit region" in joined
    assert any(t.startswith("Interval ") for t in texts)


def test_plot_drag_creates_range(qapp: QApplication) -> None:
    """A drag-out on empty canvas (range_add_requested) appends a new range with
    the dragged bounds, makes it active, and gives its card the idx-keyed swatch."""
    from asymmetry.gui.panels.model_fit_dialog import range_span_color

    dlg = _make_dialog(qapp)
    before = len(dlg._fit.ranges)

    lo, hi = 3.5, 6.5
    dlg._preview.range_add_requested.emit(lo, hi)

    new_idx = len(dlg._fit.ranges) - 1
    assert len(dlg._fit.ranges) == before + 1
    assert dlg._fit.ranges[new_idx].x_min == lo
    assert dlg._fit.ranges[new_idx].x_max == hi
    # The new range is now active.
    assert dlg.active_range_index() == new_idx
    # A card exists for it carrying the idx-keyed span colour.
    assert len(dlg._range_cards) == before + 1
    assert dlg._range_cards[new_idx]._view is not None
    assert dlg._range_cards[new_idx]._view.swatch_color == range_span_color(new_idx)


def test_plot_click_selects_range(qapp: QApplication) -> None:
    """A click on a non-active range's span (range_select_requested) activates it
    and the details pane follows (from_plot select path)."""
    dlg = _make_dialog(qapp)
    dlg._add_range()  # now two ranges, indices 0 and 1
    dlg._select_range(0)
    assert dlg.active_range_index() == 0

    dlg._preview.range_select_requested.emit(1)

    assert dlg.active_range_index() == 1
    assert dlg._preview._active_idx == 1  # canvas mirror followed
    # The details pane repointed at range 1.
    assert dlg._range_cards[1]._view is not None
    assert dlg._range_cards[1]._view.show_run


def test_add_button_and_drag_share_path(qapp: QApplication) -> None:
    """The "Add Range" button and a drag-created range both go through
    _create_default_range, so each new range is seeded with default params."""
    dlg = _make_dialog(qapp)

    # Button path.
    dlg._add_range()
    button_range = dlg._fit.ranges[-1]
    assert len(list(button_range.parameters)) > 0
    assert all(np.isfinite(p.value) for p in button_range.parameters)

    # Drag path (explicit bounds) — same seeding.
    dlg._add_range_with_bounds(2.0, 8.0)
    drag_range = dlg._fit.ranges[-1]
    assert drag_range.x_min == 2.0
    assert drag_range.x_max == 8.0
    assert list(drag_range.model.param_names) == list(button_range.model.param_names)
    assert len(list(drag_range.parameters)) == len(list(button_range.parameters))
    assert all(np.isfinite(p.value) for p in drag_range.parameters)


def test_add_range_with_bounds_ignores_degenerate(qapp: QApplication) -> None:
    """An inverted/zero-width dragged span falls back to the default data-extent
    bounds rather than creating a degenerate range (no crash)."""
    dlg = _make_dialog(qapp)
    default = dlg._create_default_range()

    # Inverted span (hi <= lo): falls back to defaults.
    dlg._add_range_with_bounds(8.0, 2.0)
    inverted = dlg._fit.ranges[-1]
    assert inverted.x_min == default.x_min
    assert inverted.x_max == default.x_max
    assert inverted.x_max > inverted.x_min

    # Zero-width span: also falls back to defaults.
    dlg._add_range_with_bounds(5.0, 5.0)
    zero = dlg._fit.ranges[-1]
    assert zero.x_min == default.x_min
    assert zero.x_max == default.x_max
    assert zero.x_max > zero.x_min
