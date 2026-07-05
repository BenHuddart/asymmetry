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
    assert dlg._range_widgets[0].x_max.value() == new_max
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


def test_exclude_disables_plain_spinboxes(qapp: QApplication) -> None:
    """After a carve the plain x_min/x_max spins are disabled (windows override)."""
    dlg = _make_dialog(qapp)
    dlg._on_preview_exclude_region(0, 4.0, 6.0)
    assert dlg._range_widgets[0].x_min.isEnabled() is False
    assert dlg._range_widgets[0].x_max.isEnabled() is False


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
