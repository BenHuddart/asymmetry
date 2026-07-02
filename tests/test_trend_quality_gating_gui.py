"""Trend quality gating — GUI surfacing + click-to-exclude (Phase 2.2 / 2.3).

Covers the trend panel's new χ²ᵣ / "(fit)" / Trend columns, the flagged-marker
overlay + provenance line, the include-toggle signal, and the headline
acceptance criterion: excluding the two garbage EuO members moves the
OrderParameter Tc back toward the ground truth (~69 K).
"""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QEventLoop, Qt, QTimer

from asymmetry.core.fitting.parameter_models import (
    ModelFitRange,
    ParameterCompositeModel,
    ParameterModelFit,
    fit_parameter_model,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel

_TC_TRUE = 69.0
_PARAM = "frequency"


def _order_curve(temp: float) -> float:
    base = max(1.0 - (temp / _TC_TRUE) ** 3.0, 0.0)
    return 30.0 * base**0.5 if base > 0 else 0.0


def _row(run: int, temp: float, value: float, *, err=0.3, include=True, flags=None) -> dict:
    return {
        "run_number": run,
        "run_label": str(run),
        "field": 0.0,
        "temperature": temp,
        "values": {_PARAM: value},
        "errors": {_PARAM: err},
        "model_name": "OrderParameter",
        "reduced_chi_squared": 1.0,
        "batch_id": "b1",
        "trend_member_key": run,
        "include_in_trend": include,
        "quality_flags": list(flags or []),
    }


def _good_rows() -> list[dict]:
    return [
        _row(2900 + i, temp, _order_curve(temp))
        for i, temp in enumerate(np.linspace(5.0, 64.0, 12))
    ]


def _garbage_rows() -> list[dict]:
    # Two members just above the true Tc that still carry a small signal — the
    # EuO run-2949/2947 pathology. A least-squares fit is forced to raise Tc to
    # keep the curve nonzero there, biasing it high (~73 K). Flagged so the UI
    # rings them; removing them lets Tc fall back to the ~69 K ground truth.
    return [
        _row(2949, 70.0, 5.0, err=0.4, flags=["spurious_reseeded"]),
        _row(2947, 71.0, 5.0, err=0.4, flags=["large_rel_err"]),
    ]


def _select_param(panel: FitParametersPanel) -> None:
    # Select the frequency parameter as the trend Y so the plot/model-fit paths
    # operate on it.
    for r in range(panel._y_selector_table.rowCount()):
        item = panel._y_selector_table.item(r, 0)
        if item is not None and item.data(Qt.ItemDataRole.UserRole) == _PARAM:
            panel._y_selector_table.selectRow(r)
            return


def _load(panel: FitParametersPanel, rows: list[dict]) -> None:
    panel.load_representation_series([("b1", "EuO", rows)])
    _select_param(panel)


# ── table columns (F14) ───────────────────────────────────────────────────────


def test_table_has_fit_suffix_chi2_and_trend_columns(qapp):
    panel = FitParametersPanel()
    _load(panel, _good_rows())
    panel._refresh_table()
    headers = [
        panel._table.horizontalHeaderItem(c).text() for c in range(panel._table.columnCount())
    ]
    assert any(h.endswith("(fit)") for h in headers)  # fitted param disambiguated
    assert "χ²ᵣ" in headers
    assert "Trend" in headers


def test_flagged_row_marks_chi2_with_tooltip(qapp):
    panel = FitParametersPanel()
    _load(panel, [*_good_rows(), *_garbage_rows()])
    panel._refresh_table()
    headers = [
        panel._table.horizontalHeaderItem(c).text() for c in range(panel._table.columnCount())
    ]
    chi2_col = headers.index("χ²ᵣ")
    tooltips = [panel._table.item(r, chi2_col).toolTip() for r in range(panel._table.rowCount())]
    assert any("Quality flags" in t for t in tooltips)


def test_trend_checkbox_toggle_emits_inclusion_change(qapp):
    panel = FitParametersPanel()
    _load(panel, _good_rows())
    panel._refresh_table()
    headers = [
        panel._table.horizontalHeaderItem(c).text() for c in range(panel._table.columnCount())
    ]
    include_col = headers.index("Trend")

    seen: list[tuple] = []
    panel.member_trend_inclusion_changed.connect(lambda *a: seen.append(a))
    item = panel._table.item(0, include_col)
    item.setCheckState(Qt.CheckState.Unchecked)
    assert seen and seen[-1][2] is False


# ── provenance line + included filtering (F5) ─────────────────────────────────


def test_provenance_line_reports_excluded_members(qapp):
    panel = FitParametersPanel()
    rows = _good_rows()
    rows[0]["include_in_trend"] = False
    _load(panel, rows)
    panel._refresh_plot()
    # The top-level panel is never shown in the test, so check the explicit
    # hidden flag (set by _update_trend_provenance) rather than isVisible().
    assert not panel._trend_provenance_label.isHidden()
    assert "excluded" in panel._trend_provenance_label.text()


def test_included_rows_drops_excluded(qapp):
    panel = FitParametersPanel()
    rows = _good_rows()
    rows[0]["include_in_trend"] = False
    _load(panel, rows)
    included = panel._included_trend_rows(panel._effective_x_key())
    assert all(r.include_in_trend for r in included)
    assert len(included) == len(rows) - 1


# ── acceptance: excluding garbage members moves Tc toward 69 K ────────────────


def _seed_order_parameter_fit(panel: FitParametersPanel) -> float:
    """Fit OrderParameter over the panel's *current* rows and store it."""
    x_key = panel._effective_x_key()
    rows = panel._included_trend_rows(x_key)
    x = np.array([panel._x_value(r, x_key) for r in rows])
    y = np.array([r.values[_PARAM] for r in rows])
    yerr = np.array([r.errors[_PARAM] for r in rows])
    model = ParameterCompositeModel(["OrderParameter"])
    params = ParameterSet(
        [
            Parameter("y0", value=30.0, min=0.0, max=100.0),
            Parameter("Tc", value=70.0, min=10.0, max=120.0),
            Parameter("beta", value=0.5, min=0.05, max=2.0),
            Parameter("alpha", value=3.0, min=0.5, max=6.0),
        ]
    )
    result = fit_parameter_model(x, y, yerr, model, params)
    fit = ParameterModelFit(
        parameter_name=_PARAM,
        x_key=x_key,
        ranges=[
            ModelFitRange(x_min=None, x_max=None, model=model, parameters=params, result=result)
        ],
    )
    panel._model_fits[_PARAM] = fit
    return float(result.parameters["Tc"].value)


def _wait_for_refit(panel: FitParametersPanel) -> None:
    """Pump a nested event loop until the off-thread re-fit worker finishes.

    refit_active_model_fits dispatches through TaskRunner (never the GUI
    thread — see its docstring); the worker→main-thread queued signal needs
    the event loop actually entered, not just polled via processEvents().
    """
    loop = QEventLoop()
    check = QTimer()
    check.timeout.connect(lambda: loop.quit() if not panel._refit_in_progress else None)
    check.start(20)
    QTimer.singleShot(10000, loop.quit)
    loop.exec()
    check.stop()


def test_excluding_garbage_members_moves_tc_toward_ground_truth(qapp):
    panel = FitParametersPanel()
    _load(panel, [*_good_rows(), *_garbage_rows()])

    tc_with_garbage = _seed_order_parameter_fit(panel)
    # The two above-Tc members drag the fitted Tc above the true 69 K.
    assert tc_with_garbage > 71.0

    # Exclude the garbage members (as the click-to-exclude / checkbox would),
    # then re-solve the attached model fit (off the GUI thread).
    for row in panel._rows:
        if row.run_number in (2949, 2947):
            row.include_in_trend = False
    panel.refit_active_model_fits()
    _wait_for_refit(panel)

    tc_excluded = float(panel._model_fits[_PARAM].ranges[0].result.parameters["Tc"].value)
    assert abs(tc_excluded - _TC_TRUE) < abs(tc_with_garbage - _TC_TRUE)
    assert tc_excluded == pytest.approx(_TC_TRUE, abs=1.5)
