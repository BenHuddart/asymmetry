"""GUI wiring for the rotating-reference-frame single fit (item 2).

The single composite fit auto-couples to the plot's RRF display via an injected
ν₀ provider: when active it fits raw data with the frequency offset and reports
the fitted frequency back in the lab frame, annotated "frame: ν_RRF".
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.gui.panels.fit_panel import SingleFitTab, _shift_rrf_parameters

pytestmark = [pytest.mark.gui]

NU_LAB = 30.0
NU_FRAME = 29.2


def _oscillatory_dataset() -> MuonDataset:
    t = np.arange(0.0, 8.0, 0.004)
    rng = np.random.default_rng(3)
    y = 20.0 * np.exp(-0.4 * t) * np.cos(2.0 * np.pi * NU_LAB * t) + rng.normal(0.0, 0.4, t.size)
    return MuonDataset(time=t, asymmetry=y, error=np.full_like(t, 0.4), metadata={"run_number": 7})


def _set_oscillatory_model(tab: SingleFitTab) -> None:
    tab._set_composite_model(CompositeModel.from_expression("Oscillatory * Exponential"))


def test_shift_rrf_parameters_round_trips_value_and_bounds():
    params = ParameterSet([Parameter("frequency", 30.0, min=0.0, max=5000.0)])
    offsets = {"frequency": 29.2}
    down = _shift_rrf_parameters(params, offsets, sign=-1)
    assert down["frequency"].value == pytest.approx(0.8)
    assert down["frequency"].min == pytest.approx(-29.2)
    up = _shift_rrf_parameters(down, offsets, sign=+1)
    assert up["frequency"].value == pytest.approx(30.0)
    assert up["frequency"].min == pytest.approx(0.0)
    assert up["frequency"].max == pytest.approx(5000.0)


def test_rrf_fit_reports_lab_frame_and_annotates(qapp: QApplication) -> None:
    tab = SingleFitTab()
    tab.set_dataset(_oscillatory_dataset())
    _set_oscillatory_model(tab)
    # Seed the frequency near the lab value (the user works in the lab frame).
    for row in range(tab._param_table.rowCount()):
        name_item = tab._param_table.item(row, 0)
        name = name_item.data(Qt.ItemDataRole.UserRole) if name_item else None
        if not isinstance(name, str):
            name = name_item.text() if name_item else ""
        if name == "frequency":
            tab._param_table.item(row, 1).setText(f"{NU_LAB + 0.1:.4f}")

    tab.set_rrf_frequency_provider(lambda: NU_FRAME)
    tab._run_fit()
    assert tab.wait_for_fit()

    # Fitted frequency is reported in the lab frame (≈ true lab value), NOT δν.
    fitted = {p.name: p.value for p in tab._last_fit_result.parameters}
    assert fitted["frequency"] == pytest.approx(NU_LAB, abs=0.05)
    # The result label carries the rotating-frame annotation.
    assert "frame: ν_RRF" in tab._result_label.text()
    assert f"{NU_FRAME:.4f}" in tab._result_label.text()


def test_rrf_fit_inactive_when_provider_returns_none(qapp: QApplication) -> None:
    tab = SingleFitTab()
    tab.set_dataset(_oscillatory_dataset())
    _set_oscillatory_model(tab)
    for row in range(tab._param_table.rowCount()):
        name_item = tab._param_table.item(row, 0)
        name = name_item.data(Qt.ItemDataRole.UserRole) if name_item else None
        if not isinstance(name, str):
            name = name_item.text() if name_item else ""
        if name == "frequency":
            tab._param_table.item(row, 1).setText(f"{NU_LAB + 0.1:.4f}")
    # Default provider returns None → ordinary lab-frame fit, no annotation.
    tab._run_fit()
    assert tab.wait_for_fit()
    fitted = {p.name: p.value for p in tab._last_fit_result.parameters}
    assert fitted["frequency"] == pytest.approx(NU_LAB, abs=0.05)
    assert "frame: ν_RRF" not in tab._result_label.text()


def test_rrf_fit_refuses_unsupported_oscillating_model(qapp: QApplication) -> None:
    tab = SingleFitTab()
    tab.set_dataset(_oscillatory_dataset())
    tab._set_composite_model(CompositeModel.from_expression("MuoniumTF + Exponential"))
    tab.set_rrf_frequency_provider(lambda: NU_FRAME)
    tab._run_fit()
    # Refused before launching a worker (no fit to wait for); the message names
    # the rotating frame so the user knows to turn it off.
    assert "rotating frame" in tab._result_label.text().lower()
