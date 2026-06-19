"""Joint K(θ) fit wired into the trend panel (Phase 6, GUI)."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from asymmetry.core.fitting.component_tracking import CrossingEvent
from asymmetry.core.fitting.knight_shift import KnightShiftConfig, KnightShiftUnit
from asymmetry.core.fitting.parameter_models import PARAMETER_MODEL_COMPONENTS
from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def _axial_field(angle, k_iso, k_ax, b_ext=7000.0):
    """A local field giving an axial K(θ) under the applied-field reference."""
    fn = PARAMETER_MODEL_COMPONENTS["KnightAnisotropy"].function
    k = fn(np.asarray(angle, dtype=float), K_iso=k_iso, K_ax=k_ax)  # dimensionless
    return b_ext * (1.0 + k)


def _setup_panel(qapp, *, swap_past_crossing: bool) -> FitParametersPanel:
    panel = FitParametersPanel()
    panel.set_angle_x_field(("Angle (°)", "angle"))
    angles = np.linspace(0.0, 90.0, 13)
    b_ext = 7000.0
    a = _axial_field(angles, 0.02, 0.03, b_ext)  # two crossing K(θ) curves
    b = _axial_field(angles, 0.02, -0.01, b_ext)
    past = angles > 54.7356103
    if swap_past_crossing:
        f1 = np.where(past, b, a)
        f2 = np.where(past, a, b)
    else:
        f1, f2 = a, b
    rows = []
    for i, ang in enumerate(angles):
        rows.append(
            {
                "run_number": i + 1,
                "run_label": str(i + 1),
                "field": b_ext,
                "temperature": 10.0,
                "values": {"field_1": float(f1[i]), "field_2": float(f2[i])},
                "errors": {"field_1": 0.2, "field_2": 0.2},
                "custom_values": {"angle": str(ang)},
            }
        )
    panel.load_representation_series([("b", "S", rows)])
    panel._x_combo.setCurrentIndex(panel._x_combo.findData("angle"))
    panel.set_knight_shift_config(KnightShiftConfig(enabled=True, unit=KnightShiftUnit.PPM))
    return panel


def test_joint_fit_button_gating(qapp):
    panel = _setup_panel(qapp, swap_past_crossing=False)
    # Angle axis is active and both K traces are auto-selected after enabling.
    panel._update_joint_fit_button()
    assert panel._joint_knight_btn.isEnabled() == (len(panel._selected_knight_traces()) >= 2)
    # Switching the x-axis away from Angle disables it.
    panel._x_combo.setCurrentIndex(panel._x_combo.findText("Run"))
    panel._update_joint_fit_button()
    assert not panel._joint_knight_btn.isEnabled()


def test_joint_fit_reorders_traces_in_place(qapp):
    panel = _setup_panel(qapp, swap_past_crossing=True)
    traces = sorted(panel._knight_shift_names)
    assert len(traces) == 2

    panel._run_joint_knight_fit(traces, "KnightAnisotropy", 25)

    # No duplicate K⟨…⟩ track columns are created — the existing K traces are reused.
    assert not any(name.startswith("K⟨") for name in panel._rows[0].values)
    # Per-curve overlays are stored on the existing K trace names.
    assert all(t in panel._model_fits for t in traces)
    assert all(t in panel._display_y_parameters() for t in traces)
    # The joint-fit state is recorded; the realignment found the crossing so the
    # assignment-derived markers are enabled and populated.
    assert panel._joint_fit is not None
    assert panel._DRAW_CROSSING_MARKERS is True
    assert panel.knight_shift_crossings()


def test_joint_reorder_survives_refresh(qapp):
    panel = _setup_panel(qapp, swap_past_crossing=True)
    traces = sorted(panel._knight_shift_names)
    panel._run_joint_knight_fit(traces, "KnightAnisotropy", 25)
    before = {t: [r.values.get(t) for r in panel._rows] for t in traces}

    # A pull refresh re-derives K from the raw field values via this chokepoint;
    # the joint reorder must be re-applied so the realignment persists.
    panel._apply_knight_shift_to_rows(panel._rows)

    after = {t: [r.values.get(t) for r in panel._rows] for t in traces}
    assert after == before


def test_reconfiguring_conversion_clears_joint_fit(qapp):
    panel = _setup_panel(qapp, swap_past_crossing=True)
    panel._run_joint_knight_fit(sorted(panel._knight_shift_names), "KnightAnisotropy", 25)
    assert panel._joint_fit is not None

    # Re-running the conversion regenerates the raw K traces and drops the joint fit.
    panel.set_knight_shift_config(KnightShiftConfig(enabled=True, unit=KnightShiftUnit.PPM))
    assert panel._joint_fit is None
    assert panel._DRAW_CROSSING_MARKERS is False


def test_joint_fit_state_round_trip(qapp):
    panel = _setup_panel(qapp, swap_past_crossing=True)
    traces = sorted(panel._knight_shift_names)
    panel._run_joint_knight_fit(traces, "KnightAnisotropy", 25)

    state = panel.get_state()
    assert state["joint_fit"] is not None

    panel2 = _setup_panel(qapp, swap_past_crossing=True)
    panel2.restore_state(state)
    assert panel2._joint_fit is not None
    assert panel2._joint_fit["model_name"] == "KnightAnisotropy"
    assert panel2._joint_fit["traces"] == list(traces)
    assert panel2._DRAW_CROSSING_MARKERS is True


def test_crossing_bands_merge_adjacent():
    events = [
        CrossingEvent(0, 0.0, 10.0, (0, 1), "order_swap"),
        CrossingEvent(1, 10.0, 20.0, (0, 1), "order_swap"),
        CrossingEvent(2, 20.0, 30.0, (0, 1), "order_swap"),
        CrossingEvent(8, 80.0, 90.0, (0, 1), "order_swap"),
    ]
    bands = FitParametersPanel._cluster_crossing_bands(events)
    # The three consecutive transitions merge into one band; the far one stays separate.
    assert len(bands) == 2
    assert bands[0][0] <= 0.0 and bands[0][1] >= 30.0
    assert bands[1][0] <= 80.0 and bands[1][1] >= 90.0


def test_joint_fit_draws_crossing_bands(qapp):
    panel = _setup_panel(qapp, swap_past_crossing=True)
    panel._run_joint_knight_fit(sorted(panel._knight_shift_names), "KnightAnisotropy", 25)
    panel._draw_plot()
    spans = [patch for ax in panel._figure.axes for patch in ax.patches]
    assert spans  # crossing bands are drawn as shaded spans
    # Bands never outnumber the raw crossing events (adjacent ones are merged).
    bands = panel._cluster_crossing_bands(panel.knight_shift_crossings())
    assert 0 < len(bands) <= len(panel.knight_shift_crossings())


def test_joint_fit_recovers_continuous_curves(qapp):
    panel = _setup_panel(qapp, swap_past_crossing=True)
    traces = sorted(panel._knight_shift_names)
    panel._run_joint_knight_fit(traces, "KnightAnisotropy", 25)
    # Each curve's fitted axial K_ax recovers one of the two physical anisotropies
    # (in ppm: 0.03 and -0.01 → 30000 and -10000).
    k_ax = sorted(
        round(p.value, -2)
        for name in traces
        for r in panel._model_fits[name].ranges
        if r.result
        for p in r.result.parameters
        if p.name == "K_ax"
    )
    assert k_ax == sorted([-10000.0, 30000.0])
