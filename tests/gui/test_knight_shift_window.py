"""Knight-shift analysis window (:mod:`asymmetry.gui.windows.knight_shift_window`)."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from asymmetry.core.fitting.knight_analysis import KnightAnalysisInput, KnightPoint
from asymmetry.core.fitting.knight_shift import (
    REFERENCE_APPLIED_FIELD,
    REFERENCE_COMPONENT,
    KnightShiftConfig,
    KnightShiftUnit,
)
from asymmetry.core.fitting.parameter_models import PARAMETER_MODEL_COMPONENTS
from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel
from asymmetry.gui.windows.knight_shift_window import KnightShiftWindow
from tests._qt_helpers import wait_for


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def _snapshot(n_points: int = 10) -> KnightAnalysisInput:
    """A 2-component field-kind snapshot spanning angles 0..90 degrees."""
    points = tuple(
        KnightPoint(
            run_number=i,
            run_label=str(1000 + i),
            x=float(i) * (90.0 / max(1, n_points - 1)),
            field_gauss=7000.0,
            values={"field_1": 7050.0 + i, "field_2": 7100.0 + 0.5 * i},
            errors={"field_1": 1.0, "field_2": 1.0},
        )
        for i in range(n_points)
    )
    return KnightAnalysisInput(
        x_key="angle",
        x_label="Angle (°)",
        components=(("field_1", "field"), ("field_2", "field")),
        points=points,
        source_label="Test series",
        batch_id="batch-1",
        group_id="group-1",
    )


def _axial_field(angle, k_iso, k_ax, b_ext=7000.0):
    """A local field giving an axial K(θ) under the applied-field reference."""
    fn = PARAMETER_MODEL_COMPONENTS["KnightAnisotropy"].function
    k = fn(np.asarray(angle, dtype=float), K_iso=k_iso, K_ax=k_ax)  # dimensionless
    return b_ext * (1.0 + k)


def _crossing_snapshot(n_points: int = 13) -> KnightAnalysisInput:
    """A 2-branch angle scan whose K(θ) curves cross at the magic angle (~54.7°).

    Mirrors the construction used by ``tests/core/test_angular_assignment.py``
    and the removed panel joint-fit tests: labels are swapped past the
    crossing (as a grouped fit would emit), so the joint fit has real work to
    do (assignment departs from identity) and is proven to converge.
    """
    angles = np.linspace(0.0, 90.0, n_points)
    b_ext = 7000.0
    a = _axial_field(angles, 0.02, 0.03, b_ext)  # two crossing K(θ) curves
    b = _axial_field(angles, 0.02, -0.01, b_ext)
    past = angles > 54.7356103
    f1 = np.where(past, b, a)
    f2 = np.where(past, a, b)
    points = tuple(
        KnightPoint(
            run_number=i + 1,
            run_label=str(i + 1),
            x=float(ang),
            field_gauss=b_ext,
            values={"field_1": float(f1[i]), "field_2": float(f2[i])},
            errors={"field_1": 0.2, "field_2": 0.2},
        )
        for i, ang in enumerate(angles)
    )
    return KnightAnalysisInput(
        x_key="angle",
        x_label="Angle (°)",
        components=(("field_1", "field"), ("field_2", "field")),
        points=points,
        source_label="Crossing series",
        batch_id="batch-crossing",
        group_id="group-crossing",
    )


# ── 1. No snapshot ───────────────────────────────────────────────────────────


def test_no_snapshot_shows_no_data_and_disables_send(qapp):
    window = KnightShiftWindow()
    assert not window._send_btn.isEnabled()
    assert "No data" in window._footer._status_label.text()


# ── 2. set_snapshot populates branches / checkboxes / combo ─────────────────


def test_set_snapshot_populates_branches_and_controls(qapp):
    window = KnightShiftWindow()
    window.set_snapshot(_snapshot())

    assert window._send_btn.isEnabled()
    assert len(window._component_checks) == 2
    assert set(window._component_checks) == {"field_1", "field_2"}
    assert window._ref_component_combo.count() == 2
    # One branch row per convertible component.
    branch_rows = [
        window._branches_layout.itemAt(i).widget() for i in range(window._branches_layout.count())
    ]
    assert len(branch_rows) == 2


# ── 3. Unit combo mapping ────────────────────────────────────────────────────


def test_unit_combo_selection_maps_to_config_unit(qapp):
    window = KnightShiftWindow()
    window.set_snapshot(_snapshot())

    expected = [
        (0, KnightShiftUnit.AUTO),
        (1, KnightShiftUnit.PPM),
        (2, KnightShiftUnit.PERCENT),
        (3, KnightShiftUnit.FRACTION),
    ]
    for index, unit in expected:
        window._unit_combo.setCurrentIndex(index)
        assert window.get_state()["config"]["unit"] == unit.value


# ── 4. Reference mode ────────────────────────────────────────────────────────


def test_component_reference_mode_enables_combo_and_sets_config(qapp):
    window = KnightShiftWindow()
    window.set_snapshot(_snapshot())

    assert not window._ref_component_combo.isEnabled()
    window._ref_component_radio.setChecked(True)
    assert window._ref_component_combo.isEnabled()

    index = window._ref_component_combo.findData("field_2")
    window._ref_component_combo.setCurrentIndex(index)

    config = window.current_config()
    assert config.reference_mode == REFERENCE_COMPONENT
    assert config.reference_component == "field_2"


def test_applied_field_reference_is_default(qapp):
    window = KnightShiftWindow()
    window.set_snapshot(_snapshot())
    assert window._ref_field_radio.isChecked()
    assert window.current_config().reference_mode == REFERENCE_APPLIED_FIELD


# ── 5. Component subset ──────────────────────────────────────────────────────


def test_unchecking_one_component_yields_one_tuple_subset(qapp):
    window = KnightShiftWindow()
    window.set_snapshot(_snapshot())

    window._component_checks["field_1"].setChecked(False)
    assert window.current_config().components == ("field_2",)


def test_all_checked_yields_empty_tuple_meaning_all(qapp):
    window = KnightShiftWindow()
    window.set_snapshot(_snapshot())

    for box in window._component_checks.values():
        box.setChecked(True)
    assert window.current_config().components == ()


# ── 6. get_state()/restore_state round trip ──────────────────────────────────


def test_state_round_trips_through_a_new_window(qapp):
    window = KnightShiftWindow()
    window.set_snapshot(_snapshot())

    window._unit_combo.setCurrentIndex(2)  # percent
    window._fold_check.setChecked(True)
    window._markers_check.setChecked(False)
    state = window.get_state()

    restored = KnightShiftWindow()
    restored.restore_state(state)

    assert restored._unit_combo.currentIndex() == 2
    assert restored._fold_check.isChecked() is True
    assert restored._markers_check.isChecked() is False
    assert restored.current_config().unit is KnightShiftUnit.PERCENT
    assert restored.get_state()["config"] == state["config"]


# ── 7. restore_state before snapshot ────────────────────────────────────────


def test_restore_state_before_snapshot_should_honour_component_subset(qapp):
    """A persisted component subset survives to the next snapshot.

    Regression guard: restore_state() runs before set_snapshot() when a saved
    project reopens the window, and its trailing re-evaluation must not
    recompute the config from the still-empty checkbox controls (which would
    collapse a partial subset to "all components").
    """
    window = KnightShiftWindow()
    state = {
        "config": KnightShiftConfig(enabled=True, components=("field_2",)).to_dict(),
        "source_batch_id": None,
        "source_group_id": None,
        "x_key": "angle",
        "fold_180": False,
        "show_markers": True,
    }
    window.restore_state(state)
    window.set_snapshot(_snapshot())

    assert window._component_checks["field_1"].isChecked() is False
    assert window._component_checks["field_2"].isChecked() is True


# ── 8. apply_config_requested ────────────────────────────────────────────────


def test_send_button_emits_apply_config_requested(qapp):
    window = KnightShiftWindow()
    window.set_snapshot(_snapshot())

    captured: list[KnightShiftConfig] = []
    window.apply_config_requested.connect(captured.append)
    window._send_btn.click()

    assert len(captured) == 1
    assert captured[0].enabled is True


# ── 9. refresh_requested ─────────────────────────────────────────────────────


def test_refresh_button_emits_refresh_requested(qapp):
    window = KnightShiftWindow()

    captured: list[object] = []
    window.refresh_requested.connect(lambda: captured.append(True))
    window._refresh_btn.click()

    assert captured == [True]


# ── 10. Panel integration ────────────────────────────────────────────────────


def _row(run: int, field: float, values: dict[str, float]) -> dict:
    return {
        "run_number": run,
        "run_label": str(run),
        "field": field,
        "temperature": 10.0,
        "values": dict(values),
        "errors": {k: 0.001 for k in values},
    }


def test_panel_knight_analysis_snapshot_matches_loaded_rows(qapp):
    panel = FitParametersPanel()
    panel.load_representation_series(
        [
            (
                "batch-1",
                "Series",
                [
                    _row(1, 7000.0, {"field_1": 7050.0, "field_2": 7080.0}),
                    _row(2, 7000.0, {"field_1": 7060.0, "field_2": 7090.0}),
                ],
            )
        ],
        knight_observables_by_id={"batch-1": {"field_1": "field", "field_2": "field"}},
    )

    snapshot = panel.knight_analysis_snapshot()
    assert snapshot is not None
    assert snapshot.component_names() == ("field_1", "field_2")
    assert len(snapshot.points) == 2
    assert snapshot.x_key


def test_panel_knight_window_button_emits_signal_once_rows_loaded(qapp):
    panel = FitParametersPanel()
    captured: list[object] = []
    panel.knight_window_requested.connect(lambda: captured.append(True))

    assert not panel._knight_window_btn.isEnabled()
    panel.load_representation_series(
        [("batch-1", "Series", [_row(1, 7000.0, {"field_1": 7050.0})])],
        knight_observables_by_id={"batch-1": {"field_1": "field"}},
    )
    assert panel._knight_window_btn.isEnabled()

    panel._knight_window_btn.click()
    assert captured == [True]


# ── 11. Matplotlib figure content ────────────────────────────────────────────


def test_figure_draws_one_axes_with_errorbar_containers_per_branch(qapp):
    window = KnightShiftWindow()
    if window._figure is None:
        pytest.skip("matplotlib is not installed")

    window.set_snapshot(_snapshot())

    assert len(window._figure.axes) == 1
    ax = window._figure.axes[0]
    assert len(ax.containers) > 0


def test_toggling_fold_and_markers_redraws_without_exception(qapp):
    window = KnightShiftWindow()
    if window._figure is None:
        pytest.skip("matplotlib is not installed")

    window.set_snapshot(_snapshot())

    window._fold_check.setChecked(True)
    window._markers_check.setChecked(False)
    window._fold_check.setChecked(False)
    window._markers_check.setChecked(True)


# ── 12. Joint K(θ) fit ────────────────────────────────────────────────────────


def test_run_joint_fit_converges_and_updates_state_and_controls(qapp):
    window = KnightShiftWindow()
    window.set_snapshot(_crossing_snapshot())

    assert window._fit_btn.isEnabled()
    window._on_run_joint_fit()
    assert not window._fit_btn.isEnabled()  # disabled while the fit runs

    wait_for(lambda: not window._joint_running, QApplication.instance(), timeout_s=20.0)

    assert window._state.joint is not None
    assert window._joint_applies()
    assert window._fit_results_label.text() != ""
    assert window._fit_btn.isEnabled()
    assert window._clear_fit_btn.isEnabled()


def test_joint_fit_state_round_trips_into_a_new_window(qapp):
    window = KnightShiftWindow()
    snapshot = _crossing_snapshot()
    window.set_snapshot(snapshot)
    window._on_run_joint_fit()
    wait_for(lambda: not window._joint_running, QApplication.instance(), timeout_s=20.0)
    assert window._joint_applies()

    state = window.get_state()
    assert state["joint"] is not None

    restored = KnightShiftWindow()
    restored.restore_state(state)
    restored.set_snapshot(snapshot)

    assert restored._state.joint is not None
    assert restored._joint_applies()


def test_clear_fit_clears_joint_state(qapp):
    window = KnightShiftWindow()
    window.set_snapshot(_crossing_snapshot())
    window._on_run_joint_fit()
    wait_for(lambda: not window._joint_running, QApplication.instance(), timeout_s=20.0)
    assert window._state.joint is not None

    window._clear_fit_btn.click()

    assert window._state.joint is None
    assert not window._clear_fit_btn.isEnabled()
    assert window._fit_results_label.text() == ""


def test_unchecking_a_component_invalidates_the_stored_fit(qapp):
    window = KnightShiftWindow()
    window.set_snapshot(_crossing_snapshot())
    window._on_run_joint_fit()
    wait_for(lambda: not window._joint_running, QApplication.instance(), timeout_s=20.0)
    assert window._joint_applies()

    window._component_checks["field_1"].setChecked(False)

    # Only one branch remains: the stored (two-branch) fit no longer applies.
    assert not window._joint_applies()
    assert "does not match" in window._fit_results_label.text()

    assert len(window._figure.axes) == 1
    assert len(window._figure.axes[0].containers) > 0
