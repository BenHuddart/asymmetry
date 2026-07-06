"""Knight-shift analysis window (:mod:`asymmetry.gui.windows.knight_shift_window`)."""

from __future__ import annotations

import os

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
from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel
from asymmetry.gui.windows.knight_shift_window import KnightShiftWindow


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

    assert len(window._figure.axes) == 1
    assert len(window._figure.axes[0].containers) > 0
