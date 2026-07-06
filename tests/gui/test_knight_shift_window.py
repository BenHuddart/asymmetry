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


def _theta0_snapshot(n_points: int = 19, theta0_true: float = 20.0) -> KnightAnalysisInput:
    """A 2-branch axial scan whose K(θ) carries a known mount offset θ0.

    Built by passing ``theta0`` straight through the ``KnightAnisotropy``
    component function (mirroring ``_axial_field`` above), so a joint fit
    against these branches should recover ``theta0_true`` for each curve.
    The two sites' K_ax have opposite sign but the same magnitude
    everywhere shy of the fold points, so — unlike ``_crossing_snapshot`` —
    the branches never cross or swap identity: the point here is isolating
    the theta0 read-out, not the assignment machinery.
    """
    angles = np.linspace(0.0, 40.0, n_points)  # short span: stays well clear of any crossing
    fn = PARAMETER_MODEL_COMPONENTS["KnightAnisotropy"].function
    b_ext = 7000.0
    k_a = fn(angles, K_iso=0.02, K_ax=0.03, theta0=theta0_true)
    k_b = fn(angles, K_iso=0.02, K_ax=-0.02, theta0=theta0_true)
    field_a = b_ext * (1.0 + k_a)
    field_b = b_ext * (1.0 + k_b)
    points = tuple(
        KnightPoint(
            run_number=i + 1,
            run_label=str(i + 1),
            x=float(ang),
            field_gauss=b_ext,
            values={"field_1": float(field_a[i]), "field_2": float(field_b[i])},
            errors={"field_1": 0.2, "field_2": 0.2},
        )
        for i, ang in enumerate(angles)
    )
    return KnightAnalysisInput(
        x_key="angle",
        x_label="Angle (°)",
        components=(("field_1", "field"), ("field_2", "field")),
        points=points,
        source_label="Theta0 scan",
        batch_id="batch-theta0",
        group_id="group-theta0",
    )


def _misfit_crossing_snapshot(n_points: int = 13) -> KnightAnalysisInput:
    """Like :func:`_crossing_snapshot` but with an added deterministic misfit.

    An alternating +/- perturbation is added on top of the smooth K(theta)
    curves so the model cannot absorb it — driving reduced chi-squared above
    one deterministically (no RNG), so the "scale errors by root chi-squared_r"
    checkbox has real work to do.
    """
    angles = np.linspace(0.0, 90.0, n_points)
    b_ext = 7000.0
    a = _axial_field(angles, 0.02, 0.03, b_ext)
    b = _axial_field(angles, 0.02, -0.01, b_ext)
    past = angles > 54.7356103
    f1 = np.where(past, b, a)
    f2 = np.where(past, a, b)
    # Deterministic alternating misfit, well above the quoted 0.2 G error bar.
    wobble = np.array([3.0 if i % 2 == 0 else -3.0 for i in range(n_points)])
    f1 = f1 + wobble
    f2 = f2 - wobble
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
        source_label="Misfit crossing series",
        batch_id="batch-misfit",
        group_id="group-misfit",
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


# ── 13. Lorentz/demag correction ─────────────────────────────────────────────


def test_correction_shifts_branch_k_by_offset_and_leaves_errors_unchanged(qapp):
    window = KnightShiftWindow()
    window.set_snapshot(_snapshot())

    # Baseline: correction unchecked (shape defaults to sphere anyway).
    baseline_k = {b.name: b.k for b in window._result.branches}
    baseline_err = {b.name: b.k_err for b in window._result.branches}
    assert "corrected" not in window._footer._status_label.text()

    window._shape_combo.setCurrentIndex(window._shape_combo.findData("plate_perpendicular"))
    window._chi_edit.setText("1e-3")
    window._correction_check.setChecked(True)

    # N = 1, chi = 1e-3 => offset = -(1/3 - 1) * 1e-3 = +2e-3/3.
    expected_offset = 2e-3 / 3.0
    for branch in window._result.branches:
        shifted = branch.k
        base = baseline_k[branch.name]
        assert shifted == pytest.approx(tuple(k + expected_offset for k in base), abs=1e-12)
        assert branch.k_err == baseline_err[branch.name]

    assert "Lorentz/demag corrected" in window._footer._status_label.text()


def test_n_field_only_enabled_for_custom_shape(qapp):
    window = KnightShiftWindow()
    window.set_snapshot(_snapshot())

    for label, key in (
        ("sphere", "sphere"),
        ("plate_parallel", "plate_parallel"),
        ("plate_perpendicular", "plate_perpendicular"),
        ("cylinder_axial", "cylinder_axial"),
        ("cylinder_transverse", "cylinder_transverse"),
    ):
        window._shape_combo.setCurrentIndex(window._shape_combo.findData(key))
        assert not window._custom_n_edit.isEnabled(), f"N enabled for shape {label!r}"

    window._shape_combo.setCurrentIndex(window._shape_combo.findData("custom"))
    assert window._custom_n_edit.isEnabled()

    window._custom_n_edit.setText("0.25")
    window._correction_check.setChecked(True)
    correction = window._current_correction()
    assert correction.shape == "custom"
    assert correction.custom_n == pytest.approx(0.25)
    assert correction.demag_factor() == pytest.approx(0.25)


# ── 14. Correction + rescale state round-trip ────────────────────────────────


def test_correction_and_rescale_round_trip_through_state(qapp):
    window = KnightShiftWindow()
    window.set_snapshot(_snapshot())

    window._shape_combo.setCurrentIndex(window._shape_combo.findData("custom"))
    window._custom_n_edit.setText("0.25")
    window._chi_edit.setText("0.001")
    window._correction_check.setChecked(True)
    window._rescale_check.setChecked(True)

    state = window.get_state()
    assert state["correction"]["enabled"] is True
    assert state["correction"]["shape"] == "custom"
    assert state["rescale_errors"] is True

    restored = KnightShiftWindow()
    restored.restore_state(state)

    assert restored._correction_check.isChecked() is True
    assert restored._rescale_check.isChecked() is True
    assert restored._shape_combo.currentData() == "custom"
    assert restored._custom_n_edit.text() == "0.25"
    assert restored._chi_edit.text() == "0.001"

    restored_correction = restored._current_correction()
    assert restored_correction.enabled is True
    assert restored_correction.shape == "custom"
    assert restored_correction.custom_n == pytest.approx(0.25)
    assert restored_correction.chi_volume_si == pytest.approx(0.001)


# ── 15. Correction offset staleness for the joint fit ────────────────────────


def test_changing_chi_after_a_joint_fit_marks_curves_stale_then_fresh_again(qapp):
    window = KnightShiftWindow()
    window.set_snapshot(_crossing_snapshot())

    # Pin a concrete display unit so AUTO can't reselect a different unit when
    # the correction offset shifts the fractions — that would confound the
    # thing this test isolates (the offset check in _joint_curves_fresh()).
    window._unit_combo.setCurrentIndex(2)  # percent
    window._shape_combo.setCurrentIndex(window._shape_combo.findData("plate_perpendicular"))
    window._chi_edit.setText("1e-3")
    window._correction_check.setChecked(True)

    window._on_run_joint_fit()
    wait_for(lambda: not window._joint_running, QApplication.instance(), timeout_s=20.0)
    assert window._joint_applies()
    assert window._joint_curves_fresh()
    assert "re-run to refresh" not in window._fit_results_label.text()

    # Change chi (correction offset changes) without re-running the fit.
    window._chi_edit.setText("2e-3")
    assert not window._joint_curves_fresh()
    window._update_fit_controls()
    assert "re-run to refresh" in window._fit_results_label.text()

    # Revert chi: freshness (and the label) is restored.
    window._chi_edit.setText("1e-3")
    assert window._joint_curves_fresh()
    window._update_fit_controls()
    assert "re-run to refresh" not in window._fit_results_label.text()


# ── 16. theta0 recovered from a known offset scan ────────────────────────────


def test_joint_fit_recovers_known_theta0_offset(qapp):
    theta0_true = 20.0
    window = KnightShiftWindow()
    window.set_snapshot(_theta0_snapshot(theta0_true=theta0_true))
    window._unit_combo.setCurrentIndex(2)  # percent

    assert window._fit_btn.isEnabled()
    window._on_run_joint_fit()
    wait_for(lambda: not window._joint_running, QApplication.instance(), timeout_s=20.0)

    assert window._state.joint is not None
    assert "theta0" in window._fit_results_label.text()

    for curve in window._state.joint.curves:
        by_name = {name: value for name, value, _err in curve.parameters}
        assert "theta0" in by_name
        assert by_name["theta0"] == pytest.approx(theta0_true, abs=2.0)


# ── 17. Rescale-errors suffix only appears for a genuine chi-squared_r > 1 misfit ──


def test_rescale_suffix_appears_only_for_chi_squared_r_above_one(qapp):
    window = KnightShiftWindow()
    window.set_snapshot(_misfit_crossing_snapshot())
    window._unit_combo.setCurrentIndex(2)  # percent

    window._on_run_joint_fit()
    wait_for(lambda: not window._joint_running, QApplication.instance(), timeout_s=20.0)
    assert window._state.joint is not None

    # Self-check: the crafted misfit must actually drive chi-squared_r above
    # one, or the rest of the test would silently check nothing.
    chi2r_values = [curve.reduced_chi_squared for curve in window._state.joint.curves]
    assert any(c == c and c > 1.0 for c in chi2r_values), (
        f"expected at least one branch with chi-squared_r > 1, got {chi2r_values}"
    )

    window._rescale_check.setChecked(True)
    assert "(errors ×√χ²ᵣ)" in window._fit_results_label.text()

    window._rescale_check.setChecked(False)
    assert "(errors ×√χ²ᵣ)" not in window._fit_results_label.text()


def test_rescale_suffix_absent_when_chi_squared_r_is_not_above_one(qapp):
    window = KnightShiftWindow()
    window.set_snapshot(_snapshot())
    window._unit_combo.setCurrentIndex(2)  # percent

    # _snapshot() is a smooth linear trend with no crossing, well inside a
    # KnightAnisotropy fit's reach: chi-squared_r stays at or below one.
    window._on_run_joint_fit()
    wait_for(lambda: not window._joint_running, QApplication.instance(), timeout_s=20.0)
    if window._state.joint is None:
        pytest.skip("joint fit did not converge for the smooth-trend snapshot")

    chi2r_values = [curve.reduced_chi_squared for curve in window._state.joint.curves]
    assert all(c != c or c <= 1.0 for c in chi2r_values)

    window._rescale_check.setChecked(True)
    assert "(errors ×√χ²ᵣ)" not in window._fit_results_label.text()

    window._rescale_check.setChecked(False)
    assert "(errors ×√χ²ᵣ)" not in window._fit_results_label.text()
