"""Knight-shift conversion wired into the parameter-trend panel (Phase 3c)."""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from asymmetry.core.fitting.knight_shift import (
    REFERENCE_COMPONENT,
    KnightShiftConfig,
    KnightShiftUnit,
    larmor_frequency_mhz,
)
from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def _select_y(panel: FitParametersPanel, names: list[str]) -> None:
    """Select the given Y-trace names in the Y-selector table."""
    from PySide6.QtCore import Qt

    wanted = set(names)
    table = panel._y_selector_table
    for i in range(table.rowCount()):
        item = table.item(i, 0)
        if item is None:
            continue
        pname = item.data(Qt.ItemDataRole.UserRole)
        item.setSelected(isinstance(pname, str) and pname in wanted)
    panel._selected_y_param_names = panel._selected_y_parameters()


def _row(run: int, field: float, values: dict[str, float]) -> dict:
    return {
        "run_number": run,
        "run_label": str(run),
        "field": field,
        "temperature": 10.0,
        "values": dict(values),
        "errors": {k: 0.001 for k in values},
    }


def _load(panel: FitParametersPanel, rows: list[dict]) -> None:
    panel.load_representation_series([("batch-1", "Series", rows)])


def test_applied_field_knight_shift_values(qapp):
    panel = FitParametersPanel()
    field = 7000.0
    nu_ref = larmor_frequency_mhz(field)
    _load(
        panel,
        [
            _row(1, field, {"frequency": nu_ref * 1.001}),
            _row(2, field, {"frequency": nu_ref * 1.002}),
        ],
    )
    panel.set_knight_shift_config(KnightShiftConfig(enabled=True, unit=KnightShiftUnit.FRACTION))

    assert "K[frequency]" in panel._display_y_parameters()
    ks = [row.values["K[frequency]"] for row in panel._rows]
    assert ks == pytest.approx([0.001, 0.002], rel=1e-6)


def test_field_parameterised_components_use_direct_field_ratio(qapp):
    # Oscillations parameterised by local field B_µ (Gauss), as in OscillatoryField
    # models: K = (B_µ − B_ext)/B_ext directly, no γ_µ. Regression for the real
    # perovskite project, whose components are field_1/field_2/field_3.
    panel = FitParametersPanel()
    b_ext = 7800.0
    _load(
        panel,
        [
            _row(1637, b_ext, {"field_1": 7849.0, "field_2": 7800.9}),
            _row(1638, b_ext, {"field_1": 7860.0, "field_2": 7805.0}),
        ],
    )
    panel.set_knight_shift_config(KnightShiftConfig(enabled=True, unit=KnightShiftUnit.FRACTION))
    assert "K[field_1]" in panel._knight_shift_names
    k1 = panel._rows[0].values["K[field_1]"]
    assert k1 == pytest.approx((7849.0 - b_ext) / b_ext, rel=1e-9)


def test_component_reference_excludes_reference_and_uses_it(qapp):
    panel = FitParametersPanel()
    _load(
        panel,
        [
            _row(1, 7000.0, {"frequency": 94.0, "frequency_2": 94.2}),
            _row(2, 7000.0, {"frequency": 95.0, "frequency_2": 95.3}),
        ],
    )
    panel.set_knight_shift_config(
        KnightShiftConfig(
            enabled=True,
            reference_mode=REFERENCE_COMPONENT,
            reference_component="frequency",
            unit=KnightShiftUnit.FRACTION,
        )
    )
    display = panel._display_y_parameters()
    # The reference component itself is not converted (K of ref vs itself = 0).
    assert "K[frequency_2]" in display
    assert "K[frequency]" not in panel._knight_shift_names
    k2 = panel._rows[0].values["K[frequency_2]"]
    assert k2 == pytest.approx((94.2 - 94.0) / 94.0, rel=1e-9)


def test_model_observable_map_excludes_non_local_field(qapp):
    # A series with two `field` params where the model marks only field_1 as a
    # local-field observable (field_2 being e.g. a muonium applied field): only
    # field_1 is converted. Mirrors the muonium-conflation fix.
    panel = FitParametersPanel()
    panel.load_representation_series(
        [("batch-1", "Series", [_row(1, 7000.0, {"field_1": 7050.0, "field_2": 7000.0})])],
        knight_observables_by_id={"batch-1": {"field_1": "field"}},
    )
    panel.set_knight_shift_config(KnightShiftConfig(enabled=True, unit=KnightShiftUnit.FRACTION))
    assert "K[field_1]" in panel._knight_shift_names
    assert "K[field_2]" not in panel._knight_shift_names


def test_component_reference_only_converts_same_kind(qapp):
    # With a field reference, a frequency component (different unit) must NOT be
    # converted against it.
    panel = FitParametersPanel()
    panel.load_representation_series(
        [
            (
                "batch-1",
                "Series",
                [_row(1, 7000.0, {"frequency": 94.0, "field_1": 7050.0, "field_2": 7080.0})],
            )
        ],
        knight_observables_by_id={
            "batch-1": {"frequency": "frequency", "field_1": "field", "field_2": "field"}
        },
    )
    panel.set_knight_shift_config(
        KnightShiftConfig(
            enabled=True,
            reference_mode=REFERENCE_COMPONENT,
            reference_component="field_1",
            unit=KnightShiftUnit.FRACTION,
        )
    )
    assert "K[field_2]" in panel._knight_shift_names
    assert "K[frequency]" not in panel._knight_shift_names  # different kind than the ref


def test_auto_unit_resolves_to_ppm_for_small_shift(qapp):
    panel = FitParametersPanel()
    field = 7000.0
    nu_ref = larmor_frequency_mhz(field)
    _load(panel, [_row(1, field, {"frequency": nu_ref * (1 + 5e-5)})])
    panel.set_knight_shift_config(KnightShiftConfig(enabled=True, unit=KnightShiftUnit.AUTO))
    # 50 ppm shift → AUTO picks ppm → stored value ≈ 50.
    assert panel._rows[0].values["K[frequency]"] == pytest.approx(50.0, rel=1e-4)


def test_stale_knight_columns_are_stripped_when_disabled(qapp):
    # A persisted K[...] column (e.g. from a project saved with the conversion on)
    # must not survive as a frozen trend parameter once the conversion is off.
    panel = FitParametersPanel()
    panel.load_representation_series(
        [("batch-1", "S", [_row(1, 7000.0, {"frequency": 94.0, "K[frequency]": 12.3})])]
    )
    # Default config is disabled.
    assert not any(name == "K[frequency]" for name in panel._rows[0].values)
    assert "K[frequency]" not in panel._display_y_parameters()


def test_disabled_config_produces_no_knight_shift(qapp):
    panel = FitParametersPanel()
    _load(panel, [_row(1, 7000.0, {"frequency": 94.0})])
    assert panel._knight_shift_names == {}
    assert not any(name.startswith("K[") for name in panel._display_y_parameters())


def test_crossing_is_flagged(qapp):
    panel = FitParametersPanel()
    # frequency and frequency_2 swap order between the two runs.
    _load(
        panel,
        [
            _row(1, 7000.0, {"frequency": 10.0, "frequency_2": 20.0}),
            _row(2, 7000.0, {"frequency": 21.0, "frequency_2": 11.0}),
        ],
    )
    panel.set_knight_shift_config(KnightShiftConfig(enabled=True, unit=KnightShiftUnit.FRACTION))
    kinds = {e.kind for e in panel.knight_shift_crossings()}
    assert "order_swap" in kinds


def test_crossings_use_raw_angle_not_folded(qapp):
    # Crossing detection must follow the raw scan order, not the folded display:
    # folding would collapse distinct orientations (e.g. 200° → 20°) onto one x.
    panel = FitParametersPanel()
    panel.set_angle_x_field(("Angle (°)", "angle"))

    def _arow(angle: str, f1: float, f2: float) -> dict:
        d = _row(int(float(angle)), 7000.0, {"field_1": f1, "field_2": f2})
        d["custom_values"] = {"angle": angle}
        return d

    panel.load_representation_series(
        [("b", "S", [_arow("0", 10.0, 20.0), _arow("90", 11.0, 30.0), _arow("200", 12.0, 5.0)])],
        knight_observables_by_id={"b": {"field_1": "field", "field_2": "field"}},
    )
    panel._x_combo.setCurrentIndex(panel._x_combo.findData("angle"))
    panel._angle_fold_combo.setCurrentIndex(panel._angle_fold_combo.findData(180.0))
    panel.set_knight_shift_config(KnightShiftConfig(enabled=True, unit=KnightShiftUnit.FRACTION))

    xs = {e.x_left for e in panel.knight_shift_crossings()}
    xs |= {e.x_right for e in panel.knight_shift_crossings()}
    assert 200.0 in xs  # raw angle used
    assert 20.0 not in xs  # not the folded value


def test_crossings_detected_but_markers_suppressed(qapp):
    panel = FitParametersPanel()
    _load(
        panel,
        [
            _row(1, 7000.0, {"frequency": 10.0, "frequency_2": 20.0}),
            _row(2, 7000.0, {"frequency": 21.0, "frequency_2": 11.0}),
        ],
    )
    panel.set_knight_shift_config(KnightShiftConfig(enabled=True, unit=KnightShiftUnit.FRACTION))
    # Crossings are still detected (the dialog reports them; they feed the future
    # realignment step) but the on-plot markers are suppressed for now.
    assert panel.knight_shift_crossings()
    panel._draw_plot()
    midpoint = 1.5  # between run 1 and run 2 on the inferred run x-axis
    axvlines = [
        line
        for ax in panel._figure.axes
        for line in ax.lines
        if len(line.get_xdata()) == 2 and line.get_xdata()[0] == line.get_xdata()[1] == midpoint
    ]
    assert not axvlines, "crossing markers should be suppressed"


def test_fraction_weights_note_shows_partition(qapp):
    # Under the n-1 scheme the free fractions are the weights and the group's last
    # term is the derived remainder; the note reports the full per-group partition
    # (free fractions plus the derived remainder) which sums to 1.
    panel = FitParametersPanel()
    panel.load_representation_series(
        [("batch-1", "S", [_row(1, 7000.0, {"field_1": 7050.0})])],
        fraction_weights_by_id={"batch-1": {"f_Aa": 0.465, "f_Bb": 0.310, "f_Cc": 0.225}},
    )
    note = panel._fraction_weights_note()
    assert "Fraction weights" in note
    assert "f_Aa = 0.465" in note
    assert "f_Cc = 0.225" in note  # the derived remainder of the group
    # Deterministically ordered (index then name).
    assert note.index("f_Aa") < note.index("f_Bb") < note.index("f_Cc")


def test_fraction_weights_note_empty_without_data(qapp):
    panel = FitParametersPanel()
    panel.load_representation_series([("batch-1", "S", [_row(1, 7000.0, {"field_1": 7050.0})])])
    assert panel._fraction_weights_note() == ""


def test_remove_button_deletes_selected_knight_trace(qapp, monkeypatch):
    # Ported from the removed test_knight_joint_fit.py: K-trace removal is a
    # panel-level behaviour (_remove_knight_traces / set_knight_shift_config)
    # independent of the joint fit, which now lives in the analysis window.
    from PySide6.QtWidgets import QMessageBox

    panel = FitParametersPanel()
    panel.load_representation_series(
        [
            (
                "batch-1",
                "S",
                [
                    _row(1, 7000.0, {"field_1": 7050.0, "field_2": 7100.0}),
                    _row(2, 7000.0, {"field_1": 7060.0, "field_2": 7110.0}),
                ],
            )
        ],
        knight_observables_by_id={"batch-1": {"field_1": "field", "field_2": "field"}},
    )
    panel.set_knight_shift_config(KnightShiftConfig(enabled=True, unit=KnightShiftUnit.PPM))
    traces = sorted(panel._knight_shift_names)
    assert len(traces) == 2

    _select_y(panel, [traces[0]])
    panel._update_composite_action_buttons()
    assert panel._remove_composite_btn.isEnabled()  # K traces are removable

    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
    )
    panel._remove_selected_composite_parameters()

    # The deleted trace is gone and excluded from the conversion so it won't regenerate.
    assert traces[0] not in panel._knight_shift_names
    assert not any(traces[0] in row.values for row in panel._rows)
    field = panel._knight_shift_names[traces[1]]
    assert panel._knight_shift_config.components == (field,)
    assert traces[1] in panel._knight_shift_names  # the unselected trace survives


def test_config_round_trips_through_state(qapp):
    panel = FitParametersPanel()
    _load(panel, [_row(1, 7000.0, {"frequency": 94.0, "frequency_2": 94.2})])
    panel.set_knight_shift_config(
        KnightShiftConfig(
            enabled=True,
            reference_mode=REFERENCE_COMPONENT,
            reference_component="frequency",
            unit=KnightShiftUnit.PPM,
        )
    )
    state = panel.get_state()

    restored = FitParametersPanel()
    restored.restore_state(state)
    cfg = restored.knight_shift_config()
    assert cfg.enabled is True
    assert cfg.reference_mode == REFERENCE_COMPONENT
    assert cfg.reference_component == "frequency"
    assert cfg.unit is KnightShiftUnit.PPM
    assert "K[frequency_2]" in restored._knight_shift_names


def test_knight_window_button_visible_when_series_has_knight_observables(qapp):
    # bed-next-angle-knight-shift.md §4.3: the button is the main-GUI shortcut,
    # shown only when the active series' model has a Knight-convertible
    # component (non-empty _knight_observables), unlike the always-available
    # menu action.
    panel = FitParametersPanel()
    panel.load_representation_series(
        [("batch-1", "S", [_row(1, 7000.0, {"field_1": 7050.0})])],
        knight_observables_by_id={"batch-1": {"field_1": "field"}},
    )
    # isHidden() reflects the explicit visibility flag without a shown ancestor
    # (isVisible() is always False until the top-level window is shown).
    assert not panel._knight_window_btn.isHidden()
    assert panel._knight_window_btn.isEnabled()


def test_knight_window_button_hidden_without_knight_observables(qapp):
    # Rows exist, but the fitted model has no Knight-convertible component
    # (no knight_observables_by_id passed): the button must stay hidden even
    # though the old behaviour (bool(self._rows)) would have enabled it.
    panel = FitParametersPanel()
    _load(panel, [_row(1, 7000.0, {"frequency": 94.0})])
    assert panel._knight_window_btn.isHidden()


def test_knight_window_button_hidden_after_clear(qapp):
    panel = FitParametersPanel()
    panel.load_representation_series(
        [("batch-1", "S", [_row(1, 7000.0, {"field_1": 7050.0})])],
        knight_observables_by_id={"batch-1": {"field_1": "field"}},
    )
    assert not panel._knight_window_btn.isHidden()  # sanity: visible before clear

    panel.clear()
    assert panel._knight_window_btn.isHidden()


def test_knight_window_button_hidden_on_fresh_panel(qapp):
    panel = FitParametersPanel()
    assert panel._knight_window_btn.isHidden()
