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
