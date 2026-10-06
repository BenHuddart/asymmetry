"""The rotating-frame bar and Auto-detect's review (rotating-frame-projection D3–D6)."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QLabel

from asymmetry.core.fourier.units import FieldUnit, convert
from asymmetry.core.transform.rotating_frame import (
    B1Axis,
    FrameEstimate,
    Provenance,
    RotatingFrame,
    RunEstimate,
)
from asymmetry.gui.widgets.rotating_frame_bar import (
    FIRST_SWITCH_HINT,
    RotatingFrameBar,
    frame_badge_text,
    rotated_ylabel,
)
from asymmetry.gui.widgets.rotating_frame_review import FOOTER_NOTE, FrameReviewDialog

pytestmark = [pytest.mark.gui]

#: Invented frames: one fresh from ν, one with estimates, one with typed values.
TYPED_NU = RotatingFrame.typed_frequency(1.5)
ESTIMATED = TYPED_NU.with_values(
    Provenance.ESTIMATED, rf_phase_deg=40.0, sense=1, baseline_x=0.3, baseline_y=-0.2, gain=0.9
)
TYPED = ESTIMATED.with_values(Provenance.TYPED, rf_phase_deg=12.0, gain=1.1)


def _edits(bar: RotatingFrameBar) -> list[tuple[list[int], str, object]]:
    events: list[tuple[list[int], str, object]] = []
    bar.field_edited.connect(lambda runs, name, value: events.append((list(runs), name, value)))
    return events


def _type(bar: RotatingFrameBar, name: str, text: str) -> None:
    edit = bar._edits[name]
    edit.setText(text)
    edit.setModified(True)
    edit.editingFinished.emit()


class TestFrameBar:
    def test_first_switch_asks_for_frequency_only(self, qapp):
        bar = RotatingFrameBar()
        bar.show_frames({7: None, 8: None}, 7)
        frequency = bar._edits["frequency_mhz"]
        assert frequency.text() == "" and frequency.placeholderText() == "required"
        assert frequency.isEnabled()
        assert all(not bar._edits[name].isEnabled() for name in ("rf_phase_deg", "gain"))
        assert bar._edits["rf_phase_deg"].text() == "0.0"  # the typed_frequency default
        assert not bar._b1_toggle.isEnabled() and not bar._sense_btn.isEnabled()
        assert not bar._detect_btn.isEnabled()
        assert not bar._hint.isHidden() and bar._hint.text() == FIRST_SWITCH_HINT
        events = _edits(bar)
        _type(bar, "frequency_mhz", "1.4925")
        assert events == [([7, 8], "frequency_mhz", 1.4925)]

    def test_frequency_typed_in_gauss_is_reported_in_mhz(self, qapp):
        bar = RotatingFrameBar()
        bar.show_frames({7: TYPED_NU}, 7)
        bar._unit_group.button(1).click()
        assert bar.frequency_unit() is FieldUnit.GAUSS
        shown = float(bar._edits["frequency_mhz"].text())
        assert shown == pytest.approx(float(convert(1.5, FieldUnit.MHZ, FieldUnit.GAUSS)), 1e-5)
        events = _edits(bar)
        _type(bar, "frequency_mhz", "100")
        expected = float(convert(100.0, FieldUnit.GAUSS, FieldUnit.MHZ))
        assert events[0][2] == pytest.approx(expected)

    def test_setup_fields_the_runs_disagree_on_read_mixed(self, qapp):
        bar = RotatingFrameBar()
        bar.show_frames({7: ESTIMATED, 8: TYPED}, 8)
        phase = bar._edits["rf_phase_deg"]
        assert phase.text() == "" and phase.placeholderText() == "mixed"
        assert bar._edits["frequency_mhz"].text() == "1.5"
        assert bar._sense_btn.text() == "↺ +1"
        events = _edits(bar)
        _type(bar, "rf_phase_deg", "30")
        assert events == [([7, 8], "rf_phase_deg", 30.0)]

    def test_run_fields_follow_the_selected_run(self, qapp):
        bar = RotatingFrameBar()
        bar.show_frames({7: ESTIMATED, 8: TYPED}, 8)
        assert bar._edits["gain"].text() == "1.1"
        assert bar._run_header.text() == "RUN 8"
        events = _edits(bar)
        _type(bar, "baseline_x", "0.5")
        assert events == [([8], "baseline_x", 0.5)]
        bar.show_frames({7: ESTIMATED, 8: TYPED}, 7)
        assert bar._edits["gain"].text() == "0.9"

    def test_toggles_report_their_new_values_for_every_run(self, qapp):
        bar = RotatingFrameBar()
        bar.show_frames({7: ESTIMATED, 8: ESTIMATED}, 7)
        events = _edits(bar)
        bar._b1_cells[1].click()
        bar._sense_btn.click()
        assert events == [([7, 8], "b1_axis", B1Axis.Y), ([7, 8], "sense", -1)]

    def test_provenance_styles(self, qapp):
        bar = RotatingFrameBar()
        bar.show_frames({7: TYPED}, 7)
        # Typed: the ✎ marker; estimated: plain; default (B₁ axis here): grey italic.
        assert not bar._markers["rf_phase_deg"].isHidden()
        assert bar._markers["baseline_x"].isHidden()
        assert "italic" not in bar._edits["baseline_x"].styleSheet()
        assert "italic" in bar._b1_cells[0].styleSheet()
        bar.show_frames({7: TYPED_NU}, 7)
        assert "italic" in bar._edits["gain"].styleSheet()
        assert not bar._markers["frequency_mhz"].isHidden()

    def test_auto_detect_needs_one_frequency_and_axis(self, qapp):
        bar = RotatingFrameBar()
        bar.show_frames({7: ESTIMATED, 8: TYPED}, 7)
        assert bar._detect_btn.isEnabled()
        other = ESTIMATED.with_values(Provenance.TYPED, frequency_mhz=1.6)
        bar.show_frames({7: ESTIMATED, 8: other}, 7)
        assert not bar._detect_btn.isEnabled()

    def test_narrow_bar_wraps_then_folds_the_run_fields(self, qapp):
        bar = RotatingFrameBar()
        bar.show_frames({7: ESTIMATED}, 7)
        bar.show()
        height = bar.sizeHint().height()
        bar.resize(bar.sizeHint().width(), height)
        qapp.processEvents()
        assert bar.presentation() == "wide"
        bar.resize(bar.sizeHint().width() - 1, height)
        qapp.processEvents()
        assert bar.presentation() == "wrapped"
        bar.resize(bar.minimumSizeHint().width(), height)
        qapp.processEvents()
        assert bar.presentation() == "folded"
        assert bar._run_btn.text() == "Run 7 ▾" and not bar._run_btn.isHidden()
        assert bar._detect_btn.text() == "Detect…"
        assert bar._run_box.parent() is bar._run_popover
        assert bar.minimumSizeHint().width() < bar.sizeHint().width() / 2


class TestPlotText:
    def test_badge_names_the_shared_frame(self):
        frame = TYPED_NU.with_values(Provenance.TYPED, frequency_mhz=1.4925, rf_phase_deg=47.0)
        assert frame_badge_text([frame]) == (
            "Rotating frame · ν = 1.4925 MHz · B₁ ∥ x′ · φ_RF = 47.0° · s = −1"
        )
        other = frame.with_values(Provenance.TYPED, rf_phase_deg=12.0)
        assert "φ_RF mixed" in frame_badge_text([frame, other])

    def test_ylabel_relates_the_rotated_axis_to_b1(self):
        along_y = TYPED_NU.with_values(Provenance.TYPED, b1_axis=B1Axis.Y)
        assert r"\parallel B_1" in rotated_ylabel("P′_x", [TYPED_NU])
        assert r"\perp B_1" in rotated_ylabel("P′_y", [TYPED_NU])
        assert r"\parallel B_1" in rotated_ylabel("P′_y", [along_y])
        assert "B_1" not in rotated_ylabel("P′_x", [TYPED_NU, along_y])


def _estimate(contrast: float = 12.0, run_contrast: tuple[float, float] = (8.0, 2.0)):
    return FrameEstimate(
        frequency_mhz=1.5006,
        sense=-1,
        rf_phase_deg=47.0,
        contrast=contrast,
        runs=(
            RunEstimate(7, 0.31, -0.18, 0.93, run_contrast[0]),
            RunEstimate(8, 0.29, -0.21, 0.95, run_contrast[1]),
        ),
    )


def _ticks(dialog: FrameReviewDialog) -> dict[str, tuple[bool, bool]]:
    return {tick.text(): (tick.isChecked(), tick.isEnabled()) for tick, _row in dialog._ticks}


class TestReview:
    def test_pre_ticks_defaults_and_estimates_never_typed_values(self, qapp):
        dialog = FrameReviewDialog(_estimate(), {7: ESTIMATED, 8: TYPED})
        assert _ticks(dialog) == {
            "ν_RF": (False, True),  # typed by the user
            "Sense": (True, True),
            "φ_RF": (False, True),  # run 8 typed it
            "Run 7": (True, True),
            "Run 8": (False, False),  # contrast 2 < 3
        }
        assert dialog._apply_btn.text() == "Apply 2 changes"
        assert FOOTER_NOTE.startswith("Ticked rows are written as estimates.")

    def test_weak_shared_contrast_disables_every_setup_row(self, qapp):
        dialog = FrameReviewDialog(_estimate(contrast=2.5), {7: TYPED_NU, 8: TYPED_NU})
        ticks = _ticks(dialog)
        assert ticks["Sense"] == (False, False) and ticks["φ_RF"] == (False, False)
        assert ticks["Run 7"] == (True, True)

    def test_apply_reports_only_ticked_values(self, qapp):
        dialog = FrameReviewDialog(_estimate(), {7: TYPED_NU, 8: TYPED_NU})
        applied: list[tuple[dict, str]] = []
        dialog.applied.connect(lambda changes, status: applied.append((changes, status)))
        sense = next(tick for tick, row in dialog._ticks if row.title == "Sense")
        sense.setChecked(False)
        assert dialog._apply_btn.text() == "Apply 2 changes"
        dialog._apply_btn.click()
        changes, status = applied[0]
        assert changes == {
            7: {"rf_phase_deg": 47.0, "baseline_x": 0.31, "baseline_y": -0.18, "gain": 0.93},
            8: {"rf_phase_deg": 47.0},
        }
        assert status == "✓ Applied 2 estimates · contrast 12"

    def test_frequency_row_is_a_check_on_the_typed_value(self, qapp):
        dialog = FrameReviewDialog(_estimate(), {7: TYPED_NU})
        labels = [label.text() for label in dialog.findChildren(QLabel)]
        assert "1.5 MHz → 1.5006 MHz" in labels
        assert "typed · agrees within 0.04 %" in labels
