"""Tests for the inline background-configuration section (Corrections panel).

Covers the embeddable body that replaced the retired ``BackgroundDialog``: mode
selection, the ``changed`` signal, and the reference-run picker (which delegates
to ``QInputDialog``/``QFileDialog``, monkeypatched here).
"""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QInputDialog

from asymmetry.core.project.profiles import BackgroundPolicy
from asymmetry.gui.windows.grouping.background_section import (
    BackgroundReferenceRunCandidate,
    BackgroundSectionWidget,
    background_status_text,
)


def _candidates() -> list[BackgroundReferenceRunCandidate]:
    return [BackgroundReferenceRunCandidate(9001, "Run 9001 (run 9001)", "/tmp/x.nxs", 1000.0)]


def test_configure_seeds_mode_without_emitting(qapp: QApplication) -> None:
    section = BackgroundSectionWidget(_candidates)
    fired: list[int] = []
    section.changed.connect(lambda: fired.append(1))
    section.configure(
        available_modes=("range", "tail_fit"),
        has_fixed_values=False,
        mode="range",
        background_run_payload=None,
    )
    assert section.mode() == "range"
    assert not fired  # seeding never emits


def test_mode_change_emits_changed(qapp: QApplication) -> None:
    section = BackgroundSectionWidget(_candidates)
    section.configure(
        available_modes=("range", "tail_fit"),
        has_fixed_values=False,
        mode="none",
        background_run_payload=None,
    )
    fired: list[int] = []
    section.changed.connect(lambda: fired.append(1))
    idx = section._mode_combo.findData("range")
    section._mode_combo.setCurrentIndex(idx)
    assert section.mode() == "range"
    assert fired  # a user edit notifies the owner


def test_reference_pick_selects_run_and_emits(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    section = BackgroundSectionWidget(_candidates)
    section.configure(
        available_modes=("reference_run",),
        has_fixed_values=False,
        mode="none",
        background_run_payload=None,
    )
    # The picker returns the candidate run.
    monkeypatch.setattr(QInputDialog, "getItem", lambda *a, **k: ("Run 9001 (run 9001)", True))
    fired: list[int] = []
    section.changed.connect(lambda: fired.append(1))
    idx = section._mode_combo.findData("reference_run")
    section._mode_combo.setCurrentIndex(idx)  # first entry prompts the picker
    assert section.mode() == "reference_run"
    payload = section.background_run_payload()
    assert payload is not None and payload["run_number"] == 9001
    assert fired


def test_cancelled_pick_falls_back_to_none(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    section = BackgroundSectionWidget(_candidates)
    section.configure(
        available_modes=("reference_run",),
        has_fixed_values=False,
        mode="none",
        background_run_payload=None,
    )
    monkeypatch.setattr(QInputDialog, "getItem", lambda *a, **k: ("", False))
    idx = section._mode_combo.findData("reference_run")
    section._mode_combo.setCurrentIndex(idx)
    assert section.mode() == "none"  # a cancelled pick resets the mode


def test_status_text_helper() -> None:
    assert background_status_text(BackgroundPolicy(mode="none")) == "Background: none"
    assert "tail fit" in background_status_text(BackgroundPolicy(mode="tail_fit")).lower()
    ref = BackgroundPolicy(mode="reference_run", details={"background_run": {"run_number": 42}})
    assert "42" in background_status_text(ref)


def test_explicit_separate_intervals_are_retained_and_editable(qapp, monkeypatch):
    section = BackgroundSectionWidget(_candidates)
    details = {"background_ranges": {"forward": [1, 71], "backward": [10, 60]}}
    section.configure(
        available_modes=("range",),
        has_fixed_values=False,
        mode="range",
        background_run_payload=None,
        details=details,
    )
    section._set_mode("none")
    section._set_mode("range")
    assert section.details() == details
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("2,72; 11,61", True))
    section._edit_range_btn.click()
    assert section.details()["background_ranges"] == [[2, 72], [11, 61]]
    assert "explicit range" in section._status_label.text()


def test_cancelled_interval_edit_preserves_authored_bounds(qapp, monkeypatch):
    section = BackgroundSectionWidget(_candidates)
    details = {"background_range": [0, 0]}
    section.configure(
        available_modes=("range",),
        has_fixed_values=False,
        mode="range",
        background_run_payload=None,
        details=details,
    )
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("", False))
    section._edit_range_btn.click()
    assert section.details() == details


def test_invalid_loaded_interval_can_be_replaced(qapp, monkeypatch):
    section = BackgroundSectionWidget(_candidates)
    section.configure(
        available_modes=("range",),
        has_fixed_values=False,
        mode="range",
        background_run_payload=None,
        details={"background_range": [4, 1]},
    )
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("1,4; 2,5", True))
    section._edit_range_btn.click()
    assert section.details() == {"background_ranges": [[1, 4], [2, 5]]}


def test_separate_aliases_are_described_as_explicit():
    policy = BackgroundPolicy(
        mode="range",
        details={"background_forward_range": [1, 71], "background_backward_range": [10, 60]},
    )
    assert background_status_text(policy) == "Background: explicit range [[1, 71], [10, 60]]"


def test_selected_green_period_is_persisted_and_displayed(qapp, monkeypatch):
    section = BackgroundSectionWidget(
        lambda: [BackgroundReferenceRunCandidate(42, "42/2", "reference.nxs", 75, period_number=2)]
    )
    section.configure(
        available_modes=("reference_run",),
        has_fixed_values=False,
        mode="none",
        background_run_payload=None,
    )
    monkeypatch.setattr(QInputDialog, "getItem", lambda *a, **k: ("42/2", True))
    section._mode_combo.setCurrentIndex(section._mode_combo.findData("reference_run"))
    assert section.background_run_payload() == {
        "run_number": 42,
        "source_file": "reference.nxs",
        "period_number": 2,
        "good_frames_reference": 75,
    }
    assert "period 2" in section._status_label.text()


def test_reference_candidates_keep_same_number_other_sources_and_periods(qapp):
    from asymmetry.core.io import select_period
    from asymmetry.gui.windows.grouping.dialog import GroupingDialog
    from tests.io.test_period_selection import _combined_two_period

    combined = _combined_two_period()
    red, green = [select_period(combined, period) for period in (1, 2)]
    other = select_period(combined, 1)
    other.run.source_file = "different.nxs"
    dialog = GroupingDialog([red, green, other], selected_run_number=red.run_number)
    candidates = dialog._background_reference_candidates()
    assert {(candidate.source_file, candidate.period_number) for candidate in candidates} == {
        ("synthetic.nxs", 2),
        ("different.nxs", 1),
    }
    dialog._clear_dirty()
    dialog.close()
    dialog.deleteLater()


@pytest.mark.parametrize(
    "mode,expected", [("green", {1}), ("green_plus_red", set()), ("green_minus_red", set())]
)
def test_reference_picker_uses_draft_sample_periods(qapp, mode, expected):
    from asymmetry.core.io import select_period
    from asymmetry.gui.windows.grouping.dialog import GroupingDialog
    from tests.io.test_period_selection import _combined_two_period

    combined = _combined_two_period()
    green = select_period(combined, 2)
    dialog = GroupingDialog([combined, green], selected_run_number=combined.run_number)
    dialog._set_period_mode(mode)
    assert {
        candidate.period_number for candidate in dialog._background_reference_candidates()
    } == expected
    dialog._clear_dirty()
    dialog.close()
    dialog.deleteLater()


def test_candidate_and_sample_exposure_snapshots_follow_selected_period(qapp):
    from asymmetry.gui.windows.grouping.dialog import GroupingDialog
    from tests.io.test_period_selection import _combined_two_period

    sample, reference = _combined_two_period(), _combined_two_period()
    reference.run.source_file = "another.nxs"
    reference.run.grouping["period_mode"] = "green"
    dialog = GroupingDialog([sample, reference], selected_run_number=sample.run_number)
    candidate = next(
        c for c in dialog._background_reference_candidates() if c.source_file == "another.nxs"
    )
    assert candidate.period_number == 2
    assert candidate.good_frames == 20
    dialog._background_section.configure(
        available_modes=("reference_run",),
        has_fixed_values=False,
        mode="reference_run",
        background_run_payload={
            "source_file": "another.nxs",
            "run_number": 12345,
            "period_number": 2,
            "good_frames_reference": 20,
        },
    )
    dialog._set_period_mode("green")
    dialog._on_background_changed()
    assert dialog._background_run_payload["good_frames_sample"] == 20
    dialog._clear_dirty()
    dialog.close()
    dialog.deleteLater()
