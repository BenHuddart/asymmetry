"""Standalone tests for the RunProgress widget."""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QLabel, QPushButton

from asymmetry.core.fitting.wizard_narrative import TrailStep
from asymmetry.gui.panels.log_panel import LogPanel
from asymmetry.gui.widgets.panel_section import PanelSection
from asymmetry.gui.widgets.run_progress import RunProgress

_STEPS = (
    TrailStep("context", "Reading series conditions…", "context", ()),
    TrailStep("screening", "Screening each run independently…", "screening", ()),
)


def test_start_sets_the_header_and_streams_pending_steps(qapp: QApplication) -> None:
    progress = RunProgress()
    progress.start("Screening the series…", _STEPS)
    labels = [label.text() for label in progress.findChildren(QLabel)]
    assert "Screening the series…" in labels
    assert progress.trail.step_keys() == ("context", "screening")
    assert progress.trail.active_step_key() is None


def test_append_log_keeps_messages_without_timestamps(qapp: QApplication) -> None:
    progress = RunProgress()
    progress.start("Screening the series…", _STEPS)
    progress.append_log("Starting screening for 4 datasets.")
    progress.append_log("Screening family 1 of 37…")
    assert progress.log_text() == "Starting screening for 4 datasets.\nScreening family 1 of 37…"
    panel_text = progress.findChild(LogPanel).to_plain_text()
    assert "Screening family 1 of 37…" in panel_text


def test_start_clears_the_previous_run(qapp: QApplication) -> None:
    progress = RunProgress()
    progress.start("Screening the series…", _STEPS)
    progress.append_log("old message")
    progress.trail.set_status("old status")
    progress.start("Optimizing selected candidates…", _STEPS[:1])
    assert progress.log_text() == ""
    assert progress.findChild(LogPanel).entry_count() == 0
    assert progress.trail.step_keys() == ("context",)


def test_cancel_button_requests_cancel(qapp: QApplication) -> None:
    progress = RunProgress()
    fired: list[bool] = []
    progress.cancel_requested.connect(lambda: fired.append(True))
    cancel = next(b for b in progress.findChildren(QPushButton) if b.text() == "Cancel")
    cancel.click()
    assert fired == [True]


def test_live_log_starts_collapsed(qapp: QApplication) -> None:
    progress = RunProgress()
    section = progress.findChild(PanelSection)
    assert section.title() == "Live log"
    assert not section.isExpanded()


def test_finish_collapses_to_a_run_log_that_keeps_the_messages(qapp: QApplication) -> None:
    progress = RunProgress()
    progress.start("Screening the series…", _STEPS)
    progress.append_log("Starting screening for 4 datasets.")
    progress.finish()
    section = progress.findChild(PanelSection)
    assert section.title() == "Run log"
    assert progress.trail.isHidden()
    cancel = next(b for b in progress.findChildren(QPushButton) if b.text() == "Cancel")
    assert not cancel.isVisibleTo(progress)
    assert progress.log_text() == "Starting screening for 4 datasets."

    progress.start("Optimizing selected candidates…", _STEPS)
    assert section.title() == "Live log"
    assert cancel.isVisibleTo(progress)


def test_restore_log_replaces_the_messages_line_by_line(qapp: QApplication) -> None:
    progress = RunProgress()
    progress.append_log("old message")
    progress.restore_log("first\nsecond")
    assert progress.log_text() == "first\nsecond"
    assert progress.findChild(LogPanel).entry_count() == 2
