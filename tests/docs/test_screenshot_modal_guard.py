"""A screenshot capture run must never be blocked by a modal message box.

Screenshots are captured with ``QT_QPA_PLATFORM=offscreen``, where nobody can
click a modal. Any ``QMessageBox`` that reaches its own event loop therefore
blocks the whole process until the capture watchdog hard-exits — and every
scenario alphabetically after it is silently never written. That is exactly what
happened to the published docs: ``GroupingDialog.closeEvent`` runs an
unsaved-changes guard (``QMessageBox.question("Discard changes", ...)``), so the
``alpha_calibration_dialog`` scenario wedged on ``dialog.close()`` and took the
rest of the run with it.

These tests pin the harness-level guard: inside
:func:`docs.screenshots.scenarios._base.auto_dismiss_modals` every message box
answers instantly with its default (or most dismissive) button, and the patches
are removed again on exit so ordinary GUI tests keep their real modals.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QMessageBox  # noqa: E402

from docs.screenshots.scenarios._base import auto_dismiss_modals  # noqa: E402

SB = QMessageBox.StandardButton


@pytest.mark.gui
def test_question_returns_default_button_without_blocking(qapp, capsys):
    """The exact call that hung capture: a discard guard defaulting to Cancel."""
    with auto_dismiss_modals():
        answer = QMessageBox.question(
            None,
            "Discard changes",
            "Discard uncommitted changes to profile 'Default (GPS)'?",
            SB.Discard | SB.Cancel,
            SB.Cancel,
        )

    assert answer == SB.Cancel
    out = capsys.readouterr().out
    assert "[screenshots] auto-dismissed modal: Discard changes" in out


@pytest.mark.gui
@pytest.mark.parametrize("kind", ["information", "warning", "critical"])
def test_notification_boxes_return_immediately(kind, qapp):
    with auto_dismiss_modals():
        answer = getattr(QMessageBox, kind)(None, "Heads up", "text")
    assert answer == SB.Ok


@pytest.mark.gui
def test_about_returns_immediately(qapp, capsys):
    """``about`` has no buttons to answer with, but it blocks just as hard.

    ``MainWindow._on_about`` reaches ``QMessageBox.about``, so a Help-menu
    scenario would wedge a capture exactly like the discard guard did.
    """
    with auto_dismiss_modals():
        assert QMessageBox.about(None, "About Asymmetry", "version 1.2.3") is None
        assert QMessageBox.aboutQt(None, "About Qt") is None

    out = capsys.readouterr().out
    assert "[screenshots] auto-dismissed modal: About Asymmetry [about] -> closed" in out
    assert "[screenshots] auto-dismissed modal: About Qt [aboutQt] -> closed" in out


@pytest.mark.gui
def test_question_without_default_picks_the_most_dismissive_button(qapp):
    """No default named: never confirm — answer as an Esc press would."""
    with auto_dismiss_modals():
        assert QMessageBox.question(None, "Save?", "text", SB.Save | SB.Cancel) == SB.Cancel
        assert QMessageBox.question(None, "Overwrite?", "text", SB.Yes | SB.No) == SB.No
        assert QMessageBox.question(None, "Note", "text", SB.Ok) == SB.Ok


@pytest.mark.gui
def test_message_box_exec_closes_itself(qapp):
    box = QMessageBox()
    box.setWindowTitle("Unsaved changes")
    box.setStandardButtons(SB.Save | SB.Discard | SB.Cancel)
    with auto_dismiss_modals():
        result = box.exec()
    assert result == int(SB.Cancel)
    assert not box.isVisible()


@pytest.mark.gui
def test_patches_are_restored_on_exit(qapp):
    """Only the capture run is patched; GUI tests keep the real modals."""
    before = {
        name: getattr(QMessageBox, name)
        for name in ("question", "information", "warning", "critical", "about", "aboutQt", "exec")
    }
    with auto_dismiss_modals():
        assert QMessageBox.question is not before["question"]
    for name, original in before.items():
        assert getattr(QMessageBox, name) is original, f"{name} was not restored"


@pytest.mark.gui
def test_qdialog_exec_is_not_patched(qapp):
    """Scenarios whose subject *is* a dialog must keep a working ``exec``."""
    from PySide6.QtWidgets import QDialog

    original = QDialog.exec
    with auto_dismiss_modals():
        assert QDialog.exec is original


def test_capture_driver_wraps_the_scenario_loop():
    """The guard is applied by the driver, not per scenario.

    Scenarios may override ``Scenario.capture`` wholesale (the α-calibration one
    does), so the only place that covers all of them is the capture loop.
    """
    from pathlib import Path

    source = Path(__file__).resolve().parents[2] / "docs" / "screenshots" / "capture.py"
    text = source.read_text(encoding="utf-8")
    assert "with auto_dismiss_modals():" in text
