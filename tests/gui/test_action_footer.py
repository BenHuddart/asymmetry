"""Tests for the shared ActionFooter pinned-footer widget."""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QLabel, QPushButton  # noqa: E402

from asymmetry.gui.widgets.action_footer import ActionFooter  # noqa: E402


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_add_primary_returns_styled_button(qapp) -> None:
    footer = ActionFooter()
    btn = footer.add_primary("Compute FFT")
    assert isinstance(btn, QPushButton)
    assert btn.text() == "Compute FFT"
    # Primary carries the accent QSS (accent-soft fill hue lives in the sheet).
    assert btn.styleSheet() != ""


def test_add_secondary_and_custom_widget(qapp) -> None:
    footer = ActionFooter()
    secondary = footer.add_secondary("Apply to selection")
    assert isinstance(secondary, QPushButton)
    assert secondary.styleSheet() == ""  # default styling
    cluster = QLabel("stepper")
    footer.add_widget(cluster)
    # Both the secondary button and the custom cluster live in the button area.
    assert footer._button_layout.count() == 2


def test_hint_hidden_by_default_and_toggles(qapp) -> None:
    footer = ActionFooter()
    assert footer._hint_label.isHidden()
    footer.set_hint("Background correction is inherited.")
    assert not footer._hint_label.isHidden()
    assert footer._hint_label.text() == "Background correction is inherited."
    footer.set_hint(None)
    assert footer._hint_label.isHidden()


def test_status_set_and_clear_accepts_rich_text(qapp) -> None:
    footer = ActionFooter()
    html = '<span style="color:#2a7a3f;font-weight:600;">Fit converged</span>'
    footer.set_status(html)
    assert "Fit converged" in footer._status_label.text()
    footer.set_status("plain text works too")
    assert footer._status_label.text() == "plain text works too"
    footer.clear_status()
    assert footer._status_label.text() == ""


def test_progress_hidden_by_default_and_toggles(qapp) -> None:
    footer = ActionFooter()
    assert footer._progress_row.isHidden()
    footer.show_progress("Reconstructing…")
    assert not footer._progress_row.isHidden()
    assert footer._progress_label.text() == "Reconstructing…"
    footer.hide_progress()
    assert footer._progress_row.isHidden()
    assert footer._progress_label.text() == ""
