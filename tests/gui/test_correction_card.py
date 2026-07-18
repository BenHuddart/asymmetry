"""Tests for the grouping dialog's collapsible CorrectionCard widget.

The card is grouping-specific chrome (richer than the shared PanelSection): a
clickable header row carrying a live status summary, a warn-tint stale state
(the α card), and an accent compare indicator shown while the card's stage is
focused in the shared preview. Expansion is plain widget state — no QSettings.
"""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel

from asymmetry.gui.styles import tokens
from asymmetry.gui.windows.grouping.correction_card import CorrectionCard


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _card_with_body(title: str = "Deadtime") -> tuple[CorrectionCard, QLabel]:
    card = CorrectionCard(title)
    body = QLabel("body content")
    card.set_body(body)
    return card, body


def test_header_click_toggles_and_emits(qapp: QApplication) -> None:
    """A click anywhere on the header row toggles expansion and emits toggled."""
    card, _body = _card_with_body()
    card.show()
    assert card.is_expanded()  # constructor default

    states: list[bool] = []
    card.toggled.connect(states.append)

    QTest.mouseClick(card._header, Qt.MouseButton.LeftButton)
    assert not card.is_expanded()
    QTest.mouseClick(card._header, Qt.MouseButton.LeftButton)
    assert card.is_expanded()
    assert states == [False, True]

    # set_expanded emits only on a real change.
    card.set_expanded(True)
    assert states == [False, True]
    card.set_expanded(False)
    assert states == [False, True, False]


def test_collapsed_card_hides_body(qapp: QApplication) -> None:
    """Collapsing hides the body container (and everything wrapped in it)."""
    card, body = _card_with_body()
    card.show()
    assert not card._body.isHidden()
    assert body.isVisible()

    card.set_expanded(False)
    assert card._body.isHidden()
    assert not body.isVisible()
    # The disclosure arrow tracks the state.
    assert card._arrow.text() == "▸"
    card.set_expanded(True)
    assert card._arrow.text() == "▾"


def test_set_comparing_shows_indicator_and_accent(qapp: QApplication) -> None:
    """set_comparing shows the indicator text and accent-tints the header."""
    card, _body = _card_with_body()
    card.show()
    assert card.comparing_text() is None
    assert card._comparing_label.isHidden()
    assert tokens.SURFACE_ALT in card._header.styleSheet()
    assert tokens.ACCENT_SOFT not in card._header.styleSheet()

    card.set_comparing("comparing: without deadtime")
    assert card.comparing_text() == "comparing: without deadtime"
    assert not card._comparing_label.isHidden()
    assert card._comparing_label.text() == "comparing: without deadtime"
    header_style = card._header.styleSheet()
    assert tokens.ACCENT_SOFT in header_style
    assert f"border-left: 3px solid {tokens.ACCENT}" in header_style

    card.set_comparing(None)
    assert card.comparing_text() is None
    assert card._comparing_label.isHidden()
    assert tokens.ACCENT_SOFT not in card._header.styleSheet()
    assert tokens.SURFACE_ALT in card._header.styleSheet()


def test_set_stale_tints_title_and_status(qapp: QApplication) -> None:
    """set_stale warn-tints the title/status labels; clearing restores them.

    The status label paints itself (it elides), so its colour is asserted via
    ``pen_color()`` rather than a stylesheet string.
    """
    from PySide6.QtGui import QColor

    card, _body = _card_with_body("α (detector balance)")
    card.set_status("1.2692 · Diamagnetic (TF)")
    assert tokens.WARN not in card._title_label.styleSheet()
    assert card._status_label.pen_color() == QColor(tokens.TEXT_MUTED)

    card.set_stale(True)
    assert tokens.WARN in card._title_label.styleSheet()
    assert card._status_label.pen_color() == QColor(tokens.WARN)

    card.set_stale(False)
    assert tokens.WARN not in card._title_label.styleSheet()
    assert card._status_label.pen_color() == QColor(tokens.TEXT_MUTED)


def test_stage_identity_color_stripe_always_on(qapp: QApplication) -> None:
    """The stage-colour stripe is the card's identity, shown in every state.

    The comparing state only deepens the header fill to the stage's soft tint —
    the stripe (which matches the pipeline chip's outline colour) never leaves.
    """
    card = CorrectionCard("Deadtime", color=tokens.STAGE_DEADTIME, soft=tokens.STAGE_DEADTIME_SOFT)
    stripe = f"border-left: 3px solid {tokens.STAGE_DEADTIME}"
    assert stripe in card._header.styleSheet()
    assert tokens.STAGE_DEADTIME_SOFT not in card._header.styleSheet()

    card.set_comparing("comparing: without deadtime")
    assert stripe in card._header.styleSheet()
    assert tokens.STAGE_DEADTIME_SOFT in card._header.styleSheet()

    card.set_comparing(None)
    assert stripe in card._header.styleSheet()
    assert tokens.STAGE_DEADTIME_SOFT not in card._header.styleSheet()


def test_status_text_round_trips(qapp: QApplication) -> None:
    card, _body = _card_with_body("Background")
    assert card.status_text() == ""
    card.set_status("pre-t0 range")
    assert card.status_text() == "pre-t0 range"
    assert card.title() == "Background"
