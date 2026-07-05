"""Standalone unit tests for the window-agnostic RangeCard widget."""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from asymmetry.gui.widgets.range_card import RangeCard, RangeCardView  # noqa: E402


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _view(**overrides) -> RangeCardView:
    base = dict(
        idx=0,
        title="Range 1",
        swatch_color="#1f4d8a",
        bounds_text="[12–88 K]",
        formula="A_1*exp(-Lambda*t)",
        status="not_run",
        status_chip_html="",
        status_tooltip="",
        can_remove=True,
        show_run=False,
    )
    base.update(overrides)
    return RangeCardView(**base)


def _press(widget) -> None:
    event = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        widget.rect().center().toPointF(),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    widget.mousePressEvent(event)


def test_set_state_renders_title_bounds_chip(qapp: QApplication) -> None:
    card = RangeCard(0)
    view = _view(
        title="Range 2",
        bounds_text="[12–40] ∪ [55–88] K",
        status="success",
        status_chip_html='<span style="color:green;">good</span>',
        status_tooltip="chi2/nu = 1.02",
    )
    card.set_state(view)

    assert card._title_label.text() == "Range 2"
    assert card._bounds_label.text() == "[12–40] ∪ [55–88] K"
    assert card._chip_label.text() == '<span style="color:green;">good</span>'
    assert card._chip_label.toolTip() == "chi2/nu = 1.02"


def test_title_uses_primary_emphasis(qapp: QApplication) -> None:
    """Line 1's title should read as the card's primary label: an explicit
    TEXT-token colour + heavier weight, distinct from the muted bounds label
    (which stays TEXT_MUTED, unweighted) — the status chip remains the only
    other strong colour on the line."""
    from asymmetry.gui.styles import tokens

    card = RangeCard(0)
    card.set_state(_view(title="Range 2"))

    title_style = card._title_label.styleSheet()
    bounds_style = card._bounds_label.styleSheet()

    assert title_style != ""
    assert tokens.TEXT in title_style
    assert "font-weight" in title_style
    assert title_style != bounds_style


def test_active_toggles_style(qapp: QApplication) -> None:
    card = RangeCard(0)
    card.set_active(True)
    active_style = card._surface.styleSheet()
    card.set_active(False)
    unselected_style = card._surface.styleSheet()

    assert active_style != unselected_style
    assert active_style != ""
    assert unselected_style != ""


def test_surface_is_frame_not_button(qapp: QApplication) -> None:
    """Regression: the card surface must be a QFrame, not a QPushButton — a
    button sizes to its own text hint and collapses the two content rows into a
    squashed ~single-line card."""
    from PySide6.QtWidgets import QFrame, QPushButton

    card = RangeCard(0)
    assert isinstance(card._surface, QFrame)
    assert not isinstance(card._surface, QPushButton)


def test_active_card_not_squashed(qapp: QApplication) -> None:
    """An active card (two content rows visible) reports a height tall enough for
    both lines — guards against the QPushButton-container squash regression."""
    card = RangeCard(0)
    card.set_state(_view(show_run=True))
    # Two stacked rows + margins: comfortably taller than a single-line control.
    assert card.sizeHint().height() >= 44


def test_show_run_controls_visibility(qapp: QApplication) -> None:
    card = RangeCard(0)

    card.set_state(_view(show_run=True))
    assert card._line2.isVisibleTo(card)
    assert card._run_button.isVisibleTo(card)
    assert card._edit_model_button.isVisibleTo(card)
    assert card._exclude_button.isVisibleTo(card)

    card.set_state(_view(show_run=False))
    assert not card._line2.isVisibleTo(card)


def test_run_button_emits_run_requested(qapp: QApplication) -> None:
    card = RangeCard(3)
    card.set_state(_view(idx=3, show_run=True))

    received = []
    card.run_requested.connect(received.append)
    card._run_button.click()

    assert received == [3]


def test_action_buttons_emit_signals(qapp: QApplication) -> None:
    card = RangeCard(2)
    card.set_state(_view(idx=2, show_run=True, can_remove=True))

    edit_model_received = []
    exclude_received = []
    remove_received = []
    card.edit_model_requested.connect(edit_model_received.append)
    card.exclude_requested.connect(exclude_received.append)
    card.remove_requested.connect(remove_received.append)

    card._edit_model_button.click()
    card._exclude_button.click()
    card._remove_button.click()

    assert edit_model_received == [2]
    assert exclude_received == [2]
    assert remove_received == [2]


def test_no_edit_params_action(qapp: QApplication) -> None:
    """The redundant "Edit Params" overflow action + signal were dropped: the
    card IS the selector, so "select this card" replaces it."""
    card = RangeCard(0)
    assert not hasattr(card, "_act_edit_params")
    assert not hasattr(card, "edit_params_requested")


def test_remove_hidden_when_cannot_remove(qapp: QApplication) -> None:
    card = RangeCard(0)

    card.set_state(_view(can_remove=False, show_run=True))
    assert not card._remove_button.isVisibleTo(card)

    card.set_state(_view(can_remove=True, show_run=True))
    assert card._remove_button.isVisibleTo(card)


def test_set_enabled_toggles_controls(qapp: QApplication) -> None:
    card = RangeCard(0)
    card.set_state(_view(show_run=True))

    card.set_enabled(False)
    assert not card._run_button.isEnabled()
    assert not card._edit_model_button.isEnabled()
    assert not card._exclude_button.isEnabled()
    assert not card._remove_button.isEnabled()

    card.set_enabled(True)
    assert card._run_button.isEnabled()
    assert card._edit_model_button.isEnabled()
    assert card._exclude_button.isEnabled()
    assert card._remove_button.isEnabled()


def test_card_click_emits_selected(qapp: QApplication) -> None:
    card = RangeCard(5)
    card.set_state(_view(idx=5))

    received = []
    card.selected.connect(received.append)
    _press(card._surface)

    assert received == [5]
