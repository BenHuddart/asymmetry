"""Standalone unit tests for ParameterCard, its stack, and the sparkline."""

from __future__ import annotations

import math
import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtCore import QMimeData, QPointF, QSize, Qt  # type: ignore
from PySide6.QtGui import QDropEvent  # type: ignore
from PySide6.QtWidgets import QApplication  # type: ignore

from asymmetry.gui.styles import tokens
from asymmetry.gui.widgets.parameter_card import (
    PARAMETER_CARD_MIME,
    ParameterCard,
    ParameterCardStack,
    sparkline_pixmap,
)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _stack(*names: str) -> ParameterCardStack:
    stack = ParameterCardStack()
    for name in names:
        stack.add_card(ParameterCard(name, name))
    return stack


# ── Card signals ─────────────────────────────────────────────────────────────


def test_card_signals_carry_the_parameter_name(qapp: QApplication) -> None:
    card = ParameterCard("lambda", "λ (µs⁻¹)")
    fits: list[str] = []
    logs: list[tuple[str, bool]] = []
    transforms: list[str] = []
    focuses: list[str] = []
    expansions: list[tuple[str, bool]] = []
    card.fit_requested.connect(fits.append)
    card.log_toggled.connect(lambda name, on: logs.append((name, on)))
    card.transform_menu_requested.connect(lambda name, _pos: transforms.append(name))
    card.focus_toggled.connect(focuses.append)
    card.expanded_changed.connect(lambda name, on: expansions.append((name, on)))

    card.fit_button.click()
    card.log_check.setChecked(True)
    card.transform_button.click()
    card.focus_button.click()
    card.set_expanded(False)

    assert fits == ["lambda"]
    assert logs == [("lambda", True)]
    assert transforms == ["lambda"]
    assert focuses == ["lambda"]
    assert expansions == [("lambda", False)]
    card.deleteLater()


def test_card_collapse_hides_body_and_controls_and_shows_sparkline(qapp: QApplication) -> None:
    card = ParameterCard("beta", "β")
    assert card.is_expanded()

    card.set_expanded(False)

    assert not card._body.isVisibleTo(card)
    assert not card.fit_button.isVisibleTo(card)
    assert not card.log_check.isVisibleTo(card)
    assert not card.transform_button.isVisibleTo(card)
    assert card._sparkline_label.isVisibleTo(card)

    card.set_expanded(True)

    assert card._body.isVisibleTo(card)
    assert card.fit_button.isVisibleTo(card)
    assert not card._sparkline_label.isVisibleTo(card)
    card.deleteLater()


def test_card_set_expanded_is_idempotent(qapp: QApplication) -> None:
    card = ParameterCard("beta", "β")
    seen: list[bool] = []
    card.expanded_changed.connect(lambda _name, on: seen.append(on))

    card.set_expanded(True)
    card.set_expanded(False)
    card.set_expanded(False)

    assert seen == [False]
    card.deleteLater()


def test_card_result_chip_hides_without_a_fit(qapp: QApplication) -> None:
    card = ParameterCard("lambda", "λ")

    card.set_result(None, "", None)

    assert not card.result_chip.isVisibleTo(card)
    assert card.result_chip.text() == ""
    card.deleteLater()


def test_card_result_chip_shows_the_value_tooltip_and_colours(qapp: QApplication) -> None:
    card = ParameterCard("lambda", "λ")
    tooltip = (
        "χ²ᵣ 0.89 · good fit (band 0.7–1.3 at 95 %)\nT꜀ = 35.8(5) K\nClick for the fit results."
    )

    card.set_result("χ²ᵣ 0.89", tooltip, (tokens.SUCCESS_BG, tokens.SUCCESS_BORDER, tokens.OK))

    assert card.result_chip.isVisibleTo(card)
    assert card.result_chip.text() == "χ²ᵣ 0.89"
    assert card.result_chip.toolTip() == tooltip
    assert tokens.SUCCESS_BG in card.result_chip.styleSheet()
    assert tokens.OK in card.result_chip.styleSheet()

    # The chip is one of the per-parameter controls, so it goes with them.
    card.set_expanded(False)
    assert not card.result_chip.isVisibleTo(card)
    card.deleteLater()


def test_card_result_chip_click_emits_results_requested(qapp: QApplication) -> None:
    card = ParameterCard("lambda", "λ")
    requested: list[str] = []
    card.results_requested.connect(requested.append)

    card.set_result("χ²ᵣ 1.2", "tip", (tokens.SURFACE_ALT, tokens.BORDER, tokens.TEXT_MUTED))
    card.result_chip.click()

    assert requested == ["lambda"]
    card.deleteLater()


def test_card_labels_and_focus_chrome(qapp: QApplication) -> None:
    card = ParameterCard("lambda", "λ", derived=True)

    card.set_fit_label("Fit ✓", "Model fit converged")
    card.set_transform_label("1/y")
    card.set_swatch(tokens.TRACE_BLUE)

    assert card.fit_button.text() == "Fit ✓"
    assert card.fit_button.toolTip() == "Model fit converged"
    assert card.transform_button.text() == "1/y"
    assert tokens.TRACE_BLUE in card._swatch.styleSheet()

    glyph = card.focus_button.text()
    card.set_focused(True)
    assert card.is_focused()
    assert tokens.ACCENT in card._surface.styleSheet()
    assert card.focus_button.text() != glyph
    # Focus is chrome only; the stack owns expansion.
    assert card.is_expanded()

    card.set_focused(False)
    assert not card.is_focused()
    assert card.focus_button.text() == glyph
    card.deleteLater()


def test_shared_tooltip_adds_a_badge_distinct_from_derived(qapp: QApplication) -> None:
    """The "Shared" badge (``docs/plans/joint-fit.md`` D5) is a sibling of the
    "derived" chip, not a replacement — a card can carry either, both, or
    neither."""
    plain = ParameterCard("A_1", "A₁")
    assert plain.shared_chip is None

    shared = ParameterCard(
        "A_bg", "A_bg", derived=True, shared_tooltip='Shared across joint fit "J1" with Series B.'
    )
    assert shared.shared_chip is not None
    assert "Shared" in shared.shared_chip.text()
    assert shared.shared_chip.toolTip() == 'Shared across joint fit "J1" with Series B.'

    plain.deleteLater()
    shared.deleteLater()


# ── Sparkline ────────────────────────────────────────────────────────────────


def test_sparkline_pixmap_has_the_requested_size(qapp: QApplication) -> None:
    size = QSize(80, 20)

    pixmap = sparkline_pixmap([1.0, 2.0, 3.0], [0.4, 0.6, 0.5], tokens.TRACE_BLUE, size)

    assert not pixmap.isNull()
    assert pixmap.size() == size


def test_sparkline_pixmap_skips_non_finite_points(qapp: QApplication) -> None:
    size = QSize(80, 20)
    xs = [1.0, 2.0, math.nan, 4.0]
    ys = [0.4, math.inf, 0.6, 0.5]

    with_holes = sparkline_pixmap(xs, ys, tokens.TRACE_BLUE, size)
    finite_only = sparkline_pixmap([1.0, 4.0], [0.4, 0.5], tokens.TRACE_BLUE, size)
    empty = sparkline_pixmap([math.nan], [math.nan], tokens.TRACE_BLUE, size)

    assert with_holes.toImage() == finite_only.toImage()
    assert not empty.isNull()
    assert empty.toImage() != finite_only.toImage()


def test_sparkline_pixmap_draws_a_flat_series(qapp: QApplication) -> None:
    flat = sparkline_pixmap([1.0, 2.0, 3.0], [0.5, 0.5, 0.5], tokens.TRACE_BLUE, QSize(80, 20))

    blank = sparkline_pixmap([], [], tokens.TRACE_BLUE, QSize(80, 20))
    assert flat.toImage() != blank.toImage()


# ── Stack: membership and order ──────────────────────────────────────────────


def test_stack_add_remove_keeps_visual_order(qapp: QApplication) -> None:
    stack = _stack("a", "b", "c")

    assert [card.name for card in stack.cards()] == ["a", "b", "c"]
    assert stack.card("b").name == "b"

    removed = stack.remove_card("b")

    assert removed.name == "b"
    assert removed.parent() is None
    assert [card.name for card in stack.cards()] == ["a", "c"]
    removed.deleteLater()
    stack.deleteLater()


def test_stack_move_card_reorders_and_emits(qapp: QApplication) -> None:
    stack = _stack("a", "b", "c")
    orders: list[list[str]] = []
    stack.order_changed.connect(orders.append)

    stack.move_card("a", 2)

    assert [card.name for card in stack.cards()] == ["b", "c", "a"]
    assert orders == [["b", "c", "a"]]

    stack.move_card("c", 0)

    assert [card.name for card in stack.cards()] == ["c", "b", "a"]
    assert orders[-1] == ["c", "b", "a"]
    stack.deleteLater()


def test_stack_set_order_ignores_unknown_and_trails_unnamed(qapp: QApplication) -> None:
    stack = _stack("a", "b", "c")

    stack.set_order(["c", "zzz", "a"])

    assert [card.name for card in stack.cards()] == ["c", "a", "b"]
    stack.deleteLater()


def test_stack_drop_reorders_by_drop_position(qapp: QApplication) -> None:
    stack = _stack("a", "b", "c")
    for index, card in enumerate(stack.cards()):
        card.setGeometry(0, index * 100, 200, 100)
    orders: list[list[str]] = []
    stack.order_changed.connect(orders.append)

    mime = QMimeData()
    mime.setData(PARAMETER_CARD_MIME, b"a")
    event = QDropEvent(
        QPointF(10.0, 290.0),
        Qt.DropAction.MoveAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    stack.dropEvent(event)

    assert [card.name for card in stack.cards()] == ["b", "c", "a"]
    assert orders == [["b", "c", "a"]]
    stack.deleteLater()


# ── Stack: focus ─────────────────────────────────────────────────────────────


def test_stack_focus_expands_one_and_restores_on_exit(qapp: QApplication) -> None:
    stack = _stack("a", "b", "c")
    stack.card("c").set_expanded(False)
    focuses: list[object] = []
    stack.focus_changed.connect(focuses.append)

    stack.set_focus("b")

    assert stack.focused() == "b"
    assert [card.is_expanded() for card in stack.cards()] == [False, True, False]
    assert [card.is_focused() for card in stack.cards()] == [False, True, False]

    stack.set_focus(None)

    assert stack.focused() is None
    assert [card.is_expanded() for card in stack.cards()] == [True, True, False]
    assert not any(card.is_focused() for card in stack.cards())
    assert focuses == ["b", None]
    stack.deleteLater()


def test_stack_focus_hop_restores_the_original_expansion(qapp: QApplication) -> None:
    stack = _stack("a", "b", "c")
    stack.card("a").set_expanded(False)

    stack.set_focus("b")
    stack.set_focus("c")
    stack.set_focus(None)

    assert [card.is_expanded() for card in stack.cards()] == [False, True, True]
    stack.deleteLater()


def test_card_focus_button_toggles_focus_through_the_stack(qapp: QApplication) -> None:
    stack = _stack("a", "b")

    stack.card("b").focus_button.click()
    assert stack.focused() == "b"

    stack.card("b").focus_button.click()
    assert stack.focused() is None
    stack.deleteLater()


def test_stack_focus_does_not_announce_expansion_changes(qapp: QApplication) -> None:
    stack = _stack("a", "b")
    seen: list[tuple[str, bool]] = []
    for each in stack.cards():
        each.expanded_changed.connect(lambda name, on: seen.append((name, on)))

    stack.set_focus("b")
    stack.set_focus(None)

    assert seen == []
    stack.deleteLater()
