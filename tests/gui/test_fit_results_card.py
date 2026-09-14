"""Standalone unit tests for the fit tabs' results card."""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QPushButton  # type: ignore

from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.widgets import VERDICT_CHIP_OBJECT_NAME
from asymmetry.gui.widgets.fit_results_card import (
    SURFACE_OBJECT_NAME,
    FitCardSummary,
    FitResultsCard,
    MemberChip,
)

_ACTIONS = (
    ("Diagnostic…", "Compare the fit's pulls against a normal distribution."),
    ("Send to Batch →", "Seed the Batch tab with this model."),
)

_GOOD = (tokens.SUCCESS_BG, tokens.SUCCESS_BORDER, tokens.OK)
_POOR = (tokens.ERROR_BG, tokens.ERROR, tokens.ERROR)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _card() -> FitResultsCard:
    return FitResultsCard(actions=_ACTIONS)


def _summary(**overrides) -> FitCardSummary:
    fields = {
        "tag": "Fit ✓",
        "tone": "ok",
        "headline": "converged",
        "meta": "ndof 595 · npar 5",
        "detail_html": "χ²/ν 0.9794 · good",
    }
    fields.update(overrides)
    return FitCardSummary(**fields)


def _member_chips(card: FitResultsCard) -> list[QPushButton]:
    layout = card._members_layout
    return [layout.itemAt(index).widget() for index in range(layout.count())]


def test_card_builds_one_segmented_button_per_action(qapp: QApplication) -> None:
    card = _card()

    buttons = [card._actions[label] for label, _ in _ACTIONS]
    assert [button.text() for button in buttons] == ["Diagnostic…", "Send to Batch →"]
    assert buttons[0].toolTip() == _ACTIONS[0][1]
    assert card._surface.objectName() == SURFACE_OBJECT_NAME
    card.deleteLater()


def test_card_action_signal_carries_the_label(qapp: QApplication) -> None:
    card = _card()
    triggered: list[str] = []
    card.action_triggered.connect(triggered.append)

    card._actions["Diagnostic…"].click()
    card._actions["Send to Batch →"].click()

    assert triggered == ["Diagnostic…", "Send to Batch →"]
    card.deleteLater()


def test_card_set_action_enabled_also_replaces_the_tooltip(qapp: QApplication) -> None:
    card = _card()

    card.set_action_enabled("Diagnostic…", False, "Run a fit first.")

    button = card._actions["Diagnostic…"]
    assert not button.isEnabled()
    assert button.toolTip() == "Run a fit first."

    card.set_action_enabled("Diagnostic…", True)

    assert button.isEnabled()
    # No tooltip given: the disabled reason it was carrying stands.
    assert button.toolTip() == "Run a fit first."
    card.deleteLater()


def test_card_set_action_enabled_rejects_an_unknown_label(qapp: QApplication) -> None:
    card = _card()

    with pytest.raises(KeyError):
        card.set_action_enabled("Trends →", True)
    card.deleteLater()


def test_card_starts_with_the_no_fit_tag_and_no_members(qapp: QApplication) -> None:
    card = _card()

    assert card._tag.text() == "No fit yet"
    assert card._tag.objectName() == VERDICT_CHIP_OBJECT_NAME
    assert tokens.SURFACE_ALT in card._tag.styleSheet()
    assert not card._members.isVisibleTo(card)
    card.deleteLater()


def test_card_summary_renders_the_tag_headline_meta_and_detail(qapp: QApplication) -> None:
    card = _card()

    card.set_summary(_summary())

    assert card._tag.text() == "Fit ✓"
    assert tokens.SUCCESS_BG in card._tag.styleSheet()
    assert card._headline.text() == "converged"
    assert card._meta.text() == "ndof 595 · npar 5"
    assert card._detail.text() == "χ²/ν 0.9794 · good"
    card.deleteLater()


@pytest.mark.parametrize(
    ("tone", "colour"),
    [
        ("ok", tokens.SUCCESS_BG),
        ("warn", tokens.WARN_BANNER_BG),
        ("error", tokens.ERROR_BG),
        ("neutral", tokens.SURFACE_ALT),
    ],
)
def test_card_tag_wears_its_tone_colours(qapp: QApplication, tone: str, colour: str) -> None:
    card = _card()

    card.set_summary(_summary(tag="Fit ⚠", tone=tone))

    assert colour in card._tag.styleSheet()
    card.deleteLater()


def test_card_members_become_clickable_verdict_chips(qapp: QApplication) -> None:
    card = _card()
    requested: list[int] = []
    card.member_requested.connect(requested.append)

    card.set_summary(
        _summary(
            members=(
                MemberChip(run=3001, text="3001 ✓ 0.98", colours=_GOOD, tooltip="converged"),
                MemberChip(run=3003, text="3003 ⚠ 1.9", colours=_POOR, tooltip="large_rel_err"),
            )
        )
    )

    chips = _member_chips(card)
    assert [chip.text() for chip in chips] == ["3001 ✓ 0.98", "3003 ⚠ 1.9"]
    assert [chip.objectName() for chip in chips] == [VERDICT_CHIP_OBJECT_NAME] * 2
    assert chips[1].toolTip() == "large_rel_err"
    assert tokens.ERROR_BG in chips[1].styleSheet()
    assert card._members.isVisibleTo(card)

    chips[1].click()

    assert requested == [3003]
    card.deleteLater()


def test_card_message_clears_the_members_and_the_meta(qapp: QApplication) -> None:
    card = _card()
    card.set_summary(
        _summary(members=(MemberChip(run=3001, text="3001 ✓ 0.98", colours=_GOOD, tooltip=""),))
    )

    card.set_message("Fitting…", tag="Fitting")

    assert card._tag.text() == "Fitting"
    assert tokens.SURFACE_ALT in card._tag.styleSheet()
    assert card._detail.text() == "Fitting…"
    assert card._headline.text() == ""
    assert card._meta.text() == ""
    assert not card._meta.isVisibleTo(card)
    assert _member_chips(card) == []
    assert not card._members.isVisibleTo(card)
    card.deleteLater()


def test_card_meta_tag_takes_the_slot_until_it_is_cleared(qapp: QApplication) -> None:
    card = _card()
    card.set_summary(_summary())

    card.set_meta_tag("seeds from 3001", "Parameter values carried forward from run 3001")

    assert card._meta_tag.isVisibleTo(card)
    assert card._meta_tag.toolTip() == "Parameter values carried forward from run 3001"
    # One read-out at a time: the tag displaces the summary's mono meta.
    assert not card._meta.isVisibleTo(card)

    card.set_meta_tag(None)

    assert not card._meta_tag.isVisibleTo(card)
    assert card._meta.isVisibleTo(card)
    card.deleteLater()


def test_content_html_reports_what_the_card_says(qapp: QApplication) -> None:
    """The fit tabs persist this line and replay it through set_message."""
    card = _card()
    assert card.content_html() == ""

    card.set_message("<b>Fit failed:</b> singular matrix", tag="Error")
    assert card.content_html() == "<b>Fit failed:</b> singular matrix"

    card.set_summary(_summary())
    # A summary folds its tag and headline in, so a restored read-out still
    # says how the fit went and not only what its statistics were.
    assert card.content_html() == "<b>Fit ✓</b> converged<br>χ²/ν 0.9794 · good"

    card.set_message(card.content_html())
    assert card._detail.text() == "<b>Fit ✓</b> converged<br>χ²/ν 0.9794 · good"
    card.deleteLater()


def test_a_message_can_carry_the_tone_of_the_outcome_it_replays(qapp: QApplication) -> None:
    """A saved read-out goes back under its own tag, not the neutral placeholder."""
    card = _card()

    card.set_message("χ²/ν 0.9794 · good", tag="Fit ✓", tone="ok")

    assert card.tag_text() == "Fit ✓"
    assert tokens.SUCCESS_BG in card._tag.styleSheet()
    # Still a message: the completed-fit read-outs stay away.
    assert card._headline.text() == ""
    assert card._meta.text() == ""
    assert card.content_html() == "χ²/ν 0.9794 · good"

    card.set_message("No fit performed yet")
    assert card.tag_text() == "No fit yet"
    assert tokens.SURFACE_ALT in card._tag.styleSheet()
    card.deleteLater()


def test_the_detail_line_carries_a_tooltip_that_the_next_message_drops(
    qapp: QApplication,
) -> None:
    """The χ² band behind a verdict chip has no room on the line; it hovers.

    Every ``set_message`` sets the hover text, so a later read-out can never
    inherit the explanation of a fit that is no longer on the card.
    """
    card = _card()

    card.set_summary(
        FitCardSummary(
            tag="Fit ✓",
            tone="ok",
            headline="converged",
            meta="",
            detail_html="χ²/ν 0.9794 · good",
            detail_tooltip="good fit (band 0.84–1.18 at 95 %)",
        )
    )
    assert card._detail.toolTip() == "good fit (band 0.84–1.18 at 95 %)"

    card.set_message("Forward/backward fit", tag="Fit ✓", tone="ok", tooltip="band 0.9–1.1")
    assert card._detail.toolTip() == "band 0.9–1.1"

    card.set_message("Fitting…", tag="Fitting")
    assert card._detail.toolTip() == ""
    card.deleteLater()
