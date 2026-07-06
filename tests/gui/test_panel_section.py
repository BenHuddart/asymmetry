"""Tests for the shared PanelSection titled-section widget."""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

from asymmetry.gui.styles.widgets import SECTION_HEADER_OBJECT_NAME  # noqa: E402
from asymmetry.gui.widgets.panel_section import PanelSection  # noqa: E402


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture
def settings(tmp_path):
    # A per-test IniFormat file rather than a shared native scope, so parallel
    # xdist workers cannot race on the same on-disk key.
    s = QSettings(str(tmp_path / "panel_section.ini"), QSettings.Format.IniFormat)
    s.clear()
    yield s
    s.clear()


def test_static_section_always_expanded(qapp) -> None:
    section = PanelSection("Model")
    assert section.isExpanded()
    assert section.title() == "Model"
    # The header uses the shared BENCH section-header label (uppercased, scoped
    # objectName) so the flat look matches make_section.
    header = section._header_label
    assert isinstance(header, QLabel)
    assert header.objectName() == SECTION_HEADER_OBJECT_NAME
    assert header.text() == "MODEL"
    # No disclosure arrow on a static section.
    assert section._arrow is None


def test_static_section_setexpanded_is_noop(qapp) -> None:
    section = PanelSection("Model")
    section.setExpanded(False)
    assert section.isExpanded()  # still expanded


def test_collapsible_starts_collapsed_by_default(qapp) -> None:
    section = PanelSection("Advanced", collapsible=True)
    assert not section.isExpanded()
    assert section._arrow is not None


def test_collapsible_expanded_true_starts_open(qapp) -> None:
    section = PanelSection("Advanced", collapsible=True, expanded=True)
    assert section.isExpanded()


def test_toggle_emits_and_flips_body_visibility(qapp) -> None:
    section = PanelSection("Advanced", collapsible=True)
    states: list[bool] = []
    section.toggled.connect(states.append)
    section._on_header_clicked()  # simulate whole-row click
    assert section.isExpanded()
    assert states == [True]
    assert section._body.isVisibleTo(section)
    section._on_header_clicked()
    assert not section.isExpanded()
    assert states == [True, False]


def test_real_click_on_header_and_child_toggles(qapp) -> None:
    """A real mouse press anywhere in the header row — including on the child
    title label — toggles the section. This is the load-bearing "whole header
    row is clickable" requirement: the plain child QLabels must not swallow the
    press, so it propagates up to the clickable row."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    section = PanelSection("Advanced", collapsible=True)
    QTest.mouseClick(section._header_row, Qt.MouseButton.LeftButton)
    assert section.isExpanded()
    # A click landing on the child header label (not the bare row) must still
    # reach the row and toggle it.
    QTest.mouseClick(section._header_label, Qt.MouseButton.LeftButton)
    assert not section.isExpanded()


def test_toggled_not_emitted_when_state_unchanged(qapp) -> None:
    section = PanelSection("Advanced", collapsible=True, expanded=True)
    states: list[bool] = []
    section.toggled.connect(states.append)
    section.setExpanded(True)  # already expanded
    assert states == []


def test_body_layout_add_widget_and_layout(qapp) -> None:
    from PySide6.QtWidgets import QHBoxLayout

    section = PanelSection("Model")
    child = QLabel("x")
    section.addWidget(child)
    section.addLayout(QHBoxLayout())
    assert section.body_layout.count() == 2
    assert section.contentLayout() is section.body_layout


def test_hint_hidden_when_none_shown_when_set(qapp) -> None:
    section = PanelSection("Model")
    assert section._hint_label.isHidden()
    section.set_hint("A muted description.")
    assert not section._hint_label.isHidden()
    assert section._hint_label.text() == "A muted description."
    section.set_hint(None)
    assert section._hint_label.isHidden()


def test_hint_constructor_shows_it(qapp) -> None:
    section = PanelSection("Model", hint="Inline hint")
    assert not section._hint_label.isHidden()
    assert section._hint_label.text() == "Inline hint"


def test_title_suffix_toggles_visibility(qapp) -> None:
    section = PanelSection("Exclusions", collapsible=True)
    assert section._suffix_label.isHidden()
    section.set_title_suffix("<b>3 exclusions</b>")
    assert not section._suffix_label.isHidden()
    assert section._suffix_label.text() == "<b>3 exclusions</b>"
    section.set_title_suffix(None)
    assert section._suffix_label.isHidden()


def test_persistence_round_trip(qapp, settings) -> None:
    key = "panels/advanced_expanded"
    # First section starts collapsed (default), then user expands it.
    first = PanelSection("Advanced", collapsible=True, settings_key=key, settings=settings)
    assert not first.isExpanded()
    first.setExpanded(True)
    # A freshly built section reads back the persisted expanded state, ignoring
    # its own expanded=False default.
    second = PanelSection(
        "Advanced", collapsible=True, expanded=False, settings_key=key, settings=settings
    )
    assert second.isExpanded()


def test_persisted_value_overrides_constructor_default(qapp, settings) -> None:
    key = "panels/collapse_me"
    settings.setValue(key, False)
    # Constructor asks for expanded=True but the persisted False wins.
    section = PanelSection(
        "Advanced", collapsible=True, expanded=True, settings_key=key, settings=settings
    )
    assert not section.isExpanded()
