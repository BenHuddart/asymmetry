"""Tests for the shared FitRunControls builder/holder widget."""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget  # noqa: E402

from asymmetry.gui.widgets.fit_run_controls import FitRunControls  # noqa: E402


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_default_construction_matches_fit_tab_style(qapp) -> None:
    """Default label/tooltip/hidden state matches the fit-tab Stop button."""
    controls = FitRunControls(button_label="Stop", tooltip="Cancel the running fit.")
    assert controls.button.text() == "Stop"
    assert controls.button.toolTip() == "Cancel the running fit."
    # A never-shown, unparented QPushButton reports isHidden()==True regardless
    # of whether .hide() was called explicitly, so parent it into a container
    # that would otherwise make an un-hidden child visible.
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.addWidget(controls.button)
    container.show()
    assert controls.button.isHidden()
    # No progress requested: both attributes stay None.
    assert controls.progress_label is None
    assert controls.progress_bar is None


def test_hidden_default_true_calls_hide(qapp) -> None:
    """hidden defaults to True: the button starts explicitly hidden, not just
    unshown, so it stays hidden even once parented into a visible container."""
    controls = FitRunControls(button_label="Stop")
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.addWidget(controls.button)
    container.show()
    assert controls.button.isHidden()


def test_on_cancel_is_wired_to_button_clicked(qapp) -> None:
    calls = []
    controls = FitRunControls(button_label="Stop", on_cancel=lambda: calls.append(1))
    controls.button.click()
    assert calls == [1]


def test_no_tooltip_when_not_given(qapp) -> None:
    controls = FitRunControls(button_label="Cancel")
    assert controls.button.toolTip() == ""


def test_hidden_false_starts_visible(qapp) -> None:
    """MaxEnt's Cancel button sits fixed in the footer, visible-but-disabled:
    hidden=False must skip the .hide() call so parenting into a shown
    container leaves the button visible."""
    controls = FitRunControls(button_label="Cancel", hidden=False)
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.addWidget(controls.button)
    container.show()
    assert not controls.button.isHidden()


def test_with_progress_builds_bar_and_label(qapp) -> None:
    controls = FitRunControls(button_label="Cancel", hidden=False, with_progress=True)
    assert controls.progress_label is not None
    assert controls.progress_label.wordWrap()
    assert controls.progress_bar is not None
    assert controls.progress_bar.isVisible() is False
    assert controls.progress_bar.minimum() == 0
    assert controls.progress_bar.maximum() == 1
    assert controls.progress_bar.value() == 0


def test_set_progress_updates_bar_and_escapes_label(qapp) -> None:
    controls = FitRunControls(button_label="Cancel", with_progress=True)
    controls.set_progress(3, 10, "cycle <3>")
    assert controls.progress_bar.minimum() == 0
    assert controls.progress_bar.maximum() == 10
    assert controls.progress_bar.value() == 3
    # HTML-escaped so a literal "<"/">" in a worker message renders as text,
    # matching MaxEnt's existing html.escape(...) behavior.
    assert controls.progress_label.text() == "cycle &lt;3&gt;"


def test_set_indeterminate_toggles_range(qapp) -> None:
    controls = FitRunControls(button_label="Cancel", with_progress=True)
    controls.set_indeterminate(True)
    assert controls.progress_bar.minimum() == 0
    assert controls.progress_bar.maximum() == 0
    controls.set_indeterminate(False)
    assert controls.progress_bar.minimum() == 0
    assert controls.progress_bar.maximum() == 1
    assert controls.progress_bar.value() == 0


def test_without_progress_set_progress_and_indeterminate_are_noop(qapp) -> None:
    controls = FitRunControls(button_label="Stop")
    # Should not raise even though progress_bar/progress_label are None.
    controls.set_progress(1, 2, "message")
    controls.set_indeterminate(True)
