"""Tests for the LoadingOverlay busy widget."""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QWidget

from asymmetry.gui.widgets.loading_overlay import LoadingOverlay


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def target(qapp):
    widget = QWidget()
    widget.resize(400, 300)
    widget.show()
    yield widget
    widget.deleteLater()


class TestLifecycle:
    def test_starts_hidden(self, target):
        overlay = LoadingOverlay(target)
        assert not overlay.isVisible()

    def test_show_message_makes_visible_with_text(self, target):
        overlay = LoadingOverlay(target)
        overlay.show_message("Computing FFT for run 42…")
        assert overlay.isVisible()
        assert "run 42" in overlay._message.text()

    def test_hide_clears(self, target):
        overlay = LoadingOverlay(target)
        overlay.show_message("working…")
        overlay.hide()
        assert not overlay.isVisible()

    def test_hide_is_idempotent(self, target):
        overlay = LoadingOverlay(target)
        # Hiding when already hidden must not raise — the "always clears"
        # contract calls hide from success, empty and error paths alike.
        overlay.hide()
        overlay.hide()
        assert not overlay.isVisible()

    def test_covers_target_rect(self, target):
        overlay = LoadingOverlay(target)
        overlay.show_message("working…")
        assert overlay.size() == target.rect().size()

    def test_tracks_target_resize(self, target):
        overlay = LoadingOverlay(target)
        overlay.show_message("working…")
        target.resize(640, 480)
        QApplication.processEvents()
        assert overlay.size() == target.rect().size()

    def test_bar_is_indeterminate(self, target):
        overlay = LoadingOverlay(target)
        # range(0, 0) is Qt's marquee/busy mode.
        assert overlay._bar.minimum() == 0
        assert overlay._bar.maximum() == 0
