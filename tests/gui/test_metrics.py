"""Tests for the font-metric-relative sizing helpers (styles/metrics.py)."""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtGui import QFont  # noqa: E402
from PySide6.QtWidgets import QApplication, QLineEdit  # noqa: E402

from asymmetry.gui.styles import metrics  # noqa: E402


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_char_width_monotonic_in_char_count(qapp) -> None:
    w5 = metrics.char_width(5)
    w10 = metrics.char_width(10)
    w20 = metrics.char_width(20)
    assert 0 < w5 < w10 < w20


def test_field_width_exceeds_char_width_by_padding(qapp) -> None:
    chars = 8
    assert metrics.field_width_for(chars) > metrics.char_width(chars)


def test_field_width_and_dialog_width_agree(qapp) -> None:
    assert metrics.dialog_width(12) == metrics.field_width_for(12)


def test_field_width_uses_widget_font(qapp) -> None:
    edit = QLineEdit()
    big = QFont(edit.font())
    big.setPointSizeF(big.pointSizeF() * 2)
    edit.setFont(big)
    # A field measured against a doubled font is wider than one measured against
    # the (smaller) application font.
    assert metrics.field_width_for(10, edit) > metrics.field_width_for(10)


def test_row_height_positive(qapp) -> None:
    assert metrics.row_height() > 0


def test_metrics_track_application_font_size(qapp) -> None:
    original = QApplication.font()
    try:
        small = QFont(original)
        small.setPointSizeF(8.0)
        QApplication.setFont(small)
        small_char = metrics.char_width(10)
        small_row = metrics.row_height()

        large = QFont(original)
        large.setPointSizeF(20.0)
        QApplication.setFont(large)
        large_char = metrics.char_width(10)
        large_row = metrics.row_height()

        assert large_char > small_char
        assert large_row > small_row
    finally:
        QApplication.setFont(original)


def test_explicit_font_beats_app_font(qapp) -> None:
    tiny = QFont()
    tiny.setPointSizeF(6.0)
    huge = QFont()
    huge.setPointSizeF(24.0)
    assert metrics.char_width(10, huge) > metrics.char_width(10, tiny)


def test_raises_without_application(monkeypatch) -> None:
    # Guard path: no QGuiApplication instance → clear error rather than crash.
    monkeypatch.setattr(metrics.QGuiApplication, "instance", staticmethod(lambda: None))
    with pytest.raises(RuntimeError, match="QGuiApplication"):
        metrics.char_width(5)
