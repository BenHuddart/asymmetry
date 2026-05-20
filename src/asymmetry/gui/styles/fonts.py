"""IBM Plex Mono font registration and convenience builders."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication

_FONTS_DIR = Path(__file__).parent.parent.parent / "resources" / "fonts"
_FAMILY = "IBM Plex Mono"
_FALLBACKS = ["Menlo", "Consolas", "Liberation Mono"]


def register_bundled_fonts() -> None:
    """Register IBM Plex Mono TTFs from the bundled resources directory.

    Must be called after QApplication is created. Safe to call more than once;
    Qt deduplicates by font ID.
    """
    if not isinstance(QApplication.instance(), QApplication):
        return
    for ttf in sorted(_FONTS_DIR.glob("IBMPlexMono-*.ttf")):
        QFontDatabase.addApplicationFont(str(ttf))


def mono_font(point_size: float = 11.0, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    """Return a QFont for tabular numeric display using IBM Plex Mono with system fallbacks."""
    font = QFont([_FAMILY, *_FALLBACKS])
    font.setPointSizeF(point_size)
    font.setWeight(weight)
    return font
