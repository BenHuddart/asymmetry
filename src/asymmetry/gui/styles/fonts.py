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


def configure_plot_fonts() -> None:
    """Configure matplotlib rcParams to match the BENCH UI font stack.

    - Registers the bundled IBM Plex Mono TTFs with matplotlib's font manager
      so they are available for annotations and tick overrides.
    - Sets sans-serif as the default family (closest to the platform UI font).
    - Uses 'dejavusans' mathtext so $...$ italic symbols use a consistent
      sans-serif face that matches the surrounding label text.
    - Sizes are aligned with the BENCH type scale (11 pt body, 10 pt small).

    Call once during application startup, after QApplication is created.
    Safe to call multiple times.
    """
    try:
        import matplotlib as mpl
        from matplotlib import font_manager
    except ImportError:
        return

    # Register IBM Plex Mono with matplotlib so it resolves by name.
    for ttf in sorted(_FONTS_DIR.glob("IBMPlexMono-*.ttf")):
        font_manager.fontManager.addfont(str(ttf))

    mpl.rcParams.update(
        {
            # Font family — sans-serif to match the Qt UI stack.
            "font.family": "sans-serif",
            # Italic symbols inside $...$ use a sans-serif math set so they
            # blend with surrounding roman label text.
            "mathtext.fontset": "dejavusans",
            # Base sizes matching the BENCH type scale.
            "font.size": 11,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.titlesize": 11,
        }
    )
