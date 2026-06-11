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


def register_matplotlib_fonts() -> None:
    """Register the bundled IBM Plex Mono TTFs with matplotlib's font manager.

    Idempotent and Qt-free, so plot-styling helpers can call it lazily —
    tick/legend text resolves the face even when a figure is created without
    the full application startup (tests, scripts).
    """
    try:
        from matplotlib import font_manager
    except ImportError:
        return
    for ttf in sorted(_FONTS_DIR.glob("IBMPlexMono-*.ttf")):
        font_manager.fontManager.addfont(str(ttf))


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
    except ImportError:
        return

    register_matplotlib_fonts()

    from asymmetry.gui.styles import tokens

    mpl.rcParams.update(
        {
            # Font family — sans-serif to match the Qt UI stack.
            "font.family": "sans-serif",
            # Italic symbols inside $...$ use a sans-serif math set so they
            # blend with surrounding roman label text.
            "mathtext.fontset": "dejavusans",
            # Sizes/colours matching the design-handoff plot grammar (the
            # per-axes treatment in styles/plots.py sets the same values
            # explicitly; these cover text created outside style_axes).
            "font.size": 11,
            "axes.labelsize": 10,
            "axes.labelcolor": tokens.PLOT_TICK_LABEL,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.color": tokens.PLOT_TICK_MARK,
            "ytick.color": tokens.PLOT_TICK_MARK,
            "xtick.labelcolor": tokens.PLOT_TICK_LABEL,
            "ytick.labelcolor": tokens.PLOT_TICK_LABEL,
            "legend.fontsize": 9,
            "axes.titlesize": 10,
            "axes.titlecolor": tokens.TEXT,
        }
    )
