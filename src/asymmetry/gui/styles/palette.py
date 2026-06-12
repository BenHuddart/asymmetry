"""Explicit BENCH QPalette and cross-platform style selection.

The bench look is defined entirely by our own stylesheet and palette. Without
an explicit style + palette the app renders on top of whatever the platform
provides ("macos" on Mac, "windowsvista" on Windows), so any widget surface
bench.qss does not cover looks different per platform, and an OS dark mode can
invert the palette underneath the light stylesheet. Forcing Fusion plus a
palette built from the BENCH tokens makes rendering identical everywhere.
"""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette

from asymmetry.gui.styles import tokens


def build_bench_palette() -> QPalette:
    """Return the light BENCH palette derived from the design tokens."""
    palette = QPalette()
    role = QPalette.ColorRole
    group = QPalette.ColorGroup

    window = QColor(tokens.BG)
    surface = QColor(tokens.SURFACE)
    surface_alt = QColor(tokens.SURFACE_ALT)
    surface_hi = QColor(tokens.SURFACE_HI)
    border = QColor(tokens.BORDER)
    border_strong = QColor(tokens.BORDER_STRONG)
    text = QColor(tokens.TEXT)
    text_muted = QColor(tokens.TEXT_MUTED)
    text_dim = QColor(tokens.TEXT_DIM)
    accent = QColor(tokens.ACCENT)
    accent_soft = QColor(tokens.ACCENT_SOFT)

    palette.setColor(role.Window, window)
    palette.setColor(role.WindowText, text)
    palette.setColor(role.Base, surface)
    palette.setColor(role.AlternateBase, surface_alt)
    palette.setColor(role.Text, text)
    palette.setColor(role.PlaceholderText, text_dim)
    palette.setColor(role.Button, surface)
    palette.setColor(role.ButtonText, text)
    palette.setColor(role.BrightText, QColor("#ffffff"))
    palette.setColor(role.ToolTipBase, surface)
    palette.setColor(role.ToolTipText, text)
    palette.setColor(role.Highlight, accent_soft)
    palette.setColor(role.HighlightedText, text)
    palette.setColor(role.Link, accent)

    # Bevel/frame shades Fusion derives 3D chrome from.
    palette.setColor(role.Light, QColor("#ffffff"))
    palette.setColor(role.Midlight, surface_hi)
    palette.setColor(role.Mid, border)
    palette.setColor(role.Dark, border_strong)
    palette.setColor(role.Shadow, QColor("#3a3c40"))

    for muted_role in (role.WindowText, role.Text, role.ButtonText):
        palette.setColor(group.Disabled, muted_role, text_dim)
    palette.setColor(group.Disabled, role.Base, surface_alt)
    palette.setColor(group.Disabled, role.Highlight, surface_hi)
    palette.setColor(group.Disabled, role.HighlightedText, text_muted)

    return palette


def apply_bench_style(app) -> None:
    """Force the Fusion style and the BENCH palette on the application."""
    if hasattr(app, "setStyle"):
        app.setStyle("Fusion")
    if hasattr(app, "setPalette"):
        app.setPalette(build_bench_palette())
