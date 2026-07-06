"""F6 — design-token discipline & UI-scale completeness (P2-6, P1-4, P3-6).

These tests lock the single-source-of-truth invariants introduced in the
token-discipline pass:

* ``bench.qss`` is a token template — no raw hex — and renders cleanly.
* the type scale lives once in ``typography`` and the chrome QSS derives from it.
* UI scale reaches the chrome/value fonts (builders + the UIManager font scan)
  and the per-widget QSS metrics.
* the plot-trace palette is centralised in ``tokens`` / ``styles.plots``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_GUI_ROOT = Path(__file__).parent.parent.parent / "src" / "asymmetry" / "gui"
_BENCH_QSS = _GUI_ROOT / "styles" / "bench.qss"

_HEX = re.compile(r"#[0-9a-fA-F]{6}\b")
_PLACEHOLDER = re.compile(r"@([A-Za-z_][A-Za-z0-9_]*)@")


# ── bench.qss templating (P2-6) ───────────────────────────────────────────────


def test_bench_template_has_no_raw_hex() -> None:
    """The bench.qss template must carry no raw hex — only @TOKEN@ placeholders."""
    text = _BENCH_QSS.read_text(encoding="utf-8")
    assert not _HEX.findall(text), "bench.qss must reference tokens, not raw hex"


def test_rendered_stylesheet_has_no_unresolved_placeholders() -> None:
    from asymmetry.gui.styles.stylesheet import render_bench_stylesheet

    rendered = render_bench_stylesheet()
    assert rendered, "rendered stylesheet should be non-empty"
    assert not _PLACEHOLDER.findall(rendered), "every @TOKEN@ must be substituted"


def test_rendered_stylesheet_substitutes_token_values() -> None:
    from asymmetry.gui.styles import tokens
    from asymmetry.gui.styles.stylesheet import render_bench_stylesheet

    rendered = render_bench_stylesheet()
    assert tokens.ACCENT in rendered  # @ACCENT@ → progress chunk / tab underline
    assert tokens.BORDER in rendered
    assert tokens.SURFACE_ALT in rendered


def test_every_placeholder_names_a_real_token() -> None:
    from asymmetry.gui.styles import tokens
    from asymmetry.gui.styles.stylesheet import load_template

    names = set(_PLACEHOLDER.findall(load_template()))
    assert names, "template should contain placeholders"
    missing = [n for n in names if not isinstance(getattr(tokens, n, None), str)]
    assert not missing, f"bench.qss references unknown tokens: {missing}"


def test_substitute_tokens_raises_on_unknown_token() -> None:
    from asymmetry.gui.styles.stylesheet import substitute_tokens

    with pytest.raises(KeyError):
        substitute_tokens("QWidget { color: @NOT_A_TOKEN@; }")


def test_render_degrades_gracefully_when_template_missing(monkeypatch) -> None:
    """A load/substitution failure must degrade to '' (Fusion), never crash startup."""
    from asymmetry.gui.styles import stylesheet

    # Unknown-token template would raise inside substitute_tokens; the production
    # entry point swallows it so app startup is never taken down by chrome.
    monkeypatch.setattr(stylesheet, "load_template", lambda: "QWidget { color: @NOPE@; }")
    assert stylesheet.render_bench_stylesheet() == ""


def test_white_token_is_used_by_palette(qapp) -> None:
    """The WHITE token must be wired (not dead) — bright-text/light bevel roles."""
    from PySide6.QtGui import QColor, QPalette

    from asymmetry.gui.styles import tokens
    from asymmetry.gui.styles.palette import build_bench_palette

    palette = build_bench_palette()
    assert palette.color(QPalette.ColorRole.BrightText) == QColor(tokens.WHITE)
    assert palette.color(QPalette.ColorRole.Light) == QColor(tokens.WHITE)


# ── Type scale single source (P3-6) ──────────────────────────────────────────


def test_chrome_qss_font_size_derives_from_typography(qapp) -> None:
    """The chrome font-size in build_stylesheet must equal scaled SIZE_HEADER."""
    from asymmetry.gui.styles import typography
    from asymmetry.gui.ui_manager import UIManager

    # build_stylesheet never touches self, so an unbound call with self=None is
    # safe and avoids constructing a whole MainWindow for a pure-string check.
    qss = UIManager.build_stylesheet(None, 1.0)  # type: ignore[arg-type]
    assert f"{round(typography.SIZE_HEADER * 1.0, 2)}pt" in qss


def test_apply_param_table_style_uses_header_font(qapp) -> None:
    """apply_param_table_style must reuse header_font() (no inline 9.5pt dup)."""
    from PySide6.QtWidgets import QTableWidget

    from asymmetry.gui.styles.typography import SIZE_HEADER, header_font
    from asymmetry.gui.styles.widgets import apply_param_table_style

    # A prior MainWindow test can leave a global app stylesheet active; Qt merges
    # its QHeaderView::section font-size into widget.font(), masking the explicit
    # setFont. Clear it so the assertion reflects what apply_param_table_style set.
    saved = qapp.styleSheet()
    qapp.setStyleSheet("")
    try:
        table = QTableWidget(1, 1)
        apply_param_table_style(table)
        applied = table.horizontalHeader().font()
        expected = header_font()
        assert applied.pointSizeF() == pytest.approx(expected.pointSizeF())
        assert applied.weight() == expected.weight()  # DemiBold, from header_font()
        assert header_font().pointSizeF() == pytest.approx(SIZE_HEADER)
    finally:
        qapp.setStyleSheet(saved)


# ── UI-scale reaches chrome/value fonts (P1-4) ───────────────────────────────


def test_font_builders_track_ui_scale(qapp) -> None:
    from asymmetry.gui.styles.fonts import mono_font, set_ui_font_scale
    from asymmetry.gui.styles.typography import (
        SIZE_HEADER,
        SIZE_NUMERIC,
        header_font,
        section_label_font,
    )

    try:
        set_ui_font_scale(1.2)
        assert header_font().pointSizeF() == pytest.approx(SIZE_HEADER * 1.2)
        assert section_label_font().pointSizeF() == pytest.approx(SIZE_HEADER * 1.2)
        assert mono_font(SIZE_NUMERIC).pointSizeF() == pytest.approx(SIZE_NUMERIC * 1.2)
        set_ui_font_scale(0.8)
        assert header_font().pointSizeF() == pytest.approx(SIZE_HEADER * 0.8)
    finally:
        set_ui_font_scale(1.0)


def test_build_stylesheet_chrome_font_sizes_scale(qapp) -> None:
    from asymmetry.gui.ui_manager import UIManager

    def header_pt(qss: str) -> float:
        m = re.search(r"QHeaderView::section \{\s*font-size: ([0-9.]+)pt", qss)
        assert m, "QHeaderView::section font-size missing from build_stylesheet"
        return float(m.group(1))

    small = header_pt(UIManager.build_stylesheet(None, 0.8))  # type: ignore[arg-type]
    large = header_pt(UIManager.build_stylesheet(None, 1.2))  # type: ignore[arg-type]
    assert large > small
    # All three chrome selectors must carry a scaled size.
    qss = UIManager.build_stylesheet(None, 1.0)  # type: ignore[arg-type]
    for selector in ("QDockWidget::title", "QHeaderView::section", "QGroupBox::title"):
        assert re.search(rf"{re.escape(selector)} \{{\s*font-size:", qss)


@pytest.mark.gui
def test_ui_scale_qss_rescales_existing_section_header(qapp) -> None:
    """A live UI-scale change must re-scale an already-built section header (P1-4).

    Section headers set an explicit ``section_label_font()`` that ignores the
    application font; the scale QSS block's ``QLabel#benchSectionHeader``
    ``font-size`` rule re-scales them live (Qt merges the size onto the existing
    font, so the builder's weight/letter-spacing survive). This is the QSS half
    of the split that replaced the per-widget font-scan machinery.
    """
    from PySide6.QtCore import QSettings

    from asymmetry.gui.mainwindow import MainWindow
    from asymmetry.gui.styles.widgets import make_section_header
    from asymmetry.gui.ui_manager import UI_SCALE_SETTINGS_KEY

    QSettings().setValue(UI_SCALE_SETTINGS_KEY, 1.0)
    window = MainWindow()
    try:
        header = make_section_header("SECTION")
        header.setParent(window)
        window._ui_manager.set_ui_scale(1.0)
        header.ensurePolished()
        base = header.font().pointSizeF()
        window._ui_manager.set_ui_scale(1.2)
        header.ensurePolished()
        grown = header.font().pointSizeF()
        window._ui_manager.set_ui_scale(0.8)
        header.ensurePolished()
        shrunk = header.font().pointSizeF()
        assert grown > base > shrunk
        assert grown == pytest.approx(base * 1.2, rel=0.02)
    finally:
        window.close()


@pytest.mark.gui
def test_ui_scale_change_updates_app_font_and_stylesheet_once(qapp) -> None:
    """A live scale change sets the QApplication font and stylesheet exactly once each.

    The font-driven zoom replaced per-widget tracking: one app-font set, one
    app-stylesheet set. Guards against a regression that re-multiplies the font
    or appends a second scale block per apply.
    """
    from unittest.mock import patch

    from PySide6.QtCore import QSettings

    from asymmetry.gui.mainwindow import MainWindow
    from asymmetry.gui.ui_manager import UI_SCALE_SETTINGS_KEY

    QSettings().setValue(UI_SCALE_SETTINGS_KEY, 1.0)
    window = MainWindow()
    try:
        with (
            patch.object(qapp, "setFont", wraps=qapp.setFont) as set_font,
            patch.object(qapp, "setStyleSheet", wraps=qapp.setStyleSheet) as set_sheet,
        ):
            window._ui_manager.set_ui_scale(1.2)
        assert set_font.call_count == 1
        assert set_sheet.call_count == 1
        # The composed sheet is base + exactly one scale block: setting the same
        # scale again is a no-op (guarded by the equality check).
        with (
            patch.object(qapp, "setFont", wraps=qapp.setFont) as set_font2,
            patch.object(qapp, "setStyleSheet", wraps=qapp.setStyleSheet) as set_sheet2,
        ):
            window._ui_manager.set_ui_scale(1.2)
        assert set_font2.call_count == 0
        assert set_sheet2.call_count == 0
    finally:
        window.close()


@pytest.mark.gui
def test_ui_scale_change_updates_param_value_cell_font_via_owner_hook(qapp) -> None:
    """A parameter table's cell font tracks a live scale change via its owner hook.

    The table sets an explicit mono cell font (``apply_param_table_style``) that
    ignores the application font, and QSS cannot reach a per-cell font. The
    owning panel's ``_on_ui_scale_changed`` re-derives it from the builders, so
    ``table.font()`` (which drives cell rendering) tracks the scale — only
    because the hook re-applied it.
    """
    from PySide6.QtCore import QSettings

    from asymmetry.gui.mainwindow import MainWindow
    from asymmetry.gui.ui_manager import UI_SCALE_SETTINGS_KEY

    QSettings().setValue(UI_SCALE_SETTINGS_KEY, 1.0)
    window = MainWindow()
    try:
        panel = window._fit_parameters_panel
        panel._ensure_ui_scale_sync()  # connect the hook (normally on showEvent)
        window._ui_manager.set_ui_scale(1.0)
        base = panel._table.font().pointSizeF()
        window._ui_manager.set_ui_scale(1.2)
        grown = panel._table.font().pointSizeF()
        window._ui_manager.set_ui_scale(0.8)
        shrunk = panel._table.font().pointSizeF()
        assert grown > base > shrunk
        assert grown == pytest.approx(base * 1.2, rel=0.02)
    finally:
        window.close()


@pytest.mark.gui
def test_ui_scale_change_updates_formula_box_font_via_owner_hook(qapp) -> None:
    """A formula box's mono label tracks a live scale change, keeping any recolour.

    ``FormulaBox`` sets an explicit ``mono_font(SIZE_NUMERIC)`` on its inner label
    that ignores the application font; QSS cannot reach it cleanly because the
    label's own local stylesheet (also used by the domain-mismatch recolour)
    wins. The box self-subscribes to the UIManager signal and re-derives the
    label font from the builders — font-only, so a recolour stylesheet survives.
    """
    from PySide6.QtCore import QSettings

    from asymmetry.gui.mainwindow import MainWindow
    from asymmetry.gui.styles.widgets import FormulaBox
    from asymmetry.gui.ui_manager import UI_SCALE_SETTINGS_KEY

    QSettings().setValue(UI_SCALE_SETTINGS_KEY, 1.0)
    window = MainWindow()
    try:
        boxes = window.findChildren(FormulaBox)
        assert boxes, "expected at least one FormulaBox in the fit panels"
        box = boxes[0]
        box._ensure_ui_scale_sync()  # connect the hook (normally on showEvent)
        # A domain-mismatch recolour lives in the label's own stylesheet; it must
        # survive the font-only refresh.
        box.label.setStyleSheet("QLabel { background: transparent; color: #ff0000; }")
        window._ui_manager.set_ui_scale(1.0)
        base = box.label.font().pointSizeF()
        window._ui_manager.set_ui_scale(1.2)
        grown = box.label.font().pointSizeF()
        window._ui_manager.set_ui_scale(0.8)
        shrunk = box.label.font().pointSizeF()
        assert grown > base > shrunk
        assert grown == pytest.approx(base * 1.2, rel=0.02)
        assert "#ff0000" in box.label.styleSheet()  # recolour not clobbered
    finally:
        window.close()


@pytest.mark.gui
def test_dock_min_widths_derive_from_font_metrics(qapp) -> None:
    """Inspector/browser dock minimum widths derive from font metrics, not px literals."""
    from PySide6.QtCore import QSettings

    from asymmetry.gui.mainwindow import (
        _BROWSER_DOCK_MIN_CHARS,
        _INSPECTOR_DOCK_MIN_CHARS,
        MainWindow,
    )
    from asymmetry.gui.styles import metrics
    from asymmetry.gui.ui_manager import UI_SCALE_SETTINGS_KEY

    QSettings().setValue(UI_SCALE_SETTINGS_KEY, 1.0)
    window = MainWindow()
    try:
        assert window._dock_data_browser.minimumWidth() == metrics.char_width(
            _BROWSER_DOCK_MIN_CHARS
        )
        for dock in (
            window._dock_fit,
            window._dock_fourier,
            window._dock_fit_parameters,
        ):
            assert dock.minimumWidth() == metrics.char_width(_INSPECTOR_DOCK_MIN_CHARS)
    finally:
        window.close()


@pytest.mark.gui
def test_repeated_mainwindow_construction_does_not_compound_app_styles(qapp) -> None:
    """A second MainWindow must reuse the QApplication's pre-scale baseline.

    UIManager used to capture ``app.styleSheet()`` / ``app.font()`` at
    construction as its "base" — i.e. the *previous* manager's output — so every
    window appended another rendered scale block to the app stylesheet and
    multiplied the app font by the scale factor again. Each GUI test constructs
    its own MainWindow against the shared QApplication, so the sheet grew (and
    ``setStyleSheet`` slowed) linearly across a test session.
    """
    from asymmetry.gui.mainwindow import MainWindow

    first = MainWindow()
    try:
        sheet_after_first = qapp.styleSheet()
        font_after_first = qapp.font().pointSizeF()
    finally:
        first.close()
        first.deleteLater()

    second = MainWindow()
    try:
        assert qapp.styleSheet() == sheet_after_first
        assert qapp.font().pointSizeF() == pytest.approx(font_after_first)
    finally:
        second.close()
        second.deleteLater()


# ── Plot-trace palette centralisation (P2-6) ─────────────────────────────────


def test_period_overlay_palette_is_token_derived() -> None:
    from asymmetry.gui.styles import tokens
    from asymmetry.gui.styles.plots import period_overlay_palette

    red = period_overlay_palette(tokens.PERIOD_RED)
    assert tokens.PERIOD_RED not in red, "overlay must exclude its own base hue"
    assert all(c in vars(tokens).values() for c in red), "overlay colours must be tokens"
    # Unknown base falls back to the full Okabe-Ito cycle.
    fallback = period_overlay_palette("#123456")
    assert tokens.TRACE_BLUE in fallback


def test_data_browser_logged_and_accent_colours_are_token_derived(qapp) -> None:
    from PySide6.QtGui import QColor

    from asymmetry.gui.panels import data_browser
    from asymmetry.gui.panels.data_browser import _RowHighlightDelegate
    from asymmetry.gui.styles import tokens

    assert data_browser._LOG_TEMPERATURE_FOREGROUND == QColor(tokens.LOGGED_VALUE_FG)
    accent = QColor(tokens.ACCENT)
    soft = _RowHighlightDelegate._ACCENT_SOFT
    assert (soft.red(), soft.green(), soft.blue()) == (accent.red(), accent.green(), accent.blue())
    assert soft.alpha() == 102
