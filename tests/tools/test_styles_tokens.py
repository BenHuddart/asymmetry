"""Verify that cleaned GUI files contain no raw hex-colour string literals.

Every file in CLEAN_FILES was explicitly converted during the BENCH redesign
phases 11-15.  Any new `"#rrggbb"` literal in those files is a regression:
colours must come from `styles/tokens.py` or a helper in `styles/widgets.py`.

Files with deliberate non-BENCH hex literals (period-mode palettes, component
colour cycles, specialist diagram colours) are intentionally excluded from this
list and should stay that way.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_GUI_ROOT = Path(__file__).parent.parent.parent / "src" / "asymmetry" / "gui"

# Regex: a hex-colour string starting with # immediately after an opening
# quote.  Matches "  "#1f4d8a"  and  '#e8eef7'  but NOT  # comments  or
# triple-quoted docstrings where the # is not right after a quote character.
_HEX_IN_STRING = re.compile(r"""["']#[0-9a-fA-F]{6}""")

# Files that were cleaned during phases 11-15 and should have zero stray hex
# colour literals.  Period-mode palettes, fit-component colour cycles, wizard
# plot colours, and detector-schematic diagram colours live elsewhere and are
# intentionally preserved.
CLEAN_FILES: list[str] = [
    "mainwindow.py",
    "ui_manager.py",
    "panels/fit_panel.py",
    # fit_panel.py was split into panels/fit/ (Phase 2 mechanical split); keep
    # the no-raw-hex guard on the relocated code.
    "panels/fit/seeding.py",
    "panels/fit/tab_base.py",
    "panels/fit/single_tab.py",
    "panels/fit/global_tab.py",
    "panels/fit/panel.py",
    "panels/fourier_panel.py",
    "panels/data_browser.py",
    # plot_panel.py's period-mode base colours and the Okabe-Ito overlay palette
    # were routed through tokens.py / styles/plots.py (F6, P2-6).
    "panels/plot_panel.py",
    "panels/cross_group_fit_dialog.py",
    "panels/model_fit_dialog.py",
    "panels/composite_parameter_dialog.py",
    "windows/detector_layout_dialog.py",
    # fit_wizard_window.py is excluded — its embedded matplotlib preview plots
    # use specialist non-BENCH hex colours for scientific visualisation.
]


@pytest.mark.parametrize("rel_path", CLEAN_FILES)
def test_no_raw_hex_literals(rel_path: str) -> None:
    """Fail if a hex-colour string literal appears in a cleaned GUI file."""
    path = _GUI_ROOT / rel_path
    if not path.exists():
        pytest.skip(f"{rel_path} not found")

    violations: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if _HEX_IN_STRING.search(line):
            violations.append(f"  line {lineno}: {line.strip()}")

    assert not violations, (
        f"{rel_path} contains raw hex-colour string literals — "
        f"use tokens.py instead:\n" + "\n".join(violations)
    )


def test_tokens_module_exports_expected_constants() -> None:
    """Spot-check that the token constants used throughout the redesign exist."""
    from asymmetry.gui.styles import tokens

    required = [
        "BG",
        "SURFACE",
        "SURFACE_ALT",
        "SURFACE_HI",
        "BORDER",
        "BORDER_STRONG",
        "TEXT",
        "TEXT_MUTED",
        "TEXT_DIM",
        "ACCENT",
        "ACCENT_SOFT",
        "ACCENT_SOFT2",
        "ACCENT_RED",
        "ACCENT_RED_SOFT",
        "ACCENT_RED_SOFT2",
        "GROUP_HEADER_BG",
        "GROUP_MEMBER_BG",
        "WARN",
        "OK",
        "FIT",
        "ERROR",
        "SUCCESS_BG",
        "SUCCESS_BORDER",
        "PLOT_AXIS",
        "PLOT_TICK_LABEL",
        "PLOT_GRID",
        "PLOT_ZERO_LINE",
        "PLOT_LEGEND_BG",
        "PLOT_DATA",
        "PLOT_FIT",
        "PLOT_FIT_RANGE_FACE",
        "PLOT_FIT_RANGE_EDGE",
        "PLOT_LOW_COUNT",
        "LOG_TAG_ACCENT",
        "LOG_TAG_OK",
        "LOG_TAG_WARN",
        # F6 (P2-6) additions: data-trace palette, period-mode bases, preview
        # fit colour, and the logged-value foreground.
        "TRACE_BLUE",
        "TRACE_SKY",
        "TRACE_GREEN",
        "TRACE_YELLOW",
        "TRACE_MAGENTA",
        "TRACE_BLACK",
        "TRACE_ORANGE",
        "TRACE_VERMILLION",
        "PERIOD_RED",
        "PERIOD_GREEN",
        "PERIOD_DIFF",
        "PERIOD_SUM",
        "PLOT_FIT_PREVIEW",
        "LOGGED_VALUE_FG",
        "WHITE",
    ]
    missing = [name for name in required if not hasattr(tokens, name)]
    assert not missing, f"Missing token constants: {missing}"


def test_widgets_helpers_exist() -> None:
    """Verify that centralised helper functions are importable."""
    from asymmetry.gui.styles.widgets import (
        build_segmented_button_qss,
        style_group_state_button,
        success_html,
    )

    # All symbols imported — no further assertion needed.
    assert callable(build_segmented_button_qss)
    assert callable(success_html)
    assert callable(style_group_state_button)


def test_plots_module_helpers_exist() -> None:
    """Verify that centralised matplotlib-style helpers are importable."""
    from asymmetry.gui.styles.plots import (
        draw_fit_range_span,
        style_axes,
    )

    assert callable(style_axes)
    assert callable(draw_fit_range_span)


def test_typography_module_constants() -> None:
    """Verify that the BENCH type-scale constants are defined."""
    from asymmetry.gui.styles.typography import (
        LETTER_SPACING_HEADER,
        LETTER_SPACING_LABEL,
        SIZE_HEADER,
        SIZE_NUMERIC,
        header_font,
        section_label_font,
    )

    assert SIZE_HEADER == 9.5
    assert SIZE_NUMERIC == 11.0
    assert LETTER_SPACING_HEADER == 0.3
    assert LETTER_SPACING_LABEL == 0.4
    assert callable(header_font)
    assert callable(section_label_font)


def test_red_accent_tokens_are_hex_strings() -> None:
    """Red accent tokens must be hex strings (not tuples or ints)."""
    from asymmetry.gui.styles import tokens

    for name in ("ACCENT_RED", "ACCENT_RED_SOFT", "ACCENT_RED_SOFT2"):
        value = getattr(tokens, name)
        assert isinstance(value, str) and value.startswith("#"), (
            f"tokens.{name} should be a '#rrggbb' string, got {value!r}"
        )


def test_data_browser_series_highlight_uses_red_token() -> None:
    """The data browser series-highlight background must derive from ACCENT_RED_SOFT."""
    from PySide6.QtGui import QColor

    from asymmetry.gui.panels import data_browser
    from asymmetry.gui.styles import tokens

    expected = QColor(tokens.ACCENT_RED_SOFT)
    actual = data_browser._SERIES_HIGHLIGHT_BACKGROUND
    assert actual.red() == expected.red()
    assert actual.green() == expected.green()
    assert actual.blue() == expected.blue()
