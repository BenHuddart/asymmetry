"""Verify the GUI colour tokens and the no-raw-hex guard.

The no-raw-hex rule is enforced repository-wide by
:func:`tools.harness.find_raw_hex_colour_violations` (over all of ``gui/``
outside ``styles/``, with a per-file allowlist for specialist matplotlib /
QPainter / palette colours). This module keeps a single test that delegates to
that function so there is *one* home for the rule, plus the token-inventory
spot-checks.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_HARNESS_PATH = Path(__file__).resolve().parents[2] / "tools" / "harness.py"


def _load_harness():
    spec = importlib.util.spec_from_file_location("asymmetry_harness", _HARNESS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_no_raw_hex_literals_anywhere_in_gui() -> None:
    """No raw hex-colour literal outside styles/ and the harness allowlist.

    Delegates to the structural harness rule so the check has one home; chrome
    colours must come from ``styles/tokens.py``.
    """
    harness = _load_harness()

    failures = harness.find_raw_hex_colour_violations()
    assert not failures, "Raw hex-colour literals in gui/:\n" + "\n".join(
        f.format() for f in failures
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
        "CAVEAT_BANNER_BG",
        "CAVEAT_BANNER_TEXT",
        "CAUTION_BANNER_BG",
        "CAUTION_BANNER_TEXT",
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
