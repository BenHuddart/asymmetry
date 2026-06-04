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

_GUI_ROOT = Path(__file__).parent.parent / "src" / "asymmetry" / "gui"

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
    "panels/fourier_panel.py",
    "panels/data_browser.py",
    "panels/cross_group_fit_dialog.py",
    "panels/model_fit_dialog.py",
    "panels/composite_parameter_dialog.py",
    "widgets/function_expression_builder.py",
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
