"""Tests for asymmetry.gui.utils.phase_colors (D1, Global Fit Wizard transitions)."""

from __future__ import annotations

from asymmetry.gui.styles import tokens
from asymmetry.gui.utils.phase_colors import (
    EXCLUDED_PHASE_HATCH_COLOR,
    EXCLUDED_PHASE_HATCH_PATTERN,
    phase_color,
    soft_phase_background,
)


def test_phase_color_follows_ordinal_and_light_dark_palettes():
    assert phase_color(1) == tokens.PHASE_COLORS[0]
    assert phase_color(2) == tokens.PHASE_COLORS[1]
    assert phase_color(5) == tokens.PHASE_COLORS[4]
    assert phase_color(1, dark=True) == tokens.PHASE_COLORS_DARK[0]
    assert phase_color(2, dark=True) == tokens.PHASE_COLORS_DARK[1]


def test_phase_color_cycles_past_five():
    assert phase_color(6) == phase_color(1)
    assert phase_color(7) == phase_color(2)
    assert phase_color(6, dark=True) == phase_color(1, dark=True)


def test_soft_phase_background_matches_soft_profile_background_shape():
    from asymmetry.gui.utils.profile_colors import soft_profile_background

    color = tokens.PHASE_COLORS[0]
    assert soft_phase_background(color) == soft_profile_background(color)
    assert soft_phase_background(color, alpha=0.3) == soft_profile_background(color, alpha=0.3)


def test_soft_phase_background_is_rgba_css_string():
    rgba = soft_phase_background("#2F4DA0")
    assert rgba == "rgba(47, 77, 160, 0.12)"


def test_excluded_phase_hatch_uses_the_muted_text_token():
    # Colour is the neutral grey already used for muted text — never a sixth
    # phase colour, so an excluded run never reads as belonging to a phase.
    assert EXCLUDED_PHASE_HATCH_COLOR == tokens.TEXT_MUTED
    assert EXCLUDED_PHASE_HATCH_COLOR not in tokens.PHASE_COLORS
    assert isinstance(EXCLUDED_PHASE_HATCH_PATTERN, str) and EXCLUDED_PHASE_HATCH_PATTERN
