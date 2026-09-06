"""Tests for asymmetry.gui.utils.phase_colors (D1/D2, Global Fit Wizard transitions)."""

from __future__ import annotations

import pytest

from asymmetry.core.representation.group import DataGroup
from asymmetry.gui.styles import tokens
from asymmetry.gui.utils.phase_colors import (
    EXCLUDED_PHASE_HATCH_COLOR,
    EXCLUDED_PHASE_HATCH_PATTERN,
    format_axis_value,
    format_phase_boundary,
    format_phase_range,
    phase_color,
    resolve_phase_color,
    soft_phase_background,
)


def _phase(**kwargs) -> DataGroup:
    return DataGroup(
        group_id="p1",
        name="Phase I",
        member_run_numbers=[1, 2, 3],
        parent_group_id="series",
        phase_ordinal=kwargs.pop("phase_ordinal", 2),
        **kwargs,
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


# ── D2: the colour resolver and the axis formatters ──────────────────────────


def test_resolve_phase_color_prefers_the_stored_swatch():
    # The stored colour is the wizard's recorded decision, so it wins over the
    # ordinal's ramp slot in both themes.
    group = _phase(phase_color="#123456")
    assert resolve_phase_color(group) == "#123456"
    assert resolve_phase_color(group, dark=True) == "#123456"


def test_resolve_phase_color_falls_back_to_the_ordinal_ramp():
    group = _phase(phase_color=None)
    assert resolve_phase_color(group) == phase_color(2)
    assert resolve_phase_color(group, dark=True) == phase_color(2, dark=True)


def test_format_axis_value_carries_the_unit_the_order_key_implies():
    assert format_axis_value(16.05, "temperature") == "16.1 K"
    assert format_axis_value(100.0, "field") == "100.0 G"
    assert format_axis_value(42.0, "run") == "42"


def test_format_phase_range_spells_the_span_with_one_unit():
    assert format_phase_range(_phase(phase_range=(1.8, 16.0), order_key="temperature")) == (
        "1.8 – 16.0 K"
    )
    assert format_phase_range(_phase(phase_range=(3.0, 9.0), order_key="run")) == "3 – 9"


def test_format_phase_range_is_empty_for_a_phase_with_no_members_left():
    # "Move to phase" can empty a phase; ProjectModel then clears its range.
    assert format_phase_range(_phase(phase_range=None, order_key="temperature")) == ""


def test_format_phase_boundary_spells_the_estimate_and_the_half_gap():
    assert format_phase_boundary((16.5, 0.5), "temperature") == "16.5 ± 0.5 K"


def test_format_phase_boundary_names_a_series_end():
    # The outer ends of the whole scan have no break beyond them.
    assert format_phase_boundary(None, "temperature") == "series end"


@pytest.mark.gui
@pytest.mark.usefixtures("qapp")
def test_is_dark_surface_reads_the_row_background_not_the_window():
    """The one place the phase ramp asks light-or-dark (``PHASE_COLORS_DARK``).

    It reads ``QPalette.Base`` — the colour a table row is actually painted on
    — so a widget deliberately hosted on a dark surface answers for itself even
    while the application palette stays light.
    """
    from PySide6.QtGui import QColor, QPalette

    from asymmetry.gui.styles.palette import build_bench_palette, is_dark_surface

    assert not is_dark_surface(build_bench_palette())

    dark = QPalette(build_bench_palette())
    dark.setColor(QPalette.ColorRole.Base, QColor("#1B1B1B"))
    dark.setColor(QPalette.ColorRole.Window, QColor("#FFFFFF"))
    assert is_dark_surface(dark)
