"""Identity colours and axis labels for phase data groups (Global Fit Wizard
transitions, D1/D2).

A series crossing one or more transitions is partitioned into ordinal
*phases* (:attr:`~asymmetry.core.representation.group.DataGroup.phase_ordinal`),
each wearing a stable colour from :data:`~asymmetry.gui.styles.tokens.PHASE_COLORS`
on its Data Browser sub-header stripe/swatch and its Fit Parameters panel
series button, range shading, and boundary lines. See ``tokens.PHASE_COLORS``
for why the ramp is ordered cold-to-hot rather than an arbitrary identity
palette like :mod:`asymmetry.gui.utils.profile_colors`.

The range/boundary formatters live here too, so the Data Browser sub-header,
its ⓘ popover and the Fit Parameters panel's boundary annotations all spell a
temperature, a field or a run index the same way.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from asymmetry.gui.styles import tokens

if TYPE_CHECKING:  # pragma: no cover - typing only
    from asymmetry.core.representation.group import DataGroup

#: How a sweep-axis value is spelled per ``DataGroup.order_key``: a format
#: string and the unit suffix. ``"run"`` is an index, not a measurement, so it
#: carries no unit and no decimals.
_AXIS_FORMATS: dict[str, tuple[str, str]] = {
    "temperature": ("{:.1f}", " K"),
    "field": ("{:.1f}", " G"),
    "run": ("{:.0f}", ""),
}
#: En dash with hair spaces, the range separator used across the phase UI.
_RANGE_DASH = " – "

#: Excluded-phase rows (a series member claimed by no phase) render with a
#: hatched stripe rather than a phase swatch — there is no ordinal colour to
#: assign an excluded run. The neutral grey already used for muted text
#: (``tokens.TEXT_MUTED``) keeps the marker reading as "no phase", never as a
#: sixth phase colour; the diagonal hatch mirrors the excluded-range treatment
#: already used for fit-window gaps (``trend_preview._draw_window_gaps``).
EXCLUDED_PHASE_HATCH_COLOR: str = tokens.TEXT_MUTED
EXCLUDED_PHASE_HATCH_PATTERN: str = "///"


def phase_color(ordinal: int, *, dark: bool = False) -> str:
    """The swatch colour for phase *ordinal* (1-based), cycling past five.

    *dark* selects :data:`tokens.PHASE_COLORS_DARK` instead of
    :data:`tokens.PHASE_COLORS` — pass the surface's own light/dark state,
    mirroring :func:`asymmetry.gui.utils.profile_colors.next_profile_color`'s
    palette-cycling shape.
    """
    palette = tokens.PHASE_COLORS_DARK if dark else tokens.PHASE_COLORS
    index = (int(ordinal) - 1) % len(palette)
    return palette[index]


def soft_phase_background(color: str, *, alpha: float = 0.12) -> str:
    """A CSS ``rgba(...)`` soft tint of *color* for backgrounds.

    Mirrors :func:`asymmetry.gui.utils.profile_colors.soft_profile_background`
    exactly — same shape, same default alpha — so a phase swatch and a
    profile swatch soften identically.
    """
    value = str(color).lstrip("#")
    r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def resolve_phase_color(group: DataGroup, *, dark: bool = False) -> str:
    """The colour a phase *group* wears: its stored swatch, else its ordinal's.

    :attr:`DataGroup.phase_color` is what the wizard assigned when it created
    the partition and is authoritative when present; a phase created without
    one (a hand-built partition, or a legacy project) falls back to the
    ordinal's slot in the ramp. *dark* only reaches the fallback — a stored
    colour is a recorded decision, not a theme-dependent lookup.
    """
    return group.phase_color or phase_color(group.phase_ordinal or 1, dark=dark)


def format_axis_value(value: float, order_key: str) -> str:
    """Spell one sweep-axis *value* with the unit *order_key* implies."""
    template, unit = _AXIS_FORMATS[order_key]
    return f"{template.format(float(value))}{unit}"


def format_axis_range(low: float, high: float, order_key: str) -> str:
    """Spell a span along the sweep axis, e.g. ``"1.8 – 16.0 K"``.

    The unit is written once, after the pair, so a range reads as one quantity.
    Shared with the Global Fit Wizard's Transitions card, which spells a phase's
    span before that phase is a :class:`DataGroup` at all.
    """
    template, unit = _AXIS_FORMATS[order_key]
    return f"{template.format(float(low))}{_RANGE_DASH}{template.format(float(high))}{unit}"


def format_phase_range(group: DataGroup) -> str:
    """Spell a phase's span, e.g. ``"1.8 – 16.0 K"`` (``""`` when unknown).

    ``phase_range`` is ``None`` on a phase that has lost every member (the
    manual "Move to phase" override can empty one), which has no span to
    print — the caller renders the name alone.
    """
    if group.phase_range is None:
        return ""
    low, high = group.phase_range
    return format_axis_range(low, high, group.order_key)


def format_phase_boundary(boundary: tuple[float, float] | None, order_key: str) -> str:
    """Spell one break estimate as ``"16.5 ± 0.5 K"``.

    *boundary* is an ``(estimate, half_gap)`` pair, or ``None`` at a series end
    — the two ends of the whole series genuinely have no break beyond them, so
    the caller prints the returned ``"series end"`` rather than a number.
    """
    if boundary is None:
        return "series end"
    estimate, half_gap = boundary
    template, unit = _AXIS_FORMATS[order_key]
    return f"{template.format(estimate)} ± {template.format(half_gap)}{unit}"


__all__ = [
    "EXCLUDED_PHASE_HATCH_COLOR",
    "EXCLUDED_PHASE_HATCH_PATTERN",
    "format_axis_value",
    "format_phase_boundary",
    "format_phase_range",
    "phase_color",
    "resolve_phase_color",
    "soft_phase_background",
]
