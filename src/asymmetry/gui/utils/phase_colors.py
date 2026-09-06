"""Identity colours for phase data groups (Global Fit Wizard transitions, D1).

A series crossing one or more transitions is partitioned into ordinal
*phases* (:attr:`~asymmetry.core.representation.group.DataGroup.phase_ordinal`),
each wearing a stable colour from :data:`~asymmetry.gui.styles.tokens.PHASE_COLORS`
on its Data Browser sub-header stripe/swatch and its Fit Parameters panel
series button, range shading, and boundary lines. See ``tokens.PHASE_COLORS``
for why the ramp is ordered cold-to-hot rather than an arbitrary identity
palette like :mod:`asymmetry.gui.utils.profile_colors`.
"""

from __future__ import annotations

from asymmetry.gui.styles import tokens

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


__all__ = [
    "EXCLUDED_PHASE_HATCH_COLOR",
    "EXCLUDED_PHASE_HATCH_PATTERN",
    "phase_color",
    "soft_phase_background",
]
