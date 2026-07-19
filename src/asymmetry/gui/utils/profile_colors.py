"""Identity colours for grouping profiles (schema v17 multi-profile projects).

Each grouping profile of an instrument wears a stable identity colour on
every surface — Data Browser run numbers, the grouping window's scope rows
and editing strip, and the profile selector swatches — so a multi-sample
project reads consistently. The colour is *stored on the profile*
(``GroupingProfile.color``, an additive project-file field) and assigned from
:data:`~asymmetry.gui.styles.tokens.PROFILE_COLORS` the first time the
profile is saved; profiles from files predating colours fall back to their
position in the instrument's profile list, which matches what a later save
assigns them.
"""

from __future__ import annotations

from asymmetry.gui.styles import tokens


def next_profile_color(used_colors) -> str:
    """The first palette colour not in *used_colors* (cycling when exhausted)."""
    used = {str(c) for c in used_colors if c}
    for color in tokens.PROFILE_COLORS:
        if color not in used:
            return color
    return tokens.PROFILE_COLORS[len(used) % len(tokens.PROFILE_COLORS)]


def effective_profile_color(profile, fingerprint_profiles) -> str:
    """The colour *profile* wears: its stored colour, else a stable fallback.

    The fallback is the profile's position in *fingerprint_profiles* (the
    instrument's profile list) taken over the palette — deterministic for
    colourless profiles from older saves, and consistent with the colour
    :func:`next_profile_color` assigns when the profile is next saved.
    """
    stored = getattr(profile, "color", None)
    if stored:
        return str(stored)
    index = 0
    for candidate in fingerprint_profiles:
        if candidate is profile or getattr(candidate, "name", None) == profile.name:
            break
        index += 1
    return tokens.PROFILE_COLORS[index % len(tokens.PROFILE_COLORS)]


def soft_profile_background(color: str, *, alpha: float = 0.12) -> str:
    """A CSS ``rgba(...)`` soft tint of *color* for backgrounds."""
    value = str(color).lstrip("#")
    r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


__all__ = ["next_profile_color", "effective_profile_color", "soft_profile_background"]
