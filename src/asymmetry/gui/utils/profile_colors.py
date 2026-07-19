"""Identity colours for grouping profiles (schema v17 multi-profile projects).

Each **non-default** grouping profile of an instrument wears a stable
identity colour on every surface — Data Browser run numbers, the grouping
window's scope rows and editing strip, and the profile selector swatches —
so a multi-sample project reads consistently. The instrument's ★ default
profile stays uncoloured (plain black run numbers), so colour itself reads
as "moved off the default".

The colour is *stored on the profile* (``GroupingProfile.color``, an
additive project-file field) and assigned from
:data:`~asymmetry.gui.styles.tokens.PROFILE_COLORS` the first time a
non-default profile is saved; colourless profiles from older saves fall back
to their position among the fingerprint's colour-bearing profiles, which
matches what a later save assigns them.
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
    """The colour *profile* would wear: its stored colour, else a fallback.

    The fallback is the profile's position among *fingerprint_profiles*
    counted over the palette, skipping colourless default profiles (they
    occupy no palette slot) — deterministic for colourless profiles from
    older saves, and consistent with what :func:`next_profile_color` assigns
    when the profile is next saved. Callers that render should prefer
    :func:`display_profile_color`, which hides the default profile's colour.
    """
    stored = getattr(profile, "color", None)
    if stored:
        return str(stored)
    index = 0
    for candidate in fingerprint_profiles:
        if candidate is profile or getattr(candidate, "name", None) == profile.name:
            break
        if getattr(candidate, "active", False) and not getattr(candidate, "color", None):
            continue  # a colourless default occupies no palette slot
        index += 1
    return tokens.PROFILE_COLORS[index % len(tokens.PROFILE_COLORS)]


def display_profile_color(profile, fingerprint_profiles) -> str | None:
    """The colour to *render* for *profile* — ``None`` for the ★ default.

    The default profile stays plain (black text) on every surface, so a
    coloured run number always means "assigned off the default".
    """
    if getattr(profile, "active", False):
        return None
    return effective_profile_color(profile, fingerprint_profiles)


def used_profile_colors(fingerprint_profiles) -> list[str]:
    """The palette colours occupied by *fingerprint_profiles*.

    Colourless default profiles occupy nothing; every other profile occupies
    its stored (or fallback) colour. Feed this to :func:`next_profile_color`
    when assigning a fresh colour — excluding the profile being coloured
    from the list, or its own fallback would occupy its slot.
    """
    return [
        effective_profile_color(p, fingerprint_profiles)
        for p in fingerprint_profiles
        if getattr(p, "color", None) or not getattr(p, "active", False)
    ]


def soft_profile_background(color: str, *, alpha: float = 0.12) -> str:
    """A CSS ``rgba(...)`` soft tint of *color* for backgrounds."""
    value = str(color).lstrip("#")
    r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


__all__ = [
    "next_profile_color",
    "effective_profile_color",
    "display_profile_color",
    "used_profile_colors",
    "soft_profile_background",
]
