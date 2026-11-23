"""Numbering templates describe acquired readout slots, not physical detector identities."""

from dataclasses import dataclass

from asymmetry.core.instruments.schema import InstrumentProfile


@dataclass(frozen=True)
class NumberingSlot:
    """Stable acquisition/channel identity with its current recorded histogram number."""

    id: str
    channel: str
    ordinal: int
    label: str | None
    physical: tuple[str, ...]


def numbering_template(profile: InstrumentProfile, acquisition: str) -> tuple[NumberingSlot, ...]:
    """Return one row per selectable channel, independent of view order or placement.

    The acquisition qualifies repeated channel keys whose histogram numbers differ
    between readout configurations. Physical associations may be empty or multiple;
    they are never inferred from the slot's drawing position.
    """
    config = next(
        (config for config in profile.detectors.acquisitions if config.key == acquisition), None
    )
    if config is None:
        raise ValueError(f"Unknown acquisition {acquisition!r} for {profile.identity.key}")
    return tuple(
        NumberingSlot(
            f"{profile.identity.key}/{config.key}/{channel.key}",
            channel.key,
            channel.ordinal,
            channel.label,
            channel.physical,
        )
        for channel in config.channels
        if channel.selectable
    )
