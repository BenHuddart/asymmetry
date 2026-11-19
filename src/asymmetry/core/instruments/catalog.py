"""One deterministic catalog of declarations bundled with the installed package."""

import re
from importlib import import_module
from pkgutil import iter_modules

from asymmetry.core import instruments
from asymmetry.core.instruments.schema import InstrumentProfile


def bundled_profiles() -> tuple[InstrumentProfile, ...]:
    """Load only reserved facility_instrument modules in this package."""
    names = sorted(
        module.name
        for module in iter_modules(instruments.__path__)
        if re.fullmatch(r"[a-z][a-z0-9]*_[a-z][a-z0-9]*", module.name)
    )
    profiles = tuple(import_module(f"{__package__}.{name}").PROFILE for name in names)
    if len({profile.identity.key for profile in profiles}) != len(profiles):
        raise ValueError("Bundled instrument profiles require unique stable identities")
    layout_keys = [
        config.layout_key or profile.identity.layout_key
        for profile in profiles
        for config in profile.detectors.acquisitions
    ]
    if len(set(layout_keys)) != len(layout_keys):
        raise ValueError("Bundled acquisitions require unique public layout keys")
    return profiles


PROFILES = bundled_profiles()


def profile_by_key(key: str) -> InstrumentProfile:
    """Return one stable bundled identity, never a display-name approximation."""
    profile = next((profile for profile in PROFILES if profile.identity.key == key), None)
    if profile is None:
        raise ValueError(f"Unknown instrument profile {key!r}")
    return profile
