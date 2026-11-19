"""Sourced instrument declarations and pure geometry expansion."""

from asymmetry.core.instruments.isis_emu import PROFILE as EMU
from asymmetry.core.instruments.isis_hifi import PROFILE as HIFI
from asymmetry.core.instruments.isis_musr import PROFILE as MUSR
from asymmetry.core.instruments.psi_alc import PROFILE as ALC
from asymmetry.core.instruments.psi_dolly import PROFILE as DOLLY
from asymmetry.core.instruments.psi_flame import PROFILE as FLAME
from asymmetry.core.instruments.psi_gpd import PROFILE as GPD
from asymmetry.core.instruments.psi_gps import PROFILE as GPS
from asymmetry.core.instruments.psi_hal9500 import PROFILE as HAL9500
from asymmetry.core.instruments.psi_lem import PROFILE as LEM
from asymmetry.core.instruments.psi_ltf import PROFILE as LTF
from asymmetry.core.instruments.resolve import layout_from_profile
from asymmetry.core.instruments.schema import InstrumentProfile

__all__ = [
    "EMU",
    "HIFI",
    "MUSR",
    "GPS",
    "FLAME",
    "HAL9500",
    "LTF",
    "GPD",
    "DOLLY",
    "ALC",
    "LEM",
    "InstrumentProfile",
    "layout_from_profile",
]
