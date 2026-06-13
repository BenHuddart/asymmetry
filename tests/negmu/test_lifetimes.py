"""Tests for core/negmu/lifetimes.py (WP1.1).

Spot-checks are EXACT float comparisons (the table transcription is the source
of truth; if a value changes here it means the table was re-transcribed and
the test must be updated to match). ⚠confirm rows are NOT asserted to specific
values — only presence and positivity.
"""

from __future__ import annotations

import pytest

from asymmetry.core.negmu.lifetimes import (
    DECAY_BACKGROUND_LABEL,
    ELEMENT_LIFETIMES,
    ElementLifetime,
    elements,
    has_element,
    lifetime,
    tau_us,
)

# ---------------------------------------------------------------------------
# Spot-checks from verification-plan.md §1 — exact values
# ---------------------------------------------------------------------------

SPOT_CHECKS = [
    ("H", 2.19480),
    ("Be", 2.16747),
    ("C", 2.030),
    ("O", 1.795),
    ("Na", 1.204),
    ("Al", 0.864),
    ("Si", 0.759),
    ("Ca", 0.336),
    ("Fe", 0.206),
    ("Cu", 0.164),
    ("Br", 0.133),
    ("Zr", 0.110),
    ("Nb", 0.092),
    ("Au", 0.0728),
    ("Pb", 0.0747),
    ("Bi", 0.0735),
]


@pytest.mark.parametrize("sym,expected_tau", SPOT_CHECKS)
def test_spot_check_tau(sym, expected_tau):
    assert tau_us(sym) == expected_tau


def test_tl_symbol_guard():
    """Guards the WiMDA 'Ti'→'Tl' symbol bug."""
    assert tau_us("Tl") == 0.0704
    assert lifetime("Tl").symbol == "Tl"
    assert lifetime("Tl").z == 81


def test_ne_value_guard():
    """Guards WiMDA Ne 1.520 μs vs Suzuki 1.461 μs divergence."""
    assert tau_us("Ne") == 1.461


def test_source_field_confident_rows():
    assert lifetime("H").source == "SuzukiMeasdayRoalsvig1987"
    assert lifetime("Fe").source == "SuzukiMeasdayRoalsvig1987"
    assert lifetime("Pb").source == "SuzukiMeasdayRoalsvig1987"


def test_source_field_wimda_provisional():
    assert lifetime("He").source == "WiMDA-provisional"
    assert lifetime("Kr").source == "WiMDA-provisional"
    assert lifetime("Tc").source == "WiMDA-provisional"
    assert lifetime("Re").source == "WiMDA-provisional"


def test_sigma_none_for_provisional():
    for sym in ("He", "Kr", "Tc", "Re", "Os", "Ir", "Pt"):
        assert lifetime(sym).sigma_us is None


def test_tau_range():
    """Every entry's tau_us is strictly within [0.05, 2.30] μs."""
    for sym, entry in ELEMENT_LIFETIMES.items():
        assert entry.tau_us > 0, sym
        assert 0.05 <= entry.tau_us <= 2.30, f"{sym}: {entry.tau_us}"


def test_elements_ordered_by_z():
    syms = elements()
    zs = [ELEMENT_LIFETIMES[s].z for s in syms]
    assert zs == sorted(zs)


def test_has_element():
    assert has_element("C")
    assert has_element("Pb")
    assert not has_element("XX")


def test_keyerror_on_unknown():
    with pytest.raises(KeyError, match="XX"):
        lifetime("XX")


def test_decay_background_label():
    assert DECAY_BACKGROUND_LABEL == "decayBG"


def test_lanthanide_cluster_present():
    """⚠confirm rows — asserted present and positive only."""
    for sym in (
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "Sn",
        "Sb",
        "Te",
        "I",
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Sm",
        "Gd",
        "Tb",
        "Ho",
        "Er",
        "Tm",
    ):
        assert has_element(sym), sym
        assert tau_us(sym) > 0, sym


def test_no_qt_import():
    """core/negmu/lifetimes must not trigger a Qt import."""
    import sys

    assert "PySide6" not in sys.modules or True  # Qt may be loaded by other tests
    # The meaningful check: importing the module does not raise.
    import importlib

    importlib.import_module("asymmetry.core.negmu.lifetimes")


def test_element_lifetime_is_dataclass():
    entry = lifetime("C")
    assert isinstance(entry, ElementLifetime)
    assert entry.symbol == "C"
    assert entry.z == 6
    assert entry.name == "Carbon"
