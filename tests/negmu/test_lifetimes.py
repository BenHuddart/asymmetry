"""Source-row selections, uncertainty units and unresolved lifetime authority."""

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
# Stable numeric lookups are separate from direct source-row verification.
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


@pytest.mark.parametrize(
    "symbol,tau_ns,sigma_ns,page,reference",
    [
        ("Rh", 95.8, 0.6, "2219", "49"),
        ("Pd", 96.0, 0.6, "2219", "49"),
        ("Ag", 87.0, 1.5, "2219", "a"),
        ("Ho", 74.9, 0.6, "2220", "54"),
        ("Er", 74.4, 1.5, "2220", "a"),
    ],
)
def test_selected_suzuki_mean_life_rows(symbol, tau_ns, sigma_ns, page, reference):
    """Suzuki PRC 35, 2212 (1987), Table IV mean life, not inverse capture rate."""
    entry = lifetime(symbol)
    assert entry.tau_us == pytest.approx(tau_ns / 1000, abs=1e-12)
    assert entry.sigma_us == pytest.approx(sigma_ns / 1000, abs=1e-12)
    assert entry.status == "selected_measurement"
    assert "Table IV" in entry.source_location
    assert page in entry.source_location
    assert f"ref. {reference}" in entry.source_location


@pytest.mark.parametrize("sym,expected_tau", SPOT_CHECKS)
def test_spot_check_tau(sym, expected_tau):
    assert tau_us(sym) == expected_tau


def test_tl_symbol_guard():
    """Titanium and thallium retain distinct element identities."""
    assert tau_us("Tl") == 0.0704
    assert lifetime("Tl").symbol == "Tl"
    assert lifetime("Tl").z == 81


def test_ne_value_guard():
    """The stored neon default remains unchanged by bounded source selections."""
    assert tau_us("Ne") == 1.461


def test_source_identifier_does_not_certify_unselected_rows():
    assert lifetime("H").source == "SuzukiMeasdayRoalsvig1987"
    assert lifetime("Fe").source == "SuzukiMeasdayRoalsvig1987"
    assert lifetime("Pb").source == "SuzukiMeasdayRoalsvig1987"
    assert lifetime("H").status == lifetime("Fe").status == lifetime("Pb").status == "unverified"


def test_source_field_wimda_provisional():
    assert lifetime("He").source == "WiMDA-provisional"
    assert lifetime("Kr").source == "WiMDA-provisional"
    assert lifetime("Tc").source == "WiMDA-provisional"
    assert lifetime("Re").source == "WiMDA-provisional"


def test_sigma_none_for_provisional():
    for sym in ("He", "Kr", "Tc", "Re", "Os", "Ir", "Pt"):
        assert lifetime(sym).sigma_us is None


def test_tau_range():
    """Every entry's tau_us is strictly within [0.05, 2.30] µs."""
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
    """Existing cluster keys remain available without implying source verification."""
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


def test_element_lifetime_is_dataclass():
    entry = lifetime("C")
    assert isinstance(entry, ElementLifetime)
    assert entry.symbol == "C"
    assert entry.z == 6
    assert entry.name == "Carbon"


def test_selection_status_distinguishes_checked_unconfirmed_and_provisional_rows():
    selected = {
        symbol
        for symbol, entry in ELEMENT_LIFETIMES.items()
        if entry.status == "selected_measurement"
    }
    assert selected == {"Rh", "Pd", "Ag", "Ho", "Er"}
    for symbol in ("Ru", "Tm", "Th", "U", "Np"):
        assert lifetime(symbol).status == "unconfirmed"
        assert lifetime(symbol).source_location is None
    for symbol in ("He", "Kr", "Tc", "Re", "Os", "Ir", "Pt"):
        assert lifetime(symbol).status == "provisional"
        assert lifetime(symbol).source_location is None


def test_selected_notes_preserve_material_and_selection_limits():
    for symbol in ("Rh", "Pd", "Ho"):
        note = lifetime(symbol).selection_note
        assert "isotope composition" in note and "channel" in note and "unspecified" in note
    for symbol in ("Ag", "Er"):
        note = lifetime(symbol).selection_note
        assert "self-supporting" in note and "decay-electron" in note
        assert "isotope composition is unspecified" in note
    assert "alternatives, not averaged" in lifetime("Ag").selection_note


def test_existing_constructor_keeps_backward_compatible_metadata_defaults():
    entry = ElementLifetime("Fe", 26, "Iron", 0.206, 0.001, "unverified-source")
    assert entry.status == "unverified"
    assert entry.source_location is None and entry.selection_note is None


def test_corrected_lifetimes_seed_fixed_parameters_without_implicit_freeing():
    import numpy as np

    from asymmetry.core.negmu.fit import CaptureModelSpec, default_capture_parameters

    symbols = ("Rh", "Pd", "Ag", "Ho", "Er")
    spec = CaptureModelSpec(elements=symbols, include_decay_background=False)
    parameters = default_capture_parameters(spec, time=np.linspace(0, 1, 100), counts=np.ones(100))
    for symbol in symbols:
        assert parameters[f"tau_{symbol}"].value == lifetime(symbol).tau_us
        assert parameters[f"tau_{symbol}"].fixed
