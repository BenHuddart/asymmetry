"""Tests for the fit-component scoping tags (geometry / physics class / cost).

These lock two things: the tag enums and coercers behave (round-trips, bad
tokens, loader-vocabulary mapping), and every built-in component in
``COMPONENTS`` carries a real, non-sentinel set of tags. The ``CUSTOM``-free
check over the whole registry is the enforcement that no built-in slipped
through untagged.
"""

from __future__ import annotations

import pytest

from asymmetry.core.fitting.component_tags import (
    ALL_GEOMETRIES,
    ComputationalCost,
    FieldGeometry,
    PhysicsClass,
    coerce_cost,
    coerce_geometries,
    coerce_physics_classes,
    geometry_from_field_direction,
)
from asymmetry.core.fitting.composite import COMPONENTS

# ── registry-wide tagging invariants ────────────────────────────────────────


def test_every_component_is_tagged_and_custom_free() -> None:
    for name, definition in COMPONENTS.items():
        assert definition.physics_classes, f"{name}: physics_classes is empty"
        assert PhysicsClass.CUSTOM not in definition.physics_classes, (
            f"{name}: still carries the CUSTOM sentinel — built-ins must be tagged"
        )
        assert definition.field_geometries, f"{name}: field_geometries is empty"
        assert all(isinstance(g, FieldGeometry) for g in definition.field_geometries)
        assert all(isinstance(c, PhysicsClass) for c in definition.physics_classes)
        assert isinstance(definition.cost, ComputationalCost), (
            f"{name}: cost not a ComputationalCost"
        )


def test_frequency_domain_components_are_spectral_or_background() -> None:
    allowed = {PhysicsClass.SPECTRAL, PhysicsClass.BACKGROUND}
    freq = {n: d for n, d in COMPONENTS.items() if d.domain == "frequency"}
    assert freq, "expected at least one frequency-domain component"
    for name, definition in freq.items():
        assert definition.physics_classes <= allowed, (
            f"{name}: frequency-domain component tagged {definition.physics_classes}"
        )


# ── spot-check pins ─────────────────────────────────────────────────────────


def test_spot_check_pins() -> None:
    vl = COMPONENTS["VortexLattice"]
    assert vl.field_geometries == frozenset({FieldGeometry.TF})
    assert vl.physics_classes == frozenset({PhysicsClass.SUPERCONDUCTIVITY})
    assert vl.cost is ComputationalCost.EXPENSIVE

    gkt = COMPONENTS["StaticGKT_ZF"]
    assert gkt.field_geometries == frozenset({FieldGeometry.ZF})
    assert gkt.physics_classes == frozenset({PhysicsClass.MAGNETISM})
    assert gkt.cost is ComputationalCost.CHEAP

    fmuf = COMPONENTS["FmuF_General"]
    assert fmuf.field_geometries == frozenset({FieldGeometry.ZF})
    assert fmuf.physics_classes == frozenset({PhysicsClass.MOLECULAR})
    assert fmuf.cost is ComputationalCost.EXPENSIVE

    keren = COMPONENTS["Keren"]
    assert keren.field_geometries == frozenset({FieldGeometry.ZF, FieldGeometry.LF})
    assert keren.cost is ComputationalCost.CHEAP

    bessel = COMPONENTS["Bessel"]
    assert FieldGeometry.ZF in bessel.field_geometries

    constant = COMPONENTS["Constant"]
    assert constant.field_geometries == ALL_GEOMETRIES
    assert constant.physics_classes == frozenset({PhysicsClass.BACKGROUND})


# ── coercers ────────────────────────────────────────────────────────────────


def test_coerce_geometries_round_trip_strings_and_enums() -> None:
    assert coerce_geometries(["ZF", "TF"]) == frozenset({FieldGeometry.ZF, FieldGeometry.TF})
    assert coerce_geometries([FieldGeometry.LF]) == frozenset({FieldGeometry.LF})
    # A bare string is treated as a single token, not iterated char-by-char.
    assert coerce_geometries("ZF") == frozenset({FieldGeometry.ZF})
    assert coerce_geometries(FieldGeometry.TF) == frozenset({FieldGeometry.TF})


def test_coerce_geometries_bad_token_names_offender() -> None:
    with pytest.raises(ValueError, match="XF"):
        coerce_geometries(["ZF", "XF"])


def test_coerce_physics_classes_round_trip_and_bad_token() -> None:
    assert coerce_physics_classes(["magnetism", PhysicsClass.DYNAMICS]) == frozenset(
        {PhysicsClass.MAGNETISM, PhysicsClass.DYNAMICS}
    )
    assert coerce_physics_classes("custom") == frozenset({PhysicsClass.CUSTOM})
    with pytest.raises(ValueError, match="not-a-class"):
        coerce_physics_classes(["not-a-class"])


def test_coerce_cost_round_trip_and_bad_token() -> None:
    assert coerce_cost("cheap") is ComputationalCost.CHEAP
    assert coerce_cost(ComputationalCost.EXPENSIVE) is ComputationalCost.EXPENSIVE
    with pytest.raises(ValueError, match="ludicrous"):
        coerce_cost("ludicrous")


# ── geometry_from_field_direction ───────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Transverse", FieldGeometry.TF),
        ("Longitudinal", FieldGeometry.LF),
        ("Zero field", FieldGeometry.ZF),
        ("transverse", FieldGeometry.TF),
        ("ZERO FIELD", FieldGeometry.ZF),
        ("TF", FieldGeometry.TF),
        ("lf", FieldGeometry.LF),
        ("zf", FieldGeometry.ZF),
        ("", None),
        ("something else", None),
    ],
)
def test_geometry_from_field_direction(text: str, expected: FieldGeometry | None) -> None:
    assert geometry_from_field_direction(text) is expected
