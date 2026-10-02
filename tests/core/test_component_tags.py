"""Tests for the fit-component tags (geometry / physics class / cost / parameter kinds).

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
    ParameterKind,
    PhysicsClass,
    coerce_cost,
    coerce_geometries,
    coerce_parameter_kind,
    coerce_physics_classes,
    geometry_from_field_direction,
)
from asymmetry.core.fitting.composite import (
    COMPONENTS,
    ComponentDefinition,
    CompositeModel,
    placeholder_component_definition,
)
from asymmetry.core.fitting.parameter_carry import GroupAmplitude

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


# ── declared parameter kinds ────────────────────────────────────────────────

#: The kind every built-in parameter declares, by local name. A new parameter
#: name must be added here: there is no default kind and no guess from the name.
_BUILT_IN_KINDS: dict[str, ParameterKind] = {
    "A": ParameterKind.AMPLITUDE,
    "height": ParameterKind.AMPLITUDE,
    "A_bg": ParameterKind.BACKGROUND,
    "bg": ParameterKind.BACKGROUND,
    "slope": ParameterKind.BACKGROUND,
    "phase": ParameterKind.PHASE,
    "beta": ParameterKind.SHAPE,
    "ratio": ParameterKind.SHAPE,
    "w_rel": ParameterKind.SHAPE,
    "Delta": ParameterKind.STATIC_WIDTH,
    "a_L": ParameterKind.STATIC_WIDTH,
    "delta_ex": ParameterKind.STATIC_WIDTH,
    "lambda_ab": ParameterKind.STATIC_WIDTH,
    "fwhm": ParameterKind.STATIC_WIDTH,
    "Lambda": ParameterKind.RATE,
    "sigma": ParameterKind.RATE,
    "lambda_T": ParameterKind.RATE,
    "lambda_L": ParameterKind.RATE,
    "Gamma": ParameterKind.RATE,
    "nu": ParameterKind.RATE,
    "tau_c": ParameterKind.RATE,
    "f_cut": ParameterKind.RATE,
    "frequency": ParameterKind.FREQUENCY,
    "delta_frequency": ParameterKind.FREQUENCY,
    "nu0": ParameterKind.FREQUENCY,
    "field": ParameterKind.FIELD,
    "B_L": ParameterKind.FIELD,
    "B_dip": ParameterKind.FIELD,
    "Bc2": ParameterKind.FIELD,
    "A_hf": ParameterKind.GEOMETRY,
    "D_mu": ParameterKind.GEOMETRY,
    "f_dip": ParameterKind.GEOMETRY,
    "f_quad": ParameterKind.GEOMETRY,
    "J_spin": ParameterKind.GEOMETRY,
    "theta_h": ParameterKind.GEOMETRY,
    "phi_h": ParameterKind.GEOMETRY,
    "theta": ParameterKind.GEOMETRY,
    "phi3": ParameterKind.GEOMETRY,
    "r_muF": ParameterKind.GEOMETRY,
    "r1": ParameterKind.GEOMETRY,
    "r2": ParameterKind.GEOMETRY,
    "r3": ParameterKind.GEOMETRY,
    "r_muH": ParameterKind.GEOMETRY,
    "r_mue": ParameterKind.GEOMETRY,
}


def test_every_built_in_parameter_declares_its_kind() -> None:
    for name, definition in COMPONENTS.items():
        assert definition.param_kinds == {
            pname: _BUILT_IN_KINDS[pname] for pname in definition.param_names
        }, name


def test_component_missing_a_parameter_kind_cannot_be_defined() -> None:
    template = COMPONENTS["Exponential"]
    with pytest.raises(ValueError, match="'Broken'.*Lambda"):
        ComponentDefinition(
            name="Broken",
            description=template.description,
            label="Broken",
            use_when=template.use_when,
            function=template.function,
            param_names=["A", "Lambda"],
            param_defaults=template.param_defaults,
            param_info=template.param_info,
            param_kinds={"A": ParameterKind.AMPLITUDE},
            formula_template=template.formula_template,
        )


def test_model_parameter_kinds_follow_the_component_not_the_name() -> None:
    model = CompositeModel.from_expression("DynamicLorentzianKT + MuoniumTF + Constant")

    kinds = model.parameter_kinds()

    assert list(kinds) == model.param_names
    # Names that only look like amplitudes: a static width and a hyperfine coupling.
    assert kinds["a_L"] is ParameterKind.STATIC_WIDTH
    assert kinds["A_hf"] is ParameterKind.GEOMETRY
    assert kinds["A_1"] is kinds["A_2"] is ParameterKind.AMPLITUDE
    assert kinds["A_bg"] is ParameterKind.BACKGROUND
    assert kinds["B_L"] is kinds["field"] is ParameterKind.FIELD


def test_fraction_group_total_is_an_amplitude_and_its_weights_are_fractions() -> None:
    model = CompositeModel.from_expression("(Gaussian + Exponential){frac} + Constant")

    kinds = model.parameter_kinds()
    identities = model.parameter_identities()

    assert kinds["A_1"] is ParameterKind.AMPLITUDE
    assert isinstance(identities["A_1"], GroupAmplitude)
    assert kinds["f_Gaussian"] is ParameterKind.FRACTION
    assert kinds["sigma"] is kinds["Lambda"] is ParameterKind.RATE
    assert kinds["A_bg"] is ParameterKind.BACKGROUND


def test_placeholder_for_a_missing_component_has_no_parameters_to_declare() -> None:
    assert placeholder_component_definition("Gone").param_kinds == {}


def test_coerce_parameter_kind_round_trip_and_bad_token() -> None:
    assert coerce_parameter_kind("static-width") is ParameterKind.STATIC_WIDTH
    assert coerce_parameter_kind(ParameterKind.RATE) is ParameterKind.RATE
    with pytest.raises(ValueError, match="'speed'"):
        coerce_parameter_kind("speed")


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

    for name in ("OverhauserPowder", "OverhauserPowderCutoff", "OverhauserPowderCentre"):
        overhauser = COMPONENTS[name]
        assert overhauser.field_geometries == frozenset({FieldGeometry.ZF})
        assert overhauser.physics_classes == frozenset({PhysicsClass.MAGNETISM})
        assert overhauser.cost is ComputationalCost.CHEAP

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
