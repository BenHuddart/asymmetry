"""Tests for composite fit-function model construction and evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.fitting.composite import (
    COMPONENTS,
    CompositeModel,
    has_legacy_fraction_values,
    migrate_legacy_fraction_parameter_entries,
    migrate_legacy_fraction_state,
    migrate_legacy_fraction_values,
)
from asymmetry.core.utils.constants import GAUSS_TO_TESLA, MUON_GYROMAGNETIC_RATIO_MHZ_PER_T


def test_single_component_matches_baseline_free_model() -> None:
    t = np.linspace(0.0, 5.0, 100)
    model = CompositeModel(["Exponential"])

    out = model.function(t, A_1=2.0, Lambda=0.3)
    expected = COMPONENTS["Exponential"].function(t, A=2.0, Lambda=0.3)

    assert np.allclose(out, expected)


def test_addition_and_multiplication_work() -> None:
    t = np.linspace(0.0, 2.0, 20)

    add_model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    add_out = add_model.function(t, A_1=1.0, Lambda=0.5, A_bg=0.2)
    assert np.allclose(add_out, COMPONENTS["Exponential"].function(t, A=1.0, Lambda=0.5) + 0.2)

    mul_model = CompositeModel(["Exponential", "Gaussian"], operators=["*"])
    mul_out = mul_model.function(t, A_1=0.8, Lambda=0.5, sigma=0.25)
    exp_vals = COMPONENTS["Exponential"].function(t, A=1.0, Lambda=0.5)
    gauss_vals = COMPONENTS["Gaussian"].function(t, A=1.0, sigma=0.25)
    expected = 0.8 * exp_vals * gauss_vals
    assert np.allclose(mul_out, expected)


def test_operator_precedence_is_respected() -> None:
    t = np.linspace(0.0, 1.0, 10)
    model = CompositeModel(["Constant", "Constant", "Constant"], operators=["+", "*"])
    out = model.function(t, A_bg_1=1.0, A_bg_2=2.0, A_bg_3=3.0)

    # 1 + (2 * 3)
    assert np.allclose(out, np.full_like(t, 7.0))


def test_duplicate_components_have_unique_parameter_names() -> None:
    model = CompositeModel(["Exponential", "Exponential"], operators=["+"])

    assert "A_1" in model.param_names
    assert "Lambda_1" in model.param_names
    assert "A_2" in model.param_names
    assert "Lambda_2" in model.param_names


def test_unique_symbols_are_not_indexed_except_amplitude() -> None:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    assert "A_1" in model.param_names
    assert "Lambda" in model.param_names
    assert "A_bg" in model.param_names


def test_multiplicative_chain_uses_single_amplitude_parameter() -> None:
    model = CompositeModel(["Exponential", "Gaussian", "Oscillatory"], operators=["*", "*"])

    assert "A_1" in model.param_names
    assert "A_2" not in model.param_names
    assert "A_3" not in model.param_names


def test_formula_and_serialization_round_trip() -> None:
    model = CompositeModel(["Oscillatory", "Constant"], operators=["-"])
    formula = model.formula_string()

    assert "frequency" in formula
    assert "A_bg" in formula

    payload = model.to_dict()
    restored = CompositeModel.from_dict(payload)

    assert restored.component_names == model.component_names
    assert restored.operators == model.operators


def test_component_expression_round_trip() -> None:
    model = CompositeModel.from_expression("Gaussian * ( Constant + Constant )")

    assert model.component_names == ["Gaussian", "Constant", "Constant"]
    assert model.operators == ["*", "+"]
    assert model.open_parentheses == [0, 1, 0]
    assert model.close_parentheses == [0, 0, 1]
    assert model.component_expression_string() == "Gaussian * (Constant + Constant)"


def test_fraction_group_expression_round_trip() -> None:
    model = CompositeModel.from_expression("( Exponential + Gaussian + Constant ){frac}")

    assert model.component_names == ["Exponential", "Gaussian", "Constant"]
    assert model.operators == ["+", "+"]
    assert model.open_parentheses == [1, 0, 0]
    assert model.close_parentheses == [0, 0, 1]
    assert model.fraction_groups == [(0, 2)]
    assert model.component_expression_string() == "(Exponential + Gaussian + Constant){frac}"


def test_fraction_group_rejects_non_additive_group() -> None:
    with pytest.raises(ValueError, match="additive"):
        CompositeModel.from_expression("( Exponential * Gaussian ){frac}")


def test_fraction_group_accepts_sum_of_products() -> None:
    t = np.linspace(0.0, 1.0, 5)
    model = CompositeModel.from_expression(
        "( Exponential * Gaussian + Exponential * Gaussian ){frac}"
    )

    # Two additive terms -> a single free fraction (n-1); the second term is the
    # derived remainder with no parameter.
    assert model.fraction_parameter_groups() == [["f_Exponential"]]
    assert model.derived_fraction_names() == ["f_Exponential_2"]
    assert model.param_names == [
        "A_1",
        "Lambda_1",
        "f_Exponential",
        "sigma_1",
        "Lambda_2",
        "sigma_2",
    ]
    assert "Lambda_1" in model.formula_string()
    assert "sigma_1" in model.formula_string()
    assert "Lambda_2" in model.formula_string()
    assert "sigma_2" in model.formula_string()

    # Free fraction = 0.25 -> remainder weight 0.75, no normalization.
    out = model.function(
        t,
        A_1=2.0,
        Lambda_1=0.3,
        sigma_1=0.2,
        Lambda_2=0.6,
        sigma_2=0.4,
        f_Exponential=0.25,
    )
    expected = 2.0 * (
        0.25 * np.exp(-0.3 * t) * np.exp(-((0.2 * t) ** 2))
        + 0.75 * np.exp(-0.6 * t) * np.exp(-((0.4 * t) ** 2))
    )

    assert np.allclose(out, expected)


def test_fraction_weights_reports_free_and_derived_remainder() -> None:
    # The n-1 free fractions carry their clamped values; the remainder carries
    # 1 - Σ free under its derived display name. No sum-normalization.
    model = CompositeModel.from_expression("( Oscillatory + Oscillatory + Oscillatory ){frac}")
    weights = model.fraction_weights({"f_Oscillatory": 0.5, "f_Oscillatory_2": 0.2})
    assert weights == pytest.approx(
        {
            "f_Oscillatory": 0.5,
            "f_Oscillatory_2": 0.2,
            "f_Oscillatory_3": 0.3,
        }
    )
    assert sum(weights.values()) == pytest.approx(1.0)


def test_fraction_weights_clamps_remainder_when_free_exceeds_one() -> None:
    # Free fractions summing above 1 drive the remainder to its 0 floor; free
    # values themselves are clamped into [0, 1].
    model = CompositeModel.from_expression("( Oscillatory + Oscillatory + Oscillatory ){frac}")
    weights = model.fraction_weights({"f_Oscillatory": 0.7, "f_Oscillatory_2": 0.6})
    assert weights == pytest.approx(
        {"f_Oscillatory": 0.7, "f_Oscillatory_2": 0.6, "f_Oscillatory_3": 0.0}
    )


def test_fraction_weights_skips_group_with_missing_free_fraction() -> None:
    # A group is skipped entirely when any of its free fractions is absent, so
    # callers never receive a partial partition.
    model = CompositeModel.from_expression("( Oscillatory + Oscillatory + Oscillatory ){frac}")
    assert model.fraction_weights({"f_Oscillatory": 0.5}) == {}


def test_with_default_fraction_groups_wraps_top_level_additive_expression() -> None:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])

    grouped_model = model.with_default_fraction_groups()

    assert grouped_model is not model
    assert grouped_model.fraction_groups == [(0, 1)]
    assert grouped_model.component_expression_string() == "(Exponential + Constant){frac}"
    # One free fraction (n-1); Constant remainder has no parameter.
    assert grouped_model.param_names == ["A_1", "Lambda", "f_Exponential"]
    assert grouped_model.derived_fraction_names() == ["f_Constant"]


def test_to_model_definition_callable() -> None:
    t = np.linspace(0.0, 1.0, 8)
    model = CompositeModel(["Constant"])
    definition = model.to_model_definition()

    out = definition.function(t, A_bg=0.4)
    assert np.allclose(out, np.full_like(t, 0.4))


def test_additive_component_indices_excludes_non_additive_terms() -> None:
    model = CompositeModel(
        ["Exponential", "Constant", "Gaussian", "Constant"],
        operators=["+", "*", "+"],
    )
    assert model.additive_component_indices() == [0, 1, 3]


def test_evaluate_components_returns_named_component_curves() -> None:
    t = np.linspace(0.0, 2.0, 25)
    model = CompositeModel(
        ["Exponential", "Constant", "Gaussian"],
        operators=["+", "*"],
    )

    curves = model.evaluate_components(
        t,
        A_1=1.2,
        Lambda=0.3,
        A_bg=0.1,
        A_2=0.8,
        sigma=0.4,
    )

    assert [name for name, _vals in curves] == ["Exponential", "Constant", "Gaussian"]
    assert all(vals.shape == t.shape for _name, vals in curves)


def test_evaluate_components_additive_only_filters_multiplicative_terms() -> None:
    t = np.linspace(0.0, 2.0, 25)
    model = CompositeModel(
        ["Exponential", "Constant", "Gaussian", "Constant"],
        operators=["+", "*", "+"],
    )

    curves = model.evaluate_components(
        t,
        additive_only=True,
        A_1=1.0,
        Lambda=0.2,
        A_bg_2=0.15,
        A_3=0.9,
        sigma=0.5,
        A_bg_4=0.03,
    )

    assert [name for name, _vals in curves] == ["Exponential", "Constant", "Constant"]


def test_formula_string_shows_single_amplitude_for_multiplicative_chain() -> None:
    model = CompositeModel(["Exponential", "Gaussian"], operators=["*"])
    formula = model.formula_string()

    assert "A_1*exp(-Lambda*t)" in formula
    assert "exp(-(sigma*t)^2)" in formula
    assert "1*exp(-(sigma*t)^2)" not in formula


def test_oscillatory_field_matches_frequency_component() -> None:
    t = np.linspace(0.0, 6.0, 200)
    field_gauss = 750.0
    amplitude = 0.7
    phase = 0.2

    frequency_mhz = MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * GAUSS_TO_TESLA * field_gauss

    by_field = COMPONENTS["OscillatoryField"].function(
        t,
        A=amplitude,
        field=field_gauss,
        phase=phase,
    )
    by_frequency = COMPONENTS["Oscillatory"].function(
        t,
        A=amplitude,
        frequency=frequency_mhz,
        phase=phase,
    )

    assert np.allclose(by_field, by_frequency)


def test_parenthesized_expression_changes_evaluation_order() -> None:
    t = np.linspace(0.0, 1.0, 10)

    # Under suppression rules for multiplied singletons: 1 * (3 + 4) = 7
    model = CompositeModel(
        ["Constant", "Constant", "Constant"],
        operators=["*", "+"],
        open_parentheses=[0, 1, 0],
        close_parentheses=[0, 0, 1],
    )
    out = model.function(t, A_bg_2=3.0, A_bg_3=4.0)

    assert np.allclose(out, np.full_like(t, 7.0))


def test_parenthesized_model_serialization_round_trip() -> None:
    model = CompositeModel(
        ["Exponential", "Constant", "Gaussian"],
        operators=["*", "+"],
        open_parentheses=[0, 1, 0],
        close_parentheses=[0, 0, 1],
    )

    restored = CompositeModel.from_dict(model.to_dict())
    assert restored.component_names == model.component_names
    assert restored.operators == model.operators
    assert restored.open_parentheses == model.open_parentheses
    assert restored.close_parentheses == model.close_parentheses


def test_fraction_group_serialization_round_trip() -> None:
    model = CompositeModel.from_expression("( Exponential + Gaussian ){frac}")

    restored = CompositeModel.from_dict(model.to_dict())
    assert restored.fraction_groups == [(0, 1)]
    assert restored.component_expression_string() == "(Exponential + Gaussian){frac}"


def test_fraction_names_disambiguate_duplicate_components() -> None:
    # Repeated component base names get bare then _2, _3 across the whole model;
    # the derived remainder continues the same scheme.
    model = CompositeModel.from_expression("( Gaussian + Gaussian + Gaussian ){frac}")
    assert model.fraction_parameter_groups() == [["f_Gaussian", "f_Gaussian_2"]]
    assert model.derived_fraction_names() == ["f_Gaussian_3"]
    # The duplicated non-fraction params still get distinct names.
    assert [p for p in model.param_names if p.startswith("sigma")] == [
        "sigma_1",
        "sigma_2",
        "sigma_3",
    ]


def test_fraction_names_span_multiple_groups() -> None:
    model = CompositeModel.from_expression(
        "( Exponential + Gaussian ){frac} + ( Gaussian + Exponential ){frac}"
    )
    # f_Exponential (group 1 free) then f_Gaussian (group 2 free, bare because
    # group 1's f_Gaussian is only a display-only remainder, not a real param).
    assert model.fraction_parameter_groups() == [["f_Exponential"], ["f_Gaussian"]]
    assert model.derived_fraction_names() == ["f_Gaussian_2", "f_Exponential_2"]
    assert model.derived_fraction_terms() == [
        ("f_Gaussian_2", (0, 1)),
        ("f_Exponential_2", (2, 3)),
    ]


def test_formula_string_renders_remainder_weight_explicitly() -> None:
    model = CompositeModel.from_expression("( Oscillatory + Oscillatory + Oscillatory ){frac}")
    formula = model.formula_string()
    # Free terms carry their parameter; the remainder shows (1-f_X-f_Y).
    assert "f_Oscillatory*" in formula
    assert "f_Oscillatory_2*" in formula
    assert "(1-f_Oscillatory-f_Oscillatory_2)*" in formula
    # The remainder is not exposed as a named parameter.
    assert "f_Oscillatory_3" not in formula


def test_normalized_parameter_values_clamps_free_fractions() -> None:
    model = CompositeModel.from_expression("( Exponential + Gaussian ){frac}")
    out = model.normalized_parameter_values(
        {"A_1": 5.0, "Lambda": 0.4, "sigma": 0.3, "f_Exponential": 1.4}
    )
    # Free fraction clamped into [0, 1]; every other entry passes through.
    assert out["f_Exponential"] == 1.0
    assert out["A_1"] == 5.0
    assert out["Lambda"] == 0.4
    assert out["sigma"] == 0.3


def test_migrate_legacy_fraction_values_round_trip() -> None:
    model = CompositeModel.from_expression("( Exponential + Exponential + Constant ){frac}")
    legacy = {
        "A_1": 12.0,
        "Lambda_1": 0.2,
        "Lambda_2": 0.5,
        "fraction_1": 1.0,
        "fraction_2": 1.0,
        "fraction_3": 2.0,
    }
    assert has_legacy_fraction_values(model, legacy)

    migrated = migrate_legacy_fraction_values(model, legacy)
    # Old normalized weights [0.25, 0.25, 0.5] -> first n-1 become the free
    # parameters; the last is dropped (now the derived remainder).
    assert migrated["f_Exponential"] == pytest.approx(0.25)
    assert migrated["f_Exponential_2"] == pytest.approx(0.25)
    assert "fraction_1" not in migrated
    assert "fraction_2" not in migrated
    assert "fraction_3" not in migrated
    # Non-fraction keys pass through untouched.
    assert migrated["A_1"] == 12.0
    assert migrated["Lambda_1"] == 0.2

    # The migrated free values reproduce the old evaluation exactly.
    t = np.linspace(0.0, 1.0, 5)
    out = model.function(t, **migrated)
    expected = 12.0 * (0.25 * np.exp(-0.2 * t) + 0.25 * np.exp(-0.5 * t) + 0.5 * np.ones_like(t))
    assert np.allclose(out, expected)


def test_migrate_legacy_fraction_values_zero_sum_uses_equal_weights() -> None:
    model = CompositeModel.from_expression("( Exponential + Exponential + Constant ){frac}")
    migrated = migrate_legacy_fraction_values(
        model, {"fraction_1": 0.0, "fraction_2": 0.0, "fraction_3": 0.0}
    )
    assert migrated["f_Exponential"] == pytest.approx(1.0 / 3.0)
    assert migrated["f_Exponential_2"] == pytest.approx(1.0 / 3.0)


def test_migrate_legacy_fraction_values_malformed_values_treated_as_zero() -> None:
    # A corrupted legacy project can carry None or a non-numeric string for a
    # fraction value; migration must never raise (TypeError/ValueError), and
    # that weight is treated as 0.0 so the others normalize consistently.
    model = CompositeModel.from_expression("( Exponential + Exponential + Constant ){frac}")
    legacy = {
        "fraction_1": None,
        "fraction_2": "garbage",
        "fraction_3": 2.0,
    }
    migrated = migrate_legacy_fraction_values(model, legacy)
    # fraction_1 and fraction_2 both coerce to 0.0; only fraction_3 contributes,
    # so the normalized weights are [0, 0, 1] and the first n-1 free params are 0.
    assert migrated["f_Exponential"] == pytest.approx(0.0)
    assert migrated["f_Exponential_2"] == pytest.approx(0.0)
    assert "fraction_1" not in migrated
    assert "fraction_2" not in migrated
    assert "fraction_3" not in migrated


def test_migrate_legacy_fraction_parameter_entries_malformed_value_does_not_raise() -> None:
    model = CompositeModel.from_expression("( Exponential + Gaussian ){frac}")
    entries = [
        {"name": "A_1", "value": 20.0},
        {"name": "fraction_1", "value": None, "fixed": False},
        {"name": "fraction_2", "value": "garbage", "fixed": False},
    ]
    migrated = migrate_legacy_fraction_parameter_entries(model, entries)
    names = [entry["name"] for entry in migrated]
    assert names == ["A_1", "f_Exponential"]
    by_name = {entry["name"]: entry for entry in migrated}
    # Both legacy values coerce to 0.0 -> equal-weight fallback (sum <= 1e-30).
    assert by_name["f_Exponential"]["value"] == pytest.approx(0.5)


def test_has_legacy_fraction_values_false_for_new_scheme() -> None:
    model = CompositeModel.from_expression("( Exponential + Gaussian ){frac}")
    new_values = {"A_1": 1.0, "Lambda": 0.3, "sigma": 0.2, "f_Exponential": 0.4}
    assert not has_legacy_fraction_values(model, new_values)
    # A migration on new-scheme values is a no-op passthrough.
    assert migrate_legacy_fraction_values(model, new_values) == new_values


def test_migrate_legacy_fraction_parameter_entries_renames_and_drops() -> None:
    model = CompositeModel.from_expression("( Exponential + Gaussian + Constant ){frac}")
    entries = [
        {"name": "A_1", "value": 20.0, "fixed": False, "min": "0", "max": "inf"},
        {"name": "Lambda", "value": 0.5, "fixed": False},
        {"name": "fraction_1", "value": 2.0, "fixed": False, "min": "0", "max": "1"},
        {"name": "sigma", "value": 0.3, "fixed": False},
        {"name": "fraction_2", "value": 1.0, "fixed": True, "min": "0", "max": "1"},
        {"name": "fraction_3", "value": 1.0, "fixed": False},
    ]
    migrated = migrate_legacy_fraction_parameter_entries(model, entries)
    names = [entry["name"] for entry in migrated]
    # Legacy fraction_1/2 → free names (order preserved); fraction_3 dropped.
    assert names == ["A_1", "Lambda", "f_Exponential", "sigma", "f_Gaussian"]
    by_name = {entry["name"]: entry for entry in migrated}
    # Raw [2, 1, 1] normalise to [0.5, 0.25, 0.25]; free params take the first n-1.
    assert by_name["f_Exponential"]["value"] == pytest.approx(0.5)
    assert by_name["f_Gaussian"]["value"] == pytest.approx(0.25)
    # Renamed entries keep their metadata (the fixed flag / bounds).
    assert by_name["f_Gaussian"]["fixed"] is True
    assert by_name["f_Exponential"]["min"] == "0"
    # No legacy keys survive.
    assert not any(name.startswith("fraction_") for name in names)


def test_migrate_legacy_fraction_parameter_entries_noop_for_new_scheme() -> None:
    model = CompositeModel.from_expression("( Exponential + Gaussian ){frac}")
    entries = [
        {"name": "A_1", "value": 1.0},
        {"name": "f_Exponential", "value": 0.4},
    ]
    migrated = migrate_legacy_fraction_parameter_entries(model, entries)
    assert [e["name"] for e in migrated] == ["A_1", "f_Exponential"]
    assert migrated == entries


def test_migrate_legacy_fraction_state_round_trip() -> None:
    model = CompositeModel.from_expression("( Exponential + Gaussian + Constant ){frac}")
    state = {
        "composite_model": model.to_dict(),
        "parameters": [
            {"name": "A_1", "value": 20.0},
            {"name": "Lambda", "value": 0.5},
            {"name": "fraction_1", "value": 2.0},
            {"name": "sigma", "value": 0.3},
            {"name": "fraction_2", "value": 1.0},
            {"name": "fraction_3", "value": 1.0},
        ],
        "result_html": "kept",
    }
    migrated = migrate_legacy_fraction_state(state)
    names = [entry["name"] for entry in migrated["parameters"]]
    assert names == ["A_1", "Lambda", "f_Exponential", "sigma", "f_Gaussian"]
    # Other keys untouched; the model payload is not rewritten.
    assert migrated["result_html"] == "kept"
    assert migrated["composite_model"] == model.to_dict()


def test_migrate_legacy_fraction_state_skips_malformed_model() -> None:
    state = {"composite_model": {"bogus": True}, "parameters": [{"name": "fraction_1"}]}
    migrated = migrate_legacy_fraction_state(state)
    # No usable model → state returned unchanged (parameters shallow-copied).
    assert migrated["parameters"] == [{"name": "fraction_1"}]


def test_fraction_group_uses_group_amplitude_and_free_fraction_parameters() -> None:
    model = CompositeModel.from_expression("( Exponential + Gaussian ){frac}")

    assert "A_1" in model.param_names
    assert "Lambda" in model.param_names
    assert "sigma" in model.param_names
    # One free fraction for the first term; Gaussian remainder has no parameter.
    assert "f_Exponential" in model.param_names
    assert model.derived_fraction_names() == ["f_Gaussian"]
    assert "f_Gaussian" not in model.param_names
    assert "A_2" not in model.param_names


def test_fraction_group_evaluation_uses_derived_remainder_weight() -> None:
    t = np.linspace(0.0, 1.0, 5)
    model = CompositeModel.from_expression("( Exponential + Exponential + Constant ){frac}")

    # Two free fractions 0.25, 0.25 -> Constant remainder weighs 1 - 0.5 = 0.5.
    out = model.function(
        t,
        A_1=12.0,
        Lambda_1=0.2,
        Lambda_2=0.5,
        f_Exponential=0.25,
        f_Exponential_2=0.25,
    )
    expected = 12.0 * (0.25 * np.exp(-0.2 * t) + 0.25 * np.exp(-0.5 * t) + 0.5 * np.ones_like(t))

    assert np.allclose(out, expected)


def test_fraction_group_default_free_fractions_are_one_over_n() -> None:
    model = CompositeModel.from_expression("( Exponential + Exponential + Constant ){frac}")
    # Three additive terms -> each free fraction defaults to 1/3.
    assert model.param_defaults["f_Exponential"] == pytest.approx(1.0 / 3.0)
    assert model.param_defaults["f_Exponential_2"] == pytest.approx(1.0 / 3.0)


def test_fraction_group_preserves_outer_multiplicative_amplitude_suppression() -> None:
    model = CompositeModel.from_expression("Oscillatory * ( Exponential + Gaussian ){frac}")

    assert "A_1" not in model.param_names
    assert "A_2" in model.param_names
    assert "f_Exponential" in model.param_names
    assert model.derived_fraction_names() == ["f_Gaussian"]


def test_parenthesized_product_suppresses_leading_amplitude() -> None:
    model = CompositeModel(
        ["Exponential", "Exponential", "Exponential"],
        operators=["*", "+"],
        open_parentheses=[0, 1, 0],
        close_parentheses=[0, 0, 1],
    )

    assert "A_1" not in model.param_names
    assert "A_2" in model.param_names
    assert "A_3" in model.param_names

    t = np.linspace(0.0, 1.0, 5)
    kwargs = {
        "Lambda_1": 0.2,
        "A_2": 2.0,
        "Lambda_2": 0.3,
        "A_3": 3.0,
        "Lambda_3": 0.4,
    }
    out = model.function(t, **kwargs)
    exp1 = np.exp(-0.2 * t)
    exp2 = np.exp(-0.3 * t)
    exp3 = np.exp(-0.4 * t)
    expected = exp1 * (2.0 * exp2 + 3.0 * exp3)
    assert np.allclose(out, expected)


def test_parenthesized_product_suppresses_trailing_amplitude() -> None:
    model = CompositeModel(
        ["Exponential", "Exponential", "Exponential"],
        operators=["+", "*"],
        open_parentheses=[1, 0, 0],
        close_parentheses=[0, 1, 0],
    )

    assert "A_1" in model.param_names
    assert "A_2" in model.param_names
    assert "A_3" not in model.param_names

    t = np.linspace(0.0, 1.0, 5)
    kwargs = {
        "A_1": 2.0,
        "Lambda_1": 0.2,
        "A_2": 3.0,
        "Lambda_2": 0.3,
        "Lambda_3": 0.4,
    }
    out = model.function(t, **kwargs)
    exp1 = np.exp(-0.2 * t)
    exp2 = np.exp(-0.3 * t)
    exp3 = np.exp(-0.4 * t)
    expected = (2.0 * exp1 + 3.0 * exp2) * exp3
    assert np.allclose(out, expected)


def test_multiplying_two_additive_groups_keeps_group_amplitudes() -> None:
    model = CompositeModel(
        ["Exponential", "Exponential", "Exponential", "Exponential"],
        operators=["+", "*", "+"],
        open_parentheses=[1, 0, 1, 0],
        close_parentheses=[0, 1, 0, 1],
    )

    assert "A_1" in model.param_names
    assert "A_2" in model.param_names
    assert "A_3" in model.param_names
    assert "A_4" in model.param_names


def test_lhs_additive_group_with_constant_suppresses_rhs_amplitude() -> None:
    model = CompositeModel(
        ["Gaussian", "Constant", "OscillatoryField"],
        operators=["+", "*"],
        open_parentheses=[1, 0, 0],
        close_parentheses=[0, 1, 0],
    )

    assert "A_1" in model.param_names
    assert "A_bg" in model.param_names
    assert "A_3" not in model.param_names

    formula = model.formula_string()
    assert "(A_1*exp(-(sigma*t)^2) + A_bg) * cos(2*pi*gamma_mu*field*t + phase)" in formula


def test_lhs_additive_group_suppresses_rhs_constant_amplitude() -> None:
    model = CompositeModel(
        ["Gaussian", "Constant", "Constant"],
        operators=["+", "*"],
        open_parentheses=[1, 0, 0],
        close_parentheses=[0, 1, 0],
    )

    assert "A" in model.param_names
    assert "A_bg_2" in model.param_names
    assert "A_bg_3" not in model.param_names

    formula = model.formula_string()
    assert "* 1" not in formula


def test_rhs_additive_group_suppresses_lhs_constant_amplitude() -> None:
    model = CompositeModel(
        ["Constant", "Gaussian", "Constant"],
        operators=["*", "+"],
        open_parentheses=[0, 1, 0],
        close_parentheses=[0, 0, 1],
    )

    assert "A_bg_1" not in model.param_names
    assert "A" in model.param_names
    assert "A_bg_3" in model.param_names

    formula = model.formula_string()
    assert "1 *" not in formula
