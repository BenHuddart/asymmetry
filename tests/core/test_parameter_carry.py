"""Parameter state follows the component instance across a model change."""

from __future__ import annotations

import pytest

from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.parameter_carry import (
    ComponentParameter,
    FractionWeight,
    GroupAmplitude,
    align_component_names,
    carry_parameter_entries,
    carry_parameter_set,
    carry_parameters,
)
from asymmetry.core.fitting.parameter_models import ParameterCompositeModel
from asymmetry.core.fitting.parameters import AffineTie, Parameter, ParameterSet


def carry_values(
    old: CompositeModel | ParameterCompositeModel,
    new: CompositeModel | ParameterCompositeModel,
    origins: tuple[int | None, ...],
    values: dict[str, float],
) -> dict[str, float]:
    """Carry a plain ``{name: value}`` payload through :func:`carry_parameters`."""
    return carry_parameters(old.parameter_identities(), new.parameter_identities(), origins, values)


def defaults_of(model: CompositeModel | ParameterCompositeModel) -> dict[str, float]:
    """Return the model's default value for every parameter, as a carry payload."""
    return {name: model.param_defaults[name] for name in model.param_names}


def entry(name: str, value: float, **overrides: object) -> dict:
    """Build one entry in the GUI's parameter-state shape."""
    return {
        "name": name,
        "value": value,
        "fixed": False,
        "min": "-inf",
        "max": "inf",
        "uncertainty": None,
        "uncertainty_asymmetric": None,
        "role": None,
        "link_group": None,
        "tie": None,
        **overrides,
    }


# --- identities ------------------------------------------------------------


@pytest.mark.parametrize(
    "expression",
    [
        "Exponential + Constant",
        "Exponential + Exponential + Constant",
        "Gaussian + Exponential + Constant",
        "Oscillatory * Exponential + Constant",
        "(Exponential + Gaussian){frac} + Constant",
        "(Exponential + Gaussian){frac} + Exponential + Constant",
    ],
)
def test_identities_cover_exactly_the_parameter_names(expression: str) -> None:
    model = CompositeModel.from_expression(expression)
    assert list(model.parameter_identities()) == model.param_names


def test_identity_kinds_for_a_fraction_group() -> None:
    model = CompositeModel.from_expression("(Exponential + Gaussian){frac} + Constant")
    assert model.parameter_identities() == {
        "A_1": GroupAmplitude(frozenset({0, 1})),
        "Lambda": ComponentParameter(0, "Lambda"),
        "f_Exponential": FractionWeight(0),
        "sigma": ComponentParameter(1, "sigma"),
        "A_bg": ComponentParameter(2, "A_bg"),
    }


def test_suppressed_unit_amplitude_is_not_an_identity() -> None:
    # In a product the second factor's scale is suppressed, so it is not a
    # parameter and must not appear as one.
    model = CompositeModel.from_expression("Oscillatory * Exponential + Constant")
    identities = model.parameter_identities()
    assert ComponentParameter(1, "A") not in identities.values()
    assert identities["Lambda"] == ComponentParameter(1, "Lambda")


def test_trend_model_identities_are_component_parameters() -> None:
    model = ParameterCompositeModel.from_expression("Linear + Linear")
    assert model.parameter_identities() == {
        "m_1": ComponentParameter(0, "m"),
        "b_1": ComponentParameter(0, "b"),
        "m_2": ComponentParameter(1, "m"),
        "b_2": ComponentParameter(1, "b"),
    }


# --- carry_parameters over the structural edits ----------------------------


def test_append_renames_lambda_and_the_value_follows() -> None:
    old = CompositeModel.from_expression("Exponential + Constant")
    new = CompositeModel.from_expression("Exponential + Exponential + Constant")
    carried = carry_values(old, new, (0, None, 1), {"A_1": 12.0, "Lambda": 3.5, "A_bg": -2.0})

    assert carried == {"A_1": 12.0, "Lambda_1": 3.5, "A_bg": -2.0}
    assert "Lambda_2" not in carried
    assert "A_2" not in carried


def test_insert_before_moves_the_value_to_the_shifted_name() -> None:
    old = CompositeModel.from_expression("Exponential + Constant")
    new = CompositeModel.from_expression("Gaussian + Exponential + Constant")
    carried = carry_values(old, new, (None, 0, 1), {"A_1": 12.0, "Lambda": 3.5, "A_bg": -2.0})

    # A_1 now denotes the inserted Gaussian, so the exponential's amplitude
    # follows the component onto A_2 rather than staying on its old name.
    assert carried == {"A_2": 12.0, "Lambda": 3.5, "A_bg": -2.0}
    assert "A_1" not in carried
    assert "sigma" not in carried


def test_delete_drops_only_the_deleted_component() -> None:
    old = CompositeModel.from_expression("Gaussian + Exponential + Constant")
    new = CompositeModel.from_expression("Gaussian + Constant")
    carried = carry_values(
        old, new, (0, 2), {"A_1": 9.0, "sigma": 0.3, "A_2": 4.0, "Lambda": 1.5, "A_bg": 0.5}
    )

    assert carried == {"A_1": 9.0, "sigma": 0.3, "A_bg": 0.5}


def test_reorder_swaps_the_amplitudes_with_their_components() -> None:
    old = CompositeModel.from_expression("Exponential + Gaussian + Constant")
    new = CompositeModel.from_expression("Gaussian + Exponential + Constant")
    carried = carry_values(
        old, new, (1, 0, 2), {"A_1": 9.0, "Lambda": 1.5, "A_2": 4.0, "sigma": 0.3, "A_bg": 0.5}
    )

    assert carried == {"A_1": 4.0, "sigma": 0.3, "A_2": 9.0, "Lambda": 1.5, "A_bg": 0.5}


def test_duplicate_component_carries_to_both_successors() -> None:
    old = CompositeModel.from_expression("Exponential + Constant")
    new = CompositeModel.from_expression("Exponential + Exponential + Constant")
    carried = carry_values(old, new, (0, 0, 1), {"A_1": 12.0, "Lambda": 3.5, "A_bg": -2.0})

    assert carried == {
        "A_1": 12.0,
        "Lambda_1": 3.5,
        "A_2": 12.0,
        "Lambda_2": 3.5,
        "A_bg": -2.0,
    }


def test_no_structural_change_carries_everything() -> None:
    model = CompositeModel.from_expression("Gaussian + Exponential + Constant")
    state = {"A_1": 9.0, "sigma": 0.3, "A_2": 4.0, "Lambda": 1.5, "A_bg": 0.5}

    assert carry_values(model, model, (0, 1, 2), state) == state


def test_replacing_linear_with_redfield_carries_nothing_for_m() -> None:
    old = ParameterCompositeModel.from_expression("Linear + Constant")
    new = ParameterCompositeModel.from_expression("Redfield + Constant")
    origins = align_component_names(old.component_names, new.component_names)
    assert origins == (None, 1)

    carried = carry_values(old, new, origins, {"m": 0.7, "b": 1.0, "c": 5.0})

    # Linear's slope and Redfield's field exponent are both spelled ``m``;
    # neither is the other, so only the surviving Constant carries.
    assert carried == {"c": 5.0}


def test_second_trend_component_keeps_the_first_component_values() -> None:
    old = ParameterCompositeModel.from_expression("Linear")
    new = ParameterCompositeModel.from_expression("Linear + Linear")
    carried = carry_values(old, new, (0, None), {"m": 0.7, "b": 1.0})

    assert carried == {"m_1": 0.7, "b_1": 1.0}


# --- fraction groups -------------------------------------------------------


def test_fraction_group_amplitude_and_weight_carry_when_the_group_survives() -> None:
    old = CompositeModel.from_expression("(Exponential + Gaussian){frac} + Constant")
    new = CompositeModel.from_expression("(Exponential + Gaussian){frac} + Exponential + Constant")
    carried = carry_values(
        old,
        new,
        (0, 1, None, 2),
        {"A_1": 20.0, "Lambda": 1.5, "f_Exponential": 0.4, "sigma": 0.3, "A_bg": 0.5},
    )

    assert carried == {
        "A_1": 20.0,
        "Lambda_1": 1.5,
        "f_Exponential": 0.4,
        "sigma": 0.3,
        "A_bg": 0.5,
    }


def test_group_amplitude_does_not_carry_when_a_group_member_is_new() -> None:
    old = CompositeModel.from_expression("(Exponential + Gaussian){frac} + Constant")
    new = CompositeModel.from_expression("(Exponential + Gaussian + Gaussian){frac} + Constant")
    carried = carry_values(
        old,
        new,
        (0, 1, None, 2),
        {"A_1": 20.0, "Lambda": 1.5, "f_Exponential": 0.4, "sigma": 0.3, "A_bg": 0.5},
    )

    # The amplitude belongs to the group, and the group is not the same group
    # any more. The weight of a surviving term start still is.
    assert "A_1" not in carried
    assert "f_Gaussian" not in carried
    assert carried == {
        "Lambda": 1.5,
        "f_Exponential": 0.4,
        "sigma_2": 0.3,
        "A_bg": 0.5,
    }


# --- ParameterSet convenience ----------------------------------------------


def test_parameter_set_carries_value_bounds_fix_and_link_group() -> None:
    old = CompositeModel.from_expression("Exponential + Constant")
    new = CompositeModel.from_expression("Exponential + Exponential + Constant")
    parameters = ParameterSet(
        [
            Parameter("A_1", value=12.0, min=0.0, max=30.0, fixed=True, link_group=2),
            Parameter("Lambda", value=3.5, min=0.1, max=9.0),
            Parameter("A_bg", value=-2.0),
        ]
    )

    carried = carry_parameter_set(old, new, (0, None, 1), parameters)

    assert carried.names == ["A_1", "Lambda_1", "A_bg"]
    amplitude = carried["A_1"]
    assert (amplitude.value, amplitude.min, amplitude.max) == (12.0, 0.0, 30.0)
    assert amplitude.fixed is True
    assert amplitude.link_group == 2
    assert carried["Lambda_1"].value == 3.5
    assert carried["Lambda_1"].min == 0.1


def test_parameter_set_retargets_a_tie_onto_the_renamed_parameter() -> None:
    old = CompositeModel.from_expression("Exponential + Constant")
    new = CompositeModel.from_expression("Exponential + Exponential + Constant")
    parameters = ParameterSet(
        [
            Parameter("A_1", value=12.0),
            Parameter("Lambda", value=3.5),
            Parameter("A_bg", value=-2.0, tie=AffineTie(main="Lambda", scale=2.0, const=1.0)),
        ]
    )

    carried = carry_parameter_set(old, new, (0, None, 1), parameters)

    assert carried["A_bg"].tie == AffineTie(main="Lambda_1", scale=2.0, const=1.0)


def test_parameter_set_drops_a_tie_to_a_deleted_parameter() -> None:
    old = CompositeModel.from_expression("Exponential + Gaussian + Constant")
    new = CompositeModel.from_expression("Exponential + Constant")
    parameters = ParameterSet(
        [
            Parameter("A_1", value=12.0),
            Parameter("Lambda", value=3.5),
            Parameter("A_2", value=8.0),
            Parameter("sigma", value=0.3),
            Parameter("A_bg", value=-2.0, tie=AffineTie(main="sigma")),
        ]
    )

    carried = carry_parameter_set(old, new, (0, 2), parameters)

    assert carried.names == ["A_1", "Lambda", "A_bg"]
    assert carried["A_bg"].tie is None


def test_tie_offset_on_an_auxiliary_parameter_survives_a_rename() -> None:
    # ``delta`` is a free auxiliary driving the tie, not a model parameter, so
    # the model edit cannot have moved it: only ``Lambda`` is re-targeted.
    old = CompositeModel.from_expression("Exponential + Constant")
    new = CompositeModel.from_expression("Exponential + Exponential + Constant")
    parameters = ParameterSet(
        [
            Parameter("Lambda", value=3.5),
            Parameter("A_bg", value=-2.0, tie=AffineTie(main="Lambda", offset="delta")),
        ]
    )

    carried = carry_parameter_set(old, new, (0, None, 1), parameters)

    assert carried["A_bg"].tie == AffineTie(main="Lambda_1", offset="delta")


def test_parameter_set_retargets_and_drops_expression_constraints() -> None:
    old = CompositeModel.from_expression("Exponential + Gaussian + Constant")
    new = CompositeModel.from_expression("Exponential + Constant")
    parameters = ParameterSet(
        [
            Parameter("A_1", value=12.0, expr="2 * Lambda"),
            Parameter("Lambda", value=3.5),
            Parameter("A_bg", value=-2.0, expr="A_1 + sigma"),
        ]
    )

    carried = carry_parameter_set(old, new, (0, 2), parameters)

    assert carried["A_1"].expr == "2 * Lambda"
    assert carried["A_bg"].expr is None


def test_parameter_set_leaves_new_parameters_out() -> None:
    old = CompositeModel.from_expression("Exponential + Constant")
    new = CompositeModel.from_expression("Gaussian + Exponential + Constant")

    carried = carry_parameter_set(
        old, new, (None, 0, 1), ParameterSet([Parameter("Lambda", value=3.5)])
    )

    assert carried.names == ["Lambda"]
    assert "sigma" not in carried
    assert "A_1" not in carried


# --- entry-list convenience ------------------------------------------------


def test_entries_carry_onto_the_shifted_names_and_pass_other_keys_through() -> None:
    old = CompositeModel.from_expression("Exponential + Constant")
    new = CompositeModel.from_expression("Gaussian + Exponential + Constant")
    entries = [
        entry("A_1", 12.0, fixed=True, min="0", max="30", role="shared", link_group=3),
        entry("Lambda", 3.5),
        entry("A_bg", -2.0),
    ]

    carried = carry_parameter_entries(old, new, (None, 0, 1), entries)

    assert [item["name"] for item in carried] == ["A_2", "Lambda", "A_bg"]
    amplitude = carried[0]
    assert amplitude["value"] == 12.0
    assert amplitude["fixed"] is True
    assert (amplitude["min"], amplitude["max"]) == ("0", "30")
    assert amplitude["role"] == "shared"
    assert amplitude["link_group"] == 3


def test_carried_entries_clear_the_uncertainty_keys() -> None:
    old = CompositeModel.from_expression("Exponential + Constant")
    new = CompositeModel.from_expression("Exponential + Exponential + Constant")
    entries = [
        entry("Lambda", 3.5, uncertainty=0.2, uncertainty_asymmetric=[-0.2, 0.3]),
    ]

    carried = carry_parameter_entries(old, new, (0, None, 1), entries)

    assert carried == [
        entry("Lambda_1", 3.5, uncertainty=None, uncertainty_asymmetric=None),
    ]
    # The source entry is not mutated.
    assert entries[0]["uncertainty"] == 0.2


def test_entry_ties_are_retargeted_and_dropped() -> None:
    old = CompositeModel.from_expression("Exponential + Gaussian + Constant")
    new = CompositeModel.from_expression("Exponential + Exponential + Gaussian + Constant")
    entries = [
        entry("A_1", 12.0, tie=AffineTie(main="sigma").to_dict()),
        entry("A_2", 8.0, tie=AffineTie(main="A_1", scale=0.5).to_dict()),
        entry("A_bg", -2.0),
    ]

    # A second exponential is inserted at index 1; the Gaussian shifts to 2.
    carried = carry_parameter_entries(old, new, (0, None, 1, 2), entries)
    by_name = {item["name"]: item for item in carried}

    assert by_name["A_1"]["tie"] == AffineTie(main="sigma").to_dict()
    assert by_name["A_3"]["tie"] == AffineTie(main="A_1", scale=0.5).to_dict()
    assert "A_2" not in by_name


def test_entry_tie_is_dropped_when_its_target_is_deleted() -> None:
    old = CompositeModel.from_expression("Exponential + Gaussian + Constant")
    new = CompositeModel.from_expression("Exponential + Constant")
    entries = [entry("A_bg", -2.0, tie=AffineTie(main="sigma").to_dict())]

    carried = carry_parameter_entries(old, new, (0, 2), entries)

    assert carried[0]["name"] == "A_bg"
    assert carried[0]["tie"] is None


def test_entries_without_a_predecessor_are_absent() -> None:
    old = CompositeModel.from_expression("Exponential + Constant")
    new = CompositeModel.from_expression("Gaussian + Exponential + Constant")

    carried = carry_parameter_entries(old, new, (None, 0, 1), [entry("Lambda", 3.5)])

    assert [item["name"] for item in carried] == ["Lambda"]


def test_defaults_of_a_new_component_are_left_for_the_caller_to_seed() -> None:
    old = CompositeModel.from_expression("Exponential + Constant")
    new = CompositeModel.from_expression("Exponential + Exponential + Constant")

    seeds = defaults_of(new)
    seeds.update(carry_values(old, new, (0, None, 1), {"Lambda": 3.5}))

    assert seeds["Lambda_1"] == 3.5
    assert seeds["Lambda_2"] == new.param_defaults["Lambda_2"]


# --- align_component_names -------------------------------------------------


@pytest.mark.parametrize(
    ("old_names", "new_names", "expected"),
    [
        # Append.
        (["Exponential", "Constant"], ["Exponential", "Constant", "Gaussian"], (0, 1, None)),
        # Insert before an existing component.
        (["Exponential", "Constant"], ["Gaussian", "Exponential", "Constant"], (None, 0, 1)),
        # Insert between.
        (["Exponential", "Constant"], ["Exponential", "Gaussian", "Constant"], (0, None, 1)),
        # Delete.
        (["Exponential", "Gaussian", "Constant"], ["Exponential", "Constant"], (0, 2)),
        # A repeated name is continued by its first occurrence.
        (
            ["Exponential", "Constant"],
            ["Exponential", "Exponential", "Constant"],
            (0, None, 1),
        ),
        # Duplicates on both sides pair up in order.
        (
            ["Exponential", "Exponential", "Constant"],
            ["Exponential", "Constant"],
            (0, 2),
        ),
        # Replacement: neither component continues the other.
        (["Linear"], ["Redfield"], (None,)),
        # Nothing to match against.
        ([], ["Exponential"], (None,)),
        (["Exponential"], [], ()),
    ],
)
def test_align_component_names(
    old_names: list[str], new_names: list[str], expected: tuple[int | None, ...]
) -> None:
    assert align_component_names(old_names, new_names) == expected


def test_align_component_names_is_the_identity_for_an_unchanged_model() -> None:
    names = ["Exponential", "Gaussian", "Constant"]

    assert align_component_names(names, names) == (0, 1, 2)


def test_text_mode_alignment_carries_the_surviving_values() -> None:
    old = CompositeModel.from_expression("Exponential + Constant")
    new = CompositeModel.from_expression("Exponential + Gaussian + Constant")
    origins = align_component_names(old.component_names, new.component_names)

    carried = carry_values(old, new, origins, {"A_1": 12.0, "Lambda": 3.5, "A_bg": -2.0})

    assert carried == {"A_1": 12.0, "Lambda": 3.5, "A_bg": -2.0}
