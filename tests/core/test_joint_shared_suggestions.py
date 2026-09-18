"""Autodetection of the default shared table (D7).

Two tiers and nothing else: an ``"exact"`` row is the same full parameter name,
same unit, once in each model, Global in each series; a ``"candidate"`` row is
the same base name at a different component index, or the same component type
where each model instantiates it exactly once. Anything more ambiguous is left
to the user, because a wrong guess here silently constrains two quantities that
are not the same measurement — and the fit will still converge and still look
fine.
"""

from __future__ import annotations

import pytest

from asymmetry.core.fitting import joint
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.joint import suggest_shared_parameters
from asymmetry.core.fitting.parameters import ParamInfo


def _all_global(model: CompositeModel) -> dict[str, str]:
    return {name: "global" for name in model.param_names}


def _by_name(suggestions) -> dict[str, object]:
    return {suggestion.name: suggestion for suggestion in suggestions}


def test_a_background_and_a_rate_shared_by_name_land_in_the_exact_tier() -> None:
    """Same name, same unit, once each, Global each: no room for doubt."""
    ordered = CompositeModel.from_expression("Oscillatory*Exponential + Constant")
    paramagnetic = CompositeModel.from_expression("Exponential + Constant")

    suggestions = _by_name(
        suggest_shared_parameters(
            [ordered, paramagnetic], [_all_global(ordered), _all_global(paramagnetic)]
        )
    )

    assert suggestions["A_bg"].tier == "exact"
    assert suggestions["A_bg"].members == {0: "A_bg", 1: "A_bg"}
    assert "Global" in suggestions["A_bg"].rationale
    assert suggestions["Lambda"].tier == "exact"

    # A parameter only one model has is never proposed.
    assert "frequency" not in suggestions
    assert "phase" not in suggestions


def test_the_same_amplitude_at_a_different_component_index_is_a_candidate() -> None:
    """``A_1`` here and ``A_2`` there: a reasonable inference, offered unticked."""
    first = CompositeModel.from_expression("Exponential + Constant")
    second = CompositeModel.from_expression("Constant + Exponential")
    assert first.param_names == ["A_1", "Lambda", "A_bg"]
    assert second.param_names == ["A_bg", "A_2", "Lambda"]

    suggestions = _by_name(
        suggest_shared_parameters([first, second], [_all_global(first), _all_global(second)])
    )

    assert suggestions["A"].tier == "candidate"
    assert suggestions["A"].members == {0: "A_1", 1: "A_2"}
    assert "A_1" in suggestions["A"].rationale and "A_2" in suggestions["A"].rationale
    # The unambiguous rows are still exact.
    assert suggestions["A_bg"].tier == "exact"
    assert suggestions["Lambda"].tier == "exact"


def test_a_base_name_with_two_instances_in_one_model_is_not_proposed() -> None:
    """Two ``A``-family parameters in one model: which one would the row mean?"""
    two_amplitudes = CompositeModel.from_expression("Gaussian + Exponential + Constant")
    one_amplitude = CompositeModel.from_expression("Constant + Exponential")
    assert two_amplitudes.param_names.count("A_1") == 1
    assert [n for n in two_amplitudes.param_names if n.startswith("A_")] == [
        "A_1",
        "A_2",
        "A_bg",
    ]

    suggestions = _by_name(
        suggest_shared_parameters(
            [two_amplitudes, one_amplitude],
            [_all_global(two_amplitudes), _all_global(one_amplitude)],
        )
    )

    assert "A" not in suggestions
    # The rows that *are* unambiguous still come through.
    assert suggestions["A_bg"].tier == "exact"
    assert suggestions["Lambda"].tier == "exact"


def test_a_unit_mismatch_is_not_proposed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same base name, different units: not the same quantity, whatever it is called."""
    first = CompositeModel.from_expression("Exponential + Constant")
    second = CompositeModel.from_expression("Constant + Exponential")

    real = joint.get_param_info

    def mismatched(name: str) -> ParamInfo:
        if name == "A_2":
            return ParamInfo(name, name, name, name, name, unit="G")
        return real(name)

    monkeypatch.setattr(joint, "get_param_info", mismatched)

    suggestions = _by_name(
        suggest_shared_parameters([first, second], [_all_global(first), _all_global(second)])
    )

    assert "A" not in suggestions
    assert suggestions["A_bg"].tier == "exact"


def test_a_local_role_parameter_is_never_proposed() -> None:
    """Only a series-Global row denotes one value for the whole series (D4)."""
    ordered = CompositeModel.from_expression("Exponential + Constant")
    paramagnetic = CompositeModel.from_expression("Exponential + Constant")
    roles = _all_global(paramagnetic) | {"A_bg": "local"}

    suggestions = _by_name(
        suggest_shared_parameters([ordered, paramagnetic], [_all_global(ordered), roles])
    )

    assert "A_bg" not in suggestions
    assert suggestions["Lambda"].tier == "exact"


def test_one_model_has_nothing_to_share() -> None:
    model = CompositeModel.from_expression("Exponential + Constant")
    assert suggest_shared_parameters([model], [_all_global(model)]) == []


def test_a_role_map_is_required_for_every_model() -> None:
    model = CompositeModel.from_expression("Exponential + Constant")
    with pytest.raises(ValueError, match="one role map per model"):
        suggest_shared_parameters([model, model], [_all_global(model)])


def test_the_same_name_on_different_component_types_is_only_a_candidate() -> None:
    """``A_1`` is Gaussian's amplitude here and Exponential's there: not an exact match."""
    ordered = CompositeModel.from_expression("Gaussian + Constant")
    paramagnetic = CompositeModel.from_expression("Exponential + Constant")
    assert "A_1" in ordered.param_names and "A_1" in paramagnetic.param_names

    suggestions = _by_name(
        suggest_shared_parameters(
            [ordered, paramagnetic], [_all_global(ordered), _all_global(paramagnetic)]
        )
    )

    assert suggestions["A_1"].tier == "candidate"
    assert suggestions["A_1"].members == {0: "A_1", 1: "A_1"}
    assert "Gaussian" in suggestions["A_1"].rationale
    assert "Exponential" in suggestions["A_1"].rationale
    # The background is owned by the same component type in both: still exact.
    assert suggestions["A_bg"].tier == "exact"
