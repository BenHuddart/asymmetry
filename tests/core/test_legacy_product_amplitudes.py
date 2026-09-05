"""Tests for the alpha-only product-amplitude migration.

Deleted at v1.0 together with
``src/asymmetry/core/fitting/legacy_product_amplitudes.py`` and its call sites
(see ``RELEASING.md`` § "Delete at v1.0").
"""

from __future__ import annotations

import math

import pytest

from asymmetry.core.fitting import legacy_product_amplitudes
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.legacy_product_amplitudes import (
    fold_legacy_product_amplitude_entries,
    fold_legacy_product_amplitude_names,
    fold_legacy_product_amplitude_set,
    fold_legacy_product_amplitude_state,
    fold_legacy_product_amplitude_values,
    legacy_parameter_mapping,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.representation import FitSlot

MULTIPLET = "(Oscillatory * Exponential) + (Oscillatory * Exponential) + Constant"


def _multiplet_entries(*, envelope_value: float = 1.0, envelope_fixed: bool = True) -> list[dict]:
    """A pre-policy saved parameter list for :data:`MULTIPLET`.

    The wizard seeded each envelope amplitude (``A_2``/``A_4``) to 1.0 and fixed
    it; the arguments let a test make one of them a genuinely free factor.
    """
    return [
        {"name": "A_1", "value": 0.2, "fixed": False, "min": "-inf", "max": "inf"},
        {"name": "frequency_1", "value": 3.0, "fixed": False},
        {"name": "phase_1", "value": 0.0, "fixed": False},
        {"name": "A_2", "value": envelope_value, "fixed": envelope_fixed},
        {"name": "Lambda_2", "value": 0.5, "fixed": False},
        {"name": "A_3", "value": 0.1, "fixed": False},
        {"name": "frequency_3", "value": 6.0, "fixed": False},
        {"name": "phase_3", "value": 0.0, "fixed": False},
        {"name": "A_4", "value": 1.0, "fixed": True},
        {"name": "Lambda_4", "value": 0.2, "fixed": False},
        {"name": "A_bg", "value": 0.05, "fixed": False},
    ]


def test_legacy_parameter_mapping_reproduces_pre_policy_names() -> None:
    """The frozen copy names amplitudes the way the model did before the policy.

    Parenthesised products kept one ``A`` per component; a flat chain shared one
    ``A`` named after the chain's first component (so ``Constant * Exponential``
    exposed both ``A_bg`` and ``A_1``, and the old evaluator multiplied by both).
    """
    multiplet = legacy_parameter_mapping(CompositeModel.from_expression(MULTIPLET))
    assert [mapping.get("A") for mapping in multiplet[:4]] == ["A_1", "A_2", "A_3", "A_4"]
    assert multiplet[4]["A_bg"] == "A_bg"

    flat = legacy_parameter_mapping(CompositeModel.from_expression("Constant * Exponential"))
    assert flat[0]["A_bg"] == "A_bg"
    assert flat[1]["A"] == "A_1"


def test_pinned_envelope_amplitudes_fold_away_leaving_line_amplitudes() -> None:
    """A multiplet saved with ``A_2 = A_4 = 1`` fixed keeps ``A_1``/``A_3`` as they were."""
    model = CompositeModel.from_expression(MULTIPLET)
    folded = fold_legacy_product_amplitude_entries(model, _multiplet_entries())

    names = [entry["name"] for entry in folded]
    assert "A_2" not in names
    assert "A_4" not in names
    by_name = {entry["name"]: entry for entry in folded}
    assert by_name["A_1"]["value"] == pytest.approx(0.2)
    assert by_name["A_3"]["value"] == pytest.approx(0.1)
    # A pinned unit factor does not make the surviving amplitude fixed.
    assert by_name["A_1"]["fixed"] is False
    # Every non-amplitude entry passes through untouched, in order.
    assert names == [
        "A_1",
        "frequency_1",
        "phase_1",
        "Lambda_2",
        "A_3",
        "frequency_3",
        "phase_3",
        "Lambda_4",
        "A_bg",
    ]
    assert set(names) <= set(model.param_names)


def test_free_envelope_amplitude_folds_value_and_uncertainty_into_survivor() -> None:
    """A free ``A_2 = 0.5`` multiplies into ``A_1``; relative errors add in quadrature."""
    model = CompositeModel.from_expression(MULTIPLET)
    entries = _multiplet_entries(envelope_value=0.5, envelope_fixed=False)
    entries[0]["uncertainty"] = 0.02  # A_1 = 0.2 ± 10%
    entries[3]["uncertainty"] = 0.05  # A_2 = 0.5 ± 10%

    by_name = {
        entry["name"]: entry for entry in fold_legacy_product_amplitude_entries(model, entries)
    }
    assert by_name["A_1"]["value"] == pytest.approx(0.1)
    assert by_name["A_1"]["uncertainty"] == pytest.approx(0.1 * math.sqrt(0.1**2 + 0.1**2))
    assert by_name["A_1"]["fixed"] is False
    # The survivor keeps its own bounds.
    assert by_name["A_1"]["min"] == "-inf"
    assert by_name["A_1"]["max"] == "inf"


def test_survivor_is_fixed_only_when_every_folded_entry_was_fixed() -> None:
    model = CompositeModel.from_expression(MULTIPLET)
    entries = _multiplet_entries(envelope_value=2.0)
    entries[0]["fixed"] = True

    by_name = {
        entry["name"]: entry for entry in fold_legacy_product_amplitude_entries(model, entries)
    }
    assert by_name["A_1"]["fixed"] is True
    assert by_name["A_1"]["value"] == pytest.approx(0.4)


def test_folding_an_entry_without_uncertainty_contributes_none() -> None:
    model = CompositeModel.from_expression(MULTIPLET)
    entries = _multiplet_entries(envelope_value=0.5, envelope_fixed=False)
    entries[0]["uncertainty"] = 0.02

    by_name = {
        entry["name"]: entry for entry in fold_legacy_product_amplitude_entries(model, entries)
    }
    assert by_name["A_1"]["uncertainty"] == pytest.approx(0.1 * 0.1)


def test_lone_parenthesised_amplitude_is_renamed_to_its_component_index() -> None:
    """``(Gaussian + Constant) * Constant`` saved ``A``/``A_bg_2``; now ``A_1``/``A_bg``."""
    model = CompositeModel.from_expression("(Gaussian + Constant) * Constant")
    entries = [
        {"name": "A", "value": 0.25, "fixed": False},
        {"name": "sigma", "value": 0.3, "fixed": False},
        {"name": "A_bg_2", "value": 0.05, "fixed": True},
    ]

    folded = fold_legacy_product_amplitude_entries(model, entries)
    assert [entry["name"] for entry in folded] == ["A_1", "sigma", "A_bg"]
    by_name = {entry["name"]: entry for entry in folded}
    assert by_name["A_1"]["value"] == pytest.approx(0.25)
    # A pure rename keeps the entry's own metadata.
    assert by_name["A_bg"]["value"] == pytest.approx(0.05)
    assert by_name["A_bg"]["fixed"] is True
    assert set(by_name) <= set(model.param_names)


def test_flat_constant_times_exponential_folds_the_chain_amplitude_into_the_background() -> None:
    """The old flat chain multiplied by ``A_bg`` *and* ``A_1``; the fold keeps the product."""
    model = CompositeModel.from_expression("Constant * Exponential")
    entries = [
        {"name": "A_bg", "value": 0.2, "fixed": False},
        {"name": "A_1", "value": 1.5, "fixed": False},
        {"name": "Lambda", "value": 0.4, "fixed": False},
    ]

    folded = fold_legacy_product_amplitude_entries(model, entries)
    assert [entry["name"] for entry in folded] == ["A_bg", "Lambda"]
    assert folded[0]["value"] == pytest.approx(0.3)


def test_leaf_amplitude_beside_an_additive_group_is_distributed_over_the_sum_terms() -> None:
    """A product with a sum factor has no surviving leaf scale of its own.

    ``(Exponential + Gaussian) * Oscillatory * Exponential`` kept ``A_4`` on the
    trailing factor under the old rule (only the factor adjacent to the group
    lost its amplitude). The sum's own terms now carry every scale, so ``A_4``
    (``k``) is distributed onto each term's surviving amplitude instead of
    dropped: ``A_1`` (the ``Exponential`` term) and ``A_2`` (the ``Gaussian``
    term) are each multiplied by ``k``, with ``k``'s relative uncertainty
    added in quadrature to each term's own. Fixedness of ``A_1``/``A_2`` is
    unchanged — they were the user's own parameters, and ``A_4`` (fixed or
    not) is simply absorbed into their values.
    """
    model = CompositeModel.from_expression("(Exponential + Gaussian) * Oscillatory * Exponential")
    entries = [
        {"name": "A_1", "value": 0.2, "fixed": False, "uncertainty": 0.02},
        {"name": "Lambda_1", "value": 0.5, "fixed": False},
        {"name": "A_2", "value": 0.1, "fixed": False, "uncertainty": 0.01},
        {"name": "sigma", "value": 0.3, "fixed": False},
        {"name": "frequency", "value": 3.0, "fixed": False},
        {"name": "phase", "value": 0.0, "fixed": False},
        {"name": "A_4", "value": 2.0, "fixed": False, "uncertainty": 0.2},
        {"name": "Lambda_4", "value": 0.2, "fixed": False},
    ]

    folded = fold_legacy_product_amplitude_entries(model, entries)
    assert [entry["name"] for entry in folded] == [
        "A_1",
        "Lambda_1",
        "A_2",
        "sigma",
        "frequency",
        "phase",
        "Lambda_4",
    ]
    by_name = {entry["name"]: entry for entry in folded}
    assert by_name["A_1"]["value"] == pytest.approx(0.4)  # 0.2 * 2.0
    assert by_name["A_2"]["value"] == pytest.approx(0.2)  # 0.1 * 2.0
    assert by_name["A_1"]["uncertainty"] == pytest.approx(0.4 * math.sqrt(0.1**2 + 0.1**2))
    assert by_name["A_2"]["uncertainty"] == pytest.approx(0.2 * math.sqrt(0.1**2 + 0.1**2))
    assert by_name["A_1"]["fixed"] is False
    assert by_name["A_2"]["fixed"] is False
    assert set(entry["name"] for entry in folded) <= set(model.param_names)


def test_leaf_amplitude_beside_a_fraction_group_is_distributed_onto_the_group_amplitude() -> None:
    """A fraction-group sum represents its terms with one group amplitude.

    ``(Exponential + Gaussian){frac} * Oscillatory * Exponential`` has a
    fraction-group sum as its first factor; the trailing ``Exponential``'s
    legacy ``A_4`` is distributed onto the group amplitude alone (``A_1``,
    named for the group's start component), not onto the group's individual
    (already-suppressed) leaf scales.
    """
    model = CompositeModel.from_expression(
        "(Exponential + Gaussian){frac} * Oscillatory * Exponential"
    )
    entries = [
        {"name": "A_1", "value": 0.3, "fixed": False, "uncertainty": 0.03},
        {"name": "Lambda_1", "value": 0.5, "fixed": False},
        {"name": "f_Exponential", "value": 0.4, "fixed": False},
        {"name": "sigma", "value": 0.3, "fixed": False},
        {"name": "frequency", "value": 3.0, "fixed": False},
        {"name": "phase", "value": 0.0, "fixed": False},
        {"name": "A_4", "value": 2.0, "fixed": False, "uncertainty": 0.2},
        {"name": "Lambda_4", "value": 0.2, "fixed": False},
    ]

    folded = fold_legacy_product_amplitude_entries(model, entries)
    assert [entry["name"] for entry in folded] == [
        "A_1",
        "Lambda_1",
        "f_Exponential",
        "sigma",
        "frequency",
        "phase",
        "Lambda_4",
    ]
    by_name = {entry["name"]: entry for entry in folded}
    assert by_name["A_1"]["value"] == pytest.approx(0.6)  # 0.3 * 2.0
    assert by_name["A_1"]["uncertainty"] == pytest.approx(0.6 * math.sqrt(0.1**2 + 0.1**2))
    assert by_name["A_1"]["fixed"] is False
    assert set(entry["name"] for entry in folded) <= set(model.param_names)


def test_already_migrated_entries_pass_through_without_touching_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Data saved under the policy skips the fold entirely — no legacy mapping, no tree walk."""

    def _fail(_model: CompositeModel) -> list[dict[str, str]]:
        raise AssertionError("the legacy mapping must not be built for migrated data")

    monkeypatch.setattr(legacy_product_amplitudes, "legacy_parameter_mapping", _fail)

    model = CompositeModel.from_expression(MULTIPLET)
    entries = [
        {"name": name, "value": 1.0, "fixed": False}
        for name in ("A_1", "Lambda_2", "A_3", "Lambda_4", "A_bg")
    ]

    folded = fold_legacy_product_amplitude_entries(model, entries)
    assert folded == entries
    assert folded[0] is not entries[0]  # a shallow copy, as the fraction migration returns


def test_values_shape_folds_and_renames() -> None:
    model = CompositeModel.from_expression(MULTIPLET)
    values = {
        "A_1": 0.2,
        "A_2": 0.5,
        "Lambda_2": 0.5,
        "A_3": 0.1,
        "A_4": 1.0,
        "Lambda_4": 0.2,
        "A_bg": 0.05,
    }

    folded = fold_legacy_product_amplitude_values(model, values)
    assert folded == {
        "A_1": pytest.approx(0.1),
        "Lambda_2": 0.5,
        "A_3": pytest.approx(0.1),
        "Lambda_4": 0.2,
        "A_bg": 0.05,
    }
    # Idempotent: the migrated dict has no legacy names left to fold.
    assert fold_legacy_product_amplitude_values(model, folded) == folded


def test_parameter_set_shape_folds_values_bounds_fixedness_and_uncertainties() -> None:
    model = CompositeModel.from_expression(MULTIPLET)
    parameters = ParameterSet(
        [
            Parameter(name="A_1", value=0.2, min=0.0, max=1.0, fixed=False),
            Parameter(name="A_2", value=0.5, fixed=True),
            Parameter(name="Lambda_2", value=0.5),
            Parameter(name="A_3", value=0.1),
            Parameter(name="A_4", value=1.0, fixed=True),
            Parameter(name="Lambda_4", value=0.2),
            Parameter(name="A_bg", value=0.05),
        ]
    )
    uncertainties = {"A_1": 0.02, "A_2": 0.05, "Lambda_2": 0.01}

    folded, folded_uncertainties = fold_legacy_product_amplitude_set(
        model, parameters, uncertainties
    )
    assert folded.names == ["A_1", "Lambda_2", "A_3", "Lambda_4", "A_bg"]
    assert folded["A_1"].value == pytest.approx(0.1)
    assert folded["A_1"].fixed is False
    # The survivor keeps its own bounds.
    assert (folded["A_1"].min, folded["A_1"].max) == (0.0, 1.0)
    assert folded["A_3"].value == pytest.approx(0.1)
    assert folded_uncertainties["A_1"] == pytest.approx(0.1 * math.sqrt(0.1**2 + 0.1**2))
    assert folded_uncertainties["Lambda_2"] == pytest.approx(0.01)
    assert "A_2" not in folded_uncertainties

    # Already migrated: the inputs come back untouched.
    again, again_uncertainties = fold_legacy_product_amplitude_set(
        model, folded, folded_uncertainties
    )
    assert again is folded
    assert again_uncertainties == folded_uncertainties


def test_state_shape_folds_the_saved_form_payload() -> None:
    model = CompositeModel.from_expression(MULTIPLET)
    state = {
        "model_name": "Composite",
        "composite_model": model.to_dict(),
        "parameters": _multiplet_entries(),
        "result_html": "<p>fit</p>",
    }

    folded = fold_legacy_product_amplitude_state(state)
    assert folded["result_html"] == "<p>fit</p>"
    names = [entry["name"] for entry in folded["parameters"]]
    assert "A_2" not in names
    assert "A_4" not in names
    # A payload without a usable model is returned unchanged.
    assert fold_legacy_product_amplitude_state({"parameters": []}) == {"parameters": []}


def test_name_tuple_shape_renames_survivors_and_drops_folded_names() -> None:
    model = CompositeModel.from_expression(MULTIPLET)
    assert fold_legacy_product_amplitude_names(
        model, ("A_1", "A_2", "Lambda_2", "A_3", "A_4", "A_bg")
    ) == ("A_1", "Lambda_2", "A_3", "A_bg")

    renaming = CompositeModel.from_expression("(Gaussian + Constant) * Constant")
    assert fold_legacy_product_amplitude_names(renaming, ("A", "sigma", "A_bg_2")) == (
        "A_1",
        "sigma",
        "A_bg",
    )


def test_fit_slot_round_trip_folds_a_saved_multiplet() -> None:
    """End to end: a pre-policy ``.asymp`` fit slot loads on the current names."""
    model = CompositeModel.from_expression(MULTIPLET)
    slot = FitSlot.from_dict(
        {
            "model": model.to_dict(),
            "parameters": _multiplet_entries(envelope_value=0.5, envelope_fixed=False),
            "provenance": "single",
        }
    )

    names = [entry["name"] for entry in slot.parameters]
    assert names == model.param_names
    by_name = {entry["name"]: entry for entry in slot.parameters}
    assert by_name["A_1"]["value"] == pytest.approx(0.1)
    assert by_name["A_3"]["value"] == pytest.approx(0.1)
