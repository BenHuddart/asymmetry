"""Permanent controls for released fraction parameter formats."""

import numpy as np
import pytest

from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.project.legacy_fractions import (
    has_legacy_fraction_values,
    migrate_legacy_fraction_parameter_entries,
    migrate_legacy_fraction_values,
)


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


def test_migrate_legacy_fraction_values_malformed_values_raise() -> None:
    model = CompositeModel.from_expression("( Exponential + Exponential + Constant ){frac}")
    legacy = {
        "fraction_1": None,
        "fraction_2": "garbage",
        "fraction_3": 2.0,
    }
    with pytest.raises(ValueError, match="stored fraction"):
        migrate_legacy_fraction_values(model, legacy)


def test_migrate_legacy_fraction_parameter_entries_malformed_value_raises() -> None:
    model = CompositeModel.from_expression("( Exponential + Gaussian ){frac}")
    entries = [
        {"name": "A_1", "value": 20.0},
        {"name": "fraction_1", "value": None, "fixed": False},
        {"name": "fraction_2", "value": "garbage", "fixed": False},
    ]
    with pytest.raises(ValueError, match="stored fraction"):
        migrate_legacy_fraction_parameter_entries(model, entries)


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
    from asymmetry.core.project.analysis_migration import fit_editor

    migrated = dict(state)
    fit_editor(migrated)
    names = [entry["name"] for entry in migrated["parameters"]]
    assert names == ["A_1", "Lambda", "f_Exponential", "sigma", "f_Gaussian"]
    # Other keys untouched; the model payload is not rewritten.
    assert migrated["result_html"] == "kept"
    assert migrated["composite_model"] == model.to_dict()


def test_fit_editor_boundary_rejects_malformed_model() -> None:
    from asymmetry.core.project.analysis_migration import fit_editor

    state = {"composite_model": {"bogus": True}, "parameters": [{"name": "fraction_1"}]}
    with pytest.raises(ValueError, match="component_names"):
        fit_editor(state)
