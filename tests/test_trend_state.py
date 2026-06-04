"""Unit tests for the structured TrendState helper (Phase 1)."""

from __future__ import annotations

from asymmetry.core.representation import TrendState


def test_empty_round_trips_to_empty_dict():
    assert TrendState().to_dict() == {}
    assert TrendState().is_empty()
    assert TrendState.from_dict({}).is_empty()
    assert TrendState.from_dict(None).is_empty()


def test_known_fields_round_trip():
    ts = TrendState(
        x_key="field",
        selected_quantities=["lambda", "A"],
        derived_params=[{"name": "ratio", "expr": "A/lambda"}],
        model_fits={"lambda": {"slope": 1.0}},
        axes_state={"y_log": True},
    )
    restored = TrendState.from_dict(ts.to_dict())
    assert restored.x_key == "field"
    assert restored.selected_quantities == ["lambda", "A"]
    assert restored.derived_params == [{"name": "ratio", "expr": "A/lambda"}]
    assert restored.model_fits == {"lambda": {"slope": 1.0}}
    assert restored.axes_state == {"y_log": True}
    assert not restored.is_empty()


def test_unknown_keys_preserved_under_legacy():
    ts = TrendState.from_dict({"x_key": "run", "future_field": 42, "other": [1, 2]})
    assert ts.x_key == "run"
    assert ts.legacy == {"future_field": 42, "other": [1, 2]}
    # Legacy survives the round trip and is not re-wrapped.
    again = TrendState.from_dict(ts.to_dict())
    assert again.legacy == {"future_field": 42, "other": [1, 2]}


def test_to_dict_omits_defaults_but_keeps_legacy():
    ts = TrendState(legacy={"keep": 1})
    assert ts.to_dict() == {"legacy": {"keep": 1}}


def test_malformed_fields_coerce_to_defaults():
    ts = TrendState.from_dict(
        {"x_key": 5, "selected_quantities": "nope", "model_fits": [], "derived_params": {}}
    )
    assert ts.x_key is None
    assert ts.selected_quantities == []
    assert ts.model_fits == {}
    assert ts.derived_params == []
