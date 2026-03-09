"""Tests for Parameter and ParameterSet behavior."""

from __future__ import annotations

from asymmetry.core.fitting.parameters import Parameter, ParameterSet


def test_parameter_is_constrained_for_fixed_or_expression() -> None:
    free = Parameter(name="A0", value=1.0)
    fixed = Parameter(name="Lambda", value=0.1, fixed=True)
    expr = Parameter(name="sigma", value=0.2, expr="Lambda")

    assert free.is_constrained is False
    assert fixed.is_constrained is True
    assert expr.is_constrained is True


def test_parameter_set_basic_operations() -> None:
    ps = ParameterSet([
        Parameter(name="A0", value=1.0),
        Parameter(name="Lambda", value=0.2, fixed=True),
    ])

    assert len(ps) == 2
    assert "A0" in ps
    assert ps["A0"].value == 1.0
    assert ps.names == ["A0", "Lambda"]
    assert ps.values_array() == [1.0, 0.2]

    free_names = [p.name for p in ps.free_parameters]
    assert free_names == ["A0"]

    ps.update_values({"A0": 3.5, "missing": 10.0})
    assert ps["A0"].value == 3.5
