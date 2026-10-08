"""The note naming a zero-field relaxation that a longitudinal field decoupled."""

from __future__ import annotations

from typing import Any

from asymmetry.core.workflow.decoupling import decoupling_note


def _series(expression: str, rate: str, rows: list[tuple[int, float, float]]) -> dict[str, Any]:
    return {
        "kind": "series",
        "expression": expression,
        "free_params": ["A_1", rate, "A_bg"],
        "results": [
            {
                "run": run,
                "success": True,
                "parameters": {"A_1": 20.0, rate: value, "A_bg": 1.0},
                "uncertainties": {"A_1": 0.1, rate: error, "A_bg": 0.1},
            }
            for run, value, error in rows
        ],
    }


def _survey(field: float, precession: str) -> dict[int, dict[str, Any]]:
    """Runs 1-3 in zero field, runs 11-13 in *field* with the given precession verdict."""
    zero = {run: {"field": 0.0, "precession": "none"} for run in (1, 2, 3)}
    applied = {run: {"field": field, "precession": precession} for run in (11, 12, 13)}
    return zero | applied


_ZF = _series(
    "StaticGKT_ZF + Constant", "Delta", [(1, 0.26, 0.001), (2, 0.26, 0.001), (3, 0.25, 0.001)]
)
_QUENCHED = _series(
    "Exponential + Constant",
    "Lambda",
    [(11, 0.0004, 0.0007), (12, 0.0014, 0.0007), (13, 0.0022, 0.0007)],
)


def test_a_relaxation_a_strong_longitudinal_field_removes_is_named_static_from_both_sides() -> None:
    stored = {"zf": _ZF, "lf": _QUENCHED}
    survey = _survey(100.0, "none")
    note = decoupling_note("lf", stored, survey)
    assert note is not None
    assert "zero-field series zf" in note and "static on the muon time scale" in note
    assert decoupling_note("zf", stored, survey) == note


def test_no_static_reading_when_the_field_or_the_data_cannot_carry_it() -> None:
    # Δ/γ_μ is about 3 G here: 5 G is not enough to decouple a static distribution.
    assert decoupling_note("lf", {"zf": _ZF, "lf": _QUENCHED}, _survey(5.0, "none")) is None
    # A rate the field leaves standing is not decoupled.
    relaxing = _series(
        "Exponential + Constant", "Lambda", [(11, 0.2, 0.01), (12, 0.2, 0.01), (13, 0.2, 0.01)]
    )
    assert decoupling_note("lf", {"zf": _ZF, "lf": relaxing}, _survey(100.0, "none")) is None
    # A Larmor line in the field runs makes the field transverse.
    assert decoupling_note("lf", {"zf": _ZF, "lf": _QUENCHED}, _survey(100.0, "larmor")) is None
