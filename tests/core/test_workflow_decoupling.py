"""The note naming a zero-field relaxation that a longitudinal field decoupled."""

from __future__ import annotations

from typing import Any

from asymmetry.core.workflow.decoupling import decoupling_note
from asymmetry.core.workflow.recipe import FitRecipe


def _series(expression: str, rows: list[tuple[int, dict[str, float], float]]) -> dict[str, Any]:
    """A stored series whose runs carry *parameters*, each free one with a relative *error*."""
    recipe = FitRecipe.from_expression(expression)
    return {
        "kind": "series",
        "expression": expression,
        "recipe": recipe.to_dict(),
        "free_params": recipe.free_parameter_names(),
        "results": [
            {
                "run": run,
                "success": True,
                "parameters": parameters,
                "uncertainties": {name: error * abs(value) for name, value in parameters.items()},
            }
            for run, parameters, error in rows
        ],
    }


def _survey(field: float, precession: str, temperature: float = 5.0) -> dict[int, dict[str, Any]]:
    """Runs 1-3 in zero field at 5 K, runs 11-13 in *field* at *temperature*."""
    zero = {run: {"field": 0.0, "precession": "none", "temperature": 5.0} for run in (1, 2, 3)}
    applied = {
        run: {"field": field, "precession": precession, "temperature": temperature}
        for run in (11, 12, 13)
    }
    return zero | applied


_ZF = _series(
    "StaticGKT_ZF + Constant",
    [(run, {"A_1": 20.0, "Delta": 0.26, "A_bg": 1.0}, 0.01) for run in (1, 2, 3)],
)
_QUENCHED = _series(
    "Exponential + Constant",
    [
        (run, {"A_1": 20.0, "Lambda": rate, "A_bg": 1.0}, 0.5)
        for run, rate in ((11, 4e-4), (12, 1e-3), (13, 2e-3))
    ],
)


def test_a_relaxation_a_strong_longitudinal_field_removes_is_named_static_from_both_sides() -> None:
    stored = {"zf": _ZF, "lf": _QUENCHED}
    survey = _survey(100.0, "none")
    note = decoupling_note("lf", stored, survey)
    assert note is not None
    assert "zero-field series zf" in note and "static on the muon time scale" in note
    assert decoupling_note("zf", stored, survey) == note


def test_the_reading_holds_whatever_model_the_field_runs_were_fitted_with() -> None:
    # A Kubo–Toyabe in 100 G is flat whatever its width: the fit leaves Delta unconstrained.
    kubo_toyabe = _series(
        "GaussianBroadenedKT + Constant",
        [
            (run, {"A_1": 5.0, "Delta": 2.0, "B_L": 100.0, "w_rel": 0.003, "A_bg": 15.0}, 3.0)
            for run in (11, 12, 13)
        ],
    )
    note = decoupling_note("lf", {"zf": _ZF, "lf": kubo_toyabe}, _survey(100.0, "none"))
    assert note is not None and "whatever the model" in note


def test_no_static_reading_when_the_data_cannot_carry_it() -> None:
    # A rate the field leaves standing is not decoupled.
    relaxing = _series(
        "Exponential + Constant",
        [(run, {"A_1": 20.0, "Lambda": 0.2, "A_bg": 1.0}, 0.05) for run in (11, 12, 13)],
    )
    assert decoupling_note("lf", {"zf": _ZF, "lf": relaxing}, _survey(100.0, "none")) is None
    # A Larmor line in the field runs makes the field transverse.
    assert decoupling_note("lf", {"zf": _ZF, "lf": _QUENCHED}, _survey(100.0, "larmor")) is None
    # Zero-field runs at other temperatures say nothing about the field runs' fields.
    assert decoupling_note("lf", {"zf": _ZF, "lf": _QUENCHED}, _survey(100.0, "none", 40.0)) is None
