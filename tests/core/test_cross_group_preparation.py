"""Cross-group config preparation preserves accepted rows, masks and execution."""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.fitting.cross_group import (
    CrossGroupFitInputError,
    prepare_cross_group_fit,
    run_cross_group_fit,
    run_cross_group_fit_from_config,
)
from asymmetry.core.fitting.parameter_models import (
    ErrorMode,
    ParameterCompositeModel,
    ParameterGroupData,
)


def _groups() -> list[ParameterGroupData]:
    x = np.linspace(1.0, 10.0, 8)
    g0 = ParameterGroupData(
        group_id="g0",
        group_name="G0",
        x=x.copy(),
        y=0.02 * x + 0.10,
        yerr=np.full_like(x, 0.005),
        group_variable_value=10.0,
    )
    g1 = ParameterGroupData(
        group_id="g1",
        group_name="G1",
        x=x.copy(),
        y=0.02 * x + 0.40,
        yerr=np.full_like(x, 0.005),
        group_variable_value=20.0,
    )
    return [g0, g1]


def _linear_config() -> dict:
    return {
        "model": ParameterCompositeModel(["Linear"], []).to_dict(),
        "fit_x_min": None,
        "fit_x_max": None,
        "parameter_rows": [
            {"name": "m", "initial": 0.02, "min": -1.0, "max": 1.0, "type": "Global"},
            {"name": "b", "initial": 0.1, "min": -1.0, "max": 1.0, "type": "Local"},
        ],
        "error_mode": "column",
        "error_value": None,
        "windows": None,
        "use_x_errors": False,
    }


def test_clean_config_fits_and_config_round_trips() -> None:
    """A config already in sync with the model is unaffected: it fits and the
    healed config carries the same rows/roles back out."""
    config = _linear_config()
    run = run_cross_group_fit_from_config(_groups(), config)

    assert run.result.success is True
    assert run.config["model"]["component_names"] == ["Linear"]
    healed_rows = {row["name"]: row["type"] for row in run.config["parameter_rows"]}
    assert healed_rows == {"m": "Global", "b": "Local"}


def test_heals_config_with_removed_param_and_missing_new_param() -> None:
    """Simulate a study whose model was edited (Linear -> Arrhenius) after the
    config was saved: parameter_rows still reference the OLD model's 'm'/'b'
    (one of which, 'b', is not even a real Arrhenius param) and are missing
    the NEW model's 'a'/'Ea' entirely. Before the fix this fed 'm'/'b' straight
    into global_fit_parameter_model, which rejects them as an unknown
    parameter classification (parameter_models.py). The healed path must drop
    the stale rows, default the new params to Global, and fit successfully.
    """
    config = _linear_config()
    config["model"] = ParameterCompositeModel(["Arrhenius"], []).to_dict()
    # parameter_rows still says "m"/"b" (the OLD Linear model's params) -- a
    # corrupted config exactly like the one _on_global_fit_refit_requested
    # would previously re-save unchanged.

    run = run_cross_group_fit_from_config(_groups(), config)

    assert run.result.success is True
    healed_names = {row["name"] for row in run.config["parameter_rows"]}
    assert healed_names == {"a", "Ea"}
    # Neither stale name survives.
    assert "m" not in healed_names
    assert "b" not in healed_names
    # New params default to Global.
    healed_roles = {row["name"]: row["type"] for row in run.config["parameter_rows"]}
    assert healed_roles == {"a": "Global", "Ea": "Global"}
    # The healed model dict matches the NEW model, not the stale one.
    assert run.config["model"]["component_names"] == ["Arrhenius"]
    assert run.model.component_names == ["Arrhenius"]


def test_heals_config_when_model_gains_an_extra_component() -> None:
    """A model edit that ADDS a component (Linear -> Linear+Constant, say via
    an equivalent single-component model with a new param) leaves an existing
    row set that undershoots param_names; the missing param must be added as
    Global with the model's default rather than left unclassified."""
    config = _linear_config()
    config["model"] = ParameterCompositeModel(["Arrhenius"], []).to_dict()
    # Keep only "a" (still meaningful-ish name overlap is irrelevant -- what
    # matters is Ea is entirely missing from parameter_rows).
    config["parameter_rows"] = [
        {"name": "a", "initial": 1.0, "min": -10.0, "max": 10.0, "type": "Global"},
    ]

    run = run_cross_group_fit_from_config(_groups(), config)

    assert run.result.success is True
    healed_by_name = {row["name"]: row for row in run.config["parameter_rows"]}
    assert set(healed_by_name) == {"a", "Ea"}
    # The pre-existing row's bounds/role are preserved.
    assert healed_by_name["a"]["type"] == "Global"
    assert healed_by_name["a"]["min"] == -10.0
    # The missing param defaults to Global with the model's default + open bounds.
    assert healed_by_name["Ea"]["type"] == "Global"
    assert healed_by_name["Ea"]["min"] == -float("inf")
    assert healed_by_name["Ea"]["max"] == float("inf")
    model = ParameterCompositeModel(["Arrhenius"], [])
    assert healed_by_name["Ea"]["initial"] == pytest.approx(model.param_defaults["Ea"])


def test_preparation_is_detached_and_preserves_extra_config_attribution():
    groups = _groups()
    config = _linear_config()
    config["source_reduced_chi_squared"] = 2.5
    config["extra"] = {"labels": ["recorded"]}
    config["windows"] = [[2.0, 4.0], [7.0, 8.0]]
    prepared = prepare_cross_group_fit(groups, config)
    expected_x = np.array([groups[0].x[i] for i in (1, 2, 5)])
    np.testing.assert_array_equal(prepared.groups[0].x, expected_x)
    config["extra"]["labels"].append("changed")
    groups[0].y[:] = 999.0
    assert prepared.config["extra"] == {"labels": ["recorded"]}
    assert prepared.config["source_reduced_chi_squared"] == 2.5
    assert not np.any(prepared.groups[0].y == 999.0)


def test_prepared_execution_matches_direct_engine_and_saved_config():
    from asymmetry.core.fitting.parameter_models import ErrorMode, global_fit_parameter_model

    groups = _groups()
    config = _linear_config()
    direct = global_fit_parameter_model(
        groups=groups,
        model=ParameterCompositeModel(["Linear"], []),
        global_params=["m"],
        local_params=["b"],
        fixed_params={},
        initial_params={"m": 0.02, "b": 0.1},
        parameter_bounds={"m": (-1.0, 1.0), "b": (-1.0, 1.0)},
        error_mode=ErrorMode.COLUMN,
    )
    prepared = run_cross_group_fit(prepare_cross_group_fit(groups, config))
    saved = run_cross_group_fit_from_config(groups, config)
    for actual in (prepared.result, saved.result):
        assert actual.success == direct.success
        assert actual.chi_squared == pytest.approx(direct.chi_squared, abs=1e-20)
        assert actual.global_parameters["m"].value == pytest.approx(
            direct.global_parameters["m"].value
        )
        for group in groups:
            assert actual.local_parameters[group.group_id]["b"].value == pytest.approx(
                direct.local_parameters[group.group_id]["b"].value
            )


@pytest.mark.parametrize(
    "updates, reason",
    [
        ({"windows": [[4.0, 2.0]]}, "windows"),
        ({"fit_x_min": 5.0, "fit_x_max": 1.0}, "range"),
        ({"fit_x_min": 100.0}, "groups"),
    ],
)
def test_user_input_errors_are_classified_once(updates, reason):
    with pytest.raises(CrossGroupFitInputError) as exc:
        prepare_cross_group_fit(_groups(), _linear_config() | updates)
    assert exc.value.reason == reason


@pytest.mark.parametrize("mode", [mode.value for mode in ErrorMode])
def test_error_modes_and_selected_x_errors_are_preserved(mode):
    groups = _groups()
    for group in groups:
        group.xerr = np.arange(group.x.size, dtype=float) / 10.0
    config = _linear_config() | {
        "error_mode": mode,
        "error_value": 0.3,
        "use_x_errors": True,
        "windows": [[2.0, 5.0]],
        "fit_x_min": 9.0,
        "fit_x_max": 1.0,
    }
    inputs = prepare_cross_group_fit(groups, config)
    assert inputs.error_mode.value == mode
    assert inputs.error_value == 0.3
    for source, selected in zip(groups, inputs.groups, strict=True):
        mask = (source.x >= 2.0) & (source.x <= 5.0)
        np.testing.assert_array_equal(inputs.xerr[source.group_id], source.xerr[mask])
        np.testing.assert_array_equal(selected.xerr, source.xerr[mask])
    without_xerr = prepare_cross_group_fit(groups, config | {"use_x_errors": False})
    assert without_xerr.xerr is None


def test_duplicate_rows_use_first_and_unknown_roles_remain_global():
    config = _linear_config()
    config["parameter_rows"] += [
        {"name": "m", "initial": 999.0, "type": "Fixed"},
        {"name": "retired", "initial": 1.0},
    ]
    config["parameter_rows"][0]["type"] = "custom"
    inputs = prepare_cross_group_fit(_groups(), config)
    assert inputs.global_params == ["m"]
    assert inputs.initial_params["m"] == 0.02
    assert [row["name"] for row in inputs.config["parameter_rows"]] == ["m", "b"]
    assert inputs.config["parameter_rows"][0]["type"] == "custom"


def test_unsuccessful_solver_result_is_returned_without_reclassification(monkeypatch):
    from asymmetry.core.fitting import cross_group
    from asymmetry.core.fitting.parameter_models import CrossGroupFitResult

    result = CrossGroupFitResult(
        success=False, chi_squared=1.0, reduced_chi_squared=1.0, message="No convergence"
    )
    monkeypatch.setattr(cross_group, "global_fit_parameter_model", lambda **kwargs: result)
    run = run_cross_group_fit_from_config(_groups(), _linear_config())
    assert run.result is result
    assert not run.result.success
