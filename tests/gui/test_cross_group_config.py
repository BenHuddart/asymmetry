"""Tests for the pure config->fit healing bridge (run_cross_group_fit_from_config).

Covers WP-C's self-heal: a config whose ``parameter_rows`` have drifted from
the model's current ``param_names`` (e.g. because the model was edited after
the config was saved, leaving stale/missing rows) must still fit successfully
by classifying roles from the model itself, and the healed config must come
back on the returned run so a caller can persist the repair.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.fitting.parameter_models import ParameterCompositeModel, ParameterGroupData
from asymmetry.gui.panels.cross_group_config import run_cross_group_fit_from_config

pytestmark = [pytest.mark.gui]


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
