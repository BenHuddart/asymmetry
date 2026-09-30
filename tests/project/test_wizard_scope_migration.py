"""Tests for the v22->v23 wizard scope migration (docs/plans/model-family-picker.md D2).

A v22 project stores fit-wizard scopes as version-1 preset payloads in the
fit-panel wizard caches: a cache ``signature``'s ``"scope"`` and the
``build_signature`` JSON string of each single-run recommendation. v23 rewrites
every one to the version-2 physics payload the runtime parses strictly.
"""

from __future__ import annotations

import json

import pytest

from asymmetry.core.fitting.component_tags import PhysicsClass
from asymmetry.core.fitting.fit_wizard import single_fit_build_signature
from asymmetry.core.fitting.wizard_scope import WizardScope
from asymmetry.core.project.schema import (
    CURRENT_SCHEMA_VERSION,
    load_project,
    migrate_to_current,
    save_project,
    validate,
)


def _v1(preset: str, include: list[str] = (), exclude: list[str] = ()) -> dict:
    return {"version": 1, "preset": preset, "include": list(include), "exclude": list(exclude)}


def _v1_build_signature(scope: dict, frequencies: list[float] = ()) -> str:
    """A ``single_fit_build_signature`` string exactly as v22 wrote it."""
    return json.dumps(
        {"scope": scope, "user_frequencies_mhz": [float(f) for f in frequencies]},
        sort_keys=True,
    )


def _single_wizard_state(scope: dict, frequencies: list[float] = ()) -> dict:
    return {
        "signature": {"run_number": 10, "model": None, "scope": scope, "user_peaks": []},
        "recommendation": {
            "recommended_key": "exp_constant",
            "build_signature": _v1_build_signature(scope, frequencies),
        },
        "log_text": "",
    }


def _global_wizard_state(scope: dict, runs: list[int]) -> dict:
    return {
        "signature": {"run_numbers": runs, "scope": scope, "effort_tier": "exhaustive"},
        "recommendation": {"recommended_key": "exp_constant"},
        "log_text": "",
        "run_numbers": runs,
    }


def _v22_project() -> dict:
    single = _single_wizard_state(_v1("lf-dynamics", include=["Oscillatory"]), [2.5])
    projection = _single_wizard_state(_v1("tf-superconductor", exclude=["Bessel"]))
    grouped = _global_wizard_state(_v1("fluoride-fmuf"), [10, 11])
    return {
        "schema_version": 22,
        "created_with_app_version": "0.21.0",
        "datasets": [
            {
                "run_number": 10,
                "source_file": "/data/run10.nxs",
                "representations": {
                    "time_fb_asymmetry": {
                        "rep_type": "time_fb_asymmetry",
                        "recipe": {"scopes": ["common"]},
                        "fit": {"model": None, "ui_state": {"wizard_state": single}},
                        "projection_fits": {
                            "real": {"model": None, "ui_state": {"wizard_state": projection}}
                        },
                    }
                },
            },
            {
                "run_number": 11,
                "source_file": "/data/run11.nxs",
                "representations": {
                    "time_grouped": {
                        "rep_type": "time_grouped",
                        "fit": {
                            "model": None,
                            "ui_state": {
                                "wizard_state": _global_wizard_state(_v1("muonium-radical"), [11]),
                                "wizard_state_by_run_set": [grouped],
                            },
                        },
                    }
                },
            },
        ],
        "multi_group_fit_state": {
            "single": {"wizard_state": _global_wizard_state(_v1("auto"), [11])},
            "batch": {"wizard_state_by_run_set": [_global_wizard_state(_v1("all"), [10, 11])]},
        },
        "joint_fits": [],
    }


def _scope(physics: set[PhysicsClass], **kwargs) -> WizardScope:
    return WizardScope(physics=frozenset(physics), **kwargs)


def test_v22_scopes_migrate_to_version_2_physics_payloads():
    result = migrate_to_current(_v22_project())
    validate(result)
    assert result["schema_version"] == CURRENT_SCHEMA_VERSION == 23

    rep = result["datasets"][0]["representations"]["time_fb_asymmetry"]
    single = rep["fit"]["ui_state"]["wizard_state"]
    expected = _scope(
        {PhysicsClass.DYNAMICS, PhysicsClass.MAGNETISM},
        include_components=frozenset({"Oscillatory"}),
    )
    assert single["signature"]["scope"] == {
        "version": 2,
        "physics": ["dynamics", "magnetism"],
        "include": ["Oscillatory"],
        "exclude": [],
        "skip_slow": False,
    }
    assert WizardScope.from_payload(single["signature"]["scope"]) == expected
    # The migrated build signature is the one the runtime builds today, so the
    # cached recommendation still counts as the answer to the same question.
    assert single["recommendation"]["build_signature"] == single_fit_build_signature(
        expected, [2.5]
    )

    projection = rep["projection_fits"]["real"]["ui_state"]["wizard_state"]
    assert WizardScope.from_payload(projection["signature"]["scope"]) == _scope(
        {PhysicsClass.SUPERCONDUCTIVITY, PhysicsClass.MAGNETISM},
        exclude_components=frozenset({"Bessel"}),
    )

    grouped_ui = result["datasets"][1]["representations"]["time_grouped"]["fit"]["ui_state"]
    assert WizardScope.from_payload(grouped_ui["wizard_state"]["signature"]["scope"]) == _scope(
        {PhysicsClass.MUONIUM}
    )
    (by_run_set,) = grouped_ui["wizard_state_by_run_set"]
    assert WizardScope.from_payload(by_run_set["signature"]["scope"]) == _scope(
        {PhysicsClass.MOLECULAR}
    )

    grouped_window = result["multi_group_fit_state"]
    for state in (
        grouped_window["single"]["wizard_state"],
        grouped_window["batch"]["wizard_state_by_run_set"][0],
    ):
        assert WizardScope.from_payload(state["signature"]["scope"]) == WizardScope()


def test_migration_leaves_everything_else_untouched():
    project = _v22_project()
    result = migrate_to_current(project)
    rep = result["datasets"][0]["representations"]["time_fb_asymmetry"]
    assert rep["recipe"] == {"scopes": ["common"]}
    assert rep["fit"]["ui_state"]["wizard_state"]["recommendation"]["recommended_key"] == (
        "exp_constant"
    )
    # The input dict is not mutated.
    assert project["schema_version"] == 22
    assert (
        project["datasets"][0]["representations"]["time_fb_asymmetry"]["fit"]["ui_state"][
            "wizard_state"
        ]["signature"]["scope"]["version"]
        == 1
    )


@pytest.mark.parametrize(
    ("scope", "expected"),
    [
        # Version-1 reads were tolerant: an unknown preset meant Auto and a
        # malformed name list meant no names.
        ({"version": 1, "preset": "made-up"}, WizardScope()),
        (
            {"version": 1, "preset": "zf-static-magnetism", "include": "Oscillatory"},
            WizardScope(physics=frozenset({PhysicsClass.MAGNETISM})),
        ),
        (
            {"version": 1, "include": ["Keren", "Keren"], "exclude": None},
            WizardScope(include_components=frozenset({"Keren"})),
        ),
    ],
)
def test_malformed_version_1_scopes_keep_their_tolerant_meaning(scope, expected):
    project = {
        "schema_version": 22,
        "datasets": [],
        "multi_group_fit_state": {"single": {"wizard_state": {"signature": {"scope": scope}}}},
    }
    migrated = migrate_to_current(project)["multi_group_fit_state"]["single"]["wizard_state"]
    assert WizardScope.from_payload(migrated["signature"]["scope"]) == expected


def test_v22_project_round_trips_through_save_and_load(tmp_path):
    path = tmp_path / "project.asymp"
    save_project(migrate_to_current(_v22_project()), path)
    loaded = load_project(path)
    validate(loaded)
    assert loaded["schema_version"] == 23
    single = loaded["datasets"][0]["representations"]["time_fb_asymmetry"]["fit"]["ui_state"][
        "wizard_state"
    ]
    assert WizardScope.from_payload(single["signature"]["scope"]).physics == {
        PhysicsClass.DYNAMICS,
        PhysicsClass.MAGNETISM,
    }
    # Re-migrating a v23 project is a no-op.
    assert migrate_to_current(loaded) == loaded
