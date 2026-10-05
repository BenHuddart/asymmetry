"""The v24 -> v25 migration: each representation's single fit becomes a saved-fit set.

docs/plans/single-fit-compare.md D4: ``fit`` and each ``projection_fits`` entry
become the sole, open fit of their projection's ``single_fits`` set; an empty
slot holds no saved fit; the caller's dict is never rewritten.
"""

from __future__ import annotations

import copy

from asymmetry.core.project.schema import CURRENT_SCHEMA_VERSION, migrate_to_current, validate
from asymmetry.core.representation import RepresentationType
from asymmetry.core.representation.project_model import ProjectModel
from tests.project.single_fits import open_fit

_MODEL = {"component_names": ["Exponential", "Constant"], "operators": ["+"]}
_EMPTY = {"model": None, "parameters": [], "result": None, "provenance": "none"}


def _slot(chi2: float) -> dict:
    return {
        "model": dict(_MODEL),
        "parameters": [{"name": "A_1", "value": 0.2}],
        "result": {"success": True, "chi_squared": chi2},
        "provenance": "single",
    }


def _v24_project() -> dict:
    return {
        "schema_version": 24,
        "created_with_app_version": "0.23.0",
        "datasets": [
            {
                "run_number": 10,
                "source_file": "/data/run10.nxs",
                "representations": {
                    "time_fb_asymmetry": {
                        "rep_type": "time_fb_asymmetry",
                        "recipe": {},
                        "fit": _slot(1.0),
                        "projection_fits": {"P_x": _slot(2.0), "P_y": dict(_EMPTY)},
                    },
                    "freq_fft": {"rep_type": "freq_fft", "recipe": {}, "fit": dict(_EMPTY)},
                },
            }
        ],
    }


def test_each_slot_becomes_the_open_fit_of_its_projection():
    result = migrate_to_current(_v24_project())
    validate(result)
    assert result["schema_version"] == CURRENT_SCHEMA_VERSION == 25
    reps = result["datasets"][0]["representations"]
    fb = reps["time_fb_asymmetry"]
    assert "fit" not in fb and "projection_fits" not in fb
    assert set(fb["single_fits"]) == {"", "P_x"}
    assert open_fit(fb)["result"]["chi_squared"] == 1.0
    assert open_fit(fb, "P_x")["result"]["chi_squared"] == 2.0
    assert reps["freq_fft"]["single_fits"] == {}


def test_migration_leaves_the_callers_project_untouched():
    project = _v24_project()
    before = copy.deepcopy(project)
    migrate_to_current(project)
    assert project == before


def test_migrated_project_loads_with_one_open_fit_per_projection():
    model = ProjectModel.from_project_state(migrate_to_current(_v24_project()))
    rep = model.representation(10, RepresentationType.TIME_FB_ASYMMETRY)
    assert [len(rep.fit_set(key).fits) for key in (None, "P_x", "P_y")] == [1, 1, 0]
    # A migrated fit recorded no window, so it has no data key and is never ranked.
    assert rep.fit.fit_range is None and rep.fit.data_key() is None
