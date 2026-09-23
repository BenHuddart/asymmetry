"""Tests for :mod:`asymmetry.core.workflow.screen`.

Screening runs the whole fit wizard, which is seconds of fitting per call, so
this file makes exactly **two** wizard builds: one with the geometry the file
implies and one with an override. Everything each build can answer is asserted
against that build rather than triggering another.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.workflow.recipe import FitRecipe
from asymmetry.core.workflow.screen import resolve_geometry, screen_run
from asymmetry.core.workflow.workdir import WorkDir
from tests.core.conftest import SCAN_RUNS

#: Categories a relaxing zero-field spectrum may legitimately be explained by.
#: The simulated signal is a single exponential, so the recommendation must
#: come from the relaxation side of the portfolio, never from precession.
_RELAXATION_CATEGORIES = {"General", "Multi-rate", "Relaxation", "Nuclear dipolar"}


# -- geometry resolution (no fitting) ---------------------------------------


def _dataset(metadata: dict) -> MuonDataset:
    time = np.linspace(0.0, 8.0, 32)
    return MuonDataset(
        time=time,
        asymmetry=np.zeros_like(time),
        error=np.ones_like(time),
        metadata=metadata,
    )


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        # A recorded field of exactly zero is zero field, whatever the file's
        # field-state stamp says — the survey's rule, so the two agree.
        ({"field": 0.0, "field_state": "TF"}, ("ZF", "field")),
        ({"field": 100.0, "field_state": "TF"}, ("TF", "file")),
        ({"field": 100.0, "field_direction": "Longitudinal"}, ("LF", "file")),
        ({"field": 100.0}, (None, "none")),
        ({}, (None, "none")),
    ],
)
def test_resolve_geometry_reports_the_value_and_where_it_came_from(
    metadata: dict, expected: tuple[str | None, str]
) -> None:
    assert resolve_geometry(_dataset(metadata), None) == expected


def test_an_override_wins_and_is_recorded_as_the_users() -> None:
    dataset = _dataset({"field": 0.0, "field_state": "TF"})
    assert resolve_geometry(dataset, "LF") == ("LF", "user")
    assert resolve_geometry(dataset, "LF", "TF") == ("LF", "user")


def test_the_surveys_geometry_beats_this_datasets_own_metadata() -> None:
    # The survey may have *measured* the geometry from Larmor precession, which
    # is the only source that can speak for a file recording no field state.
    dataset = _dataset({"field": 100.0})
    assert resolve_geometry(dataset, None) == (None, "none")
    assert resolve_geometry(dataset, None, "TF") == ("TF", "survey")


def test_an_unknown_geometry_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown geometry"):
        resolve_geometry(_dataset({}), "sideways")
    with pytest.raises(ValueError, match="Unknown geometry"):
        resolve_geometry(_dataset({}), None, "sideways")


def test_an_unknown_scope_preset_is_rejected(reduced_workdir: WorkDir) -> None:
    with pytest.raises(ValueError):
        screen_run(reduced_workdir.reduced(SCAN_RUNS[0]), scope_preset="nonsense", run_number=1)


def test_an_unknown_component_is_rejected_naming_the_vocabulary(
    reduced_workdir: WorkDir,
) -> None:
    with pytest.raises(ValueError, match="Unknown component\\(s\\) Oscilatory; .*Oscillatory"):
        screen_run(reduced_workdir.reduced(SCAN_RUNS[0]), include=["Oscilatory"], run_number=1)


# -- screening --------------------------------------------------------------


def test_screen_run_recommends_a_relaxation_model_and_writes_a_usable_recipe(
    reduced_workdir: WorkDir,
) -> None:
    run = SCAN_RUNS[0]
    dataset = reduced_workdir.reduced(run)

    result = screen_run(dataset, run_number=run)

    assert result.geometry == "ZF"
    assert result.geometry_source == "field"
    assert result.scope_preset == "auto"
    assert (result.scope_include, result.scope_exclude) == ([], [])
    assert "zero field" in result.scope_note

    # The run is a single exponential relaxation, so that is the family the
    # wizard must land in.
    recommended = next(c for c in result.candidates if c.is_recommended)
    assert recommended.key == result.recommended_key
    assert recommended.category in _RELAXATION_CATEGORIES
    assert result.confidence in {"high", "medium"}
    assert result.narrative.strip()

    # Candidates are ranked best-first by the selection score.
    scores = [c.aicc for c in result.candidates if c.aicc is not None]
    assert scores == sorted(scores)

    # The recipe is the point of the whole command: it must rebuild a model
    # that evaluates on this run's time axis.
    assert result.recipe is not None
    assert result.recipe.source == {
        "wizard_run": run,
        "template_key": result.recommended_key,
    }
    curve = result.recipe.model().function(
        dataset.time, **{p.name: p.value for p in result.recipe.parameters}
    )
    assert curve.shape == dataset.time.shape
    assert np.all(np.isfinite(curve))

    # ... and the payload round-trips, recipe included.
    data = result.to_dict()
    assert data["recommended_key"] == result.recommended_key
    assert data["recommendation"]["recommended_key"] == result.recommended_key
    assert FitRecipe.from_dict(data["recipe"]) == result.recipe


def test_a_geometry_override_changes_the_geometry_source_and_the_resolved_scope(
    reduced_workdir: WorkDir,
) -> None:
    run = SCAN_RUNS[0]
    dropped = ["Oscillatory", "OscillatoryField"]
    result = screen_run(
        reduced_workdir.reduced(run), geometry="TF", exclude=dropped, run_number=run
    )

    # Components the caller excluded never reach the candidate table, and the
    # exclusion is recorded beside the preset.
    assert result.scope_exclude == dropped
    assert result.to_dict()["scope_exclude"] == dropped
    assert not any("oscillatory" in c.key for c in result.candidates)

    assert result.geometry == "TF"
    assert result.geometry_source == "user"
    # The wizard scoped the candidate families to the geometry it was told,
    # not to the zero field the run's own metadata implies.
    assert "transverse field" in result.scope_note
    assert "zero field" not in result.scope_note
