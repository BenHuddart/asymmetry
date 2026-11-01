"""Tests for :mod:`asymmetry.core.workflow.screen`.

Screening runs the whole fit wizard, which is seconds of fitting per call, so
this file shares its wizard builds across assertions: file geometry, an
explicit override, and an unresolved survey that must clear stale file tokens.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.component_tags import PhysicsClass
from asymmetry.core.fitting.information_criteria import SelectionMetric
from asymmetry.core.workflow.recipe import FitRecipe
from asymmetry.core.workflow.screen import (
    RUNNER_UP_AICC,
    SCOPE_PRESETS,
    SurveyGeometry,
    resolve_geometry,
    screen_run,
)
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
    assert resolve_geometry(dataset, "LF", SurveyGeometry("TF")) == ("LF", "user")


def test_the_surveys_geometry_beats_this_datasets_own_metadata() -> None:
    # The survey may have *measured* the geometry from Larmor precession, which
    # is the only source that can speak for a file recording no field state.
    dataset = _dataset({"field": 100.0})
    assert resolve_geometry(dataset, None) == (None, "none")
    assert resolve_geometry(dataset, None, SurveyGeometry("TF")) == ("TF", "survey")


def test_an_unresolved_survey_does_not_revive_the_files_tf_stamp() -> None:
    dataset = _dataset({"field": 110.0, "field_state": "TF"})
    assert resolve_geometry(dataset, None, SurveyGeometry(None)) == (None, "survey")
    assert resolve_geometry(dataset, "LF", SurveyGeometry(None)) == ("LF", "user")


def test_unresolved_survey_clears_both_metadata_geometry_tokens(reduced_workdir: WorkDir) -> None:
    from dataclasses import replace

    from asymmetry.core.workflow.recipe import retained_window

    dataset = reduced_workdir.reduced(SCAN_RUNS[0])
    stamped = replace(dataset, metadata=dataset.metadata | {"field": 110.0, "field_state": "TF"})
    result = screen_run(stamped, survey=SurveyGeometry(None), run_number=SCAN_RUNS[0])
    assert result.geometry is None
    assert "field geometry not recorded" in result.scope_note
    # The window is the record the wizard fitted, after its own analysis rebinning.
    rebin = result.recommendation["rebin_factor"]
    assert result.time_window == retained_window(stamped, rebin)
    assert result.to_dict()["time_window"] == list(result.time_window)


def test_an_unknown_geometry_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown geometry"):
        resolve_geometry(_dataset({}), "sideways")
    with pytest.raises(ValueError, match="Unknown geometry"):
        resolve_geometry(_dataset({}), None, SurveyGeometry("sideways"))


def test_scope_names_are_physics_shortcuts() -> None:
    assert SCOPE_PRESETS == {
        "auto": frozenset(),
        "zf-static-magnetism": {PhysicsClass.MAGNETISM},
        "tf-knight-precession": {PhysicsClass.MAGNETISM},
        "tf-superconductor": {PhysicsClass.SUPERCONDUCTIVITY, PhysicsClass.MAGNETISM},
        "lf-dynamics": {PhysicsClass.DYNAMICS, PhysicsClass.MAGNETISM},
        "fluoride-fmuf": {PhysicsClass.MOLECULAR},
        "muonium-radical": {PhysicsClass.MUONIUM},
        "all": frozenset(),
    }


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
    assert result.recipe.seed_field == dataset.field
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

    # A single exponential leaves a near-tie among relaxation models: the best
    # other one within RUNNER_UP_AICC gets its own recipe, for the scan to decide.
    assert result.runner_up is not None
    runner_up = next(
        c for c in result.candidates if c.key == result.runner_up.source["template_key"]
    )
    assert 0.0 <= runner_up.aicc - recommended.aicc <= RUNNER_UP_AICC
    assert runner_up.parameter_count <= recommended.parameter_count + 1
    assert result.runner_up.expression != result.recipe.expression
    assert FitRecipe.from_dict(data["runner_up"]) == result.runner_up


def test_a_geometry_override_changes_the_geometry_source_and_the_resolved_scope(
    reduced_workdir: WorkDir,
) -> None:
    run = SCAN_RUNS[0]
    dropped = ["Oscillatory", "OscillatoryField"]
    result = screen_run(
        reduced_workdir.reduced(run),
        geometry="TF",
        scope_preset="lf-dynamics",
        exclude=dropped,
        run_number=run,
    )

    # Components the caller excluded never reach the candidate table, and the
    # exclusion is recorded beside the scope name.
    assert result.scope_exclude == dropped
    assert result.to_dict()["scope_exclude"] == dropped
    assert not any("oscillatory" in c.key for c in result.candidates)

    assert result.geometry == "TF"
    assert result.geometry_source == "user"
    # The wizard scoped the candidate families to the geometry it was told,
    # not to the zero field the run's own metadata implies.
    assert "transverse field" in result.scope_note
    assert "zero field" not in result.scope_note
    # A scope name chooses the physics looked for, never the geometry.
    assert result.scope_preset == "lf-dynamics"
    assert "looking for dynamics, magnetism" in result.scope_note


def _assessment(category: str, aicc: float, expression: str, parameters: int = 4):
    from types import SimpleNamespace

    from asymmetry.core.fitting.composite import CompositeModel

    model = CompositeModel.from_expression(expression)
    return SimpleNamespace(
        template=SimpleNamespace(category=category, model=model),
        aicc=aicc,
        parameter_count=parameters,
        metric_value=lambda metric: aicc,
        is_successful=True,
        is_disqualified=False,
    )


def test_a_precession_model_with_no_spectral_line_gets_the_best_relaxation_beside_it() -> None:
    from types import SimpleNamespace

    from asymmetry.core.workflow.screen import _runner_up

    bessel = _assessment("Oscillatory", 2249.1, "Bessel * Exponential + Constant", 5)
    kubo_toyabe = _assessment("KT-like", 2260.6, "DynamicGaussianKT + Constant")
    recommendation = SimpleNamespace(
        metric=SelectionMetric.AICC,
        recommended_assessment=bessel,
        assessments=[bessel, kubo_toyabe],
    )
    # However far behind: the precession model fits a relaxation's shape.
    assert _runner_up(recommendation, lined=False) is kubo_toyabe
    # A line in the spectrum is precession's own evidence.
    assert _runner_up(recommendation, lined=True) is None


def test_a_close_nested_precession_model_is_a_portable_scan_alternative() -> None:
    from types import SimpleNamespace

    from asymmetry.core.workflow.screen import _runner_up

    complex_model = _assessment(
        "Oscillatory", 100.0, "Oscillatory * Gaussian + Exponential + Constant", 7
    )
    simple = _assessment("Oscillatory", 103.0, "Oscillatory * Gaussian + Constant", 5)
    relaxation = _assessment("General", 104.0, "Exponential + Constant", 3)
    recommendation = SimpleNamespace(
        metric=SelectionMetric.AICC,
        recommended_assessment=complex_model,
        assessments=[complex_model, simple, relaxation],
    )
    assert _runner_up(recommendation, lined=True) is simple
    simple.aicc = 111.0
    assert _runner_up(recommendation, lined=True) is None


def test_screen_forwards_explicit_fit_worker_budget(monkeypatch):
    def cancelled_screen(dataset, *, max_workers, scope):
        assert max_workers == 2
        raise RuntimeError("budget reached fitting backend")

    monkeypatch.setattr(
        "asymmetry.core.workflow.screen.build_fit_wizard_recommendation",
        cancelled_screen,
    )
    with pytest.raises(RuntimeError, match="budget reached fitting backend"):
        screen_run(_dataset({"field": 0.0}), run_number=1, max_workers=2)
