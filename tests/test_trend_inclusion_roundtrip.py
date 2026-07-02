"""Trend-inclusion + member-quality persistence (Phase 2.3).

Excluding a member from the trend writes ``FitSlot.include_in_trend``; the
advisory ``quality_flags`` ride along in the series' ``results_by_run``. Both
must survive a project save/load so a reopened project keeps the user's
exclusions and the flag rings.
"""

from __future__ import annotations

from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.representation import FitSeries, RepresentationType
from asymmetry.core.representation.base import FitSlot
from asymmetry.core.representation.project_model import ProjectModel

_FB = RepresentationType.TIME_FB_ASYMMETRY


def _model() -> dict:
    return CompositeModel(["Exponential", "Constant"], operators=["+"]).to_dict()


def _build_project() -> ProjectModel:
    model = ProjectModel()
    runs = (2949, 2950, 2960)
    results_by_run = {
        r: {
            "success": True,
            "parameters": {"A_1": 20.0},
            "uncertainties": {"A_1": 0.3},
            "reduced_chi_squared": 1.1,
            "quality_flags": (["spurious_reseeded"] if r == 2949 else []),
        }
        for r in runs
    }
    model.add_batch(
        FitSeries(
            "b1",
            _FB,
            member_run_numbers=list(runs),
            canonical_model=_model(),
            param_roles={"A_1": "local"},
            results_by_run=results_by_run,
        )
    )
    for r in runs:
        rep = model.ensure_dataset(r).ensure(_FB)
        rep.fit = FitSlot(model=_model(), provenance="batch", batch_id="b1")
    return model


def test_exclusion_and_quality_flags_survive_save_load():
    model = _build_project()
    # Exclude the garbage member from the trend.
    model.set_member_trend_inclusion("b1", 2949, False)
    assert model.representation(2949, _FB).fit.include_in_trend is False

    restored = ProjectModel.from_dict(model.to_dict())

    # The per-member gate persisted.
    assert restored.representation(2949, _FB).fit.include_in_trend is False
    assert restored.representation(2960, _FB).fit.include_in_trend is True
    # The trend consumes only the still-included members.
    included = restored.trend_runs_for_batch(restored.batch("b1"))
    assert 2949 not in included
    assert {2950, 2960} <= set(included)
    # Advisory quality flags rode along in the series summary.
    summary = restored.batch("b1").results_by_run[2949]
    assert "spurious_reseeded" in summary["quality_flags"]
