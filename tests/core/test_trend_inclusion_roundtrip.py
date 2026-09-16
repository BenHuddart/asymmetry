"""Trend-exclusion + member-quality persistence (D4).

Excluding a member from the trend writes ``FitSeries.trend_excluded_runs``; the
advisory ``quality_flags`` ride along in the series' ``results_by_run``. Both
must survive a project save/load so a reopened project keeps the user's
exclusions and the flag rings. The gate lives on the *series*, so a run in two
series can be trended in one and dropped from the other.
"""

from __future__ import annotations

from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.representation import FitSeries, RepresentationType
from asymmetry.core.representation.project_model import ProjectModel

_FB = RepresentationType.TIME_FB_ASYMMETRY


def _model() -> dict:
    return CompositeModel(["Exponential", "Constant"], operators=["+"]).to_dict()


def _series(batch_id: str, runs) -> FitSeries:
    return FitSeries(
        batch_id,
        _FB,
        member_run_numbers=list(runs),
        canonical_model=_model(),
        param_roles={"A_1": "local"},
        results_by_run={
            r: {
                "success": True,
                "parameters": {"A_1": 20.0},
                "uncertainties": {"A_1": 0.3},
                "reduced_chi_squared": 1.1,
                "quality_flags": (["spurious_reseeded"] if r == 2949 else []),
            }
            for r in runs
        },
        last_fitted_members=list(runs),
    )


def _build_project() -> ProjectModel:
    model = ProjectModel()
    model.add_batch(_series("b1", (2949, 2950, 2960)))
    model.add_batch(_series("b2", (2949, 2950)))
    return model


def test_exclusion_and_quality_flags_survive_save_load():
    model = _build_project()
    # Exclude the garbage member from one series' trend.
    model.set_trend_excluded("b1", 2949, False)  # already included: a no-op
    model.set_trend_excluded("b1", 2949, True)
    assert model.batch("b1").trend_excluded_runs == [2949]

    restored = ProjectModel.from_dict(model.to_dict())

    # The per-member gate persisted, and only on the series it was set on.
    assert restored.batch("b1").trend_member_run_numbers() == [2950, 2960]
    assert restored.batch("b2").trend_member_run_numbers() == [2949, 2950]
    # Advisory quality flags rode along in the series summary.
    summary = restored.batch("b1").results_by_run[2949]
    assert "spurious_reseeded" in summary["quality_flags"]


def test_re_including_a_member_clears_the_exclusion():
    model = _build_project()
    model.set_trend_excluded("b1", 2949, True)
    model.set_trend_excluded("b1", 2949, False)
    assert model.batch("b1").trend_excluded_runs == []
    assert model.batch("b1").trend_member_run_numbers() == [2949, 2950, 2960]
