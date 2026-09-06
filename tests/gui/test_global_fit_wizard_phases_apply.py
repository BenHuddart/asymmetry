"""Applying the Global Fit Wizard's phases to the project (transitions plan, D3).

"Apply phases" turns an optimised partition into real structure: the series data
group (through the ordinary batch-group policy), one nested phase group per
non-excluded segment carrying its ordinal, members, range, boundary estimates,
colour and provenance, and one ``FitSeries`` per phase recorded through the
existing ``global_fit_completed`` seam with that phase's group bound. A run no
phase claimed stays a member of the parent alone.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.fit_wizard import CandidateTemplate, SelectionMetric
from asymmetry.core.fitting.global_fit_wizard import (
    GlobalCandidateAssessment,
    GlobalFitWizardRecommendation,
    RunResidualDiagnostic,
)
from asymmetry.core.fitting.global_search.partition import (
    PartitionPath,
    PartitionSolution,
    Segment,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.gui.utils.phase_colors import phase_color

#: Six runs of a zero-field temperature scan: two phases of three and two runs,
#: and a stub at the top end no phase claims.
_RUNS = (901, 902, 903, 904, 905, 906)
_TEMPERATURES = (1.8, 8.0, 16.0, 22.0, 28.0, 40.0)
_PHASE_I = _RUNS[0:3]
_PHASE_II = _RUNS[3:5]
_EXCLUDED = _RUNS[5]

_MODEL = CompositeModel(["Exponential", "Constant"], operators=["+"])


@pytest.fixture
def mw():
    from asymmetry.gui.mainwindow import MainWindow
    from asymmetry.gui.ui_manager import UI_SCALE_SETTINGS_KEY

    QApplication.instance() or QApplication([])
    QSettings().setValue(UI_SCALE_SETTINGS_KEY, 1.0)
    window = MainWindow()
    for run_number, temperature in zip(_RUNS, _TEMPERATURES, strict=True):
        window._data_browser.add_dataset(_dataset(run_number, temperature))
    window._plot_workspace.set_active_view("fb_asymmetry")
    return window


def _dataset(run_number: int, temperature: float) -> MuonDataset:
    run = Run(
        run_number=run_number,
        histograms=[
            Histogram(np.array([10.0, 20.0, 30.0, 40.0]), 0.1, 0),
            Histogram(np.array([8.0, 16.0, 24.0, 32.0]), 0.1, 0),
        ],
        metadata={"field": 0.0, "temperature": temperature},
        grouping={
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 3,
        },
    )
    return MuonDataset(
        np.array([0.0, 0.1, 0.2, 0.3]),
        np.array([0.1, 0.1, 0.1, 0.1]),
        np.array([0.01, 0.01, 0.01, 0.01]),
        {"run_number": run_number, "temperature": temperature, "field": 0.0},
        run,
    )


def _assessment(runs: tuple[int, ...], title: str) -> GlobalCandidateAssessment:
    template = CandidateTemplate(
        key=f"template_{title.replace(' ', '_').lower()}",
        title=title,
        category="General",
        rationale="Planted candidate",
        model=_MODEL,
    )
    time_axis = np.array([0.0, 0.1, 0.2, 0.3])
    curve = np.array([0.1, 0.09, 0.08, 0.07])
    fit_results = {
        run: FitResult(
            success=True,
            chi_squared=51.5,
            reduced_chi_squared=1.03,
            dof=50,
            parameters=ParameterSet(
                [
                    Parameter("A_1", value=0.2, min=0.0, max=1.0),
                    Parameter("Lambda", value=0.3, min=0.0, max=5.0),
                    Parameter("A_bg", value=0.01, min=-0.2, max=0.2),
                ]
            ),
            uncertainties={"A_1": 0.01, "Lambda": 0.02, "A_bg": 0.001},
            message="ok",
        )
        for run in runs
    }
    return GlobalCandidateAssessment(
        template=template,
        fit_results_by_run=fit_results,
        global_parameters=ParameterSet([Parameter("A_1", value=0.2, min=0.0, max=1.0)]),
        global_param_names=("A_1", "A_bg"),
        local_param_names=("Lambda",),
        fixed_param_names=(),
        parameter_recommendations=(),
        run_diagnostics=tuple(
            RunResidualDiagnostic(
                run_number=run,
                run_label=str(run),
                axis_value=float(_TEMPERATURES[_RUNS.index(run)]),
                residual_rms=0.8,
                runs_z_score=0.1,
                max_abs_autocorrelation=0.1,
                residual_fft_peak_snr=1.0,
                gate_passed=True,
                gate_reasons=(),
            )
            for run in runs
        ),
        series_warnings=(),
        aic=10.0,
        aicc=10.2,
        bic=12.0,
        selected_score=10.2,
        fitted_curves_by_run={run: (time_axis.copy(), curve.copy()) for run in runs},
        component_curves_by_run={run: () for run in runs},
    )


def _segment(start: int, stop: int, *, structure: str, excluded: bool = False) -> Segment:
    return Segment(
        start=start,
        stop=stop,
        run_numbers=_RUNS[start:stop],
        structure=structure,
        ic=100.0,
        excluded=excluded,
    )


def _recommendation() -> GlobalFitWizardRecommendation:
    """A two-phase-plus-stub partition, already optimised at ``k = 2``."""
    path = PartitionPath(
        solutions=(
            PartitionSolution(
                breaks=0,
                segments=(_segment(0, 6, structure="exp"),),
                total_ic=600.0,
                gain=0.0,
                admissible=True,
                boundaries=(),
            ),
            PartitionSolution(
                breaks=1,
                segments=(_segment(0, 3, structure="exp"), _segment(3, 6, structure="g")),
                total_ic=475.7,
                gain=124.3,
                admissible=True,
                boundaries=((19.0, 3.0),),
            ),
            PartitionSolution(
                breaks=2,
                segments=(
                    _segment(0, 3, structure="exp"),
                    _segment(3, 5, structure="g"),
                    _segment(5, 6, structure="stub", excluded=True),
                ),
                total_ic=452.2,
                gain=23.5,
                admissible=True,
                boundaries=((19.0, 3.0), (34.0, 6.0)),
            ),
        ),
        selected_k=2,
        beta_floor=16.0,
    )
    return GlobalFitWizardRecommendation(
        series_axis_key="temperature",
        series_axis_label="Temperature (K)",
        mixed_axes_warning=None,
        fingerprints_by_run={},
        dataset_order=_RUNS,
        templates=(),
        assessments=(),
        metric=SelectionMetric.AICC,
        recommended_key=None,
        comparable_keys=(),
        summary="2 transitions found: 19 ± 3 K and 34 ± 6 K.",
        partition_path=path,
        phase_assessments={
            (2, 0): _assessment(_PHASE_I, "Cold phase model"),
            (2, 1): _assessment(_PHASE_II, "Warm phase model"),
        },
        recommended_partition_k=2,
    )


def _apply(mw, monkeypatch) -> tuple[str, list]:
    """Run the apply, returning the parent group id and its phase groups."""
    monkeypatch.setattr(
        mw._fit_panel,
        "get_global_state",
        lambda: {
            "composite_model": _MODEL.to_dict(),
            "parameters": [
                {"name": "A_1", "type": "Global"},
                {"name": "Lambda", "type": "Local"},
                {"name": "A_bg", "type": "Global"},
            ],
            "result_html": "",
        },
    )
    mw._fit_panel.set_datasets([mw._data_browser.get_dataset(run) for run in _RUNS])
    mw._on_apply_wizard_phases(_recommendation(), 2)
    parent_id = next(
        group.group_id for group in mw._project_model.data_groups.values() if not group.is_phase
    )
    return parent_id, mw._project_model.phase_groups_for(parent_id)


def test_apply_creates_one_phase_group_per_non_excluded_segment(mw, monkeypatch) -> None:
    parent_id, phases = _apply(mw, monkeypatch)
    assert [phase.name for phase in phases] == ["Phase I", "Phase II"]
    assert [phase.phase_ordinal for phase in phases] == [1, 2]
    assert [tuple(phase.member_run_numbers) for phase in phases] == [_PHASE_I, _PHASE_II]
    assert mw._project_model.data_group(parent_id).member_run_numbers == list(_RUNS)


def test_the_unclaimed_run_stays_in_the_parent_alone(mw, monkeypatch) -> None:
    parent_id, _phases = _apply(mw, monkeypatch)
    assert mw._project_model.excluded_runs_for(parent_id) == [_EXCLUDED]


def test_each_phase_carries_its_range_boundaries_and_colour(mw, monkeypatch) -> None:
    _parent_id, phases = _apply(mw, monkeypatch)
    assert phases[0].phase_range == (1.8, 16.0)
    assert phases[1].phase_range == (22.0, 28.0)
    # A series end has no break beyond it; the interior break is shared.
    assert phases[0].phase_boundaries == {"lower": None, "upper": (19.0, 3.0)}
    assert phases[1].phase_boundaries == {"lower": (19.0, 3.0), "upper": (34.0, 6.0)}
    assert [phase.phase_color for phase in phases] == [phase_color(1), phase_color(2)]


def test_the_parent_group_trends_along_the_wizard_s_own_axis(mw, monkeypatch) -> None:
    parent_id, phases = _apply(mw, monkeypatch)
    assert mw._project_model.data_group(parent_id).order_key == "temperature"
    assert [phase.order_key for phase in phases] == ["temperature", "temperature"]


def test_provenance_records_how_the_phase_was_found_and_fitted(mw, monkeypatch) -> None:
    _parent_id, phases = _apply(mw, monkeypatch)
    provenance = phases[0].phase_provenance
    assert provenance["model_title"] == "Cold phase model"
    assert provenance["confidence"] == "high"
    assert provenance["selected_breaks"] == 2
    assert provenance["gains"] == [124.3, 23.5]
    assert provenance["axis_key"] == "temperature"
    assert provenance["shared_parameters"] == ["A_1", "A_bg"]
    assert provenance["fit_state"] == "converged"
    assert provenance["reduced_chi_squared"] == "χ²ᵣ 1.03"
    assert provenance["found_at"]
    assert phases[1].phase_provenance["model_title"] == "Warm phase model"


def test_one_fit_series_is_recorded_per_phase_bound_to_its_phase_group(mw, monkeypatch) -> None:
    _parent_id, phases = _apply(mw, monkeypatch)
    series = [s for s in mw._project_model.batches.values() if not s.is_computed]
    by_group = {s.group_id: s for s in series}
    assert len(series) == 2
    assert set(by_group) == {phase.group_id for phase in phases}
    assert by_group[phases[0].group_id].member_run_numbers == list(_PHASE_I)
    assert by_group[phases[1].group_id].member_run_numbers == list(_PHASE_II)


def test_the_batch_tab_is_left_bound_to_the_first_phase(mw, monkeypatch) -> None:
    _parent_id, phases = _apply(mw, monkeypatch)
    assert mw._fit_panel.bound_group_id() == phases[0].group_id


def test_the_partitioned_recommendation_lands_in_the_wizard_cache(mw, monkeypatch) -> None:
    _parent_id, _phases = _apply(mw, monkeypatch)
    cached = [
        entry["recommendation"]
        for entry in mw._fit_panel._global_tab._wizard_cache_by_run_set.values()
    ]
    assert cached
    assert all(rec.recommended_partition_k == 2 for rec in cached)
    assert all(len(rec.partition_path.solutions) == 3 for rec in cached)


def test_re_applying_replaces_the_partition_rather_than_stacking_it(mw, monkeypatch) -> None:
    parent_id, _phases = _apply(mw, monkeypatch)
    mw._on_apply_wizard_phases(_recommendation(), 2)
    assert len(mw._project_model.phase_groups_for(parent_id)) == 2
    assert len([s for s in mw._project_model.batches.values() if not s.is_computed]) == 2
