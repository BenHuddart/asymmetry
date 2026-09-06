"""The Global Fit Wizard's Transitions card (transitions plan, D3).

A partitioned recommendation gives the wizard a second answer beside the
series-wide one: the penalty path, one row per "exactly k breaks" solution, with
the elbow pre-selected. These tests pin what the card says (rows, boundary
text, gains, status words, the summary line and the footnote), what selecting a
row does (recolours the series overlay by phase, re-offers the actions), what
"Optimize phases" asks the core for (the selected row's ``partition_k``), and
what the per-phase strip shows once the phases have been fitted.
"""

from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

import asymmetry.gui.windows.global_fit_wizard_window as wizard_window_module
from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.fit_wizard import (
    CandidateTemplate,
    SelectionMetric,
    SpectrumFingerprint,
)
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
from asymmetry.gui.utils.phase_colors import EXCLUDED_PHASE_HATCH_COLOR, phase_color
from asymmetry.gui.windows.global_fit_wizard_window import GlobalFitWizardWindow
from tests._qt_helpers import wait_for

#: A six-run temperature scan: three cold runs, two warm ones, and a stub at the
#: top end the partition leaves out of every phase.
_RUNS = (701, 702, 703, 704, 705, 706)
_TEMPERATURES = (1.8, 8.0, 16.0, 22.0, 28.0, 40.0)
_PHASE_I = _RUNS[0:3]
_PHASE_II = _RUNS[3:5]
_STUB = _RUNS[5]


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def datasets() -> list[MuonDataset]:
    time_axis = np.linspace(0.0, 8.0, 60)
    error = np.full_like(time_axis, 0.01)
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    items: list[MuonDataset] = []
    for run_number, temperature in zip(_RUNS, _TEMPERATURES, strict=True):
        asymmetry = model.function(time_axis, A_1=0.2, Lambda=0.3, A_bg=0.01)
        items.append(
            MuonDataset(
                time=time_axis,
                asymmetry=asymmetry,
                error=error,
                metadata={
                    "run_number": run_number,
                    "run_label": str(run_number),
                    "field": 0.0,
                    "temperature": temperature,
                },
            )
        )
    return items


def _template(key: str, title: str) -> CandidateTemplate:
    return CandidateTemplate(
        key=key,
        title=title,
        category="General",
        rationale="Planted candidate",
        model=CompositeModel(["Exponential", "Constant"], operators=["+"]),
    )


def _assessment(
    datasets: list[MuonDataset],
    runs: tuple[int, ...],
    template: CandidateTemplate,
    *,
    gate_passed: bool = True,
) -> GlobalCandidateAssessment:
    """A converged coupled fit of *template* over *runs*."""
    by_run = {int(dataset.run_number): dataset for dataset in datasets}
    fit_results: dict[int, FitResult] = {}
    fitted_curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    component_curves: dict[int, tuple] = {}
    diagnostics: list[RunResidualDiagnostic] = []
    for run_number in runs:
        dataset = by_run[run_number]
        curve = template.model.function(dataset.time, A_1=0.2, Lambda=0.3, A_bg=0.01)
        fit_results[run_number] = FitResult(
            success=True,
            chi_squared=52.0,
            reduced_chi_squared=1.04,
            dof=50,
            parameters=ParameterSet(
                [
                    Parameter("A_1", value=0.2, min=0.0, max=1.0),
                    Parameter("Lambda", value=0.3, min=0.0, max=5.0),
                    Parameter("A_bg", value=0.01, min=-0.2, max=0.2),
                ]
            ),
            uncertainties={"A_1": 0.01, "Lambda": 0.02, "A_bg": 0.001},
            residuals=np.zeros_like(dataset.time),
            message="ok",
        )
        fitted_curves[run_number] = (
            np.asarray(dataset.time, dtype=float).copy(),
            np.asarray(curve, dtype=float),
        )
        component_curves[run_number] = ()
        diagnostics.append(
            RunResidualDiagnostic(
                run_number=run_number,
                run_label=dataset.run_label,
                axis_value=float(dataset.metadata["temperature"]),
                residual_rms=0.8,
                runs_z_score=0.1,
                max_abs_autocorrelation=0.1,
                residual_fft_peak_snr=1.0,
                gate_passed=gate_passed,
                gate_reasons=() if gate_passed else ("structured residuals",),
            )
        )
    return GlobalCandidateAssessment(
        template=template,
        fit_results_by_run=fit_results,
        global_parameters=ParameterSet([Parameter("A_1", value=0.2, min=0.0, max=1.0)]),
        global_param_names=("A_1", "A_bg"),
        local_param_names=("Lambda",),
        fixed_param_names=(),
        parameter_recommendations=(),
        run_diagnostics=tuple(diagnostics),
        series_warnings=(),
        aic=10.0,
        aicc=10.2,
        bic=12.0,
        selected_score=10.2,
        fitted_curves_by_run=fitted_curves,
        component_curves_by_run=component_curves,
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


def _partition_path() -> PartitionPath:
    """A three-row path over the six-run scan, elbow at one break."""
    return PartitionPath(
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
                segments=(
                    _segment(0, 3, structure="exp"),
                    _segment(3, 6, structure="gauss"),
                ),
                total_ic=475.7,
                gain=124.3,
                admissible=True,
                boundaries=((19.0, 3.0),),
            ),
            PartitionSolution(
                breaks=2,
                segments=(
                    _segment(0, 3, structure="exp"),
                    _segment(3, 5, structure="gauss"),
                    _segment(5, 6, structure="stub", excluded=True),
                ),
                total_ic=467.2,
                gain=8.5,
                admissible=False,
                boundaries=((19.0, 3.0), (34.0, 6.0)),
            ),
        ),
        selected_k=1,
        beta_floor=16.0,
    )


def _partitioned_recommendation(
    datasets: list[MuonDataset],
    *,
    optimised_k: int | None = None,
) -> GlobalFitWizardRecommendation:
    """A screening recommendation carrying the path; optionally phase-optimised."""
    template = _template("exp_constant", "Exponential + Constant")
    fingerprints = {
        int(dataset.run_number): SpectrumFingerprint(
            tail_estimate=0.01,
            initial_amplitude_estimate=0.2,
            zero_crossings=0,
            smoothed_zero_crossings=0,
            smoothed_turning_points=0,
            dominant_fft_frequency_mhz=0.0,
            dominant_fft_snr=0.0,
            dominant_fft_cycles_in_window=0.0,
            monotonic_decay_fraction=1.0,
            early_time_curvature=-0.1,
            semilog_slope_ratio=1.0,
            late_time_dip_recovery_score=0.0,
            oscillatory_hint=False,
            kt_like_hint=False,
            multi_rate_hint=False,
        )
        for dataset in datasets
    }
    phase_assessments: dict[tuple[int, int], GlobalCandidateAssessment] = {}
    if optimised_k is not None:
        solution = _partition_path().solutions[optimised_k]
        titles = ("Cold phase model", "Warm phase model", "Third phase model")
        for segment_index, segment in enumerate(solution.segments):
            if segment.excluded:
                continue
            phase_assessments[(optimised_k, segment_index)] = _assessment(
                datasets,
                segment.run_numbers,
                _template(f"phase_{segment_index}", titles[segment_index]),
            )
    return GlobalFitWizardRecommendation(
        series_axis_key="temperature",
        series_axis_label="Temperature (K)",
        mixed_axes_warning=None,
        fingerprints_by_run=fingerprints,
        dataset_order=_RUNS,
        templates=(template,),
        assessments=(),
        metric=SelectionMetric.AICC,
        recommended_key=None,
        comparable_keys=(),
        summary="Single-fit screening complete.",
        partition_path=_partition_path(),
        phase_assessments=phase_assessments,
        recommended_partition_k=optimised_k,
    )


def _window(datasets: list[MuonDataset], recommendation) -> GlobalFitWizardWindow:
    window = GlobalFitWizardWindow()
    window.set_analysis_context(datasets)
    window.set_cached_recommendation(recommendation)
    return window


def _column(window: GlobalFitWizardWindow, column: int) -> list[str]:
    table = window._transitions_card._table
    return [table.item(row, column).text() for row in range(table.rowCount())]


# ── the path table ───────────────────────────────────────────────────────────


def test_the_card_shows_one_row_per_path_solution(qapp, datasets) -> None:
    window = _window(datasets, _partitioned_recommendation(datasets))
    assert window._transitions_card.isVisibleTo(window)
    assert _column(window, 0) == ["0", "1", "2"]


def test_boundaries_carry_the_axis_unit_and_the_break_free_row_carries_none(qapp, datasets) -> None:
    window = _window(datasets, _partitioned_recommendation(datasets))
    assert _column(window, 1) == [
        "—",
        "19 ± 3 K",
        "19 ± 3 K and 34 ± 6 K",
    ]


def test_the_top_of_the_path_has_no_gain_to_report(qapp, datasets) -> None:
    window = _window(datasets, _partitioned_recommendation(datasets))
    assert _column(window, 2) == ["—", "124.3", "8.5"]


def test_the_elbow_row_is_marked_and_pre_selected(qapp, datasets) -> None:
    window = _window(datasets, _partitioned_recommendation(datasets))
    assert _column(window, 3) == ["", "elbow", "excluded: run 706"]
    assert window._transitions_card.selected_index() == 1
    assert window._partition_k == 1


def test_a_verified_row_says_so(qapp, datasets) -> None:
    window = _window(datasets, _partitioned_recommendation(datasets, optimised_k=1))
    assert _column(window, 3)[1] == "elbow · verified"


def test_the_summary_names_the_selected_row_s_transitions(qapp, datasets) -> None:
    window = _window(datasets, _partitioned_recommendation(datasets))
    assert window._transitions_card._summary_label.text() == "1 transition found: 19 ± 3 K."


def test_selecting_another_row_restates_the_summary(qapp, datasets) -> None:
    window = _window(datasets, _partitioned_recommendation(datasets))
    window._transitions_card._table.selectRow(0)
    assert window._transitions_card._summary_label.text() == (
        "No transitions found: one phase describes the whole series."
    )
    window._transitions_card._table.selectRow(2)
    assert window._transitions_card._summary_label.text() == (
        "2 transitions found: 19 ± 3 K and 34 ± 6 K."
        " Run 706 is excluded from the global fit: it looks like a different phase."
    )


def test_the_footnote_says_the_partition_is_scored_with_bic(qapp, datasets) -> None:
    window = _window(datasets, _partitioned_recommendation(datasets))
    assert window._transitions_card._footnote.text() == (
        "Transitions are scored with BIC; the ranking metric applies within a phase."
    )


def test_a_recommendation_without_a_path_hides_the_card(qapp, datasets) -> None:
    window = _window(
        datasets,
        replace(
            _partitioned_recommendation(datasets),
            partition_path=None,
            phase_assessments={},
            recommended_partition_k=None,
        ),
    )
    assert not window._transitions_card.isVisibleTo(window)
    assert window._partition_k is None


# ── the actions ──────────────────────────────────────────────────────────────


def test_optimize_phases_is_refused_on_the_break_free_row(qapp, datasets) -> None:
    window = _window(datasets, _partitioned_recommendation(datasets))
    card = window._transitions_card
    assert card._optimize_btn.isEnabled()
    card._table.selectRow(0)
    assert not card._optimize_btn.isEnabled()
    card._table.selectRow(2)
    assert card._optimize_btn.isEnabled()


def test_apply_phases_appears_only_once_the_row_is_verified(qapp, datasets) -> None:
    plain = _window(datasets, _partitioned_recommendation(datasets))
    assert not plain._transitions_card._apply_btn.isVisibleTo(plain)
    optimised = _window(datasets, _partitioned_recommendation(datasets, optimised_k=1))
    assert optimised._transitions_card._apply_btn.isVisibleTo(optimised)


def test_optimize_phases_asks_the_core_for_the_selected_row(qapp, datasets, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_build(datasets_arg, **kwargs):
        captured["partition_k"] = kwargs.get("partition_k")
        captured["partition_path"] = kwargs.get("partition_path")
        progress = kwargs.get("progress_callback")
        progress("Optimising 4 distinct phase(s) across the 2-break solution.")
        progress("Phase 701–703: separable role search over 2 candidate(s).")
        return _partitioned_recommendation(datasets_arg, optimised_k=2)

    monkeypatch.setattr(wizard_window_module, "build_global_fit_wizard_recommendation", _fake_build)
    window = _window(datasets, _partitioned_recommendation(datasets))
    window._transitions_card._table.selectRow(2)
    window._transitions_card._optimize_btn.click()
    wait_for(lambda: window._tasks.active_count == 0, qapp)

    assert captured["partition_k"] == 2
    assert captured["partition_path"] is not None
    assert window._recommendation.recommended_partition_k == 2


def test_apply_phases_emits_the_recommendation_and_the_selected_row(qapp, datasets) -> None:
    window = _window(datasets, _partitioned_recommendation(datasets, optimised_k=1))
    emitted: list[tuple] = []
    window.apply_phases_requested.connect(
        lambda recommendation, k: emitted.append((recommendation, k))
    )
    window._transitions_card._apply_btn.click()
    assert len(emitted) == 1
    assert emitted[0][1] == 1
    assert emitted[0][0] is window._recommendation


def test_the_running_trail_counts_the_phases(qapp, datasets) -> None:
    window = _window(datasets, _partitioned_recommendation(datasets))
    window._analysis_mode = "optimize_phases"
    window._show_running()
    assert window._running_trail.step_keys() == ("prepare", "phases")

    window._on_progress(
        0,
        0,
        "Optimising 4 distinct phase(s) across the 2-break solution and its verified neighbours.",
    )
    window._on_progress(0, 0, "Phase 701–703: separable role search over 2 candidate(s).")
    assert window._running_trail._rows["phases"]._header.text() == "Optimising phase 1 of 4…"

    window._on_progress(0, 0, "Phase 704–706: separable role search over 2 candidate(s).")
    assert window._running_trail._rows["phases"]._header.text() == "Optimising phase 2 of 4…"


# ── the series overlay ───────────────────────────────────────────────────────


def test_selecting_a_partition_row_colours_the_overlay_by_phase(qapp, datasets) -> None:
    window = _window(datasets, _partitioned_recommendation(datasets))
    window._transitions_card._table.selectRow(2)
    colours = [run.colour for run in window._series_card._runs]
    assert colours == [
        *[phase_color(1)] * 3,
        *[phase_color(2)] * 2,
        EXCLUDED_PHASE_HATCH_COLOR,
    ]


def test_the_break_free_row_leaves_the_axis_gradient_alone(qapp, datasets) -> None:
    window = _window(datasets, _partitioned_recommendation(datasets))
    window._transitions_card._table.selectRow(0)
    assert [run.colour for run in window._series_card._runs] == [None] * len(_RUNS)


# ── the per-phase strip ──────────────────────────────────────────────────────


def test_the_phase_strip_appears_only_after_the_phases_are_fitted(qapp, datasets) -> None:
    plain = _window(datasets, _partitioned_recommendation(datasets))
    assert plain._transitions_card._phase_buttons == []
    optimised = _window(datasets, _partitioned_recommendation(datasets, optimised_k=1))
    assert len(optimised._transitions_card._phase_buttons) == 2


def test_each_phase_states_its_range_model_roles_and_confidence(qapp, datasets) -> None:
    window = _window(datasets, _partitioned_recommendation(datasets, optimised_k=1))
    assert window._transitions_card._phase_buttons[0].text() == (
        "Phase 1 · 1.8 – 16.0 K\n"
        "Cold phase model\n"
        "Global: A_1, A_bg · Local: Lambda\n"
        "High confidence"
    )
    assert (
        window._transitions_card._phase_buttons[1]
        .text()
        .startswith("Phase 2 · 22.0 – 40.0 K\nWarm phase model\n")
    )


def test_clicking_a_phase_shows_that_phase_s_fit(qapp, datasets) -> None:
    window = _window(datasets, _partitioned_recommendation(datasets, optimised_k=1))
    window._transitions_card._phase_buttons[1].click()
    assert window._selected_phase_segment == 1
    assert window._selected_assessment().template.title == "Warm phase model"
    # Only the picked phase's runs carry a fit overlay.
    with_fits = [run.run_label for run in window._series_card._runs if run.fitted_curve is not None]
    assert with_fits == [str(run) for run in _PHASE_II + (_STUB,)]


# ── cache round-trip ─────────────────────────────────────────────────────────


def test_a_partitioned_recommendation_survives_the_cache_restore(qapp, datasets) -> None:
    recommendation = _partitioned_recommendation(datasets, optimised_k=1)
    window = _window(datasets, recommendation)
    assert window._recommendation.partition_path is recommendation.partition_path
    assert window._partition_k == 1
    assert _column(window, 3)[1] == "elbow · verified"
    assert len(window._transitions_card._phase_buttons) == 2
