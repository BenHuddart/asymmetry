"""The Global Fit Wizard's trend objective in the GUI: the objective switch, the
Compare step's sharing ladders, the trace strip, Details, and the Apply review.

The recommendations are real ones, built by the wizard on the sharing ladder's
own simulated series (``tests/core/sharing_series.py``): a hopping series with
two anomalous-amplitude runs and a poor rival model.
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
from asymmetry.core.fitting.fit_wizard import SelectionMetric
from asymmetry.core.fitting.global_fit_wizard import (
    GlobalFitWizardRecommendation,
    build_global_fit_wizard_recommendation,
)
from asymmetry.core.fitting.global_search.trend_objective import (
    SelectionObjective,
    summarise_rungs,
)
from asymmetry.core.fitting.model_comparison import summarise_candidates
from asymmetry.gui.styles import tokens
from asymmetry.gui.widgets.model_compare_panel import (
    COSTS_TOO_MUCH,
    LADDER_CAPTION,
    PRESELECTED,
    RECOMMENDED,
    CandidateRow,
    CompareRow,
    ModelComparePanel,
    RungRow,
    format_cost,
)
from asymmetry.gui.widgets.parameter_trace_strip import EXEMPT_LEGEND, NO_LOCAL_PARAMETER
from asymmetry.gui.widgets.wizard_stepper import StepState
from asymmetry.gui.windows.global_fit_wizard_window import GlobalFitWizardWindow
from tests._qt_helpers import wait_for
from tests.core.sharing_series import SimulatedSeries, hopping_series

_GKT = "Dynamic GKT + Constant"
_EXPONENTIAL = "Exponential + Constant"


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def hopping() -> tuple[SimulatedSeries, GlobalFitWizardRecommendation]:
    """Runs 3 and 8 carry 10 % more asymmetry; the exponential fits far worse than the GKT."""
    series = hopping_series(runs=12, amplitude_scale={3: 1.1, 8: 1.1})
    return series, build_global_fit_wizard_recommendation(
        series.datasets, selected_template_keys=("dynamic_gkt_constant", "exp_constant")
    )


def _panel(series: SimulatedSeries, recommendation: GlobalFitWizardRecommendation):
    panel = ModelComparePanel()
    panel.set_series(
        series.datasets,
        [dataset.run_label for dataset in series.datasets],
        series.axis,
        recommendation.series_axis_label,
    )
    rungs = summarise_rungs(recommendation.sorted_rungs(), series.datasets, recommendation.metric)
    panel.set_ladders(rungs, recommended_key=recommendation.recommended_key)
    panel.resize(1180, 600)
    return panel, {rung.key: rung for rung in rungs}


def _shown(panel: ModelComparePanel) -> list[str]:
    return [row.key for row in panel.findChildren(CompareRow) if row.isVisibleTo(panel)]


def _window(series: SimulatedSeries, recommendation) -> GlobalFitWizardWindow:
    window = GlobalFitWizardWindow()
    window.set_analysis_context(series.datasets)
    window.set_cached_recommendation(recommendation)
    return window


# ── The Compare panel's ladders ─────────────────────────────────────────────


def test_each_model_leads_with_one_rung_and_folds_the_rest(qapp, hopping) -> None:
    series, recommendation = hopping
    panel, rungs = _panel(series, recommendation)
    leads = {title: key for key, rung in reversed(rungs.items()) for title in (rung.title,)}

    assert panel._caption.text() == LADDER_CAPTION
    assert all(isinstance(row, RungRow) for row in panel._rows.values())
    # One rung per model in view: the recommended one, and the rival's pre-selected one.
    assert _shown(panel) == [leads[_GKT], leads[_EXPONENTIAL]]
    assert panel.a_key() == recommendation.recommended_key == leads[_GKT]
    assert [fold.toggle.text() for fold in panel._folds] == ["▸ 4 other rungs", "▸ 3 other rungs"]

    panel._folds[0].toggle.click()

    assert panel._folds[0].toggle.text() == "▾ 4 other rungs"
    assert _shown(panel)[:5] == [key for key, rung in rungs.items() if rung.title == _GKT]
    # The fold stays open when the board is rebuilt, e.g. by a re-rank.
    panel.set_ladders(tuple(rungs.values()), recommended_key=recommendation.recommended_key)
    assert len(_shown(panel)) == 6


def test_a_rung_row_shows_its_mark_cost_trend_and_findings(qapp, hopping) -> None:
    series, recommendation = hopping
    panel, rungs = _panel(series, recommendation)
    recommended = panel._rows[recommendation.recommended_key]
    rung = rungs[recommendation.recommended_key].rung

    assert recommended.mark.text() == RECOMMENDED
    assert [chip.text() for chip in recommended.chips[:4]] == [
        "Global A_bg",
        "Global A_1",
        "Global Δ",
        "Local ν",
    ]
    assert recommended.cost.text == format_cost(rung.series_cost)
    assert recommended.cost.colour == tokens.BORDER_STRONG
    assert "the tolerance is 2σ" in recommended.cost.toolTip()
    assert recommended.trend.text == f"{rung.trend.parameters['nu'].quality:.2f}"
    assert recommended.trend.fraction == pytest.approx(rung.trend.parameters["nu"].quality)
    assert "ν: 0.99" in recommended.trend.toolTip()
    exempt = recommended.flag_labels[0]
    assert exempt.text() == "Runs 3, 8 exempt"
    assert exempt.toolTip() == (
        "Runs 3, 8 keep their own amplitude: the asymmetry there stands apart from the series."
    )


def test_a_model_outside_the_band_is_listed_with_its_findings(qapp, hopping) -> None:
    series, recommendation = hopping
    panel, rungs = _panel(series, recommendation)
    rival = next(row for row in panel._rows.values() if rungs[row.key].title == _EXPONENTIAL)

    assert rival.mark.text() == PRESELECTED
    lines = {label.text(): label.toolTip() for label in rival.flag_labels}
    assert lines["Outside the 3 % band"] == (
        "Fits worse than the best model by more than 3 %; listed for comparison."
    )
    # The end-block finding never asserts missing asymmetry.
    assert lines["Amplitude not shared through runs 1–3"] == (
        "Amplitude could not be shared through runs 1–3: possible missing asymmetry there, "
        "or an amplitude that really changes."
    )


def test_a_rung_that_costs_too_much_says_so_and_can_still_be_picked(qapp, hopping) -> None:
    series, recommendation = hopping
    panel, rungs = _panel(series, recommendation)
    costly_key = next(
        key for key, rung in rungs.items() if rung.title == _GKT and not rung.adequate
    )
    costly = panel._rows[costly_key]

    assert costly.mark.text() == COSTS_TOO_MUCH
    assert costly.mark.toolTip().startswith("Sharing raises the series χ²ᵣ by ")
    assert costly.cost.fraction == 1.0
    assert costly.cost.colour == tokens.WARN
    # Every parameter is shared, so nothing is left to trend.
    assert costly.trend.text == "—"
    assert not costly.isVisibleTo(panel)

    picked: list[str] = []
    panel.a_changed.connect(picked.append)
    panel.set_a(costly_key)

    # Picking a folded rung opens its ladder; it is A like any other row.
    assert picked == [costly_key]
    assert costly.isVisibleTo(panel)
    assert costly.disc.letter == "A"
    assert panel.trace_strip.figure.axes[0].texts[0].get_text() == NO_LOCAL_PARAMETER


def test_the_strip_traces_every_local_parameter_of_a_and_b(qapp, hopping) -> None:
    series, recommendation = hopping
    panel, rungs = _panel(series, recommendation)
    strip = panel.trace_strip
    all_local = next(
        key
        for key, rung in rungs.items()
        if rung.title == _GKT and not rung.rung.exempt_runs and len(rung.rung.trend.parameters) > 3
    )

    assert not strip.isHidden() and panel._trend_canvas.isHidden()
    assert [axes.get_title(loc="left") for axes in strip.figure.axes] == ["ν (MHz) — trend 0.99"]
    assert strip.figure.axes[0].get_xlabel() == "Temperature (K)"
    (trace,) = strip._traces
    (a_series,) = trace.series
    assert a_series.label == "A" and a_series.filled
    assert a_series.x.tolist() == series.axis
    # Runs 3 and 8 keep their own amplitude, and are drawn apart.
    assert np.flatnonzero(a_series.exempt).tolist() == [2, 7]
    assert strip.findChild(type(panel._caption)).text() == EXEMPT_LEGEND
    assert trace.tooltip.splitlines()[:2] == [
        "ν (MHz)",
        f"A: trend quality {a_series.quality.quality:.2f} = "
        f"determined share {a_series.quality.determined:.0%} × "
        f"signal {a_series.quality.signal:.2f} × "
        f"(1 − zigzag share {a_series.quality.zigzag:.0%})",
    ]

    panel.set_b(all_local)

    names = [trace.name for trace in strip._traces]
    assert names[0] == "nu" and {"A_1", "A_bg", "Delta"} <= set(names)
    nu = strip._traces[0]
    assert [(item.label, item.filled) for item in nu.series] == [("A", True), ("B", False)]
    assert strip.trace_at(strip.figure.axes[0]) is nu
    # The table no longer picks one parameter to plot.
    assert panel._trend is None


def test_candidates_after_ladders_bring_the_ranked_view_back(qapp, hopping) -> None:
    series, recommendation = hopping
    panel, _rungs = _panel(series, recommendation)

    panel.set_candidates(
        summarise_candidates(recommendation.sorted_rungs(), series.datasets, SelectionMetric.AICC),
        "ΔAICc from best",
    )

    assert panel._caption.text() == "Role splits · ΔAICc from best"
    assert all(isinstance(row, CandidateRow) for row in panel._rows.values())
    assert len(_shown(panel)) == len(recommendation.sorted_rungs())
    assert panel.trace_strip.isHidden() and not panel._trend_canvas.isHidden()
    assert panel._trend is not None


# ── The window: objective switch, Compare, Details, Apply review ────────────


def test_the_objective_defaults_to_trending_and_reaches_the_builder(
    qapp, hopping, monkeypatch
) -> None:
    series, recommendation = hopping
    screening = replace(
        recommendation,
        assessments=tuple(
            replace(assessment, prescreen_only=True, rung=None, assessment_key=None)
            for assessment in recommendation.sorted_rungs()
            if not assessment.global_param_names
        ),
        recommended_key=None,
        comparable_keys=(),
    )
    seen: list[SelectionObjective] = []

    def fake_build(_datasets, **kwargs):
        seen.append(kwargs["objective"])
        return recommendation

    monkeypatch.setattr(
        wizard_window_module,
        "build_global_fit_wizard_screening_recommendation",
        lambda _datasets, **_kwargs: screening,
    )
    monkeypatch.setattr(wizard_window_module, "build_global_fit_wizard_recommendation", fake_build)
    window = GlobalFitWizardWindow()
    window.set_analysis_context(series.datasets)

    assert [window._objective_combo.itemText(index) for index in range(2)] == [
        "Best for trending",
        "Best statistical fit",
    ]
    assert window.current_objective() is SelectionObjective.TREND
    assert window._objective_hint.text().startswith("Recommends the sharing pattern")

    window._start_analysis()
    wait_for(lambda: window._recommendation is not None and not window._analysis_in_progress, qapp)
    window._optimise_btn.click()
    wait_for(lambda: bool(seen) and not window._analysis_in_progress, qapp)

    assert seen == [SelectionObjective.TREND]
    assert window._stepper.state("compare") is StepState.DONE
    assert window._stepper._buttons["compare"].toolTip() == "2 sharing ladders climbed"
    assert window._objective_banner.isHidden()


def test_changing_the_objective_marks_the_optimised_fits_stale(qapp, hopping) -> None:
    series, recommendation = hopping
    window = _window(series, recommendation)

    window._objective_combo.setCurrentIndex(1)

    assert window.current_objective() is SelectionObjective.STATISTICAL
    assert window._objective_hint.text() == (
        "Recommends the model and sharing pattern with the best information criterion."
    )
    # The fits still say what they were optimised for, and read stale until re-optimised.
    assert window.current_recommendation().objective is SelectionObjective.TREND
    assert window._stepper.state("compare") is StepState.STALE
    assert not window._objective_banner.isHidden()
    assert window._objective_banner.text() == (
        "These fits were optimised as “Best for trending”. Optimise the shortlist on the "
        "Screen step again, and the phases on the Phases step, to rank them as "
        "“Best statistical fit”."
    )

    window._objective_combo.setCurrentIndex(0)

    assert window._stepper.state("compare") is StepState.DONE
    assert window._objective_banner.isHidden()


def test_a_screening_takes_the_chosen_objective_and_a_restore_shows_it(qapp, hopping) -> None:
    series, recommendation = hopping
    screening = replace(
        recommendation,
        assessments=tuple(
            replace(assessment, prescreen_only=True, rung=None, assessment_key=None)
            for assessment in recommendation.sorted_rungs()
            if not assessment.global_param_names
        ),
        recommended_key=None,
    )
    window = _window(series, screening)
    cached: list[GlobalFitWizardRecommendation] = []
    window.analysis_cached.connect(lambda rec, _log, _signature: cached.append(rec))
    window._cached_signature = {"scope": window._picker.scope().to_payload()}

    window._objective_combo.setCurrentIndex(1)

    # Nothing is optimised yet, so the choice is stored with the result and nothing is stale.
    assert window.current_recommendation().objective is SelectionObjective.STATISTICAL
    assert [rec.objective for rec in cached] == [SelectionObjective.STATISTICAL]
    assert window._objective_banner.isHidden()

    reopened = _window(series, window.current_recommendation())
    assert reopened.current_objective() is SelectionObjective.STATISTICAL


def test_compare_lists_the_ladders_and_details_cost_a_run_by_run(qapp, hopping) -> None:
    series, recommendation = hopping
    window = _window(series, recommendation)
    panel = window._compare_panel
    rung = recommendation.recommended_assessment.rung

    assert window._stepper.current_key() == "compare"
    assert panel._caption.text() == LADDER_CAPTION
    assert panel.a_key() == recommendation.recommended_key
    assert window._optimised_table.rowCount() == len(recommendation.sorted_rungs())
    # Details follows A with the rung's cost, not an empty role table.
    assert window._details_stack.currentWidget() is window._rung_details
    labels = [window._rung_grid.layout().itemAtPosition(row, 0).widget().text() for row in range(4)]
    assert labels == ["χ²ᵣ, every parameter local", "χ²ᵣ, this rung", "Cost", "Trend quality"]
    table = window._run_costs_table
    assert table.rowCount() == 12
    assert [table.item(row, 3).text() for row in (1, 2, 7)] == [
        "",
        "exempt: keeps its own amplitude",
        "exempt: keeps its own amplitude",
    ]
    assert table.item(0, 2).text() == f"{rung.run_costs[1]:+.2f}"

    costly = next(
        assessment
        for assessment in recommendation.sorted_rungs()
        if assessment.template.title == _GKT and not assessment.rung.within_tolerance
    )
    panel.set_a(costly.selection_key)

    assert "over the 2σ tolerance" in {table.item(row, 3).text() for row in range(12)}
    # A costly rung can be taken on to Apply.
    assert panel._continue.isEnabled()


def test_the_apply_step_says_which_runs_are_left_out_and_why(qapp, hopping) -> None:
    series, recommendation = hopping
    window = _window(series, recommendation)

    window._compare_panel._continue.click()

    assert window._stepper.current_key() == "apply"
    assert not window._apply_left_out.isHidden()
    assert window._apply_left_out.text() == (
        "Runs 3, 8 are left out of the coupled fit: they keep their own amplitude, which the "
        "series does not share. They stay in the data group, unticked."
    )
    rationale = window._apply_rationale.text().splitlines()
    assert rationale[0].startswith("Sharing raises the series χ²ᵣ by ")
    assert rationale[1] == "Trend quality of what stays local: ν 0.99."
    assert rationale[2].startswith("Runs 3, 8 keep their own amplitude")


def test_the_statistical_objective_keeps_the_role_tables(qapp, hopping) -> None:
    series, _recommendation = hopping
    statistical = build_global_fit_wizard_recommendation(
        series.datasets,
        selected_template_keys=("dynamic_gkt_constant",),
        objective=SelectionObjective.STATISTICAL,
    )
    window = _window(series, statistical)
    panel = window._compare_panel

    assert window.current_objective() is SelectionObjective.STATISTICAL
    assert panel._caption.text() == "Role splits · ΔAICc from best"
    assert all(isinstance(row, CandidateRow) for row in panel._rows.values())
    assert panel._folds == []
    assert window._details_stack.currentWidget() is window._roles_details
    assert window._roles_table.rowCount() > 0
    assert panel.trace_strip.isHidden()
