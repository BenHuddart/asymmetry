"""Applying a sharing-ladder rung to the Batch tab (trend objective, plan D17).

A rung is applied faithfully: its model (the grouped one for a shared total),
shared → Global and local → Local, the first coupled run's values as seeds, and
its exempt runs left out of the coupled series while staying in the data group.
The recommendations are real ones on the ladder's own simulated series.
"""

from __future__ import annotations

import os
from dataclasses import replace

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QApplication, QComboBox

from asymmetry.core.fitting.global_fit_wizard import (
    GlobalCandidateAssessment,
    GlobalFitWizardRecommendation,
    build_global_fit_wizard_recommendation,
)
from asymmetry.core.fitting.global_search.partition import (
    PartitionPath,
    PartitionSolution,
    Segment,
)
from asymmetry.gui.panels.fit.global_tab import GlobalFitTab
from tests.core.sharing_series import TRANSITION, SimulatedSeries, hopping_series, two_line_series

_EXEMPT = (3, 8)
_RUNS = tuple(range(1, 13))
_COUPLED = tuple(run for run in _RUNS if run not in _EXEMPT)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def hopping() -> tuple[SimulatedSeries, GlobalFitWizardRecommendation]:
    """Amplitude, background and Δ shared; runs 3 and 8 keep their own amplitude."""
    series = hopping_series(runs=12, amplitude_scale=dict.fromkeys(_EXEMPT, 1.1))
    return series, build_global_fit_wizard_recommendation(
        series.datasets, selected_template_keys=("dynamic_gkt_constant",)
    )


@pytest.fixture(scope="module")
def two_line() -> tuple[SimulatedSeries, GlobalFitWizardRecommendation]:
    """Two lines whose fraction changes through a transition: the total is shared."""
    series = two_line_series(TRANSITION, fixed=())
    return series, build_global_fit_wizard_recommendation(
        series.datasets,
        series.model,
        current_values={parameter.name: parameter.value for parameter in series.base_by_run[1]},
        current_parameter_types={"phase_1": "Fixed", "phase_3": "Fixed"},
        selected_template_keys=("current_model",),
    )


def _applied(series: SimulatedSeries, recommendation: GlobalFitWizardRecommendation):
    """The Batch tab after the recommended rung is applied, with what it emitted."""
    tab = GlobalFitTab()
    tab.set_datasets(series.datasets)
    emitted: dict[str, object] = {"group_requests": []}
    tab.global_fit_completed.connect(
        lambda results, shared: emitted.update(results=results, shared=shared)
    )
    tab.series_group_requested.connect(emitted["group_requests"].append)
    tab._apply_fit_wizard_assessment(recommendation.recommended_assessment, recommendation)
    return tab, emitted


def _table_roles(tab: GlobalFitTab) -> dict[str, str]:
    table = tab._param_table
    roles = {}
    for row in range(table.rowCount()):
        combo = table.cellWidget(row, 2)
        if isinstance(combo, QComboBox):
            roles[table.item(row, 0).data(Qt.ItemDataRole.UserRole)] = combo.currentText()
    return roles


def _assert_form_reproduces(tab: GlobalFitTab, rung: GlobalCandidateAssessment) -> None:
    """The tab's own fit would start from the rung: its model, roles and first-run values."""
    parsed = tab._parse_parameter_configuration()
    first_run = int(tab.batch_datasets()[0].run_number)
    assert tab._composite_model.to_dict() == rung.template.model.to_dict()
    assert parsed["types"] == rung.applied_roles
    assert parsed["values"] == pytest.approx(
        {
            parameter.name: parameter.value
            for parameter in rung.fit_results_by_run[first_run].parameters
        },
        rel=1e-5,
        abs=1e-9,
    )


def test_exempt_runs_are_left_out_of_the_coupled_series(qapp, hopping) -> None:
    series, recommendation = hopping
    rung = recommendation.recommended_assessment
    assert rung.exempt_runs == _EXEMPT

    tab, emitted = _applied(series, recommendation)

    # Unticked members, not a third role: the pool keeps them, the fit does not.
    assert [int(dataset.run_number) for dataset in tab._member_pool] == list(_RUNS)
    assert [int(dataset.run_number) for dataset in tab.batch_datasets()] == list(_COUPLED)
    unticked = [
        tab._members_list.item(row).data(Qt.ItemDataRole.UserRole)
        for row in range(tab._members_list.count())
        if tab._members_list.item(row).checkState() == Qt.CheckState.Unchecked
    ]
    assert unticked == list(_EXEMPT)
    assert sorted(emitted["results"]) == list(_COUPLED)
    assert set(_table_roles(tab).values()) <= {"Global", "Local", "Fixed"}
    assert _table_roles(tab) == {
        "A_1": "Global",
        "Delta": "Global",
        "nu": "Local",
        "B_L": "Local",
        "A_bg": "Global",
    }
    _assert_form_reproduces(tab, rung)
    # The group that owns the series must hold the exempt runs too.
    assert emitted["group_requests"] == [list(_RUNS)]
    assert (
        "Runs 3, 8 are left out of the coupled fit: they keep their own amplitude"
        in tab._results_card.content_html()
    )


def test_a_rung_without_exemptions_keeps_every_run_and_asks_for_no_group(qapp, two_line) -> None:
    series, recommendation = two_line
    tab, emitted = _applied(series, recommendation)

    assert [int(dataset.run_number) for dataset in tab.batch_datasets()] == list(_RUNS)
    assert sorted(emitted["results"]) == list(_RUNS)
    assert emitted["group_requests"] == []


def test_a_shared_total_rung_lands_as_the_grouped_model(qapp, two_line) -> None:
    series, recommendation = two_line
    rung = recommendation.recommended_assessment
    grouped = rung.template.model
    assert grouped.fraction_groups and grouped.param_names != series.model.param_names

    tab, _emitted = _applied(series, recommendation)

    # One total, shared; the fraction local; the pinned phases as the grouped model names them.
    assert tab._composite_model.fraction_groups == grouped.fraction_groups
    assert _table_roles(tab) == {
        "A_1": "Global",
        "frequency_1": "Global",
        "phase_1": "Fixed",
        "f_Oscillatory": "Local",
        "sigma_1": "Local",
        "frequency_2": "Global",
        "phase_2": "Fixed",
        "sigma_2": "Local",
        "A_bg": "Global",
    }
    _assert_form_reproduces(tab, rung)


def test_the_wizard_result_is_cached_under_the_series_it_analysed(qapp, hopping) -> None:
    series, recommendation = hopping
    tab, _emitted = _applied(series, recommendation)

    # Not under the coupled runs the tab is left on: reopening those is another series.
    assert list(tab._wizard_cache_by_run_set) == [_RUNS]


# ── Through the main window: the recorded series and its group ──────────────


@pytest.fixture
def mw(hopping):
    from asymmetry.gui.mainwindow import MainWindow
    from asymmetry.gui.ui_manager import UI_SCALE_SETTINGS_KEY

    QApplication.instance() or QApplication([])
    QSettings().setValue(UI_SCALE_SETTINGS_KEY, 1.0)
    window = MainWindow()
    series, _recommendation = hopping
    for dataset in series.datasets:
        window._data_browser.add_dataset(dataset)
    window._plot_workspace.set_active_view("fb_asymmetry")
    window._fit_panel.set_datasets([window._data_browser.get_dataset(run) for run in _RUNS])
    return window


def _recorded_series(window) -> list:
    return [series for series in window._project_model.batches.values() if not series.is_computed]


def test_the_recorded_series_excludes_the_exempt_runs_and_its_group_keeps_them(mw, hopping) -> None:
    _series, recommendation = hopping

    mw._fit_panel._global_tab._apply_fit_wizard_assessment(
        recommendation.recommended_assessment, recommendation
    )

    (recorded,) = _recorded_series(mw)
    group = mw._project_model.data_group(recorded.group_id)
    assert recorded.member_run_numbers == list(_COUPLED)
    assert recorded.excluded_run_numbers == list(_EXEMPT)
    assert sorted(group.member_run_numbers) == list(_RUNS)
    assert mw._fit_panel.bound_group_id() == group.group_id
    assert recorded.param_roles == {
        name: role.lower()
        for name, role in recommendation.recommended_assessment.applied_roles.items()
    }
    assert [int(dataset.run_number) for dataset in mw._fit_panel.batch_datasets()] == list(_COUPLED)


def test_a_phase_with_an_exempt_run_keeps_it_in_the_phase_group(mw, hopping) -> None:
    _series, recommendation = hopping
    rung = recommendation.recommended_assessment
    cold, warm = _RUNS[:6], _RUNS[6:]

    def phase(runs: tuple[int, ...]) -> GlobalCandidateAssessment:
        """The series' rung restricted to one phase's runs, exemptions and all."""
        return replace(
            rung,
            fit_results_by_run={run: rung.fit_results_by_run[run] for run in runs},
            fitted_curves_by_run={run: rung.fitted_curves_by_run[run] for run in runs},
            component_curves_by_run={run: rung.component_curves_by_run[run] for run in runs},
            run_diagnostics=tuple(d for d in rung.run_diagnostics if d.run_number in runs),
            rung=replace(
                rung.rung,
                exempt_runs=tuple(run for run in _EXEMPT if run in runs),
                run_costs={run: rung.rung.run_costs[run] for run in runs},
            ),
        )

    def segment(start: int, stop: int) -> Segment:
        return Segment(start, stop, _RUNS[start:stop], "gkt", 100.0, False)

    partitioned = replace(
        recommendation,
        partition_path=PartitionPath(
            solutions=(
                PartitionSolution(0, (segment(0, 12),), 300.0, 0.0, True, ()),
                PartitionSolution(
                    1, (segment(0, 6), segment(6, 12)), 200.0, 100.0, True, ((6.5, 0.5),)
                ),
            ),
            selected_k=1,
            beta_floor=16.0,
        ),
        phase_assessments={(1, 0): phase(cold), (1, 1): phase(warm)},
        recommended_partition_k=1,
    )

    mw._on_apply_wizard_phases(partitioned, 1)

    parent_id = next(
        group.group_id for group in mw._project_model.data_groups.values() if not group.is_phase
    )
    phases = mw._project_model.phase_groups_for(parent_id)
    by_group = {series.group_id: series for series in _recorded_series(mw)}
    assert [tuple(group.member_run_numbers) for group in phases] == [cold, warm]
    assert [by_group[group.group_id].excluded_run_numbers for group in phases] == [[3], [8]]
    assert by_group[phases[1].group_id].member_run_numbers == [7, 9, 10, 11, 12]
    # The tab is left on the first phase, with that phase's exempt run unticked.
    assert mw._fit_panel.bound_group_id() == phases[0].group_id
    assert [int(dataset.run_number) for dataset in mw._fit_panel.batch_datasets()] == [
        1,
        2,
        4,
        5,
        6,
    ]
    assert "runs 3, 8" in mw.statusBar().currentMessage()
