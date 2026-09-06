"""Nested phase sub-groups in the Data Browser (transitions plan, D2).

A series the Global Fit Wizard partitioned renders three levels deep: the
series header, one sub-header per phase, and the runs beneath each phase, with
the runs no phase claimed appended as excluded members of the series. These
tests pin the row order, the fact that a phase membership is a *refinement* of
the parent's rather than a second membership (so no copy rows and no ①
markers), the stripe roles the delegate paints from, the selection semantics of
each header, the "Move to phase" override, the ⓘ popover's text, and the state
round-trip of per-phase collapse.
"""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import Qt

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.representation.base import RepresentationType
from asymmetry.core.representation.group import PhaseSpec
from asymmetry.core.representation.project_model import ProjectModel
from asymmetry.core.representation.series import FitSeries
from asymmetry.gui.panels.data_browser import (
    _BADGE_ROLE,
    _EXCLUDED_BADGE,
    _INFO_INDICATOR,
    _PHASE_STRIPE_HATCHED_ROLE,
    _PHASE_STRIPE_ROLE,
    DataBrowserPanel,
)
from asymmetry.gui.utils.phase_colors import EXCLUDED_PHASE_HATCH_COLOR, phase_color

pytestmark = [pytest.mark.gui, pytest.mark.usefixtures("qapp")]

#: The six runs of the synthetic scan: three cold, two warm, one stub at the
#: top end that the partition leaves out of every phase.
_PHASE_I_RUNS = (1, 2, 3)
_PHASE_II_RUNS = (4, 5)
_EXCLUDED_RUN = 6


def _dataset(rn: int, *, temperature: float) -> MuonDataset:
    grouping = {
        "groups": {1: [1], 2: [2]},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.0,
        "first_good_bin": 12,
        "last_good_bin": 90,
        "good_frames": 1000.0,
        "t0_bin": 10,
    }
    histograms = [
        Histogram(
            counts=np.full(100, 300.0),
            bin_width=0.016,
            t0_bin=10,
            good_bin_start=10,
            good_bin_end=90,
        )
        for _ in range(2)
    ]
    run = Run(
        run_number=rn,
        histograms=histograms,
        metadata={
            "run_number": rn,
            "title": f"Run {rn}",
            "temperature": temperature,
            "field": 100.0,
        },
        grouping=grouping,
        source_file=f"/tmp/run_{rn}.nxs",
    )
    return MuonDataset(
        time=np.arange(100.0),
        asymmetry=np.zeros(100),
        error=np.ones(100),
        metadata=dict(run.metadata),
        run=run,
    )


def _phase_spec(ordinal: int, name: str, runs: tuple[int, ...], span, *, lower, upper):
    return PhaseSpec(
        ordinal=ordinal,
        name=name,
        member_run_numbers=runs,
        phase_range=span,
        phase_boundaries={"lower": lower, "upper": upper},
        phase_color=phase_color(ordinal),
        phase_provenance={
            "model_title": "Two damped lines",
            "confidence": "high",
            "axis_key": "temperature",
            "found_at": "2026-09-06T09:00:00",
            "selected_breaks": 1,
            "gains": [12.4],
            "shared_parameters": ["lambda", "phi"],
            "fit_state": "converged",
            "reduced_chi_squared": "1.04",
        },
    )


def _panel_with_partition() -> tuple[DataBrowserPanel, ProjectModel, str, list[str]]:
    """A browser over a six-run temperature scan split into two phases + a stub."""
    panel = DataBrowserPanel()
    model = ProjectModel()
    panel.set_project_model(model)
    with panel.batch_updates():
        for rn, temperature in zip(
            (1, 2, 3, 4, 5, 6), (1.8, 8.0, 16.0, 22.0, 28.0, 40.0), strict=True
        ):
            panel.add_dataset(_dataset(rn, temperature=temperature))
    parent_id = panel.create_data_group(
        [1, 2, 3, 4, 5, 6], name="ZF temperature scan", order_key="temperature"
    )
    phase_ids = model.create_phase_groups(
        parent_id,
        [
            _phase_spec(1, "Phase I", _PHASE_I_RUNS, (1.8, 16.0), lower=None, upper=(19.0, 3.0)),
            _phase_spec(2, "Phase II", _PHASE_II_RUNS, (22.0, 28.0), lower=(19.0, 3.0), upper=None),
        ],
    )
    panel.sync_groups_from_project_model()
    return panel, model, parent_id, phase_ids


def _row_keys(panel: DataBrowserPanel) -> list[object]:
    return [
        panel._table.item(row, 0).data(panel._GROUP_ROLE) for row in range(panel._table.rowCount())
    ]


def _row_labels(panel: DataBrowserPanel) -> list[str]:
    return [panel._table.item(row, 0).text() for row in range(panel._table.rowCount())]


def _row_of_key(panel: DataBrowserPanel, key: object) -> int:
    return _row_keys(panel).index(key)


# ── rendering ────────────────────────────────────────────────────────────────


def test_phases_render_between_series_header_and_its_runs() -> None:
    panel, _model, parent_id, phase_ids = _panel_with_partition()
    assert _row_keys(panel) == [
        f"group:{parent_id}",
        f"group:{phase_ids[0]}",
        *(f"phase:{phase_ids[0]}:{rn}" for rn in _PHASE_I_RUNS),
        f"group:{phase_ids[1]}",
        *(f"phase:{phase_ids[1]}:{rn}" for rn in _PHASE_II_RUNS),
        _EXCLUDED_RUN,
    ]


def test_phase_header_shows_ordinal_swatch_and_range() -> None:
    panel, _model, _parent_id, phase_ids = _panel_with_partition()
    row = _row_of_key(panel, f"group:{phase_ids[0]}")
    assert panel._table.item(row, 0).text().strip() == "▾ ■ Phase I"
    assert panel._table.item(row, 1).text() == "1.8 – 16.0 K"
    assert panel._table.item(row, 3).text() == f"3 runs{_INFO_INDICATOR}"


def test_series_header_gains_the_info_indicator_once_partitioned() -> None:
    panel, _model, parent_id, _phase_ids = _panel_with_partition()
    row = _row_of_key(panel, f"group:{parent_id}")
    assert panel._table.item(row, 3).text() == f"6 runs{_INFO_INDICATOR}"


def test_phase_members_are_indented_one_level_deeper_than_the_phase_header() -> None:
    panel, _model, _parent_id, phase_ids = _panel_with_partition()
    labels = _row_labels(panel)
    header = labels[_row_of_key(panel, f"group:{phase_ids[0]}")]
    member = labels[_row_of_key(panel, f"phase:{phase_ids[0]}:1")]
    assert len(member) - len(member.lstrip()) == 2 * (len(header) - len(header.lstrip()))


def test_phase_members_carry_the_phase_stripe_colour() -> None:
    panel, _model, _parent_id, phase_ids = _panel_with_partition()
    for ordinal, (phase_id, runs) in enumerate(
        zip(phase_ids, (_PHASE_I_RUNS, _PHASE_II_RUNS), strict=True), start=1
    ):
        for rn in runs:
            item = panel._table.item(_row_of_key(panel, f"phase:{phase_id}:{rn}"), 0)
            assert item.data(_PHASE_STRIPE_ROLE) == phase_color(ordinal)
            assert not item.data(_PHASE_STRIPE_HATCHED_ROLE)


def test_excluded_run_is_hatched_italic_and_badged() -> None:
    panel, _model, _parent_id, _phase_ids = _panel_with_partition()
    row = _row_of_key(panel, _EXCLUDED_RUN)
    run_item = panel._table.item(row, 0)
    assert run_item.data(_PHASE_STRIPE_ROLE) == EXCLUDED_PHASE_HATCH_COLOR
    assert run_item.data(_PHASE_STRIPE_HATCHED_ROLE) is True
    assert run_item.font().italic()
    assert panel._table.item(row, 1).data(_BADGE_ROLE) == _EXCLUDED_BADGE


def test_plain_runs_carry_no_stripe() -> None:
    panel, _model, _parent_id, phase_ids = _panel_with_partition()
    header_row = _row_of_key(panel, f"group:{phase_ids[0]}")
    assert panel._table.item(header_row, 0).data(_PHASE_STRIPE_ROLE) == phase_color(1)
    panel.add_dataset(_dataset(9, temperature=3.0))
    assert panel._table.item(_row_of_key(panel, 9), 0).data(_PHASE_STRIPE_ROLE) is None


def test_phase_membership_creates_no_copy_row_or_marker() -> None:
    panel, _model, parent_id, _phase_ids = _panel_with_partition()
    assert not [key for key in _row_keys(panel) if str(key).startswith("copy:")]
    assert panel._run_to_groups[1] == [parent_id]
    assert all("①" not in label for label in _row_labels(panel))


def test_an_unrelated_group_still_makes_a_copy_row() -> None:
    panel, _model, _parent_id, phase_ids = _panel_with_partition()
    other_id = panel.create_data_group([1, 2], name="Recheck")
    assert f"copy:{other_id}:1" in _row_keys(panel)
    # …and the run keeps its single row under its phase.
    assert f"phase:{phase_ids[0]}:1" in _row_keys(panel)


def test_phase_groups_never_reach_the_top_level_display_order() -> None:
    panel, _model, parent_id, _phase_ids = _panel_with_partition()
    assert panel._display_order == [parent_id]


def test_collapsing_one_phase_hides_only_its_runs() -> None:
    panel, _model, _parent_id, phase_ids = _panel_with_partition()
    panel._toggle_group_collapsed(phase_ids[0])
    keys = _row_keys(panel)
    assert f"group:{phase_ids[0]}" in keys
    assert f"phase:{phase_ids[0]}:1" not in keys
    assert f"phase:{phase_ids[1]}:4" in keys
    assert (
        panel._table.item(_row_of_key(panel, f"group:{phase_ids[0]}"), 0).text().startswith("    ▸")
    )


def test_sorting_reorders_within_a_phase_and_keeps_the_ordinals() -> None:
    panel, _model, _parent_id, phase_ids = _panel_with_partition()
    panel._current_sort_column = 2  # T (K)
    panel._current_sort_order = Qt.SortOrder.AscendingOrder
    panel._sort_table()
    keys = _row_keys(panel)
    assert keys.index(f"group:{phase_ids[0]}") < keys.index(f"group:{phase_ids[1]}")
    assert [k for k in keys if str(k).startswith(f"phase:{phase_ids[0]}")] == [
        f"phase:{phase_ids[0]}:{rn}" for rn in _PHASE_I_RUNS
    ]


# ── selection ────────────────────────────────────────────────────────────────


def test_series_header_selects_every_phase_run_plus_the_excluded_one() -> None:
    panel, _model, parent_id, _phase_ids = _panel_with_partition()
    resolved = panel._dataset_run_numbers_from_keys([f"group:{parent_id}"])
    assert resolved == [1, 2, 3, 4, 5, 6]


def test_phase_header_selects_only_its_members() -> None:
    panel, _model, _parent_id, phase_ids = _panel_with_partition()
    assert panel._dataset_run_numbers_from_keys([f"group:{phase_ids[1]}"]) == list(_PHASE_II_RUNS)


def test_selecting_both_headers_deduplicates_the_runs() -> None:
    panel, _model, parent_id, phase_ids = _panel_with_partition()
    resolved = panel._dataset_run_numbers_from_keys([f"group:{parent_id}", f"group:{phase_ids[0]}"])
    assert resolved == [1, 2, 3, 4, 5, 6]


def test_selecting_a_phase_header_emits_group_selected_with_the_phase_id() -> None:
    panel, _model, _parent_id, phase_ids = _panel_with_partition()
    seen: list[str] = []
    panel.group_selected.connect(seen.append)
    panel._table.selectRow(_row_of_key(panel, f"group:{phase_ids[1]}"))
    assert seen == [phase_ids[1]]


def test_phase_member_row_resolves_to_its_run() -> None:
    panel, _model, _parent_id, phase_ids = _panel_with_partition()
    assert panel._dataset_run_numbers_from_keys([f"phase:{phase_ids[0]}:2"]) == [2]


# ── context menus ────────────────────────────────────────────────────────────


def _menu_texts(menu) -> list[str]:
    return [action.text() for action in menu.actions()]


def test_phase_header_menu_offers_the_phase_verbs() -> None:
    panel, _model, _parent_id, phase_ids = _panel_with_partition()
    panel._table.selectRow(_row_of_key(panel, f"group:{phase_ids[0]}"))
    texts = _menu_texts(panel._create_table_context_menu())
    assert "Collapse Phase" in texts
    assert "Rename Phase" in texts
    assert "Fit this phase…" in texts
    assert "Show series from this phase" in texts
    assert "Ungroup" in texts
    assert "Collapse Group" not in texts


def test_phase_header_ungroup_asks_the_host_to_dissolve_the_partition() -> None:
    panel, _model, parent_id, phase_ids = _panel_with_partition()
    panel._table.selectRow(_row_of_key(panel, f"group:{phase_ids[0]}"))
    menu = panel._create_table_context_menu()
    seen: list[str] = []
    panel.remove_phases_requested.connect(seen.append)
    next(a for a in menu.actions() if a.text() == "Ungroup").trigger()
    assert seen == [parent_id]


def test_phase_header_fit_emits_the_phase_id() -> None:
    panel, _model, _parent_id, phase_ids = _panel_with_partition()
    panel._table.selectRow(_row_of_key(panel, f"group:{phase_ids[1]}"))
    menu = panel._create_table_context_menu()
    seen: list[str] = []
    panel.fit_group_requested.connect(seen.append)
    next(a for a in menu.actions() if a.text() == "Fit this phase…").trigger()
    assert seen == [phase_ids[1]]


def test_member_row_menu_offers_move_to_phase_with_the_current_one_ticked() -> None:
    panel, _model, _parent_id, phase_ids = _panel_with_partition()
    panel._table.selectRow(_row_of_key(panel, f"phase:{phase_ids[0]}:2"))
    menu = panel._create_table_context_menu()
    submenu = next(a for a in menu.actions() if a.text() == "Move to phase").menu()
    entries = [(a.text(), a.isChecked()) for a in submenu.actions() if a.text()]
    assert entries == [
        ("Phase I · 1.8 – 16.0 K", True),
        ("Phase II · 22.0 – 28.0 K", False),
        ("Exclude from phases", False),
    ]


def test_excluded_row_menu_ticks_exclude_from_phases() -> None:
    panel, _model, _parent_id, _phase_ids = _panel_with_partition()
    panel._table.selectRow(_row_of_key(panel, _EXCLUDED_RUN))
    menu = panel._create_table_context_menu()
    submenu = next(a for a in menu.actions() if a.text() == "Move to phase").menu()
    assert next(a for a in submenu.actions() if a.text() == "Exclude from phases").isChecked()


def test_unpartitioned_group_member_gets_no_move_to_phase_menu() -> None:
    panel = DataBrowserPanel()
    panel.set_project_model(ProjectModel())
    panel.add_dataset(_dataset(1, temperature=2.0))
    panel.add_dataset(_dataset(2, temperature=3.0))
    panel.create_data_group([1, 2], name="Plain")
    panel._table.selectRow(_row_of_key(panel, 1))
    assert "Move to phase" not in _menu_texts(panel._create_table_context_menu())


# ── moving a run between phases ──────────────────────────────────────────────


def _phase_series(panel: DataBrowserPanel, model: ProjectModel, phase_id: str) -> FitSeries:
    """Record a fit series bound to *phase_id* over its current membership."""
    phase = model.data_group(phase_id)
    series = FitSeries(
        batch_id=f"batch-{phase_id}",
        rep_type=RepresentationType.TIME_FB_ASYMMETRY,
        member_run_numbers=list(phase.member_run_numbers),
        group_id=phase_id,
        last_fitted_members=list(phase.member_run_numbers),
    )
    model.add_batch(series)
    return series


def test_move_to_phase_moves_the_run_and_marks_both_phase_series_stale() -> None:
    panel, model, _parent_id, phase_ids = _panel_with_partition()
    series_i = _phase_series(panel, model, phase_ids[0])
    series_ii = _phase_series(panel, model, phase_ids[1])
    assert not series_i.is_stale(model.data_group(phase_ids[0]))
    assert not series_ii.is_stale(model.data_group(phase_ids[1]))

    seen: list[int] = []
    panel.group_membership_changed.connect(lambda: seen.append(1))
    panel.move_run_to_phase(3, phase_ids[1])

    assert model.data_group(phase_ids[0]).member_run_numbers == [1, 2]
    assert model.data_group(phase_ids[1]).member_run_numbers == [3, 4, 5]
    assert series_i.is_stale(model.data_group(phase_ids[0]))
    assert series_ii.is_stale(model.data_group(phase_ids[1]))
    # One report per affected phase — the vacated one and the target.
    assert len(seen) == 2


def test_move_to_phase_reorders_along_the_axis_and_updates_the_range() -> None:
    panel, model, _parent_id, phase_ids = _panel_with_partition()
    panel.move_run_to_phase(_EXCLUDED_RUN, phase_ids[1])
    phase_ii = model.data_group(phase_ids[1])
    assert phase_ii.member_run_numbers == [4, 5, 6]
    assert phase_ii.phase_range == (22.0, 40.0)
    assert model.excluded_runs_for(model.data_group(phase_ids[1]).parent_group_id) == []


def test_excluding_a_run_reports_only_the_vacated_phase() -> None:
    panel, model, parent_id, phase_ids = _panel_with_partition()
    seen: list[int] = []
    panel.group_membership_changed.connect(lambda: seen.append(1))
    panel.move_run_to_phase(1, None)
    assert model.data_group(phase_ids[0]).member_run_numbers == [2, 3]
    assert model.excluded_runs_for(parent_id) == [1, _EXCLUDED_RUN]
    assert len(seen) == 1


def test_move_to_phase_rerenders_the_run_under_its_new_phase() -> None:
    panel, _model, _parent_id, phase_ids = _panel_with_partition()
    panel.move_run_to_phase(3, phase_ids[1])
    keys = _row_keys(panel)
    assert f"phase:{phase_ids[0]}:3" not in keys
    assert f"phase:{phase_ids[1]}:3" in keys


def test_leaving_the_series_vacates_the_phase_too() -> None:
    panel, model, parent_id, phase_ids = _panel_with_partition()
    panel._remove_memberships([(1, parent_id)])
    assert 1 not in model.data_group(parent_id).member_run_numbers
    assert 1 not in model.data_group(phase_ids[0]).member_run_numbers


def test_remove_phases_returns_the_runs_to_the_series() -> None:
    panel, model, parent_id, phase_ids = _panel_with_partition()
    panel.remove_phases(parent_id)
    assert model.phase_groups_for(parent_id) == []
    assert model.data_group(parent_id).member_run_numbers == [1, 2, 3, 4, 5, 6]
    assert _row_keys(panel) == [f"group:{parent_id}", 1, 2, 3, 4, 5, 6]
    assert all(pid not in model.data_groups for pid in phase_ids)


# ── the ⓘ popover ────────────────────────────────────────────────────────────


def _popover_text(panel: DataBrowserPanel) -> str:
    popover = panel._phase_popover
    return "\n".join(
        label.text() for label in popover.findChildren(type(popover._title)) if label.text()
    )


def test_phase_popover_shows_range_boundaries_and_provenance() -> None:
    panel, _model, _parent_id, phase_ids = _panel_with_partition()
    panel._open_phase_popover(phase_ids[0], _row_of_key(panel, f"group:{phase_ids[0]}"))
    text = _popover_text(panel)
    assert "Phase I" in text
    assert "1.8 – 16.0 K · 3 runs" in text
    assert "series end" in text
    assert "19.0 ± 3.0 K" in text
    assert "Two damped lines" in text
    assert "converged · 1.04 · high" in text
    assert "lambda, phi" in text
    assert "2026-09-06T09:00:00 · 1 breaks · gains 12.4" in text


def test_series_popover_summarises_the_partition() -> None:
    panel, _model, parent_id, _phase_ids = _panel_with_partition()
    panel._open_phase_popover(parent_id, _row_of_key(panel, f"group:{parent_id}"))
    text = _popover_text(panel)
    assert "ZF temperature scan" in text
    assert "2 phases · 1 transition" in text
    assert "Phase I · 1.8 – 16.0 K" in text


def test_popover_omits_rows_for_provenance_a_phase_does_not_carry_yet() -> None:
    panel, model, parent_id, _phase_ids = _panel_with_partition()
    (bare_id,) = model.create_phase_groups(
        parent_id,
        [
            PhaseSpec(
                ordinal=1,
                name="Phase I",
                member_run_numbers=(1, 2, 3),
                phase_range=(1.8, 16.0),
                phase_boundaries={"lower": None, "upper": None},
                phase_color=None,
                phase_provenance={},
            )
        ],
    )
    panel.sync_groups_from_project_model()
    panel._open_phase_popover(bare_id, _row_of_key(panel, f"group:{bare_id}"))
    text = _popover_text(panel)
    assert "Phase I" in text
    assert "Two damped lines" not in text
    assert "series end" in text  # Boundaries always render; both ends are open.


def test_popover_actions_reach_the_panel_signals() -> None:
    panel, _model, _parent_id, phase_ids = _panel_with_partition()
    panel._open_phase_popover(phase_ids[1], _row_of_key(panel, f"group:{phase_ids[1]}"))
    seen: list[str] = []
    panel.fit_group_requested.connect(seen.append)
    panel._phase_popover._fit_button.click()
    assert seen == [phase_ids[1]]


# ── state round-trip ─────────────────────────────────────────────────────────


def test_state_round_trip_preserves_collapsed_phases() -> None:
    panel, _model, parent_id, phase_ids = _panel_with_partition()
    panel._toggle_group_collapsed(phase_ids[1])
    state = panel.get_state()
    assert [entry["group_id"] for entry in state["data_groups"]] == [parent_id, *phase_ids]
    assert [entry["collapsed"] for entry in state["data_groups"]] == [False, False, True]

    panel.restore_state(state)
    assert panel._is_collapsed(phase_ids[1])
    keys = _row_keys(panel)
    assert f"phase:{phase_ids[1]}:4" not in keys
    assert f"phase:{phase_ids[0]}:1" in keys


def test_a_standalone_restore_rebuilds_the_phases_from_the_state_block() -> None:
    source, _model, parent_id, phase_ids = _panel_with_partition()
    state = source.get_state()

    fresh = DataBrowserPanel()
    fresh.set_project_model(ProjectModel())
    with fresh.batch_updates():
        for rn, temperature in zip(
            (1, 2, 3, 4, 5, 6), (1.8, 8.0, 16.0, 22.0, 28.0, 40.0), strict=True
        ):
            fresh.add_dataset(_dataset(rn, temperature=temperature))
    fresh.restore_state(state)

    rebuilt = fresh._project_model.phase_groups_for(parent_id)
    assert [p.group_id for p in rebuilt] == phase_ids
    assert [p.phase_ordinal for p in rebuilt] == [1, 2]
    assert rebuilt[0].phase_range == (1.8, 16.0)
    assert rebuilt[0].phase_boundaries == {"lower": None, "upper": (19.0, 3.0)}
    assert rebuilt[0].phase_provenance["model_title"] == "Two damped lines"
    assert _row_keys(fresh) == _row_keys(source)


# ── header hit zones ─────────────────────────────────────────────────────────


def test_phase_chevron_zone_sits_one_indent_right_of_a_group_chevron() -> None:
    panel, _model, parent_id, phase_ids = _panel_with_partition()
    series_edge = panel._chevron_right_edge(_row_of_key(panel, f"group:{parent_id}"))
    phase_edge = panel._chevron_right_edge(_row_of_key(panel, f"group:{phase_ids[0]}"))
    assert phase_edge > series_edge


def test_info_indicator_zone_is_the_right_edge_of_the_count_column() -> None:
    panel, _model, _parent_id, phase_ids = _panel_with_partition()
    row = _row_of_key(panel, f"group:{phase_ids[0]}")
    right = panel._table.columnViewportPosition(3) + panel._table.columnWidth(3)
    assert panel._hit_info_indicator(row, right - 1)
    assert not panel._hit_info_indicator(row, panel._table.columnViewportPosition(3))


def test_an_unpartitioned_group_header_has_no_info_zone() -> None:
    panel = DataBrowserPanel()
    panel.set_project_model(ProjectModel())
    panel.add_dataset(_dataset(1, temperature=2.0))
    panel.add_dataset(_dataset(2, temperature=3.0))
    gid = panel.create_data_group([1, 2], name="Plain")
    row = _row_of_key(panel, f"group:{gid}")
    right = panel._table.columnViewportPosition(3) + panel._table.columnWidth(3)
    assert not panel._hit_info_indicator(row, right - 1)
