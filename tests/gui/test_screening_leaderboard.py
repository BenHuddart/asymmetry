"""Standalone tests for the ScreeningLeaderboard widget."""

from __future__ import annotations

import math
import os
from dataclasses import replace

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QTableWidget

from asymmetry.core.fitting.fit_wizard import SelectionMetric
from asymmetry.core.fitting.model_comparison import (
    CandidateSummary,
    RunFit,
    summarise_candidates,
)
from asymmetry.gui.styles import tokens
from asymmetry.gui.widgets.screening_leaderboard import (
    COLLAPSED_UNTICKED_ROWS,
    NUMERIC_CHI2_MAX_RUNS,
    OptimisationStatus,
    ScreeningLeaderboard,
    format_delta,
)
from tests.core.test_model_comparison import _FIELDS, _RUNS, _assessment, _datasets, _fit

_RUN_LABELS = {run: f"{_FIELDS[run]:g} G" for run in _RUNS}
_METRIC = "ΔAICc from best"


def _summaries(count: int = 3, **overrides) -> tuple[CandidateSummary, ...]:
    return summarise_candidates(
        [
            _assessment(f"family{index}", aicc=100.0 + 7.5 * index, prescreen=True, **overrides)
            for index in range(count)
        ],
        _datasets(),
        SelectionMetric.AICC,
    )


def _statuses(summaries, status=OptimisationStatus.NOT_OPTIMISED) -> dict:
    return {summary.key: status for summary in summaries}


def _board(summaries, *, ticked=(), run_labels=None) -> ScreeningLeaderboard:
    board = ScreeningLeaderboard()
    board.set_candidates(
        summaries,
        _RUN_LABELS if run_labels is None else run_labels,
        _METRIC,
        ticked=ticked,
        statuses=_statuses(summaries),
    )
    return board


def _table(board: ScreeningLeaderboard) -> QTableWidget:
    return board.findChild(QTableWidget)


def _show_all(board: ScreeningLeaderboard) -> QPushButton:
    return board.findChild(QPushButton)


def _visible_rows(board: ScreeningLeaderboard) -> list[int]:
    table = _table(board)
    return [row for row in range(table.rowCount()) if not table.isRowHidden(row)]


def test_rows_follow_rank_with_title_delta_and_ticks(qapp: QApplication) -> None:
    summaries = _summaries()
    board = _board(summaries, ticked=("family0",))
    table = _table(board)
    headers = [table.horizontalHeaderItem(c).text() for c in range(table.columnCount())]
    assert headers == ["Family", _METRIC, "100 G", "200 G", "300 G", "Status"]
    assert [table.item(row, 0).text() for row in range(3)] == [s.title for s in summaries]
    assert [table.item(row, 1).text() for row in range(3)] == ["best", "+7.5", "+15.0"]
    assert board.ticked() == ("family0",)


@pytest.mark.parametrize(
    ("delta", "text"),
    [(0.0, "best"), (3.26, "+3.3"), (19.94, "+19.9"), (1234.4, "+1 234"), (math.inf, "—")],
)
def test_format_delta(delta: float, text: str) -> None:
    assert format_delta(delta) == text


def test_chi_squared_cells_are_graded_and_always_printed(qapp: QApplication) -> None:
    fits = {
        701: replace(_fit(701), reduced_chi_squared=1.2),
        702: replace(_fit(702), reduced_chi_squared=3.0),
        703: replace(_fit(703), reduced_chi_squared=12.0),
    }
    board = _board(_summaries(1, fits=fits))
    table = _table(board)
    cells = [table.item(0, column) for column in (2, 3, 4)]
    assert [cell.text() for cell in cells] == ["1.20", "3.00", "12.00"]
    assert [cell.background().color().name() for cell in cells] == [
        tokens.SUCCESS_SOFT,
        tokens.WARN_SOFT,
        tokens.ERROR_SOFT,
    ]
    assert cells[2].toolTip() == "300 G: χ²ᵣ 12.00"


def test_a_run_the_prescreen_never_fitted_reads_as_not_fitted(qapp: QApplication) -> None:
    fits = {701: _fit(701), 702: _fit(702)}
    board = _board(_summaries(1, fits=fits))
    cell = _table(board).item(0, 4)
    assert cell.text() == "—"
    assert cell.toolTip() == "300 G: not fitted"


def test_status_badges_show_the_optimisation_state(qapp: QApplication) -> None:
    summaries = _summaries(2)
    board = ScreeningLeaderboard()
    board.set_candidates(
        summaries,
        _RUN_LABELS,
        _METRIC,
        ticked=(),
        statuses={"family0": OptimisationStatus.OPTIMISED, "family1": OptimisationStatus.FAILED},
    )
    table = _table(board)
    assert [table.item(row, 5).text() for row in range(2)] == ["Optimised", "Failed"]
    assert table.item(1, 5).background().color().name() == tokens.ERROR_SOFT


def test_every_family_needs_a_status(qapp: QApplication) -> None:
    board = ScreeningLeaderboard()
    with pytest.raises(KeyError):
        board.set_candidates(_summaries(2), _RUN_LABELS, _METRIC, ticked=(), statuses={})


def test_ticking_a_row_emits_the_ticked_keys_in_rank_order(qapp: QApplication) -> None:
    board = _board(_summaries(), ticked=("family2",))
    emitted: list[tuple] = []
    board.ticked_changed.connect(emitted.append)
    _table(board).item(0, 0).setCheckState(Qt.CheckState.Checked)
    assert emitted == [("family0", "family2")]
    assert board.ticked() == ("family0", "family2")


def test_set_candidates_is_signal_silent(qapp: QApplication) -> None:
    board = ScreeningLeaderboard()
    events: list[object] = []
    board.ticked_changed.connect(events.append)
    board.selected_changed.connect(events.append)
    summaries = _summaries()
    board.set_candidates(
        summaries, _RUN_LABELS, _METRIC, ticked=("family1",), statuses=_statuses(summaries)
    )
    assert events == []


def test_first_row_is_selected_and_a_click_selects_another(qapp: QApplication) -> None:
    board = _board(_summaries())
    assert board.selected_key() == "family0"
    selected: list[str] = []
    board.selected_changed.connect(selected.append)
    _table(board).setCurrentCell(2, 3)
    assert selected == ["family2"]
    assert board.selected_key() == "family2"


def test_selection_survives_new_candidates_that_still_list_it(qapp: QApplication) -> None:
    summaries = _summaries()
    board = _board(summaries)
    _table(board).setCurrentCell(1, 0)
    board.set_candidates(
        tuple(reversed(summaries)), _RUN_LABELS, _METRIC, ticked=(), statuses=_statuses(summaries)
    )
    assert board.selected_key() == "family1"
    board.set_candidates(
        summaries[:1], _RUN_LABELS, _METRIC, ticked=(), statuses=_statuses(summaries)
    )
    assert board.selected_key() == "family0"


def test_an_empty_board_selects_nothing(qapp: QApplication) -> None:
    board = _board(())
    assert board.selected_key() is None
    assert board.ticked() == ()


def test_many_families_collapse_to_ticked_and_leading_rows(qapp: QApplication) -> None:
    board = _board(_summaries(8), ticked=("family0", "family5"))
    unticked_shown = [1, 2, 3][:COLLAPSED_UNTICKED_ROWS]
    assert _visible_rows(board) == sorted({0, 5, *unticked_shown})
    toggle = _show_all(board)
    assert not toggle.isHidden()
    assert toggle.text() == "Show all 8 families"
    toggle.click()
    assert _visible_rows(board) == list(range(8))
    assert toggle.text() == "Show fewer families"


def test_unticking_never_hides_the_row_under_the_pointer(qapp: QApplication) -> None:
    board = _board(_summaries(8), ticked=("family5",))
    _table(board).item(5, 0).setCheckState(Qt.CheckState.Unchecked)
    assert 5 in _visible_rows(board)


def test_few_families_need_no_toggle(qapp: QApplication) -> None:
    board = _board(_summaries(3))
    assert _visible_rows(board) == [0, 1, 2]
    assert _show_all(board).isHidden()


def test_many_runs_draw_a_heat_strip_with_values_in_tooltips(qapp: QApplication) -> None:
    count = NUMERIC_CHI2_MAX_RUNS + 1
    time = np.linspace(0.0, 1.0, 5)
    runs = tuple(
        RunFit(800 + index, str(800 + index), float(index), (time, time), (time, time), 1.0 + index)
        for index in range(count)
    )
    summary = replace(_summaries(1)[0], runs=runs)
    labels = {800 + index: f"{index} K" for index in range(count)}
    board = _board((summary,), run_labels=labels)
    table = _table(board)
    cells = [table.item(0, 2 + index) for index in range(count)]
    assert all(cell.text() == "" for cell in cells)
    assert cells[3].toolTip() == "3 K: χ²ᵣ 4.00"
    assert table.horizontalHeaderItem(2).text() == ""
    assert table.horizontalHeaderItem(2).toolTip() == "0 K"
    caption = next(label for label in board.findChildren(QLabel) if "hover" in label.text())
    assert not caption.isHidden()
    assert caption.text().startswith(f"χ²ᵣ per run, 0 K → {count - 1} K")


def test_board_paints(qapp: QApplication) -> None:
    board = _board(_summaries(8), ticked=("family0",))
    board.resize(640, board.sizeHint().height())
    assert not board.grab().isNull()
