"""Standalone tests for the ModelComparePanel widget."""

from __future__ import annotations

import math
import os
from dataclasses import replace

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QScrollArea, QTableWidget

from asymmetry.core.fitting.fit_wizard import SelectionMetric
from asymmetry.core.fitting.model_comparison import (
    CandidateSummary,
    Estimate,
    ParameterFlag,
    ParameterRole,
    ParameterRow,
    summarise_candidates,
)
from asymmetry.gui.styles import tokens
from asymmetry.gui.widgets.elided_label import ElidedLabel
from asymmetry.gui.widgets.model_compare_panel import (
    EMPTY_BOARD,
    MAX_FLAG_LINES,
    NO_B_LEGEND,
    NO_LOCAL_TREND,
    CompareRow,
    ModelComparePanel,
    format_weight,
)
from tests.core.test_model_comparison import _FIELDS, _RUNS, _assessment, _datasets, _fit

_LABELS = [f"{_FIELDS[run]:g} G" for run in _RUNS]
_METRIC = "ΔAICc from best"


def _pool(*assessments, titles: tuple[str, ...]) -> tuple[CandidateSummary, ...]:
    summaries = summarise_candidates(list(assessments), _datasets(), SelectionMetric.AICC)
    return tuple(
        replace(summary, title=title) for summary, title in zip(summaries, titles, strict=True)
    )


def _default_pool() -> tuple[CandidateSummary, ...]:
    return _pool(
        _assessment("lf|shared", aicc=100.0),
        _assessment("lf|local", aicc=102.0, global_names=(), local_names=("A_1", "Lambda")),
        _assessment("gkt|x", aicc=130.0),
        titles=("LF", "LF", "GKT"),
    )


def _panel(summaries=None, **kwargs) -> ModelComparePanel:
    panel = ModelComparePanel()
    panel.set_series(_datasets(), _LABELS, [_FIELDS[run] for run in _RUNS], "Field (G)")
    panel.set_candidates(_default_pool() if summaries is None else summaries, _METRIC, **kwargs)
    panel.resize(1180, 600)
    return panel


def _rows(panel: ModelComparePanel) -> dict[str, CompareRow]:
    return {row.key: row for row in panel.findChildren(CompareRow) if not row.isHidden()}


def _board_entries(panel: ModelComparePanel) -> list[tuple[str, str]]:
    layout = panel.findChild(QScrollArea).widget().layout()
    entries = []
    for index in range(layout.count()):
        widget = layout.itemAt(index).widget()
        if isinstance(widget, CompareRow):
            entries.append(("row", widget.key))
        elif isinstance(widget, QLabel):
            entries.append(("title", widget.text()))
    return entries


def _table(panel: ModelComparePanel) -> QTableWidget:
    return panel.findChild(QTableWidget)


def _column(panel: ModelComparePanel, column: int) -> list[str]:
    table = _table(panel)
    return [table.item(row, column).text() for row in range(table.rowCount())]


def _legend(panel: ModelComparePanel) -> list[str]:
    return [
        label.text()
        for label in panel.findChildren(ElidedLabel)
        if not isinstance(label.parentWidget(), CompareRow) and not label.isHidden()
    ]


def _continue(panel: ModelComparePanel) -> QPushButton:
    return next(
        button for button in panel.findChildren(QPushButton) if button.text().startswith("Continue")
    )


def _signals(panel: ModelComparePanel) -> list[tuple[str, object]]:
    events: list[tuple[str, object]] = []
    panel.a_changed.connect(lambda key: events.append(("a", key)))
    panel.b_changed.connect(lambda key: events.append(("b", key)))
    panel.continue_requested.connect(lambda key: events.append(("continue", key)))
    return events


def _trend_axes(panel: ModelComparePanel):
    return panel.trend_figure.axes[0]


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------


def test_rows_group_consecutive_titles_in_rank_order(qapp: QApplication) -> None:
    summaries = _pool(
        _assessment("lf|a", aicc=100.0),
        _assessment("lf|b", aicc=101.0),
        _assessment("gkt|a", aicc=102.0),
        _assessment("lf|c", aicc=103.0),
        titles=("LF", "LF", "GKT", "LF"),
    )
    panel = _panel(summaries)
    assert _board_entries(panel) == [
        ("title", "LF"),
        ("row", "lf|a"),
        ("row", "lf|b"),
        ("title", "GKT"),
        ("row", "gkt|a"),
        ("title", "LF"),
        ("row", "lf|c"),
    ]


def test_rows_carry_role_chips_and_a_fixed_line(qapp: QApplication) -> None:
    rows = _rows(_panel())
    shared, local = rows["lf|shared"], rows["lf|local"]
    assert [chip.text() for chip in shared.chips] == ["Global A_1", "Local λ"]
    assert [chip.text() for chip in local.chips] == ["Local A_1", "Local λ"]
    assert tokens.ACCENT_SOFT in shared.chips[0].styleSheet()
    assert tokens.WARN_BANNER_BG in shared.chips[1].styleSheet()
    texts = [label.text() for label in shared.findChildren(QLabel)]
    assert "Fixed: A_bg" in texts

    unfixed = _pool(_assessment("x|1", aicc=1.0), titles=("X",))
    unfixed = (replace(unfixed[0], fixed_names=()),)
    row = _rows(_panel(unfixed))["x|1"]
    assert not any(label.text().startswith("Fixed:") for label in row.findChildren(QLabel))


def test_rows_show_delta_and_evidence_weight(qapp: QApplication) -> None:
    summaries = _default_pool()
    rows = _rows(_panel(summaries))
    assert [rows[s.key].delta_bar.toolTip() for s in summaries] == [
        f"{_METRIC}: best",
        f"{_METRIC}: +2.0",
        f"{_METRIC}: +30",
    ]
    assert [rows[s.key].weight.text() for s in summaries] == [
        format_weight(s.weight) for s in summaries
    ]
    assert rows["gkt|x"].weight.text() == "<1%"


@pytest.mark.parametrize(
    ("weight", "text"), [(0.0, "0%"), (0.004, "<1%"), (0.005, "0%"), (0.62, "62%"), (1.0, "100%")]
)
def test_format_weight(weight: float, text: str) -> None:
    assert format_weight(weight) == text


def test_gate_badge_reads_pass_or_warn_with_the_summary(qapp: QApplication) -> None:
    summaries = _pool(
        _assessment("x|pass", aicc=1.0),
        _assessment(
            "x|warn", aicc=2.0, gate_reasons={702: ("runs-test z score suggests structure",)}
        ),
        titles=("X", "X"),
    )
    rows = _rows(_panel(summaries))
    assert rows["x|pass"].gate.text() == "Pass"
    assert rows["x|pass"].gate.toolTip() == "Every run passes the residual gate"
    assert rows["x|warn"].gate.text() == "Warn"
    assert rows["x|warn"].gate.toolTip() == "runs-test z score suggests structure (run 702)"
    assert tokens.WARN_SOFT in rows["x|warn"].gate.styleSheet()


def test_flag_lines_name_the_runs_that_earned_each_flag(qapp: QApplication) -> None:
    fits = {run: _fit(run) for run in _RUNS}
    fits[702] = _fit(702, a_1=0.0)
    summaries = _pool(
        _assessment(
            "x|1",
            aicc=1.0,
            fits=fits,
            global_names=(),
            local_names=("A_1", "Lambda"),
            gate_reasons={702: ("A_1 at lower bound",)},
        ),
        titles=("X",),
    )
    row = _rows(_panel(summaries))["x|1"]
    # The gate's bound-hit reason is the flag's own fact, so it is said once.
    assert [label.text() for label in row.flag_labels] == [
        "A_1 at lower bound at 200 G",
        "A_1 poorly determined at 200 G",
    ]
    assert all(label.toolTip() == label.text() for label in row.flag_labels)
    assert all(label.pen_color().name() == tokens.WARN for label in row.flag_labels)


def test_a_flag_every_run_earned_reads_at_every_run(qapp: QApplication) -> None:
    fits = {run: _fit(run, a_1_err=30.0) for run in _RUNS}
    summaries = _pool(
        _assessment("x|1", aicc=1.0, fits=fits, global_names=(), local_names=("A_1", "Lambda")),
        titles=("X",),
    )
    row = _rows(_panel(summaries))["x|1"]
    assert [label.text() for label in row.flag_labels] == ["A_1 poorly determined at every run"]


def test_a_shared_global_flag_names_no_run(qapp: QApplication) -> None:
    fits = {run: _fit(run, a_1_err=30.0) for run in _RUNS}
    summaries = _pool(_assessment("x|1", aicc=1.0, fits=fits), titles=("X",))
    row = _rows(_panel(summaries))["x|1"]
    assert [label.text() for label in row.flag_labels] == ["A_1 poorly determined"]


def test_flag_lines_beyond_the_cap_sit_behind_a_more_line(qapp: QApplication) -> None:
    fits = {run: _fit(run) for run in _RUNS}
    fits[702] = _fit(702, a_1=0.0)
    (summary,) = _pool(
        _assessment(
            "x|1",
            aicc=1.0,
            fits=fits,
            global_names=(),
            local_names=("A_1", "Lambda"),
            gate_reasons={702: ("runs-test z score suggests structure",)},
        ),
        titles=("X",),
    )
    sigma = ParameterRow(
        "Sigma",
        ParameterRole.LOCAL,
        (Estimate(1.0, 0.1),) * 3,
        run_flags=((ParameterFlag.AT_UPPER_BOUND,), (), ()),
    )
    summary = replace(summary, parameters=(*summary.parameters, sigma))
    row = _rows(_panel((summary,)))["x|1"]
    texts = [label.text() for label in row.flag_labels]
    assert len(texts) == MAX_FLAG_LINES + 1
    assert texts[-1] == "+1 more"
    assert row.flag_labels[-1].toolTip() == "Sigma at upper bound at 100 G"


# ---------------------------------------------------------------------------
# A and B
# ---------------------------------------------------------------------------


def test_a_defaults_to_the_first_summary_and_set_candidates_is_silent(qapp: QApplication) -> None:
    panel = ModelComparePanel()
    events = _signals(panel)
    panel.set_series(_datasets(), _LABELS, [_FIELDS[run] for run in _RUNS], "Field (G)")
    panel.set_candidates(_default_pool(), _METRIC)
    assert panel.a_key() == "lf|shared"
    assert panel.b_key() is None
    panel.set_candidates(_default_pool(), _METRIC, a_key="gkt|x")
    assert panel.a_key() == "gkt|x"
    assert events == []
    rows = _rows(panel)
    assert [rows[key].disc.letter for key in ("lf|shared", "lf|local", "gkt|x")] == ["", "", "A"]


def test_clicking_a_row_makes_it_a(qapp: QApplication) -> None:
    panel = _panel()
    events = _signals(panel)
    row = _rows(panel)["lf|local"]
    QTest.mouseClick(row, Qt.MouseButton.LeftButton)
    assert events == [("a", "lf|local")]
    assert panel.a_key() == "lf|local"
    assert row.disc.letter == "A"
    assert row.pin_button.isHidden()
    QTest.mouseClick(row, Qt.MouseButton.LeftButton)
    assert events == [("a", "lf|local")]


def test_pin_as_b_toggles_and_emits(qapp: QApplication) -> None:
    panel = _panel()
    events = _signals(panel)
    row = _rows(panel)["gkt|x"]
    assert row.pin_button.text() == "Pin as B"
    row.pin_button.click()
    assert events == [("b", "gkt|x")]
    assert panel.b_key() == "gkt|x"
    assert row.disc.letter == "B"
    assert row.pin_button.text() == "Unpin B"
    assert not _rows(panel)["lf|shared"].pin_button.isVisibleTo(panel)
    row.pin_button.click()
    assert events == [("b", "gkt|x"), ("b", None)]
    assert panel.b_key() is None
    assert row.disc.letter == ""


def test_picking_b_as_a_clears_b(qapp: QApplication) -> None:
    panel = _panel()
    panel.set_b("lf|local")
    events = _signals(panel)
    QTest.mouseClick(_rows(panel)["lf|local"], Qt.MouseButton.LeftButton)
    assert events == [("a", "lf|local"), ("b", None)]
    assert (panel.a_key(), panel.b_key()) == ("lf|local", None)


def test_set_b_rejects_a_and_unknown_keys(qapp: QApplication) -> None:
    panel = _panel()
    with pytest.raises(ValueError, match="differ"):
        panel.set_b("lf|shared")
    with pytest.raises(KeyError):
        panel.set_b("nope")
    with pytest.raises(KeyError):
        panel.set_a("nope")
    with pytest.raises(KeyError):
        panel.set_candidates(_default_pool(), _METRIC, a_key="nope")


def test_set_candidates_keeps_b_while_listed_and_clears_it_silently(qapp: QApplication) -> None:
    panel = _panel()
    panel.set_b("lf|local")
    events = _signals(panel)
    pool = _default_pool()
    panel.set_candidates(tuple(reversed(pool)), _METRIC, a_key="lf|shared")
    assert panel.b_key() == "lf|local"
    assert _rows(panel)["lf|local"].disc.letter == "B"
    panel.set_candidates(pool, _METRIC, a_key="lf|local")
    assert panel.b_key() is None
    panel.set_b("gkt|x")
    events.clear()
    panel.set_candidates(pool[:2], _METRIC)
    assert panel.b_key() is None
    assert events == []


@pytest.mark.parametrize("key", [Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space])
def test_keyboard_sets_a_on_a_focused_row(qapp: QApplication, key: Qt.Key) -> None:
    panel = _panel()
    events = _signals(panel)
    row = _rows(panel)["gkt|x"]
    assert row.focusPolicy() == Qt.FocusPolicy.StrongFocus
    QTest.keyClick(row, key)
    assert events == [("a", "gkt|x")]


def test_legend_names_a_and_b_or_invites_a_pin(qapp: QApplication) -> None:
    panel = _panel()
    assert _legend(panel) == ["— A: LF · shares A_1", NO_B_LEGEND]
    panel.set_b("lf|local")
    assert _legend(panel) == ["— A: LF · shares A_1", "- - B: LF · all per run"]


# ---------------------------------------------------------------------------
# Parameter table
# ---------------------------------------------------------------------------


def test_table_lists_a_alone_with_roles_and_values(qapp: QApplication) -> None:
    panel = _panel()
    table = _table(panel)
    assert table.isColumnHidden(2)
    assert _column(panel, 0) == ["A_1 · shared", "λ · per run", "A_bg · fixed"]
    assert _column(panel, 1) == ["20.00(20)", "0.200(10) · 0.300(10) · 0.500(10)", "1 · 1 · 1"]
    assert table.item(1, 1).toolTip() == "100 G: 0.200(10)\n200 G: 0.300(10)\n300 G: 0.500(10)"
    assert _column(panel, 3) == ["", "", ""]


def test_table_pairs_a_and_b_with_the_sigma_difference(qapp: QApplication) -> None:
    summaries = _pool(
        _assessment("x|a", aicc=1.0, fits={r: _fit(r, a_1=20.0, a_1_err=0.3) for r in _RUNS}),
        _assessment("x|b", aicc=2.0, fits={r: _fit(r, a_1=21.55, a_1_err=0.4) for r in _RUNS}),
        titles=("X", "X"),
    )
    panel = _panel(summaries)
    panel.set_b("x|b")
    assert not _table(panel).isColumnHidden(2)
    assert _column(panel, 2)[0] == "21.6(4)"
    assert _column(panel, 3)[0] == "differs by 3.1σ"


def test_a_parameter_with_different_roles_names_both(qapp: QApplication) -> None:
    summaries = _pool(
        _assessment("x|a", aicc=1.0),
        _assessment("x|b", aicc=2.0, global_names=(), local_names=("A_1", "Lambda")),
        titles=("X", "X"),
    )
    panel = _panel(summaries)
    panel.set_b("x|b")
    assert _column(panel, 0)[0] == "A_1 · shared in A, per run in B"


def test_flagged_values_wear_the_warning_or_error_colour(qapp: QApplication) -> None:
    fits = {run: _fit(run) for run in _RUNS}
    fits[702] = _fit(702, a_1=0.0)
    fits[703] = _fit(703, a_1=math.nan)
    summaries = _pool(
        _assessment("x|a", aicc=1.0, fits=fits, global_names=(), local_names=("A_1", "Lambda")),
        _assessment(
            "x|b",
            aicc=2.0,
            fits={701: fits[701], 702: fits[702], 703: _fit(703)},
            global_names=(),
            local_names=("A_1", "Lambda"),
        ),
        titles=("X", "X"),
    )
    panel = _panel(summaries)
    panel.set_b("x|b")
    table = _table(panel)
    assert table.item(0, 1).text() == "20.00(20) · 0.00(20) · —"
    assert table.item(0, 1).foreground().color().name() == tokens.ERROR
    assert table.item(0, 2).foreground().color().name() == tokens.WARN
    assert table.item(1, 1).data(Qt.ItemDataRole.ForegroundRole) is None
    assert table.item(0, 3).text() == (
        "A: at lower bound, poorly determined, not finite · B: at lower bound, poorly determined"
    )
    assert "200 G: 0.00(20) — at lower bound — poorly determined" in table.item(0, 1).toolTip()


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------


def _click_table_row(panel: ModelComparePanel, row: int) -> None:
    panel.show()
    table = _table(panel)
    rect = table.visualItemRect(table.item(row, 0))
    QTest.mouseClick(table.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())


def test_trend_defaults_to_a_first_local_parameter(qapp: QApplication) -> None:
    panel = _panel()
    axes = _trend_axes(panel)
    assert axes.get_ylabel() == "λ (µs⁻¹)"
    assert axes.get_xlabel() == "Field (G)"
    filled = [line for line in axes.lines if line.get_marker() == "o"]
    assert [line.get_markerfacecolor() for line in filled] == [tokens.PLOT_DATA]
    assert list(filled[0].get_xdata()) == [_FIELDS[run] for run in _RUNS]
    assert _table(panel).item(1, 0).background().color().name() == tokens.ACCENT_SOFT


def test_clicking_a_local_row_switches_the_trend_and_b_is_hollow(qapp: QApplication) -> None:
    panel = _panel(a_key="lf|local")
    panel.set_b("lf|shared")
    assert _trend_axes(panel).get_ylabel() == "A_1 (%)"
    _click_table_row(panel, 1)
    axes = _trend_axes(panel)
    assert axes.get_ylabel() == "λ (µs⁻¹)"
    markers = [line for line in axes.lines if line.get_marker() == "o"]
    assert [line.get_markerfacecolor() for line in markers] == [tokens.PLOT_DATA, tokens.SURFACE]
    assert [text.get_text() for text in axes.get_legend().get_texts()] == ["A", "B"]


def test_a_shared_value_on_the_trend_is_a_flat_line(qapp: QApplication) -> None:
    panel = _panel(a_key="lf|shared")
    panel.set_b("lf|local")
    _click_table_row(panel, 0)
    axes = _trend_axes(panel)
    assert axes.get_ylabel() == "A_1 (%)"
    flat = [line for line in axes.lines if line.get_label() == "A"]
    assert list(flat[0].get_ydata()) == [20.0, 20.0]


def test_clicking_a_shared_or_fixed_row_keeps_the_trend(qapp: QApplication) -> None:
    panel = _panel()
    _click_table_row(panel, 2)
    _click_table_row(panel, 0)
    assert _trend_axes(panel).get_ylabel() == "λ (µs⁻¹)"


def test_an_a_with_no_local_parameters_shows_the_placeholder(qapp: QApplication) -> None:
    summaries = _pool(
        _assessment("x|1", aicc=1.0, global_names=("A_1", "Lambda"), local_names=()),
        titles=("X",),
    )
    panel = _panel(summaries)
    assert [text.get_text() for text in _trend_axes(panel).texts] == [NO_LOCAL_TREND]


# ---------------------------------------------------------------------------
# Empty state and Continue
# ---------------------------------------------------------------------------


def test_an_empty_panel_says_so_and_cannot_continue(qapp: QApplication) -> None:
    panel = _panel(())
    assert _board_entries(panel) == [("title", EMPTY_BOARD)]
    assert panel.a_key() is None
    assert not _continue(panel).isEnabled()
    assert _table(panel).rowCount() == 0
    assert _legend(panel) == []


def test_continue_emits_a_and_the_host_can_hold_it_back(qapp: QApplication) -> None:
    panel = _panel()
    events = _signals(panel)
    button = _continue(panel)
    assert button.text() == "Continue with A →"
    button.click()
    assert events == [("continue", "lf|shared")]
    panel.set_continue_enabled(False)
    assert not button.isEnabled()
    panel.set_a("gkt|x")
    assert not button.isEnabled()
    panel.set_continue_enabled(True)
    assert button.isEnabled()


def test_panel_paints(qapp: QApplication) -> None:
    panel = _panel()
    panel.set_b("lf|local")
    assert not panel.grab().isNull()
