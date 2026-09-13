"""The Batch tab's parameter rail, results card and outcome chip.

Phase 4 of the Fit tab refresh (``docs/plans/fit-tab-refresh.md``): the ``Bounds``
chip over the split Min/Max columns, the pop-out, the per-member verdict chips and
the windows they open, and the ``seeded from <run>`` tag.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication, QPushButton  # noqa: E402

from asymmetry.core.data.dataset import MuonDataset  # noqa: E402
from asymmetry.core.fitting.engine import FitResult  # noqa: E402
from asymmetry.core.fitting.member_quality import member_quality_flags  # noqa: E402
from asymmetry.core.fitting.parameters import Parameter, ParameterSet  # noqa: E402
from asymmetry.gui.panels.fit.global_tab import (  # noqa: E402
    _COL_MAX,
    _COL_MIN,
    _MEMBERS_LIST_MAX_ROWS,
    COLUMN_GROUPS_SETTINGS_KEY,
    GlobalFitTab,
)
from asymmetry.gui.panels.fit.tab_base import VALUE_COL_CHARS  # noqa: E402
from asymmetry.gui.styles.metrics import char_width, row_height  # noqa: E402
from asymmetry.gui.windows.fit_results_window import FitResultsWindow  # noqa: E402

#: The inspector dock the Fit tabs are designed against, in characters (~320 px).
_DOCK_CHARS = 46


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture
def settings(tmp_path):
    # A per-test IniFormat file rather than a shared native scope, so parallel
    # xdist workers cannot race on the same on-disk key.
    store = QSettings(str(tmp_path / "batch_fit.ini"), QSettings.Format.IniFormat)
    store.clear()
    yield store
    store.clear()


def _dataset(run_number: int) -> MuonDataset:
    time = np.linspace(0.1, 8.0, 32)
    return MuonDataset(
        time=time,
        asymmetry=0.2 * np.exp(-0.5 * time),
        error=np.full_like(time, 0.01),
        metadata={"run_number": run_number, "temperature": float(run_number)},
    )


def _result(chi2: float, *, success: bool = True, error: float = 0.01) -> FitResult:
    return FitResult(
        success=success,
        chi_squared=chi2 * 30.0,
        reduced_chi_squared=chi2,
        dof=30,
        parameters=ParameterSet([Parameter(name="Lambda", value=0.5)]),
        uncertainties={"Lambda": error},
    )


# ── the rail ────────────────────────────────────────────────────────────────


def test_bounds_chip_rests_off_and_persists(qapp, settings) -> None:
    tab = GlobalFitTab(member_kind="runs", settings=settings)
    try:
        assert [chip.text() for chip in tab._column_chips.values()] == ["Bounds"]
        assert tab._param_table.isColumnHidden(_COL_MIN)
        assert tab._param_table.isColumnHidden(_COL_MAX)

        tab._column_chips["bounds"].setChecked(True)
        assert not tab._param_table.isColumnHidden(_COL_MIN)
        assert not tab._param_table.isColumnHidden(_COL_MAX)
    finally:
        tab.close()
        tab.deleteLater()

    reopened = GlobalFitTab(member_kind="runs", settings=settings)
    try:
        assert reopened._column_chips["bounds"].isChecked() is True
        assert not reopened._param_table.isColumnHidden(_COL_MIN)
        assert settings.value(f"{COLUMN_GROUPS_SETTINGS_KEY}/bounds", False, type=bool) is True
    finally:
        reopened.close()
        reopened.deleteLater()


def test_the_chip_drives_every_table_on_the_surface(qapp, settings) -> None:
    """One chip, three tables — classification, physics and per-group nuisances."""
    tab = GlobalFitTab(member_kind="runs", settings=settings)
    try:
        tab._column_chips["bounds"].setChecked(True)
        minimum = tab._group_param_min_column()
        assert not tab._group_model_table.isColumnHidden(_COL_MIN)
        assert not tab._group_param_table.isColumnHidden(minimum)
        assert not tab._group_param_table.isColumnHidden(minimum + 1)

        tab._column_chips["bounds"].setChecked(False)
        assert tab._group_model_table.isColumnHidden(_COL_MAX)
        assert tab._group_param_table.isColumnHidden(minimum)
    finally:
        tab.close()
        tab.deleteLater()


def test_pop_out_takes_the_live_table_and_gives_it_back(qapp, settings) -> None:
    tab = GlobalFitTab(member_kind="runs", settings=settings)
    try:
        tab.set_datasets([_dataset(3001), _dataset(3002)])
        table = tab._param_table

        tab._show_param_table_dialog()

        assert tab._param_table_dialog.windowTitle() == "Fit parameters — 2 runs"
        assert table.parent() is tab._param_table_dialog
        assert not table.isColumnHidden(_COL_MIN)

        tab._param_table_dialog.reject()

        assert tab._rail_section_layout.indexOf(table) == 0
        # The chip is back in charge of what is shown.
        assert table.isColumnHidden(_COL_MIN)
    finally:
        tab.close()
        tab.deleteLater()


def test_two_value_columns_share_the_leftover_width(qapp, settings, monkeypatch) -> None:
    """One value column per detector group: they split the spare width evenly."""
    tab = GlobalFitTab(member_kind="groups", grouped_single=True, settings=settings)
    try:
        groups = [
            SimpleNamespace(group_id=1, group_name="Forward", counts=np.array([120.0, 118.0])),
            SimpleNamespace(group_id=2, group_name="Backward", counts=np.array([80.0, 79.0])),
        ]
        monkeypatch.setattr(tab, "_grouped_mode_context", lambda: (groups, [], "ready"))
        tab.resize(char_width(70), 1000)
        tab.show()
        tab.set_current_dataset(_dataset(3001))
        qapp.processEvents()

        table = tab._group_param_table
        forward, backward = table.columnWidth(1), table.columnWidth(2)
        # Equal but for the odd pixel a two-way split cannot halve.
        assert abs(forward - backward) <= 1
        assert forward > char_width(VALUE_COL_CHARS)
        assert (
            sum(
                table.columnWidth(column)
                for column in range(table.columnCount())
                if not table.isColumnHidden(column)
            )
            == table.viewport().width()
        )
    finally:
        tab.close()
        tab.deleteLater()


def test_the_grouped_surface_rails_its_physics_table(qapp, settings) -> None:
    """A grouped surface hides the classification section, so the rail rides the
    Fit-Function Parameters section it actually shows."""
    tab = GlobalFitTab(member_kind="groups", grouped_single=True, settings=settings)
    try:
        assert tab._rail_table is tab._group_model_table
        assert tab._rail_section_layout is tab._group_model_group.body_layout
        # Its physics table ticks Fix — there are no roles for a ⓘ to explain.
        assert tab._role_help_btn is None
    finally:
        tab.close()
        tab.deleteLater()


# ── the results card ────────────────────────────────────────────────────────


def test_a_converged_batch_renders_one_chip_per_member(qapp, settings) -> None:
    tab = GlobalFitTab(member_kind="runs", settings=settings)
    try:
        tab._render_fit_summary(
            {3001: _result(0.98), 3002: _result(1.9), 3003: _result(1.0, success=False)},
            tag_prefix="Batch",
            detail_html="avg χ²/ν = 1.3",
        )
        card = tab._results_card

        assert card.tag_text() == "Batch ⚠"
        assert "2 of 3 converged" in card.content_html()
        chips = [button.text() for button in card._members.findChildren(QPushButton)]
        assert any(text.startswith("3001 ✓") for text in chips)
        assert any(text.startswith("3003 ⚠") for text in chips)
        assert tab._outcome_chip.isVisibleTo(tab)
        # Counts only on the chip; the sentence they stand for is on the tooltip.
        assert tab._outcome_chip.text() == "2 ✓ 1 ⚠"
        assert tab._outcome_chip.toolTip() == "3 runs: 2 converged without flags, 1 flagged"
    finally:
        tab.close()
        tab.deleteLater()


def test_a_flagged_but_converged_member_still_counts_as_converged(qapp, settings) -> None:
    """Convergence and quality are separate: the headline counts one, the chips
    the other."""
    tab = GlobalFitTab(member_kind="runs", settings=settings)
    try:
        # A 500 % relative error converges but earns an advisory flag.
        flagged = _result(1.05, error=2.5)
        assert member_quality_flags(flagged)
        tab._render_fit_summary(
            {3001: _result(0.98), 3002: flagged}, tag_prefix="Batch", detail_html=""
        )
        card = tab._results_card

        assert "2 of 2 converged" in card.content_html()
        chips = [button.text() for button in card._members.findChildren(QPushButton)]
        assert [text.split()[1] for text in chips] == ["✓", "⚠"]
        assert card.tag_text() == "Batch ⚠"
        assert tab._outcome_chip.text() == "1 ✓ 1 ⚠"
        assert tab._outcome_chip.toolTip() == "2 runs: 1 converged without flags, 1 flagged"
    finally:
        tab.close()
        tab.deleteLater()


def test_an_all_good_batch_reads_as_ok(qapp, settings) -> None:
    tab = GlobalFitTab(member_kind="runs", settings=settings)
    try:
        tab._render_fit_summary(
            {3001: _result(0.98), 3002: _result(1.02)},
            tag_prefix="Batch",
            detail_html="",
        )
        assert tab._results_card.tag_text() == "Batch ✓"
        assert tab._outcome_chip.text() == "2 ✓"
        assert tab._outcome_chip.toolTip() == "2 runs: 2 converged without flags"
    finally:
        tab.close()
        tab.deleteLater()


def test_a_member_chip_opens_that_run_and_the_next_batch_refreshes_it(qapp, settings) -> None:
    tab = GlobalFitTab(member_kind="runs", settings=settings)
    try:
        tab._render_fit_summary(
            {3001: _result(0.98), 3002: _result(1.9)}, tag_prefix="Batch", detail_html=""
        )
        tab._show_member_results_window(3001)
        window = tab._member_windows[3001]
        assert isinstance(window, FitResultsWindow)
        assert window.windowTitle() == "Fit results — 3001"
        # A run fit has no editor behind it.
        assert not [
            button
            for button in window.findChildren(QPushButton)
            if button.text() == "Edit model fit…"
        ]

        tab._render_fit_summary({3001: _result(1.2)}, tag_prefix="Batch", detail_html="")
        assert tab._member_windows[3001] is window
        assert window._results.ranges[0].chi_squared == "χ²ᵣ 1.2"
        # The run that left the batch loses its window rather than showing a stale one.
        tab._show_member_results_window(3001)
    finally:
        tab.close()
        tab.deleteLater()


def test_send_to_batch_tags_the_card_with_the_source_run(qapp, settings) -> None:
    tab = GlobalFitTab(member_kind="runs", settings=settings)
    try:
        tab.show_seeded_from(3001)
        assert tab._results_card._meta_tag.text() == "seeded from 3001"
        assert "3001" in tab._results_card._meta_tag.toolTip()
    finally:
        tab.close()
        tab.deleteLater()


# ── the members list ────────────────────────────────────────────────────────


def test_the_members_list_is_as_tall_as_its_runs(qapp, settings) -> None:
    """Two runs, two rows — not a QListWidget's square default box."""
    tab = GlobalFitTab(member_kind="runs", settings=settings)
    try:
        tab.set_datasets([_dataset(3001 + i) for i in range(2)])
        members = tab._members_list

        assert members.count() == 2
        assert members.height() == 2 * row_height() + 2 * members.frameWidth()
        assert members.verticalScrollBar().maximum() == 0
    finally:
        tab.close()
        tab.deleteLater()


def test_a_long_batch_caps_the_members_list_and_scrolls(qapp, settings) -> None:
    """Past three runs the list stops growing and scrolls inside itself."""
    tab = GlobalFitTab(member_kind="runs", settings=settings)
    try:
        tab.resize(char_width(_DOCK_CHARS), 1200)
        tab.show()
        tab.set_datasets([_dataset(3001 + i) for i in range(4)])
        qapp.processEvents()
        members = tab._members_list

        assert _MEMBERS_LIST_MAX_ROWS == 3
        assert members.count() == 4
        assert members.height() == _MEMBERS_LIST_MAX_ROWS * row_height() + 2 * members.frameWidth()
        assert members.verticalScrollBar().maximum() > 0
    finally:
        tab.close()
        tab.deleteLater()


# ── the run row ─────────────────────────────────────────────────────────────


def test_the_grouped_run_row_wraps_instead_of_widening_the_dock(qapp, settings) -> None:
    """After a grouped fit the outcome chip wraps under the buttons.

    ``Run grouped fit`` · ``Preview`` · the chip on one line came to more than
    the ~300 px a 13-inch inspector dock has, so the whole tab demanded a
    horizontal scrollbar once a fit had run.
    """
    tab = GlobalFitTab(member_kind="groups", grouped_single=True, settings=settings)
    try:
        tab.resize(char_width(_DOCK_CHARS), 1200)
        tab.show()
        qapp.processEvents()

        tab._render_fit_summary(
            {0: _result(0.98), 1: _result(1.9), 2: _result(1.0, success=False), 3: _result(1.1)},
            tag_prefix="Batch",
            detail_html="",
        )
        qapp.processEvents()

        assert tab._outcome_chip.isVisible()
        assert tab._outcome_chip.toolTip().startswith("4 groups: ")
        assert tab._rail_table.horizontalScrollBar().maximum() == 0
        assert tab.minimumSizeHint().width() <= char_width(_DOCK_CHARS)
    finally:
        tab.close()
        tab.deleteLater()
