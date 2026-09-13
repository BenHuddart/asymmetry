"""The Single tab's parameter rail, its pop-out, the χ²ᵣ chip and the results card.

Phase 3 of the Fit tab refresh (``docs/plans/fit-tab-refresh.md``): the column
groups the rail chips drive, the live table's pop-out, the verdict chip that
opens a ``FitResultsWindow`` for the run, and the hand-offs that used to hide in
a ``More…`` menu.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication, QPushButton  # noqa: E402

from asymmetry.core.data.dataset import MuonDataset  # noqa: E402
from asymmetry.core.fitting.engine import FitResult  # noqa: E402
from asymmetry.core.fitting.parameters import Parameter, ParameterSet  # noqa: E402
from asymmetry.gui.panels.fit.single_tab import (  # noqa: E402
    ADD_TO_SERIES_ACTION,
    COLUMN_GROUPS_SETTINGS_KEY,
    DIAGNOSTIC_ACTION,
    RESTORED_TAG,
    SEND_TO_BATCH_ACTION,
    SingleFitTab,
)
from asymmetry.gui.styles import tokens  # noqa: E402
from asymmetry.gui.windows.fit_results_window import FitResultsWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture
def settings(tmp_path):
    # A per-test IniFormat file rather than a shared native scope, so parallel
    # xdist workers cannot race on the same on-disk key.
    store = QSettings(str(tmp_path / "single_fit.ini"), QSettings.Format.IniFormat)
    store.clear()
    yield store
    store.clear()


def _dataset(run_number: int = 3001) -> MuonDataset:
    time = np.linspace(0.1, 8.0, 64)
    return MuonDataset(
        time=time,
        asymmetry=0.2 * np.exp(-0.5 * time),
        error=np.full_like(time, 0.01),
        metadata={"run_number": run_number},
    )


def _fit(tab: SingleFitTab) -> None:
    """Run a fit against a stub engine that converges on the tab's own model."""
    names = list(tab._composite_model.param_names)

    def _engine_fit(ds, model_fn, parameters, *, minos=False, cancel_callback=None):
        return FitResult(
            success=True,
            chi_squared=60.0,
            reduced_chi_squared=1.0,
            dof=60,
            parameters=ParameterSet(
                [Parameter(name=name, value=float(i + 1)) for i, name in enumerate(names)]
            ),
            uncertainties=dict.fromkeys(names, 0.01),
        )

    from types import SimpleNamespace

    tab._fit_engine = SimpleNamespace(fit=_engine_fit)
    tab._run_fit()
    assert tab.wait_for_fit()


# ── the rail ────────────────────────────────────────────────────────────────


def test_rail_rests_on_bounds_only_and_persists_each_toggle(qapp, settings) -> None:
    tab = SingleFitTab(settings=settings)
    try:
        assert [chip.text() for chip in tab._column_chips.values()] == [
            "Bounds",
            "Links",
            "Batch",
        ]
        assert tab._param_table.column_group_visible("bounds") is True
        assert tab._param_table.column_group_visible("links") is False
        assert tab._param_table.column_group_visible("batch") is False

        tab._column_chips["links"].setChecked(True)
        tab._column_chips["bounds"].setChecked(False)
        assert tab._param_table.column_group_visible("links") is True
        assert tab._param_table.column_group_visible("bounds") is False
    finally:
        tab.close()
        tab.deleteLater()

    # A new tab on the same store opens on what the rail was left showing.
    reopened = SingleFitTab(settings=settings)
    try:
        assert reopened._column_chips["links"].isChecked() is True
        assert reopened._column_chips["bounds"].isChecked() is False
        assert reopened._param_table.column_group_visible("links") is True
        assert reopened._param_table.column_group_visible("bounds") is False
    finally:
        reopened.close()
        reopened.deleteLater()


def test_each_chip_is_its_own_settings_key(qapp, settings) -> None:
    """One boolean per group, so a rail state is readable and cannot be malformed."""
    tab = SingleFitTab(settings=settings)
    try:
        tab._column_chips["batch"].setChecked(True)
    finally:
        tab.close()
        tab.deleteLater()

    assert settings.value(f"{COLUMN_GROUPS_SETTINGS_KEY}/batch", False, type=bool) is True
    # An untouched chip leaves no key behind; its default stands in.
    assert settings.value(f"{COLUMN_GROUPS_SETTINGS_KEY}/links", None) is None

    reopened = SingleFitTab(settings=settings)
    try:
        assert reopened._param_table.column_group_visible("batch") is True
        assert reopened._param_table.column_group_visible("bounds") is True
        assert reopened._param_table.column_group_visible("links") is False
    finally:
        reopened.close()
        reopened.deleteLater()


# ── the pop-out ─────────────────────────────────────────────────────────────


def test_pop_out_takes_the_live_table_and_gives_it_back(qapp, settings) -> None:
    tab = SingleFitTab(settings=settings)
    try:
        tab.set_dataset(_dataset(3001))
        table = tab._param_table

        tab._show_param_table_dialog()

        assert tab._param_table_dialog.windowTitle() == "Fit parameters — 3001"
        # The *live* widget moves, so edits in the pop-out are edits to the tab.
        assert table.parent() is tab._param_table_dialog
        assert tab._popped_out_note.isVisibleTo(tab)
        assert all(table.column_group_visible(g) for g in ("bounds", "links", "batch"))

        tab._param_table_dialog.reject()

        assert tab._param_section_layout.indexOf(table) == 0
        assert not tab._popped_out_note.isVisibleTo(tab)
        # The chips are back in charge of what is shown.
        assert table.column_group_visible("links") is False
        assert table.column_group_visible("bounds") is True
    finally:
        tab.close()
        tab.deleteLater()


def test_copy_tsv_puts_the_whole_table_on_the_clipboard(qapp, settings) -> None:
    tab = SingleFitTab(settings=settings)
    try:
        tab._copy_param_table_tsv()
        assert QApplication.clipboard().text() == tab._param_table.as_tsv()
    finally:
        tab.close()
        tab.deleteLater()


# ── the χ²ᵣ chip ────────────────────────────────────────────────────────────


def test_chi2_chip_appears_on_a_recorded_fit_and_opens_the_results_window(qapp, settings) -> None:
    tab = SingleFitTab(settings=settings)
    try:
        tab.set_dataset(_dataset(3001))
        assert tab._chi2_chip.isHidden()

        _fit(tab)

        assert not tab._chi2_chip.isHidden()
        assert tab._chi2_chip.text() == "χ²ᵣ 1"
        tooltip = tab._chi2_chip.toolTip().splitlines()
        assert tooltip[0].startswith("good fit")
        assert any("=" in line for line in tooltip[1:])

        tab._show_fit_results_window()
        window = tab._fit_results_window
        assert isinstance(window, FitResultsWindow)
        assert window.windowTitle() == "Fit results — 3001"
        # A run fit has no model editor behind it.
        assert "Edit model fit…" not in {
            button.text() for button in window.findChildren(QPushButton)
        }

        # The window follows the next fit rather than a second one opening.
        _fit(tab)
        assert tab._fit_results_window is window

        # Binding another run drops the chip until that run is fitted.
        tab.set_dataset(_dataset(3002))
        assert tab._chi2_chip.isHidden()
        assert tab._last_fit_snapshot is None
    finally:
        tab.close()
        tab.deleteLater()


# ── the results card ────────────────────────────────────────────────────────


def test_card_hand_offs_gate_on_a_completed_fit(qapp, settings) -> None:
    tab = SingleFitTab(settings=settings)
    try:
        actions = tab._results_card._actions
        tab.set_dataset(_dataset(3001))

        # Send to Batch copies the *function*, so it never needs a fit.
        assert actions[SEND_TO_BATCH_ACTION].isEnabled()
        assert not actions[ADD_TO_SERIES_ACTION].isEnabled()
        assert not actions[DIAGNOSTIC_ACTION].isEnabled()

        _fit(tab)

        assert actions[ADD_TO_SERIES_ACTION].isEnabled()
        assert "Fit ✓" in tab._results_card.content_html()
        assert "χ²/ν" in tab._results_card.content_html()
        # The run carries no histograms, so the pull diagnostic stays out.
        assert not actions[DIAGNOSTIC_ACTION].isEnabled()
    finally:
        tab.close()
        tab.deleteLater()


def test_card_hand_offs_emit_the_tab_signals(qapp, settings) -> None:
    tab = SingleFitTab(settings=settings)
    try:
        seen: list[str] = []
        tab.send_model_to_batch_requested.connect(lambda: seen.append("batch"))
        tab.add_to_series_requested.connect(lambda: seen.append("series"))

        tab._results_card.action_triggered.emit(SEND_TO_BATCH_ACTION)
        tab._results_card.action_triggered.emit(ADD_TO_SERIES_ACTION)

        assert seen == ["batch", "series"]
    finally:
        tab.close()
        tab.deleteLater()


def test_a_restored_read_out_keeps_the_tag_it_was_saved_under(qapp, settings) -> None:
    tab = SingleFitTab(settings=settings)
    try:
        tab.set_dataset(_dataset(3001))
        _fit(tab)
        state = tab.get_state()
        assert state["result_tag"] == "Fit ✓"
    finally:
        tab.close()
        tab.deleteLater()

    restored = SingleFitTab(settings=settings)
    try:
        restored.restore_state(state)
        card = restored._results_card
        assert card.tag_text() == "Fit ✓"
        # The tone follows the tag, so a converged fit is not painted neutral.
        assert tokens.SUCCESS_BG in card._tag.styleSheet()
        assert "χ²/ν" in card.content_html()
    finally:
        restored.close()
        restored.deleteLater()


def test_a_state_saved_before_tags_reads_as_recorded_not_as_no_fit(qapp, settings) -> None:
    """A project written before `result_tag` still says a fit happened."""
    tab = SingleFitTab(settings=settings)
    try:
        tab.restore_state({"result_html": "<b>Batch fit</b><br>χ²ᵣ = 1.0500"})
        assert tab._results_card.tag_text() == RESTORED_TAG
        assert tokens.SURFACE_ALT in tab._results_card._tag.styleSheet()
        assert "Batch fit" in tab._results_card.content_html()
    finally:
        tab.close()
        tab.deleteLater()
