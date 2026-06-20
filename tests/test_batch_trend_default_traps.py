"""Two batch/trend default traps that silently produce wrong or empty results.

These come from a live new-user UX pass (Round-10 findings #7/#8):

(A) ORDERING TRAP — a custom data-browser column populated *after* a batch fit
    used to trend as all-NaN ("N/N skipped") because the batch results snapshot
    custom-column text at completion. The fix re-links the live per-run values
    into existing results (no batch re-run needed).

(B) A_1 = GLOBAL DEFAULT — the Batch tab classifies the leading amplitude as
    Global, which pins one shared value and flattens an amplitude trend. The
    trend panel now hints which Global (shared) params are held constant and how
    to trend them.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.gui.panels.data_browser import (
    CUSTOM_FIELDS_METADATA_KEY,
    DataBrowserPanel,
)
from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel, _FitRow


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def _row(run_number: int, values: dict[str, float], custom: dict[str, str]) -> _FitRow:
    return _FitRow(
        run_number=run_number,
        run_label=str(run_number),
        field=100.0 + run_number,
        temperature=10.0,
        values=dict(values),
        errors={k: 0.01 for k in values},
        custom_values=dict(custom),
    )


def _series_dict(run: int, values: dict[str, float], custom: dict[str, str]) -> dict:
    return {
        "run_number": run,
        "run_label": str(run),
        "field": 100.0,
        "temperature": 10.0,
        "values": dict(values),
        "errors": {k: 0.01 for k in values},
        "custom_values": dict(custom),
    }


def _dataset(run_number: int, custom: dict[str, str] | None = None) -> MuonDataset:
    t = np.linspace(0.0, 5.0, 20)
    meta: dict = {
        "run_number": run_number,
        "title": "sample",
        "temperature": 10.0,
        "field": 100.0,
    }
    if custom is not None:
        meta[CUSTOM_FIELDS_METADATA_KEY] = dict(custom)
    return MuonDataset(
        time=t,
        asymmetry=0.2 * np.exp(-0.2 * t),
        error=np.full_like(t, 0.01),
        metadata=meta,
    )


# --- Trap A: ordering trap — re-link custom columns into existing results -----


def test_relink_fills_stale_custom_snapshot(qapp):
    # Batch completed before the "Current (A)" column existed: rows have no value
    # for it, so the abscissa is NaN (the whole trend would be "skipped").
    panel = FitParametersPanel()
    panel._rows = [_row(1, {"A0": 0.2}, {}), _row(2, {"A0": 0.3}, {})]
    panel._varying_params = ["A0"]
    assert np.isnan(panel._x_value(panel._rows[0], "custom:cur"))

    # The user adds + populates the column; the host re-links live values.
    panel.relink_custom_values({1: {"custom:cur": "0.0"}, 2: {"custom:cur": "-0.5"}})

    assert panel._x_value(panel._rows[0], "custom:cur") == pytest.approx(0.0)
    assert panel._x_value(panel._rows[1], "custom:cur") == pytest.approx(-0.5)


def test_relink_wholesale_replace_clears_removed_value(qapp):
    # A value cleared in the browser must propagate (no longer in the live map),
    # not linger from the stale snapshot.
    panel = FitParametersPanel()
    panel._rows = [_row(1, {"A0": 0.2}, {"custom:cur": "3.0"})]
    panel._varying_params = ["A0"]

    panel.relink_custom_values({1: {}})  # browser now has no custom value for run 1

    assert panel._rows[0].custom_values == {}
    assert np.isnan(panel._x_value(panel._rows[0], "custom:cur"))


def test_relink_empty_map_is_noop(qapp):
    # No datasets loaded → don't wipe the snapshots we already have.
    panel = FitParametersPanel()
    panel._rows = [_row(1, {"A0": 0.2}, {"custom:cur": "1.5"})]
    panel._varying_params = ["A0"]

    panel.relink_custom_values({})

    assert panel._rows[0].custom_values == {"custom:cur": "1.5"}


def test_relink_reaches_inactive_series(qapp):
    # A column added later should re-link into *every* stored series, not only
    # the active one.
    panel = FitParametersPanel()
    panel.load_representation_series(
        [
            (
                "b1",
                "Series 1",
                [_series_dict(1, {"A0": 0.2}, {}), _series_dict(2, {"A0": 0.3}, {})],
            ),
            (
                "b2",
                "Series 2",
                [_series_dict(3, {"A0": 0.4}, {}), _series_dict(4, {"A0": 0.5}, {})],
            ),
        ],
        select_id="b1",
    )

    panel.relink_custom_values(
        {
            1: {"custom:cur": "0.0"},
            2: {"custom:cur": "-0.5"},
            3: {"custom:cur": "1.0"},
            4: {"custom:cur": "2.0"},
        }
    )

    inactive = panel._group_fit_results["b2"]
    by_run = {r.run_number: r.custom_values.get("custom:cur") for r in inactive.rows}
    assert by_run == {3: "1.0", 4: "2.0"}


def test_relink_keeps_active_group_snapshot_in_step(qapp):
    # After re-linking the active view, a group switch and back must not resurrect
    # the stale (pre-relink) snapshot.
    panel = FitParametersPanel()
    panel.load_representation_series(
        [("b1", "Series 1", [_series_dict(1, {"A0": 0.2}, {}), _series_dict(2, {"A0": 0.3}, {})])],
        select_id="b1",
    )
    panel.relink_custom_values({1: {"custom:cur": "0.0"}, 2: {"custom:cur": "-0.5"}})

    stored = panel._group_fit_results["b1"]
    by_run = {r.run_number: r.custom_values.get("custom:cur") for r in stored.rows}
    assert by_run == {1: "0.0", 2: "-0.5"}


def test_browser_custom_values_by_run_reads_live_metadata(qapp):
    panel = DataBrowserPanel()
    panel.add_dataset(_dataset(1))
    panel.add_dataset(_dataset(2))
    column = panel.add_custom_column("Current (A)")
    assert column is not None
    # Populate run 1 only; run 2 stays empty.
    panel._set_custom_column_value(panel._datasets[1], column.id, "0.0")

    values = panel.custom_values_by_run()
    assert values[1] == {column.id: "0.0"}
    assert values[2] == {}


def test_browser_custom_cell_edit_notifies_consumers(qapp):
    # Editing a custom cell must emit extra_columns_changed so the host re-links
    # the value into existing trend results (otherwise a value typed after the
    # batch is invisible until a structural column change or re-fit).
    panel = DataBrowserPanel()
    ds = _dataset(1)
    panel.add_dataset(ds)
    column = panel.add_custom_column("Current (A)")
    assert column is not None

    notified: list[int] = []
    panel.extra_columns_changed.connect(lambda: notified.append(1))

    visible = [c.id for c in panel._visible_extra_columns()]
    col_idx = len(panel._COLUMNS) + visible.index(column.id)
    panel._table.item(0, col_idx).setText("0.25")

    assert notified, "custom-column value edit did not emit extra_columns_changed"
    assert panel.custom_column_value(ds, column.id) == "0.25"


# --- Trap B: A_1 = Global default → flat trend, with a hint -------------------


def _load_batch_with_global(panel: FitParametersPanel, *, global_amplitude: bool) -> None:
    """Load a 3-run batch where ``f`` varies; A_1 is shared when global."""
    # A_1 identical across runs (shared/global); f varies (the real trend).
    rows = [
        _series_dict(1, {"A_1": 5.0, "f": 1.0}, {}),
        _series_dict(2, {"A_1": 5.0, "f": 1.5}, {}),
        _series_dict(3, {"A_1": 5.0, "f": 2.0}, {}),
    ]
    global_params_by_id = {"b1": {"A_1": {"value": 5.0}}} if global_amplitude else None
    panel.load_representation_series(
        [("b1", "Series 1", rows)],
        select_id="b1",
        global_params_by_id=global_params_by_id,
    )


def test_global_param_hint_shown_for_shared_amplitude(qapp):
    panel = FitParametersPanel()
    _load_batch_with_global(panel, global_amplitude=True)

    # isHidden() reflects the explicit hidden flag (the offscreen panel has no
    # shown ancestor, so isVisible() is always False here).
    assert not panel._global_param_hint.isHidden()
    text = panel._global_param_hint.text()
    assert "Global" in text
    assert "Local" in text  # points at the fix
    # The shared amplitude is named and is absent from the trendable Y list.
    assert "A" in text
    assert "A_1" not in panel._display_y_parameters()
    assert "f" in panel._display_y_parameters()


def test_no_global_param_hint_for_pure_batch(qapp):
    # All-local batch (no shared params): nothing is pinned, so no hint.
    panel = FitParametersPanel()
    _load_batch_with_global(panel, global_amplitude=False)

    assert panel._global_param_hint.isHidden()
    assert panel._global_param_hint.text() == ""


def test_shared_held_constant_excludes_varying_and_fixed(qapp):
    # A global param that (unusually) varies across rows is not flagged, and a
    # fixed param is never treated as a held-constant Global.
    panel = FitParametersPanel()
    panel._rows = [
        _row(1, {"A_1": 5.0, "lam": 0.1}, {}),
        _row(2, {"A_1": 5.0, "lam": 0.2}, {}),
    ]
    panel._varying_params = ["lam"]

    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    gp = ParameterSet()
    gp.add(Parameter(name="A_1", value=5.0))
    gp.add(Parameter(name="lam", value=0.15))  # in global set but varies → skip
    panel._global_params = gp

    assert panel._shared_held_constant_params() == ["A_1"]
