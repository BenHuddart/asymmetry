"""Batched-update behaviour of the Data Browser.

Adding N datasets one by one used to rebuild the table after every add —
O(n²) row construction, the dominant cost when loading many files or opening
a large project. ``batch_updates()`` defers the rebuild (and any deferred
sort / column auto-fit) to a single flush; these tests pin both the work
saved and the equivalence of the final table.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.gui.panels.data_browser import DataBrowserPanel

pytestmark = pytest.mark.usefixtures("qapp")


def _dataset(rn: int, *, temperature: float = 5.0) -> MuonDataset:
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
    hs = [
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
        histograms=hs,
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


def _count_row_builds(panel: DataBrowserPanel, monkeypatch) -> list[int]:
    """Spy on per-row construction; each full rebuild adds one call per run."""
    calls: list[int] = []
    original = panel._add_dataset_row

    def counting(dataset, **kwargs):
        calls.append(int(dataset.run_number))
        return original(dataset, **kwargs)

    monkeypatch.setattr(panel, "_add_dataset_row", counting)
    return calls


def test_batch_add_builds_each_row_once(monkeypatch):
    panel = DataBrowserPanel()
    calls = _count_row_builds(panel, monkeypatch)

    n = 20
    with panel.batch_updates():
        for rn in range(100, 100 + n):
            panel.add_dataset(_dataset(rn))
        # No rows materialised while the batch is open.
        assert panel._table.rowCount() == 0

    # One flush rebuild: each dataset's row built exactly once (not O(n²)).
    assert len(calls) == n
    assert panel._table.rowCount() == n


def test_unbatched_add_still_rebuilds_immediately():
    panel = DataBrowserPanel()
    panel.add_dataset(_dataset(100))
    assert panel._table.rowCount() == 1
    panel.add_dataset(_dataset(101))
    assert panel._table.rowCount() == 2


def test_batch_matches_unbatched_table():
    batched = DataBrowserPanel()
    plain = DataBrowserPanel()

    runs = [(105, 7.0), (101, 3.0), (103, 5.0)]
    with batched.batch_updates():
        for rn, temp in runs:
            batched.add_dataset(_dataset(rn, temperature=temp))
    for rn, temp in runs:
        plain.add_dataset(_dataset(rn, temperature=temp))

    assert batched._table.rowCount() == plain._table.rowCount()
    for row in range(plain._table.rowCount()):
        for col in range(plain._table.columnCount()):
            a = batched._table.item(row, col)
            b = plain._table.item(row, col)
            assert (a.text() if a else None) == (b.text() if b else None)


def test_batch_defers_active_sort_to_flush():
    panel = DataBrowserPanel()
    panel._current_sort_column = 0  # sort by run number

    with panel.batch_updates():
        for rn in (300, 100, 200):
            panel.add_dataset(_dataset(rn))

    assert panel._display_order == [100, 200, 300]
    assert [
        int(panel._table.item(row, 0).text()) for row in range(panel._table.rowCount())
    ] == [100, 200, 300]


def test_nested_batches_flush_once(monkeypatch):
    panel = DataBrowserPanel()
    calls = _count_row_builds(panel, monkeypatch)

    with panel.batch_updates():
        panel.add_dataset(_dataset(100))
        with panel.batch_updates():
            panel.add_dataset(_dataset(101))
        # Inner exit must not flush while the outer batch is open.
        assert panel._table.rowCount() == 0

    assert len(calls) == 2
    assert panel._table.rowCount() == 2


def test_batch_combined_dataset_single_flush(monkeypatch):
    panel = DataBrowserPanel()
    with panel.batch_updates():
        panel.add_dataset(_dataset(401))
        panel.add_dataset(_dataset(402))
        crn = panel.add_combined_dataset([401, 402], sign=1)
        assert crn is not None

    # Sources fold under the combined row exactly as in the unbatched path.
    assert panel._combined_datasets[crn] == [401, 402]
    assert panel._table.rowCount() == 1
