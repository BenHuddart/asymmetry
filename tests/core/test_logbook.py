"""Tests for logbook run-table management."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.data.logbook import Logbook


def _dataset(run: int, title: str, temperature: float, field: float) -> MuonDataset:
    n = 16
    return MuonDataset(
        time=np.linspace(0.0, 2.0, n),
        asymmetry=np.zeros(n),
        error=np.ones(n) * 0.01,
        metadata={
            "run_number": run,
            "title": title,
            "temperature": temperature,
            "field": field,
            "comment": f"comment-{run}",
            "source_file": f"run_{run}.nxs",
        },
    )


def test_logbook_add_get_remove_and_iter_order() -> None:
    log = Logbook()
    ds2 = _dataset(2, "second", 10.0, 100.0)
    ds1 = _dataset(1, "first", 5.0, 50.0)

    log.add(ds2, tags=["zf"])
    log.add(ds1, tags=["tf"])

    assert len(log) == 2
    assert log.run_numbers == [1, 2]
    assert [e.run_number for e in log] == [1, 2]
    assert log.get_dataset(2) is ds2

    log.remove(2)
    assert log.get_entry(2) is None
    assert log.get_dataset(2) is None


def test_logbook_filter_search_and_collections() -> None:
    log = Logbook()
    log.add(_dataset(1, "Alpha run", 5.0, 100.0), tags=["cool", "zf"])
    log.add(_dataset(2, "Beta run", 10.0, 50.0), tags=["warm", "lf"])

    filtered = log.filter(temperature=10.0)
    assert [e.run_number for e in filtered] == [2]

    searched = log.search("alpha")
    assert [e.run_number for e in searched] == [1]

    searched_tag = log.search("lf")
    assert [e.run_number for e in searched_tag] == [2]

    log.create_collection("selected", [2, 1])
    assert log.collections == ["selected"]
    assert log.get_collection("selected") == [2, 1]


def test_logbook_save_and_load_metadata(tmp_path: Path) -> None:
    log = Logbook()
    log.add(_dataset(7, "Persisted", 42.0, 123.0), tags=["persist"])
    log.create_collection("groupA", [7])

    out = tmp_path / "logbook.json"
    log.save(out)
    assert out.exists()

    restored = Logbook()
    restored.load_metadata(out)

    entry = restored.get_entry(7)
    assert entry is not None
    assert entry.title == "Persisted"
    assert restored.get_collection("groupA") == [7]
