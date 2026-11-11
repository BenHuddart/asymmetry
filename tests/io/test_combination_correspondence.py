"""Declared period order and actual selected/mapped observation identity."""

import numpy as np
import pytest

from asymmetry.core.data.combine import CombineError, coadd_compatibility_error, combine_runs
from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.io.periods import combine_mapped_periods, period_run
from tests.core.test_combination_metadata import acquisition
from tests.core.test_linear_run_arithmetic import source


def periodic(number, labels="on;off", slots=(1, 2)):
    run = source(number, periodic=True)
    run.source_file = f"synthetic-{number}.nxs"
    run.metadata["period_declaration"] = {
        "types": [1 if i + 1 in slots else 2 for i in range(max(slots))],
        "data_dae_slots": list(slots),
        "labels": labels,
    }
    return run


def test_known_implicit_period_permutation_requires_explicit_selection():
    a, b = periodic(1), periodic(2, "off;on")
    with pytest.raises(CombineError, match="period.*select"):
        combine_runs([a, b])
    assert "period" in coadd_compatibility_error([a, b])
    combine_runs([period_run(a, 0), period_run(b, 1)])


def test_dwell_slots_and_unavailable_declaration_are_not_condition_conflicts():
    a = periodic(1)
    b = periodic(2, "dwell;on;dwell;off", (2, 4))
    combine_runs([a, b])
    b.metadata["period_declaration"]["data_dae_slots"] = []
    combine_runs([a, b])


def test_distinct_periods_independent_symlink_and_renumbered_reuse_rejected(tmp_path):
    a = periodic(1)
    path = tmp_path / "original.nxs"
    path.touch()
    a.source_file = str(path)
    b = periodic(2)
    link = tmp_path / "alias.nxs"
    link.symlink_to(path)
    b.source_file = str(link)
    combine_runs([period_run(a, 0), period_run(b, 1)])
    with pytest.raises(CombineError, match="independent"):
        combine_runs([period_run(a, 0), period_run(b, 0)])


def test_ignored_and_zero_mapped_originals_do_not_create_reuse():
    a, b, c = [acquisition(i) for i in range(3)]
    datasets = [
        MuonDataset(np.arange(12), np.zeros(12), np.ones(12), r.metadata, r) for r in (a, b, c)
    ]
    mapped = period_run(combine_mapped_periods(datasets, {1: "red", 2: "ignore", 3: "red"}).run, 0)
    combine_runs([mapped, b])
    with pytest.raises(CombineError, match="independent"):
        combine_runs([mapped, c])
