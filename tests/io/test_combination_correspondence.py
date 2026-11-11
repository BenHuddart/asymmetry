"""Declared period order and actual selected/mapped observation identity."""

from dataclasses import replace

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


def test_nested_implicit_pairing_retains_original_declarations():
    a, b, c = periodic(1), periodic(2), periodic(3, "off;on")
    nested = combine_runs([a, b])
    with pytest.raises(CombineError, match="period"):
        combine_runs([nested, c])
    combine_runs([period_run(nested, 0), period_run(c, 1)])


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


def test_mapped_active_ranges_include_each_selected_original():
    runs = [acquisition(i, temperature) for i, temperature in enumerate((10, 200, 1000, 40))]
    datasets = [
        MuonDataset(np.arange(12), np.zeros(12), np.ones(12), r.metadata, r) for r in runs[:3]
    ]
    mapped = combine_mapped_periods(datasets, {1: "red", 2: "red", 3: "ignore"})
    result = combine_runs([period_run(mapped.run, 0), runs[3]])
    assert result.metadata["temperature_spread"] == (10, 200)
    ignored = next(
        record
        for record in result.metadata["combination"]["constituents"]
        if record["temperature"] == 1000
    )
    assert ignored["coefficient"] == 0


def test_selected_nested_ranges_exclude_unselected_green_conditions():
    sources = [acquisition(i, temperature) for i, temperature in enumerate((10, 200, 40, 400, 50))]
    mapped = []
    for pair in (sources[:2], sources[2:4]):
        datasets = [
            MuonDataset(np.arange(12), np.zeros(12), np.ones(12), r.metadata, r) for r in pair
        ]
        mapped.append(combine_mapped_periods(datasets, {1: "red", 2: "green"}).run)
    full = combine_runs(mapped)
    red = period_run(full, 0)
    result = combine_runs([red, sources[4]])
    assert result.metadata["temperature_spread"] == (10, 50)


def mapping_dataset(run):
    return MuonDataset(np.arange(12), np.zeros(12), np.ones(12), dict(run.metadata), run)


def test_selected_nested_mapping_list_requires_original_acquisitions():
    nested = combine_runs([acquisition(1), acquisition(2)])
    with pytest.raises(ValueError, match="original acquired"):
        combine_mapped_periods(
            [mapping_dataset(nested), mapping_dataset(acquisition(3))], {1: "red", 2: "green"}
        )


def test_direct_mapping_preserves_unit_support_and_inactive_coefficients():
    a, b = acquisition(1), acquisition(2)
    unit = combine_mapped_periods(mapping_dataset(combine_runs([a, b])), {1: "red", 2: "green"})
    assert len(unit.run.grouping["period_inputs"]) == 2
    assert "implicit_period_inputs" not in unit.run.grouping
    weighted = mapping_dataset(combine_runs([a, b], scales=[1, 0.5]))
    with pytest.raises(ValueError, match="coefficient"):
        combine_mapped_periods(weighted, {1: "red", 2: "green"})
    selected = combine_mapped_periods(weighted, {1: "red", 2: "ignore"})
    np.testing.assert_array_equal(selected.run.histograms[0].counts, a.histograms[0].counts)
    with pytest.raises(ValueError, match="coefficient"):
        combine_mapped_periods(selected, {1: "red", 2: "green"})


def test_ignored_nested_source_keeps_compact_identity_and_later_diagnosis():
    from asymmetry.core.transform.preparation import histogram_observation

    a, b = acquisition(1), acquisition(2)
    nested = combine_runs([a, b], scales=[1, 0.5])
    result = combine_mapped_periods(
        [mapping_dataset(a), mapping_dataset(nested)], {1: "red", 2: "ignore"}
    )
    snapshot = histogram_observation(result.run)
    original = snapshot.source.grouping["period_inputs"][1]
    assert all(not source.histograms for source in original.grouping["period_inputs"])
    np.testing.assert_array_equal(result.run.histograms[0].counts, a.histograms[0].counts)
    with pytest.raises(ValueError, match="original acquired"):
        combine_mapped_periods(result, {1: "red", 2: "green"})
    requested = replace(
        result.run, grouping=dict(result.run.grouping, period_mapping={"1": "red", "2": "red"})
    )
    from asymmetry.core.data.combine import reduce_combined_run

    assert "acquired" in coadd_compatibility_error([requested, acquisition(3)])
    with pytest.raises(CombineError, match="acquired"):
        combine_runs([requested, acquisition(3)])
    with pytest.raises(ValueError, match="original acquired"):
        reduce_combined_run(requested)
