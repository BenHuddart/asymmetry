"""Optional acquisition calibration never invents detector coefficients."""

import numpy as np
import pytest

from asymmetry.core.io import load, select_period
from asymmetry.core.transform.reduce import corrected_grouped_counts
from tests.io.test_nexus_loader import _write_v2_file

pytestmark = pytest.mark.io
h5py = pytest.importorskip("h5py")


@pytest.mark.parametrize("table", [None, [0.0, 0.0], [0.02], [0.02, -0.01]])
def test_optional_table_through_loading_and_final_counts(tmp_path, table):
    path = tmp_path / "optional.nxs"
    _write_v2_file(path)
    with h5py.File(path, "r+") as handle:
        detector = handle["raw_data_1/instrument/detector_1"]
        del detector["dead_time"]
        if table is not None:
            detector["dead_time"] = table
    run = load(path).run
    assert run.grouping["dead_time_us"] == ([] if table is None else table)
    options = dict(
        histograms=run.histograms,
        grouping=run.grouping,
        forward_idx=[0],
        backward_idx=[1],
        deadtime_mode="file",
        use_background=False,
    )
    if table is None or len(table) < 2:
        with pytest.warns(UserWarning, match="File deadtime correction not applied"):
            actual = corrected_grouped_counts(**options, use_deadtime=True)
        expected = corrected_grouped_counts(**options, use_deadtime=False)
        assert actual.corrections.complete
        assert actual.corrections.deadtime.status == "skipped"
        assert actual.corrections.deadtime.requested == "from_file"
        assert actual.corrections.deadtime.effective is None
        for side in ("forward", "backward"):
            np.testing.assert_array_equal(
                getattr(actual, side).counts, getattr(expected, side).counts
            )
            np.testing.assert_array_equal(
                getattr(actual, side).variance, getattr(expected, side).variance
            )
    else:
        actual = corrected_grouped_counts(**options, use_deadtime=True)
        assert actual.corrections.deadtime.status == "applied"
        np.testing.assert_array_equal(actual.corrections.deadtime.effective["tau_us"], table)


def test_short_period_tables_preserve_correspondence(tmp_path):
    path = tmp_path / "periods.nxs"
    _write_v2_file(path, multiperiod=True)
    with h5py.File(path, "r+") as handle:
        detector = handle["raw_data_1/instrument/detector_1"]
        del detector["dead_time"]
        detector["dead_time"] = [[0.02], [0.03]]
    dataset = load(path)
    assert select_period(dataset, 1).run.grouping["dead_time_us"] == [0.02]
    assert select_period(dataset, 2).run.grouping["dead_time_us"] == [0.03]


@pytest.mark.parametrize(
    "relative",
    [
        "Magnetism/Magnetic ordering in EuO/data/deltat_pta_gps_2941.bin",
        "Magnetism/AFM transition in high TF/data/tdc_hifi_2020_00730.mdu",
    ],
)
def test_acquired_psi_without_file_table_reduces_uncorrected(relative):
    from tests.corpus import corpus_path

    path = corpus_path(relative)
    if path is None:
        pytest.skip("WiMDA corpus not configured")
    run = load(path).run
    if path.suffix == ".mdu":
        from asymmetry.core.transform.deadtime import prepare_histograms_with_deadtime
        from asymmetry.core.workflow.reduction import ReductionSettings, reduce_run

        # The acquired default grouping requests an absent channel; this remains fatal.
        with pytest.warns(UserWarning, match="File deadtime correction not applied"):
            prepared, outcome = prepare_histograms_with_deadtime(run.histograms, run.grouping, True)
        assert outcome.status == "skipped"
        for original, actual in zip(run.histograms, prepared, strict=True):
            np.testing.assert_array_equal(actual.counts, original.counts)
            np.testing.assert_array_equal(actual.time_axis, original.time_axis)
        with pytest.raises(ValueError, match="do not exist"):
            reduce_run(run, ReductionSettings(deadtime="from_file"))
        assert "correction_error" in run.metadata
    else:
        _assert_acquired_file_fallback(run)


@pytest.mark.parametrize("name", ["lem15_his_2994.root", "lem12_his_2466.root", "07770.root"])
def test_acquired_root_without_file_table_reduces_uncorrected(name):
    from tests.io.test_root_loader import _musrfit_example_file

    _assert_acquired_file_fallback(load(_musrfit_example_file(name)).run)


def _assert_acquired_file_fallback(run):
    """Compare the same acquired observations under Off and unresolved File."""
    from asymmetry.core.workflow.reduction import ReductionSettings, reduce_run

    before = [hist.counts.copy() for hist in run.histograms]
    expected = reduce_run(run, ReductionSettings())
    with pytest.warns(UserWarning, match="File deadtime correction not applied"):
        actual = reduce_run(run, ReductionSettings(deadtime="from_file"))
    for field in ("time", "asymmetry", "error"):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
    assert "correction_error" not in actual.metadata
    assert all(
        record["deadtime"]["status"] == "skipped"
        for record in actual.metadata["correction_outcomes"].values()
    )
    for hist, counts in zip(run.histograms, before, strict=True):
        np.testing.assert_array_equal(hist.counts, counts)
