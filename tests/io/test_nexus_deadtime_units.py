"""Explicit v2 deadtime declarations become canonical coefficients at loading."""

import numpy as np
import pytest

from asymmetry.core.io import load, select_period
from asymmetry.core.project.schema import CURRENT_SCHEMA_VERSION, load_project, save_project
from asymmetry.core.transform.deadtime import prepare_histograms_with_deadtime
from tests.io.test_nexus_loader import _write_v1_file, _write_v2_file

pytestmark = pytest.mark.io
h5py = pytest.importorskip("h5py")


@pytest.mark.parametrize(
    "micro,nano",
    [
        ("us", "ns"),
        ("µs", "nanoseconds"),
        ("μs", "nanosecond"),
        ("Microseconds", "Nanoseconds"),
        (np.bytes_("microseconds"), np.bytes_("ns")),
        (np.bytes_(b"\xb5s"), np.bytes_("nanoseconds")),
        (np.array([b"us"]), np.array([b"ns"])),
    ],
)
@pytest.mark.parametrize("coefficient", [0.02, -0.02, 0.0])
def test_v2_equivalent_units_reach_the_same_correction(tmp_path, micro, nano, coefficient):
    corrected_counts = []
    for suffix, declaration, scale in [("us", micro, 1), ("ns", nano, 1000)]:
        path = tmp_path / f"{suffix}.nxs"
        _write_v2_file(path)
        with h5py.File(path, "r+") as handle:
            node = handle["raw_data_1/instrument/detector_1/dead_time"]
            node[...] = [coefficient * scale, 0.0]
            node.attrs["units"] = declaration
        run = load(path).run
        np.testing.assert_array_equal(run.grouping["dead_time_us"], [coefficient, 0.0])
        raw = [h.counts.copy() for h in run.histograms]
        grouping = {**run.grouping, "deadtime_mode": "file"}
        corrected, applied = prepare_histograms_with_deadtime(run.histograms, grouping, True)
        assert applied.status.value == "applied"
        for before, original, after, deadtime in zip(
            raw, run.histograms, corrected, [coefficient, 0.0], strict=True
        ):
            expected = before / (1 - before * deadtime / (0.02 * 200000))
            np.testing.assert_allclose(after.counts, expected, rtol=1e-14)
            np.testing.assert_array_equal(original.counts, before)
            np.testing.assert_array_equal(
                after.counts if after.count_variance is None else after.count_variance, before
            )
            assert after.bin_width == original.bin_width == 0.02
            assert after.t0_bin == original.t0_bin
            assert after.good_bin_start == original.good_bin_start
            assert after.good_bin_end == original.good_bin_end
        corrected_counts.append([h.counts for h in corrected])
    np.testing.assert_array_equal(*corrected_counts)


@pytest.mark.parametrize("location", ["dead_time", "deadtime", "deadtimes"])
@pytest.mark.parametrize("layout", ["shared", "table", "flat", "shared_3d", "table_3d"])
def test_v2_selected_dataset_owns_units_before_period_mapping(tmp_path, location, layout):
    path = tmp_path / "periods.nxs"
    _write_v2_file(path, multiperiod=True)
    table = np.array([[20.0, -10.0], [0.0, 30.0]])
    values = {
        "shared": table[0],
        "table": table,
        "flat": table.ravel(),
        "shared_3d": table[0].reshape(1, 1, 2),
        "table_3d": table.reshape(2, 1, 2),
    }[layout]
    with h5py.File(path, "r+") as handle:
        detector = handle["raw_data_1/instrument/detector_1"]
        del detector["dead_time"]
        # Earlier empty aliases must not donate their units to the selected table.
        for name in ("dead_time", "deadtime", "deadtimes"):
            node = detector.create_dataset(name, data=values if name == location else [])
            node.attrs["units"] = "ns" if name == location else "us"
        detector.attrs["units"] = "us"
        handle["raw_data_1/good_frames"][...] = [200000]
    combined = load(path)
    for index in range(2):
        selected = select_period(combined, index + 1)
        expected = table[0 if layout.startswith("shared") else index] * 0.001
        np.testing.assert_array_equal(selected.run.grouping["dead_time_us"], expected)
        corrected, applied = prepare_histograms_with_deadtime(
            selected.run.histograms, selected.run.grouping, True
        )
        assert applied.status.value == "applied"
        for original, after, deadtime in zip(
            selected.run.histograms, corrected, expected, strict=True
        ):
            np.testing.assert_allclose(
                after.counts,
                original.counts / (1 - original.counts * deadtime / (0.02 * 200000)),
                rtol=1e-14,
            )


@pytest.mark.parametrize("declaration", [None, "s", "ps", "unknown", np.array([b"ns", b"us"])])
def test_v2_missing_and_other_units_keep_stored_values(tmp_path, declaration):
    path = tmp_path / "unchanged.nxs"
    _write_v2_file(path)
    with h5py.File(path, "r+") as handle:
        node = handle["raw_data_1/instrument/detector_1/dead_time"]
        if declaration is not None:
            node.attrs["units"] = declaration
    assert load(path).run.grouping["dead_time_us"] == [0.01, 0.02]


@pytest.mark.parametrize("declaration", [None, "ns", "microseconds"])
def test_v1_keeps_fixed_microsecond_coefficients(tmp_path, declaration):
    path = tmp_path / "legacy.nxs"
    _write_v1_file(path)
    with h5py.File(path, "r+") as handle:
        node = handle["run/histogram_data_1/dead_time"]
        if declaration is not None:
            node.attrs["units"] = declaration
    assert load(path).run.grouping["dead_time_us"] == [0.015, 0.025]


def test_saved_canonical_coefficients_are_not_reconverted(tmp_path):
    path = tmp_path / "canonical.asymp"
    grouping = {"dead_time_us": [0.02, -0.01], "period_dead_time_us": [[0.02, -0.01], [0, 0.03]]}
    save_project(
        {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "datasets": [
                {"run_number": 1, "source_file": "unit-test.nxs", "grouping_overrides": grouping}
            ],
        },
        path,
    )
    assert load_project(path)["datasets"][0]["grouping_overrides"] == grouping


@pytest.mark.parametrize(
    "relative",
    [
        "Magnetism/Ferromagnetic nickel/Data/emu00124218.nxs",
        "Semiconductors/Photo-muSR in silicon/Data/HIFI00103277.nxs",
    ],
)
def test_acquired_isis_file_coefficients_preserve_counts_and_variance(relative):
    from tests.corpus import corpus_path

    path = corpus_path(relative)
    if path is None:
        pytest.skip("WiMDA corpus not configured (set ASYMMETRY_CORPUS_ROOT)")
    dataset = load(path)
    for period in range(1, dataset.run.metadata["period_count"] + 1):
        run = select_period(dataset, period).run
        raw_counts = [h.counts.copy() for h in run.histograms]
        corrected, outcome = prepare_histograms_with_deadtime(run.histograms, run.grouping, True)
        assert outcome.requested == "from_file" and outcome.status.value == "applied"
        for before, original, after, tau in zip(
            raw_counts, run.histograms, corrected, run.grouping["dead_time_us"], strict=True
        ):
            np.testing.assert_allclose(
                after.counts,
                before / (1 - before * tau / (original.bin_width * run.grouping["good_frames"])),
                rtol=1e-14,
            )
            np.testing.assert_array_equal(original.counts, before)
            np.testing.assert_array_equal(after.count_variance, before)
            np.testing.assert_array_equal(after.time_axis, original.time_axis)
