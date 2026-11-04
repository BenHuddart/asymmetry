"""Serialized PSI fixtures written from independently authored tagged counts."""

from pathlib import Path

import numpy as np
import pytest

from asymmetry.core.io import load

pytestmark = pytest.mark.io
uproot = pytest.importorskip("uproot")
FIXTURES = Path(__file__).parents[1] / "fixtures" / "psi_root"


def test_serialized_lem_header_calibration():
    path = FIXTURES / "legacy-lem.root"
    with uproot.open(path) as root_file:
        header = root_file["RunInfo"].member("fFolders")[0]
        assert (header.classname, header.class_version) == ("TLemRunHeader", 5)
        assert header.member("fTimeResolution") == 10.0
        np.testing.assert_array_equal(header.member("fTimeZero")[:2], [4, 7])
    dataset = load(str(path))
    run = dataset.run
    assert run.run_number == 4321
    assert run.metadata["temperature"] == 50.0
    assert run.metadata["field"] == 100.0
    assert run.metadata["title"] == "Synthetic format equivalence"
    assert run.metadata["root_format"] == "lem-root-v5"
    declaration = run.metadata["declared_units"]["time"]["RunInfo/Time Resolution"]
    assert declaration["unit"] == "ns"
    assert declaration["source"] == "format"
    assert run.metadata["root_histo_numbers"] == [0, 1]
    assert run.grouping["t0_source"] == "file"
    for index, histogram in enumerate(run.histograms):
        np.testing.assert_array_equal(
            histogram.counts[:128], (index * 2 + 1) * (100 + np.arange(128))
        )
        np.testing.assert_array_equal(histogram.counts[128:], [0])
        assert histogram.bin_width == pytest.approx(0.01)
        assert histogram.t0_bin == [4, 7][index]
        assert histogram.t0_time_us == pytest.approx(([4, 7][index] + 0.5) * 0.01)


@pytest.mark.parametrize(
    "filename",
    ["synthetic.bin", "legacy-lem.root", "musrroot-folder.root", "musrroot-directory.root"],
)
def test_cross_format_selected_observations(filename):
    import json
    from dataclasses import replace

    from asymmetry.core.simulate import reduce_run_to_dataset
    from asymmetry.core.transform.grouping import resolve_alignment, sum_aligned_counts

    run = load(str(FIXTURES / filename)).run
    for index, histogram in enumerate(run.histograms):
        np.testing.assert_array_equal(
            histogram.counts[:128], (2 * index + 1) * (100 + np.arange(128))
        )
        np.testing.assert_array_equal(
            histogram.counts[128:], [] if filename.endswith(".bin") else [0]
        )
    alignment = resolve_alignment(run.histograms, [0], [1])
    # Local bins 20..59 are forward raw 20..59 and backward raw 23..62.
    selected = slice(20, 60)
    raw = np.arange(20, 60)
    forward = sum_aligned_counts(run.histograms, [0], alignment)[selected]
    backward = sum_aligned_counts(run.histograms, [1], alignment)[selected]
    np.testing.assert_array_equal(forward, 100 + raw)
    np.testing.assert_array_equal(backward, 300 + 3 * (raw + 3))
    grouping = dict(
        run.grouping,
        groups={1: [1], 2: [2]},
        forward_group=1,
        backward_group=2,
        alpha=1.0,
        beta=1.0,
        t0_bin=7,
        # The shared reduction's inclusive window uses uncropped coordinates.
        first_good_bin=23,
        last_good_bin=62,
        deadtime_correction=False,
        background_correction=False,
    )
    reduced = reduce_run_to_dataset(replace(run, grouping=grouping))
    expected = (forward - backward) / (forward + backward)
    error = 2 * np.sqrt(forward * backward / (forward + backward) ** 3)
    np.testing.assert_allclose(reduced.asymmetry, 100 * expected, rtol=1e-14)
    np.testing.assert_allclose(reduced.error, 100 * error, rtol=1e-14)
    # BIN float32 width and legacy decimal ns differ by 2.24e-8 relative.
    np.testing.assert_allclose(reduced.time, (raw - 4) * 0.01, rtol=3e-8, atol=1e-16)
    reference = json.loads((FIXTURES / "reference.json").read_text())
    counts_export = np.asarray(reference["summed_counts"])
    fb_export = np.asarray(reference["asymmetry"])
    np.testing.assert_array_equal(counts_export[:, 1], forward + backward)
    np.testing.assert_allclose(counts_export[:, 2], np.sqrt(forward + backward), rtol=5e-6)
    np.testing.assert_allclose(reduced.time, counts_export[:, 0], rtol=3e-8)
    np.testing.assert_allclose(reduced.asymmetry / 100, fb_export[:, 1], rtol=5e-6)
    np.testing.assert_allclose(reduced.error / 100, fb_export[:, 2], rtol=5e-6)


def test_lem_numeric_members_own_calibration():
    from asymmetry.core.io.root_header import read_lem_header

    with uproot.open(FIXTURES / "legacy-lem.root") as root_file:
        record = root_file["RunInfo"].member("fFolders")[0]
        # Change only the decoded numeric payload; its display strings stay stale.
        record.members["fTimeResolution"] = 25.0
        record.members["fNHist"] = 3
        record.members["fTimeZero"] = np.array([0.0, 7.25, -1.0])
        header = read_lem_header(record)
    assert header["RunInfo/Time Resolution"] == "25.0"
    assert header["DetectorInfo/Detector000/Time Zero Bin"] == "0.0"
    assert header["DetectorInfo/Detector001/Time Zero Bin"] == "7.25"
    assert "DetectorInfo/Detector002/Time Zero Bin" not in header
    assert header["DetectorInfo/Detector002/Histo Number"] == "2"
