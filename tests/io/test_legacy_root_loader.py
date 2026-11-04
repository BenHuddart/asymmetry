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


@pytest.mark.parametrize(
    "anchors, usable",
    [
        ((0.0, 7.25), (True, True)),
        ((7.25, 0.0), (True, True)),
        ((-1.0, 7.25), (False, True)),
        ((np.inf, 7.25), (False, True)),
        ((-np.inf, 7.25), (False, True)),
        ((np.nan, 7.25), (False, True)),
        ((-1.0, -1.0), (False, False)),
    ],
)
def test_lem_numeric_calibration_through_production_loader(monkeypatch, anchors, usable):
    path = FIXTURES / "legacy-lem.root"
    with uproot.open(path) as root_file:
        record = root_file["RunInfo"].member("fFolders")[0]
        # Inject edge payloads into the decoded serialized record, preserving
        # its actual layout and stale display strings for production dispatch.
        record.members["fTimeResolution"] = 25.0
        record.members["fTimeZero"] = record.member("fTimeZero").copy()
        record.members["fTimeZero"][:2] = anchors
        monkeypatch.setattr(uproot, "open", lambda path: root_file)
        dataset = load(str(path))
    run = dataset.run
    assert run.metadata["root_format"] == "lem-root-v5"
    assert run.metadata["root_histo_numbers"] == [0, 1]
    assert run.metadata["declared_units"]["time"]["RunInfo/Time Resolution"] == {
        "unit": "ns",
        "source": "format",
        "reference": "TLemRunHeader v5; musrfit 1.12.1 ROOT-NPP",
    }
    assert run.grouping["t0_source"] == ("file" if any(usable) else "missing")
    for index, histogram in enumerate(run.histograms):
        np.testing.assert_array_equal(
            histogram.counts, np.append((2 * index + 1) * (100 + np.arange(128)), 0)
        )
        assert histogram.bin_width == pytest.approx(0.025)
        if usable[index]:
            assert histogram.t0_bin == round(anchors[index])
            assert histogram.t0_time_us == pytest.approx((anchors[index] + 0.5) * 0.025)
        else:
            assert histogram.t0_bin == 0
            assert histogram.t0_time_us is None
    assert np.all(np.isfinite(dataset.time))
    assert np.all(np.isfinite(dataset.asymmetry))
    assert np.all(np.isfinite(dataset.error))
