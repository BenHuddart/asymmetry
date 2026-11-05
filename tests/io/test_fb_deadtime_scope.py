"""Loaded acquisitions and CLI requests retain selected original observations."""

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from asymmetry.core.io import load
from asymmetry.core.project.profiles import (
    ProfileFingerprint,
    profile_from_payload,
    resolve_effective_grouping,
)
from asymmetry.core.transform.grouping import group_forward_backward
from asymmetry.core.transform.reduce import corrected_grouped_counts
from tests.corpus import corpus_path
from tests.io.test_file_window_timing import CORPUS
from tests.io.test_psi_root_examples import EXAMPLES

pytestmark = pytest.mark.io


@pytest.mark.parametrize("source", [*range(len(CORPUS)), *EXAMPLES])
def test_acquired_selected_correction_matches_original_raw_samples(source):
    if isinstance(source, int):
        path = corpus_path(CORPUS[source])
    else:
        locator = os.environ.get("ASYMMETRY_MUSRFIT_DATA")
        path = None if locator is None else Path(locator) / EXAMPLES[source][0]
    if path is None:
        pytest.skip("optional acquired inputs are not configured")
    dataset = load(path)
    if isinstance(dataset, list):
        dataset = dataset[0]
    run = dataset.run
    profile = profile_from_payload(
        run.grouping, "Selected pair", ProfileFingerprint("", len(run.histograms))
    )
    profile = replace(
        profile, groups={1: [1], 2: [2]}, forward_group=1, backward_group=2, projections=[]
    )
    grouping = resolve_effective_grouping(profile, run)
    raw = group_forward_backward(run.histograms, grouping).counts
    width = raw.bin_width_us
    frames = grouping["good_frames"]
    # Synthetic coefficients test transport, not a physical acquisition calibration.
    coefficient = 0.01 * width * frames / max(float(np.max(h.counts)) for h in run.histograms[:2])
    grouping = dict(grouping, deadtime_mode="file", deadtime_file_us=[coefficient, -coefficient])
    original = [hist.counts.copy() for hist in run.histograms]
    result = corrected_grouped_counts(
        histograms=run.histograms,
        grouping=grouping,
        forward_idx=[0],
        backward_idx=[1],
        use_deadtime=True,
        deadtime_mode="file",
        use_background=False,
    )
    assert result.corrections.deadtime.status == "applied"
    assert result.alignment == raw.alignment
    assert result.window == raw.window
    assert result.t0_time_us == raw.t0_time_us
    for index, group in enumerate((result.forward, result.backward)):
        offset = dict(result.alignment.detector_offsets)[index]
        bins = np.arange(result.window.first, result.window.last + 1) - offset
        observed = run.histograms[index].counts[bins]
        tau = coefficient if index == 0 else -coefficient
        np.testing.assert_allclose(
            group.counts[result.window.slice],
            observed / (1 - observed * tau / (width * frames)),
            rtol=1e-13,
        )
        variance = run.histograms[index].count_variance
        np.testing.assert_array_equal(
            group.variance[result.window.slice], observed if variance is None else variance[bins]
        )
    for histogram, counts in zip(run.histograms, original, strict=True):
        np.testing.assert_array_equal(histogram.counts, counts)


def test_cli_applies_file_correction_despite_excluded_saturated_prefix(tmp_path, capsys):
    import h5py

    from asymmetry import cli
    from asymmetry.core.workflow.workdir import WorkDir
    from tests.io.test_nexus_loader import _write_v2_file

    source = tmp_path / "input"
    source.mkdir()
    path = source / "EMU00012345.nxs"
    _write_v2_file(path, first_good_bin_attr=2, last_good_bin_attr=128)
    with h5py.File(path, "r+") as handle:
        detector = handle["raw_data_1/instrument/detector_1"]
        detector["dead_time"][:] = [1.0, -0.1]
        attributes = dict(detector["counts"].attrs)
        del detector["counts"]
        f = np.resize([120.0, 140.0, 160.0], 127)
        b = np.resize([95.0, 110.0, 130.0], 127)
        detector.create_dataset(
            "counts", data=np.array([np.r_[4000, f], np.r_[80, b]])
        ).attrs.update(attributes)
        del detector["raw_time"]
        detector["raw_time"] = np.arange(129) * 0.02
    workdir = tmp_path / "work"
    cli.main(
        [
            "reduce",
            str(source),
            "--runs",
            "12345",
            "--deadtime",
            "from_file",
            "--allow-aliasing",
            "--workdir",
            str(workdir),
            "--json",
        ]
    )
    capsys.readouterr()
    actual = WorkDir(workdir).reduced(12345)
    corrected_f, corrected_b = f / (1 - f / 4000), b / (1 + b * 0.1 / 4000)
    expected = 100 * (corrected_f - corrected_b) / (corrected_f + corrected_b)
    np.testing.assert_allclose(actual.asymmetry, expected, rtol=1e-13)
    assert all(
        record["deadtime"]["status"] == "applied"
        for record in actual.metadata["correction_outcomes"].values()
    )
    np.testing.assert_allclose(actual.time, (np.arange(1, 128) + 0.5) * 0.02, atol=1e-14)
