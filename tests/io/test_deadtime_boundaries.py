"""Signed file coefficients keep their acquisition normalisation through reduction."""

from copy import deepcopy

import numpy as np
import pytest

from asymmetry import cli
from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.io import load
from asymmetry.core.io.nexus_writer import write_nexus_v1
from asymmetry.core.io.periods import combine_mapped_periods, select_period_histograms
from asymmetry.core.transform import reduce_grouped_asymmetry
from asymmetry.core.transform.corrections import CorrectionUnavailableError
from asymmetry.core.transform.rebin import binning_spec

h5py = pytest.importorskip("h5py")


@pytest.fixture
def signed_file(tmp_path):
    path = tmp_path / "SIM00000001.nxs"
    run = Run(
        run_number=1,
        histograms=[
            Histogram(np.array(counts), bin_width=1, good_bin_end=1)
            for counts in ([5, 20], [10, 20])
        ],
        grouping={
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "first_good_bin": 0,
            "last_good_bin": 1,
            "dead_time_us": [-0.1, 0.1],
            "good_frames": 10,
        },
        metadata={"instrument": "SIM"},
    )
    write_nexus_v1(run, path)
    return path


@pytest.mark.parametrize("frames", [None, 0, -1, np.nan, np.inf])
def test_nexus_does_not_invent_missing_or_invalid_frames(signed_file, frames):
    with h5py.File(signed_file, "r+") as handle:
        del handle["run/good_frames"]
        if frames is not None:
            handle["run/good_frames"] = frames
    run = load(signed_file).run
    assert "good_frames" not in run.grouping
    with pytest.raises(CorrectionUnavailableError, match="good frames"):
        reduce_grouped_asymmetry(
            histograms=run.histograms,
            binning=binning_spec(run.grouping),
            grouping=run.grouping,
            forward_idx=[0],
            backward_idx=[1],
            alpha=1,
            use_deadtime=True,
            deadtime_mode="file",
            use_background=False,
        )


def test_negative_file_table_reaches_the_reduction(signed_file):
    run = load(signed_file).run
    assert run.grouping["dead_time_us"] == [-0.1, 0.1]
    result = reduce_grouped_asymmetry(
        histograms=run.histograms,
        binning=binning_spec(run.grouping),
        grouping=run.grouping,
        forward_idx=[0],
        backward_idx=[1],
        alpha=1,
        use_deadtime=True,
        deadtime_mode="file",
        use_background=False,
    )
    forward = np.array([100 / 21, 50 / 3])
    backward = np.array([100 / 9, 25])
    np.testing.assert_allclose(result.asymmetry, 100 * (forward - backward) / (forward + backward))
    assert result.corrections.deadtime.status == "applied"


def test_writer_preserves_absent_frames(signed_file, tmp_path):
    run = load(signed_file).run
    del run.grouping["good_frames"]
    path = tmp_path / "no_frames.nxs"
    write_nexus_v1(run, path)
    with h5py.File(path) as handle:
        assert "good_frames" not in handle["run"]
    assert "good_frames" not in load(path).run.grouping


@pytest.mark.parametrize("short_table", [False, True])
def test_period_selection_does_not_inherit_another_periods_frames(signed_file, short_table):
    red = load(signed_file)
    green = deepcopy(red)
    del green.run.grouping["good_frames"]
    mapped = combine_mapped_periods([red, green], {1: "red", 2: "green"})
    assert mapped.run.grouping["period_good_frames"] == [10, None]
    if short_table:
        mapped.run.grouping["period_good_frames"] = [10]
    histograms, grouping = select_period_histograms(mapped.run.histograms, mapped.run.grouping, 1)
    assert "good_frames" not in grouping
    with pytest.raises(CorrectionUnavailableError, match="good frames"):
        reduce_grouped_asymmetry(
            histograms=histograms,
            binning=binning_spec(grouping),
            grouping=grouping,
            forward_idx=[0],
            backward_idx=[1],
            alpha=1,
            use_deadtime=True,
            deadtime_mode="file",
            use_background=False,
        )


def test_short_nexus_frame_table_does_not_repeat_another_periods_value(signed_file):
    with h5py.File(signed_file, "r+") as handle:
        group = handle["run/histogram_data_1"]
        counts = group["counts"][...]
        attrs = dict(group["counts"].attrs)
        del group["counts"]
        dataset = group.create_dataset("counts", data=np.stack([counts] * 3))
        dataset.attrs.update(attrs)
        del handle["run/good_frames"]
        handle["run/good_frames"] = [10, 20]
    periods = load(signed_file)
    assert [period.run.grouping.get("good_frames") for period in periods] == [10, 20, None]


def test_cli_rejects_saturated_file_with_one_line(signed_file, capsys, tmp_path):
    with h5py.File(signed_file, "r+") as handle:
        handle["run/good_frames"][...] = 1
    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "reduce",
                str(signed_file.parent),
                "--runs",
                "1",
                "--deadtime",
                "from_file",
                "--workdir",
                str(tmp_path / "work"),
            ]
        )
    assert error.value.code == 1
    stderr = capsys.readouterr().err
    assert len(stderr.splitlines()) == 1
    assert "Deadtime correction rejected for detector 2" in stderr
