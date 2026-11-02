"""The CLI stores the same counts-first binning as the core API."""

import numpy as np

from asymmetry import cli
from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.io import load
from asymmetry.core.io.nexus_writer import write_nexus_v1
from asymmetry.core.transform.rebin import BinningSpec
from asymmetry.core.transform.reduce import reduce_grouped_asymmetry
from asymmetry.core.workflow.workdir import WorkDir


def test_cli_rebin_stores_the_count_level_result(tmp_path, capsys):
    source = tmp_path / "input"
    source.mkdir()
    run = Run(
        run_number=1,
        metadata={"run_number": 1, "instrument": "EMU"},
        histograms=[
            Histogram(np.tile(np.array(y, dtype=float), 64), 0.1, good_bin_end=127)
            for y in ([90, 1], [10, 9])
        ],
        grouping={
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
        },
    )
    path = source / "EMU00000001.nxs"
    write_nexus_v1(run, path)
    loaded = load(str(path))
    expected = reduce_grouped_asymmetry(
        histograms=loaded.run.histograms,
        grouping=loaded.run.grouping,
        binning=BinningSpec(merge=2),
        forward_idx=[0],
        backward_idx=[1],
        alpha=1.0,
        use_deadtime=False,
        deadtime_mode="off",
        use_background=False,
    )
    workdir = tmp_path / "work"
    cli.main(
        [
            "reduce",
            str(source),
            "--runs",
            "1",
            "--rebin",
            "2",
            "--allow-aliasing",
            "--workdir",
            str(workdir),
            "--json",
        ]
    )
    capsys.readouterr()
    actual = WorkDir(workdir).reduced(1)
    for key in ("time", "asymmetry", "error"):
        np.testing.assert_array_equal(getattr(actual, key), getattr(expected, key))
    np.testing.assert_allclose(actual.asymmetry, np.full(64, 100 * 72 / 110), rtol=1e-12)
