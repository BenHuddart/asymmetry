"""The CLI stores the same counts-first binning as the core API."""

import json
from dataclasses import replace

import numpy as np
import pytest

from asymmetry import cli
from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.io import load
from asymmetry.core.io.nexus_writer import write_nexus_v1
from asymmetry.core.transform.reduce import reduce_grouped_asymmetry
from asymmetry.core.workflow.workdir import WorkDir


@pytest.mark.parametrize("mode", ["fixed", "variable", "constant_error"])
@pytest.mark.parametrize("merge", [1, 2, 3])
def test_cli_rebin_stores_the_count_level_result(tmp_path, capsys, monkeypatch, mode, merge):
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
    loaded.run.grouping.update(binning_mode=mode, bunching_factor=2, bin0_us=0.12, bin10_us=0.25)
    loaded.run.metadata["field"] = 0.0
    from asymmetry.core.transform.rebin import binning_spec
    from asymmetry.core.workflow import reduction, survey

    monkeypatch.setattr(reduction, "load_reduction_source", lambda *args: loaded)

    def unsolicited_spectrum(*args, **kwargs):
        pytest.fail("CLI reduction requested spectral analysis")

    monkeypatch.setattr(survey, "fingerprint_spectrum", unsolicited_spectrum)
    expected = reduce_grouped_asymmetry(
        histograms=loaded.run.histograms,
        grouping=loaded.run.grouping,
        binning=replace(binning_spec(loaded.run.grouping), merge=merge),
        forward_idx=[0],
        backward_idx=[1],
        alpha=1.0,
        use_deadtime=False,
        deadtime_mode="off",
        use_background=False,
    )
    workdir = tmp_path / "work"
    args = [
        "reduce",
        str(source),
        "--runs",
        "1",
        "--rebin",
        str(merge),
        "--workdir",
        str(workdir),
        "--json",
    ]
    cli.main(args)
    first = json.loads(capsys.readouterr().out)
    actual = WorkDir(workdir).reduced(1)
    for key in ("time", "asymmetry", "error"):
        np.testing.assert_array_equal(getattr(actual, key), getattr(expected, key))
    assert first["entries"][0]["recomputed"] is True
    cli.main(args)
    replay = json.loads(capsys.readouterr().out)
    assert replay["entries"][0]["recomputed"] is False
    assert WorkDir(workdir).entry(1).settings.rebin == merge
    for key in ("time", "asymmetry", "error"):
        np.testing.assert_array_equal(
            getattr(WorkDir(workdir).reduced(1), key), getattr(actual, key)
        )
