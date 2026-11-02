"""File-boundary shape checks and named detector-alignment provenance."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from asymmetry.core.io.nexus import NexusLoader
from asymmetry.core.io.psi import PsiLoader
from asymmetry.core.io.root import RootLoader, _RootHistogram
from asymmetry.core.transform.grouping import align_forward_backward, effective_group_indices
from asymmetry.core.transform.rebin import binning_spec
from asymmetry.core.transform.reduce import reduce_grouped_asymmetry
from tests.io.test_nexus_loader import _write_v2_file
from tests.io.test_psi_loader import _write_psi_bin
from tests.io.test_root_loader import _write_musrroot_directory

pytestmark = pytest.mark.io


def test_psi_rejects_ragged_raw_acquisition_counts(tmp_path):
    path = tmp_path / "synthetic.bin"
    _write_psi_bin(path)
    loader = PsiLoader()
    raw = loader._read_bin(path)
    raw.counts[0] = raw.counts[0][:-1]
    with pytest.raises(ValueError, match="equal shape"):
        loader._build_dataset(raw)


def test_root_rejects_ragged_raw_acquisition_counts():
    histograms = [_RootHistogram(i + 1, np.ones(n), "") for i, n in enumerate((10, 8))]
    with pytest.raises(ValueError, match="equal shape"):
        RootLoader()._build_dataset(
            Path("synthetic.root"), {"RunInfo/Time Resolution": "10 ns"}, "musrroot", histograms
        )


@pytest.mark.parametrize("kind", ["psi", "root", "nexus"])
def test_loader_records_the_named_group_alignment(tmp_path, kind):
    writer, loader, extension = {
        "psi": (_write_psi_bin, PsiLoader(), "bin"),
        "root": (_write_musrroot_directory, RootLoader(), "root"),
        "nexus": (_write_v2_file, NexusLoader(), "nxs"),
    }[kind]
    path = tmp_path / f"synthetic.{extension}"
    writer(path)
    dataset = loader.load(str(path))
    run = dataset.run
    forward, backward = (
        effective_group_indices(run.grouping, run.grouping[key])
        for key in ("forward_group", "backward_group")
    )
    _, _, crop = align_forward_backward(run.histograms, forward, backward)
    assert dataset.metadata["grouped_alignment"] == asdict(crop)

    reduction = reduce_grouped_asymmetry(
        histograms=run.histograms,
        binning=binning_spec(run.grouping),
        grouping=run.grouping,
        forward_idx=forward,
        backward_idx=backward,
        alpha=run.grouping["alpha"],
        use_deadtime=False,
        deadtime_mode="off",
        use_background=False,
    )
    np.testing.assert_allclose(dataset.time, reduction.time, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(dataset.asymmetry, reduction.asymmetry, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(dataset.error, reduction.error, rtol=1e-13, atol=1e-13)
