"""Default quick-asymmetry cannot turn shared channels into independent data."""

import numpy as np
import pytest

from asymmetry.core.io.nexus import NexusLoader
from asymmetry.core.io.psi import PsiLoader
from asymmetry.core.io.root import RootLoader
from tests.io.test_nexus_loader import _write_v2_file
from tests.io.test_psi_loader import _write_psi_bin
from tests.io.test_root_loader import _write_red_green_offset_directory

pytestmark = pytest.mark.io


@pytest.mark.parametrize("loader_kind", ["nexus", "psi", "root"])
def test_raw_loading_survives_unavailable_default_pair(tmp_path, monkeypatch, loader_kind):
    if loader_kind == "nexus":
        import h5py

        path = tmp_path / "single.nxs"
        _write_v2_file(path)
        with h5py.File(path, "r+") as handle:
            data = handle["raw_data_1/instrument/detector_1/counts"]
            values = data[...][:1]
            del handle["raw_data_1/instrument/detector_1/counts"]
            handle["raw_data_1/instrument/detector_1/counts"] = values
        dataset = NexusLoader().load(str(path))
    elif loader_kind == "root":
        path = tmp_path / "single.root"
        _write_red_green_offset_directory(path)
        dataset = RootLoader().load(str(path))
    else:
        path = tmp_path / "shared.bin"
        _write_psi_bin(path)
        monkeypatch.setattr(
            PsiLoader, "_default_groups", lambda *args: ({1: [1], 2: [1]}, {1: "F", 2: "B"}, 1, 2)
        )
        dataset = PsiLoader().load(str(path))
    if isinstance(dataset, list):
        dataset = dataset[0]
    assert dataset.run.histograms
    assert np.sum(dataset.run.histograms[0].counts) > 0
    assert dataset.asymmetry.size == 0
    assert "shared acquired channels" in dataset.metadata["correction_error"]
    grouping = dataset.run.grouping
    assert (
        grouping["groups"][grouping["forward_group"]]
        == grouping["groups"][grouping["backward_group"]]
    )
