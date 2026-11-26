"""Authored copies preserve acquisition facts and establish new source identity."""

from contextlib import nullcontext

import h5py
import numpy as np
import pytest

from asymmetry.core.data.acquisition import ContentIdentity, SourceBinding
from asymmetry.core.io import load
from asymmetry.core.io.nexus_grouping import NexusGroupingWarning
from tests.corpus import authored_grouping_copy
from tests.io.test_nexus_grouping_declarations import grouping_file

pytestmark = pytest.mark.io


@pytest.mark.parametrize("container", ["hdf4", "hdf5"])
def test_authored_copy_keeps_invalid_original_and_all_other_acquisition_fields(tmp_path, container):
    source = tmp_path / "original.nxs"
    if container == "hdf4":
        from tests.io.test_hdf4_loader import _write_hdf4_v1

        pyhdf_sd = pytest.importorskip("pyhdf.SD")

        _write_hdf4_v1(source, counts=np.full((2, 24), 1000), time_zero=0.032)
        handle = pyhdf_sd.SD(str(source), pyhdf_sd.SDC.WRITE)
        node = handle.select("grouping")
        node[:] = [0, 0]
        for name in ("available", "number_groups"):
            node.attr(name).set(pyhdf_sd.SDC.INT32, 1)
        node.endaccess()
        handle.end()
        from asymmetry.core.io.hdf4 import open_hdf4

        def opener(path):
            return nullcontext(open_hdf4(path))
    else:
        grouping_file(source, "v1", table=[0] * 2, count=2)
        with h5py.File(source, "r+") as handle:
            handle["run/histogram_data_1/grouping"].attrs["number_groups"] = 1
        opener = h5py.File
    source.with_suffix(".log").write_text("synthetic sidecar preserved verbatim\n")
    original_bytes = source.read_bytes()
    with pytest.raises(ValueError, match="no positive"):
        load(source)
    target = authored_grouping_copy(source, tmp_path / "authored", [1, 2])
    assert source.read_bytes() == original_bytes
    assert target.read_bytes() != original_bytes
    assert target.with_suffix(".log").read_bytes() == source.with_suffix(".log").read_bytes()
    with opener(str(source)) as original, opener(str(target)) as authored:

        def compare(left, right):
            assert set(left.keys()) == set(right.keys())
            for key in left.keys():
                if key == "grouping":
                    np.testing.assert_array_equal(left[key][()], [0, 0])
                    np.testing.assert_array_equal(right[key][()], [1, 2])
                    for name in ("available", "number_groups"):
                        assert left[key].attrs[name] == 1
                        assert right[key].attrs[name] == 2
                elif isinstance(left[key], h5py.Group) or hasattr(left[key], "keys"):
                    compare(left[key], right[key])
                else:
                    np.testing.assert_array_equal(left[key][()], right[key][()])
                    assert dict(left[key].attrs) == dict(right[key].attrs)

        compare(original, authored)
    dataset = load(target)
    assert dataset.run.grouping["groups"] == {1: [1], 2: [2]}
    identity = ContentIdentity.read(target)
    assert dataset.run.source_binding == SourceBinding(expected=identity, observed=identity)
    assert identity != ContentIdentity.read(source)
    with pytest.raises(ValueError, match="no positive"):
        load(source)


@pytest.mark.parametrize(
    ("relative", "recover_emu"),
    [
        ("Semiconductors/Shallow donor state in cadmium sulphide/Data/EMU00020711.nxs", True),
        ("Chemistry/Muonium reaction with maleic acid/Data/emu00078251.nxs", True),
        ("Chemistry/Muon spectroscopy of benzene/data/RF resonance/56426.nxs", False),
    ],
)
def test_original_corpus_all_zero_declared_grouping_policy(relative, recover_emu):
    from tests.corpus import corpus_path

    path = corpus_path(relative)
    if path is None:
        pytest.skip("WiMDA corpus not configured (set ASYMMETRY_CORPUS_ROOT)")
    original_bytes = path.read_bytes()
    if recover_emu:
        with pytest.warns(NexusGroupingWarning, match="standard EMU forward/backward grouping"):
            dataset = load(path)
        grouping = dataset.run.grouping
        assert grouping["groups"] == {1: list(range(1, 49)), 2: list(range(49, 97))}
        assert (grouping["forward_group"], grouping["backward_group"]) == (1, 2)
        assert grouping["grouping_preset"] == "Longitudinal"
        assert grouping["nexus_grouping_declaration"]["values"] == [0] * 96
        provenance = grouping["grouping_provenance"]
        assert provenance["profile_key"] == "isis.emu"
        assert provenance["fallback_reason"] == "nexus_v1_all_zero"
        assert dataset.effective_recipe.to_dict()["grouping_provenance"] == provenance
    else:
        with pytest.raises(ValueError, match="no positive"):
            load(path)
    assert path.read_bytes() == original_bytes
