"""Synthetic tests for the HDF4 -> h5py adapter and the v1 reduction path.

No real data and no HDF4 runtime: the adapter wrappers and ``_load_v1`` are
driven by hand-built :class:`Group` / :class:`Dataset` trees fed straight to
the loader, so these run in CI everywhere. (Container parsing via pyhdf is
covered against the real corpus by the env-gated sweep in
``tests/test_hdf4_loader.py``.)
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

pytestmark = [pytest.mark.io]

from asymmetry.core.io.hdf4 import (
    HDF4_MAGIC,
    Dataset,
    Group,
    _Hdf4Group,
    _to_str,
    is_hdf4,
)
from asymmetry.core.io.nexus import NexusLoader


def _char(name: str, text: str) -> Dataset:
    """A char SDS, stored as a signed-char array like pyhdf returns."""
    return Dataset(name, np.frombuffer(text.encode("latin1"), dtype=np.int8).copy(), is_char=True)


def _sample_tree() -> _Hdf4Group:
    root = Group("/", "NXroot")
    run = Group("run", "NXentry")
    run.children["analysis"] = _char("analysis", "muonTD")
    run.children["number"] = Dataset("number", np.array([44989], dtype=np.int32))
    sample = Group("sample", "NXsample")
    sample.children["temperature"] = Dataset(
        "temperature", np.array([290.0], dtype=np.float32), attrs={"units": "Kelvin"}
    )
    run.children["sample"] = sample
    hist = Group("histogram_data_1", "NXdata")
    hist.children["counts"] = Dataset(
        "counts",
        np.arange(8, dtype=np.int32).reshape(2, 4),
        attrs={"t0_bin": 30, "first_good_bin": 36},
    )
    run.children["histogram_data_1"] = hist
    root.children["run"] = run
    return _Hdf4Group(root)


# --- adapter surface --------------------------------------------------------


def test_group_membership_and_keys() -> None:
    handle = _sample_tree()
    assert "run" in handle
    assert "missing" not in handle
    assert "run" in list(handle.keys())
    assert set(iter(handle)) == {"run"}


def test_group_getitem_and_get() -> None:
    handle = _sample_tree()
    run = handle["run"]
    assert hasattr(run, "keys")  # a group, not a dataset
    assert not hasattr(run, "dtype")
    with pytest.raises(KeyError):
        _ = handle["missing"]
    assert handle.get("missing") is None
    assert handle.get("run") is not None


def test_group_attrs_expose_nxclass() -> None:
    handle = _sample_tree()
    assert handle.attrs.get("NX_class") == "NXroot"
    assert handle["run"].attrs.get("NX_class") == "NXentry"


def test_numeric_dataset_surface() -> None:
    counts = _sample_tree()["run"]["histogram_data_1"]["counts"]
    assert counts.dtype == np.int32
    assert counts.shape == (2, 4)
    np.testing.assert_array_equal(counts[()], np.arange(8).reshape(2, 4))
    np.testing.assert_array_equal(np.asarray(counts), np.arange(8).reshape(2, 4))
    assert counts.attrs.get("t0_bin") == 30
    assert counts.attrs.get("first_good_bin") == 36


def test_nested_chained_getitem() -> None:
    handle = _sample_tree()
    temp = handle["run"]["sample"]["temperature"]
    assert float(np.asarray(temp[()]).flat[0]) == pytest.approx(290.0)
    assert temp.attrs.get("units") == "Kelvin"


@pytest.mark.parametrize(
    "data",
    [
        np.frombuffer(b"muonTD", dtype=np.int8).copy(),  # signed char SDS
        np.frombuffer(b"muonTD", dtype=np.uint8).copy(),  # unsigned char SDS
        np.array([b"m", b"u", b"o", b"n", b"T", b"D"], dtype="S1"),  # |S1 char SDS
    ],
    ids=["int8", "uint8", "S1"],
)
def test_char_dataset_decodes_full_string(data) -> None:
    """A char SDS must yield the whole decoded string on ``[()]`` (not byte 0)."""
    grp = Group("g", "NXentry")
    grp.children["analysis"] = Dataset("analysis", data, is_char=True)
    assert _Hdf4Group(grp)["analysis"][()] == "muonTD"


def test_to_str_strips_trailing_nul_and_whitespace() -> None:
    assert _to_str(np.frombuffer(b"EMU\x00\x00", dtype=np.int8).copy()) == "EMU"


# --- v1 reduction over the adapter (the counts-attr fix) --------------------


def _v1_handle(*, with_bin_attrs: bool) -> _Hdf4Group:
    """A minimal but representative v1 ``/run`` tree, as the HDF4 reader yields.

    Eight bins, two detectors, two groups. ``corrected_time`` is centred so the
    bin closest to t=0 is index 2 (0-based); the counts attributes use 1-based
    numbering (``t0_bin=3``, ``first_good_bin=4``, ``last_good_bin=8``) exactly
    as real ISIS v1 files do.
    """
    counts_attrs = {"t0_bin": 3, "first_good_bin": 4, "last_good_bin": 8} if with_bin_attrs else {}
    root = Group("/", "NXroot")
    run = Group("run", "NXentry")
    run.children["analysis"] = _char("analysis", "muonTD")
    run.children["IDF_version"] = Dataset("IDF_version", np.array([1], dtype=np.int32))
    run.children["number"] = Dataset("number", np.array([4242], dtype=np.int32))
    run.children["title"] = _char("title", "Synthetic v1")

    instrument = Group("instrument", "NXinstrument")
    instrument.children["name"] = _char("name", "MUSR")
    detector = Group("detector", "NXdetector")
    detector.children["orientation"] = _char("orientation", "L")
    instrument.children["detector"] = detector
    run.children["instrument"] = instrument

    sample = Group("sample", "NXsample")
    sample.children["temperature"] = Dataset(
        "temperature", np.array([10.0], dtype=np.float32), attrs={"units": "Kelvin"}
    )
    sample.children["magnetic_field"] = Dataset("magnetic_field", np.array([100.0], np.float32))
    sample.children["magnetic_field_state"] = _char("magnetic_field_state", "TF")
    run.children["sample"] = sample

    hist = Group("histogram_data_1", "NXdata")
    counts = np.array(
        [[10, 12, 14, 16, 18, 20, 22, 24], [8, 9, 10, 11, 12, 13, 14, 15]], dtype=np.int32
    )
    hist.children["counts"] = Dataset("counts", counts, attrs=counts_attrs)
    hist.children["corrected_time"] = Dataset(
        "corrected_time", ((np.arange(8) - 2) * 0.1).astype(np.float32)
    )
    hist.children["grouping"] = Dataset("grouping", np.array([1, 2], dtype=np.int32))
    hist.children["dead_time"] = Dataset("dead_time", np.array([0.01, 0.02], np.float32))
    run.children["histogram_data_1"] = hist

    root.children["run"] = run
    return _Hdf4Group(root)


def test_v1_reads_good_bin_window_and_t0_from_counts_attrs() -> None:
    """v1 must read t0/good-bin from the counts SDS attributes and normalize
    the 1-based numbering — the gap the HDF4 port surfaced."""
    loader = NexusLoader()
    result = loader._reduce_handle(_v1_handle(with_bin_attrs=True), "synthetic")
    ds = result[0] if isinstance(result, list) else result

    assert ds.metadata["nexus_version"] == "v1"
    assert ds.metadata["run_number"] == 4242
    assert ds.metadata["instrument"] == "MUSR"
    assert ds.metadata["title"] == "Synthetic v1"
    assert ds.metadata["field_state"] == "TF"
    assert ds.metadata["temperature"] == pytest.approx(10.0)
    assert ds.metadata["field"] == pytest.approx(100.0)

    g = ds.run.grouping
    # 1-based 4/8 normalized to 0-based 3/7; t0_bin 3 -> 2; base recorded as 1.
    assert g["first_good_bin"] == 3
    assert g["last_good_bin"] == 7
    assert g["bin_index_base"] == 1
    assert ds.run.histograms[0].t0_bin == 2
    # Window [3..7] inclusive -> 5 points (would be 8 if attrs were ignored).
    assert ds.n_points == 5


def test_v1_multiperiod_flat_counts_split_by_switching_states() -> None:
    """Legacy v1 stores multi-period counts flat as [n_periods*n_spectra, nb];
    ``switching_states`` must drive a period-major split (HiFi RF/ALC dialect)."""
    root = Group("/", "NXroot")
    run = Group("run", "NXentry")
    run.children["analysis"] = _char("analysis", "muonTD")
    run.children["number"] = Dataset("number", np.array([56426], dtype=np.int32))
    run.children["switching_states"] = Dataset("switching_states", np.array([2], dtype=np.int32))
    hist = Group("histogram_data_1", "NXdata")
    # 4 rows = 2 periods x 2 spectra, 6 bins.
    hist.children["counts"] = Dataset("counts", np.arange(24, dtype=np.int32).reshape(4, 6))
    hist.children["corrected_time"] = Dataset("corrected_time", (np.arange(6) * 0.1).astype(float))
    hist.children["grouping"] = Dataset("grouping", np.array([1, 2], dtype=np.int32))
    run.children["histogram_data_1"] = hist
    root.children["run"] = run

    loader = NexusLoader()
    result = loader._reduce_handle(_Hdf4Group(root), "synthetic")
    # Two periods collapse to a single combined red/green dataset.
    ds = result[0] if isinstance(result, list) else result
    assert len(result) == 1
    assert ds.metadata.get("period_count") == 2


def test_v1_without_counts_attrs_falls_back_to_full_range() -> None:
    """Attr-less v1 files keep the prior behaviour (no window, zero-based)."""
    loader = NexusLoader()
    result = loader._reduce_handle(_v1_handle(with_bin_attrs=False), "synthetic")
    ds = result[0] if isinstance(result, list) else result

    g = ds.run.grouping
    assert g["first_good_bin"] == 0
    assert g["last_good_bin"] == 7
    assert g["bin_index_base"] == 0
    assert ds.n_points == 8


# --- detection + dependency guard ------------------------------------------


def test_is_hdf4_detection(tmp_path) -> None:
    hdf4_file = tmp_path / "hdf4.nxs"
    hdf4_file.write_bytes(HDF4_MAGIC + b"\x00" * 64)
    not_hdf4 = tmp_path / "other.nxs"
    not_hdf4.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 64)  # HDF5 magic
    assert is_hdf4(str(hdf4_file)) is True
    assert is_hdf4(str(not_hdf4)) is False
    assert is_hdf4(str(tmp_path / "missing.nxs")) is False


def test_pyhdf_absent_raises_clear_error_but_hdf5_still_loads(monkeypatch, tmp_path) -> None:
    """With pyhdf unavailable, HDF4 load raises a pointing error; HDF5 unaffected."""
    h5py = pytest.importorskip("h5py")
    import asymmetry.core.io.hdf4 as hdf4_mod

    # Simulate pyhdf not being importable, and reset the one-shot DLL guard.
    for name in ("pyhdf", "pyhdf.V", "pyhdf.HDF", "pyhdf.SD"):
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setattr(hdf4_mod, "_dll_runtime_registered", False, raising=False)

    # An HDF4-magic file routes to the pyhdf path, which must raise before parsing.
    hdf4_file = tmp_path / "legacy.nxs"
    hdf4_file.write_bytes(HDF4_MAGIC + b"\x00" * 64)

    loader = NexusLoader()
    with pytest.raises(ImportError, match=r"pyhdf|asymmetry\[hdf4\]"):
        loader.load(str(hdf4_file))

    # The HDF4 dependency must not become a hard requirement for HDF5 files.
    hdf5_file = tmp_path / "modern.nxs"
    with h5py.File(hdf5_file, "w") as f:
        run = f.create_group("run")
        run.create_dataset("analysis", data=np.bytes_("muonTD"))
        run.create_dataset("IDF_version", data=1)
        run.create_dataset("number", data=7)
        h_data = run.create_group("histogram_data_1")
        h_data.create_dataset("counts", data=np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=float))
        h_data.create_dataset("corrected_time", data=np.array([0.0, 0.1, 0.2, 0.3]))
        h_data.create_dataset("grouping", data=np.array([1, 2], dtype=np.int32))

    result = loader.load(str(hdf5_file))
    ds = result[0] if isinstance(result, list) else result
    assert ds.metadata["run_number"] == 7
