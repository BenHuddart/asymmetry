"""pyhdf-backed HDF4 read tests: synthetic round-trips + real-corpus parity.

Two layers, both requiring a working pyhdf/HDF4 runtime (so they run on CI
Linux/macOS, where the wheel is self-contained, and locally on Windows when
``ASYMMETRY_HDF4_DLL_DIR`` points at the HDF4 DLLs — see
``packaging/windows/fetch_hdf4_dlls.py``):

* **Synthetic round-trip** — write a tiny HDF4 v1 file with pyhdf, then read it
  back through ``NexusLoader``. This exercises the real container walk
  (``read_tree`` / ``_build_root`` / ``_read_vgroup`` / ``_read_sds``) end to
  end without committing any data.
* **Corpus parity** — opt-in via ``ASYMMETRY_WIMDA_CORPUS``: every legacy HDF4
  ``/run`` file must reduce identically to its ``nxs4to5``-converted HDF5 v2
  twin (a known-good oracle that loads like genuine ISISICP v2). Never commits
  data.

Pure-adapter and ``_load_v1`` logic (no HDF4 runtime needed) is covered by
``tests/test_hdf4_adapter.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.io]

h5py = pytest.importorskip("h5py")

# Register the Windows HDF4 runtime DLL directory before importing pyhdf; a
# no-op on Linux/macOS (self-contained wheel) and when the env var is unset.
_DLL_DIR = os.environ.get("ASYMMETRY_HDF4_DLL_DIR")
if _DLL_DIR and hasattr(os, "add_dll_directory") and Path(_DLL_DIR).is_dir():
    os.add_dll_directory(_DLL_DIR)
# Skip when pyhdf is missing *or* installed but its HDF4 runtime DLLs cannot be
# loaded (Windows without ASYMMETRY_HDF4_DLL_DIR) — both surface as ImportError.
pytest.importorskip("pyhdf.SD", exc_type=ImportError)

from asymmetry.core.io.hdf4 import is_hdf4
from asymmetry.core.io.nexus import NexusLoader

_CORPUS_ROOT = os.environ.get("ASYMMETRY_WIMDA_CORPUS")


@pytest.fixture()
def loader() -> NexusLoader:
    return NexusLoader()


def _as_single(result):
    return result[0] if isinstance(result, list) else result


# --- synthetic HDF4 writer (round-trip coverage of the pyhdf container walk) -


def _write_hdf4_v1(
    path,
    *,
    counts: np.ndarray,
    t0_bin: int | None = None,
    first_good_bin: int | None = None,
    last_good_bin: int | None = None,
    switching_states: int | None = None,
    number: int = 4242,
    title: str = "Synthetic v1",
    temp_log_name: str | None = None,
    temp_log_values: tuple[float, ...] = (5.0, 5.0, 5.0),
) -> None:
    """Write a minimal HDF4 ``/run`` muonTD file the same way the NeXus C API
    does (Vgroups + SDS), so ``read_tree`` parses it like a real legacy file.

    When ``temp_log_name`` is given, also write a generically-named NXlog Vgroup
    (``log_1``) carrying that sensor label in a ``name`` SDS plus ``time`` /
    ``values`` arrays — the native shape in which v1 files store thermometer
    logs (the converted HDF5 twin bakes the name into the selog path)."""
    import pyhdf.V  # noqa: F401  registers the V interface used by vgstart
    from pyhdf.HDF import HC, HDF
    from pyhdf.SD import SD, SDC

    counts = np.asarray(counts, dtype=np.int32)
    n_bins = counts.shape[-1]
    corrected_time = ((np.arange(n_bins) - 2) * 0.1).astype(np.float64)
    grouping = np.array([1, 2], dtype=np.int32)

    hdf = HDF(str(path), HC.WRITE | HC.CREATE)
    v = hdf.vgstart()
    sd = SD(str(path), SDC.WRITE | SDC.CREATE)
    try:
        np_of = {SDC.INT32: np.int32, SDC.FLOAT64: np.float64}

        def _num(name, arr, dtype, attrs=None):
            arr = np.asarray(arr, dtype=np_of[dtype])
            shape = arr.shape if arr.ndim else (1,)
            s = sd.create(name, dtype, shape)
            s[:] = arr.reshape(shape)
            for key, val in (attrs or {}).items():
                s.attr(key).set(SDC.INT32, int(val))
            ref = s.ref()
            s.endaccess()
            return ref

        def _char(name, text):
            s = sd.create(name, SDC.CHAR8, len(text))
            s[:] = text
            ref = s.ref()
            s.endaccess()
            return ref

        def _vgroup(name, cls):
            vg = v.create(name)
            vg._class = cls
            return vg

        run = _vgroup("run", "NXentry")
        run.add(HC.DFTAG_NDG, _char("analysis", "muonTD"))
        run.add(HC.DFTAG_NDG, _num("number", np.array([number]), SDC.INT32))
        run.add(HC.DFTAG_NDG, _char("title", title))
        if switching_states is not None:
            run.add(HC.DFTAG_NDG, _num("switching_states", np.array([switching_states]), SDC.INT32))

        bin_attrs = {}
        if t0_bin is not None:
            bin_attrs["t0_bin"] = t0_bin
        if first_good_bin is not None:
            bin_attrs["first_good_bin"] = first_good_bin
        if last_good_bin is not None:
            bin_attrs["last_good_bin"] = last_good_bin

        hist = _vgroup("histogram_data_1", "NXdata")
        hist.add(HC.DFTAG_NDG, _num("counts", counts, SDC.INT32, bin_attrs))
        hist.add(HC.DFTAG_NDG, _num("corrected_time", corrected_time, SDC.FLOAT64))
        hist.add(HC.DFTAG_NDG, _num("grouping", grouping, SDC.INT32))
        run.insert(hist)
        hist.detach()

        if temp_log_name is not None:
            log_values = np.asarray(temp_log_values, dtype=np.float64)
            log_times = np.arange(log_values.size, dtype=np.float64) * 10.0
            nxlog = _vgroup("log_1", "NXlog")  # generic Vgroup name
            nxlog.add(HC.DFTAG_NDG, _char("name", temp_log_name))
            nxlog.add(HC.DFTAG_NDG, _num("time", log_times, SDC.FLOAT64))
            nxlog.add(HC.DFTAG_NDG, _num("values", log_values, SDC.FLOAT64))
            run.insert(nxlog)
            nxlog.detach()

        run.detach()
    finally:
        v.end()
        sd.end()
        hdf.close()


def test_synthetic_hdf4_v1_roundtrip(loader: NexusLoader, tmp_path) -> None:
    """Write then read an HDF4 v1 file: covers the real pyhdf Vgroup/SDS walk,
    the char-SDS decode, and the 1-based counts-attr good-bin normalization."""
    path = tmp_path / "synth_v1.nxs"
    counts = np.arange(16, dtype=np.int32).reshape(2, 8)
    _write_hdf4_v1(path, counts=counts, t0_bin=3, first_good_bin=4, last_good_bin=8)

    assert is_hdf4(str(path))
    ds = _as_single(loader.load(str(path)))

    assert ds.metadata["nexus_version"] == "v1"
    assert ds.metadata["run_number"] == 4242
    assert ds.metadata["title"] == "Synthetic v1"  # char SDS decoded in full
    # 1-based 4/8 normalized to 0-based 3/7; t0_bin 3 -> 2; window [3..7] = 5.
    assert ds.run.grouping["first_good_bin"] == 3
    assert ds.run.grouping["last_good_bin"] == 7
    assert ds.run.histograms[0].t0_bin == 2
    assert ds.n_points == 5
    np.testing.assert_array_equal(
        np.asarray(ds.run.histograms[0].counts, dtype=np.int32), counts[0]
    )


def test_synthetic_hdf4_multiperiod_roundtrip(loader: NexusLoader, tmp_path) -> None:
    """A flat [n_periods*n_spectra, n_bins] HDF4 block with switching_states=2
    must split period-major into two periods through the real container walk."""
    path = tmp_path / "synth_mp.nxs"
    counts = np.arange(32, dtype=np.int32).reshape(4, 8)  # 2 periods x 2 spectra
    _write_hdf4_v1(path, counts=counts, switching_states=2)

    # Two periods collapse to a single combined red/green dataset.
    ds = _as_single(loader.load(str(path)))
    assert not isinstance(ds, list)
    assert ds.metadata.get("period_count") == 2
    # Period-major C-order: period 1's first spectrum is input row 0.
    np.testing.assert_array_equal(
        np.asarray(ds.run.histograms[0].counts, dtype=np.int32), counts[0]
    )


def test_synthetic_hdf4_surfaces_logged_temperature_via_nxlog_name(
    loader: NexusLoader, tmp_path
) -> None:
    """Through the real pyhdf Vgroup/SDS walk, a generically-named NXlog whose
    sensor label is in a ``name`` SDS must populate ``sample_temperature_logged``
    and appear in ``nexus_time_series`` — closing the native-HDF4 logged-T gap."""
    path = tmp_path / "synth_log.nxs"
    counts = np.arange(16, dtype=np.int32).reshape(2, 8)
    _write_hdf4_v1(
        path, counts=counts, temp_log_name="Temp_Sample", temp_log_values=(4.8, 5.0, 5.2)
    )

    ds = _as_single(loader.load(str(path)))
    series = ds.metadata["nexus_time_series"]
    assert "log_1" in series
    assert series["log_1"]["name"] == "Temp_Sample"
    assert ds.sample_temperature_logged == pytest.approx(5.0)


def test_hdf4_load_does_not_lock_file(loader: NexusLoader, tmp_path) -> None:
    """After loading, the file is unlocked (pyhdf V/SD/HDF handles closed) —
    matters on Windows: a left-open handle would block the unlink below."""
    path = tmp_path / "synth_lock.nxs"
    _write_hdf4_v1(path, counts=np.arange(16, dtype=np.int32).reshape(2, 8))
    _ = _as_single(loader.load(str(path)))
    path.unlink()  # raises PermissionError on Windows if still locked
    assert not path.exists()


# --- real-corpus parity sweep (opt-in via ASYMMETRY_WIMDA_CORPUS) -----------


def _assert_reduced_parity(ds4, ds5) -> None:
    """Assert two reduced datasets (HDF4-v1 vs HDF5-v2 twin) match."""
    assert ds4.time.shape == ds5.time.shape, (ds4.time.shape, ds5.time.shape)
    assert np.allclose(ds4.time, ds5.time, rtol=1e-6, atol=1e-6)
    assert np.allclose(ds4.asymmetry, ds5.asymmetry, rtol=1e-6, atol=1e-6)
    assert np.allclose(ds4.error, ds5.error, rtol=1e-6, atol=1e-6)

    assert int(ds4.metadata["run_number"]) == int(ds5.metadata["run_number"])
    assert ds4.metadata.get("instrument") == ds5.metadata.get("instrument")
    assert ds4.metadata.get("title") == ds5.metadata.get("title")
    assert ds4.metadata.get("field_state") == ds5.metadata.get("field_state")
    assert pytest.approx(float(ds5.metadata["temperature"]), rel=1e-5) == float(
        ds4.metadata["temperature"]
    )
    assert pytest.approx(float(ds5.metadata["field"]), abs=1e-6) == float(ds4.metadata["field"])

    # Logged sample temperature must agree across containers — the native-HDF4
    # logged-T gap (a Temp_Sample NXlog whose Vgroup is generically named and
    # whose sensor label lives in a ``name`` child). Same value, or both absent.
    t4 = ds4.metadata.get("sample_temperature_logged")
    t5 = ds5.metadata.get("sample_temperature_logged")
    if t4 is None or t5 is None:
        assert t4 is None and t5 is None, (t4, t5)
    else:
        assert pytest.approx(float(t5), rel=1e-5) == float(t4)

    g4, g5 = ds4.run.grouping, ds5.run.grouping
    assert int(g4["first_good_bin"]) == int(g5["first_good_bin"])
    assert int(g4["last_good_bin"]) == int(g5["last_good_bin"])
    assert g4["groups"] == g5["groups"]

    assert len(ds4.run.histograms) == len(ds5.run.histograms)
    for h4, h5 in zip(ds4.run.histograms, ds5.run.histograms):
        np.testing.assert_array_equal(
            np.asarray(h4.counts, dtype=np.int64), np.asarray(h5.counts, dtype=np.int64)
        )


def _corpus_pairs() -> list[tuple[Path, Path]]:
    """Every corpus HDF4 file paired with its converted ``*_hdf5`` twin."""
    if not _CORPUS_ROOT:
        return []
    root = Path(_CORPUS_ROOT)
    pairs: list[tuple[Path, Path]] = []
    for nxs in root.rglob("*.nxs"):
        parts = nxs.parts
        if any(p.lower().endswith("_hdf5") for p in parts):
            continue
        if not is_hdf4(str(nxs)):
            continue
        # Map ``.../data/<f>`` (or ``Data``) to its ``.../data_hdf5/<f>`` twin,
        # preserving the original case of the data segment.
        twin = None
        for i, part in enumerate(parts):
            if part.lower() == "data":
                twin = Path(*parts[:i], part + "_hdf5", *parts[i + 1 :])
                break
        if twin is not None and twin.exists():
            pairs.append((nxs, twin))
    return pairs


@pytest.mark.slow
@pytest.mark.skipif(not _CORPUS_ROOT, reason="ASYMMETRY_WIMDA_CORPUS not set")
@pytest.mark.parametrize("pair", _corpus_pairs(), ids=lambda p: p[0].stem if p else "none")
def test_corpus_hdf4_parity(loader: NexusLoader, pair: tuple[Path, Path]) -> None:
    hdf4_path, hdf5_path = pair
    ds4 = _as_single(loader.load(str(hdf4_path)))
    ds5 = _as_single(loader.load(str(hdf5_path)))
    _assert_reduced_parity(ds4, ds5)
