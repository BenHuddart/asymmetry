"""Real-corpus parity sweep: direct HDF4 v1 read vs converted HDF5 v2 twin.

The WiMDA-muon-school corpus ships every legacy HDF4 ``/run`` muonTD file
alongside a converted HDF5 (v2 ``/raw_data_1``) twin produced by ``nxs4to5``,
which loads identically to genuine ISISICP v2 files. That makes the converted
twin a known-good oracle: a direct HDF4 read (``_load_v1``) must reduce to the
*same* asymmetry / counts / metadata as the twin read through ``_load_v2``.

This is **opt-in and never commits data**: it runs only when
``ASYMMETRY_WIMDA_CORPUS`` points at the corpus root. CI coverage of the
loader logic on synthetic data lives in ``tests/test_hdf4_adapter.py``.

On Windows, pyhdf's wheel needs an external HDF4 C runtime
(``hdf.dll`` / ``mfhdf.dll``); point ``ASYMMETRY_HDF4_DLL_DIR`` at a directory
holding them (see ``packaging/windows/fetch_hdf4_dlls.py``). On Linux/macOS the
wheel bundles HDF4, so no setup is needed.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.io, pytest.mark.slow]

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


_PAIRS = _corpus_pairs()


@pytest.mark.skipif(not _CORPUS_ROOT, reason="ASYMMETRY_WIMDA_CORPUS not set")
@pytest.mark.parametrize("pair", _PAIRS, ids=lambda p: p[0].stem if p else "none")
def test_corpus_hdf4_parity(loader: NexusLoader, pair: tuple[Path, Path]) -> None:
    hdf4_path, hdf5_path = pair
    ds4 = _as_single(loader.load(str(hdf4_path)))
    ds5 = _as_single(loader.load(str(hdf5_path)))
    _assert_reduced_parity(ds4, ds5)


@pytest.mark.skipif(not _PAIRS, reason="ASYMMETRY_WIMDA_CORPUS not set / no HDF4 pairs")
def test_hdf4_load_does_not_lock_file(loader: NexusLoader, tmp_path) -> None:
    """After loading, the file is unlocked (pyhdf V/SD/HDF handles closed) —
    matters on Windows: a left-open handle would block the unlink below."""
    import shutil

    src = _PAIRS[0][0]
    copied = tmp_path / src.name
    shutil.copy(src, copied)
    _ = _as_single(loader.load(str(copied)))
    copied.unlink()  # raises PermissionError on Windows if still locked
    assert not copied.exists()
