"""HDF4 container support for the ISIS muon NeXus v1 (``/run``) layout.

Legacy ISIS muon NeXus files (the format WiMDA reads natively) store the v1
``muonTD`` schema in an **HDF4** container, which ``h5py`` cannot open
(``OSError: file signature not found``). The schema itself is already
understood by :class:`asymmetry.core.io.nexus.NexusLoader` via ``_load_v1`` —
only the container differs.

This module closes that gap with two pieces:

* a small reader that walks the HDF4 Vgroup/SDS hierarchy (via ``pyhdf``) into
  an in-memory :class:`Group` / :class:`Dataset` tree, and
* a thin adapter (:func:`open_hdf4`) exposing the read-only slice of the
  ``h5py`` API that ``_load_v1`` relies on (``in`` / ``[]`` / ``keys`` /
  ``get`` / ``.attrs`` / array coercion), so the existing v1 reader runs over
  an HDF4 file unchanged.

``pyhdf`` is an optional dependency (``asymmetry[hdf4]``). On Linux/macOS its
wheels bundle the HDF4 C library; on Windows the wheel needs an external HDF4
runtime (``hdf.dll`` / ``mfhdf.dll``) — see :func:`_register_hdf4_runtime` and
``packaging/windows/fetch_hdf4_dlls.py``.

The reader logic is re-implemented from the MIT-compatible WiMDA-muon-school
``nxs4to5`` tooling; it carries no Mantid (GPL) lineage.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

#: HDF4 file magic. ``h5py`` cannot open these; sniff before ``h5py.File``.
HDF4_MAGIC = b"\x0e\x03\x13\x01"

#: Vgroup classes used internally by the HDF4/netCDF libraries (not NeXus
#: groups); skipped when locating top-level entries.
_INTERNAL_CLASSES = frozenset(
    {
        "CDF0.0",
        "Var0.0",
        "Dim0.0",
        "UDim0.0",
        "Attr0.0",
        "RIG0.0",
        "RI0.0",
        "DimVal0.0",
        "DimVal0.1",
    }
)

_dll_runtime_registered = False


def is_hdf4(path: str) -> bool:
    """Return ``True`` when *path* is an HDF4 container (by file magic)."""
    try:
        with open(path, "rb") as fh:
            return fh.read(4) == HDF4_MAGIC
    except OSError:
        return False


def _register_hdf4_runtime() -> None:
    """Make the external HDF4 C runtime discoverable before importing pyhdf.

    Only relevant on Windows, where pyhdf's wheel does not bundle
    ``hdf.dll`` / ``mfhdf.dll``. Searches, in order, the frozen-app bundle
    directory (PyInstaller) and ``ASYMMETRY_HDF4_DLL_DIR``. A no-op on
    Linux/macOS (no ``os.add_dll_directory``) and when nothing is found.
    """
    global _dll_runtime_registered
    if _dll_runtime_registered or not hasattr(os, "add_dll_directory"):
        return

    candidates: list[Path] = []
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).parent)
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(Path(meipass))
    env_dir = os.environ.get("ASYMMETRY_HDF4_DLL_DIR")
    if env_dir:
        candidates.append(Path(env_dir))

    for directory in candidates:
        try:
            if directory.is_dir():
                os.add_dll_directory(str(directory))
        except OSError:
            continue
    _dll_runtime_registered = True


def _require_pyhdf():
    """Import pyhdf, raising a clear, actionable error when unavailable."""
    _register_hdf4_runtime()
    try:
        import pyhdf.V  # noqa: F401  registers the V interface used by vgstart
        from pyhdf.HDF import HC, HDF
        from pyhdf.SD import SD, SDC
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(
            "Reading HDF4 NeXus (.nxs) files requires pyhdf. Install it with "
            "'pip install asymmetry[hdf4]' (or 'pip install pyhdf'). On Windows "
            "pyhdf also needs the HDF4 C runtime (hdf.dll/mfhdf.dll): install "
            "conda-forge 'hdf4', or run packaging/windows/fetch_hdf4_dlls.py and "
            "point ASYMMETRY_HDF4_DLL_DIR at the result."
        ) from exc
    return HDF, HC, SD, SDC


# --- In-memory HDF4 tree ----------------------------------------------------


@dataclass
class Dataset:
    """A scientific dataset (SDS) read from an HDF4 file."""

    name: str
    data: np.ndarray
    attrs: dict = field(default_factory=dict)
    #: ``True`` for HDF4 char SDS (CHAR8/UCHAR8) holding a string.
    is_char: bool = False


@dataclass
class Group:
    """A NeXus group (HDF4 Vgroup) and its named children."""

    name: str
    nxclass: str
    children: dict = field(default_factory=dict)  # name -> Group | Dataset


def _to_str(arr: np.ndarray) -> str:
    """Decode an HDF4 char SDS (stored as int8/uint8/bytes) to ``str``."""
    a = np.asarray(arr)
    if a.dtype.kind in ("i", "u"):
        b = a.astype(np.uint8).tobytes()
    elif a.dtype.kind == "S":
        b = a.tobytes()
    else:
        return str(a)
    return b.decode("latin1").rstrip("\x00").strip()


def read_tree(path: str) -> Group:
    """Read an HDF4 NeXus file into a :class:`Group` tree rooted at ``'/'``.

    Closes every pyhdf ``V`` / ``SD`` / ``HDF`` handle before returning so the
    file is not left locked (matters on Windows).
    """
    hdf_cls, hc, sd_cls, sdc = _require_pyhdf()
    hdf = hdf_cls(str(path))
    v = hdf.vgstart()
    sd = sd_cls(str(path))
    try:
        return _build_root(v, sd, hc, sdc)
    finally:
        for closer in (v.end, sd.end, hdf.close):
            try:
                closer()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass


def _build_root(v, sd, hc, sdc) -> Group:
    """Assemble the top-level groups (members of no other vgroup)."""
    refs: list[int] = []
    ref = -1
    while True:
        try:
            ref = v.getid(ref)
        except Exception:  # noqa: BLE001 - pyhdf signals end-of-iteration by raising
            break
        refs.append(ref)

    members: set[int] = set()
    info: dict[int, tuple[str, str]] = {}
    for r in refs:
        vg = v.attach(r)
        try:
            info[r] = (vg._name, vg._class)
            for tag, mref in vg.tagrefs():
                if tag == hc.DFTAG_VG:
                    members.add(mref)
        finally:
            vg.detach()

    root = Group("/", "NXroot")
    for r in refs:
        name, cls = info[r]
        if r in members or cls in _INTERNAL_CLASSES:
            continue
        root.children[name] = _read_vgroup(v, sd, r, hc, sdc)
    return root


def _read_vgroup(v, sd, ref, hc, sdc) -> Group:
    vg = v.attach(ref)
    try:
        grp = Group(vg._name, vg._class)
        for tag, r in vg.tagrefs():
            if tag == hc.DFTAG_VG:
                child = _read_vgroup(v, sd, r, hc, sdc)
                grp.children[child.name] = child
            elif tag == hc.DFTAG_NDG:  # scientific dataset
                ds = _read_sds(sd, r, sdc)
                if ds is not None:
                    grp.children[ds.name] = ds
            # DFTAG_VH (vdata) does not occur in ISIS muon v1 files; ignored.
        return grp
    finally:
        vg.detach()


def _read_sds(sd, ref, sdc) -> Dataset | None:
    sds = None
    try:
        sds = sd.select(sd.reftoindex(ref))
        name, _rank, _dims, dtype, _natt = sds.info()
        data = np.asarray(sds.get())
        attrs = dict(sds.attributes())
        is_char = dtype in (sdc.CHAR8, sdc.UCHAR8)
        return Dataset(name, data, attrs, is_char=is_char)
    except Exception:  # noqa: BLE001 - skip unreadable SDS rather than abort the tree
        return None
    finally:
        if sds is not None:
            try:
                sds.endaccess()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass


# --- h5py-compatible adapter ------------------------------------------------
#
# ``_load_v1`` consumes its handle through a small, uniform surface:
#   "name" in node | node["name"] | node.keys() | node.get(name) | node.attrs
#   and np.asarray(dataset) / dataset[()]
# These wrappers reproduce exactly that slice over the HDF4 tree, so the v1
# schema reader stays format-agnostic.


class _Hdf4Dataset:
    """h5py-dataset-shaped view of a :class:`Dataset`."""

    def __init__(self, dataset: Dataset) -> None:
        self._dataset = dataset

    @property
    def dtype(self):
        return self._dataset.data.dtype

    @property
    def shape(self):
        return self._dataset.data.shape

    @property
    def attrs(self) -> dict:
        return self._dataset.attrs

    def __array__(self, dtype=None):
        data = self._dataset.data
        return data.astype(dtype) if dtype is not None else data

    def __getitem__(self, key):
        # A char SDS is a string field: h5py would return the whole string as
        # bytes, so callers (_safe_str) decode it correctly. Return the decoded
        # string here rather than the raw per-character array (whose .flat[0]
        # would otherwise be mistaken for the whole value).
        if self._dataset.is_char:
            return _to_str(self._dataset.data)
        return self._dataset.data[key]


class _Hdf4Group:
    """h5py-group-shaped view of a :class:`Group`."""

    def __init__(self, group: Group) -> None:
        self._group = group

    @property
    def attrs(self) -> dict:
        # Expose the NeXus class the way h5py would surface a group attribute.
        return {"NX_class": self._group.nxclass} if self._group.nxclass else {}

    def keys(self):
        return self._group.children.keys()

    def __iter__(self):
        return iter(self._group.children)

    def __contains__(self, name: str) -> bool:
        return name in self._group.children

    def __getitem__(self, name: str):
        try:
            child = self._group.children[name]
        except KeyError:
            raise KeyError(name) from None
        return _wrap(child)

    def get(self, name: str, default: Any = None):
        child = self._group.children.get(name)
        if child is None:
            return default
        return _wrap(child)


def _wrap(node):
    """Wrap a tree node in its h5py-shaped adapter."""
    if isinstance(node, Group):
        return _Hdf4Group(node)
    return _Hdf4Dataset(node)


def open_hdf4(path: str) -> _Hdf4Group:
    """Open an HDF4 NeXus file as an h5py-compatible read-only handle."""
    return _Hdf4Group(read_tree(path))
