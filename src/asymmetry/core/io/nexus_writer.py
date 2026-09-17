"""Minimal ISIS muon NeXus V1 writer for synthetic and derived runs.

Writes a fresh HDF5 file containing exactly the layout
:class:`~asymmetry.core.io.nexus.NexusLoader` consumes for legacy ``/run``
files (see ``docs/porting/simulate-mode/implementation-options.md`` for the
field table), plus a ``/run/simulation`` provenance group that the loader
surfaces through ``metadata["nexus_fields"]``. Unlike WiMDA's template-copy
save, the file is built from the :class:`Run` alone, so any loaded run — PSI
``.bin``/``.mdu`` included — can act as the instrument template, and no stale
sample logs leak into the synthetic file.

Notes on the V1 contract (verified against ``nexus.py``):

* Time zero is written the way ISIS writes it: a **1-based** ``t0_bin``
  attribute on the ``counts`` dataset (one value per detector, so PSI-style
  staggered t0 survives) plus ``histogram_data_1/time_zero`` in **microseconds**
  from the start of acquisition. ``first_good_bin`` / ``last_good_bin`` are
  1-based inclusive attributes on the same dataset. A round trip through
  :class:`~asymmetry.core.io.nexus.NexusLoader` is the identity.
* The file carries *either* the run's file t0 throughout or its policy-resolved
  effective t0 throughout (``effective=True``) — never a mix (F7).
* The loader treats the two lowest group ids in ``grouping`` as forward and
  backward, so the writer renumbers the run's forward group to 1 and
  backward to 2 (further groups follow in stable order).
* ISIS NeXus V1 has no balance-factor field, so α does **not** survive a
  round trip — the generating α is recorded in ``/run/simulation`` for
  reference only, exactly as with real data where α is an analysis-side
  calibration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from asymmetry.core.data.dataset import Run
from asymmetry.core.transform.grouping import resolve_group_indices
from asymmetry.core.transform.t0 import effective_detector_t0_bins

try:
    import h5py  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - exercised only without the hdf5 extra
    h5py = None


def _write_str(group: Any, name: str, value: str) -> None:
    # UTF-8 bytes: titles/labels may carry non-ASCII (e.g. the ×factor badge).
    group.create_dataset(name, data=np.bytes_(str(value).encode("utf-8")))


def _detector_group_ids(grouping: dict, n_detectors: int) -> np.ndarray:
    """Per-detector group-id array with forward → 1 and backward → 2."""
    groups = grouping.get("groups")
    ids = np.zeros(n_detectors, dtype=np.int32)
    if not isinstance(groups, dict) or not groups:
        # No grouping recorded: loader falls back to a half/half split.
        return ids

    forward_gid = int(grouping.get("forward_group", 1))
    backward_gid = int(grouping.get("backward_group", 2))
    ordered: list[int] = []
    for key in groups:
        try:
            gid = int(key)
        except (TypeError, ValueError):
            continue
        if gid not in ordered:
            ordered.append(gid)

    id_map: dict[int, int] = {}
    if forward_gid in ordered:
        id_map[forward_gid] = 1
    if backward_gid in ordered:
        id_map[backward_gid] = 2
    next_id = 3
    for gid in ordered:
        if gid not in id_map:
            id_map[gid] = next_id
            next_id += 1

    for gid, mapped in id_map.items():
        for det in resolve_group_indices(groups, gid):
            if 0 <= det < n_detectors and ids[det] == 0:
                ids[det] = mapped
    return ids


def _provenance_items(metadata: dict) -> dict[str, Any]:
    """Flatten simulation/degrade provenance into writable scalar fields."""
    items: dict[str, Any] = {}
    for source_key in ("simulation", "degraded"):
        info = metadata.get(source_key)
        if not isinstance(info, dict):
            continue
        for key, value in info.items():
            name = key if source_key == "simulation" else f"degrade_{key}"
            if isinstance(value, dict):
                items[name] = json.dumps(value, sort_keys=True)
            elif isinstance(value, (int, float, np.integer, np.floating)):
                items[name] = value
            else:
                items[name] = str(value)
    return items


def write_nexus_v1(run: Run, path: str | Path, *, effective: bool = False) -> None:
    """Write *run* as a loadable ISIS muon NeXus V1 (HDF5) file.

    The file reloads through :func:`asymmetry.core.io.load` with identical
    per-detector counts, bin width, per-detector t0 bins, exact t0, good-bin
    window, grouping (forward/backward renumbered to 1/2) and good frames.
    Raises :class:`ValueError` for runs without histograms or with ragged
    histogram lengths, and :class:`ImportError` when ``h5py`` is unavailable.

    With ``effective=True`` every t0-derived field — the per-detector bins, the
    exact ``time_zero``, the good window and ``corrected_time`` — comes from the
    grouping's policy-resolved t0 instead of the file's, so the written run is
    the analysed one. The two sets are never mixed.
    """
    if h5py is None:
        raise ImportError(
            "h5py is required to write NeXus files. Install with "
            "'pip install h5py' or 'pip install asymmetry[hdf5]'."
        )
    if not run.histograms:
        raise ValueError("Cannot write a NeXus file for a run without histograms.")

    lengths = {hist.n_bins for hist in run.histograms}
    if len(lengths) != 1:
        raise ValueError("All detector histograms must have the same length.")
    n_bins = lengths.pop()
    n_det = len(run.histograms)

    grouping = run.grouping if isinstance(run.grouping, dict) else {}
    metadata = run.metadata if isinstance(run.metadata, dict) else {}

    bin_width = float(run.histograms[0].bin_width)
    if effective:
        t0_bins = np.asarray(effective_detector_t0_bins(run.histograms, grouping), dtype=np.int64)
        # A policy shift is a whole number of bins (D4), so the sub-bin part of
        # each detector's exact t0 rides along with it.
        time_zero = np.asarray(
            [
                hist.t0_time_us_effective + (int(bin_) - int(hist.t0_bin)) * bin_width
                for hist, bin_ in zip(run.histograms, t0_bins, strict=True)
            ],
            dtype=np.float64,
        )
        first_good = int(grouping.get("first_good_bin", 0))
        last_good = int(grouping.get("last_good_bin", n_bins - 1))
    else:
        t0_bins = np.asarray([int(hist.t0_bin) for hist in run.histograms], dtype=np.int64)
        time_zero = np.asarray(
            [hist.t0_time_us_effective for hist in run.histograms], dtype=np.float64
        )
        first_good = max(int(hist.good_bin_start) for hist in run.histograms)
        last_good = min(int(hist.good_bin_end) for hist in run.histograms)
    # ``corrected_time`` is a single run-level axis, so it is stamped from the
    # common (latest) detector t0 — the bin reduction aligns the groups onto.
    common_time_zero = float(time_zero[int(np.argmax(t0_bins))])

    counts = np.vstack([np.asarray(hist.counts, dtype=np.float64) for hist in run.histograms])
    counts = np.clip(np.rint(counts), 0, None)
    counts_dtype = np.int32 if counts.max(initial=0.0) <= np.iinfo(np.int32).max else np.int64

    dead_time = np.asarray(grouping.get("dead_time_us", []), dtype=np.float64)
    if dead_time.size != n_det:
        dead_time = np.zeros(n_det, dtype=np.float64)

    try:
        good_frames = float(grouping.get("good_frames", 1.0))
    except (TypeError, ValueError):
        good_frames = 1.0

    with h5py.File(str(path), "w") as handle:
        entry = handle.create_group("run")
        _write_str(entry, "analysis", "muonTD")
        entry.create_dataset("IDF_version", data=np.int32(1))
        entry.create_dataset("number", data=np.int32(run.run_number))
        _write_str(entry, "title", metadata.get("title", ""))
        _write_str(entry, "notes", metadata.get("comment", ""))
        for key, name in (("started", "start_time"), ("stopped", "stop_time")):
            value = metadata.get(key)
            if value:
                _write_str(entry, name, str(value))
        entry.create_dataset("good_frames", data=np.float64(good_frames))

        instrument = entry.create_group("instrument")
        _write_str(instrument, "name", metadata.get("instrument", ""))
        detector = instrument.create_group("detector")
        orientation = metadata.get("detector_orientation")
        if orientation:
            _write_str(detector, "orientation", str(orientation))

        sample = entry.create_group("sample")
        sample.create_dataset("temperature", data=np.float64(metadata.get("temperature", 0.0)))
        sample.create_dataset("magnetic_field", data=np.float64(metadata.get("field", 0.0)))
        field_state = metadata.get("field_state")
        if field_state:
            _write_str(sample, "magnetic_field_state", str(field_state))

        h_data = entry.create_group("histogram_data_1")
        counts_ds = h_data.create_dataset("counts", data=counts.astype(counts_dtype))
        # ISIS bin metadata is 1-based and inclusive
        # (docs/porting/t0-determination/isis-header-index-base.md).
        counts_ds.attrs["t0_bin"] = (t0_bins + 1).astype(np.int32)
        counts_ds.attrs["first_good_bin"] = np.int32(first_good + 1)
        counts_ds.attrs["last_good_bin"] = np.int32(last_good + 1)
        h_data.create_dataset(
            "corrected_time",
            data=(np.arange(n_bins, dtype=np.float64) + 0.5) * bin_width - common_time_zero,
        )
        h_data.create_dataset("grouping", data=_detector_group_ids(grouping, n_det))
        h_data.create_dataset("dead_time", data=dead_time)
        # time_zero is the exact t0 in µs from the start of acquisition, one
        # value per detector (WiMDA quantised a single value to whole µs here).
        time_zero_ds = h_data.create_dataset("time_zero", data=time_zero)
        time_zero_ds.attrs["units"] = np.bytes_(b"microseconds")

        provenance = _provenance_items(metadata)
        if metadata.get("synthetic") or provenance:
            sim = entry.create_group("simulation")
            sim.create_dataset("synthetic", data=np.int32(1 if metadata.get("synthetic") else 0))
            for name, value in sorted(provenance.items()):
                if isinstance(value, str):
                    _write_str(sim, name, value)
                elif isinstance(value, (int, np.integer)):
                    sim.create_dataset(name, data=np.int64(value))
                else:
                    sim.create_dataset(name, data=np.float64(value))
