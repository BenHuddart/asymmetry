"""Survey the ISIS NeXus t0 header convention across a corpus (study scaffolding).

For every file listed (one path per line in the file given as argv[1]) read
``time_zero`` (µs), ``resolution``, the ``t0_bin`` / ``first_good_bin`` /
``last_good_bin`` attributes on ``counts`` and the summed pulse, then report
which integer rule maps ``time_zero / resolution`` onto ``t0_bin`` and how far
the detected pulse rise sits from the header t0. Evidence and conclusions are
recorded in ``docs/porting/t0-determination/isis-header-index-base.md``.

Usage::

    .venv/bin/python tests/porting/t0-determination/isis_t0_header_survey.py files.txt

Reads HDF5 (v2, v1-hdf5) with h5py and HDF4 v1 through
``asymmetry.core.io.hdf4``. Prints only instrument prefixes, eras and bin
values — no paths, titles or run numbers — so the output is safe to paste into
the study.
"""

from __future__ import annotations

import collections
import math
import os
import statistics
import sys

import h5py
import numpy as np

from asymmetry.core.io.hdf4 import is_hdf4, open_hdf4


def _get(group, path):
    for part in path.split("/"):
        if part not in group:
            return None
        group = group[part]
    return group


def _scalar_attr(node, name):
    value = node.attrs.get(name)
    if value is None:
        return None
    flat = np.asarray(value).ravel()
    if flat.dtype.kind in "SU":
        text = flat[0].decode() if isinstance(flat[0], bytes) else str(flat[0])
        try:
            return float(text)
        except ValueError:
            return None
    return float(flat[0])


def _pulse(counts):
    c = counts.astype(float)
    peak = int(np.argmax(c))
    half = c[peak] / 2.0
    below = np.flatnonzero(c[: peak + 1] < half)
    if below.size == 0:
        return peak, float("nan")
    x = int(below[-1]) + 1
    frac = 0.5 if c[x] == c[x - 1] else (half - c[x - 1]) / (c[x] - c[x - 1])
    return peak, x - 1 + frac


def _fields(path):
    if is_hdf4(path):
        root, version = open_hdf4(path), "v1-hdf4"
    else:
        root, version = h5py.File(path, "r"), None
    if "raw_data_1" in root:
        version = version or "v2"
        h = _get(root, "raw_data_1/instrument/detector_1") or _get(root, "raw_data_1/detector_1")
    else:
        version = version or "v1-hdf5"
        h = _get(root, "run/histogram_data_1")
    counts = h["counts"]
    summed = np.asarray(counts)
    summed = summed.reshape(-1, summed.shape[-1]).sum(axis=0)
    return {
        "version": version,
        "instrument": os.path.basename(path)[:4].upper(),
        "resolution_us": float(np.asarray(h["resolution"]).ravel()[0]) * 1e-6,
        "time_zero_us": float(np.asarray(h["time_zero"]).ravel()[0]),
        "t0_bin": _scalar_attr(counts, "t0_bin"),
        "first_good_bin": _scalar_attr(counts, "first_good_bin"),
        "last_good_bin": _scalar_attr(counts, "last_good_bin"),
        "n_bins": int(summed.size),
        "pulse": _pulse(summed),
    }


def main(list_path: str) -> None:
    rows = []
    errors: collections.Counter[str] = collections.Counter()
    for line in open(list_path, encoding="utf-8"):
        path = line.strip()
        if not path:
            continue
        try:
            d = _fields(path)
        except Exception as exc:  # noqa: BLE001 - survey tool, keep going
            errors[type(exc).__name__] += 1
            continue
        if d["t0_bin"] is None:
            errors["missing-t0_bin"] += 1
            continue
        rows.append(d)
    print(f"files parsed: {len(rows)}  errors: {dict(errors)}")

    rule: collections.Counter = collections.Counter()
    lgb: collections.Counter = collections.Counter()
    offsets: dict = collections.defaultdict(collections.Counter)
    rise: dict = collections.defaultdict(list)
    for d in rows:
        q = d["time_zero_us"] / d["resolution_us"]
        t0b = int(d["t0_bin"])
        frac = q - math.floor(q)
        if 0.05 < frac < 0.95:
            fit = "floor+1" if t0b == math.floor(q) + 1 else "floor" if t0b == math.floor(q) else "round" if t0b == round(q) else "other"
            rule[(d["instrument"], fit)] += 1
        if d["last_good_bin"] is not None:
            lgb[(d["instrument"], int(d["last_good_bin"]) == d["n_bins"])] += 1
        if d["first_good_bin"] is not None:
            offsets[d["instrument"]][int(d["first_good_bin"]) - t0b] += 1
        _, mid = d["pulse"]
        if not math.isnan(mid):
            rise[(d["instrument"], d["version"], d["resolution_us"])].append((mid + 0.5) - q)

    print("\nt0_bin rule on files with a clearly fractional time_zero/resolution (instrument, rule): n")
    for key, n in sorted(rule.items()):
        print(f"  {key}: {n}")
    print("\nlast_good_bin == n_bins (instrument, equal): n")
    for key, n in sorted(lgb.items()):
        print(f"  {key}: {n}")
    print("\nfirst_good_bin - t0_bin per instrument:")
    for inst, counter in sorted(offsets.items()):
        print(f"  {inst}: {dict(sorted(counter.items()))}")
    print("\nrise midpoint (edge units) - time_zero/resolution, bins: (instrument, version, res) n mean sd min max")
    for key, values in sorted(rise.items()):
        print(
            f"  {key}: {len(values)} {statistics.mean(values):+.2f} "
            f"{statistics.pstdev(values):.2f} {min(values):+.2f} {max(values):+.2f}"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
