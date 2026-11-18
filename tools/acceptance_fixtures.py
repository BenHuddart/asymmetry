"""Write original integer-count fixtures using the existing ROOT/HDF4 writer contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def write_fixtures(directory: Path) -> dict:
    """Create tiny actual containers; neither the app nor frozen code imports this writer."""
    import pyhdf.V  # noqa: F401
    import uproot
    from pyhdf.HDF import HC, HDF
    from pyhdf.SD import SD, SDC

    directory.mkdir(parents=True, exist_ok=False)
    counts = np.array([120 * np.arange(1, 21), 80 * np.arange(1, 21)], dtype=np.int32)
    root_path = directory / "counts.root"
    with uproot.recreate(root_path) as root:
        for key, value in {
            "Run Number": "4451",
            "Run Title": "Original acceptance counts",
            "No of Histos": "2",
            "Time Resolution": "125 ns",
        }.items():
            root[f"RunHeader/RunInfo/{key}"] = value
        for index, label in enumerate(("Forward", "Backward"), 1):
            prefix = f"RunHeader/DetectorInfo/Detector{index:03d}"
            for key, value in {
                "Name": label,
                "Histo Number": str(index),
                "Time Zero Bin": "2",
                "First Good Bin": "3",
                "Last Good Bin": "18",
            }.items():
                root[f"{prefix}/{key}"] = value
            root[f"histos/DecayAnaModule/hDecay{index:03d}"] = (
                counts[index - 1].astype(float),
                np.arange(21, dtype=float) - 0.5,
            )

    path = directory / "counts.nxs"
    hdf = HDF(str(path), HC.WRITE | HC.CREATE)
    groups = hdf.vgstart()
    sd = SD(str(path), SDC.WRITE | SDC.CREATE)
    try:

        def number(parent, name, values, dtype, attrs=()):
            values = np.asarray(values, dtype=np.int32 if dtype == SDC.INT32 else np.float64)
            dataset = sd.create(name, dtype, values.shape)
            dataset[:] = values
            for key, value in attrs:
                dataset.attr(key).set(SDC.CHAR8 if isinstance(value, str) else SDC.INT32, value)
            parent.add(HC.DFTAG_NDG, dataset.ref())
            dataset.endaccess()

        def text(parent, name, value):
            dataset = sd.create(name, SDC.CHAR8, len(value))
            dataset[:] = value
            parent.add(HC.DFTAG_NDG, dataset.ref())
            dataset.endaccess()

        run = groups.create("run")
        run._class = "NXentry"
        text(run, "analysis", "muonTD")
        text(run, "title", "Original acceptance counts")
        number(run, "number", [4452], SDC.INT32)
        hist = groups.create("histogram_data_1")
        hist._class = "NXdata"
        number(
            hist,
            "counts",
            counts,
            SDC.INT32,
            (("t0_bin", 3), ("first_good_bin", 4), ("last_good_bin", 19)),
        )
        number(hist, "corrected_time", (np.arange(20) - 2) * 0.125, SDC.FLOAT64)
        number(hist, "resolution", [125000], SDC.INT32, (("units", "picoseconds"),))
        number(hist, "time_zero", [0.3125], SDC.FLOAT64, (("units", "microseconds"),))
        number(hist, "grouping", [1, 2], SDC.INT32)
        run.insert(hist)
        hist.detach()
        run.detach()
    finally:
        groups.end()
        sd.end()
        hdf.close()
    manifest = {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in ("counts.root", "counts.nxs")
    }
    (directory / "fixtures.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    write_fixtures(parser.parse_args().directory)
