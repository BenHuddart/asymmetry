"""Create and query a logbook of synthetic runs."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.data.logbook import Logbook


def _dataset(run_number: int, field: float, temp: float, comment: str) -> MuonDataset:
    time = np.linspace(0.0, 4.0, 100)
    asym = 18.0 * np.exp(-0.5 * time)
    err = np.full_like(time, 0.5)
    return MuonDataset(
        time=time,
        asymmetry=asym,
        error=err,
        metadata={
            "run_number": run_number,
            "title": f"Run {run_number}",
            "field": field,
            "temperature": temp,
            "comment": comment,
        },
    )


def main() -> None:
    logbook = Logbook()
    logbook.add(_dataset(3101, 30.0, 5.0, "ZF baseline"), tags=["zf", "calibration"])
    logbook.add(_dataset(3102, 200.0, 5.0, "LF sweep"), tags=["lf"])

    print("runs:", logbook.run_numbers)
    print("field=200G:", [e.run_number for e in logbook.filter(field=200.0)])
    print("search 'baseline':", [e.run_number for e in logbook.search("baseline")])

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "logbook.json"
        logbook.save(out)

        restored = Logbook()
        restored.load_metadata(out)
        print("restored runs:", restored.run_numbers)


if __name__ == "__main__":
    main()
