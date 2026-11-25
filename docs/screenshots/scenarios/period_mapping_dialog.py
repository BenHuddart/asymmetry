"""Period → red/green/ignore mapping dialog on a two-period silicon run.

Companion screenshot to :doc:`/workflows/photomusr_silicon_periods`.
Two synthetic stored periods each carry 28,108 good frames and identical
file-declared memberships. The first maps to **Red** and the second to **Green**.

The rows are stored data indices; no dwell rows are synthesised. Red-only
mapping is checked through core preparation while this default two-set dialog
retains the same controls. Reduction uses each acquisition's correction and
clock before pooling; the rows retain their acquired good-frame counts.
The dialog receives deterministic synthetic V1 data loaded through the real
period-major grouping decoder. No fit runs in this scenario.
"""

from __future__ import annotations

from PySide6.QtWidgets import QWidget

from ..data import make_silicon_photomusr_periods
from ._base import Scenario, register


class PeriodMappingDialogScenario(Scenario):
    name = "period_mapping_dialog"
    description = (
        "Period mapping dialog on a two-period silicon photo-μSR run, with the "
        "first period mapped to Red and the second to Green."
    )
    size = (640, 150)

    def build(self) -> QWidget:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        import h5py
        import numpy as np

        from asymmetry.core.io import load, select_period
        from asymmetry.gui.windows.period_mapping_dialog import PeriodMappingDialog

        sources = make_silicon_photomusr_periods()
        runs = [source.run for source in sources]
        histograms = [histogram for run in runs for histogram in run.histograms]
        anchor = histograms[0]
        clocks = {(h.bin_width, h.t0_bin, h.good_bin_start, h.good_bin_end) for h in histograms}
        if len(clocks) != 1:
            raise RuntimeError("The synthetic V1 scenario requires one shared acquisition clock")
        tables = np.zeros((len(runs), len(runs[0].histograms)), dtype=int)
        for row, run in zip(tables, runs, strict=True):
            for gid, members in run.grouping["groups"].items():
                row[np.asarray(members) - 1] = gid
        # V1 histogram ordinals include both switching states; rows share memberships.
        with TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic-periods.nxs"
            with h5py.File(path, "w") as handle:
                entry = handle.create_group("run")
                entry.create_dataset("analysis", data=b"muonTD")
                entry.create_dataset("number", data=runs[0].metadata["source_run_number"])
                entry.create_dataset("switching_states", data=2)
                entry.create_dataset(
                    "good_frames", data=[run.grouping["good_frames"] for run in runs]
                )
                detector = entry.create_group("histogram_data_1")
                counts = detector.create_dataset(
                    "counts", data=np.stack([h.counts for h in histograms])
                )
                counts.attrs.update(
                    t0_bin=anchor.t0_bin + 1,
                    first_good_bin=anchor.good_bin_start + 1,
                    last_good_bin=anchor.good_bin_end + 1,
                )
                detector.create_dataset("resolution", data=anchor.bin_width * 1e6).attrs[
                    "units"
                ] = "ps"
                detector.create_dataset("grouping", data=tables.ravel())
            combined = load(path)
            periods = [select_period(combined, number) for number in (1, 2)]
        if any(
            period.run.grouping["groups"] != source.run.grouping["groups"]
            for period, source in zip(periods, sources, strict=True)
        ):
            raise RuntimeError("Loaded period-major declarations must retain detector membership")
        return PeriodMappingDialog(periods)


register(PeriodMappingDialogScenario())
