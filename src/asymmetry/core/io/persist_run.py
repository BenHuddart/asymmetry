"""Lossless, verified file backing for a single in-memory histogram acquisition."""

from __future__ import annotations

import os
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from asymmetry.core.data.dataset import Run
from asymmetry.core.io import load
from asymmetry.core.io.nexus_writer import write_nexus_v1


def needs_histogram_file(run: Run) -> bool:
    """A locator alone cannot establish that memory-only counts were captured."""
    return bool(run.metadata.get("synthetic") or run.metadata.get("degraded")) and (
        not run.source_file or run.source_binding.observed is None
    )


def persist_histogram_run(run: Run, path: str | Path, *, checkpoint=lambda: None) -> Run:
    """Atomically write raw counts, verify a real load, and return detached source facts.

    The V1 writer represents one integer-count acquisition, without supplied
    variances or multiple periods. Unsupported observations must never acquire
    a file identity from a rounded, clipped or incomplete representation.
    """
    checkpoint()
    if (
        not run.histograms
        or run.grouping.get("period_histograms")
        or run.grouping.get("period_inputs")
    ):
        raise ValueError("Save as NeXus requires a single-period histogram run")
    for hist in run.histograms:
        checkpoint()
        counts = np.asarray(hist.counts)
        if hist.count_variance is not None or not (
            np.all(np.isfinite(counts))
            and np.all(counts >= 0)
            and np.all(counts <= 2**53)
            and np.array_equal(counts, np.rint(counts))
        ):
            raise ValueError("Save as NeXus requires non-negative integer counts without variance")
    destination = Path(path).absolute()
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.stem}-", suffix=".nxs", dir=destination.parent
    )
    os.close(fd)
    try:
        write_nexus_v1(run, temporary)
        checkpoint()
        decoded = load(temporary, checkpoint=checkpoint)
        if isinstance(decoded, list) or decoded.run is None:
            raise ValueError("Saved NeXus file did not reload a single histogram run")
        captured = decoded.run
        if captured.run_number != run.run_number:
            raise ValueError("Saved NeXus file changed the run number")
        if captured.source_binding.observed is None:
            raise ValueError("Saved NeXus file could not be verified")
        if len(captured.histograms) != len(run.histograms):
            raise ValueError("Saved NeXus file changed the acquired detector count")
        for original, restored in zip(run.histograms, captured.histograms, strict=True):
            checkpoint()
            if not (
                np.array_equal(original.counts, restored.counts)
                and original.bin_width == restored.bin_width
                and original.t0_bin == restored.t0_bin
                and original.t0_time_us_effective == restored.t0_time_us_effective
                and original.good_bin_start == restored.good_bin_start
                and original.good_bin_end == restored.good_bin_end
            ):
                raise ValueError("Saved NeXus file cannot preserve the acquired counts and timing")
        checkpoint()
        os.replace(temporary, destination)
        return replace(run, source_file=str(destination), source_binding=captured.source_binding)
    finally:
        Path(temporary).unlink(missing_ok=True)
