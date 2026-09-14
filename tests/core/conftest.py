"""A synthetic folder of runs for the workflow-façade and CLI tests.

Built once per session from :mod:`asymmetry.core.simulate` and written as ISIS
NeXus V1 with :mod:`asymmetry.core.io.nexus_writer`, so the tests exercise the
real loader path without any research data in the repository. The folder holds:

* one weak transverse-field calibration run at 100 G, simulated with a
  detector balance of :data:`CALIBRATION_ALPHA` (≠ 1) built into the group
  efficiencies, so ``alpha`` has something real to recover;
* a five-run zero-field temperature scan whose relaxation rate rises with
  temperature along the known curve :func:`scan_rate`, so a fit of the scan can
  be checked against the physics it was built from;
* one deliberately broken zero-field run at the top of the scan
  (:data:`FLAT_RUN`) carrying no signal at all — a flat spectrum whose
  amplitude collapses and whose relaxation rate is undetermined, so the
  quality flags a series fit raises have something to fire on.

Every run is stamped ``field_state = "TF"``, including the zero-field ones —
that is what ISIS files actually do to a scan that opened with a weak-TF
calibration, and the survey's geometry rule has to see past it.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from asymmetry.core.io.nexus_writer import write_nexus_v1
from asymmetry.core.simulate import BUILTIN_TEMPLATES, simulate_run

#: Detector balance simulated into the calibration run.
CALIBRATION_ALPHA = 1.25

#: Run number of the weak-TF calibration run.
CALIBRATION_RUN = 101

#: Run numbers of the zero-field temperature scan, and their temperatures.
SCAN_RUNS = (102, 103, 104, 105, 106)
SCAN_TEMPERATURES = (10.0, 20.0, 30.0, 40.0, 50.0)

#: The broken run: zero-field, at the top of the scan, with no signal at all.
FLAT_RUN = 107
FLAT_TEMPERATURE = 60.0

#: Every run in the folder, in the order the survey lists them.
ALL_RUNS = (CALIBRATION_RUN, *SCAN_RUNS, FLAT_RUN)

#: The zero-field temperature scan including its broken member.
ZF_RUNS = (*SCAN_RUNS, FLAT_RUN)
ZF_TEMPERATURES = (*SCAN_TEMPERATURES, FLAT_TEMPERATURE)

#: The one scan run whose file carries non-zero per-detector deadtime values.
DEADTIME_RUN = 104
DEADTIME_US = 0.012

#: Transverse field of the calibration run, in gauss, and its precession
#: frequency (γ_μ/2π = 0.01355 MHz/G). 100 G puts ~11 cycles inside the
#: good-bin window, so the oscillation averages away and the integral-ratio
#: alpha estimate sees the simulated detector balance rather than a residual
#: part-cycle.
CALIBRATION_FIELD_G = 100.0
_CALIBRATION_FREQ_MHZ = 0.01355 * CALIBRATION_FIELD_G

_TOTAL_EVENTS = 2.0e6
_A0_PERCENT = 20.0

#: A small pulsed-source spectrometer: 8 detectors in two groups of four,
#: 600 bins of 16 ns. Small enough that six files build in well under a
#: second, structured exactly like the ISIS templates. ``good_frames`` is the
#: deadtime normaliser (``N·τ/(Δt·frames)``); an ISIS-scale count keeps the
#: ``from_file`` correction a few-percent perturbation instead of the
#: clipped, asymmetry-preserving blow-up a frame count of 1 would give.
_TEMPLATE = dataclasses.replace(
    BUILTIN_TEMPLATES["ideal_pulsed_fb"],
    key="test_pulsed_fb",
    n_detectors=8,
    n_bins=600,
    forward_detectors=(1, 2, 3, 4),
    backward_detectors=(5, 6, 7, 8),
    good_frames=40000.0,
    instrument_name="SIM",
)


def _calibration_signal(time_us: np.ndarray) -> np.ndarray:
    """A weak-TF precession signal in percent."""
    return (
        _A0_PERCENT
        * np.cos(2.0 * np.pi * _CALIBRATION_FREQ_MHZ * time_us)
        * np.exp(-0.02 * time_us)
    )


def _relaxation_signal(rate_per_us: float):
    """An exponentially relaxing zero-field signal in percent."""

    def signal(time_us: np.ndarray) -> np.ndarray:
        return _A0_PERCENT * np.exp(-rate_per_us * time_us)

    return signal


def scan_rate(temperature: float) -> float:
    """The relaxation rate the scan run at *temperature* was simulated with."""
    return 0.10 + 0.004 * temperature


def _flat_signal(time_us: np.ndarray) -> np.ndarray:
    """No signal at all — the broken run's spectrum."""
    return np.zeros_like(time_us)


def _template_for(*, temperature: float, field: float, title: str):
    template = _TEMPLATE.build()
    template.metadata.update(
        {
            "temperature": temperature,
            "field": field,
            # ISIS stamps TF on every run of a scan that opened with a
            # weak-TF calibration, zero-field runs included.
            "field_state": "TF",
            "title": title,
            # A detector-bank orientation, so the survey's reading of it has
            # something to report. (The run's free-text note is stamped in
            # :func:`_write`; ``comment`` is not an inherited template key.)
            "detector_orientation": "L",
        }
    )
    return template


def _write(run, folder: Path, *, started: str, stopped: str, deadtime_us: float) -> None:
    run.metadata["started"] = started
    run.metadata["stopped"] = stopped
    # The writer puts ``comment`` into the NeXus ``notes`` node, which the
    # loaders read back as ``comment`` — the run's free-text note.
    run.metadata["comment"] = f"simulated {run.metadata['title']}"
    run.grouping["dead_time_us"] = [deadtime_us] * len(run.histograms)
    write_nexus_v1(run, folder / f"SIM{run.run_number:08d}.nxs")


@pytest.fixture(scope="session")
def workflow_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A directory of synthetic NeXus runs: one TF calibration plus a ZF scan."""
    pytest.importorskip("h5py")
    folder = tmp_path_factory.mktemp("workflow_runs")

    calibration = simulate_run(
        _template_for(
            temperature=5.0,
            field=CALIBRATION_FIELD_G,
            title=f"Calibrant T=5.0 K B={CALIBRATION_FIELD_G} G",
        ),
        _calibration_signal,
        total_events=_TOTAL_EVENTS,
        seed=1,
        alpha=CALIBRATION_ALPHA,
        run_number=CALIBRATION_RUN,
        title=f"Calibrant T=5.0 K B={CALIBRATION_FIELD_G} G",
    )
    _write(
        calibration,
        folder,
        started="2024-03-01T09:00:00",
        stopped="2024-03-01T09:30:00",
        deadtime_us=0.0,
    )

    for index, (run_number, temperature) in enumerate(zip(SCAN_RUNS, SCAN_TEMPERATURES)):
        title = f"Sample T={temperature} K B=0.0 G"
        run = simulate_run(
            _template_for(temperature=temperature, field=0.0, title=title),
            _relaxation_signal(scan_rate(temperature)),
            total_events=_TOTAL_EVENTS,
            seed=10 + index,
            alpha=1.0,
            run_number=run_number,
            title=title,
        )
        _write(
            run,
            folder,
            started=f"2024-03-01T1{index}:00:00",
            stopped=f"2024-03-01T1{index}:20:00",
            deadtime_us=DEADTIME_US if run_number == DEADTIME_RUN else 0.0,
        )

    flat_title = f"Sample T={FLAT_TEMPERATURE} K B=0.0 G"
    flat = simulate_run(
        _template_for(temperature=FLAT_TEMPERATURE, field=0.0, title=flat_title),
        _flat_signal,
        total_events=_TOTAL_EVENTS,
        seed=20,
        alpha=1.0,
        run_number=FLAT_RUN,
        title=flat_title,
    )
    _write(
        flat,
        folder,
        started="2024-03-01T15:00:00",
        stopped="2024-03-01T15:20:00",
        deadtime_us=0.0,
    )

    return folder


@pytest.fixture(scope="session")
def reduced_workdir(workflow_folder: Path, tmp_path_factory: pytest.TempPathFactory):
    """A work directory with every zero-field run reduced under the defaults.

    The screening, fitting and series tests all start from reduced spectra —
    that is how the commands themselves reach their data — so the reduction is
    done once here rather than in each test.
    """
    from asymmetry.core.io import load
    from asymmetry.core.workflow.reduction import (
        ReductionSettings,
        reduce_run,
        resolve_reduction_grouping,
    )
    from asymmetry.core.workflow.survey import build_run_row
    from asymmetry.core.workflow.workdir import ReducedEntry, WorkDir, reduction_digest

    workdir = WorkDir(tmp_path_factory.mktemp("workflow_workdir"))
    workdir.ensure()
    settings = ReductionSettings()
    for run_number in ZF_RUNS:
        path = workflow_folder / f"SIM{run_number:08d}.nxs"
        source = load(str(path)).run
        grouping = resolve_reduction_grouping(source, settings)
        dataset = reduce_run(source, settings)
        workdir.write_reduced(
            dataset,
            ReducedEntry(
                run_number=run_number,
                digest=reduction_digest(source_file=path, grouping=grouping, settings=settings),
                source_file=str(path),
                n_points=dataset.n_points,
                settings=settings,
                run=build_run_row(
                    dataset, path=path, prefix="SIM", run_number=run_number
                ).to_dict(),
                alpha=float(grouping["alpha"]),
                deadtime_mode=str(grouping["deadtime_mode"]),
                forward_group=int(grouping["forward_group"]),
                backward_group=int(grouping["backward_group"]),
            ),
        )
    return workdir


@pytest.fixture(scope="session")
def first_scan_run() -> int:
    """The lowest-temperature run of the zero-field scan."""
    return SCAN_RUNS[0]
