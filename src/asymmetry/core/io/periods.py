"""Period (red/green) selection for multi-period muon data.

Pulsed-source muon runs can be recorded in *period mode*: a single ``.nxs``
file holds several period histograms (for example light-OFF / light-ON in a
photo-µSR experiment, RF-on / RF-off, or ALC steps). The NeXus loader keeps
the raw per-period histograms on the loaded :class:`~asymmetry.core.data.dataset.Run`
(see :mod:`asymmetry.core.io.nexus`); this module is the GUI-free, scriptable
way to *select* a single period from a loaded run.

Two public layers live here:

* **High level** — :func:`select_period` returns a fully reduced
  :class:`MuonDataset` for one period of a loaded run, preserving provenance.
  It is what scripts (and :func:`asymmetry.core.io.load` via ``period=``) use.
* **Low level** — :func:`select_period_histograms` and
  :func:`combine_period_asymmetry` hold the histogram-selection and the
  green/red combination arithmetic. The GUI grouping/reduction path calls
  these so there is a single implementation of the rule rather than a copy in
  the desktop app.

Period conventions
------------------
Periods are numbered from ``1`` (matching ``metadata['period_number']``).
For the common two-period case the first period (index 0) is labelled
``"red"`` and the second (index 1) ``"green"`` — the same convention the
loader and the GUI "RG box" use. In a photo-µSR experiment the **light-OFF**
spectrum is conventionally *Green* (period 2) and **light-ON** is *Red*
(period 1); confirm this against the relaxation for a given instrument.

This module must stay free of Qt / matplotlib / ``asymmetry.gui`` imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.transform.grouping import good_frames
from asymmetry.core.transform.integral import (
    _X_LABELS,
    FieldScan,
    FieldScanPoint,
    _excluded_run_number,
    _order_value,
    integrate_curve,
)
from asymmetry.core.utils.constants import ORDER_KEYS, PeriodMode

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.typing import NDArray

__all__ = [
    "RED_INDEX",
    "GREEN_INDEX",
    "PERIOD_MAPPING_TARGETS",
    "PeriodMode",
    "build_rf_difference_scan",
    "combine_mapped_periods",
    "combine_period_asymmetry",
    "encode_period_run_number",
    "normalise_period_mapping",
    "period_count",
    "period_labels",
    "resolve_period_index",
    "select_period",
    "select_period_histograms",
    "sum_period_histograms",
]

#: Histogram index of the first ("red") period in a two-period file.
RED_INDEX = 0
#: Histogram index of the second ("green") period in a two-period file.
GREEN_INDEX = 1


def encode_period_run_number(run_number: int, period_number: int) -> int:
    """Return a stable, unique integer run-number key for one period row.

    The data browser keys datasets by integer run number, so the two periods
    of a run must not share the source run's number or the second add would
    overwrite (collapse) the first. This encodes ``run * 1000 + period`` — the
    same scheme the loader's 3+-period list path uses in
    ``NeXusReader._encode_period_run_number`` — while the user-facing
    ``run/period`` label and the true ``source_run_number`` are carried in
    metadata. Real ISIS/PSI run numbers are well under 1000×, so the encoded
    keys never clash with single-period runs.
    """
    return int(run_number) * 1000 + int(period_number)


# Friendly labels accepted for the two-period red/green case (lower-cased).
_RG_LABEL_INDEX: dict[str, int] = {
    "red": RED_INDEX,
    "r": RED_INDEX,
    str(PeriodMode.RED): RED_INDEX,
    "green": GREEN_INDEX,
    "g": GREEN_INDEX,
    str(PeriodMode.GREEN): GREEN_INDEX,
}


# --- introspection ------------------------------------------------------------


def period_count(data: MuonDataset | list[MuonDataset] | Run) -> int:
    """Return the number of periods reachable from ``data``.

    Accepts a loaded :class:`MuonDataset`, a ``list`` of datasets (as returned
    for 3+ period files), or a :class:`Run`.
    """
    if isinstance(data, list):
        return len(data)
    run = data if isinstance(data, Run) else data.run
    if run is not None and isinstance(run.grouping, dict):
        reduced = run.grouping.get("period_reduced")
        if isinstance(reduced, list) and reduced:
            return len(reduced)
    metadata = data.metadata if isinstance(data, (MuonDataset,)) else getattr(data, "metadata", {})
    try:
        return max(1, int(metadata.get("period_count", 1)))
    except (TypeError, ValueError):
        return 1


def period_labels(data: MuonDataset | list[MuonDataset] | Run) -> list[str]:
    """Return human-facing labels for each selectable period.

    Two-period runs return ``["red", "green"]``; otherwise the 1-based period
    numbers as strings.
    """
    count = period_count(data)
    if count == 2:
        return [str(PeriodMode.RED), str(PeriodMode.GREEN)]
    return [str(i) for i in range(1, count + 1)]


def resolve_period_index(period: int | str | PeriodMode, count: int) -> int:
    """Translate a user-supplied period selector to a 0-based index.

    ``period`` may be a 1-based integer period number, a numeric string, or —
    when ``count == 2`` — a ``"red"``/``"green"`` label (or :class:`PeriodMode`
    member). Raises :class:`ValueError` for unknown labels or out-of-range
    numbers and :class:`TypeError` for unsupported types.
    """
    if isinstance(period, bool):  # bool is an int subclass; reject explicitly
        raise TypeError(f"Period selector must be an int or str, not {type(period).__name__}")
    if isinstance(period, (int, np.integer)):
        return _index_from_number(int(period), count)
    if isinstance(period, str):
        key = period.strip().lower()
        if count == 2 and key in _RG_LABEL_INDEX:
            return _RG_LABEL_INDEX[key]
        try:
            number = int(key)
        except ValueError:
            allowed = "1.." + str(count)
            if count == 2:
                allowed += " or 'red'/'green'"
            raise ValueError(f"Unknown period selector {period!r}; expected {allowed}") from None
        return _index_from_number(number, count)
    raise TypeError(f"Period selector must be an int or str, not {type(period).__name__}")


def _index_from_number(number: int, count: int) -> int:
    if 1 <= number <= count:
        return number - 1
    raise ValueError(f"Period {number} out of range; this run has {count} period(s) (1..{count})")


# --- high-level selection -----------------------------------------------------


def select_period(
    data: MuonDataset | list[MuonDataset],
    period: int | str | PeriodMode,
) -> MuonDataset:
    """Return a reduced :class:`MuonDataset` for a single period.

    Parameters
    ----------
    data
        A loaded run: either a single :class:`MuonDataset` (single-period file,
        or a combined two-period file) or the ``list`` returned for 3+ period
        files.
    period
        Period selector — see :func:`resolve_period_index`.

    Returns
    -------
    MuonDataset
        The requested period with its own ``time``/``asymmetry``/``error`` and
        per-period provenance (``period_number``, ``run_label``, ``good_frames``,
        ``dead_time_us``). t0, good-bin window, grouping, field and temperature
        are preserved from the parent run.

    Notes
    -----
    The arrays returned are the loader's default reduction (``alpha = 1.0``,
    no deadtime/background correction), identical to what the loader produces
    for a single-period file.
    """
    if isinstance(data, list):
        index = resolve_period_index(period, len(data))
        return data[index]

    if not isinstance(data, MuonDataset):
        raise TypeError(f"select_period expects a MuonDataset or list, got {type(data).__name__}")

    grouping = data.run.grouping if data.run is not None else {}
    reduced = grouping.get("period_reduced") if isinstance(grouping, dict) else None

    if isinstance(reduced, list) and len(reduced) >= 2:
        index = resolve_period_index(period, len(reduced))
        return _build_period_dataset(data, index)

    # Single-period dataset: only period 1 is selectable.
    count = period_count(data)
    index = resolve_period_index(period, count)
    if count <= 1 and index == 0:
        return data
    raise ValueError(
        "Per-period histograms are unavailable for this dataset; reload the "
        "source file with asymmetry.core.io.load(...) to access individual periods"
    )


def _build_period_dataset(combined: MuonDataset, index: int) -> MuonDataset:
    """Construct a per-period dataset from a combined two-period run."""
    assert combined.run is not None  # guaranteed by caller
    grouping = combined.run.grouping
    reduced = grouping["period_reduced"]
    time, asymmetry, error = reduced[index]
    count = len(reduced)

    period_grouping = {
        key: value
        for key, value in grouping.items()
        if key not in {"period_histograms", "period_reduced", "period_mode"}
    }

    good_frames = grouping.get("period_good_frames")
    if isinstance(good_frames, list) and index < len(good_frames):
        try:
            period_grouping["good_frames"] = float(good_frames[index])
        except (TypeError, ValueError):
            pass

    dead_time = grouping.get("period_dead_time_us")
    if (
        isinstance(dead_time, list)
        and index < len(dead_time)
        and isinstance(dead_time[index], list)
    ):
        period_grouping["dead_time_us"] = [float(v) for v in dead_time[index]]

    period_histograms = grouping.get("period_histograms")
    if isinstance(period_histograms, list) and index < len(period_histograms):
        histograms = _clone_histograms(period_histograms[index])
    else:
        histograms = _clone_histograms(combined.run.histograms)

    metadata = dict(combined.metadata)
    source_run = metadata.get("source_run_number", metadata.get("run_number", 0))
    # Each period gets its own encoded run-number key so both survive in the
    # run-number-keyed data browser (otherwise the second add collapses the
    # first). The source run and the friendly run/period label stay in metadata.
    period_run_number = encode_period_run_number(source_run, index + 1)
    metadata["run_number"] = period_run_number
    metadata["source_run_number"] = source_run
    metadata["period_number"] = index + 1
    metadata["period_count"] = count
    metadata["run_label"] = f"{source_run}/{index + 1}"
    if count == 2:
        metadata["period_label"] = period_labels(combined)[index]

    run = Run(
        run_number=period_run_number,
        histograms=histograms,
        metadata=metadata,
        grouping=period_grouping,
        source_file=combined.run.source_file,
    )
    return MuonDataset(
        time=np.asarray(time, dtype=np.float64).copy(),
        asymmetry=np.asarray(asymmetry, dtype=np.float64).copy(),
        error=np.asarray(error, dtype=np.float64).copy(),
        metadata=metadata,
        run=run,
    )


# --- low-level helpers shared with the GUI reduction path ---------------------


def select_period_histograms(
    histograms: list[Histogram],
    grouping: dict,
    period_index: int,
) -> tuple[list[Histogram], dict]:
    """Return period-specific histograms plus effective grouping metadata.

    This is the single source of truth for "pick the histograms for period N
    and the good-frames / deadtime that go with them" used by both the
    scriptable API and the GUI grouping/reduction path. Falls back to a clone
    of ``histograms`` when period data is missing or malformed.
    """
    period_grouping = dict(grouping)

    period_good_frames = grouping.get("period_good_frames")
    if isinstance(period_good_frames, list) and period_index < len(period_good_frames):
        try:
            period_grouping["good_frames"] = float(period_good_frames[period_index])
        except (TypeError, ValueError):
            pass

    period_dead_time_us = grouping.get("period_dead_time_us")
    if isinstance(period_dead_time_us, list) and period_index < len(period_dead_time_us):
        raw_deadtime = period_dead_time_us[period_index]
        if isinstance(raw_deadtime, list):
            cleaned_deadtime: list[float] = []
            for value in raw_deadtime:
                try:
                    cleaned_deadtime.append(float(value))
                except (TypeError, ValueError):
                    continue
            period_grouping["dead_time_us"] = cleaned_deadtime

    period_histograms = grouping.get("period_histograms")
    if not (isinstance(period_histograms, list) and period_index < len(period_histograms)):
        return _clone_histograms(histograms), period_grouping

    selected_period = period_histograms[period_index]
    if not isinstance(selected_period, list):
        return _clone_histograms(histograms), period_grouping

    cloned_period: list[Histogram] = []
    for hist in selected_period:
        if not isinstance(hist, Histogram):
            return _clone_histograms(histograms), period_grouping
        cloned_period.append(_clone_histogram(hist))
    return cloned_period or _clone_histograms(histograms), period_grouping


def combine_period_asymmetry(
    red_time: NDArray[np.float64],
    red_asymmetry: NDArray[np.float64],
    red_error: NDArray[np.float64],
    green_time: NDArray[np.float64],
    green_asymmetry: NDArray[np.float64],
    green_error: NDArray[np.float64],
    mode: PeriodMode | str,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Combine reduced red/green spectra into a green∓red difference or sum.

    Implements the "G minus R" and "G plus R" RG modes:

    * ``GREEN_MINUS_RED``: ``green - red``
    * ``GREEN_PLUS_RED``: ``green + red``

    with errors added in quadrature. Arrays are truncated to their common
    length (the time axis from the red spectrum is used). Returns empty arrays
    when there is no overlap.
    """
    n = min(
        len(red_time),
        len(green_time),
        len(red_asymmetry),
        len(green_asymmetry),
        len(red_error),
        len(green_error),
    )
    if n <= 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty.copy(), empty.copy()

    time = np.asarray(red_time[:n], dtype=np.float64).copy()
    red_a = np.asarray(red_asymmetry[:n], dtype=np.float64)
    green_a = np.asarray(green_asymmetry[:n], dtype=np.float64)
    red_e = np.asarray(red_error[:n], dtype=np.float64)
    green_e = np.asarray(green_error[:n], dtype=np.float64)

    mode_key = str(mode)
    if mode_key == str(PeriodMode.GREEN_MINUS_RED):
        asymmetry = green_a - red_a
    elif mode_key == str(PeriodMode.GREEN_PLUS_RED):
        asymmetry = green_a + red_a
    else:
        raise ValueError(
            f"combine_period_asymmetry expects GREEN_MINUS_RED or GREEN_PLUS_RED, got {mode!r}"
        )

    error = np.sqrt(np.square(green_e) + np.square(red_e))
    return time, np.asarray(asymmetry, dtype=np.float64), error


# --- internal -----------------------------------------------------------------


def _clone_histogram(hist: Histogram) -> Histogram:
    return Histogram(
        counts=np.asarray(hist.counts, dtype=np.float64).copy(),
        bin_width=float(hist.bin_width),
        t0_bin=int(hist.t0_bin),
        good_bin_start=int(hist.good_bin_start),
        good_bin_end=int(hist.good_bin_end),
    )


def _clone_histograms(histograms: list[Histogram]) -> list[Histogram]:
    return [_clone_histogram(hist) for hist in histograms]


# --- multi-period subset -> red/green mapping ---------------------------------

#: Valid targets in a period mapping (WiMDA ``PeriodMappingUnit`` semantics).
PERIOD_MAPPING_TARGETS = ("red", "green", "ignore")


def normalise_period_mapping(mapping: dict, n_periods: int) -> dict[int, str]:
    """Validate a ``{period_number: target}`` mapping.

    Keys are 1-based period numbers (``int`` or numeric ``str`` — JSON round
    trips turn them into strings); targets are ``"red"``, ``"green"`` or
    ``"ignore"``. At least one period must map to red. Raises ``ValueError``
    on out-of-range periods, unknown targets, or an empty red set, so a stale
    persisted mapping fails loudly instead of silently reducing nothing.
    """
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Period mapping must be a non-empty dict")
    normalised: dict[int, str] = {}
    for raw_period, raw_target in mapping.items():
        try:
            period = int(raw_period)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Period mapping key {raw_period!r} is not a period number") from exc
        if not 1 <= period <= n_periods:
            raise ValueError(f"Period {period} is outside this run's 1..{n_periods} periods")
        target = str(raw_target).strip().lower()
        if target not in PERIOD_MAPPING_TARGETS:
            raise ValueError(
                f"Period mapping target {raw_target!r} is not one of {PERIOD_MAPPING_TARGETS}"
            )
        normalised[period] = target
    if "red" not in normalised.values():
        raise ValueError("Period mapping must send at least one period to red")
    return normalised


def sum_period_histograms(period_histograms: list[list[Histogram]]) -> list[Histogram]:
    """Sum per-period histogram lists detector-wise (count level).

    All periods of a run share detector layout, bin width and t0, so the sum
    is exact Poisson addition. Arrays are truncated to the shortest common
    bin count per detector.
    """
    if not period_histograms:
        raise ValueError("No period histograms to sum")
    n_detectors = min(len(period) for period in period_histograms)
    if n_detectors == 0:
        raise ValueError("Period histogram lists are empty")
    summed: list[Histogram] = []
    for det in range(n_detectors):
        first = period_histograms[0][det]
        n_bins = min(len(period[det].counts) for period in period_histograms)
        counts = np.zeros(n_bins, dtype=np.float64)
        for period in period_histograms:
            counts += np.asarray(period[det].counts[:n_bins], dtype=np.float64)
        summed.append(
            Histogram(
                counts=counts,
                bin_width=first.bin_width,
                t0_bin=first.t0_bin,
                good_bin_start=first.good_bin_start,
                good_bin_end=first.good_bin_end,
            )
        )
    return summed


def _periods_for_target(mapping: dict[int, str], target: str) -> list[int]:
    """1-based period numbers mapped to *target*, in ascending order."""
    return [p for p in sorted(mapping) if mapping[p] == target]


def _sum_period_set(runs: list[Run], periods: list[int]) -> list[Histogram]:
    """Detector-wise count sum over the given 1-based periods."""
    return sum_period_histograms([runs[p - 1].histograms for p in periods])


def _sum_period_good_frames(runs: list[Run], periods: list[int]) -> float:
    """Total good frames across the given 1-based periods (summed exposure)."""
    return float(sum(good_frames(runs[p - 1].grouping) for p in periods))


def _combined_dead_times(runs: list[Run], periods: list[int]) -> list[float]:
    """One deadtime table for the summed period set.

    Equal per-period tables pass through verbatim; differing tables get a
    frame-weighted mean (the best single table for the summed counts). Each
    surviving table keeps ITS period's frame weight, so periods without a table
    do not shift the others. Ragged tables (malformed metadata) reduce to their
    common detector count rather than crashing.
    """
    tables: list[tuple[list[float], float]] = []
    for p in periods:
        grouping = runs[p - 1].grouping if isinstance(runs[p - 1].grouping, dict) else {}
        values = grouping.get("dead_time_us")
        if isinstance(values, list) and values:
            tables.append(([float(v) for v in values], _sum_period_good_frames(runs, [p])))
    if not tables:
        return []
    n_detectors = min(len(values) for values, _ in tables)
    reference = tables[0][0][:n_detectors]
    if all(np.allclose(values[:n_detectors], reference) for values, _ in tables[1:]):
        return list(reference)
    stacked = np.asarray([values[:n_detectors] for values, _ in tables], dtype=np.float64)
    weights = [frames for _, frames in tables]
    table = np.average(stacked, axis=0, weights=weights)
    return [float(v) for v in table]


def combine_mapped_periods(
    period_datasets: list[MuonDataset],
    mapping: dict,
    *,
    source_run_number: int | None = None,
    source_file: str = "",
) -> MuonDataset:
    """Build one red/green dataset from per-period datasets and a mapping.

    ``period_datasets`` are the loader's per-period datasets for one run
    (3+-period files load as such a list); ``mapping`` sends each period to
    red, green or ignore (WiMDA's period-mapping form). Counts and good
    frames are summed per set at the count level — periods of one run share
    beam exposure bookkeeping exactly, so no scaling is involved (unlike
    background-run subtraction). The result carries the standard two-period
    structure (``period_histograms``, ``period_good_frames``,
    ``period_dead_time_us``, ``period_mode``) that the RG reduction and the
    grouping dialog already understand; with no green periods the result is
    a single-set run (G±R modes do not apply). The mapping is recorded on
    the grouping as ``period_mapping`` for provenance.

    The trivial mapping ``{1: red, 2: green}`` of a two-period run
    reproduces the loader's combined dataset structure.
    """
    runs = [ds.run for ds in period_datasets]
    if any(run is None or not run.histograms for run in runs):
        raise ValueError("Every period dataset needs run histograms to map periods")
    n_periods = len(period_datasets)
    normalised = normalise_period_mapping(mapping, n_periods)

    red_periods = _periods_for_target(normalised, "red")
    green_periods = _periods_for_target(normalised, "green")

    # Counts are summed fresh by sum_period_histograms (it never mutates its
    # inputs), so there is no need to clone each period before summing — only
    # the final Run gets an independent copy below.
    red_histograms = _sum_period_set(runs, red_periods)
    sets: list[list[Histogram]] = [red_histograms]
    frames = [_sum_period_good_frames(runs, red_periods)]
    dead_times = [_combined_dead_times(runs, red_periods)]
    if green_periods:
        sets.append(_sum_period_set(runs, green_periods))
        frames.append(_sum_period_good_frames(runs, green_periods))
        dead_times.append(_combined_dead_times(runs, green_periods))

    base_run = runs[red_periods[0] - 1]
    run_number = (
        int(source_run_number)
        if source_run_number is not None
        else int(base_run.metadata.get("source_run_number", base_run.run_number))
    )
    metadata = dict(period_datasets[red_periods[0] - 1].metadata)
    metadata["run_number"] = run_number
    metadata["source_run_number"] = run_number
    metadata["run_label"] = str(run_number)
    metadata["period_number"] = 1
    metadata["period_count"] = len(sets)
    metadata["period_mapping"] = {str(k): v for k, v in sorted(normalised.items())}

    grouping = dict(base_run.grouping) if isinstance(base_run.grouping, dict) else {}
    grouping["period_histograms"] = sets
    grouping["period_good_frames"] = frames
    grouping["period_dead_time_us"] = dead_times
    grouping["period_mode"] = str(PeriodMode.RED)
    grouping["period_mapping"] = {str(k): v for k, v in sorted(normalised.items())}
    grouping["good_frames"] = frames[0]
    if dead_times[0]:
        grouping["dead_time_us"] = list(dead_times[0])
    grouping.pop("period_reduced", None)

    run = Run(
        run_number=run_number,
        histograms=_clone_histograms(red_histograms),
        metadata=metadata,
        grouping=grouping,
        source_file=source_file or base_run.source_file,
    )
    reference = period_datasets[red_periods[0] - 1]
    return MuonDataset(
        time=np.asarray(reference.time, dtype=np.float64).copy(),
        asymmetry=np.asarray(reference.asymmetry, dtype=np.float64).copy(),
        error=np.asarray(reference.error, dtype=np.float64).copy(),
        metadata=metadata,
        run=run,
    )


# --- RF (Green − Red) field-difference scan -----------------------------------

#: RF combination modes accepted by :func:`build_rf_difference_scan` (the WiMDA
#: RF-µSR observable is ``Green − Red``; ``Green + Red`` is offered for symmetry).
_RF_DIFFERENCE_MODES = (str(PeriodMode.GREEN_MINUS_RED), str(PeriodMode.GREEN_PLUS_RED))


def _resolve_run_for_rf(item: MuonDataset | Run) -> Run:
    """Return the :class:`Run` backing a loaded dataset or run.

    Unlike :func:`asymmetry.core.transform.integral._resolve_run` (which needs
    histograms), the RF scan reads the ``period_reduced`` cache, so the error
    message names that instead.
    """
    if isinstance(item, Run):
        return item
    if isinstance(item, MuonDataset):
        if item.run is None:
            raise ValueError("MuonDataset has no source run; RF scan needs per-period data.")
        return item.run
    raise TypeError(f"Expected a MuonDataset or Run, got {type(item).__name__}")


def _red_green_reduced(
    run: Run,
) -> (
    tuple[
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    ]
    | None
):
    """Return the red and green reduced ``(time, asym, err)`` curves, or ``None``.

    Reads the loader's ``period_reduced`` cache (percent-scale asymmetry,
    ``red = period 1``, ``green = period 2`` — see this module's period
    conventions). Returns ``None`` when ``run`` is not a two-period (red/green)
    run, so the caller can exclude it with a reason rather than crashing.
    """
    grouping = run.grouping if isinstance(run.grouping, dict) else {}
    reduced = grouping.get("period_reduced")
    if not (isinstance(reduced, list) and len(reduced) >= 2):
        return None
    red = reduced[RED_INDEX]
    green = reduced[GREEN_INDEX]
    if not (isinstance(red, tuple) and isinstance(green, tuple)):
        return None
    if len(red) != 3 or len(green) != 3:
        return None
    return red, green


def build_rf_difference_scan(
    runs: object,
    *,
    t_min: float | None = None,
    t_max: float | None = None,
    mode: PeriodMode | str = PeriodMode.GREEN_MINUS_RED,
    order_key: str = "field",
) -> FieldScan:
    """Assemble an RF-µSR period-difference integral-asymmetry field scan.

    This is the RF-resonance analogue of
    :func:`asymmetry.core.transform.build_field_scan`. For each run it forms the
    **(Green − Red)** time-domain asymmetry from the two acquisition periods
    (Red = RF-on = period 1, Green = RF-off = period 2) via
    :func:`combine_period_asymmetry`, time-integrates it over ``[t_min, t_max]``
    with :func:`asymmetry.core.transform.integrate_curve`, and collects one value
    per run ordered by *order_key* (the swept static field). The result is the
    W-shaped double dip that :class:`RFResonanceMuP` fits for ``A_µ`` and ``A_p``.

    Parameters
    ----------
    runs
        Iterable of loaded :class:`MuonDataset` or :class:`Run` objects. Each must
        be a two-period (red/green) run; single-period runs are skipped and listed
        in :attr:`FieldScan.excluded`.
    t_min, t_max
        Inclusive integration window in µs. ``None`` integrates the whole curve.
    mode
        ``GREEN_MINUS_RED`` (default, the RF-µSR observable) or ``GREEN_PLUS_RED``.
    order_key
        ``"field"`` (default), ``"temperature"`` or ``"run"`` — the x-axis the
        points are ordered by.

    Returns
    -------
    FieldScan
        Sorted parallel arrays plus the list of excluded runs. ``value``/``error``
        are **fractional** (the percent ``period_reduced`` curves divided by 100),
        matching :func:`build_field_scan`'s convention so the same display/fit
        path applies.
    """
    mode_key = str(mode)
    if mode_key not in _RF_DIFFERENCE_MODES:
        raise ValueError(f"mode must be PeriodMode.GREEN_MINUS_RED or GREEN_PLUS_RED, got {mode!r}")
    if order_key not in ORDER_KEYS:
        raise ValueError(f"order_key must be one of {ORDER_KEYS}, got {order_key!r}")

    points: list[FieldScanPoint] = []
    excluded: list[tuple[int, str]] = []

    for item in runs:
        try:
            run = _resolve_run_for_rf(item)
        except (TypeError, ValueError) as exc:
            excluded.append((_excluded_run_number(item), str(exc)))
            continue
        run_number = int(run.run_number)

        rg = _red_green_reduced(run)
        if rg is None:
            excluded.append((run_number, "not a two-period (red/green) run"))
            continue
        (red_t, red_a, red_e), (green_t, green_a, green_e) = rg

        time, diff, err = combine_period_asymmetry(
            red_t, red_a, red_e, green_t, green_a, green_e, mode_key
        )
        if time.size == 0:
            excluded.append((run_number, "red/green spectra do not overlap"))
            continue

        x_value = _order_value(run, order_key)
        if x_value is None:
            excluded.append((run_number, f"no {order_key} value"))
            continue

        try:
            value, error = integrate_curve(time, diff, err, t_min=t_min, t_max=t_max)
        except ValueError as exc:
            excluded.append((run_number, str(exc)))
            continue

        # period_reduced is percent-scale; divide by 100 to return the fractional
        # FieldScan convention build_field_scan uses (the ALC display path ×100s).
        points.append(
            FieldScanPoint(
                run_number=run_number,
                x=x_value,
                value=float(value) / 100.0,
                error=float(error) / 100.0,
            )
        )

    points.sort(key=lambda p: (p.x, p.run_number))
    label = "Green − Red" if mode_key == str(PeriodMode.GREEN_MINUS_RED) else "Green + Red"
    return FieldScan(
        x=np.array([p.x for p in points], dtype=np.float64),
        value=np.array([p.value for p in points], dtype=np.float64),
        error=np.array([p.error for p in points], dtype=np.float64),
        run_numbers=[p.run_number for p in points],
        order_key=order_key,
        method="integral",
        x_label=_X_LABELS[order_key],
        y_label=f"Integral asymmetry ({label})",
        excluded=excluded,
    )
