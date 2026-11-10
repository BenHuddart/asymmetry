"""Select acquired periods and combine independently reduced red/green curves.

Periods use 1-based public numbering and 0-based held-array indices. Red is
period 1; green is period 2. Cached curves do not supply raw histograms for
an automatic timing search. Selection preserves acquired calibration and
exposure independently of authored corrections.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from itertools import zip_longest
from typing import TYPE_CHECKING

import numpy as np

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.io.frame_exposure import summed_period_exposures
from asymmetry.core.io.period_counts import sum_period_histograms
from asymmetry.core.io.quick_asymmetry import quick_asymmetry
from asymmetry.core.transform.deadtime import file_deadtime_table
from asymmetry.core.transform.grouping import good_frames
from asymmetry.core.transform.integral import (
    _X_LABELS,
    FieldScan,
    FieldScanPoint,
    _excluded_run_number,
    _order_value,
    integrate_curve,
)
from asymmetry.core.transform.reduce import correction_flags_from_grouping
from asymmetry.core.transform.reference_scale import ReferenceExposure
from asymmetry.core.utils.constants import ORDER_KEYS, PeriodMode

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.typing import NDArray

    #: A reduced ``(time, asymmetry, error)`` curve.
    _Curve = tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]

__all__ = [
    "source_run_of",
    "RED_INDEX",
    "GREEN_INDEX",
    "PERIOD_MAPPING_TARGETS",
    "PeriodMode",
    "build_rf_difference_scan",
    "period_run",
    "combine_mapped_periods",
    "combine_period_asymmetry",
    "encode_period_run_number",
    "normalise_period_mapping",
    "held_period_count",
    "period_count",
    "period_labels",
    "resolve_period_index",
    "reference_mapping_requires_replay",
    "selected_period_index",
    "select_period",
    "select_period_histograms",
    "select_period_grouping",
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


def source_run_of(run: Run) -> int:
    """*run*'s file run number — its own, or the one a period-selected run was cut from."""
    return int(run.metadata.get("source_run_number", run.run_number))


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


def held_period_count(run: Run) -> int:
    """Number of raw or cached periods held by this run."""
    periods = run.grouping.get("period_histograms") or run.grouping.get("period_reduced")
    return len(periods) if periods else 1


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

    if isinstance(reduced, list) and reduced:
        index = resolve_period_index(period, len(reduced))
        time, asymmetry, error = reduced[index]
        run = period_run(data.run, index)
        return MuonDataset(
            ratio_validity=(
                data.histogram_observation.cached_validities[index]
                if data.histogram_observation is not None
                and data.histogram_observation.cached_validities
                else None
            ),
            histogram_observation=(
                data.histogram_observation.cached_periods[index]
                if data.histogram_observation is not None
                and data.histogram_observation.cached_periods
                else None
            ),
            time=np.asarray(time, dtype=np.float64).copy(),
            asymmetry=np.asarray(asymmetry, dtype=np.float64).copy(),
            error=np.asarray(error, dtype=np.float64).copy(),
            metadata=run.metadata,
            run=run,
        )

    if data.run is not None and grouping.get("period_histograms"):
        index = resolve_period_index(period, len(grouping["period_histograms"]))
        return _quick_period_dataset(period_run(data.run, index))

    # Single-period dataset: only period 1 is selectable.
    count = period_count(data)
    index = resolve_period_index(period, count)
    if count <= 1 and index == 0:
        return data
    raise ValueError(
        "Per-period histograms are unavailable for this dataset; reload the "
        "source file with asymmetry.core.io.load(...) to access individual periods"
    )


def period_run(combined: Run, index: int) -> Run:
    """Extract a held period with its acquired timing, exposure, and corrections."""
    grouping = combined.grouping
    from asymmetry.core.project.profiles import GoodWindowPolicy

    count = len(grouping.get("period_reduced") or grouping["period_histograms"])

    histograms, period_grouping = select_period_histograms(combined.histograms, grouping, index)
    for key in (
        "period_histograms",
        "period_reduced",
        "period_t0_sources",
    ):
        period_grouping.pop(key, None)

    period_grouping["period_mode"] = ("red", "green")[index]
    metadata = dict(combined.metadata)
    metadata["raw_period_histograms_available"] = (
        acquired_period_histograms(grouping, index) is not None
    )
    source_run = source_run_of(combined)
    # Each period gets its own encoded run-number key so both survive in the
    # run-number-keyed data browser (otherwise the second add collapses the
    # first). The source run and the friendly run/period label stay in metadata.
    period_run_number = encode_period_run_number(source_run, index + 1)
    metadata["run_number"] = period_run_number
    metadata["source_run_number"] = source_run
    metadata["period_number"] = index + 1
    metadata["period_count"] = count
    metadata["run_label"] = f"{source_run}/{index + 1}"
    inputs = grouping.get("period_inputs")
    if inputs is not None:
        members = _periods_for_target(
            {int(k): v for k, v in grouping["period_mapping"].items()}, ("red", "green")[index]
        )
        for key in ("data_period_number", "dae_period_number"):
            metadata.pop(key, None)
            if len(members) == 1 and key in inputs[members[0] - 1].metadata:
                metadata[key] = inputs[members[0] - 1].metadata[key]
    if count == 2:
        metadata["period_label"] = str((PeriodMode.RED, PeriodMode.GREEN)[index])

    run = Run(
        run_number=period_run_number,
        histograms=histograms,
        metadata=metadata,
        grouping=period_grouping,
        source_file=combined.source_file,
    )

    # Per-detector file facts follow the selected period; scalar-only requests
    # retain their recorded window. Extraction never runs a timing search.
    if (
        metadata["raw_period_histograms_available"]
        and "detector_first_good_bins" in period_grouping
        and GoodWindowPolicy.from_dict(period_grouping.get("good_window_policy")).mode
        == "from_file"
    ):
        quick = quick_asymmetry(histograms, period_grouping, metadata=metadata)
        run.grouping.pop("t0_time_us", None)
        run.grouping.update(quick.file_grouping)
        metadata.pop("correction_error", None)
        metadata.update(quick.metadata)
    return run


# --- low-level helpers shared with the GUI reduction path ---------------------


def reference_mapping_requires_replay(mapping: dict | None) -> bool:
    """Whether held red/green arrays cannot name the same acquired period on disk.

    Ordinary two-period loading also records the identity mapping. A nontrivial
    mapping needs its contributor recipe, rather than an acquired-period ordinal.
    """
    return any(
        str(number) != {"red": "1", "green": "2"}.get(target)
        for number, target in (mapping or {}).items()
        if target != "ignore"
    )


def selected_period_index(grouping: dict, mode: str | None = None) -> int | None:
    """Index for a single held Red/Green selection; combined modes return None."""
    if "period_histograms" not in grouping:
        return None
    return {PeriodMode.RED: RED_INDEX, PeriodMode.GREEN: GREEN_INDEX}.get(
        mode or grouping.get("period_mode", PeriodMode.RED)
    )


def _period_timing_facts(grouping: dict, period_index: int) -> dict:
    """Select acquisition timing status; an absent period table is not inherited."""
    facts = dict(grouping)
    if "period_t0_sources" in grouping:
        facts["t0_source"] = grouping["period_t0_sources"][period_index]
    if "period_detector_t0_sources" in grouping:
        sources = grouping["period_detector_t0_sources"][period_index]
        if sources is None:
            facts.pop("detector_t0_sources", None)
        else:
            facts["detector_t0_sources"] = sources
    return facts


def select_period_grouping(grouping: dict, period_index: int) -> dict:
    """Select acquired calibration and exposure, preserving authored coefficients."""
    period_grouping = _period_timing_facts(grouping, period_index)
    period_grouping["period_mode"] = ("red", "green")[period_index]
    origin = _period_timing_facts(grouping, grouping.get("effective_t0_period", period_index))
    manual_file = grouping.get("t0_policy", {}).get("mode") == "manual" and all(
        facts.get("t0_source") != "missing"
        and "missing" not in facts.get("detector_t0_sources", [])
        for facts in (origin, period_grouping)
    )
    if (
        "effective_t0_period" in grouping
        and grouping["effective_t0_period"] != period_index
        and not manual_file
    ):
        for key in (
            "effective_detector_t0_bins",
            "effective_t0_sources",
            "effective_t0_period",
            "t0_search_strategy",
            "t0_search_spread_bins",
        ):
            period_grouping.pop(key, None)

    period_grouping["reference_exposure"] = (
        grouping["period_reference_exposure"][period_index]
        if "period_reference_exposure" in grouping
        else grouping.get("reference_exposure", ReferenceExposure().to_dict())
        if period_index == 0
        else ReferenceExposure(period_number=period_index + 1).to_dict()
    )

    period_good_frames = grouping.get("period_good_frames")
    if isinstance(period_good_frames, list):
        period_grouping.pop("good_frames", None)
        if period_index < len(period_good_frames) and period_good_frames[period_index] is not None:
            period_grouping["good_frames"] = period_good_frames[period_index]

    period_dead_time_us = grouping.get("period_dead_time_us")
    stored = correction_flags_from_grouping(grouping).deadtime_mode in {"manual", "estimate"}
    if isinstance(period_dead_time_us, list):
        values = (
            period_dead_time_us[period_index] if period_index < len(period_dead_time_us) else []
        )
        period_grouping["deadtime_file_us"] = list(values) if isinstance(values, list) else []
        if not stored:
            period_grouping["dead_time_us"] = list(period_grouping["deadtime_file_us"])

    return period_grouping


def select_period_histograms(
    histograms: list[Histogram],
    grouping: dict,
    period_index: int,
) -> tuple[list[Histogram], dict]:
    """Select acquired histograms, exposure, and calibration for one period."""
    period_grouping = select_period_grouping(grouping, period_index)

    selected = held_period_histograms(histograms, grouping, period_index)
    facts = histogram_grouping_facts(selected, period_grouping)
    if "effective_t0_sources" in facts:
        facts["effective_t0_period"] = period_index
    return _clone_histograms(selected), facts


def held_period_histograms(
    histograms: list[Histogram], grouping: dict, period_index: int
) -> list[Histogram]:
    """Read raw counts for a held acquisition, retaining the acquired fallback."""
    return acquired_period_histograms(grouping, period_index) or histograms


def acquired_period_histograms(grouping: dict, period_index: int) -> list[Histogram] | None:
    """Raw held counts only; cached curves cannot establish an acquired period."""
    period_histograms = grouping.get("period_histograms")
    selected = (
        period_histograms[period_index]
        if isinstance(period_histograms, list) and 0 <= period_index < len(period_histograms)
        else None
    )
    return (
        selected
        if isinstance(selected, list)
        and selected
        and all(isinstance(hist, Histogram) for hist in selected)
        else None
    )


def histogram_grouping_facts(histograms: list[Histogram], grouping: dict) -> dict:
    """Refresh recorded detector facts for selected histograms without copying counts."""
    period_grouping = dict(grouping)
    from asymmetry.core.transform.t0 import EFFECTIVE_DETECTOR_T0_KEY, detector_t0_overrides

    policy = grouping.get("t0_policy", {})
    sources = grouping.get("effective_t0_sources", [])
    estimated = (
        policy.get("mode") == "auto_detect" or "estimated" in sources
    ) and "effective_t0_period" not in grouping
    override = (
        detector_t0_overrides(grouping, len(histograms)) if histograms and not estimated else None
    )
    if estimated:
        period_grouping.pop(EFFECTIVE_DETECTOR_T0_KEY, None)
        period_grouping.pop("effective_t0_sources", None)
    anchors = grouping.get("detector_t0_bins")
    if (
        override is not None
        and isinstance(anchors, (list, tuple))
        and len(anchors) == len(histograms)
    ):
        period_grouping[EFFECTIVE_DETECTOR_T0_KEY] = [
            max(0, hist.t0_bin + effective - acquired)
            for hist, effective, acquired in zip(histograms, override, anchors, strict=True)
        ]
    for key, values in (
        ("detector_t0_bins", [hist.t0_bin for hist in histograms]),
        ("detector_first_good_bins", [hist.good_bin_start for hist in histograms]),
        (
            "detector_last_good_bins",
            [
                hist.good_bin_end if hist.good_bin_end >= 0 else hist.n_bins - 1
                for hist in histograms
            ],
        ),
    ):
        if key in grouping:
            period_grouping[key] = values
    return period_grouping


def combine_period_asymmetry(
    red_time: NDArray[np.float64],
    red_asymmetry: NDArray[np.float64],
    red_error: NDArray[np.float64],
    green_time: NDArray[np.float64],
    green_asymmetry: NDArray[np.float64],
    green_error: NDArray[np.float64],
    mode: PeriodMode | str,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Difference independent curves; pooled G+R requires prepared counts.

    Arrays alone cannot determine the pooled count ratio. Both curves must have
    identical time coordinates; errors of the difference add in quadrature.
    """
    if str(mode) == str(PeriodMode.GREEN_PLUS_RED):
        raise ValueError("Pooled G+R requires prepared counts; reload raw period data.")
    arrays = (red_time, red_asymmetry, red_error, green_time, green_asymmetry, green_error)
    if any(np.asarray(array).ndim != 1 or len(array) != len(red_time) for array in arrays):
        raise ValueError(
            "Period asymmetry arrays must have identical lengths and be one-dimensional."
        )
    if not np.array_equal(red_time, green_time):
        raise ValueError("Period asymmetry time arrays must be identical.")
    time = np.asarray(red_time, dtype=np.float64).copy()
    red_a, green_a = np.asarray(red_asymmetry), np.asarray(green_asymmetry)
    red_e, green_e = np.asarray(red_error), np.asarray(green_error)

    mode_key = str(mode)
    if mode_key == str(PeriodMode.GREEN_MINUS_RED):
        asymmetry = green_a - red_a
    else:
        raise ValueError(f"combine_period_asymmetry expects GREEN_MINUS_RED, got {mode!r}")

    error = np.sqrt(np.square(green_e) + np.square(red_e))
    return time, np.asarray(asymmetry, dtype=np.float64), error


# --- internal -----------------------------------------------------------------


def _clone_histogram(hist: Histogram) -> Histogram:
    return replace(hist, counts=np.asarray(hist.counts, dtype=np.float64).copy())


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


def _periods_for_target(mapping: dict[int, str], target: str) -> list[int]:
    """1-based period numbers mapped to *target*, in ascending order."""
    return [p for p in sorted(mapping) if mapping[p] == target]


def _sum_period_set(
    runs: list[Run], periods: list[int], cancel_check: Callable[[], None] | None = None
) -> list[Histogram]:
    """Detector-wise count sum over the given 1-based periods."""
    return sum_period_histograms(
        [runs[p - 1].histograms for p in periods], cancel_check=cancel_check
    )


def _sum_period_good_frames(runs: list[Run], periods: list[int]) -> float | None:
    """Total good frames across the given 1-based periods (summed exposure)."""
    frames = [good_frames(runs[p - 1].grouping, default=0.0) for p in periods]
    return float(sum(frames)) if all(value > 0 for value in frames) else None


def _combined_dead_times(runs: list[Run], periods: list[int]) -> list[float | None]:
    """Keep detector positions, combining only coefficients supplied by every period."""
    tables = [file_deadtime_table(runs[p - 1].grouping) for p in periods]
    if len(tables) == 1:
        return list(tables[0])
    # A held view has no effective coefficient for a nonlinear correction.
    return [
        column[0] if all(value == column[0] for value in column) else None
        for column in zip_longest(*tables)
    ]


def combine_mapped_periods(
    period_datasets: list[MuonDataset] | MuonDataset,
    mapping: dict,
    *,
    source_run_number: int | None = None,
    source_file: str = "",
    cancel_check: Callable[[], None] | None = None,
) -> MuonDataset:
    """Sum acquired periods into red/green sets without exposure scaling.

    Counts, DT normalisers, and evidenced exposures sum within each set.
    Held views retain raw-index inspection; reduction corrects and aligns originals.
    Pass a mapped dataset to remap its
    retained original acquisitions, including those previously ignored.
    """
    if cancel_check is not None:
        cancel_check()
    if isinstance(period_datasets, MuonDataset):
        if period_datasets.run is None or "period_inputs" not in period_datasets.run.grouping:
            raise ValueError("Remapping requires retained acquired period inputs")
        source_file = source_file or period_datasets.run.source_file
        if source_run_number is None:
            source_run_number = source_run_of(period_datasets.run)
        runs = list(period_datasets.run.grouping["period_inputs"])
        metadata_sources = [run.metadata for run in runs]
    else:
        runs = [ds.run for ds in period_datasets]
        metadata_sources = [ds.metadata for ds in period_datasets]
    if any(run is None or not run.histograms for run in runs):
        raise ValueError("Every period dataset needs run histograms to map periods")
    n_periods = len(runs)
    normalised = normalise_period_mapping(mapping, n_periods)

    red_periods = _periods_for_target(normalised, "red")
    green_periods = _periods_for_target(normalised, "green")

    # Summation preserves the input arrays; only the returned Run owns new counts.
    red_histograms = _sum_period_set(runs, red_periods, cancel_check)
    sets: list[list[Histogram]] = [red_histograms]
    frames = [_sum_period_good_frames(runs, red_periods)]
    dead_times = [_combined_dead_times(runs, red_periods)]
    if green_periods:
        sets.append(_sum_period_set(runs, green_periods, cancel_check))
        frames.append(_sum_period_good_frames(runs, green_periods))
        dead_times.append(_combined_dead_times(runs, green_periods))

    base_run = runs[red_periods[0] - 1]
    run_number = (
        int(source_run_number) if source_run_number is not None else source_run_of(base_run)
    )
    metadata = dict(metadata_sources[red_periods[0] - 1])
    metadata["run_number"] = run_number
    metadata["source_run_number"] = run_number
    metadata["run_label"] = str(run_number)
    metadata["period_number"] = 1
    metadata["period_count"] = len(sets)
    metadata["period_mapping"] = {str(k): v for k, v in sorted(normalised.items())}
    for key in ("data_period_number", "dae_period_number"):
        if len(red_periods) != 1:
            metadata.pop(key, None)

    grouping = dict(base_run.grouping) if isinstance(base_run.grouping, dict) else {}
    # Acquired inputs own constituent facts; held sums are pooled display views.
    grouping["period_inputs"] = tuple(
        replace(
            run,
            histograms=[replace(h) for h in run.histograms],
            metadata=deepcopy(metadata_source),
            grouping=deepcopy(
                {
                    key: value
                    for key, value in run.grouping.items()
                    if key not in {"period_inputs", "period_histograms", "period_reduced"}
                }
            ),
        )
        for run, metadata_source in zip(runs, metadata_sources, strict=True)
    )
    grouping["period_histograms"] = sets
    # Held raw views borrow the first clock; numerical preparation uses originals.
    grouping["period_t0_sources"] = [
        runs[periods[0] - 1].grouping.get("t0_source", "file")
        for periods in (red_periods, green_periods)
        if periods
    ]
    grouping["period_detector_t0_sources"] = [
        runs[ps[0] - 1].grouping.get("detector_t0_sources")
        for ps in (red_periods, green_periods)
        if ps
    ]
    exposures = summed_period_exposures(runs, (red_periods, green_periods))
    grouping["period_reference_exposure"] = exposures
    grouping["reference_exposure"] = exposures[0]
    grouping["period_good_frames"] = frames
    grouping["period_dead_time_us"] = dead_times
    grouping["period_mode"] = str(PeriodMode.RED)
    grouping["period_mapping"] = {str(k): v for k, v in sorted(normalised.items())}
    grouping.pop("good_frames", None)
    if frames[0] is not None:
        grouping["good_frames"] = frames[0]
    grouping["deadtime_file_us"] = list(dead_times[0])
    if correction_flags_from_grouping(grouping).deadtime_mode not in {"manual", "estimate"}:
        grouping["dead_time_us"] = list(dead_times[0])
    grouping.pop("period_reduced", None)

    red_histograms, grouping = select_period_histograms(red_histograms, grouping, RED_INDEX)
    run = Run(
        run_number=run_number,
        histograms=red_histograms,
        metadata=metadata,
        grouping=grouping,
        source_file=source_file or base_run.source_file,
    )
    return _quick_period_dataset(run, cancel_check=cancel_check)


def _quick_period_dataset(
    run: Run, *, cancel_check: Callable[[], None] | None = None
) -> MuonDataset:
    """Capture a mapped/selected raw quick look with its actual replay settings."""
    if cancel_check is not None:
        cancel_check()
    grouping, metadata = run.grouping, run.metadata
    quick = quick_asymmetry(run.histograms, grouping, metadata=metadata, cancel_check=cancel_check)
    grouping.pop("t0_time_us", None)
    grouping.update(quick.file_grouping)
    metadata.update(quick.metadata)
    window = quick.window.slice if quick.window is not None else slice(0, 0)
    from asymmetry.core.transform.preparation import histogram_observation

    # Quick looks use selected alpha, beta one, file support, and native bins.
    source = replace(
        run,
        grouping=dict(
            grouping,
            beta=1.0,
            binning_mode="fixed",
            bunching_factor=1,
            deadtime_correction=False,
            background_correction=False,
        ),
    )
    source.grouping.pop("good_window_policy", None)
    return MuonDataset(
        histogram_observation=histogram_observation(source, cancel_check=cancel_check),
        ratio_validity=quick.ratio_validity.select(window)
        if quick.ratio_validity is not None
        else None,
        time=quick.time[window],
        asymmetry=quick.asymmetry[window],
        error=quick.error[window],
        metadata=metadata,
        run=run,
    )


# --- RF (Green − Red) field-difference scan -----------------------------------

#: RF combination modes accepted by :func:`build_rf_difference_scan` (the WiMDA
#: RF-µSR observable is ``Green − Red``; ``Green + Red`` is offered for symmetry).
_RF_DIFFERENCE_MODES = (str(PeriodMode.GREEN_MINUS_RED), str(PeriodMode.GREEN_PLUS_RED))


def _resolve_run_for_rf(item: MuonDataset | Run) -> Run:
    """Return the :class:`Run` backing a loaded dataset or run.

    Count-backed requests use current preparation; historical differences may
    still use the stored per-period curves.
    """
    if isinstance(item, Run):
        return item
    if isinstance(item, MuonDataset):
        if item.run is None:
            raise ValueError("MuonDataset has no source run; RF scan needs per-period data.")
        return item.run
    raise TypeError(f"Expected a MuonDataset or Run, got {type(item).__name__}")


def _red_green_reduced(run: Run) -> tuple[_Curve, _Curve] | None:
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
    selected time-domain observation from two acquisition periods, then integrates
    it over ``[t_min, t_max]``
    with :func:`asymmetry.core.transform.integrate_curve`, and collects one value
    per run ordered by *order_key* (the swept static field). The result is the
    W-shaped double dip that :class:`RFResonanceMuP` fits for ``A_µ`` and ``A_p``
    when the acquisition conditions identify that RF contrast. Colour labels
    alone do not identify RF-on/off. G+R pools prepared counts. Curve-only
    historical differences remain available; pooled reconstruction requires
    verified count preparation and otherwise records an explicit reload reason.

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

        # A retained observation establishes count replay; an attached Run alone
        # does not establish the provenance of an arbitrary dataset's curve.
        raw = isinstance(item, Run) or item.histogram_observation is not None
        if (
            raw
            and run.histograms
            and all(acquired_period_histograms(run.grouping, i) is not None for i in (0, 1))
        ):
            from asymmetry.core.io import resolve_background_reference
            from asymmetry.core.transform.preparation import prepare_histogram_asymmetry

            effective = dict(run.grouping, period_mode=mode_key)
            reference = (
                resolve_background_reference(effective.get("background_run"))
                if effective.get("background_correction")
                and effective.get("background_mode") == "reference_run"
                else None
            )
            result = prepare_histogram_asymmetry(
                replace(run, grouping=effective),
                merge=1,
                facility=None,
                reference=reference,
            )
            time, diff = result.time, result.asymmetry
            err = result.ratio_validity.mathematical_error(result.error)
        else:
            if mode_key == str(PeriodMode.GREEN_PLUS_RED):
                excluded.append(
                    (
                        run_number,
                        "pooled G+R unavailable: reload raw period data with verified count preparation",
                    )
                )
                continue
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
