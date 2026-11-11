"""Independent linear inspection views with retained acquired reduction inputs.

Counts use signed coefficients and independent variances use their squares.
Histogram.count_variance carries retained noise through every consumer. Raw
independent unit sums retain the implicit Poisson variance. Correction and
ordinary/period timing policies are supplied by the existing reduction owners.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.io.periods import sum_period_histograms
from asymmetry.core.transform.corrections import (
    independent_observations,
    reference_observation_metadata,
)
from asymmetry.core.transform.grouping import good_frames
from asymmetry.core.transform.t0 import run_t0_time_us

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.typing import NDArray

__all__ = [
    "CombineError",
    "coadd_member_windows",
    "coadd_compatibility_error",
    "combine_runs",
    "reduce_combined_run",
    "runs_with_dataset_metadata",
]

#: Scalar metadata keys a dataset may override (display/log scalars not on the
#: underlying run); merged onto run copies for combine's event-weighting.
_DATASET_SCALAR_KEYS = ("temperature", "field", "title")

#: Bin-width agreement tolerance (µs) for the count-level compatibility check.
_BIN_WIDTH_RTOL = 1e-6

#: Grouping keys holding the per-period payload (mirrors simulate._PERIOD_KEYS).
_PERIOD_KEYS = (
    "period_histograms",
    "period_inputs",
    "period_reduced",
    "period_good_frames",
    "period_dead_time_us",
)


class CombineError(ValueError):
    """Raised when runs cannot be combined at the count level."""


# --- public API ---------------------------------------------------------------


def combine_runs(
    runs: Sequence[Run],
    *,
    sign: int = 1,
    scales: Sequence[float] | None = None,
    run_number: int | None = None,
    label: str | None = None,
    subtract_method: str = "reference",
    cancel_check=None,
) -> Run:
    """Combine the raw histograms of ``runs`` at the count level.

    Parameters
    ----------
    runs
        Two or more loaded runs with detector histograms. Every run must share
        the same detector count and (per detector) bin width; differing
        per-detector ``t0`` values are aligned before summing (the ISIS norm of
        a single shared ``t0`` takes the trivial path).
    sign
        ``+1`` co-adds (``runs[0] + runs[1] + …``); ``-1`` co-subtracts
        (``runs[0] − Σ scaleₖ·runsₖ`` for ``k ≥ 1``). Co-subtract is the
        reference-run path: ``runs`` is ``[sample, reference]`` and ``scales``
        is ``[1.0, frame_ratio]``.
    scales
        Finite signed per-run multipliers (length ``len(runs)``), default one.
        Zero terms do not participate in compatibility or noise requirements.
        ``runs[0]`` is always taken at its own scale (normally 1.0).
    run_number
        Run number for the combined run. Defaults to ``runs[0].run_number``
        (the GUI passes a negative synthetic id).
    label
        Optional explicit ``run_label``; otherwise built from the constituents
        (``"a + b"`` for add, ``"a − b"`` for subtract).
    subtract_method
        For ``sign == -1`` only. ``"reference"`` (default) is the frame-scaled
        reference subtraction ``[sample, reference]`` with ``scales =
        [1.0, frame_ratio]`` (exactly two runs). ``"signed"`` is the symmetric
        N-run signed co-subtract ``runs[0] − Σ scaleₖ·runsₖ`` (``k ≥ 1``), every
        term contributing its retained independent variance — for photo-µSR laser-on/off
        and background-style differences over more than two runs.

    cancel_check
        Optional cooperative cancellation callback, checked between detectors
        and original acquisition records. Raising aborts without changing sources.

    Returns
    -------
    Run
        A first-class run with summed (or differenced) histograms, mirrored
        grouping, accumulated good frames, event-weighted scalar metadata with
        spread keys, and a ``metadata["combination"]`` provenance block. For a
        subtraction the independent noise is retained in each histogram's
        ``count_variance``.

    Raises
    ------
    CombineError
        Fewer than two runs, empty histograms, or mismatched detector
        count / bin width.
    """
    if cancel_check is not None:
        cancel_check()
    if sign not in (1, -1):
        raise CombineError("sign must be +1 (co-add) or -1 (co-subtract)")
    subtract_method = str(subtract_method)
    if sign == -1 and subtract_method not in ("reference", "signed"):
        raise CombineError("subtract_method must be 'reference' or 'signed'")

    runs = list(runs)
    if len(runs) < 2:
        raise CombineError("combine_runs needs at least two runs")
    if sign == -1 and subtract_method == "reference" and len(runs) != 2:
        # The reference-run co-subtract is one sample, one reference. The
        # symmetric N-run signed subtraction uses subtract_method="signed".
        raise CombineError("reference co-subtract takes exactly two runs (sample, reference)")

    if scales is None:
        scales = [1.0] * len(runs)
    scales = [float(s) for s in scales]
    if len(scales) != len(runs):
        raise CombineError("scales must match the number of runs")

    if not all(np.isfinite(scales)):
        raise CombineError("scales must be finite")
    coefficients = [scale if i == 0 else sign * scale for i, scale in enumerate(scales)]
    active = [
        (run, coefficient)
        for run, coefficient in zip(runs, coefficients, strict=True)
        if coefficient != 0
    ]
    if not active:
        raise CombineError("combine_runs requires at least one nonzero coefficient")
    contributors, weights = map(list, zip(*active, strict=True))
    if any(run.metadata.get("combination") for run in contributors):
        from asymmetry.core.transform.background import resolve_background_mode

        references = [
            {
                key: run.grouping.get("background_run", {}).get(key)
                for key in ("source_file", "run_number", "period_number", "scale_request", "timing")
            }
            if run.grouping.get("background_correction")
            and resolve_background_mode(run.grouping) == "reference_run"
            else None
            for run in contributors
        ]
        if any(request != references[0] for request in references[1:]):
            raise CombineError(
                "Derived run combinations require one shared reference-background request; choose original acquisitions and a shared reference."
            )
    error = _compatibility_error(contributors)
    if error is not None:
        raise CombineError(error)
    observations = [combination_observations(run) for run in contributors]
    for i, members in enumerate(observations):
        for previous in observations[:i]:
            if any(
                not independent_observations(histograms, metadata, other_histograms, other_metadata)
                for histograms, metadata in members
                for other_histograms, other_metadata in previous
            ):
                raise CombineError(
                    "Run arithmetic requires independent observations; a source is reused"
                )

    period_sizes = [len(run.grouping.get("period_histograms", ())) for run in contributors]
    if any(period_sizes) and (not all(period_sizes) or len(set(period_sizes)) != 1):
        raise CombineError("Incompatible period payloads; select corresponding periods explicitly")
    period_payload = None
    if any(period_sizes):
        histograms, period_payload = _combine_periodic(
            contributors, weights, sign=sign, cancel_check=cancel_check
        )
    else:
        histograms = _combine_histograms(contributors, weights, cancel_check=cancel_check)
    negative_bins = sum(int(np.count_nonzero(h.counts < 0)) for h in histograms)

    base = runs[0]
    number = base.run_number if run_number is None else int(run_number)
    # Frame totals label the inspection view. Correction reads original
    # denominators; measured reference exposure is resolved independently.
    if sign == 1:
        exposure = float(sum(max(good_frames(run.grouping, 0.0), 0.0) for run in contributors))
    else:
        exposure = max(good_frames(contributors[0].grouping, 0.0), 0.0)
    grouping = _mirror_grouping(contributors[0], period_payload, exposure)
    _retain_acquisitions(grouping, contributors, weights, cancel_check=cancel_check)
    if sign == -1 and subtract_method == "reference" and scales[0] == 1:
        from asymmetry.core.transform.reference_scale import ReferenceExposure

        grouping["reference_exposure"] = ReferenceExposure.from_grouping(runs[0].grouping).to_dict()
        if "period_reference_exposure" in runs[0].grouping:
            grouping["period_reference_exposure"] = copy.deepcopy(
                runs[0].grouping["period_reference_exposure"]
            )
    metadata = _combined_metadata(
        runs,
        scales=scales,
        sign=sign,
        subtract_method=subtract_method,
        number=number,
        label=label,
        exposure=exposure,
        negative_bins=negative_bins,
    )

    return Run(
        run_number=number,
        histograms=histograms,
        metadata=metadata,
        grouping=grouping,
        source_file="",
    )


def _retain_acquisitions(grouping, runs, coefficients, *, cancel_check=None):
    """Retain borrowed originals and counterpart identities for acquired preparation."""
    from asymmetry.core.io.frame_exposure import summed_period_exposures
    from asymmetry.core.io.periods import histogram_grouping_facts, select_period_grouping
    from asymmetry.core.transform.reference_scale import ReferenceExposure

    originals, weights, mapping = [], [], {}
    for run, coefficient in zip(runs, coefficients, strict=True):
        if cancel_check is not None:
            cancel_check()
        retained = run.grouping.get("period_inputs")
        if retained is not None:
            for number, (original, weight) in enumerate(
                zip(
                    retained,
                    run.grouping.get("constituent_coefficients", [1.0] * len(retained)),
                    strict=True,
                ),
                1,
            ):
                originals.append(original)
                weights.append(coefficient * weight)
                mapping[str(len(originals))] = run.grouping["period_mapping"].get(
                    str(number), run.grouping["period_mapping"].get(number)
                )
        elif run.grouping.get("period_histograms"):
            for index in range(len(run.grouping["period_histograms"])):
                histograms = run.grouping["period_histograms"][index]
                facts = histogram_grouping_facts(
                    histograms, select_period_grouping(run.grouping, index)
                )
                for key in ("period_histograms", "period_reduced"):
                    facts.pop(key, None)
                originals.append(
                    replace(
                        run,
                        histograms=histograms,
                        grouping=facts,
                        metadata=dict(run.metadata, data_period_number=index + 1),
                    )
                )
                weights.append(coefficient)
                mapping[str(len(originals))] = ("red", "green")[index]
        else:
            originals.append(run)
            weights.append(coefficient)
            mapping[str(len(originals))] = grouping.get("period_mode", "red")
    grouping.update(
        period_inputs=tuple(originals),
        constituent_coefficients=tuple(weights),
        period_mapping=mapping,
    )
    grouping.setdefault("period_mode", "red")
    # Only unit acquisition sums carry measured pooled exposure. The correction
    # denominator remains attached to each original, independently of this fact.
    exposures = []
    for target in ("red", "green"):
        selected = [i for i in range(1, len(originals) + 1) if mapping[str(i)] == target]
        unit = all(weight == 1 for i, weight in enumerate(weights, 1) if mapping[str(i)] == target)
        exposures.append(
            summed_period_exposures(originals, [selected])[0]
            if selected and unit
            else ReferenceExposure().to_dict()
        )
    grouping["reference_exposure"] = exposures[int(grouping.get("period_mode") == "green")]
    if grouping.get("period_histograms"):
        grouping["period_reference_exposure"] = exposures


def reduce_combined_run(run: Run, *, reference=None, cancel_check=None) -> MuonDataset:
    """Reduce canonical counts and retained variance through the normal owner."""
    from asymmetry.core.io import resolve_background_reference
    from asymmetry.core.transform.background import resolve_background_mode
    from asymmetry.core.transform.preparation import (
        histogram_observation,
        prepare_histogram_asymmetry,
    )
    from asymmetry.core.transform.reduce import correction_flags_from_grouping
    from asymmetry.core.transform.reference_scale import ReferenceExposure

    if (
        reference is None
        and correction_flags_from_grouping(run.grouping).use_background
        and resolve_background_mode(run.grouping) == "reference_run"
    ):
        reference = resolve_background_reference(
            run.grouping.get("background_run"),
            sample_exposure=ReferenceExposure.from_grouping(run.grouping),
        )
    reduced = prepare_histogram_asymmetry(
        run, merge=1, facility=None, reference=reference, cancel_check=cancel_check
    )
    if reduced.effective_alpha is not None:
        run = replace(run, grouping=dict(run.grouping, alpha=reduced.effective_alpha))
    return MuonDataset(
        reduced.time,
        reduced.asymmetry,
        reduced.error,
        reduced.metadata_with_alignment(run.metadata),
        run,
        histogram_observation(run, reference=reference, cancel_check=cancel_check),
        reduced.ratio_validity,
    )


# --- in-batch co-add windowing ------------------------------------------------


def runs_with_dataset_metadata(datasets: Sequence[MuonDataset]) -> list[Run]:
    """Shallow run copies whose scalar metadata reflects each dataset's overrides.

    A :class:`MuonDataset`'s metadata is the display source of truth for scalars
    like temperature/field (it may carry browser or from-log overrides that the
    underlying :attr:`MuonDataset.run` metadata does not). Combining the raw runs
    directly would event-weight on the stale run scalars, so this merges the
    dataset's :data:`_DATASET_SCALAR_KEYS` onto a shallow run copy (sharing
    histograms/grouping, which :func:`combine_runs` never mutates). Datasets
    without a backing run are skipped. Use this to feed :func:`combine_runs` from
    GUI datasets so the combined run's scalars match what the user sees.
    """
    from dataclasses import replace

    runs: list[Run] = []
    for dataset in datasets:
        run = dataset.run
        if run is None:
            continue
        merged = dict(run.metadata)
        ds_metadata = dataset.metadata if isinstance(dataset.metadata, dict) else {}
        for key in _DATASET_SCALAR_KEYS:
            if key in ds_metadata:
                merged[key] = ds_metadata[key]
        runs.append(replace(run, metadata=merged))
    return runs


def coadd_member_windows(
    n_members: int,
    *,
    mode: str,
    window: int,
) -> list[list[int]]:
    """Index windows for in-batch co-add of successive batch-series members.

    Mirrors WiMDA's batch-fit sequential co-add (the "+ N runs" control):
    a window co-adds ``window`` successive members. The two stepping modes match
    WiMDA's Smooth/Bin radio buttons:

    * ``"smooth"`` — sliding window, **step 1**: windows
      ``[0, W)``, ``[1, 1 + W)``, …. Yields ``n - W + 1`` windows.
    * ``"bin"`` — non-overlapping, **step W**: windows ``[0, W)``, ``[W, 2W)``, …
      Yields ``n // W`` windows.

    In both modes, as in WiMDA, the loop requires a *full*
    window, so a trailing partial window is dropped — this matches that exactly.

    Parameters
    ----------
    n_members
        Number of ordered members available to fit.
    mode
        ``"smooth"`` or ``"bin"``. Any other value (e.g. ``"off"``) returns one
        singleton window per member (no co-add).
    window
        Members co-added per window (WiMDA ``jump + 1``). A value ``<= 1`` is the
        no-op singleton partition; values are clamped to ``>= 1``.

    Returns
    -------
    list[list[int]]
        Each inner list holds the member indices for one co-add window, in order.
        Empty when no full window fits (``window > n_members`` in a co-add mode);
        callers fall back to no co-add and report it.
    """
    n = max(0, int(n_members))
    width = max(1, int(window))
    normalized = str(mode).strip().lower()
    if normalized not in ("smooth", "bin") or width <= 1:
        return [[i] for i in range(n)]
    if width > n:
        return []
    step = 1 if normalized == "smooth" else width
    return [list(range(start, start + width)) for start in range(0, n - width + 1, step)]


# --- compatibility ------------------------------------------------------------


def combination_observations(run: Run) -> list[tuple[list[Histogram], dict]]:
    """Resolve direct acquired identities from active mapped or held period facts."""
    originals = run.grouping.get("period_inputs")
    if originals is not None:
        mapping = run.grouping["period_mapping"]
        selected = (
            {run.grouping.get("period_mode", "red")}
            if not run.grouping.get("period_histograms")
            else {"red", "green"}
        )
        return [
            (original.histograms, reference_observation_metadata(original))
            for number, original in enumerate(originals, 1)
            if mapping.get(str(number), mapping.get(number)) in selected
        ]
    periods = run.grouping.get("period_histograms")
    metadata = reference_observation_metadata(run)
    if periods:
        return [
            (histograms, dict(metadata, data_period_number=number))
            for number, histograms in enumerate(periods, 1)
        ]
    return [(run.histograms, metadata)]


def _compatibility_error(runs: list[Run]) -> str | None:
    """Enforce the count-level invariants: histograms, detector count, width."""
    for run in runs:
        if not run.histograms:
            return f"run {run.run_number} has no histograms to combine"

    n_detectors = len(runs[0].histograms)
    for run in runs[1:]:
        if len(run.histograms) != n_detectors:
            return (
                "runs have different detector counts "
                f"({n_detectors} vs {len(run.histograms)}); cannot combine"
            )

    for det in range(n_detectors):
        width0 = float(runs[0].histograms[det].bin_width)
        for run in runs[1:]:
            width = float(run.histograms[det].bin_width)
            if not np.isclose(width, width0, rtol=_BIN_WIDTH_RTOL, atol=0.0):
                return (
                    f"detector {det + 1} bin widths differ ({width0} vs {width} µs); cannot combine"
                )
    return None


def _aligned_detector_arrays(
    counts: list[NDArray[np.float64]],
    t0s: list[int],
) -> tuple[list[NDArray[np.float64]], int]:
    """Shift each detector array to a common t0 (max), zero-padding the front.

    Mirrors :func:`asymmetry.core.transform.grouping.apply_grouping_aligned`'s
    per-detector shift, but across runs for one detector. Returns the shifted
    arrays truncated to their common length and the common t0 bin.
    """
    common_t0 = max(0, max(int(t) for t in t0s))
    shifted: list[NDArray[np.float64]] = []
    for arr, t0 in zip(counts, t0s, strict=True):
        # common_t0 is the maximum, so offset is always >= 0 (front zero-pad).
        offset = common_t0 - int(t0)
        if offset == 0:
            shifted.append(np.asarray(arr, dtype=np.float64).copy())
        else:
            out = np.zeros(len(arr) + offset, dtype=np.float64)
            out[offset:] = arr
            shifted.append(out)
    min_len = min(len(a) for a in shifted)
    return [a[:min_len] for a in shifted], common_t0


def _combine_histograms(
    runs: list[Run], coefficients: list[float], *, cancel_check=None
) -> list[Histogram]:
    """Keep the ordinary path's whole-bin padded alignment for counts and noise."""
    aligned = [[] for _ in runs]
    for det in range(len(runs[0].histograms)):
        if cancel_check is not None:
            cancel_check()
        histograms = [run.histograms[det] for run in runs]
        t0s = [h.t0_bin for h in histograms]
        counts, common_t0 = _aligned_detector_arrays([h.counts for h in histograms], t0s)
        variances, _ = _aligned_detector_arrays(
            [h.counts if h.count_variance is None else h.count_variance for h in histograms], t0s
        )
        exact = run_t0_time_us(histograms, common_t0)
        for target, hist, values, variance in zip(
            aligned, histograms, counts, variances, strict=True
        ):
            target.append(
                replace(
                    hist,
                    counts=values,
                    t0_bin=common_t0,
                    t0_time_us=exact,
                    count_variance=variance if hist.count_variance is not None else None,
                )
            )
    summed = sum_period_histograms(aligned, coefficients=coefficients, cancel_check=cancel_check)
    return [
        replace(hist, good_bin_start=base.good_bin_start, good_bin_end=base.good_bin_end)
        for hist, base in zip(summed, aligned[0], strict=True)
    ]


def _combine_periodic(
    runs: list[Run],
    coefficients: list[float],
    *,
    sign: int,
    cancel_check=None,
) -> tuple[list[Histogram], dict[str, Any]]:
    """Apply coefficients to every counterpart on existing common raw-index support."""
    n_periods = len(runs[0].grouping["period_histograms"])
    summed_periods: list[list[Histogram]] = []
    for p in range(n_periods):
        per_run = [run.grouping["period_histograms"][p] for run in runs]
        # Correspondence is required for each counterpart, not only the flat view.
        error = _compatibility_error(
            [replace(run, histograms=h) for run, h in zip(runs, per_run, strict=True)]
        )
        if error is not None:
            raise CombineError(error)
        summed_periods.append(
            sum_period_histograms(per_run, coefficients=coefficients, cancel_check=cancel_check)
        )

    # These held arrays are inspection views. Original acquired records own
    # the correction normalisers and coefficient tables used by reduction.
    good = _accumulate_period_good_frames(runs if sign == 1 else runs[:1], n_periods)
    base_grouping = runs[0].grouping
    payload: dict[str, Any] = {
        "period_histograms": summed_periods,
        "period_good_frames": good,
        "period_mode": base_grouping.get("period_mode"),
    }
    dead = base_grouping.get("period_dead_time_us")
    if isinstance(dead, list):
        payload["period_dead_time_us"] = copy.deepcopy(dead)

    # The run's flat histograms mirror the first period set (the loader/RG path
    # reads period_histograms for reduction; the flat list is the red set).
    return list(summed_periods[0]), payload


def _accumulate_period_good_frames(runs: list[Run], n_periods: int) -> list[float]:
    """Element-wise sum of per-period good frames across runs."""
    totals = [0.0] * n_periods
    for run in runs:
        per = run.grouping.get("period_good_frames")
        if not isinstance(per, list):
            continue
        for p in range(min(n_periods, len(per))):
            try:
                totals[p] += float(per[p])
            except (TypeError, ValueError):
                continue
    return totals


# --- grouping / metadata ------------------------------------------------------


def _mirror_grouping(
    base: Run,
    period_payload: dict[str, Any] | None,
    exposure: float,
) -> dict[str, Any]:
    """Mirror the base run's grouping onto the combined run.

    Period payloads (which would otherwise carry the *base* run's per-period
    histograms) are stripped first and replaced with the freshly summed payload,
    for either sign.
    Frame totals label the inspection view; correction reads original inputs.
    """
    # ``t0_time_us`` is a per-run fact of the *base* run; the combined run's is
    # the mean over its own aligned detectors, which the combined histograms
    # carry, so drop it and let common_t0_time_us re-derive it.
    dropped = (*_PERIOD_KEYS, "t0_time_us", "reference_exposure", "period_reference_exposure")
    grouping = copy.deepcopy({k: v for k, v in base.grouping.items() if k not in dropped})
    if exposure > 0.0:
        grouping["good_frames"] = float(exposure)
    if period_payload is not None:
        for key, value in period_payload.items():
            if value is not None:
                grouping[key] = value
        if period_payload.get("period_good_frames"):
            grouping["good_frames"] = float(period_payload["period_good_frames"][0])
    return grouping


def _combined_metadata(
    runs: list[Run],
    *,
    scales: list[float],
    sign: int,
    subtract_method: str = "reference",
    number: int,
    label: str | None,
    exposure: float,
    negative_bins: int = 0,
) -> dict[str, Any]:
    """Build the combined run's metadata: identity, event-weighted scalars,
    spread keys, ``combined_from`` and the nested ``combination`` block (W3)."""
    weights = [max(good_frames(run.grouping, 0.0), 0.0) for run in runs]
    if not any(w > 0.0 for w in weights):
        weights = [1.0] * len(runs)

    source_numbers = [int(run.run_number) for run in runs]
    join = " + " if sign == 1 else " − "
    run_label = label if label is not None else join.join(str(n) for n in source_numbers)
    title = _common_title(runs, sign)

    temperature, temperature_spread = _weighted_scalar(runs, weights, "temperature")
    field, field_spread = _weighted_scalar(runs, weights, "field")

    good = float(exposure)

    metadata: dict[str, Any] = {
        "run_number": number,
        "run_label": run_label,
        "title": title,
        "temperature": temperature,
        "field": field,
        "combined_from": list(source_numbers),
    }
    if temperature_spread is not None:
        metadata["temperature_spread"] = temperature_spread
    if field_spread is not None:
        metadata["field_spread"] = field_spread

    constituents = [
        {
            "run_number": int(run.run_number),
            "source_file": run.source_file,
            "good_frames": float(max(good_frames(run.grouping, 0.0), 0.0)),
            "temperature": _scalar(run, "temperature"),
            "field": _scalar(run, "field"),
        }
        for run in runs
    ]
    if sign == 1:
        method = "coadd"
    else:
        method = "subtract_reference" if subtract_method == "reference" else "subtract_signed"
    combination: dict[str, Any] = {
        "method": method,
        "sign": sign,
        "scales": list(scales),
        "constituents": constituents,
        "good_frames": good,
        "independence_assumption": "distinct_observations_when_identity_unknown",
        "detector_correspondence": "channel_order",
    }
    if good > 0.0:
        metadata.setdefault("good_frames", good)
    if sign == -1:
        combination["negative_count_bins"] = int(negative_bins)
        if subtract_method == "reference":
            # The reference-run path records its single designated reference.
            combination["reference_run_number"] = int(runs[1].run_number)
            combination["reference_scale"] = float(scales[1])

    metadata["combination"] = combination
    return metadata


def _common_title(runs: list[Run], sign: int) -> str:
    titles = [str(run.metadata.get("title", "")).strip() for run in runs]
    non_empty = [t for t in titles if t]
    if non_empty and all(t == non_empty[0] for t in non_empty):
        return non_empty[0]
    verb = "Combined" if sign == 1 else "Difference of"
    return f"{verb} {len(runs)} runs"


def _scalar(run: Run, key: str) -> float | None:
    try:
        return float(run.metadata.get(key))
    except (TypeError, ValueError):
        return None


def _weighted_scalar(
    runs: list[Run],
    weights: list[float],
    key: str,
) -> tuple[float, tuple[float, float] | None]:
    """Event-weighted mean of a scalar metadata field and its (min, max) spread.

    Runs without a parseable value are skipped. Returns ``(0.0, None)`` when no
    run carries the field.
    """
    values: list[float] = []
    used_weights: list[float] = []
    for run, weight in zip(runs, weights, strict=True):
        value = _scalar(run, key)
        if value is None:
            continue
        values.append(value)
        used_weights.append(weight if weight > 0.0 else 0.0)
    if not values:
        return 0.0, None
    weight_total = sum(used_weights)
    if weight_total > 0.0:
        mean = float(np.average(values, weights=used_weights))
    else:
        mean = float(np.mean(values))
    spread = (float(min(values)), float(max(values))) if len(values) > 1 else None
    return mean, spread


def coadd_compatibility_error(runs):
    """Compare shared authored intent without comparing acquired correction/clock facts."""
    from asymmetry.core.project.profiles import profile_fingerprint_for_run, profile_from_payload

    if any(run is None or not run.histograms or not run.grouping.get("groups") for run in runs):
        return "Co-add requires identical grouping intent. Apply grouping to each source run before combining them."
    error = _compatibility_error(runs)
    if error is not None:
        return error
    fingerprint = profile_fingerprint_for_run(runs[0])
    profiles = [profile_from_payload(run.grouping, "coadd", fingerprint) for run in runs]
    if any(profile != profiles[0] for profile in profiles[1:]):
        return "Co-add requires identical grouping intent. Align groups, alpha/beta, binning, and authored timing/correction policies first."
    return None
