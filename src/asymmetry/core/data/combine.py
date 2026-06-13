"""Histogram-level run arithmetic: co-add and reference-run co-subtract.

This is the Qt-free kernel behind the Data Browser's "Co-add Selected" and
"Subtract Reference Run…" actions, and the scriptable entry point for combining
runs at the **raw-count** level.

The governing principle is *sum counts, then reduce* — never average reduced
asymmetry curves. Errors and any nonlinear correction (deadtime, α, background)
must see the total statistics, so the combined object is a first-class
:class:`~asymmetry.core.data.dataset.Run` with real summed histograms that the
normal grouping / deadtime / count-fit / MaxEnt pipelines can operate on.

Two operations share one entry point, :func:`combine_runs`:

* **Co-add** (``sign=+1``) — detector-wise count addition of repeated
  measurements. The summed counts stay Poisson, so the result reduces through
  the ordinary path (:func:`reduce_combined_run` delegates to
  :func:`asymmetry.core.simulate.reduce_run_to_dataset`).
* **Co-subtract** (``sign=-1``) — ``runs[0] − Σ scaleₖ·runsₖ``. Per-detector
  arithmetic goes through :func:`asymmetry.core.transform.background.subtract_scaled_counts`
  (the single count-level subtraction seam; F9 reconciliation), so variances
  add. A difference spectrum is no longer Poisson, so the per-detector
  propagated errors are carried on the run and consumed at reduction.

WiMDA reference: ``$WIMDA_SRC/src/muondata.pas`` 2418–2491 (``cosign=-1``
subtraction, frame/spill + per-period accumulation, ACC ``tshift`` alignment).
See ``docs/porting/run-arithmetic/`` for the full study and divergences.

This module must stay free of Qt / matplotlib / ``asymmetry.gui`` imports.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.io.periods import sum_period_histograms
from asymmetry.core.transform.asymmetry import (
    compute_asymmetry_with_count_errors,
    slice_to_good_window,
)
from asymmetry.core.transform.background import subtract_scaled_counts
from asymmetry.core.transform.grouping import good_frames, group_forward_backward

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.typing import NDArray

__all__ = [
    "CombineError",
    "coadd_member_windows",
    "combine_runs",
    "reduce_combined_run",
]

#: Bin-width agreement tolerance (µs) for the count-level compatibility check.
_BIN_WIDTH_RTOL = 1e-6

#: Grouping keys holding the per-period payload (mirrors simulate._PERIOD_KEYS).
_PERIOD_KEYS = (
    "period_histograms",
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
        Per-run multipliers (length ``len(runs)``). Defaults to all ones.
        ``runs[0]`` is always taken at its own scale (normally 1.0).
    run_number
        Run number for the combined run. Defaults to ``runs[0].run_number``
        (the GUI passes a negative synthetic id).
    label
        Optional explicit ``run_label``; otherwise built from the constituents
        (``"a + b"`` for add, ``"a − b"`` for subtract).

    Returns
    -------
    Run
        A first-class run with summed (or differenced) histograms, mirrored
        grouping, accumulated good frames, event-weighted scalar metadata with
        spread keys, and a ``metadata["combination"]`` provenance block. For a
        subtraction the per-detector propagated errors are stored under
        ``combination["detector_variance"]`` for :func:`reduce_combined_run`.

    Raises
    ------
    CombineError
        Fewer than two runs, empty histograms, or mismatched detector
        count / bin width.
    """
    sign = int(sign)
    if sign not in (1, -1):
        raise CombineError("sign must be +1 (co-add) or -1 (co-subtract)")

    runs = list(runs)
    if len(runs) < 2:
        raise CombineError("combine_runs needs at least two runs")
    if sign == -1 and len(runs) != 2:
        # The shipped co-subtract surface is reference-run only (one sample,
        # one reference); symmetric / N-run signed subtraction is a recorded
        # follow-on (docs/porting/run-arithmetic).
        raise CombineError("co-subtract takes exactly two runs (sample, reference)")

    if scales is None:
        scales = [1.0] * len(runs)
    scales = [float(s) for s in scales]
    if len(scales) != len(runs):
        raise CombineError("scales must match the number of runs")

    _validate_compatible(runs)

    detector_variance: list[NDArray[np.float64]] | None = None
    period_payload: dict[str, Any] | None = None
    negative_bins = 0

    if sign == 1 and _all_periodic(runs):
        # W12: sum per-period histograms through the one periods.py helper.
        histograms, period_payload = _combine_periodic(runs)
    elif sign == 1:
        histograms = _combine_histograms_add(runs, scales)
    else:
        histograms, detector_variance, negative_bins = _combine_histograms_subtract(runs, scales)

    base = runs[0]
    number = base.run_number if run_number is None else int(run_number)
    # Exposure for the combined run: co-add sums good frames (the summed counts
    # correspond to the summed frames, so the deadtime normaliser rate =
    # counts/(dt·frames) stays correct); a reference subtract keeps the sample's
    # exposure (the reference is frame-scaled and consumed).
    if sign == 1:
        exposure = float(sum(max(good_frames(run.grouping, 0.0), 0.0) for run in runs))
    else:
        exposure = max(good_frames(base.grouping, 0.0), 0.0)
    grouping = _mirror_grouping(base, period_payload, exposure)
    metadata = _combined_metadata(
        runs,
        scales=scales,
        sign=sign,
        number=number,
        label=label,
        exposure=exposure,
        detector_variance=detector_variance,
        negative_bins=negative_bins,
    )

    return Run(
        run_number=number,
        histograms=histograms,
        metadata=metadata,
        grouping=grouping,
        source_file="",
    )


def reduce_combined_run(run: Run) -> MuonDataset:
    """Reduce a run produced by :func:`combine_runs` to a :class:`MuonDataset`.

    Co-added runs hold Poisson counts, so this delegates to the ordinary
    :func:`asymmetry.core.simulate.reduce_run_to_dataset`. Subtraction runs hold
    a difference spectrum whose errors are *not* ``√counts``; this path groups
    the carried per-detector variances in quadrature and forms the asymmetry
    with :func:`compute_asymmetry_with_count_errors`, so the displayed error
    bars are correct.
    """
    # Imported lazily to avoid a module-load cycle (simulate imports widely).
    from asymmetry.core.simulate import reduce_run_to_dataset

    grouping = run.grouping if isinstance(run.grouping, dict) else {}
    combination = run.metadata.get("combination") if isinstance(run.metadata, dict) else None
    is_subtract = isinstance(combination, dict) and int(combination.get("sign", 1)) == -1

    if not is_subtract:
        return reduce_run_to_dataset(run)

    if not run.histograms:
        raise ValueError("Reduction requires a run with detector histograms.")

    variance = combination.get("detector_variance")
    variance_hists = _variance_histograms(run.histograms, variance)

    fb = group_forward_backward(run.histograms, grouping)
    var_fb = group_forward_backward(variance_hists, grouping)
    n = min(fb.forward.size, fb.backward.size, var_fb.forward.size, var_fb.backward.size)
    asymmetry, error = compute_asymmetry_with_count_errors(
        fb.forward[:n],
        fb.backward[:n],
        np.sqrt(np.maximum(var_fb.forward[:n], 0.0)),
        np.sqrt(np.maximum(var_fb.backward[:n], 0.0)),
        fb.alpha,
    )
    asymmetry = asymmetry * 100.0
    error = error * 100.0

    time, asymmetry, error = slice_to_good_window(
        asymmetry,
        error,
        grouping,
        common_t0=fb.common_t0,
        bin_width=float(run.histograms[0].bin_width),
    )

    metadata = dict(run.metadata)
    metadata.setdefault("run_number", run.run_number)
    metadata.setdefault("run_label", str(run.run_number))
    return MuonDataset(time=time, asymmetry=asymmetry, error=error, metadata=metadata, run=run)


# --- in-batch co-add windowing ------------------------------------------------


def coadd_member_windows(
    n_members: int,
    *,
    mode: str,
    window: int,
) -> list[list[int]]:
    """Index windows for in-batch co-add of successive batch-series members.

    Mirrors WiMDA's ``BatchFit.pas`` sequential co-add (the "+ N runs" control):
    a window co-adds ``window`` successive members. The two stepping modes match
    WiMDA's Smooth/Bin radio buttons (``$WIMDA_SRC/src/BatchFit.pas`` ~375):

    * ``"smooth"`` — sliding window, **step 1**: windows
      ``[0, W)``, ``[1, 1 + W)``, … (``inc(i)``). Yields ``n - W + 1`` windows.
    * ``"bin"`` — non-overlapping, **step W**: windows ``[0, W)``, ``[W, 2W)``, …
      (``i := i + jump + 1``). Yields ``n // W`` windows.

    In both modes WiMDA's loop guard (``until i + jump > nff``) requires a *full*
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


def _validate_compatible(runs: list[Run]) -> None:
    """Enforce the count-level invariants: histograms, detector count, width."""
    for run in runs:
        if not run.histograms:
            raise CombineError(f"run {run.run_number} has no histograms to combine")

    n_detectors = len(runs[0].histograms)
    for run in runs[1:]:
        if len(run.histograms) != n_detectors:
            raise CombineError(
                "runs have different detector counts "
                f"({n_detectors} vs {len(run.histograms)}); cannot combine"
            )

    for det in range(n_detectors):
        width0 = float(runs[0].histograms[det].bin_width)
        for run in runs[1:]:
            width = float(run.histograms[det].bin_width)
            if not np.isclose(width, width0, rtol=_BIN_WIDTH_RTOL, atol=0.0):
                raise CombineError(
                    f"detector {det + 1} bin widths differ ({width0} vs {width} µs); cannot combine"
                )


def _all_periodic(runs: list[Run]) -> bool:
    """True when every run carries a >=2-period histogram payload."""
    for run in runs:
        grouping = run.grouping if isinstance(run.grouping, dict) else {}
        periods = grouping.get("period_histograms")
        if not (isinstance(periods, list) and len(periods) >= 2):
            return False
    counts = {len(run.grouping["period_histograms"]) for run in runs}
    return len(counts) == 1


# --- count arithmetic ---------------------------------------------------------


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


def _combine_histograms_add(
    runs: list[Run],
    scales: list[float],
) -> list[Histogram]:
    """Detector-wise scaled count sum across runs, aligning per-detector t0."""
    n_detectors = len(runs[0].histograms)
    out: list[Histogram] = []
    for det in range(n_detectors):
        arrays = [np.asarray(run.histograms[det].counts, dtype=np.float64) for run in runs]
        t0s = [int(run.histograms[det].t0_bin) for run in runs]
        shifted, common_t0 = _aligned_detector_arrays(arrays, t0s)
        total = np.zeros_like(shifted[0])
        for arr, scale in zip(shifted, scales, strict=True):
            total += scale * arr
        out.append(_clone_geometry(runs[0].histograms[det], total, common_t0))
    return out


def _combine_histograms_subtract(
    runs: list[Run],
    scales: list[float],
) -> tuple[list[Histogram], list[NDArray[np.float64]], int]:
    """Detector-wise ``sample − scale·reference`` via ``subtract_scaled_counts``.

    Returns the difference histograms, the per-detector variances (error²) so
    the reduction can propagate them, and the count of negative difference bins
    (the unphysical-counts guard, RA5). ``runs`` is ``[sample, reference]``; the
    reference scale is ``scales[1]``. The sample is taken at unit scale (a
    reference subtraction is ``sample − scale·reference``, so ``scales[0]`` is
    1.0 by contract and recorded for provenance only); passing it through the
    chokepoint's variance term would give the wrong, linear-in-scale variance.
    """
    sample, reference = runs
    reference_scale = scales[1]
    n_detectors = len(sample.histograms)
    out: list[Histogram] = []
    variances: list[NDArray[np.float64]] = []
    negative_bins = 0
    for det in range(n_detectors):
        arrays = [
            np.asarray(sample.histograms[det].counts, dtype=np.float64),
            np.asarray(reference.histograms[det].counts, dtype=np.float64),
        ]
        t0s = [int(sample.histograms[det].t0_bin), int(reference.histograms[det].t0_bin)]
        (s_counts, r_counts), common_t0 = _aligned_detector_arrays(arrays, t0s)
        # subtract_scaled_counts is the single count-level subtraction seam
        # (F9): difference = sample − reference_scale·reference,
        # variance = sample + reference_scale²·reference.
        diff, error = subtract_scaled_counts(s_counts, r_counts, reference_scale)
        negative_bins += int(np.count_nonzero(diff < 0.0))
        out.append(_clone_geometry(sample.histograms[det], diff, common_t0))
        variances.append(error * error)
    return out, variances, negative_bins


def _combine_periodic(
    runs: list[Run],
) -> tuple[list[Histogram], dict[str, Any]]:
    """Co-add period-mode runs: sum each period set across runs (W12).

    Uses :func:`asymmetry.core.io.periods.sum_period_histograms` so there is no
    second period-summing implementation. Periods of repeated measurements
    share detector layout, bin width and t0, so the helper's detector-wise sum
    is exact.
    """
    n_periods = len(runs[0].grouping["period_histograms"])
    summed_periods: list[list[Histogram]] = []
    for p in range(n_periods):
        per_run = [run.grouping["period_histograms"][p] for run in runs]
        summed_periods.append(sum_period_histograms(per_run))

    # Per-period good frames accumulate across runs; deadtime tables come from
    # the first run (co-add requires identical grouping at the GUI gate).
    good = _accumulate_period_good_frames(runs, n_periods)
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


def _clone_geometry(
    template: Histogram,
    counts: NDArray[np.float64],
    t0_bin: int,
) -> Histogram:
    """A histogram with ``counts`` but ``template``'s bin geometry at ``t0_bin``."""
    return Histogram(
        counts=np.asarray(counts, dtype=np.float64),
        bin_width=float(template.bin_width),
        t0_bin=int(t0_bin),
        good_bin_start=int(template.good_bin_start),
        good_bin_end=int(template.good_bin_end),
    )


def _variance_histograms(
    histograms: list[Histogram],
    variance: Any,
) -> list[Histogram]:
    """Wrap per-detector variances as histograms so they group like counts.

    Falls back to the difference counts' own values (treated as Poisson) only
    when the stored variance is missing or malformed — a defensive path that
    should not trigger for runs built by :func:`combine_runs`.
    """
    if isinstance(variance, list) and len(variance) == len(histograms):
        return [
            _clone_geometry(hist, np.asarray(var, dtype=np.float64), hist.t0_bin)
            for hist, var in zip(histograms, variance, strict=True)
        ]
    return [
        _clone_geometry(hist, np.abs(np.asarray(hist.counts, dtype=np.float64)), hist.t0_bin)
        for hist in histograms
    ]


# --- grouping / metadata ------------------------------------------------------


def _mirror_grouping(
    base: Run,
    period_payload: dict[str, Any] | None,
    exposure: float,
) -> dict[str, Any]:
    """Mirror the base run's grouping onto the combined run.

    Period payloads (which would otherwise carry the *base* run's per-period
    histograms) are stripped first and replaced with the freshly summed payload,
    or dropped entirely for a subtraction (period co-subtract is out of scope).
    The run-level ``good_frames`` is set to the accumulated ``exposure`` so the
    deadtime normaliser sees the combined frame count.
    """
    grouping = copy.deepcopy({k: v for k, v in base.grouping.items() if k not in _PERIOD_KEYS})
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
    number: int,
    label: str | None,
    exposure: float,
    detector_variance: list[NDArray[np.float64]] | None,
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
    combination: dict[str, Any] = {
        "method": "coadd" if sign == 1 else "subtract_reference",
        "sign": sign,
        "scales": list(scales),
        "constituents": constituents,
        "good_frames": good,
    }
    if good > 0.0:
        metadata.setdefault("good_frames", good)
    if sign == -1:
        combination["reference_run_number"] = int(runs[1].run_number)
        combination["reference_scale"] = float(scales[1])
        combination["negative_count_bins"] = int(negative_bins)
        if detector_variance is not None:
            combination["detector_variance"] = detector_variance

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
