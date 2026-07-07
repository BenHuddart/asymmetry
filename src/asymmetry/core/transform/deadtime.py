"""Deadtime correction utilities for raw detector histograms."""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from asymmetry.core.data.dataset import Histogram
from asymmetry.core.transform.grouping import good_frames
from asymmetry.core.utils.constants import MUON_LIFETIME_US
from asymmetry.core.utils.perf import perf_timer


def apply_deadtime_correction(
    counts,
    tau_us: float,
    bin_width_us: float,
    *,
    num_good_frames: float = 1.0,
):
    """Apply the musrfit/Mantid non-paralyzable deadtime correction.

    ``N_corr = N / (1 - N * tau / (dt * n_frames))``
    """
    n = np.asarray(counts, dtype=np.float64)
    tau_us = float(tau_us)
    if not np.isfinite(tau_us) or tau_us == 0.0 or bin_width_us <= 0.0 or num_good_frames <= 0.0:
        return n.copy()

    denom = 1.0 - (n * tau_us / (float(bin_width_us) * float(num_good_frames)))
    denom = np.clip(denom, 1.0e-6, None)
    return n / denom


def estimate_deadtime_from_histograms(
    histograms: list[Histogram],
    *,
    t_good_offset: int = 0,
    last_good_bin: int | None = None,
    num_good_frames: float = 1.0,
    max_bins: int = 400,
) -> float | None:
    """Estimate a uniform deadtime value from early-time detector counts.

    The fit mirrors WiMDA's ``countfit`` workflow: average the early-time count
    rate per detector and fit a muon-lifetime decay with a deadtime-loss term.
    The returned value is a single detector deadtime in microseconds that can be
    broadcast across grouped histograms.
    """
    calibrated = calibrate_deadtime_from_histograms(
        histograms,
        t_good_offset=t_good_offset,
        last_good_bin=last_good_bin,
        num_good_frames=num_good_frames,
        max_bins=max_bins,
    )
    if not calibrated:
        return None
    return float(np.mean(calibrated))


def calibrate_deadtime_from_histograms(
    histograms: list[Histogram],
    *,
    t_good_offset: int = 0,
    last_good_bin: int | None = None,
    num_good_frames: float = 1.0,
    max_bins: int = 400,
) -> list[float] | None:
    """Calibrate per-detector deadtime values from early-time count data.

    This mirrors WiMDA's ``Cal`` button behavior: fit each detector histogram
    independently using the same ``countfit`` model and return one deadtime
    value per detector in microseconds.
    """
    if not histograms:
        return None

    try:
        frame_scale = float(num_good_frames) * float(histograms[0].bin_width)
    except (TypeError, ValueError):
        return None
    if frame_scale <= 0.0:
        return None

    try:
        offset = max(0, int(t_good_offset))
        window_limit = max(1, int(max_bins))
    except (TypeError, ValueError):
        return None

    calibrated: list[float] = []
    for histogram in histograms:
        counts = np.asarray(histogram.counts, dtype=np.float64)
        if counts.size <= 0:
            return None

        start = max(0, int(histogram.t0_bin) + offset)
        end = counts.size - 1
        if last_good_bin is not None:
            try:
                end = min(end, int(last_good_bin))
            except (TypeError, ValueError):
                pass
        if start > end:
            return None

        n_fit = min(window_limit, end - start + 1)
        if n_fit < 3:
            return None

        observed = counts[start : start + n_fit]
        sigma = np.sqrt(np.clip(observed, 1.0, None))
        times_us = (np.arange(n_fit, dtype=np.float64) + 1.0) * float(histogram.bin_width)
        tau_us = _fit_deadtime_tau(times_us, observed, sigma, frame_scale)
        if tau_us is None:
            return None
        calibrated.append(tau_us)

    return calibrated


def _fit_deadtime_tau(
    times_us: np.ndarray,
    observed: np.ndarray,
    sigma: np.ndarray,
    frame_scale: float,
) -> float | None:
    """Fit the WiMDA deadtime model and return deadtime in microseconds."""
    if observed.size < 3:
        return None

    def _countfit(time_us, amplitude, tau_us):
        cc = amplitude * np.exp(-time_us / MUON_LIFETIME_US)
        n_rate = cc / frame_scale
        return cc * (1.0 - n_rate * MUON_LIFETIME_US * (1.0 - np.exp(-tau_us / MUON_LIFETIME_US)))

    guess_amplitude = max(float(observed[0]), 1.0)
    tau_upper = max(0.5, 10.0 * float(times_us[0]))
    try:
        params, _ = curve_fit(
            _countfit,
            times_us,
            observed,
            p0=(guess_amplitude, 0.01),
            sigma=sigma,
            absolute_sigma=True,
            bounds=((0.0, 0.0), (np.inf, tau_upper)),
            maxfev=10000,
        )
    except (RuntimeError, ValueError, TypeError):
        return None

    try:
        tau_us = float(params[1])
    except (TypeError, ValueError, IndexError):
        return None
    if not np.isfinite(tau_us):
        return None
    return max(0.0, tau_us)


def parse_deadtime_calibration_text(text: str, *, n_histograms: int | None = None) -> list[float]:
    """Parse a WiMDA-style deadtime calibration text file.

    The accepted format is the simple line-oriented form written by WiMDA:

    - optional first line with the histogram count
    - subsequent lines of ``<index> <tau_us>``
    - optional trailing metadata lines such as ``Run 1234``
    """
    lines = [
        line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")
    ]
    if not lines:
        raise ValueError("Deadtime calibration file is empty.")

    expected_count = None
    start_index = 0
    first_tokens = lines[0].split()
    if len(first_tokens) == 1:
        try:
            expected_count = int(first_tokens[0])
            start_index = 1
        except ValueError:
            expected_count = None

    indexed_values: dict[int, float] = {}
    sequential_values: list[float] = []
    for line in lines[start_index:]:
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0].lower() == "run":
            break
        if len(tokens) >= 2:
            try:
                detector_index = int(tokens[0])
                tau_us = float(tokens[1])
            except ValueError:
                continue
            indexed_values[detector_index] = tau_us
            continue
        try:
            sequential_values.append(float(tokens[0]))
        except ValueError:
            continue

    if indexed_values:
        count = expected_count or max(indexed_values)
        values = [0.0] * count
        for detector_index, tau_us in indexed_values.items():
            if detector_index <= 0:
                continue
            if detector_index > len(values):
                values.extend([0.0] * (detector_index - len(values)))
            values[detector_index - 1] = tau_us
    else:
        values = sequential_values

    if expected_count is not None and len(values) != expected_count:
        raise ValueError(
            f"Deadtime calibration expected {expected_count} values but parsed {len(values)}."
        )
    if n_histograms is not None and len(values) != int(n_histograms):
        raise ValueError(
            f"Deadtime calibration provides {len(values)} values but the run has {int(n_histograms)} histograms."
        )
    return [float(value) for value in values]


#: Higher-order count-loss coefficients carried alongside the promoted DT0.
DEADTIME_MODEL_TERM_NAMES: tuple[str, ...] = ("DT1", "C2", "C3", "C4")


def promote_deadtime_to_grouping(
    grouping: dict,
    dead_time_us_value: float | list[float] | tuple[float, ...] | np.ndarray,
    *,
    n_histograms: int,
    detector_indices: list[int] | None = None,
    additive: bool = False,
    model: str | None = None,
    extra_terms: dict[str, float] | None = None,
    method: str = "value",
) -> dict[str, dict[int, float]]:
    """Write a fitted deadtime (µs) into the grouping correction (WiMDA Send-to-Group).

    Mirrors WiMDA's ``SendToGroupClick``: a deadtime obtained from a count fit is
    promoted into the grouping's per-detector ``dead_time_us`` so the next
    reduction applies it. ``additive`` accumulates onto the existing value
    (WiMDA's ``DTmodelChanges``) instead of replacing it; ``detector_indices``
    restricts the write to the fitted group's detectors (default: all). The
    correction is enabled and the ``deadtime_method`` marked ``value`` by
    default; pass ``method`` to record a different provenance label (e.g.
    ``"maxent_fit"`` for the MaxEnt calibration route).

    ``dead_time_us_value`` is either a single deadtime broadcast to every target
    detector (the count-fit DT₀ case) or a per-detector sequence — one value per
    detector index — for calibrators that fit each detector independently (the
    MaxEnt route). A per-detector sequence aligns to detector index ``i``;
    targets beyond its length get ``0.0``.

    For a polynomial or power-law count-loss fit, pass the loss ``model`` and its
    higher-order ``extra_terms`` (``DT1``/``C2``/``C3``/``C4``). The dominant
    ``DT0`` term still drives the per-detector ``dead_time_us`` that the reduction
    applies (Asymmetry's grouping stores a single non-paralyzable deadtime per
    histogram), while the model name and the higher-order coefficients are
    recorded in ``deadtime_model`` / ``deadtime_model_terms`` so the full
    calibration round-trips with the run. ``additive`` accumulates the extra
    terms too.

    Returns ``{"before": {idx: value}, "after": {idx: value}}`` for the affected
    detectors, so the GUI can show the change before/after.
    """
    n = max(0, int(n_histograms))
    values = [0.0] * n
    existing = grouping.get("dead_time_us")
    if isinstance(existing, list):
        for i in range(min(len(existing), n)):
            try:
                values[i] = float(existing[i])
            except (TypeError, ValueError):
                values[i] = 0.0

    if detector_indices is None:
        targets = list(range(n))
    else:
        targets = [int(i) for i in detector_indices if 0 <= int(i) < n]

    incoming = _coerce_deadtime_value(dead_time_us_value, n)
    before = {i: values[i] for i in targets}
    for i in targets:
        dt = incoming[i]
        values[i] = values[i] + dt if additive else dt
    after = {i: values[i] for i in targets}

    grouping["dead_time_us"] = values
    grouping["deadtime_correction"] = True
    grouping["deadtime_method"] = str(method)

    if model is not None or extra_terms:
        _promote_deadtime_model_terms(grouping, model, extra_terms, additive=additive)
    return {"before": before, "after": after}


def _coerce_deadtime_value(
    dead_time_us_value: float | list[float] | tuple[float, ...] | np.ndarray,
    n: int,
) -> list[float]:
    """Return one deadtime per detector index in ``[0, n)``.

    A scalar is broadcast to every detector; a sequence aligns by index and
    pads short tails with ``0.0`` so per-detector calibrations (MaxEnt) and the
    scalar count-fit DT₀ share one write path.
    """
    if isinstance(dead_time_us_value, (list, tuple, np.ndarray)):
        seq = list(dead_time_us_value)
        result = [0.0] * n
        for i in range(min(len(seq), n)):
            try:
                result[i] = float(seq[i])
            except (TypeError, ValueError):
                result[i] = 0.0
        return result
    return [float(dead_time_us_value)] * n


def _promote_deadtime_model_terms(
    grouping: dict,
    model: str | None,
    extra_terms: dict[str, float] | None,
    *,
    additive: bool,
) -> None:
    """Record the count-loss model name and higher-order coefficients in grouping."""
    if model is not None:
        grouping["deadtime_model"] = str(model)
    incoming = {
        name: float(value)
        for name, value in (extra_terms or {}).items()
        if name in DEADTIME_MODEL_TERM_NAMES and value not in (None, 0.0)
    }
    if not incoming:
        return
    stored = grouping.get("deadtime_model_terms")
    stored = dict(stored) if isinstance(stored, dict) else {}
    for name, value in incoming.items():
        if additive:
            try:
                base = float(stored.get(name, 0.0))
            except (TypeError, ValueError):
                base = 0.0
            stored[name] = base + value
        else:
            stored[name] = value
    grouping["deadtime_model_terms"] = stored


def has_file_deadtime(grouping: dict, n_histograms: int = 0) -> bool:
    """Return ``True`` when grouping metadata contains file deadtime values."""
    if not isinstance(grouping, dict):
        return False
    method = str(grouping.get("deadtime_method", "")).strip().lower()
    if method not in {"", "file"}:
        return False
    return has_resolved_deadtime(grouping, n_histograms)


def has_resolved_deadtime(grouping: dict, n_histograms: int = 0) -> bool:
    """Return ``True`` when grouping metadata contains usable deadtime values."""
    if not isinstance(grouping, dict):
        return False
    dead_time_us = grouping.get("dead_time_us")
    if not isinstance(dead_time_us, list):
        return False
    required = max(1, int(n_histograms))
    return len(dead_time_us) >= required and any(_finite_nonzero(value) for value in dead_time_us)


def prepare_histograms_with_deadtime(
    histograms: list[Histogram],
    grouping: dict,
    use_deadtime: bool,
) -> tuple[list[Histogram], bool]:
    """Return histograms with optional deadtime correction applied."""
    with perf_timer(
        "core.deadtime.prepare",
        n_detectors=len(histograms),
        use_deadtime=use_deadtime,
    ) as perf:
        if not use_deadtime:
            perf.detail(applied=False)
            return list(histograms), False

        dead_time_us = grouping.get("dead_time_us") if isinstance(grouping, dict) else None
        if isinstance(dead_time_us, list) and len(dead_time_us) >= len(histograms):
            corrected, applied = _prepare_histograms_with_resolved_deadtime(
                histograms, grouping, dead_time_us
            )
            perf.detail(applied=applied)
            return corrected, applied

        perf.detail(applied=False)
        return list(histograms), False


def _prepare_histograms_with_resolved_deadtime(
    histograms: list[Histogram],
    grouping: dict,
    dead_time_us: list,
) -> tuple[list[Histogram], bool]:
    frames = good_frames(grouping)

    corrected: list[Histogram] = []
    applied_any = False
    for i, hist in enumerate(histograms):
        try:
            tau_us = float(dead_time_us[i])
        except (TypeError, ValueError):
            tau_us = 0.0

        counts = np.asarray(hist.counts, dtype=np.float64)
        if tau_us != 0.0 and np.isfinite(tau_us):
            counts = apply_deadtime_correction(
                counts,
                tau_us,
                hist.bin_width,
                num_good_frames=frames,
            )
            applied_any = True

        corrected.append(_copy_histogram_with_counts(hist, counts))

    if applied_any and isinstance(grouping, dict):
        method = str(grouping.get("deadtime_method", "")).strip().lower()
        grouping["deadtime_method"] = method or "file"

    return corrected, applied_any


def _finite_nonzero(value) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(number) and number != 0.0)


def _copy_histogram_with_counts(histogram: Histogram, counts) -> Histogram:
    return Histogram(
        counts=np.asarray(counts, dtype=np.float64),
        bin_width=histogram.bin_width,
        t0_bin=histogram.t0_bin,
        good_bin_start=histogram.good_bin_start,
        good_bin_end=histogram.good_bin_end,
    )
