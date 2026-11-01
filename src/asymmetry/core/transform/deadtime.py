"""Deadtime correction utilities for raw detector histograms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

from asymmetry.core.data.dataset import Histogram
from asymmetry.core.utils.constants import MUON_LIFETIME_US
from asymmetry.core.utils.perf import perf_timer


class DeadtimeCorrectionError(ValueError):
    """A correction outside its raw-count or inverse-model domain."""

    def __init__(self, reason, *, detector=None, bin_index=None, occupancy=None):
        self.detector = detector
        self.bin_index = bin_index
        self.occupancy = occupancy
        location = "" if detector is None else f" for detector {detector + 1}"
        super().__init__(f"Deadtime correction rejected{location}: {reason}")


def _deadtime_number(value, name, *, detector=None, positive=False) -> float:
    """Parse a finite calibration or normalisation scalar at an input boundary."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise DeadtimeCorrectionError(f"{name} must be finite", detector=detector) from None
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(number) or (positive and number <= 0):
        requirement = "finite and > 0" if positive else "finite"
        raise DeadtimeCorrectionError(f"{name} must be {requirement}", detector=detector)
    return number


@dataclass(frozen=True)
class DeadtimeCorrection:
    """Finite signed detector coefficients with an explicit acquisition normalisation."""

    tau_us: tuple[float, ...]
    good_frames: float | None

    def __post_init__(self):
        coefficients = tuple(
            _deadtime_number(value, "deadtime coefficient (µs)", detector=i)
            for i, value in enumerate(self.tau_us)
        )
        object.__setattr__(self, "tau_us", coefficients)
        if any(value != 0 for value in coefficients):
            object.__setattr__(
                self,
                "good_frames",
                _deadtime_number(self.good_frames, "good frames", positive=True),
            )

    @classmethod
    def from_grouping(cls, grouping: dict, n_histograms: int) -> DeadtimeCorrection:
        return cls(tuple(grouping["dead_time_us"][:n_histograms]), grouping.get("good_frames"))


def deadtime_occupancy(counts, tau_us, bin_width_us, good_frames, *, detector=None):
    """Return N·τ/(Δt·F) for valid raw counts and an invertible signed correction."""
    tau = _deadtime_number(tau_us, "deadtime coefficient (µs)", detector=detector)
    width = _deadtime_number(bin_width_us, "bin width (µs)", detector=detector, positive=True)
    frames = _deadtime_number(good_frames, "good frames", detector=detector, positive=True)
    try:
        raw = np.asarray(counts)
        n = raw.astype(np.float64) if not np.iscomplexobj(raw) else None
    except (TypeError, ValueError, OverflowError):
        n = None
    if n is None or n.ndim != 1 or not np.all(np.isfinite(n) & (n >= 0)):
        raise DeadtimeCorrectionError(
            "raw counts must be finite and non-negative", detector=detector
        )
    with np.errstate(over="ignore", invalid="ignore"):
        occupancy = (n / frames) * (tau / width)
    if not np.all(np.isfinite(occupancy)):
        raise DeadtimeCorrectionError("N·τ/(Δt·F) must be finite", detector=detector)
    saturated = occupancy >= 1
    if np.any(saturated):
        index = int(np.argmax(occupancy))
        peak = float(occupancy[index])
        raise DeadtimeCorrectionError(
            f"occupancy N·τ/(Δt·F) reaches {peak:.6g} at bin {index} "
            f"({np.count_nonzero(saturated)} bins at or above 1); "
            "the non-paralysable model requires < 1.",
            detector=detector,
            bin_index=index,
            occupancy=peak,
        )
    return occupancy


def apply_deadtime_correction(
    counts,
    tau_us: float,
    bin_width_us: float,
    *,
    num_good_frames: float,
    detector: int | None = None,
):
    """Invert the non-paralysable model on every raw bin, rejecting N·τ/(Δt·F) ≥ 1.

    ``N_corr = N / (1 - N * tau / (dt * n_frames))``
    """
    # ISIS calibration coefficients are signed fit parameters, in µs. A negative
    # coefficient reduces counts; it is not a negative physical recovery interval.
    occupancy = deadtime_occupancy(counts, tau_us, bin_width_us, num_good_frames, detector=detector)
    with np.errstate(over="ignore"):
        corrected = np.asarray(counts, dtype=np.float64) / (1 - occupancy)
    if not np.all(np.isfinite(corrected)):
        raise DeadtimeCorrectionError("corrected counts must be finite", detector=detector)
    return corrected


def estimate_deadtime_from_histograms(
    histograms: list[Histogram],
    *,
    t_good_offset: int = 0,
    last_good_bin: int | None = None,
    num_good_frames: float = 1.0,
    max_bins: int = 400,
    detector_t0_bins: list[int] | None = None,
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
        detector_t0_bins=detector_t0_bins,
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
    detector_t0_bins: list[int] | None = None,
) -> list[float] | None:
    """Calibrate per-detector deadtime values from early-time count data.

    This mirrors WiMDA's ``Cal`` button behavior: fit each detector histogram
    independently using the same ``countfit`` model and return one deadtime
    value per detector in microseconds.

    ``detector_t0_bins`` optionally overrides each detector's alignment t0
    (D10) so a Manual/Auto-detect t0 policy moves the calibration window with
    it; ``None`` uses each histogram's own file ``t0_bin``.
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
    for i, histogram in enumerate(histograms):
        counts = np.asarray(histogram.counts, dtype=np.float64)
        if counts.size <= 0:
            return None

        t0_bin = int(detector_t0_bins[i]) if detector_t0_bins is not None else int(histogram.t0_bin)
        start = max(0, t0_bin + offset)
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
            if expected_count > 0:
                start_index = 1
            else:
                expected_count = None
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
            except ValueError:
                continue
            tau_us = _deadtime_number(
                tokens[1], "deadtime coefficient (µs)", detector=detector_index - 1
            )
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
    return [
        _deadtime_number(value, "deadtime coefficient (µs)", detector=i)
        for i, value in enumerate(values)
    ]


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
            values[i] = _deadtime_number(existing[i], "deadtime coefficient (µs)", detector=i)

    if detector_indices is None:
        targets = list(range(n))
    else:
        targets = [int(i) for i in detector_indices if 0 <= int(i) < n]

    incoming = _coerce_deadtime_value(dead_time_us_value, n)
    before = {i: values[i] for i in targets}
    for i in targets:
        dt = incoming[i]
        values[i] = _deadtime_number(
            values[i] + dt if additive else dt, "deadtime coefficient (µs)", detector=i
        )
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
            result[i] = _deadtime_number(seq[i], "deadtime coefficient (µs)", detector=i)
        return result
    return [_deadtime_number(dead_time_us_value, "deadtime coefficient (µs)")] * n


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
    """Return whether a complete non-zero table is present, pending domain validation."""
    if not isinstance(grouping, dict):
        return False
    dead_time_us = grouping.get("dead_time_us")
    if not isinstance(dead_time_us, list):
        return False
    required = max(1, int(n_histograms))
    return len(dead_time_us) >= required and any(value != 0 for value in dead_time_us)


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
            correction = DeadtimeCorrection.from_grouping(grouping, len(histograms))
            corrected, applied = _prepare_histograms_with_resolved_deadtime(
                histograms, grouping, correction
            )
            perf.detail(applied=applied)
            return corrected, applied

        perf.detail(applied=False)
        return list(histograms), False


def _prepare_histograms_with_resolved_deadtime(
    histograms: list[Histogram],
    grouping: dict,
    correction: DeadtimeCorrection,
) -> tuple[list[Histogram], bool]:
    applied_any = any(value != 0 for value in correction.tau_us)
    if not applied_any:
        return list(histograms), False
    corrected: list[Histogram] = []
    for i, hist in enumerate(histograms):
        tau_us = correction.tau_us[i]
        counts = apply_deadtime_correction(
            hist.counts,
            tau_us,
            hist.bin_width,
            num_good_frames=correction.good_frames,
            detector=i,
        )

        corrected.append(_copy_histogram_with_counts(hist, counts))

    method = str(grouping.get("deadtime_method", "")).strip().lower()
    grouping["deadtime_method"] = method or "file"
    return corrected, True


def _copy_histogram_with_counts(histogram: Histogram, counts) -> Histogram:
    return Histogram(
        counts=np.asarray(counts, dtype=np.float64),
        bin_width=histogram.bin_width,
        t0_bin=histogram.t0_bin,
        good_bin_start=histogram.good_bin_start,
        good_bin_end=histogram.good_bin_end,
        t0_time_us=histogram.t0_time_us,
    )
