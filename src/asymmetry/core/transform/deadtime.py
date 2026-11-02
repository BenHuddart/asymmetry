"""Deadtime correction utilities for raw detector histograms."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import CancelledError
from dataclasses import dataclass

import numpy as np
from iminuit import Minuit

from asymmetry.core.data.dataset import Histogram
from asymmetry.core.transform.corrections import (
    CorrectionOutcome,
    CorrectionOutcomes,
    CorrectionStatus,
)
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
    requirement = "finite and > 0" if positive else "finite"
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise DeadtimeCorrectionError(f"{name} must be {requirement}", detector=detector) from None
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(number) or (positive and number <= 0):
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
    num_good_frames: float,
    max_bins: int = 400,
    detector_t0_bins: list[int] | None = None,
) -> float | None:
    """Estimate a uniform deadtime value from early-time detector counts.

    Average the independently fitted rational-model coefficients across detectors.
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
    num_good_frames: float,
    max_bins: int = 400,
    detector_t0_bins: list[int] | None = None,
) -> list[float] | None:
    """Return per-detector coefficients in µs from the rational decay model."""
    result = fit_deadtime_calibration(
        histograms,
        num_good_frames=num_good_frames,
        t_good_offset=t_good_offset,
        last_good_bin=last_good_bin,
        max_bins=max_bins,
        detector_t0_bins=detector_t0_bins,
    )
    return None if result is None else list(result.tau_us)


@dataclass(frozen=True)
class DeadtimeCalibration:
    """Total per-detector coefficients and uncertainties in µs, ready for promotion."""

    tau_us: tuple[float, ...]
    errors_us: tuple[float | None, ...]
    method: str
    source_run: int | None

    def __post_init__(self):
        if not self.tau_us or len(self.errors_us) != len(self.tau_us):
            raise ValueError("a calibration requires one coefficient and error per detector")
        if self.method not in {"count_fit", "calibrate", "estimate", "maxent_fit"}:
            raise ValueError("unknown deadtime calibration method")
        if self.source_run is not None and (
            not isinstance(self.source_run, (int, np.integer))
            or isinstance(self.source_run, (bool, np.bool_))
        ):
            raise ValueError("deadtime calibration source run must be an integer")
        coefficients = tuple(
            _deadtime_number(v, "deadtime coefficient (µs)", detector=i)
            for i, v in enumerate(self.tau_us)
        )
        errors = tuple(
            None if v is None else _deadtime_number(v, "deadtime error (µs)")
            for v in self.errors_us
        )
        if any(v is not None and v < 0 for v in errors):
            raise ValueError("deadtime calibration errors must be non-negative")
        object.__setattr__(self, "tau_us", coefficients)
        object.__setattr__(self, "errors_us", errors)
        if self.source_run is not None:
            object.__setattr__(self, "source_run", int(self.source_run))

    @classmethod
    def from_increment(
        cls,
        base: CorrectionOutcome,
        increment_us: float,
        *,
        n_histograms: int,
        detector_indices: list[int],
        error_us: float | None,
        source_run: int,
    ):
        """Compose the fitted residual with the exact table applied to its data."""
        CorrectionOutcomes(deadtime=base).require_complete()
        values = (
            list(base.effective["tau_us"])
            if base.status == CorrectionStatus.APPLIED
            else [0.0] * n_histograms
        )
        if len(values) != n_histograms or not detector_indices:
            raise ValueError(
                "deadtime calibration requires a complete base table and fitted detectors"
            )
        errors = [None] * n_histograms
        delta = _deadtime_number(increment_us, "deadtime increment (µs)")
        if len(set(detector_indices)) != len(detector_indices):
            raise ValueError("fitted deadtime detector indices must be unique")
        for i in detector_indices:
            if not 0 <= i < n_histograms:
                raise ValueError("fitted deadtime detector is outside the run")
            # A nonzero base has no recorded covariance with the residual fit.
            errors[i] = error_us if values[i] == 0 else None
            values[i] += delta
        return cls(tuple(values), tuple(errors), "count_fit", source_run)


def fit_deadtime_calibration(
    histograms: list[Histogram],
    *,
    num_good_frames: float,
    t_good_offset: int = 0,
    last_good_bin: int | None = None,
    max_bins: int = 400,
    detector_t0_bins: list[int] | None = None,
    source_run: int | None = None,
    method: str = "calibrate",
    cancel_callback: Callable[[], bool] | None = None,
) -> DeadtimeCalibration | None:
    """Fit independent signed coefficients using RUMDA (RAL-94-080), §4's mean.

    M = N/(1 + N d/(Δt F)); d and Δt are in µs, F is dimensionless.
    This nonlinear weighted fit uses raw-count Poisson standard deviations;
    the published transformed regression supplies a signed starting point.
    """
    frames = _deadtime_number(num_good_frames, "good frames", positive=True)
    if not histograms:
        return None
    offset, window_limit = int(t_good_offset), int(max_bins)
    if offset < 0 or window_limit < 3:
        raise ValueError(
            "deadtime calibration requires a non-negative offset and at least three bins"
        )
    if detector_t0_bins is not None and len(detector_t0_bins) != len(histograms):
        raise ValueError("deadtime calibration requires one t0 per detector")
    coefficients, errors = [], []
    for i, histogram in enumerate(histograms):
        if cancel_callback is not None and cancel_callback():
            raise CancelledError
        width = _deadtime_number(histogram.bin_width, "bin width (µs)", positive=True)
        counts = np.asarray(histogram.counts, dtype=float)
        if not np.all(np.isfinite(counts)) or np.any(counts < 0):
            raise DeadtimeCorrectionError(
                "calibration counts must be finite and non-negative", detector=i
            )
        t0 = histogram.t0_bin if detector_t0_bins is None else detector_t0_bins[i]
        start = max(0, int(t0) + offset)
        end = counts.size if last_good_bin is None else min(counts.size, int(last_good_bin) + 1)
        observed = counts[start : min(end, start + window_limit)]
        if observed.size < 3 or np.count_nonzero(observed) < 3:
            return None
        times = np.arange(observed.size, dtype=float) * width
        fitted = _fit_deadtime_tau(
            times,
            observed,
            np.sqrt(np.maximum(observed, 1.0)),
            frames * width,
            cancel_callback=cancel_callback,
        )
        if fitted is None:
            return None
        coefficient, error = fitted
        coefficients.append(coefficient)
        errors.append(error)
    if method == "estimate":
        mean = float(np.mean(coefficients))
        error = float(np.sqrt(np.sum(np.square(errors))) / len(errors))
        coefficients = [mean] * len(coefficients)
        errors = [error] * len(errors)
    return DeadtimeCalibration(tuple(coefficients), tuple(errors), method, source_run)


def _fit_deadtime_tau(times_us, observed, sigma, frame_scale, *, cancel_callback=None):
    """Fit the rational mean without a sign bound; invalid trial means cost infinity."""
    decay = np.exp(-times_us / MUON_LIFETIME_US)
    slope, amplitude = np.polyfit(observed, observed / decay, 1)
    coefficient = -slope * frame_scale / amplitude if amplitude > 0 else 0.0
    amplitude = amplitude if amplitude > 0 else float(np.max(observed))

    def objective(amplitude, tau_us):
        if cancel_callback is not None and cancel_callback():
            raise CancelledError
        true = amplitude * decay
        denominator = 1 + true * tau_us / frame_scale
        if np.any(denominator <= 0) or not np.all(np.isfinite(denominator)):
            return np.inf
        return float(np.sum(((observed - true / denominator) / sigma) ** 2))

    fit = Minuit(objective, amplitude=amplitude, tau_us=coefficient)
    fit.errordef = Minuit.LEAST_SQUARES
    fit.limits["amplitude"] = (0, None)
    fit.tol = 1e-8
    fit.migrad()
    fit.hesse()
    if not fit.valid or not np.isfinite(fit.values["tau_us"]):
        return None
    return float(fit.values["tau_us"]), float(fit.errors["tau_us"])


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


def promote_deadtime_to_grouping(grouping: dict, calibration: DeadtimeCalibration) -> dict:
    """Write one validated total table atomically, enabling its manual correction."""
    old = grouping.get("dead_time_us", [])
    before = {i: old[i] if i < len(old) else 0.0 for i in range(len(calibration.tau_us))}
    grouping.update(
        {
            "dead_time_us": list(calibration.tau_us),
            "deadtime_correction": True,
            "deadtime_mode": "manual",
            "deadtime_method": calibration.method,
            "deadtime_errors_us": list(calibration.errors_us),
            "deadtime_reference_run": calibration.source_run,
        }
    )
    return {"before": before, "after": dict(enumerate(calibration.tau_us))}


def resolve_deadtime(
    grouping: dict, n_histograms: int, use: bool, mode: str | None = None
) -> CorrectionOutcome:
    """Resolve one request without mutating its table or coefficient provenance."""
    if not use:
        return CorrectionOutcome("deadtime", "off", CorrectionStatus.OFF)
    raw_mode = (
        str(
            mode
            if mode is not None
            else grouping.get("deadtime_mode", grouping.get("deadtime_method", "file"))
        )
        .strip()
        .lower()
    )
    modes = {
        "file": "from_file",
        "from_file": "from_file",
        "manual": "manual",
        "load": "manual",
        "value": "manual",
        "estimate": "estimate",
        "calibrate": "manual",
        "count_fit": "manual",
        "maxent_fit": "manual",
        "off": "off",
        "none": "off",
        "": "from_file",
    }
    if raw_mode not in modes:
        raise ValueError(f"unknown deadtime correction mode {raw_mode!r}")
    requested = modes[raw_mode]
    if requested == "off":
        return CorrectionOutcome("deadtime", "off", CorrectionStatus.OFF)
    table = grouping.get("dead_time_us", grouping.get("deadtime_loaded_us"))
    reason = None
    if not isinstance(table, (list, tuple)) or len(table) < n_histograms:
        reason = "a complete per-detector deadtime table is unavailable."
    elif requested == "from_file" and str(
        grouping.get("deadtime_method", "")
    ).strip().lower() not in {
        "",
        "file",
        "from_file",
    }:
        reason = "the deadtime table is not from the file."
    if reason is not None:
        return CorrectionOutcome("deadtime", requested, CorrectionStatus.UNAVAILABLE, reason)
    if any(value != 0 for value in table[:n_histograms]) and grouping.get("good_frames") is None:
        return CorrectionOutcome(
            "deadtime",
            requested,
            CorrectionStatus.UNAVAILABLE,
            "good frames must be finite and > 0; acquisition normalisation is unavailable",
        )
    correction = DeadtimeCorrection(tuple(table[:n_histograms]), grouping.get("good_frames"))
    if requested == "from_file" and not any(correction.tau_us):
        return CorrectionOutcome(
            "deadtime",
            requested,
            CorrectionStatus.UNAVAILABLE,
            "the file has no non-zero per-detector deadtime table.",
        )
    return CorrectionOutcome(
        "deadtime",
        requested,
        CorrectionStatus.APPLIED,
        effective={
            "tau_us": correction.tau_us,
            "good_frames": correction.good_frames if any(correction.tau_us) else None,
            "source": str(grouping.get("deadtime_method") or requested).strip().lower(),
        },
        uncertainty="Poisson; deadtime derivative and coefficient uncertainty not propagated",
    )


def prepare_histograms_with_deadtime(
    histograms: list[Histogram],
    grouping: dict,
    use_deadtime: bool,
    *,
    mode: str | None = None,
    allow_incomplete: bool = False,
) -> tuple[list[Histogram], CorrectionOutcome]:
    """Return prepared histograms and the exact requested/effective correction."""
    with perf_timer(
        "core.deadtime.prepare", n_detectors=len(histograms), use_deadtime=use_deadtime
    ) as perf:
        outcome = resolve_deadtime(grouping, len(histograms), use_deadtime, mode)
        if not allow_incomplete:
            CorrectionOutcomes(deadtime=outcome).require_complete()
        perf.detail(applied=outcome.status == CorrectionStatus.APPLIED)
        if outcome.status != CorrectionStatus.APPLIED or not any(outcome.effective["tau_us"]):
            return list(histograms), outcome
        corrected = [
            _copy_histogram_with_counts(
                hist,
                apply_deadtime_correction(
                    hist.counts,
                    tau_us,
                    hist.bin_width,
                    num_good_frames=outcome.effective["good_frames"],
                    detector=i,
                ),
            )
            for i, (hist, tau_us) in enumerate(
                zip(histograms, outcome.effective["tau_us"], strict=True)
            )
        ]
        return corrected, outcome


def _copy_histogram_with_counts(histogram: Histogram, counts) -> Histogram:
    return Histogram(
        counts=np.asarray(counts, dtype=np.float64),
        bin_width=histogram.bin_width,
        t0_bin=histogram.t0_bin,
        good_bin_start=histogram.good_bin_start,
        good_bin_end=histogram.good_bin_end,
        t0_time_us=histogram.t0_time_us,
    )
