"""Background subtraction for grouped raw histograms.

The grouped-histogram path follows musrfit's ``PRunAsymmetry`` convention:
background is subtracted from the grouped forward and backward histograms
before the asymmetry is calculated. A fixed forward/backward background may be
provided, otherwise the background is estimated as the mean count over a bin
range. When no range is provided, the musrfit fallback range
``0.1 * t0`` to ``0.6 * t0`` is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

_BEAM_PERIOD_US = {
    "psi": 0.01975,
    "triumf": 0.04337,
    "ral": 0.0,
    "isis": 0.0,
}


@dataclass(frozen=True)
class BackgroundCorrectionResult:
    """Result of grouped background subtraction."""

    forward: NDArray[np.float64]
    backward: NDArray[np.float64]
    forward_error: NDArray[np.float64] | None
    backward_error: NDArray[np.float64] | None
    applied: bool
    method: str
    values: tuple[float, float] | None = None
    ranges: tuple[tuple[int, int], tuple[int, int]] | None = None


def supports_background_correction(
    *,
    metadata: dict[str, Any] | None = None,
    source_file: str = "",
) -> bool:
    """Return whether musrfit-style grouped background subtraction applies."""
    metadata = metadata if isinstance(metadata, dict) else {}
    facility = str(metadata.get("facility", "")).lower()
    instrument = str(metadata.get("instrument", "")).lower()
    if "psi" in facility or instrument == "lem":
        return True
    if metadata.get("psi_format"):
        return True
    path = str(source_file or metadata.get("source_file", "") or "").lower()
    is_psi_root = "psi" in facility or instrument == "lem" or metadata.get("root_format")
    return path.endswith((".bin", ".mdu")) or (path.endswith(".root") and bool(is_psi_root))


def apply_grouped_background_correction(
    forward: NDArray[np.float64],
    backward: NDArray[np.float64],
    *,
    grouping: dict[str, Any] | None,
    t0_bin: int,
    bin_width_us: float,
    facility: str = "",
) -> BackgroundCorrectionResult:
    """Subtract fixed or estimated background from grouped counts.

    Parameters
    ----------
    forward, backward
        Grouped forward/backward count arrays.
    grouping
        Grouping payload. Recognized keys are ``background_fixed_values`` for
        fixed values and ``background_ranges``/``background_range`` for bin
        ranges. Ranges are inclusive, matching musrfit.
    t0_bin
        Common time-zero bin used to derive musrfit's fallback range.
    bin_width_us
        Histogram bin width in microseconds, used only for the optional
        facility beam-period adjustment.
    facility
        Optional facility label. PSI and TRIUMF ranges are shortened to an
        integer number of accelerator periods, following musrfit.
    """
    f = np.asarray(forward, dtype=np.float64)
    b = np.asarray(backward, dtype=np.float64)
    grouping = grouping if isinstance(grouping, dict) else {}

    fixed = _fixed_background_values(grouping)
    if fixed is not None:
        return BackgroundCorrectionResult(
            forward=f - fixed[0],
            backward=b - fixed[1],
            forward_error=_fixed_background_error(f),
            backward_error=_fixed_background_error(b),
            applied=True,
            method="fixed",
            values=fixed,
            ranges=None,
        )

    ranges = _background_ranges(grouping, int(t0_bin))
    if ranges is None:
        return BackgroundCorrectionResult(f, b, None, None, False, "none")

    f_range = _adjust_range_for_beam_period(
        ranges[0],
        bin_width_us=bin_width_us,
        facility=facility,
    )
    b_range = _adjust_range_for_beam_period(
        ranges[1],
        bin_width_us=bin_width_us,
        facility=facility,
    )
    if not _range_is_valid(f_range, len(f)) or not _range_is_valid(b_range, len(b)):
        return BackgroundCorrectionResult(
            f,
            b,
            None,
            None,
            False,
            "invalid_range",
            ranges=(f_range, b_range),
        )

    f_slice = f[f_range[0] : f_range[1] + 1]
    b_slice = b[b_range[0] : b_range[1] + 1]
    f_value = float(np.mean(f_slice))
    b_value = float(np.mean(b_slice))
    f_bkg_error = _estimated_background_error(f_slice)
    b_bkg_error = _estimated_background_error(b_slice)
    return BackgroundCorrectionResult(
        forward=f - f_value,
        backward=b - b_value,
        forward_error=_estimated_background_count_error(f, f_bkg_error),
        backward_error=_estimated_background_count_error(b, b_bkg_error),
        applied=True,
        method="estimated",
        values=(f_value, b_value),
        ranges=(f_range, b_range),
    )


def _fixed_background_values(grouping: dict[str, Any]) -> tuple[float, float] | None:
    for key in ("background_fixed_values", "background_fix", "bkg_fix"):
        value = grouping.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                return float(value[0]), float(value[1])
            except (TypeError, ValueError):
                return None
    return None


def _fixed_background_error(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return musrfit-style count errors for fixed-background subtraction."""
    arr = np.asarray(values, dtype=np.float64)
    return np.where(arr != 0.0, np.sqrt(np.maximum(arr, 0.0)), 1.0)


def _estimated_background_error(values: NDArray[np.float64]) -> float:
    """Return musrfit's error on an estimated constant background."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    total = float(np.sum(arr))
    if total <= 0.0:
        return 0.0
    return float(np.sqrt(total) / float(arr.size))


def _estimated_background_count_error(
    values: NDArray[np.float64],
    background_error: float,
) -> NDArray[np.float64]:
    """Return musrfit-style per-bin count errors after estimated subtraction."""
    arr = np.asarray(values, dtype=np.float64)
    return np.where(arr > 0.0, np.sqrt(np.maximum(arr + background_error**2, 0.0)), 1.0)


def _background_ranges(
    grouping: dict[str, Any],
    t0_bin: int,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    raw = grouping.get("background_ranges")
    parsed = _parse_range_pair(raw)
    if parsed is not None:
        return parsed

    forward = _parse_range(grouping.get("background_forward_range"))
    backward = _parse_range(grouping.get("background_backward_range"))
    if forward is not None and backward is not None:
        return forward, backward

    shared = _parse_range(grouping.get("background_range"))
    if shared is not None:
        return shared, shared

    start = int(float(t0_bin) * 0.1)
    end = int(float(t0_bin) * 0.6)
    return _ordered_range((start, end)), _ordered_range((start, end))


def _parse_range_pair(value: Any) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if isinstance(value, dict):
        forward = _parse_range(value.get("forward"))
        backward = _parse_range(value.get("backward"))
        if forward is not None and backward is not None:
            return forward, backward
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        first = _parse_range(value[0])
        second = _parse_range(value[1])
        if first is not None and second is not None:
            return first, second
        shared = _parse_range(value)
        if shared is not None:
            return shared, shared
    return None


def _parse_range(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return _ordered_range((int(float(value[0])), int(float(value[1]))))
    except (TypeError, ValueError):
        return None


def _ordered_range(value: tuple[int, int]) -> tuple[int, int]:
    start, end = int(value[0]), int(value[1])
    return (start, end) if start <= end else (end, start)


def _adjust_range_for_beam_period(
    value: tuple[int, int],
    *,
    bin_width_us: float,
    facility: str,
) -> tuple[int, int]:
    start, end = value
    if bin_width_us <= 0.0:
        return value
    period = _beam_period_us(facility)
    if period <= 0.0:
        return value
    interval_us = float(end - start) * float(bin_width_us)
    full_cycles = int(interval_us / period)
    adjusted_end = start + int((full_cycles * period) / float(bin_width_us))
    if adjusted_end == start:
        return value
    return _ordered_range((start, adjusted_end))


def _beam_period_us(facility: str) -> float:
    text = str(facility or "").lower()
    for key, period in _BEAM_PERIOD_US.items():
        if key in text:
            return period
    return 0.0


def _range_is_valid(value: tuple[int, int], length: int) -> bool:
    start, end = value
    return length > 0 and 0 <= start <= end < length
