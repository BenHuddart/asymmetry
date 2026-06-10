"""Background subtraction for grouped raw histograms.

Four modes are supported, selected by the ``background_mode`` grouping key
(with back-compatible derivation from the older keys when absent):

- ``"fixed"`` — subtract user-supplied per-group constants.
- ``"range"`` — musrfit's ``PRunAsymmetry`` convention: the mean count over
  a pre-t0 bin range (fallback range ``0.1·t0`` to ``0.6·t0``), only
  meaningful at continuous sources where a pre-t0 region exists.
- ``"tail_fit"`` — WiMDA's pulsed-source mode (``Group.pas estBG/BGfit``):
  a bin-integrated muon exponential plus a flat rate fitted to the late-time
  spectrum; the flat rate is subtracted. Asymmetry fits by Poisson maximum
  likelihood instead of WiMDA's √N weights (study divergence D4) and reports
  the rate with an uncertainty.
- ``"reference_run"`` — WiMDA's File BG: subtract a designated reference
  run's grouped counts scaled by the good-frame ratio (sample holder /
  silver / laser-off references), with error propagation (divergence D7).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from asymmetry.core.utils.constants import MUON_LIFETIME_US

_BEAM_PERIOD_US = {
    "psi": 0.01975,
    "triumf": 0.04337,
    "ral": 0.0,
    "isis": 0.0,
}

BACKGROUND_MODES = ("none", "fixed", "range", "tail_fit", "reference_run")


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
    details: dict[str, Any] | None = None


@dataclass(frozen=True)
class TailFitResult:
    """Late-time exponential + flat fit to one grouped count spectrum.

    ``rate_per_us`` is the flat background rate (counts/µs for the group);
    ``amplitude_per_us`` the muon-decay rate extrapolated to t0.
    ``consistent_with_zero`` flags a background below two standard errors —
    the expected outcome on pulsed data, where the duty factor suppresses
    the uncorrelated rate (textbook §14.3); a significantly non-zero value
    is itself a diagnostic.
    """

    rate_per_us: float
    rate_error_per_us: float | None
    amplitude_per_us: float
    window: tuple[int, int]
    n_bins: int
    ok: bool
    consistent_with_zero: bool
    message: str = ""


def supports_background_correction(
    *,
    metadata: dict[str, Any] | None = None,
    source_file: str = "",
) -> bool:
    """Return whether pre-t0 range-average background subtraction applies.

    Kept for back-compatibility: this is the gate for the ``"range"`` mode
    only. Use :func:`available_background_modes` for the full mode list.
    """
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


def available_background_modes(
    *,
    metadata: dict[str, Any] | None = None,
    source_file: str = "",
) -> tuple[str, ...]:
    """Background modes applicable to a dataset.

    ``fixed``, ``tail_fit`` and ``reference_run`` apply everywhere; ``range``
    needs a pre-t0 region, which only continuous-source data provide (pulsed
    ISIS files start at the muon pulse).
    """
    modes = ["fixed", "tail_fit", "reference_run"]
    if supports_background_correction(metadata=metadata, source_file=source_file):
        modes.insert(0, "range")
    return tuple(modes)


def resolve_background_mode(grouping: dict[str, Any] | None) -> str:
    """Resolve the active background mode from a grouping payload.

    An explicit ``background_mode`` wins; otherwise the pre-existing keys
    decide (fixed values → ``fixed``; anything else → ``range``, which is
    the historical behaviour of the single enable flag).
    """
    grouping = grouping if isinstance(grouping, dict) else {}
    explicit = str(grouping.get("background_mode", "")).strip().lower()
    if explicit in BACKGROUND_MODES:
        return explicit
    if _fixed_background_values(grouping) is not None:
        return "fixed"
    return "range"


def fit_tail_background(
    counts: NDArray[np.float64],
    *,
    bin_width_us: float,
    t0_bin: int,
    last_good_bin: int | None = None,
    fit_start_bin: int | None = None,
) -> TailFitResult:
    """Fit exponential + flat background to the late-time spectrum.

    The model is WiMDA's ``BGfit`` (``Group.pas:1114``), the muon exponential
    averaged across each bin plus a flat rate:

        µ_i = [p₁ · e^{−t_i/τ_µ} · sinh(w/2τ_µ)/(w/2τ_µ) + p₂] · w

    with w the bin width and t_i the bin centre relative to t0. Estimation is
    by Poisson maximum likelihood — the intensity is linear in (p₁, p₂), so
    the negative log-likelihood is convex and the minimum unique. WiMDA
    instead weights by √N and deletes bins with ≤ 4 counts, which biases the
    rate low exactly where this fit operates (study divergence D4). The rate
    uncertainty comes from the expected information matrix at the optimum.

    The default window is the late half of (t0, last_good) — WiMDA's
    ``estBG`` convention expressed on raw bins rather than display bins.
    """
    n = np.asarray(counts, dtype=np.float64)
    width = float(bin_width_us)
    end = int(last_good_bin) if last_good_bin is not None else n.size - 1
    end = min(end, n.size - 1)
    start = (
        int(fit_start_bin) if fit_start_bin is not None else int(t0_bin) + (end - int(t0_bin)) // 2
    )
    start = max(start, 0)

    def _failure(message: str) -> TailFitResult:
        return TailFitResult(
            rate_per_us=0.0,
            rate_error_per_us=None,
            amplitude_per_us=0.0,
            window=(start, end),
            n_bins=max(0, end - start + 1),
            ok=False,
            consistent_with_zero=True,
            message=message,
        )

    if width <= 0.0:
        return _failure("Bin width must be positive")
    if end - start + 1 < 5:
        return _failure("Tail-fit window has fewer than 5 bins")
    window = np.clip(n[start : end + 1], 0.0, None)
    if float(np.sum(window)) <= 0.0:
        return _failure("Tail-fit window contains no counts")

    t = (np.arange(start, end + 1, dtype=np.float64) - float(t0_bin)) * width
    x = width / MUON_LIFETIME_US
    bin_factor = float(np.sinh(x / 2.0) / (x / 2.0)) if x > 0.0 else 1.0
    shape = np.exp(-t / MUON_LIFETIME_US) * bin_factor

    def negative_log_likelihood(params: np.ndarray) -> tuple[float, np.ndarray]:
        p1, p2 = params
        mu = np.maximum((p1 * shape + p2) * width, 1e-12)
        nll = float(np.sum(mu - window * np.log(mu)))
        residual = 1.0 - window / mu
        grad = np.array(
            [
                float(np.sum(residual * shape * width)),
                float(np.sum(residual * width)),
            ]
        )
        return nll, grad

    mean_rate = float(np.mean(window)) / width
    p1_init = max(mean_rate / max(float(np.mean(shape)), 1e-12), 1e-6)
    initial = np.array([p1_init, max(0.1 * mean_rate, 1e-9)])
    result = minimize(
        negative_log_likelihood,
        initial,
        jac=True,
        method="L-BFGS-B",
        bounds=[(0.0, None), (0.0, None)],
    )
    if not result.success:
        return _failure(f"Tail fit did not converge: {result.message}")
    p1, p2 = (float(v) for v in result.x)

    # Expected information for the linear-intensity Poisson model.
    mu = np.maximum((p1 * shape + p2) * width, 1e-12)
    j1 = shape * width
    j2 = np.full_like(mu, width)
    info = np.array(
        [
            [float(np.sum(j1 * j1 / mu)), float(np.sum(j1 * j2 / mu))],
            [float(np.sum(j1 * j2 / mu)), float(np.sum(j2 * j2 / mu))],
        ]
    )
    rate_error: float | None = None
    try:
        covariance = np.linalg.inv(info)
        if covariance[1, 1] > 0.0:
            rate_error = float(np.sqrt(covariance[1, 1]))
    except np.linalg.LinAlgError:
        rate_error = None

    consistent = rate_error is None or p2 < 2.0 * rate_error
    return TailFitResult(
        rate_per_us=p2,
        rate_error_per_us=rate_error,
        amplitude_per_us=p1,
        window=(start, end),
        n_bins=end - start + 1,
        ok=True,
        consistent_with_zero=consistent,
    )


def subtract_scaled_counts(
    counts: NDArray[np.float64],
    reference_counts: NDArray[np.float64],
    scale: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Subtract a scaled reference spectrum with error propagation.

    Returns ``counts − scale·reference`` and per-bin errors
    ``√(counts + scale²·reference)`` (both inputs treated as Poisson; WiMDA
    subtracts the raw reference without touching the errors — study
    divergence D7). Arrays are truncated to their common length. This is the
    shared frame-scaled arithmetic seam the run-arithmetic project reuses.
    """
    a = np.asarray(counts, dtype=np.float64)
    r = np.asarray(reference_counts, dtype=np.float64)
    m = min(a.size, r.size)
    a = a[:m]
    r = r[:m]
    s = float(scale)
    corrected = a - s * r
    variance = np.clip(a, 0.0, None) + s * s * np.clip(r, 0.0, None)
    errors = np.where(variance > 0.0, np.sqrt(variance), 1.0)
    return corrected, errors


def apply_grouped_background_correction(
    forward: NDArray[np.float64],
    backward: NDArray[np.float64],
    *,
    grouping: dict[str, Any] | None,
    t0_bin: int,
    bin_width_us: float,
    facility: str = "",
    last_good_bin: int | None = None,
    reference_forward: NDArray[np.float64] | None = None,
    reference_backward: NDArray[np.float64] | None = None,
    reference_scale: float | None = None,
) -> BackgroundCorrectionResult:
    """Subtract background from grouped counts by the active mode.

    Parameters
    ----------
    forward, backward
        Grouped forward/backward count arrays.
    grouping
        Grouping payload. ``background_mode`` selects the mode (see
        :func:`resolve_background_mode` for the back-compatible default);
        ``background_fixed_values`` supplies fixed values and
        ``background_ranges``/``background_range`` bin ranges (inclusive,
        matching musrfit).
    t0_bin
        Common time-zero bin (musrfit fallback range; tail-fit time axis).
    bin_width_us
        Histogram bin width in microseconds.
    facility
        Optional facility label. PSI and TRIUMF ranges are shortened to an
        integer number of accelerator periods, following musrfit.
    last_good_bin
        End of the good window; bounds the tail-fit window.
    reference_forward, reference_backward, reference_scale
        Grouped counts of the resolved background reference run and the
        good-frame ratio, required by the ``reference_run`` mode (resolution
        from the ``background_run`` payload happens at the caller, which has
        loader access).
    """
    f = np.asarray(forward, dtype=np.float64)
    b = np.asarray(backward, dtype=np.float64)
    grouping = grouping if isinstance(grouping, dict) else {}
    mode = resolve_background_mode(grouping)

    if mode == "none":
        return BackgroundCorrectionResult(f, b, None, None, False, "none")

    if mode == "tail_fit":
        forward_fit = fit_tail_background(
            f,
            bin_width_us=bin_width_us,
            t0_bin=t0_bin,
            last_good_bin=last_good_bin,
        )
        backward_fit = fit_tail_background(
            b,
            bin_width_us=bin_width_us,
            t0_bin=t0_bin,
            last_good_bin=last_good_bin,
        )
        if not forward_fit.ok or not backward_fit.ok:
            return BackgroundCorrectionResult(
                f,
                b,
                None,
                None,
                False,
                "tail_fit_failed",
                details={
                    "forward_message": forward_fit.message,
                    "backward_message": backward_fit.message,
                },
            )
        f_value = forward_fit.rate_per_us * bin_width_us
        b_value = backward_fit.rate_per_us * bin_width_us
        f_sigma = (forward_fit.rate_error_per_us or 0.0) * bin_width_us
        b_sigma = (backward_fit.rate_error_per_us or 0.0) * bin_width_us
        return BackgroundCorrectionResult(
            forward=f - f_value,
            backward=b - b_value,
            forward_error=_estimated_background_count_error(f, f_sigma),
            backward_error=_estimated_background_count_error(b, b_sigma),
            applied=True,
            method="tail_fit",
            values=(f_value, b_value),
            ranges=(forward_fit.window, backward_fit.window),
            details={
                "forward_rate_per_us": forward_fit.rate_per_us,
                "forward_rate_error_per_us": forward_fit.rate_error_per_us,
                "backward_rate_per_us": backward_fit.rate_per_us,
                "backward_rate_error_per_us": backward_fit.rate_error_per_us,
                "forward_consistent_with_zero": forward_fit.consistent_with_zero,
                "backward_consistent_with_zero": backward_fit.consistent_with_zero,
            },
        )

    if mode == "reference_run":
        if reference_forward is None or reference_backward is None or reference_scale is None:
            return BackgroundCorrectionResult(f, b, None, None, False, "missing_reference")
        f_corr, f_err = subtract_scaled_counts(f, reference_forward, reference_scale)
        b_corr, b_err = subtract_scaled_counts(b, reference_backward, reference_scale)
        return BackgroundCorrectionResult(
            forward=f_corr,
            backward=b_corr,
            forward_error=f_err,
            backward_error=b_err,
            applied=True,
            method="reference_run",
            details={"scale": float(reference_scale)},
        )

    fixed = _fixed_background_values(grouping)
    if mode == "fixed":
        if fixed is None:
            # An explicit fixed mode without usable values must not silently
            # degrade to a range estimate the user never asked for.
            return BackgroundCorrectionResult(f, b, None, None, False, "missing_fixed_values")
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
