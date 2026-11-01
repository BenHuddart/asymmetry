"""Background subtraction for grouped raw histograms.

Four modes are supported, selected by the ``background_mode`` grouping key
(with back-compatible derivation from the older keys when absent):

- ``"fixed"`` — subtract user-supplied per-group constants.
- ``"range"`` — musrfit's ``PRunAsymmetry`` convention: the mean count over
  a pre-t0 bin range (fallback range ``0.1·t0`` to ``0.6·t0``), only
  meaningful at continuous sources where a pre-t0 region exists.
- ``"tail_fit"`` — WiMDA's pulsed-source mode (``estBG``/``BGfit``):
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

from asymmetry.core.transform.fb_counts import GroupCounts, SharedComponent, validated_count_arrays
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

    forward: GroupCounts
    backward: GroupCounts
    applied: bool
    method: str
    values: tuple[float, float] | None = None
    ranges: tuple[tuple[int, int], tuple[int, int]] | None = None
    details: dict[str, Any] | None = None


@dataclass(frozen=True)
class TailFitResult:
    """Late-time exponential + flat fit to one grouped count spectrum.

    ``rate_per_us`` is the flat background rate in counts/µs;
    ``amplitude_per_us`` is the muon-decay rate extrapolated to t0.
    ``consistent_with_zero`` flags a background below two standard errors.
    ``count_covariance`` is cov(raw count, fitted rate) in counts²/µs on
    ``window``. ``expected_counts`` supplies the fitted Poisson variance basis.
    """

    rate_per_us: float
    rate_error_per_us: float | None
    amplitude_per_us: float
    window: tuple[int, int]
    n_bins: int
    ok: bool
    consistent_with_zero: bool
    message: str = ""

    count_covariance: NDArray[np.float64] | None = None
    expected_counts: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        if not self.ok:
            return
        if (
            self.rate_error_per_us is None
            or self.count_covariance is None
            or self.expected_counts is None
        ):
            raise ValueError(
                "a converged tail fit requires its Poisson information and influence row."
            )
        covariance, expected = validated_count_arrays(self.count_covariance, self.expected_counts)
        if (
            covariance.size != self.n_bins
            or np.any(expected <= 0)
            or not np.isfinite(self.rate_error_per_us)
            or self.rate_error_per_us <= 0
        ):
            raise ValueError("tail-fit uncertainty requires finite positive Poisson information.")
        for name, array in (("count_covariance", covariance), ("expected_counts", expected)):
            owned = np.array(array, copy=True)
            owned.setflags(write=False)
            object.__setattr__(self, name, owned)


def resolve_facility(
    *,
    metadata: dict[str, Any] | None = None,
    grouping: dict[str, Any] | None = None,
) -> str:
    """Resolve the facility label from run metadata, or ``""`` when unknown.

    The single place this resolution lives. The loaders record the facility in
    ``metadata["facility"]`` (``"PSI"`` for the PSI reader, the MusrRoot
    ``Laboratory`` header for ROOT files); ``metadata["instrument"]`` is the
    fallback, since the facility can be read off an instrument name where the
    label is missing, and ``metadata["psi_format"]`` is decisive on its own. A
    grouping payload's ``instrument`` — the canonical registry identity the
    loaders stamp there — is the last resort.

    The string matters because :func:`apply_grouped_background_correction` trims
    a pre-t0 background window to whole accelerator periods only for facilities
    it recognises (see :func:`_beam_period_us`). It used to be the caller's job
    to pass it, and a caller who did not know that silently lost the trimming.
    """
    metadata = metadata if isinstance(metadata, dict) else {}
    grouping = grouping if isinstance(grouping, dict) else {}
    facility = str(metadata.get("facility", "") or "").strip()
    if facility:
        return facility
    if metadata.get("psi_format"):
        return "PSI"
    instrument = str(metadata.get("instrument", "") or "").strip()
    if instrument:
        return instrument
    return str(grouping.get("instrument", "") or "").strip()


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

    The model is WiMDA's ``BGfit``, the muon exponential
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
    if np.linalg.matrix_rank(info) < 2:
        return _failure("Tail-fit information matrix is singular")
    try:
        covariance = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        return _failure("Tail-fit information matrix is singular")
    if not np.all(np.isfinite(covariance)) or covariance[1, 1] <= 0:
        return _failure("Tail-fit information matrix is not positive definite")
    rate_error = float(np.sqrt(covariance[1, 1]))
    # Cov(raw count j, fitted rate) = J_j · I⁻¹[:, rate] for Poisson ML.
    count_covariance = j1 * covariance[0, 1] + j2 * covariance[1, 1]
    consistent = p2 < 2.0 * rate_error
    return TailFitResult(
        rate_per_us=p2,
        rate_error_per_us=rate_error,
        amplitude_per_us=p1,
        window=(start, end),
        n_bins=end - start + 1,
        ok=True,
        consistent_with_zero=consistent,
        count_covariance=count_covariance,
        expected_counts=mu,
    )


def subtract_scaled_counts(
    counts: NDArray[np.float64],
    reference_counts: NDArray[np.float64],
    scale: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Subtract a scaled reference spectrum with error propagation.

    Returns ``counts − scale·reference`` and independent per-bin variance
    ``counts + scale²·reference`` (both inputs treated as Poisson; WiMDA
    subtracts the raw reference without touching the errors — study
    divergence D7). Arrays must have equal one-dimensional shape. This is the
    shared frame-scaled arithmetic seam the run-arithmetic project reuses.
    """
    a, r = validated_count_arrays(counts, reference_counts)
    s = float(scale)
    corrected = a - s * r
    variance = np.clip(a, 0.0, None) + s * s * np.clip(r, 0.0, None)
    return corrected, variance


def apply_grouped_background_correction(
    forward: NDArray[np.float64],
    backward: NDArray[np.float64],
    *,
    grouping: dict[str, Any] | None,
    t0_bin: int,
    bin_width_us: float,
    facility: str | None = None,
    metadata: dict[str, Any] | None = None,
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
        Facility label. PSI and TRIUMF ``range``-mode windows are shortened to
        an integer number of accelerator periods, following musrfit. The default
        ``None`` means *derive it* from ``metadata``/``grouping`` via
        :func:`resolve_facility` — the trimming used to be silently skipped for
        every caller who did not know to pass this. Pass ``""`` to opt out
        explicitly (no facility, no trimming); any non-empty string overrides.
    metadata
        Run metadata, used only to resolve ``facility`` when it is not given.
    last_good_bin
        End of the good window; bounds the tail-fit window.
    reference_forward, reference_backward, reference_scale
        Grouped counts of the resolved background reference run and the
        good-frame ratio, required by the ``reference_run`` mode (resolution
        from the ``background_run`` payload happens at the caller, which has
        loader access).
    """
    f, b = validated_count_arrays(forward, backward)
    grouping = grouping if isinstance(grouping, dict) else {}
    if facility is None:
        facility = resolve_facility(metadata=metadata, grouping=grouping)
    mode = resolve_background_mode(grouping)

    if mode == "none":
        return BackgroundCorrectionResult(GroupCounts(f), GroupCounts(b), False, "none")

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
                GroupCounts(f),
                GroupCounts(b),
                False,
                "tail_fit_failed",
                details={
                    "forward_message": forward_fit.message,
                    "backward_message": backward_fit.message,
                },
            )
        f_value = forward_fit.rate_per_us * bin_width_us
        b_value = backward_fit.rate_per_us * bin_width_us
        corrected = []
        for raw, fitted, level in ((f, forward_fit, f_value), (b, backward_fit, b_value)):
            first, last = fitted.window
            poisson_variance = raw.copy()
            poisson_variance[first : last + 1] = fitted.expected_counts
            covariance = np.zeros(raw.size)
            covariance[first : last + 1] = bin_width_us * fitted.count_covariance
            corrected.append(
                _estimated_background_group(
                    raw,
                    level,
                    (bin_width_us * fitted.rate_error_per_us) ** 2,
                    covariance,
                    poisson_variance,
                )
            )
        return BackgroundCorrectionResult(
            forward=corrected[0],
            backward=corrected[1],
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
            return BackgroundCorrectionResult(
                GroupCounts(f), GroupCounts(b), False, "missing_reference"
            )
        f_corr, f_var = subtract_scaled_counts(f, reference_forward, reference_scale)
        b_corr, b_var = subtract_scaled_counts(b, reference_backward, reference_scale)
        return BackgroundCorrectionResult(
            forward=GroupCounts(f_corr, f_var),
            backward=GroupCounts(b_corr, b_var),
            applied=True,
            method="reference_run",
            details={"scale": float(reference_scale)},
        )

    fixed = _fixed_background_values(grouping)
    if mode == "fixed":
        if fixed is None:
            # An explicit fixed mode without usable values must not silently
            # degrade to a range estimate the user never asked for.
            return BackgroundCorrectionResult(
                GroupCounts(f), GroupCounts(b), False, "missing_fixed_values"
            )
        return BackgroundCorrectionResult(
            forward=GroupCounts(f - fixed[0], f),
            backward=GroupCounts(b - fixed[1], b),
            applied=True,
            method="fixed",
            values=fixed,
            ranges=None,
        )

    ranges = _background_ranges(grouping, int(t0_bin))
    if ranges is None:
        return BackgroundCorrectionResult(GroupCounts(f), GroupCounts(b), False, "none")

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
            GroupCounts(f),
            GroupCounts(b),
            False,
            "invalid_range",
            ranges=(f_range, b_range),
        )

    f_slice = f[f_range[0] : f_range[1] + 1]
    b_slice = b[b_range[0] : b_range[1] + 1]
    f_value = float(np.mean(f_slice))
    b_value = float(np.mean(b_slice))
    corrected = []
    for raw, values, limits, level in (
        (f, f_slice, f_range, f_value),
        (b, b_slice, b_range, b_value),
    ):
        covariance = np.zeros(raw.size)
        covariance[limits[0] : limits[1] + 1] = values / values.size
        corrected.append(
            _estimated_background_group(
                raw,
                level,
                float(np.sum(values)) / values.size**2,
                covariance,
                raw,
            )
        )
    return BackgroundCorrectionResult(
        forward=corrected[0],
        backward=corrected[1],
        applied=True,
        method="estimated",
        values=(f_value, b_value),
        ranges=(f_range, b_range),
    )


def _estimated_background_group(
    raw: NDArray[np.float64],
    level: float,
    level_variance: float,
    count_covariance: NDArray[np.float64],
    independent_variance: NDArray[np.float64],
) -> GroupCounts:
    """Keep independent Poisson noise separate from one shared subtracted level."""
    return GroupCounts(
        raw - level,
        independent_variance,
        shared=(
            SharedComponent(
                "background",
                level_variance,
                -np.ones(raw.size),
                count_covariance,
            ),
        ),
    )


def _fixed_background_values(grouping: dict[str, Any]) -> tuple[float, float] | None:
    for key in ("background_fixed_values", "background_fix", "bkg_fix"):
        candidate = grouping.get(key)
        if isinstance(candidate, (list, tuple)) and len(candidate) >= 2:
            try:
                return float(candidate[0]), float(candidate[1])
            except (TypeError, ValueError):
                return None
    return None


def _background_ranges(
    grouping: dict[str, Any],
    t0_bin: int,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    pair = _parse_range_pair(grouping.get("background_ranges"))
    if pair is not None:
        return pair
    forward = _parse_range(grouping.get("background_forward_range"))
    backward = _parse_range(grouping.get("background_backward_range"))
    if forward is not None and backward is not None:
        return forward, backward
    shared = _parse_range(grouping.get("background_range"))
    if shared is None:
        shared = _ordered_range((int(0.1 * t0_bin), int(0.6 * t0_bin)))
    return shared, shared


def _parse_range_pair(value: Any) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if isinstance(value, dict):
        forward = _parse_range(value.get("forward"))
        backward = _parse_range(value.get("backward"))
    elif isinstance(value, (list, tuple)) and len(value) >= 2:
        forward = _parse_range(value[0])
        backward = _parse_range(value[1])
        if forward is None or backward is None:
            shared = _parse_range(value)
            return (shared, shared) if shared is not None else None
    else:
        return None
    return (forward, backward) if forward is not None and backward is not None else None


def _parse_range(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        endpoints = int(float(value[0])), int(float(value[1]))
    except (TypeError, ValueError):
        return None
    return _ordered_range(endpoints)


def _ordered_range(value: tuple[int, int]) -> tuple[int, int]:
    first, second = int(value[0]), int(value[1])
    return min(first, second), max(first, second)


def _adjust_range_for_beam_period(
    value: tuple[int, int],
    *,
    bin_width_us: float,
    facility: str,
) -> tuple[int, int]:
    period = _beam_period_us(facility)
    if bin_width_us <= 0.0 or period <= 0.0:
        return value
    periods = int((value[1] - value[0]) * bin_width_us / period)
    bins = int(periods * period / bin_width_us)
    if bins == 0:
        return value
    return _ordered_range((value[0], value[0] + bins))


def _beam_period_us(facility: str) -> float:
    label = facility.lower()
    return next((period for name, period in _BEAM_PERIOD_US.items() if name in label), 0.0)


def _range_is_valid(value: tuple[int, int], length: int) -> bool:
    return length > 0 and 0 <= value[0] <= value[1] < length
