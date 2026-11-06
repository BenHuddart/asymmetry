"""Background subtraction for grouped raw histograms.

Four modes are supported, selected by the ``background_mode`` grouping key
(with back-compatible derivation from the older keys when absent):

- ``"fixed"`` — subtract user-supplied per-group constants.
- ``"range"`` — a pre-t0 background convention: the mean count over
  a pre-t0 bin range (fallback range ``0.1·t0`` to ``0.6·t0``), only
  meaningful at continuous sources where a pre-t0 region exists.
- ``"tail_fit"`` — a pulsed-source background mode:
  a bin-integrated muon exponential plus a flat rate fitted to the late-time
  spectrum; the flat rate is subtracted. Asymmetry fits by Poisson maximum
  likelihood on native bins and reports the rate with an uncertainty.
- ``"reference_run"`` — WiMDA's File BG: subtract a designated reference
  run's grouped counts scaled by the good-frame ratio (sample holder /
  silver / laser-off references), with error propagation (divergence D7).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from asymmetry.core.transform.corrections import CorrectionOutcome, CorrectionStatus
from asymmetry.core.transform.fb_counts import (
    BinWindow,
    GroupCounts,
    SharedComponent,
    validated_count_arrays,
)
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
    outcome: CorrectionOutcome


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
    counts: NDArray[np.float64] | GroupCounts,
    *,
    bin_width_us: float,
    t0_bin: int,
    last_good_bin: int | None = None,
    fit_start_bin: int | None = None,
) -> TailFitResult:
    """Fit exponential + flat background to the late-time spectrum.

    The model is the muon exponential averaged across each bin plus a flat rate:

        µ_i = [p₁ · e^{−t_i/τ_µ} · sinh(w/2τ_µ)/(w/2τ_µ) + p₂] · w

    with w the bin width and t_i the bin centre relative to t0. Estimation is
    by nonnegative Poisson maximum likelihood, profiling the total expected
    count and solving the remaining monotone mixture score on [0, 1].
    The rate uncertainty comes from the expected information for raw counts.
    With GroupCounts, propagate supplied variance through the estimator's
    linearised influence, including its covariance with the input bins.

    The default window is the late half of (t0, last_good), on acquisition bins.
    """
    source = GroupCounts.from_value(counts)
    n, supplied_variance = source.counts, source.variance
    width = float(bin_width_us)
    start, end = tail_background_window(n.size, t0_bin, last_good_bin, fit_start_bin=fit_start_bin)

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

    if not np.isfinite(width) or width <= 0.0:
        return _failure("Bin width must be finite and positive")
    if end - start + 1 < 5:
        return _failure("Tail-fit window has fewer than 5 bins")
    window = np.clip(n[start : end + 1], 0.0, None)
    if float(np.sum(window)) <= 0.0:
        return _failure("Tail-fit window contains no counts")

    # Normalised component shapes remove the origin/amplitude scale. For
    # mu_i = T p_i(f), sum(p)=1 implies T_hat=sum(n); the mixture log
    # likelihood is concave, so its score root or an endpoint is the optimum.
    x = width / MUON_LIFETIME_US
    shape = np.exp(-np.arange(window.size, dtype=np.float64) * x)
    decay = shape / np.sum(shape)
    flat = np.full(window.size, 1.0 / window.size)
    delta = decay - flat
    total = float(np.sum(window))
    positive = window > 0

    def mixture_score(fraction: float) -> float:
        probability = (1.0 - fraction) * flat[positive] + fraction * decay[positive]
        return float(np.sum((window[positive] / total) * delta[positive] / probability))

    if mixture_score(0.0) <= 0.0:
        fraction = 0.0
    elif mixture_score(1.0) >= 0.0:
        fraction = 1.0
    else:
        fraction = brentq(mixture_score, 0.0, 1.0, xtol=1e-15)
    mu = total * ((1.0 - fraction) * flat + fraction * decay)
    if not np.all(np.isfinite(mu)) or np.any(mu <= 0.0):
        return _failure("Tail-fit expected counts must be finite and positive")
    p2 = total * (1.0 - fraction) / window.size / width
    # Restore the exponential rate at t0, including the bin-average factor.
    log_bin_factor = x / 2.0 + np.log(-np.expm1(-x) / x)
    p1 = 0.0
    if fraction > 0.0:
        log_amplitude = (
            np.log(total * fraction / np.sum(shape) / width)
            + (start - float(t0_bin)) * x
            - log_bin_factor
        )
        limits = np.finfo(float)
        if not np.log(limits.smallest_subnormal) <= log_amplitude <= np.log(limits.max):
            return _failure("Tail-fit amplitude at time zero is not representable")
        p1 = float(np.exp(log_amplitude))

    # Information in anchored counts/bin coordinates avoids origin and width
    # scaling in the rank check; convert the flat-count influence to counts/us.
    j1 = shape
    j2 = np.ones_like(mu)
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
    rate_error = float(np.sqrt(covariance[1, 1])) / width
    count_covariance = (j1 * covariance[0, 1] + j2 * covariance[1, 1]) / width
    if supplied_variance is not None:
        influence = count_covariance / mu
        source_variance = supplied_variance[start : end + 1]
        count_covariance = source_variance * influence
        rate_error = float(np.sqrt(np.sum(source_variance * influence**2)))
        if not np.isfinite(rate_error) or rate_error <= 0:
            return _failure("Tail-fit propagated variance must be finite and positive")
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
    counts: NDArray[np.float64] | GroupCounts,
    reference_counts: NDArray[np.float64] | GroupCounts,
    scale: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Subtract a scaled reference spectrum with error propagation.

    Returns ``counts − scale·reference`` with variance ``v + scale²·v_ref``.
    Array inputs use the Poisson approximation; GroupCounts retain supplied
    independent variance. Arrays must have equal one-dimensional shape.
    """
    sample, reference = (GroupCounts.from_value(value) for value in (counts, reference_counts))
    if sample.shared or reference.shared:
        raise ValueError("scaled count subtraction requires independent input counts")
    a, r = validated_count_arrays(sample.counts, reference.counts)
    s = float(scale)
    corrected = a - s * r
    variance = np.clip(sample.independent_variance, 0.0, None) + s * s * np.clip(
        reference.independent_variance, 0.0, None
    )
    return corrected, variance


def apply_grouped_background_correction(
    forward: NDArray[np.float64] | GroupCounts,
    backward: NDArray[np.float64] | GroupCounts,
    *,
    grouping: dict[str, Any] | None,
    t0_bin: int,
    acquisition_start_bin: int = 0,
    bin_width_us: float,
    facility: str | None = None,
    metadata: dict[str, Any] | None = None,
    last_good_bin: int | None = None,
    reference_forward: NDArray[np.float64] | GroupCounts | None = None,
    reference_backward: NDArray[np.float64] | GroupCounts | None = None,
    reference_scale: float | None = None,
) -> BackgroundCorrectionResult:
    """Subtract background from grouped counts by the active mode.

    Parameters
    ----------
    forward, backward
        Grouped forward/backward count arrays, or GroupCounts carrying supplied
        independent variance. Background estimation requires independent inputs.
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
    f_input, b_input = (GroupCounts.from_value(value) for value in (forward, backward))
    if f_input.shared or b_input.shared:
        raise ValueError("background estimation requires independent input counts")
    f, b = validated_count_arrays(f_input.counts, b_input.counts)
    vf, vb = f_input.independent_variance, b_input.independent_variance
    grouping = grouping if isinstance(grouping, dict) else {}
    if facility is None:
        facility = resolve_facility(metadata=metadata, grouping=grouping)
    mode = resolve_background_mode(grouping)

    if mode == "none":
        return _background_result(f_input, b_input, mode, status=CorrectionStatus.OFF)

    if mode == "tail_fit":
        forward_fit = fit_tail_background(
            f_input,
            bin_width_us=bin_width_us,
            t0_bin=t0_bin,
            last_good_bin=last_good_bin,
        )
        backward_fit = fit_tail_background(
            b_input,
            bin_width_us=bin_width_us,
            t0_bin=t0_bin,
            last_good_bin=last_good_bin,
        )
        if not forward_fit.ok or not backward_fit.ok:
            return _background_result(
                f_input,
                b_input,
                mode,
                status=CorrectionStatus.FAILED,
                reason=f"{forward_fit.message}; {backward_fit.message}",
            )
        f_value = forward_fit.rate_per_us * bin_width_us
        b_value = backward_fit.rate_per_us * bin_width_us
        corrected = []
        for source, fitted, level in (
            (f_input, forward_fit, f_value),
            (b_input, backward_fit, b_value),
        ):
            raw = source.counts
            first, last = fitted.window
            independent_variance = source.independent_variance.copy()
            if source.variance is None:
                independent_variance[first : last + 1] = fitted.expected_counts
            covariance = np.zeros(raw.size)
            covariance[first : last + 1] = bin_width_us * fitted.count_covariance
            corrected.append(
                _estimated_background_group(
                    raw,
                    level,
                    (bin_width_us * fitted.rate_error_per_us) ** 2,
                    covariance,
                    independent_variance,
                )
            )
        return _background_result(
            forward=corrected[0],
            backward=corrected[1],
            mode=mode,
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
            return _background_result(
                f_input,
                b_input,
                mode,
                status=CorrectionStatus.UNAVAILABLE,
                reason="reference run is not loaded",
            )
        f_corr, f_var = subtract_scaled_counts(f_input, reference_forward, reference_scale)
        b_corr, b_var = subtract_scaled_counts(b_input, reference_backward, reference_scale)
        return _background_result(
            forward=GroupCounts(f_corr, f_var),
            backward=GroupCounts(b_corr, b_var),
            mode=mode,
            details={"scale": float(reference_scale)},
        )

    if mode == "fixed":
        fixed = _fixed_background_values(grouping)
        if fixed is None:
            # An explicit fixed mode without usable values must not silently
            # degrade to a range estimate the user never asked for.
            return _background_result(
                f_input,
                b_input,
                mode,
                status=CorrectionStatus.UNAVAILABLE,
                reason="fixed background values are unavailable",
            )
        return _background_result(
            forward=GroupCounts(f - fixed[0], vf),
            backward=GroupCounts(b - fixed[1], vb),
            mode=mode,
            values=fixed,
            ranges=None,
        )

    f_range, b_range = grouped_background_ranges(
        grouping,
        t0_bin=t0_bin,
        acquisition_start_bin=acquisition_start_bin,
        length=f.size,
        bin_width_us=bin_width_us,
        facility=facility,
    )
    f_slice = f[f_range.slice]
    b_slice = b[b_range.slice]
    f_value = float(np.mean(f_slice))
    b_value = float(np.mean(b_slice))
    corrected = []
    for raw, variance, limits, level in (
        (f, vf, f_range, f_value),
        (b, vb, b_range, b_value),
    ):
        values = variance[limits.slice]
        covariance = np.zeros(raw.size)
        covariance[limits.slice] = values / values.size
        corrected.append(
            _estimated_background_group(
                raw,
                level,
                float(np.sum(values)) / values.size**2,
                covariance,
                variance,
            )
        )
    return _background_result(
        forward=corrected[0],
        backward=corrected[1],
        mode=mode,
        values=(f_value, b_value),
        ranges=((f_range.first, f_range.last), (b_range.first, b_range.last)),
    )


def tail_background_window(
    length: int, t0_bin: int, last_good_bin: int | None, *, fit_start_bin: int | None = None
) -> tuple[int, int]:
    """Inclusive late-half window shared by dependency selection and the fit."""
    end = min(length - 1, int(last_good_bin) if last_good_bin is not None else length - 1)
    start = (
        int(fit_start_bin) if fit_start_bin is not None else int(t0_bin) + (end - int(t0_bin)) // 2
    )
    return max(start, 0), end


def grouped_background_ranges(
    grouping: dict,
    *,
    t0_bin: int,
    acquisition_start_bin: int,
    length: int,
    bin_width_us: float,
    facility: str,
) -> tuple[BinWindow, BinWindow]:
    """Resolve observed local ranges, including established beam-period trimming."""
    ranges = resolve_background_ranges(grouping, int(t0_bin) + acquisition_start_bin)
    local = tuple(
        BinWindow(
            value.first - acquisition_start_bin,
            value.last - acquisition_start_bin,
            label="background range",
        )
        for value in ranges
    )
    for window in local:
        window.validate_length(length, label="background range")
    return tuple(
        _adjust_range_for_beam_period(window, bin_width_us=bin_width_us, facility=facility)
        for window in local
    )


def _background_result(
    forward,
    backward,
    mode,
    *,
    status=CorrectionStatus.APPLIED,
    reason=None,
    values=None,
    ranges=None,
    details=None,
):
    """Record the estimation inputs and uncertainty carried by the corrected counts."""
    effective = None
    uncertainty = None
    if status == CorrectionStatus.APPLIED:
        effective = {"method": "estimated" if mode == "range" else mode}
        if values is not None:
            effective["values"] = values
            effective["value_errors"] = [
                float(
                    np.sqrt(
                        sum(
                            component.variance
                            for component in group.shared
                            if component.name == "background"
                        )
                    )
                )
                for group in (forward, backward)
            ]
        if ranges is not None:
            effective["ranges"] = ranges
        if details is not None:
            effective["details"] = details
        uncertainty = {
            "fixed": "fixed background values treated as exact",
            "reference_run": "independent supplied sample and reference count variances",
            "range": "shared estimated levels with count-estimator covariance",
            "tail_fit": "shared fitted levels with count-estimator influence covariance",
        }[mode]
    return BackgroundCorrectionResult(
        forward,
        backward,
        CorrectionOutcome(
            "background",
            mode,
            status,
            reason,
            effective,
            uncertainty,
        ),
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
        if isinstance(candidate, (list, tuple)) and len(candidate) == 2:
            try:
                values = float(candidate[0]), float(candidate[1])
            except (TypeError, ValueError):
                raise ValueError("fixed background values must be two finite numbers") from None
            if not np.all(np.isfinite(values)):
                raise ValueError("fixed background values must be two finite numbers")
            return values
        if candidate is not None:
            raise ValueError("fixed background values must be two finite numbers")
    return None


def background_window(value: object) -> BinWindow:
    """Validate two inclusive acquisition-bin endpoints at the input boundary."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("background range requires two integer bin indices.")
    return BinWindow(*value, label="background range")


def resolve_background_ranges(grouping: dict[str, Any], t0_bin: int) -> tuple[BinWindow, BinWindow]:
    """Resolve both inclusive background windows on the declared acquisition grid."""
    if "background_ranges" in grouping:
        value = grouping["background_ranges"]
        if isinstance(value, dict):
            return background_window(value.get("forward")), background_window(value.get("backward"))
        if isinstance(value, (list, tuple)) and len(value) == 2:
            if all(isinstance(item, (list, tuple)) for item in value):
                return background_window(value[0]), background_window(value[1])
        shared = background_window(value)
    elif "background_forward_range" in grouping or "background_backward_range" in grouping:
        return (
            background_window(grouping.get("background_forward_range")),
            background_window(grouping.get("background_backward_range")),
        )
    elif "background_range" in grouping:
        shared = background_window(grouping["background_range"])
    else:
        shared = BinWindow(int(0.1 * t0_bin), int(0.6 * t0_bin))
    return shared, shared


def _adjust_range_for_beam_period(
    value: BinWindow, *, bin_width_us: float, facility: str
) -> BinWindow:
    period = _beam_period_us(facility)
    if bin_width_us <= 0.0 or period <= 0.0:
        return value
    periods = int((value.last - value.first) * bin_width_us / period)
    bins = int(periods * period / bin_width_us)
    return value if bins == 0 else BinWindow(value.first, value.first + bins)


def _beam_period_us(facility: str) -> float:
    label = facility.lower()
    return next((period for name, period in _BEAM_PERIOD_US.items() if name in label), 0.0)
