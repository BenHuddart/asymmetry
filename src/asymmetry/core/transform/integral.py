r"""Time-integral asymmetry and field-scan assembly.

The *integral-counting* method reduces a whole muon time spectrum to a **single
number per run** — the asymmetry over a time window — and a *field scan*
collects that number across a series of runs ordered by a run variable (field,
temperature, or run number).  This is the model-free observable behind ALC
(avoided level crossing), LF repolarisation, and QLCR analysis.

This module is the porting target documented in
``docs/porting/time-integral-asymmetry/``.  The behavioural contract follows
Mantid's ``PlotAsymmetryByLogValue`` (alpha-aware, two reduction *methods*,
single time window, ordering by a run log) with WiMDA's count-integral as the
default ``"integral"`` formula (``alpha = 1.0`` reproduces WiMDA exactly).
musrfit has no native integral observable, so nothing is ported from it.

Two reduction methods are provided:

* ``"integral"`` (default) — **sum the counts** over ``[t_min, t_max]`` and then
  form ``(F_int - α B_int) / (F_int + α B_int)``.  This is the WiMDA ALC method
  and Mantid's ``Type=Integral``.
* ``"differential"`` — form the per-bin asymmetry first and then take its
  **mean** over the window.  This is Mantid's ``Type=Differential`` normalised by
  the window so the result is independent of window width and bin size (Mantid's
  single-period path returns an un-normalised time-integral; the normalised mean
  is the more useful observable and matches Mantid's dual-period normalisation).

The two methods agree only when the asymmetry is flat across the window.

Consistency with the time-domain asymmetry: per-run reduction shares the
grouping path with :class:`asymmetry.core.representation.time.TimeFBAsymmetry`
(via :func:`asymmetry.core.transform.group_forward_backward` and
:func:`~asymmetry.core.transform.effective_grouping`), and forms its counts
through the reduction's own correction stage
(:func:`~asymmetry.core.transform.reduce.corrected_grouped_counts`), so the two
agree on detector grouping, t0 alignment, deadtime, the balance ``alpha``, and
recipe ``grouping_ref`` overrides by construction — the grouping's constant
background level included, whose correlated error the integral propagates (see
:func:`integrate_asymmetry`).  The integral observable intentionally operates on **native
bins**: it ignores the time-domain display ``bunching_factor`` (which is a
plotting smoothing). The ``"integral"`` method is bunching-invariant anyway, and
integrating native bins is the more faithful observable.

Native period selection and correction preparation share
:func:`~asymmetry.core.transform.reduce.corrected_period_counts`. G+R uses one
pooled observation before the selected estimator; G−R differences independently
reduced set estimates. Red, Green, and extracted sets use their actual
acquired clocks and support.

Scale: every integral asymmetry this module produces is the dimensionless
**fraction** — the scale the field-scan / ALC parameter models
expect — *not* the percent of a loaded
:class:`~asymmetry.core.data.dataset.MuonDataset`. The exception is
:func:`integrate_curve`, which averages a caller-supplied curve and so returns
whatever scale it was given. See :mod:`asymmetry.core.transform.units` and
"Asymmetry units across the API" in the documentation.

This module must stay free of Qt / matplotlib / ``asymmetry.gui`` imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from asymmetry.core.data.dataset import MuonDataset, Run
from asymmetry.core.transform.asymmetry import (
    compute_asymmetry,
    compute_asymmetry_with_count_errors,
)
from asymmetry.core.transform.corrections import (
    BackgroundReference,
    CorrectionOutcomes,
    reference_observation_metadata,
)
from asymmetry.core.transform.fb_counts import (
    GroupCounts,
    linear_combination_variance,
    native_count_time,
    validated_count_arrays,
)
from asymmetry.core.transform.grouping import (
    effective_group_indices,
    effective_grouping,
    grouping_balances,
)
from asymmetry.core.transform.rebin import BinningSpec
from asymmetry.core.transform.reduce import (
    corrected_period_counts,
    correction_flags_from_grouping,
)
from asymmetry.core.transform.units import ASYMMETRY_FRACTION, AsymmetryUnit
from asymmetry.core.utils.constants import ORDER_KEYS, PeriodMode

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

    from numpy.typing import NDArray

__all__ = [
    "METHODS",
    "ORDER_KEYS",
    "integrate_asymmetry",
    "integrate_curve",
    "integrate_run",
    "build_field_scan",
    "differentiate_scan",
    "FieldScan",
    "FieldScanPoint",
]

#: Supported per-run reduction methods.
METHODS = ("integral", "differential")

# ORDER_KEYS ("field", "temperature", "run") is imported from
# asymmetry.core.utils.constants — the same tuple FitSeries uses — so the field
# scan and a fit series cannot diverge on which orderings are allowed.

#: Human-facing axis labels for each ordering variable.
_X_LABELS = {"field": "B (G)", "temperature": "T (K)", "run": "Run"}


# --- per-run reduction --------------------------------------------------------


def integrate_asymmetry(
    forward: GroupCounts,
    backward: GroupCounts,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    time: NDArray[np.float64] | None = None,
    t_min: float | None = None,
    t_max: float | None = None,
    method: str = "integral",
) -> tuple[float, float]:
    r"""Reduce grouped counts to one fractional asymmetry and marginal error.

    ``integral`` sums native counts before forming the ratio. ``differential``
    averages per-bin asymmetries over the selected non-zero denominators.
    Both propagate the independent count variances and shared components with
    :func:`~asymmetry.core.transform.fb_counts.linear_combination_variance`.
    Its weights are 1 for an integral and the asymmetry derivatives divided by
    the number of averaged bins for a differential mean.

    ``time`` is an optional aligned axis in μs, required with inclusive bounds
    ``t_min`` or ``t_max``. Without bounds the full record is selected. α and β
    must be positive. Degenerate first-order asymmetry errors use the unit
    no-information sentinel at the observable level.
    """
    _validate_method(method)
    _validate_alpha(alpha)
    _validate_beta(beta)
    f, b = forward.counts, backward.counts
    if f.shape != b.shape:
        raise ValueError("count arrays must have equal shape.")
    n = f.size
    if n == 0:
        raise ValueError("integrate_asymmetry requires non-empty count arrays.")
    if time is not None:
        _, _, time = validated_count_arrays(f, b, time)
    mask = _window_mask(time, t_min, t_max, n)
    if not np.any(mask):
        raise ValueError("Integration window selects no bins.")
    edges = np.array([0, n], dtype=np.intp)
    if method == "integral":
        variances = [
            linear_combination_variance(group, edges, mask)[0] for group in (forward, backward)
        ]
        asym, error = compute_asymmetry_with_count_errors(
            np.array([np.sum(f[mask])]),
            np.array([np.sum(b[mask])]),
            np.array([np.sqrt(variances[0])]),
            np.array([np.sqrt(variances[1])]),
            alpha,
            beta,
        )
        return float(asym[0]), float(error[0])

    denominator = beta * f + alpha * b
    valid = mask & (denominator != 0.0)
    if not np.any(valid):
        raise ValueError("Integration window selects no bins with non-zero counts.")
    asym, _ = compute_asymmetry(f, b, alpha, beta)
    scale = alpha * (1.0 + beta) / denominator[valid] ** 2 / np.count_nonzero(valid)
    f_weights, b_weights = np.zeros(n), np.zeros(n)
    f_weights[valid] = scale * b[valid]
    b_weights[valid] = -scale * f[valid]
    variance = (
        linear_combination_variance(forward, edges, f_weights)[0]
        + linear_combination_variance(backward, edges, b_weights)[0]
    )
    error = float(np.sqrt(variance)) if variance > 0 else 1.0
    return float(np.mean(asym[valid])), error


def integrate_curve(
    time: NDArray[np.float64],
    asymmetry: NDArray[np.float64],
    error: NDArray[np.float64],
    *,
    t_min: float | None = None,
    t_max: float | None = None,
) -> tuple[float, float]:
    """Mean of an already-formed asymmetry curve over ``[t_min, t_max]``.

    Useful when only the reduced curve is available (for example a combined
    green∓red spectrum) rather than the forward/backward counts.  The error is
    the error on the mean, ``sqrt(Σ eᵢ²) / N``.

    Parameters
    ----------
    time
        Time axis in µs.
    asymmetry, error
        The reduced curve, on **either** asymmetry scale — fraction or percent.
        This is the one function in the module that does not fix the scale.
    t_min, t_max
        Inclusive averaging window in µs.

    Returns
    -------
    (value, error)
        **On the same scale as the input** ``asymmetry`` — a mean is
        unit-preserving. Pass ``ds.asymmetry`` and the result is percent; pass
        ``ds.asymmetry_fraction`` and it is a fraction, matching
        :func:`integrate_asymmetry` and :func:`integrate_run`.
    """
    t, a, e = validated_count_arrays(time, asymmetry, error)
    n = t.size
    if n == 0:
        raise ValueError("integrate_curve requires non-empty arrays.")
    mask = _window_mask(t[:n], t_min, t_max, n)
    if not np.any(mask):
        raise ValueError("Integration window selects no bins.")
    return _mean_over_window(a[:n], e[:n], mask)


@dataclass(frozen=True)
class IntegralReduction:
    """Integral value and error with the correction decisions that produced them."""

    value: float
    error: float
    corrections: CorrectionOutcomes


def integrate_run(
    data: MuonDataset | Run,
    *,
    t_min: float | None = None,
    t_max: float | None = None,
    method: str = "integral",
    alpha: float | None = None,
    grouping_ref: dict | None = None,
    reference: BackgroundReference | None = None,
) -> tuple[float, float]:
    """Return fractional integral asymmetry and error, warning for accepted skips."""
    result = reduce_period_integral(
        data,
        t_min=t_min,
        t_max=t_max,
        method=method,
        alpha=alpha,
        grouping_ref=grouping_ref,
        reference=reference,
    )
    return result.value, result.error


def reduce_integral_run(
    data: MuonDataset | Run,
    *,
    t_min: float | None = None,
    t_max: float | None = None,
    method: str = "integral",
    alpha: float | None = None,
    grouping_ref: dict | None = None,
    reference: BackgroundReference | None = None,
) -> IntegralReduction:
    """Reduce a loaded run to a single integral-asymmetry ``(value, error)``.

    The run's forward/backward groups are formed through the same shared path as
    :class:`asymmetry.core.representation.time.TimeFBAsymmetry`
    (:func:`asymmetry.core.transform.effective_grouping` +
    :func:`~asymmetry.core.transform.group_forward_backward`), so the integral
    observable agrees with the time-domain asymmetry on grouping and ``alpha``.
    When the window is unspecified it defaults to the run's good-bin range
    (matching the WiMDA ALC / Mantid full-range behaviour).

    Parameters
    ----------
    data
        A :class:`MuonDataset` (its ``.run`` is used) or a :class:`Run`.  For
        multi-period runs, select the period first with
        :func:`asymmetry.core.io.periods.select_period`.
    t_min, t_max
        Inclusive window in µs.  Default: the run's good-bin range.
    method
        See :func:`integrate_asymmetry`.
    alpha
        Balance parameter; when ``None`` it is resolved for the selected pair
        in the effective grouping (default 1.0), exactly as the
        time-domain reduction does. An explicit value is validated (> 0, finite).
    grouping_ref
        Optional grouping override (forward/backward group ids, alpha, good-bin
        window) merged over ``run.grouping`` — the same ``grouping_ref`` a
        :class:`TimeFBAsymmetry` recipe carries, so a GUI can pass the user's
        effective grouping and the scan matches the displayed asymmetry.
    reference
        Explicitly resolved reference acquisition for a configured reference-run
        background. No reference files are opened by this transform.

    Returns
    -------
    (value, error)
        **As a fraction**, from :func:`integrate_asymmetry` —
        *not* the percent scale of ``data.asymmetry`` when ``data`` is a
        :class:`MuonDataset`. This function reduces the run's **counts**, so the
        dataset's own percent-scale curve is never involved.
    """
    run = _resolve_run(data)
    grouping = effective_grouping(run, grouping_ref)
    mode = PeriodMode(grouping.get("period_mode", "red"))
    if mode in {PeriodMode.GREEN_PLUS_RED, PeriodMode.GREEN_MINUS_RED}:
        raise ValueError("Use reduce_period_integral for combined period correction outcomes.")
    result = reduce_period_integral(
        data,
        t_min=t_min,
        t_max=t_max,
        method=method,
        alpha=alpha,
        grouping_ref=grouping_ref,
        reference=reference,
    )
    return IntegralReduction(result.value, result.error, result.corrections[str(mode)])


@dataclass(frozen=True)
class PeriodIntegralReduction:
    """A period integral combination and every contributing correction decision."""

    value: float
    error: float
    corrections: dict[str, CorrectionOutcomes]


def reduce_period_integral(
    data,
    *,
    t_min=None,
    t_max=None,
    method="integral",
    alpha=None,
    grouping_ref=None,
    reference=None,
):
    """Coordinate background before integrating held-period sums or differences."""
    run = _resolve_run(data)
    grouping = effective_grouping(run, grouping_ref)
    if alpha is not None:
        grouping.pop("alpha_policy", None)
    mode = PeriodMode(grouping.get("period_mode", "red"))
    _validate_method(method)
    alpha_used, beta_used = grouping_balances(grouping)
    flags = correction_flags_from_grouping(grouping)
    from asymmetry.core.io import background_reference_for_grouping

    counts = corrected_period_counts(
        run.histograms,
        grouping,
        str(mode),
        binning=BinningSpec(),
        forward_idx=effective_group_indices(
            grouping, int(grouping.get("forward_group", 1)), n_histograms=len(run.histograms)
        ),
        backward_idx=effective_group_indices(
            grouping, int(grouping.get("backward_group", 2)), n_histograms=len(run.histograms)
        ),
        use_deadtime=flags.use_deadtime,
        deadtime_mode=flags.deadtime_mode,
        use_background=flags.use_background,
        metadata=reference_observation_metadata(run),
        native_time_range=(t_min, t_max),
        reference_resolver=partial(background_reference_for_grouping, reference),
    )
    from asymmetry.core.transform.grouping import shared_automatic_alpha
    from asymmetry.core.transform.reduce import estimate_prepared_alpha

    if alpha is None and shared_automatic_alpha(grouping, reference_observation_metadata(run)):
        target = str(PeriodMode.RED) if mode == PeriodMode.GREEN_MINUS_RED else str(mode)
        alpha_used = estimate_prepared_alpha(counts[target])
    values = {}
    for label, pair in counts.items():
        time = native_count_time(
            pair.forward.counts.size, pair.bin_width_us, pair.common_t0_bin, pair.t0_time_us
        )
        values[label] = integrate_asymmetry(
            pair.forward,
            pair.backward,
            alpha=alpha_used if alpha is None else alpha,
            beta=beta_used,
            time=time,
            t_min=time[pair.window.first] if t_min is None else t_min,
            t_max=time[pair.window.last] if t_max is None else t_max,
            method=method,
        )
    if mode != PeriodMode.GREEN_MINUS_RED:
        value, error = values[str(mode)]
        return PeriodIntegralReduction(value, error, {k: v.corrections for k, v in counts.items()})
    red, green = values["red"], values["green"]
    return PeriodIntegralReduction(
        green[0] - red[0],
        float(np.hypot(red[1], green[1])),
        {label: pair.corrections for label, pair in counts.items()},
    )


# --- field-scan assembly ------------------------------------------------------


@dataclass
class FieldScanPoint:
    """One run's contribution to a field scan.

    ``value``/``error`` are the integral asymmetry **as a fraction**
    carried straight from the owning
    :class:`FieldScan` — see its ``units`` field.
    """

    run_number: int
    x: float
    value: float
    error: float
    label: str = ""
    corrections: dict[str, CorrectionOutcomes] = field(default_factory=dict)


@dataclass
class FieldScan:
    """An ordered series of integral-asymmetry values vs a run variable.

    The arrays are parallel and sorted by ``x`` ascending.  ``excluded`` lists
    ``(run_number, reason)`` for runs that could not contribute (for example a
    missing field log — Mantid skips such runs rather than failing the scan).

    Scale note: ``value``/``error`` are the integral asymmetry **as a fraction**
    (from :func:`compute_asymmetry` on summed counts) — the
    scale the field-scan / ALC parameter models expect (e.g. ``GaussianLCR``
    seeds ``f`` near 0.1), and the one the ``units`` field records. This differs
    from a loaded :class:`~asymmetry.core.data.dataset.MuonDataset`, whose
    ``asymmetry`` is on the **percent** scale; multiply by 100 (or call
    :func:`asymmetry.core.transform.units.to_percent`) if you need to compare the
    two directly.
    """

    x: NDArray[np.float64]
    value: NDArray[np.float64]
    error: NDArray[np.float64]
    run_numbers: list[int]
    order_key: str
    method: str
    derivative: bool = False
    x_label: str = ""
    y_label: str = "Integral asymmetry"
    excluded: list[tuple[int, str]] = field(default_factory=list)
    #: Scale of :attr:`value` and :attr:`error`. Always
    #: :data:`~asymmetry.core.transform.units.ASYMMETRY_FRACTION` — the
    #: counterpart of
    #: :attr:`~asymmetry.core.transform.reduce.GroupedAsymmetryReduction.units`
    #: on the other side of the divide. For a ``derivative`` scan the values are
    #: fraction *per x-unit* (:func:`differentiate_scan` divides by Δx, which
    #: leaves the asymmetry scale untouched).
    units: AsymmetryUnit = ASYMMETRY_FRACTION
    correction_outcomes: dict[int, dict[str, CorrectionOutcomes]] = field(default_factory=dict)

    @property
    def n_points(self) -> int:
        return int(self.x.size)

    @property
    def points(self) -> list[FieldScanPoint]:
        return [
            FieldScanPoint(
                run_number=self.run_numbers[i],
                x=float(self.x[i]),
                value=float(self.value[i]),
                error=float(self.error[i]),
                corrections=self.correction_outcomes.get(self.run_numbers[i], {}),
            )
            for i in range(self.n_points)
        ]

    def __repr__(self) -> str:
        kind = "derivative" if self.derivative else self.method
        return (
            f"FieldScan({kind}, n={self.n_points}, order_key={self.order_key!r}, "
            f"excluded={len(self.excluded)})"
        )


def build_field_scan(
    runs,
    *,
    t_min: float | None = None,
    t_max: float | None = None,
    method: str = "integral",
    alpha: float | None = None,
    order_key: str = "field",
    grouping_ref: dict | None = None,
    reference: BackgroundReference | None = None,
    filter: Callable[[Run], bool] | None = None,
) -> FieldScan:
    """Assemble a field scan from a series of loaded runs.

    Parameters
    ----------
    runs
        Iterable of loaded objects — each a :class:`MuonDataset` or :class:`Run`.
        For multi-period runs, select the period upstream and pass the per-period
        datasets in.
    t_min, t_max, method, alpha, grouping_ref, reference
        Passed through to :func:`integrate_run` (applied to every run in the
        scan — the grouping/alpha is normally common across a scan).
    order_key
        ``"field"``, ``"temperature"``, or ``"run"`` — the x-axis variable the
        points are ordered by (the same :data:`ORDER_KEYS` ``FitSeries`` uses).
    filter
        Optional predicate ``run -> bool`` used to keep a *subset* of the runs in
        a single scan, so distinct measurement periods or run types taken at the
        same field are not mixed into one curve. It is called with each resolved
        :class:`Run` (giving its ``run_number`` and ``metadata``) **before**
        reduction; runs for which it returns a falsy value are dropped and listed
        in ``excluded`` with the reason ``"excluded by filter"``. Use it to keep
        one period (``run.metadata.get("period_label") == "red"``), one run type,
        or to drop interleaved calibration runs (``run.run_number not in cal``).
        A predicate that raises excludes that one run (reason ``"filter raised:
        …"``) rather than aborting the scan. Default ``None`` keeps every run —
        identical to the historical behaviour. (To split a multi-period *file*
        into its periods first, use
        :func:`asymmetry.core.io.periods.select_period` upstream — period
        extraction lives in ``io``, which depends on this transform, not the
        reverse.)

    Returns
    -------
    FieldScan
        Whose ``value``/``error`` are **as a fraction** (with
        ``units`` set to
        :data:`~asymmetry.core.transform.units.ASYMMETRY_FRACTION`) — *not*
        percent, since every point comes from :func:`integrate_run`. Sorted
        parallel arrays plus the list of excluded runs.
    """
    _validate_method(method)
    if order_key not in ORDER_KEYS:
        raise ValueError(f"order_key must be one of {ORDER_KEYS}, got {order_key!r}")
    if filter is not None and not callable(filter):
        raise TypeError(f"filter must be callable or None, got {type(filter).__name__}")

    points: list[FieldScanPoint] = []
    excluded: list[tuple[int, str]] = []

    for item in runs:
        try:
            run = _resolve_run(item)
        except (TypeError, ValueError) as exc:
            # One un-resolvable item must not abort the whole scan; exclude it
            # with a reason, like a run missing its log value.
            excluded.append((_excluded_run_number(item), str(exc)))
            continue
        run_number = int(run.run_number)
        if filter is not None:
            try:
                keep = bool(filter(run))
            except Exception as exc:  # noqa: BLE001 - one bad predicate must not abort the scan
                excluded.append((run_number, f"filter raised: {exc}"))
                continue
            if not keep:
                excluded.append((run_number, "excluded by filter"))
                continue
        x_value = _order_value(run, order_key)
        if x_value is None:
            excluded.append((run_number, f"no {order_key} value"))
            continue
        try:
            reduced = reduce_period_integral(
                run,
                t_min=t_min,
                t_max=t_max,
                method=method,
                alpha=alpha,
                grouping_ref=grouping_ref,
                reference=reference,
            )
        except ValueError as exc:
            excluded.append((run_number, str(exc)))
            continue
        points.append(
            FieldScanPoint(
                run_number, x_value, reduced.value, reduced.error, corrections=reduced.corrections
            )
        )

    points.sort(key=lambda p: (p.x, p.run_number))
    return FieldScan(
        x=np.array([p.x for p in points], dtype=np.float64),
        value=np.array([p.value for p in points], dtype=np.float64),
        error=np.array([p.error for p in points], dtype=np.float64),
        run_numbers=[p.run_number for p in points],
        order_key=order_key,
        method=method,
        x_label=_X_LABELS[order_key],
        excluded=excluded,
        units=ASYMMETRY_FRACTION,
        correction_outcomes={p.run_number: p.corrections for p in points},
    )


def differentiate_scan(scan: FieldScan, *, max_gap: float | None = None) -> FieldScan:
    """Forward-difference derivative ``dA/dx`` of a scan (WiMDA ``dA/dB``).

    Each output point sits at the midpoint of an adjacent pair, with
    ``dA/dx = (A₂ - A₁) / (x₂ - x₁)`` and error ``sqrt(e₁² + e₂²) / |Δx|``.
    Pairs with non-positive spacing, or spacing greater than ``max_gap`` (when
    given), are skipped — mirroring WiMDA's adjacent-field threshold.

    Returns
    -------
    FieldScan
        With ``derivative=True``. Dividing by Δx leaves the asymmetry scale
        untouched, so the values stay **on the input scan's scale** — a fraction
        per x-unit, and ``units`` is carried through unchanged.
    """
    if scan.derivative:
        raise ValueError("differentiate_scan expects an integral scan, not a derivative.")

    xs, ys, es = scan.x, scan.value, scan.error
    mid_x: list[float] = []
    deriv: list[float] = []
    deriv_err: list[float] = []
    run_pairs: list[int] = []

    for i in range(scan.n_points - 1):
        dx = float(xs[i + 1] - xs[i])
        if dx <= 0.0 or (max_gap is not None and dx > max_gap):
            continue
        mid_x.append(0.5 * (float(xs[i]) + float(xs[i + 1])))
        deriv.append((float(ys[i + 1]) - float(ys[i])) / dx)
        deriv_err.append(float(np.hypot(es[i], es[i + 1])) / dx)
        run_pairs.append(scan.run_numbers[i + 1])

    return FieldScan(
        x=np.array(mid_x, dtype=np.float64),
        value=np.array(deriv, dtype=np.float64),
        error=np.array(deriv_err, dtype=np.float64),
        run_numbers=run_pairs,
        order_key=scan.order_key,
        method=scan.method,
        derivative=True,
        x_label=scan.x_label,
        y_label=f"d({scan.y_label})/d{scan.order_key}",
        excluded=list(scan.excluded),
        correction_outcomes=dict(scan.correction_outcomes),
        units=scan.units,
    )


# --- internal -----------------------------------------------------------------


def _validate_method(method: str) -> None:
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")


def _validate_alpha(alpha: float) -> None:
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError(f"alpha must be a positive, finite number, got {alpha!r}")


def _validate_beta(beta: float) -> None:
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError(f"beta must be a positive, finite number, got {beta!r}")


def _validate_window(t_min: float, t_max: float) -> None:
    # Equal bounds are allowed: they select a single bin (for example a run
    # whose good-bin range is one bin). Only an inverted window is rejected.
    if t_min > t_max:
        raise ValueError(f"t_min must be <= t_max, got t_min={t_min}, t_max={t_max}")


def _window_mask(
    time: NDArray[np.float64] | None,
    t_min: float | None,
    t_max: float | None,
    n: int,
) -> NDArray[np.bool_]:
    if t_min is None and t_max is None:
        return np.ones(n, dtype=bool)
    if time is None:
        raise ValueError("A time axis is required when t_min/t_max are supplied.")
    if t_min is not None and t_max is not None:
        _validate_window(float(t_min), float(t_max))
    t = np.asarray(time, dtype=np.float64)[:n]
    mask = np.ones(n, dtype=bool)
    if t_min is not None:
        mask &= t >= t_min
    if t_max is not None:
        mask &= t <= t_max
    return mask


def _mean_over_window(
    asym: NDArray[np.float64],
    err: NDArray[np.float64],
    mask: NDArray[np.bool_],
) -> tuple[float, float]:
    selected = asym[mask]
    selected_err = err[mask]
    count = selected.size
    value = float(np.mean(selected))
    error = float(np.sqrt(np.sum(np.square(selected_err))) / count)
    return value, error


def _resolve_run(data: MuonDataset | Run) -> Run:
    if isinstance(data, Run):
        return data
    if isinstance(data, MuonDataset):
        if data.run is None:
            raise ValueError(
                "This MuonDataset has no source run; integral reduction needs the "
                "run's histograms and grouping."
            )
        return data.run
    if isinstance(data, list):
        raise TypeError(
            "Got a list of per-period datasets; select one period first with "
            "asymmetry.core.io.periods.select_period(...)."
        )
    raise TypeError(f"Expected a MuonDataset or Run, got {type(data).__name__}")


def _order_value(run: Run, order_key: str) -> float | None:
    if order_key == "run":
        return float(run.run_number)
    # Read metadata directly rather than via Run.field / Run.temperature: those
    # properties default a missing value to 0.0, which we must NOT conflate with
    # a genuine 0 G / 0 K scan point (a TF run at 0 G is a valid point — see the
    # field-geometry study). Absent metadata -> None -> the run is excluded.
    raw = run.metadata.get(order_key)
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    # Reject non-finite logs (NaN/inf) so they cannot corrupt the scan order or
    # produce NaN derivative points downstream.
    if not np.isfinite(value):
        return None
    return value


def _excluded_run_number(item: object) -> int:
    """Best-effort run number for an item that could not be resolved to a Run.

    ``MuonDataset.run_number`` and ``Run.run_number`` both work via ``getattr``;
    anything else (a stray ``list``/``None``) falls back to ``0``.
    """
    try:
        return int(getattr(item, "run_number", None))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
