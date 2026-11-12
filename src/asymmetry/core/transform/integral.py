r"""Time-integral asymmetry and field-scan assembly.

The *integral-counting* method reduces a whole muon time spectrum to a **single
number per run** — the asymmetry over a time window — and a *field scan*
collects that number across a series of runs ordered by a run variable (field,
temperature, or run number).  This is the model-free observable behind ALC
(avoided level crossing), LF repolarisation, and QLCR analysis.

Two distinct reduction methods are provided. ``integral`` sums selected
corrected native F/B counts before forming ``(F-alpha*B)/(beta*F+alpha*B)``.
``differential`` is the explicit equal-time-bin mean of pointwise ratios,
not a field derivative. They need not agree for a changing count ratio.
Mantid's single-period Integral uses the same count-first ordering; its
Differential and dual-period operations are distinct observables.

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
pooled observation before the selected estimator; G−R combines retained
responses before evaluating its variance. Red, Green, and extracted sets use their actual
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

from dataclasses import dataclass, field, replace
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from asymmetry.core import computation
from asymmetry.core.data.dataset import MuonDataset, Run
from asymmetry.core.data.ratio_validity import RatioReason
from asymmetry.core.data.reduction_recipe import EffectiveReductionRecipe, field_scan_recipe
from asymmetry.core.transform.corrections import (
    BackgroundReference,
    CorrectionOutcomes,
    reference_observation_metadata,
)
from asymmetry.core.transform.counting_noise import (
    CountingCapture,
    CountingNoise,
    capture_counting_observations,
)
from asymmetry.core.transform.fb_counts import (
    GroupCounts,
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
from asymmetry.core.transform.scalar_observation import (
    ScalarObservation,
    count_ratio_observations,
    curve_observations,
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
    averages usable native ratio observations and reports omissions in typed results.
    Both propagate the independent count variances and shared components with
    :func:`~asymmetry.core.transform.fb_counts.linear_combination_variance`.
    Its weights are 1 for an integral and the asymmetry derivatives divided by
    the number of averaged bins for a differential mean.

    ``time`` is an optional aligned axis in μs, required with inclusive bounds
    ``t_min`` or ``t_max``. Without bounds the full record is selected. α and β
    must be positive. An undefined scalar returns ``(nan, nan)``; a defined
    zero-variance ratio retains its value with error zero. Neither has Gaussian
    fit weight. A mean with no usable observations raises ``ValueError``.
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
    ratios = count_ratio_observations(
        forward, backward, alpha, beta, mask if method == "integral" else None
    )
    result = ratios.scalar() if method == "integral" else ratios.mean(mask)
    return result.value, result.error


def integrate_curve(
    time: NDArray[np.float64],
    asymmetry: NDArray[np.float64],
    error: NDArray[np.float64],
    *,
    t_min: float | None = None,
    t_max: float | None = None,
) -> tuple[float, float]:
    """Mean of an already-formed asymmetry curve over ``[t_min, t_max]``.

    Explicit zero errors are omitted; no surviving observations raises ``ValueError``.
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
    result = curve_observations(a, e).mean(mask)
    return result.value, result.error


@dataclass(frozen=True)
class IntegralReduction:
    """Scalar value/error and known count corrections, absent for a curve-only mean."""

    value: float
    error: float
    corrections: CorrectionOutcomes | None
    observation: ScalarObservation
    effective_recipe: EffectiveReductionRecipe | None = None


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
    with whole native bins selected by inclusive sample centres.

    Parameters
    ----------
    data
        A :class:`MuonDataset` with a verified histogram observation, or an
        explicitly supplied raw :class:`Run`. Curve-only datasets support the
        explicit ``differential`` mean without count-setting overrides. For
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
        dataset's own percent-scale curve is not used for count-backed reductions.
        A curve-only ``differential`` mean uses that stored curve, converted to
        fractions, and returns ``corrections=None`` because count preparation
        is not evidenced.
    """
    mode = None
    if not isinstance(data, MuonDataset) or data.histogram_observation is not None:
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
    return IntegralReduction(
        result.value,
        result.error,
        result.corrections.get(str(mode)),
        result.observation,
        result.effective_recipe,
    )


@dataclass(frozen=True)
class PeriodIntegralReduction:
    """A period integral combination and every contributing correction decision."""

    value: float
    error: float
    corrections: dict[str, CorrectionOutcomes]
    observation: ScalarObservation
    noise: CountingNoise | None = None
    effective_recipe: EffectiveReductionRecipe | None = None


def reduce_period_integral(
    data,
    *,
    t_min=None,
    t_max=None,
    method="integral",
    alpha=None,
    grouping_ref=None,
    reference=None,
    cancel_check=None,
    counting_capture=None,
):
    """Coordinate background before integrating held-period sums or differences."""
    _validate_method(method)
    facility = None
    if isinstance(data, MuonDataset):
        observation = data.histogram_observation
        if observation is None:
            if method == "differential":
                if alpha is not None or grouping_ref is not None or reference is not None:
                    raise ValueError(
                        "A curve-only time-bin mean cannot apply count calibration, grouping, "
                        "or reference overrides; reload verified raw data for those settings."
                    )
                mask = _window_mask(data.time, t_min, t_max, data.time.size)
                result = curve_observations(
                    data.asymmetry_fraction,
                    data.error_fraction,
                    data.ratio_validity,
                    data.counting_noise.scale(np.full(data.time.size, 0.01))
                    if data.counting_noise is not None
                    else None,
                ).mean(mask)
                return PeriodIntegralReduction(
                    result.value,
                    result.error,
                    {},
                    result.observation,
                    result.noise,
                    data.effective_recipe.curve_operation(
                        "time_bin_mean",
                        t_min=t_min,
                        t_max=t_max,
                        units="fraction",
                        observation=result.observation.to_dict(),
                    )
                    if data.effective_recipe is not None
                    else None,
                )
            raise ValueError(
                "Count integral unavailable: reload raw data with verified count preparation; "
                "an attached Run does not establish count replay for this curve. "
                "Use method='differential' for an explicit time-bin mean."
            )
        run = observation.source
        reference = observation.reference if reference is None else reference
        facility = observation.facility
    else:
        run = _resolve_run(data)
    if cancel_check is not None:
        cancel_check()
    run, reference = capture_counting_observations(
        run, reference, cancel_check=cancel_check, capture=counting_capture
    )
    grouping = effective_grouping(run, grouping_ref)
    if alpha is not None:
        grouping.pop("alpha_policy", None)
    mode = PeriodMode(grouping.get("period_mode", "red"))
    alpha_used, beta_used = grouping_balances(grouping)
    _validate_alpha(alpha_used if alpha is None else alpha)
    _validate_beta(beta_used)
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
        facility=facility,
        cancel_check=cancel_check,
        reference_resolver=partial(background_reference_for_grouping, reference),
    )
    from asymmetry.core.transform.grouping import shared_automatic_alpha
    from asymmetry.core.transform.reduce import estimate_prepared_alpha

    if alpha is None and shared_automatic_alpha(grouping, reference_observation_metadata(run)):
        target = str(PeriodMode.RED) if mode == PeriodMode.GREEN_MINUS_RED else str(mode)
        alpha_used = estimate_prepared_alpha(counts[target])
    ratios, masks = {}, {}
    for label, pair in counts.items():
        time = native_count_time(
            pair.forward.counts.size, pair.bin_width_us, pair.common_t0_bin, pair.t0_time_us
        )
        mask = _window_mask(
            time,
            time[pair.window.first] if t_min is None else t_min,
            time[pair.window.last] if t_max is None else t_max,
            time.size,
        )
        masks[label] = mask
        ratios[label] = count_ratio_observations(
            pair.forward,
            pair.backward,
            alpha_used if alpha is None else alpha,
            beta_used,
            mask if method == "integral" else None,
        )
    if mode == PeriodMode.GREEN_MINUS_RED:
        ratio = ratios["green"].difference(ratios["red"])
        mask = masks["green"] & masks["red"]
    else:
        ratio, mask = ratios[str(mode)], masks[str(mode)]
    result = ratio.scalar() if method == "integral" else ratio.mean(mask)
    return PeriodIntegralReduction(
        result.value,
        result.error,
        {k: v.corrections for k, v in counts.items()},
        result.observation,
        result.noise,
        EffectiveReductionRecipe.from_dict(
            {
                "version": 1,
                "operation": "native_count_" + method,
                "period_mode": str(mode),
                "observable": "asymmetry",
                "units": "fraction",
                "alpha": alpha_used if alpha is None else alpha,
                "beta": beta_used,
                "time_range": [t_min, t_max],
                "periods": {
                    label: pair.effective_recipe.with_facts(
                        corrections=pair.corrections.to_dict(),
                        selected_native_bins=np.flatnonzero(masks[label]).tolist(),
                    ).to_dict()
                    for label, pair in counts.items()
                },
                "observation": result.observation.to_dict(),
                "uncertainty": {
                    "calibration": "conditional",
                    "counting_noise": "retained" if result.noise is not None else "marginal",
                },
            }
        ),
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
    observation: ScalarObservation | None = None
    noise: CountingNoise | None = field(default=None, repr=False)
    effective_recipe: EffectiveReductionRecipe | None = None


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
    #: Numerator scale of values/errors: builders return fractions; explicitly
    #: supplied percent scans retain percent. A derivative divides that scale by
    #: the selected x-unit without changing this recorded numerator unit.
    units: AsymmetryUnit = ASYMMETRY_FRACTION
    correction_outcomes: dict[int, dict[str, CorrectionOutcomes]] = field(default_factory=dict)
    observations: tuple[ScalarObservation | None, ...] = ()
    correlated_pairs: tuple[tuple[int, int], ...] = ()
    source_assumption: str = "Unknown acquisition identity is assumed distinct."
    effective_recipe: EffectiveReductionRecipe | None = None
    computation_revision: str | None = None

    def select(self, selection):
        """Crop/reorder parallel observations without losing fit-support facts."""
        indices = np.arange(self.n_points)[selection]
        runs = [self.run_numbers[i] for i in indices]
        positions = {old: new for new, old in enumerate(indices)}
        return replace(
            self,
            effective_recipe=self.effective_recipe.scan_selection(
                indices.tolist(), derivative=self.derivative
            )
            if self.effective_recipe is not None
            else None,
            x=self.x[selection],
            value=self.value[selection],
            error=self.error[selection],
            run_numbers=runs,
            observations=tuple(self.observations[i] for i in indices) if self.observations else (),
            correlated_pairs=tuple(
                (positions[a], positions[b])
                for a, b in self.correlated_pairs
                if a in positions and b in positions
            ),
        )

    @property
    def n_points(self) -> int:
        return int(self.x.size)

    @property
    def points(self) -> list[FieldScanPoint]:
        recipes = (
            self.effective_recipe.to_dict().get("point_recipes", [])
            if self.effective_recipe is not None and not self.derivative
            else []
        )
        return [
            FieldScanPoint(
                run_number=self.run_numbers[i],
                x=float(self.x[i]),
                value=float(self.value[i]),
                error=float(self.error[i]),
                corrections=self.correction_outcomes.get(self.run_numbers[i], {}),
                effective_recipe=EffectiveReductionRecipe.from_dict(recipes[i])
                if recipes and recipes[i] is not None
                else None,
                observation=self.observations[i] if self.observations else None,
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
    progress_callback: Callable[[int, int, str], None] | None = None,
    cancel_check: Callable[[], None] | None = None,
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

    runs = tuple(runs)
    capture = CountingCapture()
    for index, item in enumerate(runs):
        if progress_callback is not None:
            progress_callback(index, len(runs), "Reducing native counts")
        if cancel_check is not None:
            cancel_check()
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
                item,
                t_min=t_min,
                t_max=t_max,
                method=method,
                alpha=alpha,
                grouping_ref=grouping_ref,
                reference=reference,
                cancel_check=cancel_check,
                counting_capture=capture,
            )
        except ValueError as exc:
            excluded.append((run_number, str(exc)))
            continue
        points.append(
            FieldScanPoint(
                run_number,
                x_value,
                reduced.value,
                reduced.error,
                corrections=reduced.corrections,
                observation=reduced.observation,
                noise=reduced.noise,
                effective_recipe=reduced.effective_recipe,
            )
        )

    points.sort(key=lambda p: (p.x, p.run_number))
    scan = FieldScan(
        x=np.array([p.x for p in points], dtype=np.float64),
        value=np.array([p.value for p in points], dtype=np.float64),
        error=np.array([p.error for p in points], dtype=np.float64),
        run_numbers=[p.run_number for p in points],
        order_key=order_key,
        method=method,
        y_label="Integral asymmetry" if method == "integral" else "Time-bin mean",
        x_label=_X_LABELS[order_key],
        excluded=excluded,
        units=ASYMMETRY_FRACTION,
        correction_outcomes={p.run_number: p.corrections for p in points},
        observations=tuple(p.observation for p in points),
        correlated_pairs=tuple(
            (i, j)
            for i, left in enumerate(points)
            for j, right in enumerate(points[i + 1 :], i + 1)
            if left.noise is not None
            and right.noise is not None
            and abs(float(left.noise.covariance(right.noise)[0]))
            > 128 * np.finfo(float).eps * left.error * right.error
        ),
        computation_revision=computation.COMPUTATION_REVISION,
    )

    scan.effective_recipe = field_scan_recipe(points, order_key, method).with_facts(
        correlated_pairs=scan.correlated_pairs,
        source_assumption=scan.source_assumption,
    )
    return scan


def differentiate_scan(scan: FieldScan, *, max_gap: float | None = None) -> FieldScan:
    """Forward-difference derivative ``dA/dx`` of a scan (WiMDA ``dA/dB``).

    Each output point sits at the midpoint of an adjacent pair, with
    ``dA/dx = (A₂ - A₁) / (x₂ - x₁)``. Independent input errors give the marginal
    ``hypot(e₁, e₂) / |Δx|``. For a known dependent input pair its covariance is
    unsupported here: retain the value and report error NaN, not zero variance.
    Undefined source ratios remain undefined. Derived points have no new ratio
    classification; the original scan retains its scalar observation reports.
    Pairs with non-positive spacing, or spacing greater than explicit ``max_gap``,
    are skipped without interpolation or an automatic gap threshold.
    Adjacent differences share observations. These scans support display/export;
    fit the original scan rather than assigning independent derivative weights.

    Returns
    -------
    FieldScan
        With ``derivative=True``. Dividing by Δx leaves the asymmetry scale
        untouched, so values stay on the input fraction or percent scale per
        x-unit. ``units`` records that numerator scale and is carried unchanged.
    """
    if scan.derivative:
        raise ValueError("differentiate_scan expects an integral scan, not a derivative.")

    xs, ys, es = scan.x, scan.value, scan.error
    mid_x: list[float] = []
    deriv: list[float] = []
    deriv_err: list[float] = []
    run_pairs: list[int] = []

    dependent = {tuple(sorted(pair)) for pair in scan.correlated_pairs}
    for i in range(scan.n_points - 1):
        dx = float(xs[i + 1] - xs[i])
        if dx <= 0.0 or (max_gap is not None and dx > max_gap):
            continue
        mid_x.append(0.5 * (float(xs[i]) + float(xs[i + 1])))
        undefined = bool(scan.observations) and any(
            scan.observations[j] is not None
            and scan.observations[j].reason == RatioReason.UNDEFINED
            for j in (i, i + 1)
        )
        deriv.append(float("nan") if undefined else (float(ys[i + 1]) - float(ys[i])) / dx)
        deriv_err.append(
            float("nan")
            if undefined or (i, i + 1) in dependent
            else float(np.hypot(es[i], es[i + 1])) / dx
        )
        run_pairs.append(scan.run_numbers[i + 1])

    return FieldScan(
        effective_recipe=scan.effective_recipe.curve_operation(
            "forward_difference",
            max_gap=max_gap,
            output_points=len(mid_x),
            fit_eligibility="display_export_only",
        )
        if scan.effective_recipe is not None
        else None,
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
        source_assumption=(
            scan.source_assumption + " Derivative display/export only; fit the original scan. "
            "Adjacent differences share observations. Errors assume independent input pairs; "
            "known dependent pairs have unavailable uncertainty (NaN). "
            "Source ratio classifications remain on the original scan."
        ),
        computation_revision=computation.COMPUTATION_REVISION,
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


def _resolve_run(data: MuonDataset | Run) -> Run:
    if isinstance(data, Run):
        return data
    if isinstance(data, MuonDataset):
        if data.histogram_observation is not None:
            source = data.histogram_observation.source
            current = data.run.metadata if data.run is not None else source.metadata
            conditions = {
                key: current.get(key, data.metadata.get(key)) for key in ("field", "temperature")
            }
            return replace(source, metadata=source.metadata | conditions)
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
