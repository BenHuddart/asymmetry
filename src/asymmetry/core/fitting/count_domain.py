"""Fit raw single-group or forward/backward counts with Gaussian or Cash costs.

The mean is N₀ exp(-t/τμ) (1 + s A P(t)) + background; times are in µs,
physics asymmetries are percent. DT₀ is a rational count-loss increment on the
applied correction. Detector shares are inferred at each trial coefficient.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Hashable

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.engine import (
    FitResult,
    _make_cancel_guard,
    _minuit_status_message,
    _reject_affine_ties,
    drive_minuit,
    poisson_cash,
)
from asymmetry.core.fitting.grouped_time_domain import (
    GroupedTimeDomainFitResult,
    build_count_group,
    build_count_groups,
    build_fb_count_model,
    build_grouped_count_model,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.transform.deadtime import DeadtimeCalibration, _deadtime_number
from asymmetry.core.utils.constants import MUON_LIFETIME_US

#: Selectable count-fit cost functions.
COUNT_COSTS: tuple[str, ...] = ("poisson", "gaussian")

#: Optional count-loss (deadtime) parameters, applied when present in the fit set.
DEADTIME_PARAMS: tuple[str, ...] = ("DT0",)

#: Seed sentinel: give this as a parameter's ``value`` and the count-fit driver
#: computes the seed from the data instead. Supported for ``N0``, whose right
#: scale is a per-run count level no caller can know in advance — a generic seed
#: drops the forward/backward fit into a spurious minimum (see
#: :func:`fit_fb_alpha`). Omitting ``N0`` from the set entirely does the same
#: thing; the sentinel exists for callers that want to state the bounds.
SEED_FROM_DATA: float = float("nan")

#: Names the count-fit driver consumes structurally or as optional nuisances. A
#: physics-model parameter sharing one of these would be silently swallowed by the
#: name-based dispatch (popped as a nuisance, or shadowed by a fixed structural
#: value), so a collision is rejected up front.
RESERVED_COUNT_PARAMS: frozenset[str] = frozenset(
    {
        "N0",
        "background",
        "background_b",
        "amplitude",
        "relative_phase",
        "alpha",
        "sign",
        "t0",
        "tau",
        "dpsep",
        "lambda_base",
        "beta_base",
        *DEADTIME_PARAMS,
    }
)

_INF = float("inf")


def _guard_model_param_collisions(model_fn: Callable[..., NDArray]) -> None:
    """Reject a physics model whose declared parameters collide with reserved names.

    The count-fit dispatch pops its nuisances (``t0``, the deadtime terms,
    ``dpsep``, ``tau``, ``lambda_base``/``beta_base``) by name and fixes the
    structural ones (``N0``/``background``/``alpha``/``sign``/``amplitude``/
    ``relative_phase``). A model parameter literally named one of these would be
    consumed by the driver and never reach the model — a silent wrong result.
    This inspects the model's *declared* parameters (skipping the leading time
    axis; models taking only ``**kwargs`` expose their names via
    ``param_names``, guarded at the GUI seam) and fails loudly on a collision.
    """
    parameters = list(inspect.signature(model_fn).parameters.values())
    named = {
        p.name
        for p in parameters[1:]
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    collisions = sorted(named & RESERVED_COUNT_PARAMS)
    if collisions:
        raise ValueError(
            f"Model parameter(s) {collisions} collide with reserved count-fit names "
            f"{sorted(RESERVED_COUNT_PARAMS)}; rename the model parameter(s) so the "
            "count-fit dispatch is unambiguous."
        )


def _guard_beta_collision(model_fn: Callable[..., NDArray]) -> None:
    """Reject a physics model owning a ``beta`` parameter on the β-estimation path.

    The β-estimation path adds a structural shared ``beta`` scaling the backward
    polarization amplitude. ``beta`` is deliberately **not** in
    :data:`RESERVED_COUNT_PARAMS` (the stretched-exponential model's stretch
    exponent is legitimately named ``beta``, and must stay usable on the ordinary
    count-fit paths), so the clash is caught here — scoped to the β path only —
    rather than globally.
    """
    parameters = list(inspect.signature(model_fn).parameters.values())
    named = {
        p.name
        for p in parameters[1:]
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    if "beta" in named:
        raise ValueError(
            "Model parameter 'beta' collides with the β-estimation structural "
            "parameter (which scales the backward polarization amplitude). Rename "
            "the model parameter (e.g. the stretched-exponential exponent) or "
            "disable β estimation."
        )


# --- cost functions ---------------------------------------------------------


def _gaussian_chi2(counts: NDArray[np.float64], model: NDArray[np.float64]) -> float:
    """sqrt(N) Gaussian least-squares cost (WiMDA weighting)."""
    sigma = np.sqrt(np.clip(counts, 1.0, None))
    return float(np.sum(((counts - model) / sigma) ** 2))


def _cost_value(counts: NDArray[np.float64], model: NDArray[np.float64], cost: str) -> float:
    if not np.all(np.isfinite(model)):
        return _INF
    if cost == "gaussian":
        return _gaussian_chi2(counts, model)
    # Cash statistic — the single primitive shared with the grouped/global
    # driver (factored into engine.poisson_cash).
    return poisson_cash(counts, model)


def _validate_cost(cost: str) -> str:
    if cost not in COUNT_COSTS:
        raise ValueError(f"Unknown count-fit cost {cost!r}; expected one of {COUNT_COSTS}")
    return cost


# --- raw-count model wrapping ----------------------------------------------


def _percent_to_fraction(model_fn: Callable[..., NDArray]) -> Callable[..., NDArray]:
    """Scale a physics model's percent asymmetry to a fraction (WiMDA's 0.01·MusrFun).

    Asymmetry's fit models (and :func:`simulate_run`) express the asymmetry in
    percent; the count models combine it as ``1 + a`` with ``a`` a fraction, so
    the percent convention is divided out here once, keeping the model builders
    convention-agnostic.

    ``functools.wraps`` preserves the wrapped model's signature so the count
    model's phase-parameter detection (which inspects the signature) is not
    fooled by this wrapper's ``**kwargs``.
    """

    @functools.wraps(model_fn)
    def fraction(t, **kwargs):
        return 0.01 * np.asarray(model_fn(t, **kwargs), dtype=float)

    return fraction


def _with_baseline_drift(fraction_fn: Callable[..., NDArray]) -> Callable[..., NDArray]:
    """Optionally damp the polarization by a stretched-exponential baseline drift.

    WiMDA's ``Bsln lambda`` / ``Bsln beta``: the polarization is multiplied by
    ``exp(-(lambda_base*t)^beta_base)``. The parameters are popped from the call
    kwargs so they never reach the user model; when ``lambda_base`` is absent or
    zero this is an exact no-op (the default state).
    """

    @functools.wraps(fraction_fn)
    def damped(t, **kwargs):
        lam = float(kwargs.pop("lambda_base", 0.0))
        beta = float(kwargs.pop("beta_base", 1.0))
        base = np.asarray(fraction_fn(t, **kwargs), dtype=float)
        if lam > 0.0:
            arg = np.maximum(lam * np.asarray(t, dtype=float), 0.0)
            base = base * np.exp(-np.power(arg, beta))
        return base

    return damped


def _split_time_offset(kw: dict) -> float:
    """Pop and return the fittable ``t0`` time offset (microseconds), default 0."""
    return float(kw.pop("t0", 0.0))


def _split_lifetime(kw: dict, default: float) -> float:
    """Pop the optional free muon lifetime ``tau`` (µs); default = physical value.

    Popping it keeps ``tau`` from reaching the model builders (which would pass it
    to the physics model). When absent this returns ``default`` and leaves ``kw``
    untouched, so the fixed-lifetime path is unchanged.
    """
    return float(kw.pop("tau", default))


# --- count loss (deadtime) --------------------------------------------------


def _deadtime_frame_norm(dataset: MuonDataset, group) -> float:
    """Exposure Δt·F per detector, with Δt the bunched bin width in µs."""
    frames = _deadtime_number(dataset.run.grouping.get("good_frames"), "good frames", positive=True)
    width = _deadtime_number(dataset.run.histograms[0].bin_width, "bin width (µs)", positive=True)
    return width * group.metadata["grouped_time_domain_bunch_factor"] * frames


def _apply_deadtime(counts, tau_us, *, frame_norm, detector_counts):
    """Non-paralysable mean Σᵢ wᵢN/(1 + τwᵢN/(Δt·F)).

    Knoll, Radiation Detection and Measurement, 4th ed., ch. 4, Dead Time.
    Detector shares are inferred by inverting the observed per-detector traces
    at the trial coefficient; all detectors must share their time dependence.
    Undefined trial means return NaNs for the objective to reject with +∞.
    """
    if tau_us == 0.0:
        return counts
    a = tau_us / frame_norm
    inverse_denominator = 1.0 - a * detector_counts
    if not np.all(np.isfinite(inverse_denominator) & (inverse_denominator > 0)):
        return np.full_like(counts, np.nan)
    totals = np.sum(detector_counts / inverse_denominator, axis=1)
    if not np.isfinite(totals.sum()) or totals.sum() <= 0:
        return np.full_like(counts, np.nan)
    shares = totals / totals.sum()
    true_counts = shares[:, None] * counts
    denominator = 1.0 + a * true_counts
    if not np.all(np.isfinite(denominator) & (denominator > 0)):
        return np.full_like(counts, np.nan)
    return np.sum(true_counts / denominator, axis=0)


# --- double pulse -----------------------------------------------------------


def _double_pulse_factor(
    fraction_fn: Callable[..., NDArray],
    time: NDArray[np.float64],
    dpsep: float,
    sign: float,
    physics: dict,
) -> NDArray[np.float64]:
    """The two-pulse ``(1 + sign·P)`` envelope shared by the single and F+B models.

    Two muon pulses separated by ``dpsep`` (µs) each carry the polarization,
    evaluated at ``t ± dpsep/2`` and weighted by ``exp(∓dpsep/2τ_μ)`` (WiMDA
    ``ArrayMusrFunc``); the 0.5 normalization recovers the single-pulse limit as
    ``dpsep → 0`` and the second pulse is gated to ``t > dpsep/2``. The gated-out
    times are clamped to ``>= 0`` before evaluating the model, so a model
    undefined for ``t < 0`` cannot poison the (zero-weighted) early bins.
    ``sign`` is the per-histogram ±1 (single: the orientation; F+B: the forward/
    backward sign).
    """
    tau = float(MUON_LIFETIME_US)
    dpsep2 = float(dpsep) / 2.0
    c1 = np.exp(-dpsep2 / tau)
    c2 = np.exp(dpsep2 / tau)
    gate = (time > dpsep2).astype(float)
    a1 = np.asarray(fraction_fn(time + dpsep2, **physics), dtype=float)
    a2 = np.asarray(fraction_fn(np.maximum(time - dpsep2, 0.0), **physics), dtype=float)
    return 0.5 * (c1 * (1.0 + sign * a1) + gate * c2 * (1.0 + sign * a2))


def _double_pulse_single_model(fraction_fn: Callable[..., NDArray]) -> Callable[..., NDArray]:
    """Raw single-histogram count model for an ISIS double-pulse source.

    The decay envelope and ``N0``/background stay at ``t``; only the polarization
    is shifted (see :func:`_double_pulse_factor`). The per-histogram ``amplitude``
    (fixed ±1) carries the forward/backward orientation.
    """
    tau = float(MUON_LIFETIME_US)

    def model(t, *, N0, background, amplitude, relative_phase, dpsep, **physics):  # noqa: N803
        time = np.asarray(t, dtype=float)
        factor = _double_pulse_factor(fraction_fn, time, dpsep, amplitude, physics)
        return float(N0) * np.exp(-time / tau) * factor + float(background)

    return model


def _double_pulse_fb_model(fraction_fn: Callable[..., NDArray]) -> Callable[..., NDArray]:
    """Raw forward/backward count model for an ISIS double-pulse source.

    The ``fgFB`` double-pulse analogue of :func:`_double_pulse_single_model`: the
    same two-pulse envelope (:func:`_double_pulse_factor`) is carried with the
    forward/backward ``sign``, and the shared ``N0`` is split by the detector
    balance as ``N0·√alpha`` (forward) / ``N0/√alpha`` (backward) exactly as in
    :func:`build_fb_count_model`. Each side keeps its own background.
    """
    tau = float(MUON_LIFETIME_US)

    def model(t, *, alpha, N0, background, sign, dpsep, **physics):  # noqa: N803
        time = np.asarray(t, dtype=float)
        ralp = np.sqrt(abs(float(alpha)))
        if sign >= 0.0:
            scale = ralp
        else:
            scale = 1.0 / ralp if ralp > 0.0 else 0.0
        factor = _double_pulse_factor(fraction_fn, time, dpsep, sign, physics)
        return float(N0) * scale * np.exp(-time / tau) * factor + float(background)

    return model


def _raw_model(lifetime_corrected_model: Callable[..., NDArray]) -> Callable[..., NDArray]:
    """Turn a lifetime-corrected count model into a raw-count model.

    N_raw(t) = exp(-t / tau_mu) * [ N0*(1 + s*P) + bg*exp(t / tau_mu) ]
             = exp(-t / tau_mu) * N0*(1 + s*P) + bg.
    """
    tau = float(MUON_LIFETIME_US)

    def raw(t, **kwargs):
        time = np.asarray(t, dtype=float)
        corrected = np.asarray(lifetime_corrected_model(time, **kwargs), dtype=float)
        return np.exp(-time / tau) * corrected

    return raw


# --- iminuit driver ---------------------------------------------------------


def _solve(
    free: list[Parameter],
    total_cost: Callable[..., float],
    *,
    minos: bool = False,
    cancel_callback: Callable[[], bool] | None = None,
):
    """Minimise with cooperative cancellation and optional MINOS intervals."""
    from iminuit import Minuit

    guard = _make_cancel_guard(cancel_callback)

    def cost(*args) -> float:
        guard()
        return total_cost(*args)

    cost.errordef = 1.0  # Cash and sqrt-N chi-square both use errordef = 1
    names = [p.name for p in free]
    initial = [p.value for p in free]
    m = Minuit(cost, *initial, name=names)
    for i, p in enumerate(free):
        lo = p.min if p.min != -_INF else m.limits[i][0]
        hi = p.max if p.max != _INF else m.limits[i][1]
        m.limits[i] = (lo, hi)
    drive_minuit(m, migrad_kwargs={"iterate": 5, "use_simplex": True}, minos=minos)
    return m


def _result_from_minuit(
    m,
    params: ParameterSet,
    free_names: list[str],
    *,
    time: NDArray[np.float64],
    counts: NDArray[np.float64],
    model_values: NDArray[np.float64],
    keep_params: list[str] | None = None,
    chi_squared: float | None = None,
    followers: dict[str, str] | None = None,
    success_message: str = "Count fit successful",
    failure_prefix: str = "Count fit failed",
) -> FitResult:
    """Pack a :class:`FitResult` for one count-domain (or one F/B side).

    ``chi_squared`` overrides the reported cost; pass the per-side cost for a
    joint forward/backward fit (``m.fval`` is the combined cost). Defaults to
    ``m.fval`` (correct for a single-domain fit). ``followers`` (``{follower:
    main}``) maps equality-link followers to their group main so a tied parameter
    is reported at the main's *fitted* value, not its pre-fit seed.
    """
    keep = set(keep_params) if keep_params is not None else None
    link_followers = followers or {}
    # MINOS intervals (when run) live on m.merrors keyed by the free parameter name;
    # empty when MINOS was not requested. A display-only overlay — the symmetric
    # HESSE errors below are unchanged.
    merrors = getattr(m, "merrors", None)
    result_params = ParameterSet()
    uncertainties: dict[str, float] = {}
    minos_errors: dict[str, tuple[float, float]] = {}
    for p in params:
        if keep is not None and p.name not in keep:
            continue
        # A link-follower mirrors its group main: report the main's fitted value.
        source = link_followers.get(p.name, p.name)
        if source in free_names:
            idx = free_names.index(source)
            result_params.add(Parameter(name=p.name, value=m.values[idx], min=p.min, max=p.max))
            if m.errors[idx] is not None:
                uncertainties[p.name] = float(m.errors[idx])
            if merrors is not None:
                try:
                    merror = merrors[source]
                except (KeyError, TypeError):
                    merror = None
                if merror is not None and getattr(merror, "is_valid", False):
                    minos_errors[p.name] = (float(merror.lower), float(merror.upper))
        else:
            result_params.add(Parameter(name=p.name, value=p.value, fixed=True))

    covariance = None
    covariance_order: list[str] = []
    if m.valid and getattr(m, "covariance", None) is not None:
        cov = np.asarray(m.covariance, dtype=float)
        cov_names = [n for n in free_names if keep is None or n in keep]
        cov_idx = [free_names.index(n) for n in cov_names]
        if cov_idx:
            covariance = cov[np.ix_(cov_idx, cov_idx)]
            covariance_order = cov_names

    ndata = int(np.size(counts))
    nfree = len(free_names if keep is None else covariance_order)
    residuals = np.asarray(counts, dtype=float) - np.asarray(model_values, dtype=float)
    chi2 = float(m.fval) if chi_squared is None else float(chi_squared)
    return FitResult(
        success=bool(m.valid),
        chi_squared=chi2,
        reduced_chi_squared=chi2 / max(ndata - nfree, 1),
        parameters=result_params,
        uncertainties=uncertainties,
        covariance=covariance,
        covariance_parameters=covariance_order,
        residuals=residuals,
        message=_minuit_status_message(
            m, success_message=success_message, failure_prefix=failure_prefix
        ),
        function_calls=int(getattr(m, "nfcn", 0) or 0),
        dof=ndata - nfree,
        minos_errors=minos_errors or None,
    )


def _record_deadtime_calibration(result, dataset, groups):
    """Capture the applied base and fitted detectors before mutable GUI edits."""
    if result.success and "DT0" in result.parameters:
        result.deadtime_calibration = DeadtimeCalibration.from_increment(
            groups[0].deadtime,
            result.parameters["DT0"].value,
            n_histograms=len(dataset.run.histograms),
            detector_indices=sorted({i for group in groups for i in group.detector_indices}),
            error_us=result.uncertainties.get("DT0"),
            source_run=dataset.run_number,
        )


# --- dpsep refinement -------------------------------------------------------

#: Grid sizes for the coarse -> fine dpsep scan. The second-pulse onset gate is
#: non-smooth in dpsep, so migrad cannot descend it; a grid scan locates the
#: separation and migrad still refines the other parameters at each grid point.
_DPSEP_SCAN_COARSE = 17
_DPSEP_SCAN_FINE = 9


def _with_fixed_dpsep(params: ParameterSet, value: float) -> ParameterSet:
    """Return a copy of ``params`` with ``dpsep`` fixed at ``value``."""
    out = ParameterSet()
    for p in params:
        if p.name == "dpsep":
            out.add(Parameter(name="dpsep", value=float(value), fixed=True))
        else:
            out.add(p)
    return out


def _dpsep_scan_bounds(dpsep_param: Parameter) -> tuple[float, float]:
    """Return the ``(lo, hi)`` scan window for a free dpsep.

    Uses the parameter's finite bounds when present; otherwise falls back to a
    half-to-three-halves window around the seed (a refinement around the
    instrument value).
    """
    lo = float(dpsep_param.min) if np.isfinite(dpsep_param.min) else None
    hi = float(dpsep_param.max) if np.isfinite(dpsep_param.max) else None
    seed = max(float(dpsep_param.value), 1.0e-6)
    if lo is None:
        lo = max(0.0, 0.5 * seed)
    if hi is None:
        hi = 1.5 * seed
    lo = max(0.0, lo)
    if hi <= lo:
        hi = lo + max(seed, 1.0e-3)
    return lo, hi


def _refine_free_dpsep(
    dpsep_param: Parameter,
    fit_fixed: Callable[[float], object],
    cost_of: Callable[[object], float],
):
    """Locate a free dpsep by a coarse -> fine grid scan and return the best fit.

    ``fit_fixed(value)`` runs the standard count fit with dpsep fixed at
    ``value`` (migrad refines the remaining parameters); ``cost_of(result)``
    extracts that fit's cost. The coarse pass spans the scan window; the fine
    pass refines around the coarse minimum to roughly the fine-grid spacing.
    """

    def _cost(result) -> float:
        try:
            value = float(cost_of(result))
        except (TypeError, ValueError):
            return _INF
        return value if np.isfinite(value) else _INF

    def _evaluate(grid):
        return [(g, result := fit_fixed(g), _cost(result)) for g in grid]

    lo, hi = _dpsep_scan_bounds(dpsep_param)
    coarse = _evaluate(np.linspace(lo, hi, _DPSEP_SCAN_COARSE))
    best_g, best_result, best_cost = min(coarse, key=lambda item: item[2])

    step = (hi - lo) / (_DPSEP_SCAN_COARSE - 1)
    fine_lo = max(lo, best_g - step)
    fine_hi = min(hi, best_g + step)
    for _g, result, cost in _evaluate(np.linspace(fine_lo, fine_hi, _DPSEP_SCAN_FINE)):
        if cost < best_cost:
            best_result, best_cost = result, cost
    return best_result


def _has_free_dpsep(params: ParameterSet) -> bool:
    return "dpsep" in params and not params["dpsep"].fixed


# --- single histogram -------------------------------------------------------


def _single_histogram_amplitude(side: str) -> float:
    return -1.0 if str(side).lower() == "backward" else 1.0


def fit_single_histogram(
    dataset: MuonDataset,
    group_id: int,
    model_fn: Callable[..., NDArray],
    params: ParameterSet,
    *,
    side: str = "forward",
    cost: str = "poisson",
    t_min: float | None = None,
    t_max: float | None = None,
    exclude: tuple[float, float] | None = None,
    minos: bool = False,
    cancel_callback: Callable[[], bool] | None = None,
) -> FitResult:
    """Fit one detector group's raw counts (the musrfit fittype-0 analogue).

    The count model is ``N0 * exp(-t/tau) * (1 + s*A*P(t)) + bg`` with ``s`` the
    forward (+1) / backward (-1) sign set by ``side`` (``"selected"`` uses +1).
    ``params`` supplies the nuisances ``N0`` and ``background`` plus the physics
    model parameters; the muon lifetime is held fixed at the physical value.

    Scale: the physics model's asymmetry parameters (``A``/``A0``, ``baseline``)
    are **in percent (0–100)**, as everywhere else in the fitting layer —
    :func:`_percent_to_fraction` divides the 100 out internally so ``1 + a``
    combines a true fraction with the counts. Seed and read ``A`` in percent. The
    *data* here is raw counts, so the percent/fraction question touches only the
    parameters, not the arrays.

    Optional window/nuisance flexibility, all no-ops unless requested:
    ``exclude`` drops an interior time window from the fit; a free ``t0``
    parameter shifts the model time axis; free ``lambda_base`` / ``beta_base``
    parameters add a stretched-exponential baseline drift; the deadtime
    parameter ``DT0`` adds a rational non-paralysable count-loss term;
    a ``dpsep``
    parameter switches on the ISIS double-pulse count model — fixed by default,
    or located by a coarse->fine grid scan when supplied free (the non-smooth
    pulse-onset gate defeats gradient fitting). A ``tau`` parameter frees the muon
    lifetime (musrfit-style; default fixed at the physical value); it is not
    supported together with the double-pulse model.
    """
    _validate_cost(cost)
    _guard_model_param_collisions(model_fn)
    if "tau" in params and "dpsep" in params:
        raise ValueError("Free muon lifetime is not supported with the double-pulse model")
    if _has_free_dpsep(params):
        return _refine_free_dpsep(
            params["dpsep"],
            fit_fixed=lambda value: fit_single_histogram(
                dataset,
                group_id,
                model_fn,
                _with_fixed_dpsep(params, value),
                side=side,
                cost=cost,
                t_min=t_min,
                t_max=t_max,
                exclude=exclude,
            ),
            cost_of=lambda result: result.chi_squared,
        )
    group = build_count_group(
        dataset, group_id, t_min=t_min, t_max=t_max, lifetime_corrected=False, exclude=exclude
    )
    time = np.asarray(group.time, dtype=float)
    counts = np.asarray(group.counts, dtype=float)

    tau = float(MUON_LIFETIME_US)
    base_decay = np.exp(-time / tau)  # hoisted: loop-invariant unless t0 is free
    fraction_fn = _with_baseline_drift(_percent_to_fraction(model_fn))
    double_pulse = "dpsep" in params
    free_lifetime = "tau" in params  # fit (or fix at a non-physical value) the lifetime
    dp_model = _double_pulse_single_model(fraction_fn) if double_pulse else None
    corrected_model = None if double_pulse else build_grouped_count_model(fraction_fn)
    frame_norm = _deadtime_frame_norm(dataset, group) if "DT0" in params else 1.0

    # The per-group amplitude/phase are fixed nuisances here: the sign carries the
    # forward/backward orientation and the physics model owns the real amplitude.
    params = _with_fixed(
        params,
        amplitude=_single_histogram_amplitude(side),
        relative_phase=0.0,
    )
    _reject_affine_ties([params], "Count-domain fitting")
    free = params.free_parameters
    free_names = [p.name for p in free]
    fixed_kw = {p.name: p.value for p in params if p.fixed}
    followers = params.link_followers()

    def predict(args) -> NDArray[np.float64]:
        kw = {**fixed_kw, **dict(zip(free_names, args, strict=False))}
        for follower, main in followers.items():
            kw[follower] = kw[main]
        t0 = _split_time_offset(kw)
        dt0 = float(kw.pop("DT0", 0.0))
        tau_eval = _split_lifetime(kw, tau)
        if free_lifetime or t0:
            t_eval = time + t0 if t0 else time
            decay = np.exp(-t_eval / tau_eval)
        else:
            t_eval = time
            decay = base_decay
        if double_pulse:
            model = np.asarray(dp_model(t_eval, **kw), dtype=float)
        elif free_lifetime:
            # Evaluate the raw model directly so the flat background does not pick
            # up the module lifetime: N_raw = exp(-t/tau)*N0*(1 + s*P) + bg.
            bg = float(kw["background"])
            n0_term = np.asarray(corrected_model(t_eval, **{**kw, "background": 0.0}), dtype=float)
            model = decay * n0_term + bg
        else:
            # raw count = exp(-t/tau) * [lifetime-corrected model]; the cached
            # decay avoids re-exponentiating the invariant time axis each call.
            model = decay * np.asarray(corrected_model(t_eval, **kw), dtype=float)
        return _apply_deadtime(
            model, dt0, frame_norm=frame_norm, detector_counts=group.detector_counts
        )

    def total_cost(*args) -> float:
        return _cost_value(counts, predict(args), cost)

    m = _solve(free, total_cost, minos=minos, cancel_callback=cancel_callback)
    model_values = predict([m.values[i] for i in range(len(free_names))])
    result = _result_from_minuit(
        m,
        params,
        free_names,
        time=time,
        counts=counts,
        model_values=model_values,
        followers=followers,
        success_message="Single-histogram count fit successful",
        failure_prefix="Single-histogram count fit failed",
    )
    _record_deadtime_calibration(result, dataset, [group])
    return result


# --- forward/backward with free alpha --------------------------------------


def fit_fb_alpha(
    dataset: MuonDataset,
    forward_group: int,
    backward_group: int,
    model_fn: Callable[..., NDArray],
    params: ParameterSet,
    *,
    cost: str = "poisson",
    t_min: float | None = None,
    t_max: float | None = None,
    exclude: tuple[float, float] | None = None,
    estimate_beta: bool = False,
    minos: bool = False,
    cancel_callback: Callable[[], bool] | None = None,
) -> GroupedTimeDomainFitResult:
    """Simultaneously fit forward and backward counts with the balance alpha free.

    Recovers the detector balance ``alpha`` from a transverse-field calibration
    run the statistically proper way (WiMDA ``fgFB``), reporting its correlation
    with the physics amplitude — strong in TF runs. ``params`` must contain the
    shared ``alpha``, a forward background ``background`` and a backward
    background ``background_b``, plus the physics model parameters — whose
    asymmetry amplitudes (``A``/``A0``, ``baseline``) are **in percent (0–100)**,
    as in :func:`fit_single_histogram`.

    Returns a :class:`GroupedTimeDomainFitResult` whose ``group_results`` are
    keyed by ``forward_group`` / ``backward_group`` and whose
    ``shared_parameters`` hold the fitted ``alpha``/``N0``/physics. The (alpha,
    amplitude) covariance is available on either group result.

    Seeding ``N0``
    --------------
    ``N0`` may be **omitted**, or supplied with the value
    :data:`SEED_FROM_DATA` (or any non-positive / non-finite value), in which
    case it is seeded from the data: the forward window's first positive bin
    lifted back to ``t = 0`` by ``exp(t/tau_mu)``. This is the default because the
    right ``N0`` is a per-run count level no caller can know in advance, and a
    generically scaled seed does not merely converge slowly — it converges
    *cleanly* on a spurious minimum, with a tiny ``alpha`` and a compensating
    amplitude balance. An explicit positive ``N0`` is used as given, and a fixed
    ``N0`` is never overruled.

    Scale sanity check
    ------------------
    Because that spurious minimum is a valid minimum as far as Migrad is
    concerned, a converged fit is additionally checked against the counts: the
    fitted ``alpha`` must sit within :data:`_FB_ALPHA_SCALE_FACTOR` of the
    window's observed forward/backward count ratio, and the fitted forward scale
    ``N0·√alpha`` within :data:`_FB_N0_SCALE_FACTOR` of the observed count level
    at ``t = 0``. A violation returns ``success=False`` with a message naming the
    discrepancy (and the same text appended to each group result's ``warnings``)
    rather than presenting the walk-off as a clean convergence. Neither check
    applies to a parameter the caller fixed.

    The same optional nuisances as :func:`fit_single_histogram` apply, all
    no-ops unless requested: ``exclude`` drops an interior window; a free ``t0``
    shifts the model time axis; free ``lambda_base`` / ``beta_base`` add a
    baseline drift; ``DT0`` adds rational non-paralysable count loss;
    a ``dpsep`` parameter switches on the ISIS double-pulse
    model (the √α-tied model evaluated at ``t ± dpsep/2``); a ``tau`` parameter
    frees the shared muon lifetime (reported in ``shared_parameters``; not
    supported with the double-pulse model).

    When ``estimate_beta`` is true a shared free ``beta`` is added (positive
    lower-bound clamp, fit box [0.01, 10], seed 1.0) that scales the backward
    polarization amplitude only — the musrfit fit-type-2 asymmetry balance
    measured on a weak-TF calibration run. ``shared_parameters`` then also
    reports ``beta``, and its correlation with ``alpha`` is available on either
    group result's covariance block. This path rejects a physics model owning a
    ``beta`` parameter (e.g. the stretched exponential) so the structural name is
    unambiguous, and is not supported with the double-pulse model. Existing
    ``estimate_beta=False`` callers are unaffected.
    """
    _validate_cost(cost)
    _guard_model_param_collisions(model_fn)
    if estimate_beta:
        _guard_beta_collision(model_fn)
        if "dpsep" in params:
            raise ValueError("β estimation is not supported with the double-pulse model")
    if "tau" in params and "dpsep" in params:
        raise ValueError("Free muon lifetime is not supported with the double-pulse model")
    for required in ("alpha", "background", "background_b"):
        if required not in params:
            raise ValueError(f"Forward/backward count fit requires a {required!r} parameter")
    if int(forward_group) == int(backward_group):
        raise ValueError("Forward/backward count fit needs two distinct groups")
    # alpha is physically positive and the model is sign-degenerate (sqrt(abs)),
    # so enforce a positive lower bound rather than report a negative balance.
    params = _clamp_alpha_positive(params)
    # beta scales the backward amplitude only and is likewise positive; add it
    # (or clamp an existing free one) only on the β-estimation path.
    if estimate_beta:
        params = _ensure_beta_param(params)

    # A free dpsep is located by a grid scan, not migrad (non-smooth pulse gate).
    if _has_free_dpsep(params):
        return _refine_free_dpsep(
            params["dpsep"],
            fit_fixed=lambda value: fit_fb_alpha(
                dataset,
                forward_group,
                backward_group,
                model_fn,
                _with_fixed_dpsep(params, value),
                cost=cost,
                t_min=t_min,
                t_max=t_max,
                exclude=exclude,
                cancel_callback=cancel_callback,
            ),
            cost_of=lambda result: sum(gr.chi_squared for gr in result.group_results.values()),
        )

    # Build both banks in ONE context so they share a common t0 alignment.
    g_fwd, g_bwd = build_count_groups(
        dataset,
        [forward_group, backward_group],
        t_min=t_min,
        t_max=t_max,
        lifetime_corrected=False,
        exclude=exclude,
    )
    time_f = np.asarray(g_fwd.time, dtype=float)
    counts_f = np.asarray(g_fwd.counts, dtype=float)
    time_b = np.asarray(g_bwd.time, dtype=float)
    counts_b = np.asarray(g_bwd.counts, dtype=float)

    # Seed N0 from the data unless the caller supplied a usable value. The count
    # scale is a per-run fact; a caller who does not know it lands in a spurious
    # minimum that converges cleanly, so good seeding is the default here rather
    # than something only the beta-calibration wrapper knew to do.
    n0_seed = _count_level_at_zero(time_f, counts_f)
    if _needs_data_seed(params, "N0"):
        params = _with_seeded_value(params, "N0", n0_seed if n0_seed is not None else 1.0)

    tau = float(MUON_LIFETIME_US)
    base_decay_f = np.exp(-time_f / tau)
    base_decay_b = np.exp(-time_b / tau)
    fraction_fn = _with_baseline_drift(_percent_to_fraction(model_fn))
    double_pulse = "dpsep" in params
    free_lifetime = "tau" in params
    fb_dp_model = _double_pulse_fb_model(fraction_fn) if double_pulse else None
    fb_corrected = (
        None if double_pulse else build_fb_count_model(fraction_fn, with_beta=estimate_beta)
    )

    frame_norm_f = _deadtime_frame_norm(dataset, g_fwd) if "DT0" in params else 1.0
    frame_norm_b = _deadtime_frame_norm(dataset, g_bwd) if "DT0" in params else 1.0

    _reject_affine_ties([params], "Count-domain fitting")
    free = params.free_parameters
    free_names = [p.name for p in free]
    fixed_kw = {p.name: p.value for p in params if p.fixed}
    followers = params.link_followers()
    # Shared physics/scale params: everything except the two per-side backgrounds,
    # the time offset, the deadtime terms and the lifetime (each applied per eval,
    # not handed to the physics model).
    _per_eval = ("background", "background_b", "t0", "tau", *DEADTIME_PARAMS)
    shared_keys = [n for n in free_names if n not in _per_eval]
    shared_fixed = {k: v for k, v in fixed_kw.items() if k not in _per_eval}  # hoisted

    def _eval(args, time, base_decay, counts_key, sign, frame_norm, group) -> NDArray[np.float64]:
        kw = {**fixed_kw, **dict(zip(free_names, args, strict=False))}
        for follower, main in followers.items():
            kw[follower] = kw[main]
        t0 = _split_time_offset(kw)
        dt0 = float(kw.pop("DT0", 0.0))
        tau_eval = _split_lifetime(kw, tau)
        if free_lifetime or t0:
            t_eval = time + t0 if t0 else time
            decay = np.exp(-t_eval / tau_eval)
        else:
            t_eval = time
            decay = base_decay
        shared = {**shared_fixed, **{k: kw[k] for k in shared_keys if k in kw}}
        if double_pulse:
            # The double-pulse model carries its own decay envelope and background.
            model = np.asarray(
                fb_dp_model(t_eval, sign=sign, background=kw[counts_key], **shared), dtype=float
            )
        elif free_lifetime:
            # Evaluate the raw model directly so the flat background does not pick
            # up the module lifetime (the √α-tied N0 term carries the free tau).
            bg = float(kw[counts_key])
            n0_term = np.asarray(
                fb_corrected(t_eval, sign=sign, background=0.0, **shared), dtype=float
            )
            model = decay * n0_term + bg
        else:
            model = decay * np.asarray(
                fb_corrected(t_eval, sign=sign, background=kw[counts_key], **shared), dtype=float
            )
        return _apply_deadtime(
            model, dt0, frame_norm=frame_norm, detector_counts=group.detector_counts
        )

    def predict_f(args):
        return _eval(args, time_f, base_decay_f, "background", +1.0, frame_norm_f, g_fwd)

    def predict_b(args):
        return _eval(args, time_b, base_decay_b, "background_b", -1.0, frame_norm_b, g_bwd)

    def total_cost(*args) -> float:
        return _cost_value(counts_f, predict_f(args), cost) + _cost_value(
            counts_b, predict_b(args), cost
        )

    # MINOS runs on the fixed-dpsep solve below (the common case). When dpsep is
    # free it is located by the non-smooth grid scan above, where a likelihood-walk
    # interval is not well defined, so that path stays HESSE-only.
    m = _solve(free, total_cost, minos=minos, cancel_callback=cancel_callback)
    fitted = [m.values[i] for i in range(len(free_names))]
    model_f = predict_f(fitted)
    model_b = predict_b(fitted)
    chi2_f = _cost_value(counts_f, model_f, cost)
    chi2_b = _cost_value(counts_b, model_b, cost)

    # Both sides retain shared parameters and their own background.
    fwd_keep = [p.name for p in params if p.name != "background_b"]
    bwd_keep = [p.name for p in params if p.name != "background"]
    result_f = _result_from_minuit(
        m,
        params,
        free_names,
        time=time_f,
        counts=counts_f,
        model_values=model_f,
        keep_params=fwd_keep,
        chi_squared=chi2_f,
        followers=followers,
        success_message="Forward/backward count fit successful",
        failure_prefix="Forward/backward count fit failed",
    )
    result_b = _result_from_minuit(
        m,
        params,
        free_names,
        time=time_b,
        counts=counts_b,
        model_values=model_b,
        keep_params=bwd_keep,
        chi_squared=chi2_b,
        followers=followers,
        success_message="Forward/backward count fit successful",
        failure_prefix="Forward/backward count fit failed",
    )

    for result in (result_f, result_b):
        _record_deadtime_calibration(result, dataset, [g_fwd, g_bwd])

    shared_parameters = ParameterSet()
    # Every free parameter except the two per-side backgrounds is shared: the
    # scale/physics params plus the free per-eval nuisances (t0, tau, deadtime
    # terms). The backgrounds are reported on each group result instead.
    side_only = ("background", "background_b")
    for name in free_names:
        if name in side_only:
            continue
        p = params[name]
        idx = free_names.index(name)
        shared_parameters.add(Parameter(name=name, value=m.values[idx], min=p.min, max=p.max))
    for p in params:
        if p.fixed and p.name not in side_only:
            shared_parameters.add(Parameter(name=p.name, value=p.value, fixed=True))

    group_results: dict[Hashable, FitResult] = {
        int(forward_group): result_f,
        int(backward_group): result_b,
    }
    success = bool(m.valid)
    message = (
        "Forward/backward count fit successful" if success else "Forward/backward count fit failed"
    )
    if success:
        free_values = dict(zip(free_names, fitted, strict=True))
        complaint = _fb_scale_complaint(
            params,
            {name: float(free_values[name]) for name in ("alpha", "N0") if name in free_values},
            counts_f=counts_f,
            counts_b=counts_b,
            n0_seed=n0_seed,
        )
        if complaint:
            success = False
            message = f"Forward/backward count fit {complaint}"
            for result in (result_f, result_b):
                result.warnings.append(message)
    return GroupedTimeDomainFitResult(
        success=success,
        group_results=group_results,
        shared_parameters=shared_parameters,
        message=message,
    )


# --- plot overlay -----------------------------------------------------------


def _corrected_overlay(group, fit_result: FitResult) -> tuple[NDArray, NDArray]:
    """Recover the fitted model on the displayed (lifetime-corrected) scale.

    The count fit minimises a raw-count model; the Individual-Groups plot shows
    lifetime-corrected counts. ``FitResult.residuals`` already holds
    ``counts - model`` at the fit points, so the raw model is recovered exactly
    as ``counts - residuals`` (no model re-evaluation) and scaled to the displayed
    axis by ``exp(t / tau_mu)``. Returns ``(time, corrected_model)``.
    """
    time = np.asarray(group.time, dtype=float)
    counts = np.asarray(group.counts, dtype=float)
    residuals = np.asarray(fit_result.residuals, dtype=float)
    if residuals.shape != counts.shape:
        raise ValueError("Count-fit residuals do not match the rebuilt group trace")
    model_raw = counts - residuals
    return time, model_raw * np.exp(time / float(MUON_LIFETIME_US))


def single_histogram_overlay(
    dataset: MuonDataset,
    group_id: int,
    fit_result: FitResult,
    *,
    t_min: float | None = None,
    t_max: float | None = None,
    exclude: tuple[float, float] | None = None,
) -> dict[int, tuple[NDArray, NDArray]]:
    """Overlay curve for a single-histogram count fit, keyed by group id.

    Rebuilds the same raw-count trace the fit used and returns
    ``{group_id: (time, corrected_model)}`` for plotting on the lifetime-corrected
    Individual-Groups axis. The window arguments must match the fit call.
    """
    group = build_count_group(
        dataset, group_id, t_min=t_min, t_max=t_max, lifetime_corrected=False, exclude=exclude
    )
    return {int(group_id): _corrected_overlay(group, fit_result)}


def fb_overlay_curves(
    dataset: MuonDataset,
    forward_group: int,
    backward_group: int,
    grouped_result: GroupedTimeDomainFitResult,
    *,
    t_min: float | None = None,
    t_max: float | None = None,
    exclude: tuple[float, float] | None = None,
) -> dict[int, tuple[NDArray, NDArray]]:
    """Overlay curves for a forward/backward count fit, keyed by group id.

    Rebuilds both banks in ONE shared context (matching :func:`fit_fb_alpha`, so
    the time axes line up with the residuals stored on each side's result) and
    returns ``{forward_group: (t, y), backward_group: (t, y)}`` on the displayed
    lifetime-corrected scale.
    """
    g_fwd, g_bwd = build_count_groups(
        dataset,
        [forward_group, backward_group],
        t_min=t_min,
        t_max=t_max,
        lifetime_corrected=False,
        exclude=exclude,
    )
    overlays: dict[int, tuple[NDArray, NDArray]] = {}
    for gid, group in ((int(forward_group), g_fwd), (int(backward_group), g_bwd)):
        side_result = grouped_result.group_results.get(gid)
        if side_result is not None:
            overlays[gid] = _corrected_overlay(group, side_result)
    return overlays


# --- helpers ----------------------------------------------------------------


def _clamp_alpha_positive(params: ParameterSet) -> ParameterSet:
    """Return a copy of ``params`` with a positive lower bound on a free ``alpha``.

    ``build_fb_count_model`` uses ``sqrt(abs(alpha))``, so the model is degenerate
    under ``alpha → -alpha``; without a positive floor the fit can settle on a
    physically meaningless negative balance. A no-op if alpha is fixed or already
    bounded above zero.
    """
    alpha = params["alpha"]
    if alpha.fixed or alpha.min > 0.0:
        return params
    floor = 1.0e-6
    out = ParameterSet()
    for p in params:
        if p.name == "alpha":
            # The model is sign-degenerate, so |seed| is the physically equivalent
            # start — and a far better one than the floor for a negative seed.
            out.add(
                Parameter(
                    name=p.name,
                    value=max(abs(float(p.value)), floor),
                    min=floor,
                    max=p.max,
                    fixed=p.fixed,
                    link_group=p.link_group,
                )
            )
        else:
            out.add(p)
    return out


#: Default fit box for the β-estimation shared parameter (the UI spin stays
#: wider at [0.01, 1000]; the fit box is tight around physical values).
_BETA_FIT_BOUNDS: tuple[float, float] = (0.01, 10.0)


def _ensure_beta_param(params: ParameterSet) -> ParameterSet:
    """Return a copy of ``params`` carrying a free, positive-clamped ``beta``.

    The backward-amplitude balance ``beta`` scales only the backward polarization,
    so — like ``alpha`` — it is physically positive and seeded at 1.0. A missing
    ``beta`` is added at the default seed/box; an existing free ``beta`` gets a
    positive lower bound (mirroring :func:`_clamp_alpha_positive`). A fixed
    ``beta`` is left untouched.
    """
    floor, ceil = _BETA_FIT_BOUNDS
    if "beta" not in params:
        out = ParameterSet()
        for p in params:
            out.add(p)
        out.add(Parameter(name="beta", value=1.0, min=floor, max=ceil))
        return out
    beta = params["beta"]
    if beta.fixed or beta.min > 0.0:
        return params
    out = ParameterSet()
    for p in params:
        if p.name == "beta":
            out.add(
                Parameter(
                    name=p.name,
                    value=max(abs(float(p.value)), floor),
                    min=floor,
                    max=p.max if p.max != _INF else ceil,
                    fixed=p.fixed,
                    link_group=p.link_group,
                )
            )
        else:
            out.add(p)
    return out


def _count_level_at_zero(time: NDArray[np.float64], counts: NDArray[np.float64]) -> float | None:
    """Lifetime-corrected count level extrapolated to ``t = 0``, or ``None``.

    The count models are all ``N0 · scale · exp(-t/tau_mu) · (1 + …) + bg``, so the
    right scale for ``N0`` is the fit window's first bin lifted back to ``t = 0``.
    This is the windowed generalisation of the private first-good-bin seed the
    beta-calibration module used, and matches it exactly when the window starts at
    ``t0``. The first *positive* bin is used, so an empty leading bin does not
    collapse the seed.
    """
    positive = np.flatnonzero(np.isfinite(counts) & (counts > 0.0))
    if positive.size == 0:
        return None
    index = int(positive[0])
    level = float(counts[index]) * float(np.exp(float(time[index]) / float(MUON_LIFETIME_US)))
    return level if np.isfinite(level) and level > 0.0 else None


def _needs_data_seed(params: ParameterSet, name: str) -> bool:
    """Whether *name* should be seeded from the data rather than from *params*."""
    if name not in params:
        return True
    parameter = params[name]
    if parameter.fixed:
        # A fixed value is a declaration, not a seed; never overrule it.
        return False
    value = float(parameter.value)
    return not np.isfinite(value) or value <= 0.0


def _with_seeded_value(params: ParameterSet, name: str, value: float) -> ParameterSet:
    """Copy of *params* with *name* present at *value*, keeping bounds and order."""
    out = ParameterSet()
    seeded = False
    for p in params:
        if p.name == name:
            out.add(
                Parameter(
                    name=p.name,
                    value=float(value),
                    min=p.min,
                    max=p.max,
                    fixed=p.fixed,
                    link_group=p.link_group,
                )
            )
            seeded = True
        else:
            out.add(p)
    if not seeded:
        out.add(Parameter(name=name, value=float(value), min=0.0))
    return out


#: Scale sanity checks applied to a converged forward/backward count fit.
#:
#: ``alpha`` is a detector balance, so it must sit within a broad factor of the
#: window's observed forward/backward count ratio. The model fixes how far apart
#: they can legitimately be: ``sum(F)/sum(B) = alpha·(1 + a)/(1 − beta·a)`` with
#: ``a`` the window-mean asymmetry, which for a real µSR run is at most a few
#: tenths (a surface-muon initial asymmetry is ~0.25, and relaxation or
#: oscillation averages it down further) — a factor of two at the extreme. Three
#: leaves margin for a synthetic or pathological amplitude without admitting the
#: order-of-magnitude walk-offs a bad seed produces. Likewise the fitted forward
#: scale ``N0·√alpha`` must land within two decades of the observed count level at
#: ``t = 0``. Both compare the fit against the *data*, so unlike a fixed
#: plausibility band on alpha they cannot be defeated by a caller who widened the
#: fit bounds as a matter of course.
_FB_ALPHA_SCALE_FACTOR = 3.0
_FB_N0_SCALE_FACTOR = 100.0


def _fb_scale_complaint(
    params: ParameterSet,
    fitted: dict[str, float],
    *,
    counts_f: NDArray[np.float64],
    counts_b: NDArray[np.float64],
    n0_seed: float | None,
) -> str:
    """Message describing an implausible converged scale, or ``""`` when sane.

    Migrad reports a valid minimum whether or not it is *the* minimum. Driven
    from a badly scaled ``N0`` seed the forward/backward fit has a spurious one
    at a tiny alpha with a compensating amplitude balance, and it converges there
    cleanly. These two checks compare the converged scale against the counts
    themselves, which is what a caller cannot see from ``success`` alone.

    ``fitted`` holds the *free* parameters' converged values; a parameter the
    caller fixed is a declaration and is never complained about, though a fixed
    alpha is still read (from ``params``) to form the forward count scale.
    """
    alpha = fitted.get("alpha", float(params["alpha"].value))
    if "alpha" in fitted:
        sum_f = float(np.sum(np.clip(counts_f, 0.0, None)))
        sum_b = float(np.sum(np.clip(counts_b, 0.0, None)))
        if sum_f > 0.0 and sum_b > 0.0 and alpha > 0.0:
            observed = sum_f / sum_b
            if not (
                observed / _FB_ALPHA_SCALE_FACTOR <= alpha <= observed * _FB_ALPHA_SCALE_FACTOR
            ):
                return (
                    f"converged at alpha={alpha:.4g}, more than "
                    f"{_FB_ALPHA_SCALE_FACTOR:g}x away from the window's "
                    f"forward/backward count ratio {observed:.4g} — this is a "
                    "spurious minimum, not a detector balance; check the N0 seed "
                    "(omit it, or pass SEED_FROM_DATA, to have it seeded from the "
                    "data) and the fit window"
                )
    n0 = fitted.get("N0")
    if n0 is not None and n0_seed is not None:
        scale = n0 * float(np.sqrt(abs(alpha))) if np.isfinite(alpha) else n0
        if scale > 0.0 and not (
            n0_seed / _FB_N0_SCALE_FACTOR <= scale <= n0_seed * _FB_N0_SCALE_FACTOR
        ):
            return (
                f"converged at a forward count scale N0*sqrt(alpha)={scale:.4g}, "
                f"more than {_FB_N0_SCALE_FACTOR:g}x away from the observed level "
                f"{n0_seed:.4g} at t=0 — this is a spurious minimum; check the N0 "
                "seed (omit it, or pass SEED_FROM_DATA, to have it seeded from "
                "the data)"
            )
    return ""


def _with_fixed(params: ParameterSet, **fixed_values: float) -> ParameterSet:
    """Return a copy of ``params`` with the named parameters present and fixed.

    Existing entries are overwritten with a fixed parameter at the given value;
    missing entries are added. Used to pin the single-histogram amplitude/phase
    nuisances that the count model requires but that are not free here.
    """
    out = ParameterSet()
    for p in params:
        if p.name in fixed_values:
            out.add(Parameter(name=p.name, value=float(fixed_values[p.name]), fixed=True))
        else:
            out.add(p)
    for name, value in fixed_values.items():
        if name not in out:
            out.add(Parameter(name=name, value=float(value), fixed=True))
    return out
