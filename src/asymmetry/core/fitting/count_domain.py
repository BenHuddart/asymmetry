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
from asymmetry.core.fitting.count_calibration import record_count_deadtime_calibration
from asymmetry.core.fitting.count_inputs import count_model_frame_norm as _deadtime_frame_norm
from asymmetry.core.fitting.count_results import (
    _result_from_minuit,
    count_bank_view,
    count_fit_domain,
    joint_count_fit,
)
from asymmetry.core.fitting.engine import (
    FitCancelledError,
    FitResult,
    _make_cancel_guard,
    _reject_affine_ties,
    drive_minuit,
)
from asymmetry.core.fitting.grouped_time_domain import (
    GroupedTimeDomainFitResult,
    build_count_group,
    build_count_groups,
    build_fb_count_model,
    build_grouped_count_model,
    group_observations,
)
from asymmetry.core.fitting.likelihood import (
    GaussianObservations,
    Objective,
    PoissonCounts,
    gaussian_chi2,
    poisson_deviance,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.maxent.pulse import PulseRequestError
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


def _cost_value(observations, model: NDArray[np.float64]) -> float:
    if not np.all(np.isfinite(model)):
        return _INF
    if isinstance(observations, GaussianObservations):
        return gaussian_chi2(observations.values, model, observations.sigma)
    return poisson_deviance(observations.counts, model)


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
    explicit_hesse: bool = False,
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
    drive_minuit(
        m,
        migrad_kwargs={"iterate": 5, "use_simplex": True},
        minos=minos,
        explicit_hesse=explicit_hesse,
    )
    return m


def _validate_fixed_separation(params: ParameterSet) -> None:
    """Double-delta counts consume a supplied separation, never a source width."""
    if "dpsep" not in params:
        return
    separation = params["dpsep"]
    if not separation.fixed:
        raise PulseRequestError(
            "dpsep must be fixed at a supplied separation in µs; separation estimation is unsupported."
        )
    if not np.isfinite(separation.value) or separation.value < 0.0:
        raise PulseRequestError(
            "dpsep must be nonnegative and finite in µs; supply a fixed separation."
        )


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
    parameter selects the double-delta count model with fixed supplied separation. A ``tau`` parameter frees the muon
    lifetime (musrfit-style; default fixed at the physical value); it is not
    supported together with the double-pulse model.
    """
    _validate_cost(cost)
    _guard_model_param_collisions(model_fn)
    if "tau" in params and "dpsep" in params:
        raise ValueError("Free muon lifetime is not supported with the double-pulse model")
    _validate_fixed_separation(params)
    group = build_count_group(
        dataset, group_id, t_min=t_min, t_max=t_max, lifetime_corrected=False, exclude=exclude
    )
    observations = group_observations(group, Objective(cost))
    time = observations.time

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
        return _cost_value(observations, predict(args))

    m = _solve(free, total_cost, minos=minos, cancel_callback=cancel_callback)
    model_values = predict([m.values[i] for i in range(len(free_names))])
    result = _result_from_minuit(
        m,
        params,
        free_names,
        observations=observations,
        model_values=model_values,
        followers=followers,
        success_message="Single-histogram count fit successful",
        failure_prefix="Single-histogram count fit failed",
    )
    record_count_deadtime_calibration(result, dataset, [group])
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
    """Prepare and jointly fit F/B counts with shared physics and balance.

    Physics amplitudes are percent; scales/backgrounds are counts per packed bin.
    ``estimate_beta`` permits a shared backward amplitude ratio, including fixed
    beta constraints. Otherwise beta is absent. Gaussian cost uses retained count
    errors; Poisson cost requires uncorrected integer acquisition observations.
    Optional t0, lifetime, deadtime and pulse nuisances use the same joint solver.
    """
    groups = build_count_groups(
        dataset,
        [forward_group, backward_group],
        t_min=t_min,
        t_max=t_max,
        lifetime_corrected=False,
        exclude=exclude,
    )
    return _fit_fb_count_groups(
        dataset,
        groups,
        model_fn,
        params,
        cost=cost,
        estimate_beta=estimate_beta,
        minos=minos,
        cancel_callback=cancel_callback,
    )


def _fit_fb_count_groups(
    dataset,
    groups,
    model_fn,
    params,
    *,
    cost="poisson",
    estimate_beta=False,
    minos=False,
    cancel_callback=None,
) -> GroupedTimeDomainFitResult:
    """Solve prepared F/B banks once, retaining the full nuisance covariance."""
    g_fwd, g_bwd = groups
    forward_group, backward_group = g_fwd.group_id, g_bwd.group_id
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

    _validate_fixed_separation(params)

    time_f = np.asarray(g_fwd.time, dtype=float)
    counts_f = np.asarray(g_fwd.counts, dtype=float)
    time_b = np.asarray(g_bwd.time, dtype=float)
    counts_b = np.asarray(g_bwd.counts, dtype=float)
    observations_f = group_observations(g_fwd, Objective(cost))
    observations_b = group_observations(g_bwd, Objective(cost))
    time_f, time_b = observations_f.time, observations_b.time

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
        return _cost_value(observations_f, predict_f(args)) + _cost_value(
            observations_b, predict_b(args)
        )

    m = _solve(free, total_cost, minos=minos, explicit_hesse=True, cancel_callback=cancel_callback)
    fitted = [m.values[i] for i in range(len(free_names))]
    model_f = predict_f(fitted)
    model_b = predict_b(fitted)
    chi2_f = _cost_value(observations_f, model_f)
    chi2_b = _cost_value(observations_b, model_b)

    if isinstance(observations_f, PoissonCounts):
        observations = PoissonCounts(
            np.concatenate((time_f, time_b)),
            np.concatenate((counts_f, counts_b)),
            observations_f.source,
            observations_f.run_number,
        )
    else:
        observations = GaussianObservations(
            np.concatenate((time_f, time_b)),
            np.concatenate((counts_f, counts_b)),
            np.concatenate((observations_f.sigma, observations_b.sigma)),
            observations_f.run_number,
        )
    joint = _result_from_minuit(
        m,
        params,
        free_names,
        observations=observations,
        model_values=np.concatenate((model_f, model_b)),
        chi_squared=chi2_f + chi2_b,
        followers=followers,
        success_message="Forward/backward count fit successful",
        failure_prefix="Forward/backward count fit failed",
    )
    result_f = count_bank_view(
        joint, observations_f, model_f, omit="background_b", statistic=chi2_f
    )
    result_b = count_bank_view(joint, observations_b, model_b, omit="background", statistic=chi2_b)

    minima = (result_f.minimum_expected_count, result_b.minimum_expected_count)
    minimum = min(minima) if all(value is not None for value in minima) else None
    for result in (result_f, result_b):
        result.minimum_expected_count = minimum
        record_count_deadtime_calibration(result, dataset, [g_fwd, g_bwd])

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
    success = joint.success
    message = joint.message
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
    joint.success, joint.message = success, message
    if cancel_callback is not None and cancel_callback():
        raise FitCancelledError("Fit cancelled.")
    return GroupedTimeDomainFitResult(
        joint_fit=joint_count_fit(
            joint, tuple(count_fit_domain(g, dataset) for g in (g_fwd, g_bwd))
        ),
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
