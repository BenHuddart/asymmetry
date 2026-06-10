"""Raw-count-domain fitting: single-histogram and forward/backward (alpha-free).

These are WiMDA's ``fgForward``/``fgBackward``/``fgSelected`` (single histogram)
and ``fgFB`` (simultaneous forward+backward with the detector balance ``alpha``
free) modes. Both fit **raw** detector counts to

    N(t) = N0 * exp(-t / tau_mu) * (1 + s * A * P(t)) + bg

with a selectable cost:

- ``"poisson"`` (default) — the Cash statistic C = 2*sum(mu - n + n*ln(n/mu)),
  the correct treatment for the low-count late-time and continuous-source bins
  these modes target;
- ``"gaussian"`` — sqrt(N) least squares, matching WiMDA's weighting, for parity
  and speed.

The model is reused, not rebuilt: the raw count model is the existing
lifetime-corrected builder (:func:`build_grouped_count_model` /
:func:`build_fb_count_model`) multiplied by ``exp(-t / tau_mu)``. This module
owns only the count-space statistics and a small iminuit driver. The multi-group
``fgAll`` path stays on :meth:`FitEngine.global_fit`.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Hashable

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.grouped_time_domain import (
    GroupedTimeDomainFitResult,
    build_count_group,
    build_fb_count_model,
    build_grouped_count_model,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.transform.grouping import effective_group_indices
from asymmetry.core.utils.constants import MUON_LIFETIME_US

#: Selectable count-fit cost functions.
COUNT_COSTS: tuple[str, ...] = ("poisson", "gaussian")

#: Single-histogram nuisance parameters (the rest are physics-model parameters).
SINGLE_HISTOGRAM_NUISANCE: tuple[str, ...] = ("N0", "background")

#: Optional count-loss (deadtime) parameters, applied when present in the fit set.
DEADTIME_PARAMS: tuple[str, ...] = ("DT0", "C2", "C3", "C4")

_INF = float("inf")


# --- cost functions ---------------------------------------------------------


def _gaussian_chi2(counts: NDArray[np.float64], model: NDArray[np.float64]) -> float:
    """sqrt(N) Gaussian least-squares cost (WiMDA weighting)."""
    sigma = np.sqrt(np.clip(counts, 1.0, None))
    return float(np.sum(((counts - model) / sigma) ** 2))


def _poisson_cash(counts: NDArray[np.float64], model: NDArray[np.float64]) -> float:
    """Cash statistic 2*sum(mu - n + n*ln(n/mu)).

    Scaled so that ``errordef = 1`` yields correct parameter errors (Delta C
    behaves like Delta chi-square near the minimum). The ``n = 0`` bins reduce to
    ``2*mu`` with no logarithm.
    """
    mu = np.clip(model, 1.0e-12, None)
    term = mu - counts
    positive = counts > 0.0
    term[positive] += counts[positive] * np.log(counts[positive] / mu[positive])
    return 2.0 * float(np.sum(term))


def _cost_value(counts: NDArray[np.float64], model: NDArray[np.float64], cost: str) -> float:
    if cost == "gaussian":
        return _gaussian_chi2(counts, model)
    return _poisson_cash(counts, model)


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


# --- count loss (deadtime) --------------------------------------------------


def _pop_deadtime(kw: dict) -> tuple[float, float, float, float]:
    """Pop the optional deadtime parameters ``(DT0, C2, C3, C4)`` (all default 0)."""
    return (
        float(kw.pop("DT0", 0.0)),
        float(kw.pop("C2", 0.0)),
        float(kw.pop("C3", 0.0)),
        float(kw.pop("C4", 0.0)),
    )


def _deadtime_frame_norm(dataset: MuonDataset, group_id: int) -> float:
    """Per-frame, per-detector, per-bin count-rate normalization for deadtime.

    Matches the convention of :func:`apply_deadtime_correction`
    (``loss = N*tau/(bin_width*n_frames)`` per detector). The grouped count sums
    ``n_det`` detectors, so the per-detector rate divides the group count by
    ``n_det``. Returns ``n_det * bin_width_us * n_frames``; the loss term is then
    ``DT0 * N_group / frame_norm``.
    """
    grouping = (
        dataset.run.grouping if dataset.run and isinstance(dataset.run.grouping, dict) else {}
    )
    n_det = max(1, len(effective_group_indices(grouping, int(group_id))))
    try:
        n_frames = float(grouping.get("good_frames", 1.0)) or 1.0
    except (TypeError, ValueError):
        n_frames = 1.0
    bin_width = float(dataset.run.histograms[0].bin_width) if dataset.run.histograms else 1.0
    return max(n_det * bin_width * n_frames, 1e-30)


def _apply_deadtime(
    counts: NDArray[np.float64], dt_terms: tuple[float, float, float, float], frame_norm: float
) -> NDArray[np.float64]:
    """Multiply true counts by the non-paralyzable count-loss factor ``1 - L(qq)``.

    With ``qq = counts / frame_norm`` the per-frame, per-detector rate,
    ``L = DT0*qq + C2*qq^2 + C3*qq^3 + C4*qq^4`` (Simple = DT0 only; the higher
    terms are the polynomial extension). An exact no-op when every coefficient is
    zero. The factor is clipped at 0 so a runaway trial cannot make counts
    negative.
    """
    dt0, c2, c3, c4 = dt_terms
    if dt0 == 0.0 and c2 == 0.0 and c3 == 0.0 and c4 == 0.0:
        return counts
    qq = np.asarray(counts, dtype=float) / frame_norm
    loss = dt0 * qq + c2 * qq**2 + c3 * qq**3 + c4 * qq**4
    return counts * np.clip(1.0 - loss, 0.0, None)


# --- double pulse -----------------------------------------------------------


def _double_pulse_single_model(fraction_fn: Callable[..., NDArray]) -> Callable[..., NDArray]:
    """Raw single-histogram count model for an ISIS double-pulse source.

    Two muon pulses separated by ``dpsep`` (μs) each carry the polarization,
    evaluated at ``t ± dpsep/2`` and weighted by ``exp(∓dpsep/2τ_μ)`` (WiMDA
    ``ArrayMusrFunc``). The decay envelope and ``N0``/background stay at ``t``;
    only the polarization is shifted. The 0.5 normalization recovers the
    single-pulse limit as ``dpsep → 0``; the second pulse is gated to
    ``t > dpsep/2``.
    """
    tau = float(MUON_LIFETIME_US)

    def model(t, *, N0, background, amplitude, relative_phase, dpsep, **physics):  # noqa: N803
        time = np.asarray(t, dtype=float)
        dpsep2 = float(dpsep) / 2.0
        c1 = np.exp(-dpsep2 / tau)
        c2 = np.exp(dpsep2 / tau)
        a1 = np.asarray(fraction_fn(time + dpsep2, **physics), dtype=float)
        a2 = np.asarray(fraction_fn(time - dpsep2, **physics), dtype=float)
        gate = (time > dpsep2).astype(float)
        factor = 0.5 * (c1 * (1.0 + amplitude * a1) + gate * c2 * (1.0 + amplitude * a2))
        return float(N0) * np.exp(-time / tau) * factor + float(background)

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


def _solve(free: list[Parameter], total_cost: Callable[..., float]):
    """Minimise ``total_cost`` over the free parameters and return the Minuit object."""
    from iminuit import Minuit

    total_cost.errordef = 1.0  # Cash and sqrt-N chi-square both use errordef = 1
    names = [p.name for p in free]
    initial = [p.value for p in free]
    m = Minuit(total_cost, *initial, name=names)
    for i, p in enumerate(free):
        lo = p.min if p.min != -_INF else m.limits[i][0]
        hi = p.max if p.max != _INF else m.limits[i][1]
        m.limits[i] = (lo, hi)
    m.migrad(iterate=5, use_simplex=True)
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
    success_message: str = "Count fit successful",
    failure_prefix: str = "Count fit failed",
) -> FitResult:
    """Pack a :class:`FitResult` for one count-domain (or one F/B side)."""
    keep = set(keep_params) if keep_params is not None else None
    result_params = ParameterSet()
    uncertainties: dict[str, float] = {}
    for p in params:
        if keep is not None and p.name not in keep:
            continue
        if p.name in free_names:
            idx = free_names.index(p.name)
            result_params.add(Parameter(name=p.name, value=m.values[idx], min=p.min, max=p.max))
            if m.errors[idx] is not None:
                uncertainties[p.name] = float(m.errors[idx])
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
    chi2 = float(m.fval)
    return FitResult(
        success=bool(m.valid),
        chi_squared=chi2,
        reduced_chi_squared=chi2 / max(ndata - nfree, 1),
        parameters=result_params,
        uncertainties=uncertainties,
        covariance=covariance,
        covariance_parameters=covariance_order,
        residuals=residuals,
        message=success_message if m.valid else failure_prefix,
        function_calls=int(getattr(m, "nfcn", 0) or 0),
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
) -> FitResult:
    """Fit one detector group's raw counts (the musrfit fittype-0 analogue).

    The count model is ``N0 * exp(-t/tau) * (1 + s*A*P(t)) + bg`` with ``s`` the
    forward (+1) / backward (-1) sign set by ``side`` (``"selected"`` uses +1).
    ``params`` supplies the nuisances ``N0`` and ``background`` plus the physics
    model parameters; the muon lifetime is held fixed at the physical value.

    Optional window/nuisance flexibility, all no-ops unless requested:
    ``exclude`` drops an interior time window from the fit; a free ``t0``
    parameter shifts the model time axis; free ``lambda_base`` / ``beta_base``
    parameters add a stretched-exponential baseline drift; ``DT0`` (+ ``C2``,
    ``C3``, ``C4``) add a count-loss/deadtime term; a ``dpsep`` parameter
    switches on the ISIS double-pulse count model.
    """
    _validate_cost(cost)
    group = build_count_group(
        dataset, group_id, t_min=t_min, t_max=t_max, lifetime_corrected=False, exclude=exclude
    )
    time = np.asarray(group.time, dtype=float)
    counts = np.asarray(group.counts, dtype=float)

    fraction_fn = _with_baseline_drift(_percent_to_fraction(model_fn))
    if "dpsep" in params:
        base_model = _double_pulse_single_model(fraction_fn)
    else:
        base_model = _raw_model(build_grouped_count_model(fraction_fn))
    frame_norm = _deadtime_frame_norm(dataset, group_id)

    # The per-group amplitude/phase are fixed nuisances here: the sign carries the
    # forward/backward orientation and the physics model owns the real amplitude.
    params = _with_fixed(
        params,
        amplitude=_single_histogram_amplitude(side),
        relative_phase=0.0,
    )
    free = params.free_parameters
    free_names = [p.name for p in free]
    fixed_kw = {p.name: p.value for p in params if p.fixed}
    followers = params.link_followers()

    def predict(args) -> NDArray[np.float64]:
        kw = {**fixed_kw, **dict(zip(free_names, args, strict=False))}
        for follower, main in followers.items():
            kw[follower] = kw[main]
        t0 = _split_time_offset(kw)
        dt_terms = _pop_deadtime(kw)
        model = np.asarray(base_model(time + t0, **kw), dtype=float)
        return _apply_deadtime(model, dt_terms, frame_norm)

    def total_cost(*args) -> float:
        return _cost_value(counts, predict(args), cost)

    m = _solve(free, total_cost)
    model_values = predict([m.values[i] for i in range(len(free_names))])
    return _result_from_minuit(
        m,
        params,
        free_names,
        time=time,
        counts=counts,
        model_values=model_values,
        success_message="Single-histogram count fit successful",
        failure_prefix="Single-histogram count fit failed",
    )


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
) -> GroupedTimeDomainFitResult:
    """Simultaneously fit forward and backward counts with the balance alpha free.

    Recovers the detector balance ``alpha`` from a transverse-field calibration
    run the statistically proper way (WiMDA ``fgFB``), reporting its correlation
    with the physics amplitude — strong in TF runs. ``params`` must contain the
    shared ``alpha`` and ``N0``, a forward background ``background`` and a
    backward background ``background_b``, plus the physics model parameters.

    Returns a :class:`GroupedTimeDomainFitResult` whose ``group_results`` are
    keyed by ``forward_group`` / ``backward_group`` and whose
    ``shared_parameters`` hold the fitted ``alpha``/``N0``/physics. The (alpha,
    amplitude) covariance is available on either group result.
    """
    _validate_cost(cost)
    for required in ("alpha", "N0", "background", "background_b"):
        if required not in params:
            raise ValueError(f"Forward/backward count fit requires a {required!r} parameter")

    g_fwd = build_count_group(
        dataset, forward_group, t_min=t_min, t_max=t_max, lifetime_corrected=False, exclude=exclude
    )
    g_bwd = build_count_group(
        dataset, backward_group, t_min=t_min, t_max=t_max, lifetime_corrected=False, exclude=exclude
    )
    time_f = np.asarray(g_fwd.time, dtype=float)
    counts_f = np.asarray(g_fwd.counts, dtype=float)
    time_b = np.asarray(g_bwd.time, dtype=float)
    counts_b = np.asarray(g_bwd.counts, dtype=float)

    raw_model = _raw_model(
        build_fb_count_model(_with_baseline_drift(_percent_to_fraction(model_fn)))
    )

    frame_norm_f = _deadtime_frame_norm(dataset, forward_group)
    frame_norm_b = _deadtime_frame_norm(dataset, backward_group)

    free = params.free_parameters
    free_names = [p.name for p in free]
    fixed_kw = {p.name: p.value for p in params if p.fixed}
    followers = params.link_followers()
    # Shared physics/scale params: everything except the two per-side backgrounds,
    # the time offset (applied by shifting the model time axis) and the deadtime
    # terms (applied as a post-multiply loss factor).
    _per_eval = ("background", "background_b", "t0", *DEADTIME_PARAMS)
    shared_keys = [n for n in free_names if n not in _per_eval]

    def _eval(args, time, counts_key: str, sign: float, frame_norm: float) -> NDArray[np.float64]:
        kw = {**fixed_kw, **dict(zip(free_names, args, strict=False))}
        for follower, main in followers.items():
            kw[follower] = kw[main]
        t0 = _split_time_offset(kw)
        dt_terms = _pop_deadtime(kw)
        shared = {k: kw[k] for k in shared_keys if k in kw}
        shared.update({k: v for k, v in fixed_kw.items() if k not in _per_eval})
        model = np.asarray(
            raw_model(time + t0, sign=sign, background=kw[counts_key], **shared), dtype=float
        )
        return _apply_deadtime(model, dt_terms, frame_norm)

    def predict_f(args):
        return _eval(args, time_f, "background", +1.0, frame_norm_f)

    def predict_b(args):
        return _eval(args, time_b, "background_b", -1.0, frame_norm_b)

    def total_cost(*args) -> float:
        return _cost_value(counts_f, predict_f(args), cost) + _cost_value(
            counts_b, predict_b(args), cost
        )

    m = _solve(free, total_cost)
    fitted = [m.values[i] for i in range(len(free_names))]

    # Forward result keeps the shared params + the forward background; backward
    # keeps the shared params + its own background. alpha (shared) appears in both.
    fwd_keep = shared_keys + ["background"]
    bwd_keep = shared_keys + ["background_b"]
    result_f = _result_from_minuit(
        m,
        params,
        free_names,
        time=time_f,
        counts=counts_f,
        model_values=predict_f(fitted),
        keep_params=fwd_keep,
        success_message="Forward/backward count fit successful",
        failure_prefix="Forward/backward count fit failed",
    )
    result_b = _result_from_minuit(
        m,
        params,
        free_names,
        time=time_b,
        counts=counts_b,
        model_values=predict_b(fitted),
        keep_params=bwd_keep,
        success_message="Forward/backward count fit successful",
        failure_prefix="Forward/backward count fit failed",
    )

    shared_parameters = ParameterSet()
    for name in shared_keys:
        p = params[name]
        idx = free_names.index(name)
        shared_parameters.add(Parameter(name=name, value=m.values[idx], min=p.min, max=p.max))
    for p in params:
        if p.fixed and p.name not in ("background", "background_b"):
            shared_parameters.add(Parameter(name=p.name, value=p.value, fixed=True))

    group_results: dict[Hashable, FitResult] = {
        int(forward_group): result_f,
        int(backward_group): result_b,
    }
    success = bool(m.valid)
    return GroupedTimeDomainFitResult(
        success=success,
        group_results=group_results,
        shared_parameters=shared_parameters,
        message="Forward/backward count fit successful"
        if success
        else "Forward/backward count fit failed",
    )


# --- helpers ----------------------------------------------------------------


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
