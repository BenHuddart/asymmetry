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
from asymmetry.core.fitting.engine import FitResult, _minuit_status_message
from asymmetry.core.fitting.grouped_time_domain import (
    GroupedTimeDomainFitResult,
    build_count_group,
    build_count_groups,
    build_fb_count_model,
    build_grouped_count_model,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.transform.grouping import effective_group_indices
from asymmetry.core.utils.constants import MUON_LIFETIME_US

#: Selectable count-fit cost functions.
COUNT_COSTS: tuple[str, ...] = ("poisson", "gaussian")

#: Optional count-loss (deadtime) parameters, applied when present in the fit set.
DEADTIME_PARAMS: tuple[str, ...] = ("DT0", "DT1", "C2", "C3", "C4")

#: Selectable count-loss models (WiMDA ``CountLossModelling``).
DEADTIME_MODELS: tuple[str, ...] = ("simple", "linear", "polynomial", "power")

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


def _split_lifetime(kw: dict, default: float) -> float:
    """Pop the optional free muon lifetime ``tau`` (μs); default = physical value.

    Popping it keeps ``tau`` from reaching the model builders (which would pass it
    to the physics model). When absent this returns ``default`` and leaves ``kw``
    untouched, so the fixed-lifetime path is unchanged.
    """
    return float(kw.pop("tau", default))


# --- count loss (deadtime) --------------------------------------------------


def _pop_deadtime(kw: dict) -> tuple[float, float, float, float, float]:
    """Pop the deadtime parameters ``(DT0, DT1, C2, C3, C4)`` (all default 0)."""
    return (
        float(kw.pop("DT0", 0.0)),
        float(kw.pop("DT1", 0.0)),
        float(kw.pop("C2", 0.0)),
        float(kw.pop("C3", 0.0)),
        float(kw.pop("C4", 0.0)),
    )


def _validate_deadtime_model(model: str) -> str:
    if model not in DEADTIME_MODELS:
        raise ValueError(f"Unknown deadtime model {model!r}; expected one of {DEADTIME_MODELS}")
    return model


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


def _deadtime_event_fraction(dataset: MuonDataset, group_id: int) -> float:
    """Per-group event fraction ``evfr`` for the linear / power-law loss forms.

    WiMDA's ``evfr`` is the fraction of the run's events landing in this group.
    The loaders do not carry an ISIS event-fraction block, so this reads an
    optional ``event_fractions`` (per group id) or scalar ``event_fraction`` from
    the grouping and **falls back to 1.0** (the whole event budget) for
    PSI/synthetic runs that lack it. With ``evfr = 1`` the linear form reduces to
    ``(DT0 + DT1)·qq`` and the power-law base to ``DT0``.
    """
    grouping = (
        dataset.run.grouping if dataset.run and isinstance(dataset.run.grouping, dict) else {}
    )
    fractions = grouping.get("event_fractions")
    if isinstance(fractions, dict):
        try:
            return float(fractions[int(group_id)])
        except (KeyError, TypeError, ValueError):
            pass
    try:
        value = float(grouping.get("event_fraction", 1.0))
    except (TypeError, ValueError):
        return 1.0
    return value if np.isfinite(value) else 1.0


def _apply_deadtime(
    counts: NDArray[np.float64],
    dt_terms: tuple[float, float, float, float, float],
    *,
    frame_norm: float,
    time: NDArray[np.float64],
    evfr: float,
    model: str,
) -> NDArray[np.float64]:
    """Multiply true counts by the non-paralyzable count-loss factor ``1 - L``.

    Transcribes WiMDA ``ArrayMusrFunc`` (``AsymFitFunction.pas``) with
    ``qq = counts / frame_norm`` the per-frame, per-detector rate and ``evfr`` the
    group event fraction. An exact no-op when every coefficient is zero; the
    factor is clipped at 0 so a runaway trial cannot make counts negative.

    - ``"simple"``: ``L = DT0·qq``.
    - ``"linear"``: ``L = (DT0 + DT1·evfr)·qq``.
    - ``"polynomial"``: ``L = DT0·qq + C2·10³·qq² + C3·10⁶·qq³ + C4·10⁹·qq⁴``.
    - ``"power"``: ``L = (evfr·DT0)^C2 · exp(-(C4·λ_μ·t)^C3)`` (rate-independent;
      ``λ_μ = 1/τ_μ``).
    """
    dt0, dt1, c2, c3, c4 = dt_terms
    if dt0 == 0.0 and dt1 == 0.0 and c2 == 0.0 and c3 == 0.0 and c4 == 0.0:
        return counts
    counts = np.asarray(counts, dtype=float)
    if model == "power":
        base = max(float(evfr) * dt0, 0.0)
        decay_arg = np.maximum(c4 * np.asarray(time, dtype=float) / float(MUON_LIFETIME_US), 0.0)
        loss = (base**c2) * np.exp(-np.power(decay_arg, c3))
    else:
        qq = counts / frame_norm
        if model == "simple":
            loss = dt0 * qq
        elif model == "linear":
            loss = (dt0 + dt1 * float(evfr)) * qq
        else:  # polynomial
            loss = dt0 * qq + c2 * 1.0e3 * qq**2 + c3 * 1.0e6 * qq**3 + c4 * 1.0e9 * qq**4
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
        gate = (time > dpsep2).astype(float)
        # The second pulse only contributes for t > dpsep/2. Clamp the gated-out
        # times to >= 0 before evaluating the model, so a model that raises or
        # returns non-finite values for t < 0 cannot poison the (zero-weighted)
        # early bins. Where the gate is active, time - dpsep2 > 0 already, so the
        # clamp is a no-op there.
        a1 = np.asarray(fraction_fn(time + dpsep2, **physics), dtype=float)
        a2 = np.asarray(fraction_fn(np.maximum(time - dpsep2, 0.0), **physics), dtype=float)
        factor = 0.5 * (c1 * (1.0 + amplitude * a1) + gate * c2 * (1.0 + amplitude * a2))
        return float(N0) * np.exp(-time / tau) * factor + float(background)

    return model


def _double_pulse_fb_model(fraction_fn: Callable[..., NDArray]) -> Callable[..., NDArray]:
    """Raw forward/backward count model for an ISIS double-pulse source.

    The ``fgFB`` double-pulse analogue of :func:`_double_pulse_single_model`: the
    two-pulse polarization (evaluated at ``t ± dpsep/2`` and weighted by
    ``exp(∓dpsep/2τ_μ)``, second pulse gated to ``t > dpsep/2``) is carried with
    the forward/backward ``sign``, and the shared ``N0`` is split by the detector
    balance as ``N0·√alpha`` (forward) / ``N0/√alpha`` (backward) exactly as in
    :func:`build_fb_count_model`. The single-pulse limit (``dpsep → 0``) recovers
    that model. Each side keeps its own background.
    """
    tau = float(MUON_LIFETIME_US)

    def model(t, *, alpha, N0, background, sign, dpsep, **physics):  # noqa: N803
        time = np.asarray(t, dtype=float)
        dpsep2 = float(dpsep) / 2.0
        c1 = np.exp(-dpsep2 / tau)
        c2 = np.exp(dpsep2 / tau)
        gate = (time > dpsep2).astype(float)
        # Clamp the gated-out times to >= 0 before evaluating, matching the
        # single-histogram path: a model undefined for t < 0 must not poison the
        # zero-weighted early bins.
        a1 = np.asarray(fraction_fn(time + dpsep2, **physics), dtype=float)
        a2 = np.asarray(fraction_fn(np.maximum(time - dpsep2, 0.0), **physics), dtype=float)
        ralp = np.sqrt(abs(float(alpha)))
        if sign >= 0.0:
            scale = ralp
        else:
            scale = 1.0 / ralp if ralp > 0.0 else 0.0
        factor = 0.5 * (c1 * (1.0 + sign * a1) + gate * c2 * (1.0 + sign * a2))
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
    chi_squared: float | None = None,
    success_message: str = "Count fit successful",
    failure_prefix: str = "Count fit failed",
) -> FitResult:
    """Pack a :class:`FitResult` for one count-domain (or one F/B side).

    ``chi_squared`` overrides the reported cost; pass the per-side cost for a
    joint forward/backward fit (``m.fval`` is the combined cost). Defaults to
    ``m.fval`` (correct for a single-domain fit).
    """
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
    deadtime_model: str = "polynomial",
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
    parameters add a stretched-exponential baseline drift; the deadtime
    parameters (``DT0``, ``DT1``, ``C2``, ``C3``, ``C4``) add a count-loss term
    whose form is set by ``deadtime_model`` (``"simple"`` / ``"linear"`` /
    ``"polynomial"`` / ``"power"`` — see :func:`_apply_deadtime`); a ``dpsep``
    parameter switches on the ISIS double-pulse count model — fixed by default,
    or located by a coarse->fine grid scan when supplied free (the non-smooth
    pulse-onset gate defeats gradient fitting). A ``tau`` parameter frees the muon
    lifetime (musrfit-style; default fixed at the physical value); it is not
    supported together with the double-pulse model.
    """
    _validate_cost(cost)
    _validate_deadtime_model(deadtime_model)
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
                deadtime_model=deadtime_model,
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
    frame_norm = _deadtime_frame_norm(dataset, group_id)
    evfr = _deadtime_event_fraction(dataset, group_id)

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
            model, dt_terms, frame_norm=frame_norm, time=t_eval, evfr=evfr, model=deadtime_model
        )

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
    deadtime_model: str = "polynomial",
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

    The same optional nuisances as :func:`fit_single_histogram` apply, all
    no-ops unless requested: ``exclude`` drops an interior window; a free ``t0``
    shifts the model time axis; free ``lambda_base`` / ``beta_base`` add a
    baseline drift; the deadtime parameters add a count-loss term (form set by
    ``deadtime_model``); a ``dpsep`` parameter switches on the ISIS double-pulse
    model (the √α-tied model evaluated at ``t ± dpsep/2``); a ``tau`` parameter
    frees the shared muon lifetime (reported in ``shared_parameters``; not
    supported with the double-pulse model).
    """
    _validate_cost(cost)
    _validate_deadtime_model(deadtime_model)
    if "tau" in params and "dpsep" in params:
        raise ValueError("Free muon lifetime is not supported with the double-pulse model")
    for required in ("alpha", "N0", "background", "background_b"):
        if required not in params:
            raise ValueError(f"Forward/backward count fit requires a {required!r} parameter")
    if int(forward_group) == int(backward_group):
        raise ValueError("Forward/backward count fit needs two distinct groups")
    # alpha is physically positive and the model is sign-degenerate (sqrt(abs)),
    # so enforce a positive lower bound rather than report a negative balance.
    params = _clamp_alpha_positive(params)

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
                deadtime_model=deadtime_model,
                t_min=t_min,
                t_max=t_max,
                exclude=exclude,
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

    tau = float(MUON_LIFETIME_US)
    base_decay_f = np.exp(-time_f / tau)
    base_decay_b = np.exp(-time_b / tau)
    fraction_fn = _with_baseline_drift(_percent_to_fraction(model_fn))
    double_pulse = "dpsep" in params
    free_lifetime = "tau" in params
    fb_dp_model = _double_pulse_fb_model(fraction_fn) if double_pulse else None
    fb_corrected = None if double_pulse else build_fb_count_model(fraction_fn)

    frame_norm_f = _deadtime_frame_norm(dataset, forward_group)
    frame_norm_b = _deadtime_frame_norm(dataset, backward_group)
    evfr_f = _deadtime_event_fraction(dataset, forward_group)
    evfr_b = _deadtime_event_fraction(dataset, backward_group)

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

    def _eval(args, time, base_decay, counts_key, sign, frame_norm, evfr) -> NDArray[np.float64]:
        kw = {**fixed_kw, **dict(zip(free_names, args, strict=False))}
        for follower, main in followers.items():
            kw[follower] = kw[main]
        t0 = _split_time_offset(kw)
        dt_terms = _pop_deadtime(kw)
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
            model, dt_terms, frame_norm=frame_norm, time=t_eval, evfr=evfr, model=deadtime_model
        )

    def predict_f(args):
        return _eval(args, time_f, base_decay_f, "background", +1.0, frame_norm_f, evfr_f)

    def predict_b(args):
        return _eval(args, time_b, base_decay_b, "background_b", -1.0, frame_norm_b, evfr_b)

    def total_cost(*args) -> float:
        return _cost_value(counts_f, predict_f(args), cost) + _cost_value(
            counts_b, predict_b(args), cost
        )

    m = _solve(free, total_cost)
    fitted = [m.values[i] for i in range(len(free_names))]
    model_f = predict_f(fitted)
    model_b = predict_b(fitted)
    # Each side reports its OWN cost, not the joint m.fval (which is cost_f + cost_b).
    chi2_f = _cost_value(counts_f, model_f, cost)
    chi2_b = _cost_value(counts_b, model_b, cost)

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
        model_values=model_f,
        keep_params=fwd_keep,
        chi_squared=chi2_f,
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
        success_message="Forward/backward count fit successful",
        failure_prefix="Forward/backward count fit failed",
    )

    shared_parameters = ParameterSet()
    reported: set[str] = set()
    # Free shared scale/physics params, plus the free per-eval nuisances that are
    # genuinely shared across both banks (t0, tau, deadtime terms). The two
    # backgrounds are per-side and reported on each group result instead.
    for name in [*shared_keys, *free_names]:
        if name in reported or name in ("background", "background_b"):
            continue
        p = params[name]
        idx = free_names.index(name)
        shared_parameters.add(Parameter(name=name, value=m.values[idx], min=p.min, max=p.max))
        reported.add(name)
    for p in params:
        if p.fixed and p.name not in ("background", "background_b") and p.name not in reported:
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
