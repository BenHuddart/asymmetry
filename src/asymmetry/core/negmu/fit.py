"""EXPERIMENTAL — WORK IN PROGRESS. Negative-muon (μ⁻) capture-lifetime analysis.

This API is UNVALIDATED against real μ⁻ elemental-analysis data. No μ⁻ corpus
exists in this project; every result here has been exercised only against
synthetic histograms. The element lifetime values are literature-anchored
(Suzuki, Measday & Roalsvig, Phys. Rev. C 35, 2212 (1987), via Blundell et al.,
Muon Spectroscopy: An Introduction, OUP 2022, Table C.1), but the fitting,
capture-ratio, and background machinery have NOT been checked against an
established tool (WiMDA, Mantid) on measured data. The API, parameter names, and
return shapes MAY CHANGE without notice. Do not rely on results for publication
without independent verification. This feature is deliberately NOT exposed in the
GUI fit builders. Promotion trigger for a GUI: real ISIS μ⁻ data AND a user.

Fit a μ⁻ capture-lifetime histogram (raw counts) to the multi-exponential model.
Reuses the shared minimiser drive (drive_minuit), FitResult, and
Parameter/ParameterSet. The Poisson (Cash) and Gaussian (√N) count costs are
replicated locally (the fitting engine's helpers are private API, off-limits).
Lifetimes are FIXED at the table value by default; free any via spec.free_tau.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.fitting.engine import (
    FitResult,
    _make_cancel_guard,
    _minuit_status_message,
    drive_minuit,
)
from asymmetry.core.fitting.grouped_time_domain import (
    GroupedTimeDomainFitResult,
    build_count_group,
    build_count_groups,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.negmu.lifetimes import DECAY_BACKGROUND_LABEL, tau_us
from asymmetry.core.negmu.model import CaptureComponent, build_capture_count_model
from asymmetry.core.utils.constants import MUON_LIFETIME_US

COUNT_COSTS: tuple[str, ...] = ("poisson", "gaussian")

_INF = float("inf")


# ---------------------------------------------------------------------------
# Cost functions (replicated locally; the fitting engine's helpers are private API)
# ---------------------------------------------------------------------------


# Cash statistic: 2 Σ (μ − n + n ln(n/μ)), skipping n=0 log terms
def _poisson_cash(counts: NDArray[np.float64], model: NDArray[np.float64]) -> float:
    mu = np.clip(model, 1.0e-12, None)
    term = mu - counts
    pos = counts > 0.0
    term[pos] += counts[pos] * np.log(counts[pos] / mu[pos])
    return 2.0 * float(np.sum(term))


# Gaussian chi-squared: Σ (n − μ)² / max(σ², ε)
def _gaussian_chi2(
    counts: NDArray[np.float64],
    model: NDArray[np.float64],
    variance: NDArray[np.float64] | None = None,
) -> float:
    sigma2 = variance if variance is not None else np.clip(counts, 1.0, None)
    return float(np.sum((counts - model) ** 2 / np.clip(sigma2, 1e-12, None)))


# ---------------------------------------------------------------------------
# Model spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaptureModelSpec:
    """Which elemental components are in the fit, with τ fixed by default."""

    elements: tuple[str, ...]
    include_decay_background: bool = True
    free_tau: frozenset[str] = field(default_factory=frozenset)

    def components(self) -> tuple[CaptureComponent, ...]:
        """Element components (τ from table) + optional decayBG (τ_μ)."""
        comps: list[CaptureComponent] = [
            CaptureComponent(label=sym, tau_us=tau_us(sym)) for sym in self.elements
        ]
        if self.include_decay_background:
            comps.append(CaptureComponent(label=DECAY_BACKGROUND_LABEL, tau_us=MUON_LIFETIME_US))
        return tuple(comps)

    def labels(self) -> tuple[str, ...]:
        """Component labels in component order."""
        return tuple(c.label for c in self.components())


# ---------------------------------------------------------------------------
# Default parameter seeding
# ---------------------------------------------------------------------------


def default_capture_parameters(
    spec: CaptureModelSpec,
    *,
    time: NDArray[np.float64],
    counts: NDArray[np.float64],
    seeds: Mapping[str, float] | None = None,
) -> ParameterSet:
    """Seeded ParameterSet for a capture fit.

    ``amp_<label>`` (free, ≥0) — derived from the signal window integral using
    the exact telescoping normalisation so that seeds scale correctly with τ:
    amp_seed_i = (signal / n_comp) · (1 − exp(−Δt/τ_i)) / (1 − exp(−T/τ_i)).
    ``tau_<label>`` (fixed unless label in spec.free_tau; freed τ bounded ±50 %).
    ``background`` (free, ≥0) — seeded from the late-time floor.
    ``seeds`` overrides any amp_/tau_/background value before Parameter creation.
    """
    comps = spec.components()
    n_comp = len(comps)
    override = dict(seeds) if seeds else {}

    n_bins = len(counts)
    bin_width = float(time[1] - time[0]) if n_bins > 1 else 0.016
    bin_width = max(bin_width, 1e-9)
    t_window = float(n_bins) * bin_width

    n_late = max(1, n_bins // 10)
    bg_seed = max(0.0, float(np.mean(counts[-n_late:])))
    signal_total = max(1.0, float(np.sum(counts)) - bg_seed * n_bins)
    signal_per_comp = signal_total / max(n_comp, 1)

    params = ParameterSet()
    for comp in comps:
        tau_i = max(float(comp.tau_us), 1e-9)
        exp_dt = float(np.exp(-bin_width / tau_i))
        exp_T = float(np.exp(-t_window / tau_i))
        # amp ≈ signal_per_comp * (1-exp(-Δt/τ)) / (1-exp(-T/τ))  (exact telescoping)
        denom = max(1.0 - exp_T, 1e-12)
        amp_seed = max(1.0, signal_per_comp * (1.0 - exp_dt) / denom)

        amp_name = f"amp_{comp.label}"
        amp_val = float(override.get(amp_name, amp_seed))
        params.add(Parameter(name=amp_name, value=amp_val, min=0.0))

        tau_name = f"tau_{comp.label}"
        tau_seed_val = float(comp.tau_us)
        if comp.label in spec.free_tau:
            tau_lo = max(1e-4, tau_seed_val * 0.5)
            tau_hi = tau_seed_val * 2.0
            tau_val = float(override.get(tau_name, tau_seed_val))
            params.add(Parameter(name=tau_name, value=tau_val, min=tau_lo, max=tau_hi))
        else:
            tau_val = float(override.get(tau_name, tau_seed_val))
            params.add(Parameter(name=tau_name, value=tau_val, fixed=True))

    bg_val = float(override.get("background", bg_seed))
    params.add(Parameter(name="background", value=bg_val, min=0.0))

    return params


# ---------------------------------------------------------------------------
# Array-level fitter (no dataset dependency — testable without a loaded run)
# ---------------------------------------------------------------------------


def fit_capture_histogram(
    time: NDArray[np.float64],
    counts: NDArray[np.float64],
    spec: CaptureModelSpec,
    *,
    variance: NDArray[np.float64] | None = None,
    cost: str = "poisson",
    parameters: ParameterSet | None = None,
    minos: bool = False,
    cancel_callback: Callable[[], bool] | None = None,
) -> FitResult:
    """Fit raw counts to Σ_i amp_i·exp(−t/τ_i)+bg.

    The array-level entry: testable without a dataset. ``variance`` (if given)
    is used by the Gaussian cost; Poisson (Cash) ignores it. Returns a
    :class:`FitResult` with ``dof`` set so
    :func:`~asymmetry.core.fitting.fit_quality.assess_fit_quality` applies.
    """
    if cost not in COUNT_COSTS:
        raise ValueError(f"Unknown cost {cost!r}; expected one of {COUNT_COSTS}")

    from iminuit import Minuit

    time_arr = np.asarray(time, dtype=np.float64)
    counts_arr = np.asarray(counts, dtype=np.float64)

    params = (
        parameters
        if parameters is not None
        else default_capture_parameters(spec, time=time_arr, counts=counts_arr)
    )

    model_fn = build_capture_count_model(spec.components())
    free = params.free_parameters
    free_names = [p.name for p in free]
    fixed_vals = {p.name: p.value for p in params if p.name not in set(free_names)}

    guard = _make_cancel_guard(cancel_callback)

    def total_cost(*args: float) -> float:
        guard()
        kw = dict(fixed_vals)
        kw.update(zip(free_names, args))
        model_vals = model_fn(time_arr, **kw)
        if cost == "gaussian":
            return _gaussian_chi2(counts_arr, model_vals, variance)
        return _poisson_cash(counts_arr, model_vals)

    total_cost.errordef = 1.0  # Cash and sqrt-N chi-square both use errordef = 1

    initial = [p.value for p in free]
    m = Minuit(total_cost, *initial, name=free_names)
    for i, p in enumerate(free):
        lo = None if not np.isfinite(p.min) else float(p.min)
        hi = None if not np.isfinite(p.max) else float(p.max)
        m.limits[i] = (lo, hi)

    # Capture fits have degenerate directions (components with similar τ).
    # Run simplex first to escape the basin-of-seeding, then refine with
    # strategy=2 (smaller gradient steps) for accurate Hessian in flat valleys.
    m.strategy = 2
    m.simplex(ncall=5000)
    drive_minuit(m, migrad_kwargs={"iterate": 5, "use_simplex": True}, minos=minos)

    # Reconstruct fitted parameter values
    fitted_kw = dict(fixed_vals)
    fitted_kw.update({name: float(m.values[i]) for i, name in enumerate(free_names)})
    model_values = model_fn(time_arr, **fitted_kw)

    # Pack ParameterSet with post-fit values
    result_params = ParameterSet()
    uncertainties: dict[str, float] = {}
    minos_errors: dict[str, tuple[float, float]] = {}
    merrors = getattr(m, "merrors", None)

    for p in params:
        if p.name in free_names:
            idx = free_names.index(p.name)
            result_params.add(
                Parameter(name=p.name, value=float(m.values[idx]), min=p.min, max=p.max)
            )
            err = float(m.errors[idx]) if m.errors[idx] is not None else 0.0
            if err > 0.0:
                uncertainties[p.name] = err
            if merrors is not None:
                try:
                    me = merrors[p.name]
                    if me is not None and getattr(me, "is_valid", False):
                        minos_errors[p.name] = (float(me.lower), float(me.upper))
                except (KeyError, TypeError):
                    pass
        else:
            result_params.add(Parameter(name=p.name, value=p.value, fixed=True))

    covariance = None
    cov_params: list[str] = []
    if m.valid and getattr(m, "covariance", None) is not None:
        covariance = np.asarray(m.covariance, dtype=float)
        cov_params = list(free_names)

    n_data = len(time_arr)
    n_free = len(free_names)
    dof = max(n_data - n_free, 1)
    chi2 = float(m.fval)
    residuals = counts_arr - model_values

    return FitResult(
        success=bool(m.valid),
        chi_squared=chi2,
        reduced_chi_squared=chi2 / dof,
        parameters=result_params,
        uncertainties=uncertainties,
        covariance=covariance,
        covariance_parameters=cov_params,
        residuals=residuals,
        message=_minuit_status_message(
            m,
            success_message="Capture fit successful",
            failure_prefix="Capture fit failed",
        ),
        function_calls=int(getattr(m, "nfcn", 0) or 0),
        dof=dof,
        minos_errors=minos_errors or None,
        covariance_accurate=getattr(m, "accurate", False),
    )


# ---------------------------------------------------------------------------
# Dataset-level entry (wraps build_count_group)
# ---------------------------------------------------------------------------


def fit_capture_group(
    dataset,
    group_id: int,
    spec: CaptureModelSpec,
    *,
    t_min: float | None = None,
    t_max: float | None = None,
    cost: str = "poisson",
    parameters: ParameterSet | None = None,
    exclude: tuple[float, float] | None = None,
    minos: bool = False,
    cancel_callback: Callable[[], bool] | None = None,
) -> FitResult:
    """Build the raw (time, counts) trace for ``group_id`` and fit.

    Uses :func:`~asymmetry.core.fitting.grouped_time_domain.build_count_group`
    with ``lifetime_corrected=False`` to obtain raw counts on which the Poisson
    cost is exact, then delegates to :func:`fit_capture_histogram`.
    ``parameters`` is forwarded verbatim to allow chain-seeding across fits.
    """
    group = build_count_group(
        dataset,
        group_id,
        t_min=t_min,
        t_max=t_max,
        lifetime_corrected=False,
        exclude=exclude,
    )
    return fit_capture_histogram(
        group.time,
        group.counts,
        spec,
        cost=cost,
        parameters=parameters,
        minos=minos,
        cancel_callback=cancel_callback,
    )


# ---------------------------------------------------------------------------
# Forward/backward simultaneous fit with shared amplitudes and free α (WP2.1)
# ---------------------------------------------------------------------------

# Parameters not passed to the shared model function (handled outside model_fn).
_FB_PER_EVAL: frozenset[str] = frozenset({"alpha", "bg_F", "bg_B"})


def fit_capture_fb_alpha(
    dataset,
    forward_group: int,
    backward_group: int,
    spec: CaptureModelSpec,
    *,
    cost: str = "poisson",
    t_min: float | None = None,
    t_max: float | None = None,
    exclude: tuple[float, float] | None = None,
    alpha_seed: float = 1.0,
    minos: bool = False,
    cancel_callback: Callable[[], bool] | None = None,
) -> GroupedTimeDomainFitResult:
    """Simultaneously fit forward and backward capture histograms with shared
    per-element amplitudes amp_<label>, shared τ_i, and a free detector balance α:

        N_F(t) = √α · Σ_i amp_i·exp(−t/τ_i) + bg_F
        N_B(t) = (1/√α) · Σ_i amp_i·exp(−t/τ_i) + bg_B

    mirroring ``build_count_groups`` geometry and the √α detector-balance split.
    Both banks built in ONE context (``build_count_groups``) for a common t0.
    Returns a :class:`GroupedTimeDomainFitResult`: ``group_results`` keyed by
    ``forward_group`` / ``backward_group``; ``shared_parameters`` holds α,
    amplitudes, and τ.

    DIVERGENCE from WiMDA: WiMDA fits independent per-side amplitudes (NF, NB);
    this shares ``amp_i`` (isotropic capture populations), so per-side capture
    ratios are identical by construction. Use :func:`fit_capture_group` on each
    side independently when a genuine F/B amplitude difference is wanted.
    """
    if cost not in COUNT_COSTS:
        raise ValueError(f"Unknown cost {cost!r}; expected one of {COUNT_COSTS}")
    if int(forward_group) == int(backward_group):
        raise ValueError("Forward/backward capture fit needs two distinct groups")

    from iminuit import Minuit

    # Build both banks in ONE context so they share a common t0 alignment.
    g_fwd, g_bwd = build_count_groups(
        dataset,
        [forward_group, backward_group],
        t_min=t_min,
        t_max=t_max,
        lifetime_corrected=False,
        exclude=exclude,
    )
    time_f = np.asarray(g_fwd.time, dtype=np.float64)
    counts_f = np.asarray(g_fwd.counts, dtype=np.float64)
    time_b = np.asarray(g_bwd.time, dtype=np.float64)
    counts_b = np.asarray(g_bwd.counts, dtype=np.float64)

    # Seed shared amp_*/tau_* from the forward group (representative of shared physics).
    seed_params = default_capture_parameters(spec, time=time_f, counts=counts_f)

    # Build the combined ParameterSet:
    #   alpha (shared, free, floor 1e-6)
    #   amp_*/tau_* from seeding (shared; skip 'background' — replaced by bg_F/bg_B)
    #   bg_F / bg_B (per-side flat backgrounds)
    params = ParameterSet()
    params.add(Parameter(name="alpha", value=max(1e-6, abs(float(alpha_seed))), min=1e-6))
    for p in seed_params:
        if p.name != "background":
            params.add(p)
    n_late_f = max(1, len(counts_f) // 10)
    bg_f_seed = max(0.0, float(np.mean(counts_f[-n_late_f:])))
    params.add(Parameter(name="bg_F", value=bg_f_seed, min=0.0))
    n_late_b = max(1, len(counts_b) // 10)
    bg_b_seed = max(0.0, float(np.mean(counts_b[-n_late_b:])))
    params.add(Parameter(name="bg_B", value=bg_b_seed, min=0.0))

    model_fn = build_capture_count_model(spec.components())

    free = params.free_parameters
    free_names = [p.name for p in free]
    fixed_vals = {p.name: p.value for p in params if p.name not in set(free_names)}

    guard = _make_cancel_guard(cancel_callback)

    def total_cost(*args: float) -> float:
        guard()
        kw = dict(fixed_vals)
        kw.update(zip(free_names, args))
        alpha = max(1e-6, float(kw["alpha"]))
        sqrt_a = float(np.sqrt(alpha))
        bg_f = float(kw["bg_F"])
        bg_b = float(kw["bg_B"])
        # Shared model sum excludes bg_F/bg_B/alpha (model_fn defaults background=0.0).
        model_kw = {k: v for k, v in kw.items() if k not in _FB_PER_EVAL}
        model_f = sqrt_a * model_fn(time_f, **model_kw) + bg_f
        model_b = (1.0 / sqrt_a) * model_fn(time_b, **model_kw) + bg_b
        if cost == "gaussian":
            return _gaussian_chi2(counts_f, model_f) + _gaussian_chi2(counts_b, model_b)
        return _poisson_cash(counts_f, model_f) + _poisson_cash(counts_b, model_b)

    total_cost.errordef = 1.0

    initial = [p.value for p in free]
    m = Minuit(total_cost, *initial, name=free_names)
    for i, p in enumerate(free):
        lo = None if not np.isfinite(p.min) else float(p.min)
        hi = None if not np.isfinite(p.max) else float(p.max)
        m.limits[i] = (lo, hi)

    m.strategy = 2
    m.simplex(ncall=5000)
    drive_minuit(m, migrad_kwargs={"iterate": 5, "use_simplex": True}, minos=minos)

    # Reconstruct fitted parameter values for model evaluation.
    fitted_kw = dict(fixed_vals)
    fitted_kw.update({name: float(m.values[i]) for i, name in enumerate(free_names)})

    alpha_fit = max(1e-6, float(fitted_kw["alpha"]))
    sqrt_a_fit = float(np.sqrt(alpha_fit))
    bg_f_fit = float(fitted_kw.get("bg_F", 0.0))
    bg_b_fit = float(fitted_kw.get("bg_B", 0.0))
    model_kw_fit = {k: v for k, v in fitted_kw.items() if k not in _FB_PER_EVAL}
    model_f_fit = sqrt_a_fit * model_fn(time_f, **model_kw_fit) + bg_f_fit
    model_b_fit = (1.0 / sqrt_a_fit) * model_fn(time_b, **model_kw_fit) + bg_b_fit
    residuals_f = counts_f - model_f_fit
    residuals_b = counts_b - model_b_fit

    # Pack result ParameterSet from iminuit values.
    result_params = ParameterSet()
    uncertainties: dict[str, float] = {}
    minos_errors: dict[str, tuple[float, float]] = {}
    merrors = getattr(m, "merrors", None)

    for p in params:
        if p.name in set(free_names):
            idx = free_names.index(p.name)
            result_params.add(
                Parameter(name=p.name, value=float(m.values[idx]), min=p.min, max=p.max)
            )
            err = float(m.errors[idx]) if m.errors[idx] is not None else 0.0
            if err > 0.0:
                uncertainties[p.name] = err
            if merrors is not None:
                try:
                    me = merrors[p.name]
                    if me is not None and getattr(me, "is_valid", False):
                        minos_errors[p.name] = (float(me.lower), float(me.upper))
                except (KeyError, TypeError):
                    pass
        else:
            result_params.add(Parameter(name=p.name, value=p.value, fixed=True))

    covariance: np.ndarray | None = None
    cov_params: list[str] = []
    if m.valid and getattr(m, "covariance", None) is not None:
        covariance = np.asarray(m.covariance, dtype=float)
        cov_params = list(free_names)

    n_free_params = len(free_names)
    msg = _minuit_status_message(
        m,
        success_message="Capture F+B fit successful",
        failure_prefix="Capture F+B fit failed",
    )

    # Per-side chi-squared from the evaluated fitted model.
    if cost == "gaussian":
        chi2_f = _gaussian_chi2(counts_f, model_f_fit)
        chi2_b = _gaussian_chi2(counts_b, model_b_fit)
    else:
        chi2_f = _poisson_cash(counts_f, model_f_fit)
        chi2_b = _poisson_cash(counts_b, model_b_fit)

    # DOF: both sides share all free parameters (simultaneous fit).
    dof_f = max(len(time_f) - n_free_params, 1)
    dof_b = max(len(time_b) - n_free_params, 1)
    n_calls = int(getattr(m, "nfcn", 0) or 0)
    cov_accurate = getattr(m, "accurate", False)
    is_valid = bool(m.valid)
    minos_out = minos_errors or None

    # Build independent per-group ParameterSets: shared params + own bg only.
    # This prevents result_f and result_b from aliasing the same ParameterSet
    # (forward result should not expose bg_B; backward should not expose bg_F).
    def _side_params(exclude_bg: str) -> ParameterSet:
        ps = ParameterSet()
        for p in result_params:
            if p.name != exclude_bg:
                ps.add(p)
        return ps

    params_f = _side_params("bg_B")
    params_b = _side_params("bg_F")
    unc_f = {k: v for k, v in uncertainties.items() if k != "bg_B"}
    unc_b = {k: v for k, v in uncertainties.items() if k != "bg_F"}

    def _pack(
        chi2: float,
        dof_g: int,
        resid: np.ndarray,
        side_params: ParameterSet,
        side_unc: dict[str, float],
    ) -> FitResult:
        return FitResult(
            success=is_valid,
            chi_squared=chi2,
            reduced_chi_squared=chi2 / dof_g,
            parameters=side_params,
            uncertainties=side_unc,
            covariance=covariance,
            covariance_parameters=cov_params,
            residuals=resid,
            message=msg,
            function_calls=n_calls,
            dof=dof_g,
            minos_errors=minos_out,
            covariance_accurate=cov_accurate,
        )

    result_f = _pack(chi2_f, dof_f, residuals_f, params_f, unc_f)
    result_b = _pack(chi2_b, dof_b, residuals_b, params_b, unc_b)

    # shared_parameters: alpha + amp_* + tau_* (excluding per-side bg_F/bg_B).
    shared = ParameterSet()
    for p in result_params:
        if p.name not in {"bg_F", "bg_B"}:
            shared.add(p)

    return GroupedTimeDomainFitResult(
        success=is_valid,
        group_results={int(forward_group): result_f, int(backward_group): result_b},
        shared_parameters=shared,
        message=msg,
    )
