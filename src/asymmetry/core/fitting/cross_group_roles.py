"""Role-suggestion engine for cross-group parameter fits.

This module is the cross-group analogue of the spectrum-level global fit
wizard (:mod:`asymmetry.core.fitting.global_fit_wizard`). Where that wizard
recommends which model parameters should be shared across a *series of
spectra*, :func:`suggest_cross_group_roles` recommends which parameters of a
parameter-vs-x model (e.g. ``λ(B)``) should be **Global** (one shared value
across all groups) versus **Local** (one value per group) when the same model
is fitted jointly across ``N`` groups via
:func:`asymmetry.core.fitting.parameter_models.global_fit_parameter_model`.

It reuses the wizard's vocabulary and scoring style: per-parameter
``global_score`` / ``local_score`` / ``score_delta`` /
``total_variation`` / ``roughness`` / ``rationale``, and AIC/AICc/BIC ranking
of candidate partitions.

Search strategy (bounded and deterministic — **not** ``3^k``)
--------------------------------------------------------------
Parameters the user pinned as *Fixed* are never flipped; they stay fixed at
the value the user set. Every other (non-fixed) model parameter is a candidate
for Global vs Local. Rather than enumerate all partitions, the search is:

1. **Baseline** — fit "all non-fixed params Global". Also fit the "all
   non-fixed params Local" reference (the block-separable extreme), which
   supplies the per-group value traces used for ``total_variation`` /
   ``roughness``.
2. **Single flips** — for each non-fixed param, fit "that param Local, the
   rest Global". Rank flips by criterion improvement over the baseline.
3. **Greedy accumulation** — apply the best improving flip, then re-evaluate
   the remaining single flips *on top of* it (a beam of width up to 2 when the
   budget allows), repeating until no flip improves the criterion or the
   ``max_fits`` budget is exhausted.

Every attempted candidate is recorded. Failed fits are kept with
``success=False`` and excluded from ranking. ``cancel_callback`` is checked
before each fit; on cancellation the partial results collected so far are
returned with ``message="cancelled"``.

Criterion definitions (with ``k = n_free`` free Minuit parameters and
``n = n_points`` from the fit result)::

    AIC  = chi2 + 2k
    AICc = AIC + 2k(k+1)/(n - k - 1)     (inf when n - k - 1 <= 0)
    BIC  = chi2 + k * ln(n)

Here ``chi2`` is the cross-group fit cost (``m.fval``), which equals ``-2 log
L`` up to an additive constant **only** under the ``COLUMN`` / ``ABSOLUTE`` /
``PERCENT`` error modes. Under ``NONE`` / ``SCATTER`` the unit y-weights carry
no physical scale, so the absolute criterion value is not a likelihood; the
*relative* comparison between candidates that share ``n_points`` and error
mode is still meaningful, and the recommendation flags this in ``message``.

Seeding
-------
To keep the many candidate fits fast to converge, each candidate seeds its
Minuit start values from an already-fitted parent: single flips seed from the
all-global baseline fit, and greedily-accumulated candidates seed from the
parent candidate they extend. Seeds are folded into the ``initial_params`` dict
passed to :func:`global_fit_parameter_model` (which seeds every global and
local instance of a name from that name's entry).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from asymmetry.core.fitting.parameter_models import (
    CrossGroupFitResult,
    ErrorMode,
    ParameterCompositeModel,
    ParameterGroupData,
    global_fit_parameter_model,
)

__all__ = [
    "CrossGroupCandidate",
    "CrossGroupParameterRecommendation",
    "CrossGroupRoleRecommendation",
    "suggest_cross_group_roles",
]

# Criterion improvement (in criterion units) a Local flip must clear before it
# is applied by the greedy search and recommended. Mirrors the wizard's
# ``_ROLE_DELTA_THRESHOLD`` so the two layers speak the same language: a flip
# that only marginally lowers the penalized score is treated as a tie in favour
# of the simpler (Global) assignment.
_ROLE_DELTA_THRESHOLD = 2.0

# Beam width for the greedy accumulation phase. Width 2 keeps a runner-up
# branch alive when the budget allows, without exploding toward ``3^k``.
_BEAM_WIDTH = 2

_VALID_CRITERIA = ("aic", "aicc", "bic")


@dataclass
class CrossGroupCandidate:
    """One attempted Global/Local/Fixed partition and its fit outcome.

    ``score`` is the value of the active selection criterion (AIC / AICc /
    BIC). ``n_free`` is the total number of free Minuit parameters:
    ``len(global) + len(local) * n_groups``. Failed fits keep
    ``success=False`` and are excluded from ranking (their criteria are
    ``inf``). ``result`` holds the underlying :class:`CrossGroupFitResult` for
    the best few candidates (populated for successful fits so callers can seed
    an actual fit from the recommended partition).
    """

    global_params: tuple[str, ...]
    local_params: tuple[str, ...]
    fixed_params: tuple[str, ...]
    success: bool
    chi_squared: float
    reduced_chi_squared: float
    n_free: int
    n_points: int
    aic: float
    aicc: float
    bic: float
    result: CrossGroupFitResult | None = None

    def criterion_value(self, criterion: str) -> float:
        """Return this candidate's value of ``criterion`` ("aic"/"aicc"/"bic")."""
        if criterion == "aic":
            return self.aic
        if criterion == "bic":
            return self.bic
        return self.aicc


@dataclass
class CrossGroupParameterRecommendation:
    """Recommended Global/Local role for one non-fixed model parameter.

    ``score_delta`` follows the sign convention ``criterion(global) -
    criterion(local)``: a **positive** delta means the all-else-global fit
    scores worse than localizing this parameter, i.e. the data reward making it
    Local. ``total_variation`` and ``roughness`` describe the spread of that
    parameter's per-group values in the all-local reference fit, ordered by
    ``group_variable_value`` and normalized by the value span (scale-free, as in
    the wizard).
    """

    name: str
    recommended_role: str  # "global" | "local"
    score_delta: float
    total_variation: float
    roughness: float
    rationale: str


@dataclass
class CrossGroupRoleRecommendation:
    """Full role-suggestion payload for one cross-group fit setup."""

    candidates: list[CrossGroupCandidate] = field(default_factory=list)
    recommended: CrossGroupCandidate | None = None
    parameters: list[CrossGroupParameterRecommendation] = field(default_factory=list)
    criterion: str = "aicc"
    message: str = ""


def _information_criteria(
    chi_squared: float, n_free: int, n_points: int
) -> tuple[float, float, float]:
    """Return (AIC, AICc, BIC) for a fit cost, free-param count and point count.

    ``chi_squared`` is the fit cost (``-2 log L`` up to a constant under the
    column/absolute/percent error modes). AICc guards a non-positive
    ``n - k - 1`` denominator by returning ``inf``.
    """
    k = float(n_free)
    n = float(n_points)
    if not np.isfinite(chi_squared):
        return float("inf"), float("inf"), float("inf")

    aic = chi_squared + 2.0 * k
    denom = n - k - 1.0
    if denom <= 0.0:
        aicc = float("inf")
    else:
        aicc = aic + (2.0 * k * (k + 1.0)) / denom
    bic = chi_squared + k * math.log(n) if n > 0.0 else float("inf")
    return aic, aicc, bic


def _trace_variation_and_roughness(values: np.ndarray) -> tuple[float, float]:
    """Total variation and roughness of a per-group value trace (wizard style).

    ``total_variation`` is the sum of absolute successive differences divided by
    the value span; ``roughness`` is the RMS of second differences divided by
    the span. Both are scale-free so rationale thresholds do not depend on the
    parameter's magnitude. Mirrors
    ``global_fit_wizard._parameter_trace_roughness_from_results``.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return 0.0, 0.0
    span = max(
        float(np.max(values) - np.min(values)),
        float(np.max(np.abs(values))),
        1e-9,
    )
    total_variation = float(np.sum(np.abs(np.diff(values))) / span)
    if values.size < 3:
        return total_variation, 0.0
    roughness = float(np.sqrt(np.mean(np.square(np.diff(values, n=2)))) / span)
    return total_variation, roughness


def _local_value_trace(
    result: CrossGroupFitResult,
    ordered_group_ids: list[str],
    name: str,
) -> np.ndarray:
    """Per-group values of local parameter ``name`` from an all-local-style fit.

    Ordered to match ``ordered_group_ids`` (groups sorted by
    ``group_variable_value``). Groups missing the parameter contribute NaN,
    which the variation/roughness helper drops.
    """
    trace: list[float] = []
    for group_id in ordered_group_ids:
        pset = result.local_parameters.get(group_id)
        if pset is not None and name in pset:
            trace.append(float(pset[name].value))
        else:
            trace.append(float("nan"))
    return np.asarray(trace, dtype=float)


class _FitRunner:
    """Runs and caches cross-group fits, enforcing the ``max_fits`` budget.

    Fits are memoized on the (global, local, fixed) partition so the greedy
    search never re-runs an identical partition. ``cancel_callback`` is checked
    before each *new* fit; a cached partition never re-checks. Once the budget
    is exhausted, :meth:`fit` returns ``None`` (callers treat that as "no more
    candidates available").
    """

    def __init__(
        self,
        groups: list[ParameterGroupData],
        model: ParameterCompositeModel,
        *,
        base_initial_params: dict[str, float],
        parameter_bounds: dict[str, tuple[float, float]] | None,
        fixed_values: dict[str, float],
        error_mode: ErrorMode,
        error_value: float | None,
        windows,
        xerr,
        criterion: str,
        max_fits: int,
        cancel_callback: Callable[[], bool] | None,
    ) -> None:
        self._groups = groups
        self._model = model
        self._base_initial_params = base_initial_params
        self._parameter_bounds = parameter_bounds
        self._fixed_values = fixed_values
        self._error_mode = error_mode
        self._error_value = error_value
        self._windows = windows
        self._xerr = xerr
        self._criterion = criterion
        self._max_fits = max(0, int(max_fits))
        self._cancel_callback = cancel_callback

        self._n_groups = len(groups)
        self._cache: dict[tuple[tuple[str, ...], tuple[str, ...]], CrossGroupCandidate] = {}
        self.candidates: list[CrossGroupCandidate] = []
        self.fits_used = 0
        self.cancelled = False

    @property
    def budget_exhausted(self) -> bool:
        return self.fits_used >= self._max_fits

    def _check_cancel(self) -> bool:
        if self._cancel_callback is None:
            return False
        try:
            if self._cancel_callback():
                self.cancelled = True
                return True
        except Exception:
            # A misbehaving callback must never take down the search.
            return False
        return False

    def fit(
        self,
        global_params: tuple[str, ...],
        local_params: tuple[str, ...],
        *,
        seed_from: CrossGroupCandidate | None = None,
    ) -> CrossGroupCandidate | None:
        """Fit one partition (or return the cached candidate).

        Returns ``None`` if the budget is exhausted or the search was
        cancelled before this fit ran. ``seed_from`` supplies start values from
        an already-fitted parent candidate.
        """
        key = (tuple(sorted(global_params)), tuple(sorted(local_params)))
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        if self.cancelled or self.budget_exhausted:
            return None
        if self._check_cancel():
            return None

        initial_params = dict(self._base_initial_params)
        if seed_from is not None and seed_from.result is not None:
            initial_params.update(_seed_values_from_result(seed_from.result))

        self.fits_used += 1
        result = global_fit_parameter_model(
            self._groups,
            self._model,
            global_params=list(global_params),
            local_params=list(local_params),
            fixed_params=dict(self._fixed_values),
            initial_params=initial_params,
            parameter_bounds=self._parameter_bounds,
            error_mode=self._error_mode,
            error_value=self._error_value,
            windows=self._windows,
            xerr=self._xerr,
        )

        n_free = len(global_params) + len(local_params) * self._n_groups
        n_points = int(result.n_points)
        success = bool(result.success)
        if success:
            aic, aicc, bic = _information_criteria(result.chi_squared, n_free, n_points)
        else:
            aic = aicc = bic = float("inf")

        candidate = CrossGroupCandidate(
            global_params=tuple(sorted(global_params)),
            local_params=tuple(sorted(local_params)),
            fixed_params=tuple(sorted(self._fixed_values)),
            success=success,
            chi_squared=float(result.chi_squared),
            reduced_chi_squared=float(result.reduced_chi_squared),
            n_free=int(n_free),
            n_points=n_points,
            aic=float(aic),
            aicc=float(aicc),
            bic=float(bic),
            result=result if success else None,
        )
        self._cache[key] = candidate
        self.candidates.append(candidate)
        return candidate


def _seed_values_from_result(result: CrossGroupFitResult) -> dict[str, float]:
    """Flatten a fit result into a name→value seed dict.

    Global values map directly. For a local parameter the *median* over groups
    is used as the shared seed (a stable single value that
    ``global_fit_parameter_model`` will apply to every instance of that name).
    """
    seeds: dict[str, float] = {}
    for parameter in result.global_parameters:
        seeds[parameter.name] = float(parameter.value)
    local_values: dict[str, list[float]] = {}
    for pset in result.local_parameters.values():
        for parameter in pset:
            local_values.setdefault(parameter.name, []).append(float(parameter.value))
    for name, values in local_values.items():
        finite = [v for v in values if np.isfinite(v)]
        if finite:
            seeds[name] = float(np.median(np.asarray(finite, dtype=float)))
    return seeds


def _rank_candidates(
    candidates: list[CrossGroupCandidate],
    criterion: str,
) -> list[CrossGroupCandidate]:
    """Successful candidates, best criterion first; deterministic tie-break.

    Ties on the criterion break toward fewer free parameters (simpler model),
    then lexicographically by the local-parameter tuple, so ordering is
    reproducible across runs.
    """
    successful = [c for c in candidates if c.success and np.isfinite(c.criterion_value(criterion))]
    return sorted(
        successful,
        key=lambda c: (c.criterion_value(criterion), c.n_free, c.local_params),
    )


def suggest_cross_group_roles(
    groups: list[ParameterGroupData],
    model: ParameterCompositeModel,
    *,
    initial_params: dict[str, float] | None = None,
    parameter_bounds: dict[str, tuple[float, float]] | None = None,
    fixed_params: dict[str, float] | None = None,
    error_mode: ErrorMode | str = ErrorMode.COLUMN,
    error_value: float | None = None,
    windows=None,
    xerr=None,
    criterion: str = "aicc",
    max_fits: int = 40,
    cancel_callback: Callable[[], bool] | None = None,
) -> CrossGroupRoleRecommendation:
    """Recommend Global/Local roles for a cross-group parameter fit.

    ``groups`` and ``model`` are the same objects passed to
    :func:`global_fit_parameter_model`. ``fixed_params`` maps the parameters the
    user pinned as *Fixed* to their held values; those are never flipped.
    ``criterion`` selects the ranking statistic (``"aic"`` / ``"aicc"`` /
    ``"bic"``). ``max_fits`` caps the total number of candidate fits.
    ``cancel_callback`` is polled before each fit; returning ``True`` stops the
    search and yields a partial result with ``message="cancelled"``.

    Returns a :class:`CrossGroupRoleRecommendation` whose ``candidates`` are in
    deterministic best-first order, ``recommended`` is the best-scoring
    successful candidate (or ``None``), and ``parameters`` gives the per-param
    Global/Local recommendation with wizard-style diagnostics.
    """
    criterion = criterion.lower().strip()
    if criterion not in _VALID_CRITERIA:
        criterion = "aicc"

    error_mode = ErrorMode(error_mode)
    fixed_params = dict(fixed_params or {})
    parameter_bounds = dict(parameter_bounds or {})

    all_names = list(model.param_names)
    fixed_names = set(fixed_params)
    free_names = tuple(name for name in all_names if name not in fixed_names)

    base_initial_params = dict(model.param_defaults)
    if initial_params:
        base_initial_params.update(initial_params)

    ordered_group_ids = [
        group.group_id
        for group in sorted(groups, key=lambda g: (float(g.group_variable_value), g.group_id))
    ]

    # ``NONE``/``SCATTER`` weight uniformly, so the criterion value is not a
    # likelihood; relative comparison is still well-defined but we flag it.
    relative_only = error_mode in (ErrorMode.NONE, ErrorMode.SCATTER)

    recommendation = CrossGroupRoleRecommendation(criterion=criterion)

    if len(groups) < 2:
        recommendation.message = "Need at least two groups for cross-group role suggestion."
        return recommendation
    if not free_names:
        recommendation.message = "All parameters are fixed; nothing to classify as Global vs Local."
        return recommendation

    runner = _FitRunner(
        groups,
        model,
        base_initial_params=base_initial_params,
        parameter_bounds=parameter_bounds,
        fixed_values=fixed_params,
        error_mode=error_mode,
        error_value=error_value,
        windows=windows,
        xerr=xerr,
        criterion=criterion,
        max_fits=max_fits,
        cancel_callback=cancel_callback,
    )

    # (1) Baseline: all free params global.
    baseline = runner.fit(free_names, ())
    if runner.cancelled:
        return _finalize(runner, recommendation, ordered_group_ids, free_names, relative_only, None)

    # All-local reference: supplies the per-group value traces and a
    # block-separable comparison point.
    all_local = runner.fit((), free_names, seed_from=baseline)
    if runner.cancelled:
        return _finalize(
            runner, recommendation, ordered_group_ids, free_names, relative_only, all_local
        )

    # (2) Single flips: each free param local, the rest global, seeded from the
    # baseline all-global fit.
    single_flip_by_name: dict[str, CrossGroupCandidate] = {}
    for name in free_names:
        if runner.budget_exhausted or runner.cancelled:
            break
        global_side = tuple(n for n in free_names if n != name)
        candidate = runner.fit(global_side, (name,), seed_from=baseline)
        if candidate is None:
            break
        single_flip_by_name[name] = candidate

    if runner.cancelled:
        return _finalize(
            runner, recommendation, ordered_group_ids, free_names, relative_only, all_local
        )

    # (3) Greedy / beam accumulation over the improving single flips.
    if baseline is not None and baseline.success:
        _greedy_accumulate(
            runner,
            baseline=baseline,
            free_names=free_names,
            single_flip_by_name=single_flip_by_name,
            criterion=criterion,
        )

    return _finalize(
        runner, recommendation, ordered_group_ids, free_names, relative_only, all_local
    )


def _greedy_accumulate(
    runner: _FitRunner,
    *,
    baseline: CrossGroupCandidate,
    free_names: tuple[str, ...],
    single_flip_by_name: dict[str, CrossGroupCandidate],
    criterion: str,
) -> None:
    """Greedily add the best improving Local flip, beam width up to 2.

    Starting from the incumbent all-global baseline, repeatedly try localizing
    each still-global free param on top of the current local set, keep the
    frontier of the best branches, and stop when nothing clears the incumbent's
    criterion (minus the role threshold) or the budget runs out.
    """
    # A beam entry is (candidate, local_set).
    frontier: list[tuple[CrossGroupCandidate, frozenset[str]]] = [(baseline, frozenset())]

    while frontier and not runner.budget_exhausted and not runner.cancelled:
        next_frontier: list[tuple[CrossGroupCandidate, frozenset[str]]] = []
        for incumbent, local_set in frontier:
            incumbent_score = incumbent.criterion_value(criterion)
            remaining = [name for name in free_names if name not in local_set]
            for name in remaining:
                if runner.budget_exhausted or runner.cancelled:
                    break
                new_local = local_set | {name}
                # Seed from the single-flip when extending the empty set (its
                # parent is the baseline); otherwise from the incumbent.
                seed = single_flip_by_name.get(name) if not local_set else incumbent
                seed = seed if seed is not None and seed.success else incumbent
                global_side = tuple(n for n in free_names if n not in new_local)
                candidate = runner.fit(global_side, tuple(sorted(new_local)), seed_from=seed)
                if candidate is None or not candidate.success:
                    continue
                improvement = incumbent_score - candidate.criterion_value(criterion)
                if improvement > _ROLE_DELTA_THRESHOLD:
                    next_frontier.append((candidate, frozenset(new_local)))

        if not next_frontier:
            break
        # Keep the best few branches by criterion (deterministic tie-break).
        next_frontier.sort(
            key=lambda item: (
                item[0].criterion_value(criterion),
                item[0].n_free,
                item[0].local_params,
            )
        )
        frontier = next_frontier[:_BEAM_WIDTH]


def _finalize(
    runner: _FitRunner,
    recommendation: CrossGroupRoleRecommendation,
    ordered_group_ids: list[str],
    free_names: tuple[str, ...],
    relative_only: bool,
    all_local: CrossGroupCandidate | None,
) -> CrossGroupRoleRecommendation:
    """Rank candidates, choose the recommendation, and build per-param rows."""
    criterion = recommendation.criterion
    recommendation.candidates = _rank_candidates(runner.candidates, criterion)
    # Record every attempted candidate too (failed ones excluded from ranking
    # but discoverable). Ranked successes come first; append the rest.
    ranked_keys = {(c.global_params, c.local_params) for c in recommendation.candidates}
    for candidate in runner.candidates:
        if (candidate.global_params, candidate.local_params) not in ranked_keys:
            recommendation.candidates.append(candidate)

    recommended = recommendation.candidates[0] if recommendation.candidates else None
    if recommended is not None and not recommended.success:
        recommended = None
    recommendation.recommended = recommended

    recommendation.parameters = _parameter_recommendations(
        runner,
        recommended=recommended,
        free_names=free_names,
        ordered_group_ids=ordered_group_ids,
        all_local=all_local,
        criterion=criterion,
    )

    messages: list[str] = []
    if runner.cancelled:
        messages.append("cancelled")
    if recommended is None and not runner.cancelled:
        messages.append("No candidate fit converged; roles left unchanged.")
    if relative_only:
        messages.append(
            f"Error mode is {runner._error_mode.value!r}: {criterion.upper()} values are "
            "not likelihoods, only their relative ordering across candidates is meaningful."
        )
    if recommended is not None and not runner.cancelled:
        local_label = ", ".join(recommended.local_params) or "none"
        messages.append(
            f"Recommended partition: local = [{local_label}] "
            f"({criterion.upper()} = {recommended.criterion_value(criterion):.2f}, "
            f"{runner.fits_used} candidate fit(s) evaluated)."
        )
    recommendation.message = " ".join(messages)
    return recommendation


def _parameter_recommendations(
    runner: _FitRunner,
    *,
    recommended: CrossGroupCandidate | None,
    free_names: tuple[str, ...],
    ordered_group_ids: list[str],
    all_local: CrossGroupCandidate | None,
    criterion: str,
) -> list[CrossGroupParameterRecommendation]:
    """Per-parameter Global/Local recommendation with wizard-style diagnostics.

    For each free parameter, ``score_delta = criterion(global) -
    criterion(local)`` compares the two single-role fits that differ only in
    this parameter, relative to the recommended partition's assignment for the
    other parameters. ``total_variation`` / ``roughness`` come from the
    all-local reference fit's per-group value trace.
    """
    recommended_local = set(recommended.local_params) if recommended is not None else set()
    rows: list[CrossGroupParameterRecommendation] = []

    for name in free_names:
        # Variation / roughness from the all-local reference (best available
        # per-group trace); fall back to the recommended fit if it localizes it.
        trace_source = None
        if all_local is not None and all_local.result is not None:
            trace_source = all_local.result
        elif (
            recommended is not None and recommended.result is not None and name in recommended_local
        ):
            trace_source = recommended.result
        if trace_source is not None:
            trace = _local_value_trace(trace_source, ordered_group_ids, name)
            total_variation, roughness = _trace_variation_and_roughness(trace)
        else:
            total_variation, roughness = 0.0, 0.0

        # Compare "name local vs name global" holding the other params at the
        # recommended assignment. Look up both partitions in the runner cache.
        others_local = tuple(sorted(recommended_local - {name}))
        local_with = tuple(sorted({*others_local, name}))
        global_side_with_local = tuple(n for n in free_names if n not in local_with)
        global_side_without = tuple(n for n in free_names if n not in others_local)

        local_candidate = _lookup(runner, global_side_with_local, local_with)
        global_candidate = _lookup(runner, global_side_without, others_local)

        local_score = (
            local_candidate.criterion_value(criterion)
            if local_candidate is not None and local_candidate.success
            else float("inf")
        )
        global_score = (
            global_candidate.criterion_value(criterion)
            if global_candidate is not None and global_candidate.success
            else float("inf")
        )
        score_delta = global_score - local_score

        recommended_role = "local" if name in recommended_local else "global"
        rationale = _parameter_rationale(
            name=name,
            recommended_role=recommended_role,
            score_delta=score_delta,
            total_variation=total_variation,
            roughness=roughness,
            local_available=local_candidate is not None and local_candidate.success,
            global_available=global_candidate is not None and global_candidate.success,
            criterion=criterion,
        )
        rows.append(
            CrossGroupParameterRecommendation(
                name=name,
                recommended_role=recommended_role,
                score_delta=float(score_delta) if np.isfinite(score_delta) else float("inf"),
                total_variation=float(total_variation),
                roughness=float(roughness),
                rationale=rationale,
            )
        )
    return rows


def _lookup(
    runner: _FitRunner,
    global_params: tuple[str, ...],
    local_params: tuple[str, ...],
) -> CrossGroupCandidate | None:
    key = (tuple(sorted(global_params)), tuple(sorted(local_params)))
    return runner._cache.get(key)


def _parameter_rationale(
    *,
    name: str,
    recommended_role: str,
    score_delta: float,
    total_variation: float,
    roughness: float,
    local_available: bool,
    global_available: bool,
    criterion: str,
) -> str:
    crit = criterion.upper()
    if recommended_role == "local":
        if not local_available:
            return (
                f"{name} is recommended Local, but no comparable shared fit was available to "
                "quantify the improvement."
            )
        if np.isfinite(score_delta):
            return (
                f"Localizing {name} improves {crit} by {score_delta:.2f} "
                f"(per-group spread: variation {total_variation:.2f}, roughness {roughness:.2f})."
            )
        return (
            f"Sharing {name} across groups did not converge, so it is kept Local "
            f"(per-group spread: variation {total_variation:.2f})."
        )
    # Recommended global.
    if not global_available:
        return (
            f"{name} is recommended Global, but the all-else-global reference fit did not "
            "converge to quantify the comparison."
        )
    if np.isfinite(score_delta):
        return (
            f"Sharing {name} is favoured: localizing it changes {crit} by {score_delta:.2f}, "
            f"below the {_ROLE_DELTA_THRESHOLD:.1f} threshold to justify per-group values "
            f"(per-group spread: variation {total_variation:.2f}, roughness {roughness:.2f})."
        )
    return (
        f"Localizing {name} did not converge or gave no improvement, so it is shared across groups."
    )
