"""Bayesian experimental design for parameter trending.

Given a converged trend-model fit (``fit_parameter_model``), suggest where
the next measurement should go — and how many events it should collect —
to most constrain the model's parameters. The acquisition is the Laplace
(Gaussian-posterior) expected information gain for a single new datum:

    IG(x) = 1/2 log[1 + g(x)^T Sigma g(x) / sigma_new^2(x)]        (D-optimal)

    DeltaVar_k(x) = (Sigma g(x))_k^2 / (sigma_new^2(x) + g(x)^T Sigma g(x))
                                                                    (c-optimal)

where g(x) is the model's parameter sensitivity at the fitted values and
Sigma the fit covariance. The rank-one c-optimal update is exact for a
linear model; for strongly nonlinear models (critical exponents) it ranks
candidates correctly but *underestimates* the realised post-fit variance —
see ``docs/studies/prototype-results.md`` — so the counting recommendation
must be calibrated with ``calibrate_suggestion`` before being shown to a
user.

Everything here is GUI-free and pure numpy; refits happen only inside
``calibrate_suggestion``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.fitting.parameter_models import (
    ParameterCompositeModel,
    fit_parameter_model,
)
from asymmetry.core.fitting.parameters import ParameterSet

#: Condition number above which the covariance is flagged as ill-conditioned.
_COVARIANCE_CONDITION_WARN = 1.0e4

#: Utility ratio outside which the argmax is flagged as gradient-unstable
#: (recomputed with a 10x finite-difference step).
_GRADIENT_STABILITY_SPAN = (0.5, 2.0)

#: Realised/predicted variance ratio above which the analytic figure is
#: called out as optimistic in calibration warnings.
_CALIBRATION_OPTIMISM_WARN = 2.0

#: Fraction of failed calibration refits above which a warning is added.
_CALIBRATION_FAILURE_FRACTION_WARN = 0.25


@dataclass(frozen=True)
class NextPointSuggestion:
    """Utility curve and recommendation for the next measurement.

    ``utility`` is ``DeltaVar_k`` (variance reduction of the target
    parameter) in c-optimal mode, or the information gain ``IG`` in
    D-optimal mode (``target is None``). Two degenerate shapes exist, both
    explained by ``warnings``: unusable inputs (bad covariance, no noise
    model, ...) yield empty arrays with ``best_x`` NaN, while a valid setup
    where no candidate carries information keeps the full curve (all-zero
    ``utility``) with ``best_x`` NaN.
    """

    x_candidates: NDArray[np.float64]
    utility: NDArray[np.float64]
    extrapolated: NDArray[np.bool_]
    best_x: float
    target: str | None
    sigma_new: NDArray[np.float64]
    #: Predicted post-measurement sigma of the target at ``best_x`` for a
    #: reference-statistics point (c-optimal mode only). Rank-one estimate —
    #: optimistic near critical points; calibrate before display.
    predicted_post_sigma: float | None = None
    #: N/N_ref needed at ``best_x`` to reach ``sigma_goal`` (rank-one
    #: estimate). ``0.0`` means the goal is already met without a new point.
    events_factor_to_target: float | None = None
    target_unreachable: bool = False
    #: sigma of the target in the N -> infinity limit of a single new point
    #: at ``best_x`` — the floor set by the other parameters' uncertainty.
    floor_sigma: float | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class SuggestionCalibration:
    """Monte-Carlo check of a suggestion: simulate the point, refit, measure.

    ``realized_post_sigma`` is the median post-fit sigma of the target over
    the successful trials; the p16/p84 band shows its spread.
    ``calibration_ratio`` is realised variance / rank-one predicted variance
    (> 1 means the analytic figure was optimistic).
    """

    x_new: float
    events_factor: float
    target: str
    n_trials: int
    n_failed: int
    realized_post_sigma: float
    realized_post_sigma_p16: float
    realized_post_sigma_p84: float
    predicted_post_sigma: float | None = None
    calibration_ratio: float | None = None
    warnings: tuple[str, ...] = ()


def _empty_suggestion(target: str | None, warnings: list[str]) -> NextPointSuggestion:
    empty = np.zeros(0, dtype=float)
    return NextPointSuggestion(
        x_candidates=empty,
        utility=empty,
        extrapolated=np.zeros(0, dtype=bool),
        best_x=float("nan"),
        target=target,
        sigma_new=empty,
        warnings=tuple(warnings),
    )


def _validated_covariance(
    covariance: tuple[list[str], list[list[float]]] | None,
    warnings: list[str],
) -> tuple[list[str], NDArray[np.float64]] | None:
    if not covariance:
        warnings.append("No covariance matrix available from the fit.")
        return None
    names, matrix = covariance
    cov = np.asarray(matrix, dtype=float)
    if cov.ndim != 2 or cov.shape != (len(names), len(names)) or not names:
        warnings.append("Covariance matrix shape does not match its parameter names.")
        return None
    if not np.all(np.isfinite(cov)):
        warnings.append("Covariance matrix contains non-finite entries.")
        return None
    cov = 0.5 * (cov + cov.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    max_eig = float(eigenvalues[-1])
    min_eig = float(eigenvalues[0])
    if max_eig <= 0.0 or min_eig < -1e-12 * max(max_eig, 1.0):
        warnings.append("Covariance matrix is not positive definite.")
        return None
    if min_eig <= 0.0 or max_eig / min_eig > _COVARIANCE_CONDITION_WARN:
        warnings.append(
            "Fit covariance is ill-conditioned (strong parameter correlations); "
            "utilities are approximate."
        )
    return list(names), cov


def _empirical_sigma(
    x_data: NDArray[np.float64],
    y_err: NDArray[np.float64],
    warnings: list[str],
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Return (sorted x, sigma at those x) from the finite positive errors."""
    xd = np.asarray(x_data, dtype=float)
    ed = np.asarray(y_err, dtype=float)
    ok = np.isfinite(xd) & np.isfinite(ed) & (ed > 0.0)
    if not np.any(ok):
        warnings.append("No finite positive y-errors to build a noise model from.")
        return None
    order = np.argsort(xd[ok])
    return xd[ok][order], ed[ok][order]


def _sensitivities(
    model: ParameterCompositeModel,
    values: dict[str, float],
    free_names: list[str],
    x: NDArray[np.float64],
    step_scale: float = 1.0e-6,
) -> NDArray[np.float64]:
    """Central-difference d f/d theta_j over the whole candidate grid.

    Returns shape ``(len(free_names), len(x))``. Non-finite entries are
    zeroed — a candidate in a region where the model is undefined simply
    carries no information about that parameter.
    """
    grads = np.zeros((len(free_names), len(x)), dtype=float)
    for j, name in enumerate(free_names):
        theta = float(values[name])
        h = step_scale * max(1.0, abs(theta))
        hi = dict(values)
        lo = dict(values)
        hi[name] = theta + h
        lo[name] = theta - h
        f_hi = np.asarray(model.function(x, **hi), dtype=float)
        f_lo = np.asarray(model.function(x, **lo), dtype=float)
        grads[j] = (f_hi - f_lo) / (2.0 * h)
    return np.nan_to_num(grads, nan=0.0, posinf=0.0, neginf=0.0)


def _utility_curve(
    grads: NDArray[np.float64],
    cov: NDArray[np.float64],
    sigma_new: NDArray[np.float64],
    target_index: int | None,
) -> NDArray[np.float64]:
    projected = cov @ grads  # (n_params, n_candidates): Sigma g per candidate
    quadratic = np.einsum("jc,jc->c", grads, projected)  # g^T Sigma g
    variance = sigma_new**2
    if target_index is None:
        with np.errstate(invalid="ignore", divide="ignore"):
            utility = 0.5 * np.log1p(quadratic / variance)
    else:
        utility = projected[target_index] ** 2 / (variance + quadratic)
    return np.nan_to_num(utility, nan=0.0, posinf=0.0, neginf=0.0)


def suggest_next_point(
    model: ParameterCompositeModel,
    parameters: ParameterSet,
    covariance: tuple[list[str], list[list[float]]] | None,
    x_data: NDArray[np.float64],
    y_err: NDArray[np.float64],
    x_min: float,
    x_max: float,
    *,
    target: str | None = None,
    sigma_goal: float | None = None,
    n_candidates: int = 257,
) -> NextPointSuggestion:
    """Rank candidate x values by how much a new point there constrains the fit.

    ``covariance`` is the ``(names, matrix)`` pair from
    ``ParameterModelFitResult.covariance``; ``x_data``/``y_err`` are the
    measured series (the per-point errors define the empirical noise model
    for the hypothetical new point at reference statistics). ``target``
    selects c-optimality for that parameter; ``None`` means D-optimal.
    ``sigma_goal`` (c-optimal only) additionally solves for the event-count
    factor N/N_ref needed at the best x — a rank-one planning figure that
    should be calibrated with :func:`calibrate_suggestion` before display.

    Never raises on degenerate inputs: returns an empty suggestion with an
    explanatory warning instead.
    """
    warnings: list[str] = []

    validated = _validated_covariance(covariance, warnings)
    if validated is None:
        return _empty_suggestion(target, warnings)
    free_names, cov = validated

    if target is not None and target not in free_names:
        warnings.append(f"Target parameter {target!r} is not a free fitted parameter.")
        return _empty_suggestion(target, warnings)

    noise = _empirical_sigma(x_data, y_err, warnings)
    if noise is None:
        return _empty_suggestion(target, warnings)
    x_meas, sigma_meas = noise

    if not (np.isfinite(x_min) and np.isfinite(x_max)) or x_max <= x_min:
        warnings.append("Invalid candidate range.")
        return _empty_suggestion(target, warnings)

    if len(x_meas) <= len(free_names) + 1:
        warnings.append(
            "Fit is barely constrained (few points for the number of free "
            "parameters); the suggestion is unreliable — consider a coarse scan."
        )

    grid = np.linspace(float(x_min), float(x_max), int(n_candidates))
    in_range = x_meas[(x_meas >= x_min) & (x_meas <= x_max)]
    candidates = np.unique(np.concatenate([grid, in_range]))
    sigma_new = np.interp(candidates, x_meas, sigma_meas)
    extrapolated = (candidates < x_meas[0]) | (candidates > x_meas[-1])
    if np.any(extrapolated):
        warnings.append(
            "Candidate range extends beyond the measured points; suggestions "
            "there rely entirely on the assumed model form."
        )

    values = {p.name: float(p.value) for p in parameters}
    missing = [n for n in free_names if n not in values]
    if missing:
        warnings.append(f"Fitted values missing for parameters: {missing}.")
        return _empty_suggestion(target, warnings)

    target_index = free_names.index(target) if target is not None else None
    grads = _sensitivities(model, values, free_names, candidates)
    utility = _utility_curve(grads, cov, sigma_new, target_index)

    if not np.any(utility > 0.0):
        warnings.append("No candidate carries information about the fit parameters.")
        return NextPointSuggestion(
            x_candidates=candidates,
            utility=utility,
            extrapolated=extrapolated,
            best_x=float("nan"),
            target=target,
            sigma_new=sigma_new,
            warnings=tuple(warnings),
        )

    best_index = int(np.argmax(utility))
    best_x = float(candidates[best_index])

    # Gradient-stability probe at the winner only: near a model domain
    # boundary (e.g. x ~ fitted T_c) the finite-difference sensitivity is
    # finite but step-dependent, so a 10x step materially moving the utility
    # marks the suggestion as approximate.
    probe_x = np.array([best_x], dtype=float)
    probe_sigma = sigma_new[best_index : best_index + 1]
    coarse = _utility_curve(
        _sensitivities(model, values, free_names, probe_x, step_scale=1.0e-5),
        cov,
        probe_sigma,
        target_index,
    )[0]
    fine = utility[best_index]
    low, high = _GRADIENT_STABILITY_SPAN
    if coarse <= 0.0 or not (low <= coarse / fine <= high):
        warnings.append(
            "The suggested point sits near a model domain boundary; its "
            "utility is step-sensitive and approximate."
        )

    predicted_post_sigma: float | None = None
    events_factor: float | None = None
    unreachable = False
    floor_sigma: float | None = None
    if target_index is not None:
        g_best = grads[:, best_index]
        projected = cov @ g_best
        quadratic = float(g_best @ projected)
        prior_var = float(cov[target_index, target_index])
        gain = float(projected[target_index]) ** 2
        post_var = prior_var - gain / (float(sigma_new[best_index]) ** 2 + quadratic)
        predicted_post_sigma = float(np.sqrt(max(post_var, 0.0)))
        if quadratic > 0.0:
            floor_var = prior_var - gain / quadratic
            floor_sigma = float(np.sqrt(max(floor_var, 0.0)))
        if sigma_goal is not None and np.isfinite(sigma_goal) and sigma_goal > 0.0:
            goal_var = float(sigma_goal) ** 2
            if prior_var <= goal_var:
                events_factor = 0.0
            elif quadratic <= 0.0 or gain <= 0.0:
                unreachable = True
            else:
                needed = gain / (prior_var - goal_var) - quadratic
                if needed <= 0.0:
                    unreachable = True
                else:
                    events_factor = float(sigma_new[best_index]) ** 2 / needed
            if unreachable:
                warnings.append(
                    "The precision goal cannot be reached with a single new "
                    "point at the suggested x — the other parameters' "
                    "uncertainty sets the floor."
                )

    return NextPointSuggestion(
        x_candidates=candidates,
        utility=utility,
        extrapolated=extrapolated,
        best_x=best_x,
        target=target,
        sigma_new=sigma_new,
        predicted_post_sigma=predicted_post_sigma,
        events_factor_to_target=events_factor,
        target_unreachable=unreachable,
        floor_sigma=floor_sigma,
        warnings=tuple(warnings),
    )


def calibrate_suggestion(
    model: ParameterCompositeModel,
    parameters: ParameterSet,
    x_data: NDArray[np.float64],
    y_data: NDArray[np.float64],
    y_err: NDArray[np.float64],
    x_new: float,
    *,
    target: str,
    events_factor: float = 1.0,
    n_trials: int = 50,
    seed: int = 0,
    predicted_post_sigma: float | None = None,
) -> SuggestionCalibration:
    """Monte-Carlo-calibrate a suggestion by simulating the point and refitting.

    Each trial draws the hypothetical new datum from the fitted model plus
    Gaussian noise at the empirical sigma for ``x_new`` (scaled by
    ``1/sqrt(events_factor)``), appends it to the measured series, refits
    from the fitted values, and records the post-fit sigma of ``target``.
    The median over trials is the calibrated post-measurement sigma — the
    honest counterpart of the rank-one ``predicted_post_sigma``, which is
    known to be optimistic near critical points (see
    ``docs/studies/prototype-results.md``).

    ``y_err`` must be the effective per-point sigmas the trend fit was
    weighted with (already resolved through the fit's error mode).
    """
    warnings: list[str] = []
    xd = np.asarray(x_data, dtype=float)
    yd = np.asarray(y_data, dtype=float)
    ed = np.asarray(y_err, dtype=float)

    noise = _empirical_sigma(xd, ed, warnings)
    if noise is None or not np.isfinite(x_new) or events_factor <= 0.0 or n_trials < 1:
        warnings.append("Calibration inputs are degenerate; nothing simulated.")
        return SuggestionCalibration(
            x_new=float(x_new),
            events_factor=float(events_factor),
            target=target,
            n_trials=0,
            n_failed=0,
            realized_post_sigma=float("nan"),
            realized_post_sigma_p16=float("nan"),
            realized_post_sigma_p84=float("nan"),
            predicted_post_sigma=predicted_post_sigma,
            warnings=tuple(warnings),
        )
    x_meas, sigma_meas = noise
    sigma_new = float(np.interp(x_new, x_meas, sigma_meas)) / float(np.sqrt(events_factor))

    values = {p.name: float(p.value) for p in parameters}
    y_model_new = float(np.asarray(model.function(np.array([x_new]), **values))[0])

    realized: list[float] = []
    n_failed = 0
    for trial in range(int(n_trials)):
        rng = np.random.default_rng(seed + trial)
        y_new = y_model_new + rng.normal(0.0, sigma_new)
        result = fit_parameter_model(
            np.append(xd, x_new),
            np.append(yd, y_new),
            np.append(ed, sigma_new),
            model,
            parameters,
        )
        err = result.uncertainties.get(target) if result.success else None
        if err is not None and np.isfinite(err) and err > 0.0:
            realized.append(float(err))
        else:
            n_failed += 1

    if not realized:
        warnings.append("All calibration refits failed; no realised sigma available.")
        return SuggestionCalibration(
            x_new=float(x_new),
            events_factor=float(events_factor),
            target=target,
            n_trials=int(n_trials),
            n_failed=n_failed,
            realized_post_sigma=float("nan"),
            realized_post_sigma_p16=float("nan"),
            realized_post_sigma_p84=float("nan"),
            predicted_post_sigma=predicted_post_sigma,
            warnings=tuple(warnings),
        )

    sigmas = np.array(realized, dtype=float)
    median = float(np.median(sigmas))
    ratio: float | None = None
    if predicted_post_sigma is not None and predicted_post_sigma > 0.0:
        ratio = float(median**2 / predicted_post_sigma**2)
        if ratio > _CALIBRATION_OPTIMISM_WARN:
            warnings.append(
                "The analytic post-fit sigma is substantially optimistic here "
                "(strong model nonlinearity); trust the calibrated figure."
            )
    if n_failed > n_trials * _CALIBRATION_FAILURE_FRACTION_WARN:
        warnings.append("A large fraction of calibration refits failed.")

    return SuggestionCalibration(
        x_new=float(x_new),
        events_factor=float(events_factor),
        target=target,
        n_trials=int(n_trials),
        n_failed=n_failed,
        realized_post_sigma=median,
        realized_post_sigma_p16=float(np.percentile(sigmas, 16.0)),
        realized_post_sigma_p84=float(np.percentile(sigmas, 84.0)),
        predicted_post_sigma=predicted_post_sigma,
        calibration_ratio=ratio,
        warnings=tuple(warnings),
    )


def calibrate_events_for_goal(
    model: ParameterCompositeModel,
    parameters: ParameterSet,
    x_data: NDArray[np.float64],
    y_data: NDArray[np.float64],
    y_err: NDArray[np.float64],
    x_new: float,
    *,
    target: str,
    sigma_goal: float,
    initial_events_factor: float = 1.0,
    n_trials: int = 30,
    seed: int = 0,
    max_events_factor: float = 1024.0,
    max_iterations: int = 6,
) -> tuple[float | None, SuggestionCalibration]:
    """Find the event-count factor whose *calibrated* post-sigma meets the goal.

    Multiplicative search starting from ``initial_events_factor`` (use the
    rank-one ``events_factor_to_target`` when available): each step scales
    the factor by the shortfall ``(median/goal)^2`` — exact if the new
    point's variance dominates, conservative otherwise. Returns
    ``(factor, last_calibration)``; factor is ``None`` when the goal is not
    met by ``max_events_factor`` (the single-point floor applies).
    """
    factor = max(float(initial_events_factor), 1.0e-3)
    calibration: SuggestionCalibration | None = None
    for iteration in range(max(1, int(max_iterations))):
        calibration = calibrate_suggestion(
            model,
            parameters,
            x_data,
            y_data,
            y_err,
            x_new,
            target=target,
            events_factor=factor,
            n_trials=n_trials,
            seed=seed + 1000 * iteration,
        )
        median = calibration.realized_post_sigma
        if not np.isfinite(median):
            return None, calibration
        if median <= sigma_goal:
            return factor, calibration
        if factor >= max_events_factor:
            break
        factor = min(factor * min((median / sigma_goal) ** 2, 8.0), max_events_factor)
    assert calibration is not None
    return None, calibration
