"""Fit driver: single-run and global (simultaneous) fitting.

Uses :mod:`iminuit` as the fitting back-end, providing robust minimization
without scipy dependencies (important for Python 3.13+ compatibility).
"""

from __future__ import annotations

import inspect
import warnings
from collections import Counter
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.parameters import Parameter, ParameterSet


class AsymmetryScaleWarning(UserWarning):
    """Emitted when fit seeds and data appear to be on different asymmetry scales.

    The classic trap: a model seeded with fraction-scale amplitudes
    (``A ∈ [-1, 1]``) is fitted against a loaded ``MuonDataset`` whose
    ``asymmetry`` is on the **percent** scale (``×100``). The fit either
    converges to a degenerate amplitude or to the wrong minimum. See
    :class:`asymmetry.core.data.dataset.MuonDataset` and its
    ``asymmetry_percent`` / ``asymmetry_fraction`` accessors.
    """


class FixedFrequencyFieldMismatchWarning(UserWarning):
    """Emitted when a fit pins a precession frequency far from γ_μ·B(field).

    The TF trap: a transverse-field oscillation is fitted with its
    ``frequency`` parameter *fixed* (``Parameter.fixed=True``) to a value that
    disagrees with the Larmor frequency ``ν = γ_μ B / 2π`` implied by the run's
    ``field`` metadata. Pinning the line away from its true position pushes the
    misfit into the damping term, inflating the fitted Gaussian ``sigma`` (~8%
    on a vortex-state superconductor, where the diamagnetic shift moves the
    line below T_c). Letting ``frequency`` float removes the bias. See
    :func:`asymmetry.core.fourier.units.gauss_to_mhz` for the conversion and the
    "transverse-field frequency" cookbook entry.
    """


#: Upper bound on a peak magnitude that could plausibly be a *fraction*
#: asymmetry. A true ``A`` lies in ``[-1, 1]``; the headroom absorbs noise and
#: the ±1 one-sided sentinel. A peak above this can only be percent-scale.
_FRACTION_SCALE_CEILING = 1.5

#: Relative gap between a *fixed* frequency seed and γ_μ·B(field) above which the
#: pin is flagged. 2% comfortably clears fit/rounding noise while still catching
#: the few-percent vortex-state diamagnetic shift that inflates σ.
_FIXED_FREQ_FIELD_REL_TOLERANCE = 0.02

#: Floor on the field-implied reference frequency γ_μ·B (MHz) below which the
#: guard stays silent. A single floor covers three cases at once: zero-field
#: (field 0 → reference 0), a few gauss of stray/residual field, and the
#: "needs a meaningful reference" requirement — below ~0.1 MHz (≈7 G) a fixed
#: frequency is almost certainly not a γ_μ·B Larmor line, so second-guessing it
#: would be a false positive.
_MIN_REFERENCE_MHZ = 0.1

#: Minimum peak ratio (on top of a fraction/percent boundary crossing) before a
#: mismatch is flagged — keeps near-boundary noise quiet without masking the
#: ~100× percent-vs-fraction trap.
_SCALE_MISMATCH_RATIO = 10.0


def _warn_on_scale_mismatch(
    time: NDArray[np.float64],
    asymmetry: NDArray[np.float64],
    model_wrapper: Callable[..., NDArray[np.float64]],
    initial_values: Sequence[float],
) -> None:
    """Warn when the data and the seeded model curve sit on different scales.

    Compares the peak magnitude (max |·|) of the data to that of the model
    evaluated at its seed. The peak is the amplitude proxy: a decaying model's
    *median* sits in its near-zero tail and would mismeasure the amplitude,
    whereas its peak tracks the seeded amplitude.

    The flagged condition is a *fraction/percent boundary crossing*: one peak is
    fraction-scale (``≤ 1.5`` — a true ``|A| ≤ 1`` cannot be more) while the
    other can only be percent. An on-scale-but-poorly-guessed amplitude (both
    peaks clearly percent) is deliberately NOT flagged, however far apart they
    are — this guard is about scale confusion, not seed quality. Advisory only:
    it never raises and never blocks the fit; any evaluation failure is swallowed
    so the guard cannot change fit outcomes.
    """
    try:
        asym = np.asarray(asymmetry, dtype=np.float64)
        data_finite = asym[np.isfinite(asym) & (asym != 0.0)]
        if data_finite.size == 0:
            return
        model_vals = np.asarray(model_wrapper(time, *initial_values), dtype=np.float64)
        model_finite = model_vals[np.isfinite(model_vals) & (model_vals != 0.0)]
        if model_finite.size == 0:
            return
        data_mag = float(np.max(np.abs(data_finite)))
        model_mag = float(np.max(np.abs(model_finite)))
        if data_mag <= 0.0 or model_mag <= 0.0:
            return
        low, high = sorted((data_mag, model_mag))
        straddles_scale = low <= _FRACTION_SCALE_CEILING < high
        ratio = high / low
        if not straddles_scale or ratio < _SCALE_MISMATCH_RATIO:
            return
    except Exception:  # noqa: BLE001 - a guard must never break the fit it guards
        return

    data_is_larger = data_mag > model_mag
    likely = (
        "data looks percent-scale while the seeds look fraction-scale"
        if data_is_larger
        else "data looks fraction-scale while the seeds look percent-scale"
    )
    warnings.warn(
        f"Asymmetry scale mismatch: the data magnitude (~{data_mag:.3g}) and the "
        f"seeded model magnitude (~{model_mag:.3g}) differ by ~{ratio:.0f}×, so {likely}. "
        "Loaded MuonDataset.asymmetry is on the percent scale (×100); seed amplitudes "
        "to match, or pass ds.asymmetry_fraction / ds.asymmetry_percent explicitly.",
        AsymmetryScaleWarning,
        stacklevel=3,
    )


def _warn_on_fixed_frequency_far_from_field(
    dataset: MuonDataset,
    parameters: ParameterSet,
) -> None:
    """Warn when a *fixed* precession frequency disagrees with γ_μ·B(field).

    Iterates the seeds for any parameter that is both ``fixed`` and named like a
    frequency (``"freq"`` substring, matching the model convention), and compares
    each to the Larmor frequency ``ν = γ_μ B / 2π`` implied by the run's ``field``
    metadata. A pin further than :data:`_FIXED_FREQ_FIELD_REL_TOLERANCE` from that
    reference is flagged, because the misfit then leaks into the damping term and
    inflates the fitted ``sigma`` (the vortex-state TF trap).

    Deliberately quiet for the cases where γ_μ·B is not the relevant line: a
    *free* frequency (only ``fixed`` seeds are checked), a run with no ``field``
    metadata, and zero-/stray-field runs whose reference frequency falls below
    :data:`_MIN_REFERENCE_MHZ` (which also covers genuine ZF/LF relaxation models,
    as those carry no frequency parameter at all). Advisory only: it never raises
    and never blocks the fit; any failure is swallowed so the guard cannot change
    fit outcomes.
    """
    try:
        from asymmetry.core.fourier.units import gauss_to_mhz

        field = dataset.field
        if field is None:
            return
        reference = float(gauss_to_mhz(field))
        if not np.isfinite(reference) or abs(reference) < _MIN_REFERENCE_MHZ:
            return

        for p in parameters:
            if not getattr(p, "fixed", False) or "freq" not in p.name.lower():
                continue
            value = float(p.value)
            if not np.isfinite(value):
                continue
            rel_gap = abs(value - reference) / abs(reference)
            if rel_gap <= _FIXED_FREQ_FIELD_REL_TOLERANCE:
                continue
            warnings.warn(
                f"Fixed-frequency trap: parameter {p.name!r} is pinned at "
                f"{value:.4g} MHz, ~{rel_gap * 100:.0f}% away from γ_μ·B = "
                f"{reference:.4g} MHz implied by the run's {float(field):.0f} G "
                "field. Pinning a transverse-field precession line away from its "
                "true position pushes the misfit into the damping term and "
                "inflates the fitted Gaussian sigma (~8% on a vortex-state "
                "superconductor, where the diamagnetic shift moves the line below "
                f"T_c). Let {p.name!r} float (Parameter.fixed=False) to remove the "
                "bias, or correct the pinned value to match the field.",
                FixedFrequencyFieldMismatchWarning,
                stacklevel=3,
            )
    except Exception:  # noqa: BLE001 - a guard must never break the fit it guards
        return


class FitCancelledError(RuntimeError):
    """Raised when a fit is cancelled cooperatively via a ``cancel_callback``.

    Mirrors :class:`~asymmetry.core.maxent.engine.MaxEntCancelledError`. A cancelled
    fit records **no** result: the partially-minimised state is discarded entirely,
    so callers must let this propagate and not pack a :class:`FitResult` from it.
    """


#: Poll the cancel callback once every this many cost-function evaluations. A flag
#: read is nanoseconds against a microsecond model evaluation, so this is small
#: enough to abort even short fits (which converge in tens of calls) while keeping
#: the poll off the very hottest path; between-fit checks in series/global loops
#: guarantee a clean stop regardless.
_CANCEL_POLL_INTERVAL = 8


def _make_cancel_guard(cancel_callback: Callable[[], bool] | None) -> Callable[[], None]:
    """Return a guard that raises :class:`FitCancelledError` when cancellation is set.

    The returned callable is invoked inside the cost function; it polls
    ``cancel_callback`` every :data:`_CANCEL_POLL_INTERVAL` calls (in-fit abort
    granularity). A ``None`` callback yields a no-op guard.
    """
    if cancel_callback is None:
        return lambda: None

    counter = {"n": 0}

    def guard() -> None:
        counter["n"] += 1
        if counter["n"] % _CANCEL_POLL_INTERVAL == 0 and bool(cancel_callback()):
            raise FitCancelledError("Fit cancelled.")

    return guard


def _validate_tie_references(parameters, ties: dict) -> None:
    """Validate affine-tie references before fitting.

    Each ``main``/``offset`` must name a parameter in the set, must not be a
    tie follower itself (no tie-to-tie chaining — the engine resolves ties in a
    single pass), and a tied parameter must not also be link-grouped or fixed
    (the tie would silently win, discarding the other constraint). Raising here
    gives a clear error instead of a cryptic ``KeyError`` deep in the cost
    function or a silently-ignored ``fixed`` flag.
    """
    for name, tie in ties.items():
        if parameters[name].link_group is not None:
            raise ValueError(
                f"Parameter '{name}' cannot be both link-grouped and affinely tied; "
                "use one constraint or the other."
            )
        if parameters[name].fixed:
            raise ValueError(
                f"Parameter '{name}' cannot be both fixed and affinely tied; "
                "a tie derives its value, so drop the fixed flag."
            )
        for ref in tie.references():
            if ref not in parameters:
                raise ValueError(f"Affine tie on '{name}' references unknown parameter '{ref}'.")
            if ref in ties:
                raise ValueError(
                    f"Affine tie on '{name}' references tied parameter '{ref}'; "
                    "ties may not chain to other ties."
                )


def _reject_affine_ties(parameter_sets, context: str) -> None:
    """Raise if any parameter set carries an affine tie on a path that lacks support.

    Affine ties are currently honoured only by the single-run :meth:`FitEngine.fit`
    path. Global, count-domain, and grouped/chained fits build their own parameter
    partition and cost wrappers and do not resolve ``tie_followers()``; silently
    ignoring a tie there would reintroduce the very free-frequency scatter the
    feature removes. Fail loudly with a pointer to the supported path instead.
    """
    for ps in parameter_sets:
        if ps is not None and ps.tie_followers():
            raise NotImplementedError(
                f"{context} does not support affine parameter ties yet; tied "
                "parameters are honoured only by single-run FitEngine.fit(). "
                "Fit each run individually, or remove the tie."
            )


def _model_kwarg_names(model_fn) -> set[str] | None:
    """Keyword names ``model_fn`` accepts, or ``None`` if it takes ``**kwargs``.

    Affine ties may introduce free *auxiliary* parameters (e.g. a half-splitting
    ``delta``) that the model never consumes. Those must not be forwarded to a
    model whose signature is explicit — it would raise ``TypeError`` on the
    unexpected kwarg. A model that declares ``**kwargs`` (e.g.
    :meth:`CompositeModel.function`) accepts everything, so no filtering is
    needed (return ``None``). Un-introspectable callables are treated as
    permissive.
    """
    try:
        sig = inspect.signature(model_fn)
    except (TypeError, ValueError):
        return None
    names: set[str] = set()
    for p in sig.parameters.values():
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            return None
        names.add(p.name)
    return names


def _minuit_status_message(minuit, *, success_message: str, failure_prefix: str) -> str:
    if getattr(minuit, "valid", False):
        return success_message

    details: list[str] = []
    fmin = getattr(minuit, "fmin", None)
    if fmin is not None:
        if getattr(fmin, "has_reached_call_limit", False):
            details.append("call limit reached")
        if getattr(fmin, "is_above_max_edm", False):
            details.append("EDM above threshold")
        if getattr(fmin, "has_parameters_at_limit", False):
            details.append("parameters at limit")
        if not getattr(fmin, "has_valid_parameters", True):
            details.append("invalid parameters")
        if getattr(fmin, "hesse_failed", False):
            details.append("hesse failed")
        if not getattr(fmin, "is_valid", True):
            details.append("minimum invalid")

    if not details:
        return failure_prefix
    return f"{failure_prefix}: {', '.join(details)}"


def _clamp_minuit_step_size(step: float, lower: float, upper: float) -> float:
    clipped = abs(float(step))
    if not np.isfinite(clipped) or clipped <= 0.0:
        return 0.0
    if np.isfinite(lower) and np.isfinite(upper) and upper > lower:
        width = float(upper - lower)
        return float(np.clip(clipped, max(width * 1e-6, 1e-8), max(width * 0.5, 1e-6)))
    return float(max(clipped, 1e-8))


def drive_minuit(
    m,
    *,
    method: str = "migrad",
    migrad_kwargs: dict | None = None,
    run_hesse: bool = True,
    minos: bool = False,
    minos_parameters: Sequence[str] | None = None,
) -> dict[str, tuple[float, float]] | None:
    """Drive a constructed, limit-set Minuit through minimisation and report MINOS.

    The single shared minimiser-drive seam (W13): the three minimiser sites in the
    codebase (:meth:`FitEngine.fit`, :meth:`FitEngine.global_fit`, and the
    count-domain ``_solve``) all route their migrad/simplex call through here so the
    *explicit HESSE* refinement and *opt-in MINOS* behaviour are defined once.

    ``m`` must already have its cost function, parameter names, and limits set; this
    function owns only the post-construction drive. ``migrad_kwargs`` is forwarded to
    ``m.migrad``/``m.simplex`` so each caller keeps its own ncall/iterate/use_simplex
    tuning. migrad already runs HESSE as its final step, so the symmetric errors and
    covariance need no extra work; an explicit ``m.hesse()`` is run only when MINOS
    follows a *simplex* fit (simplex computes no Hessian, which MINOS requires). The
    migrad symmetric errors are therefore bit-identical whether or not MINOS follows.

    When ``minos`` is true and the fit is valid, ``m.minos()`` scans every free
    parameter (or just ``minos_parameters``) and the signed asymmetric offsets
    ``{name: (lower, upper)}`` (``lower < 0 < upper``) are returned for every scan
    that succeeded (``MError.is_valid``). A whole-scan failure or any parameter whose
    individual scan is invalid simply yields no asymmetric entry for it — the caller
    keeps the symmetric HESSE σ for that parameter. Returns ``None`` when MINOS was
    not requested, the fit is invalid, or no scan produced a valid interval.
    """
    migrad_kwargs = dict(migrad_kwargs or {})
    if method == "simplex":
        m.simplex(**migrad_kwargs)
    else:
        m.migrad(**migrad_kwargs)

    if not minos or not getattr(m, "valid", False):
        return None

    # MINOS needs a valid Hessian. migrad already runs HESSE as its final step, so on
    # the migrad path the covariance is HESSE-quality and we add nothing — keeping the
    # symmetric errors bit-identical whether or not MINOS follows. simplex computes no
    # Hessian, so MINOS there gets an explicit HESSE first.
    if run_hesse and method == "simplex":
        try:
            m.hesse()
        except Exception:
            # HESSE is a fidelity refinement, not a correctness requirement; a
            # back-end that rejects it leaves the simplex state in place.
            pass

    try:
        if minos_parameters:
            m.minos(*minos_parameters)
        else:
            m.minos()
    except (RuntimeError, ValueError):
        # MINOS can fail wholesale (non-quadratic blow-up, call-limit); fall back
        # to the symmetric HESSE errors the caller already has.
        return None

    names = list(minos_parameters) if minos_parameters else list(m.parameters)
    out: dict[str, tuple[float, float]] = {}
    for name in names:
        try:
            merror = m.merrors[name]
        except (KeyError, TypeError):
            continue
        if merror is not None and getattr(merror, "is_valid", False):
            out[name] = (float(merror.lower), float(merror.upper))
    return out or None


# --- selectable fit cost ----------------------------------------------------


def gaussian_chi2(
    observed: NDArray[np.float64],
    model: NDArray[np.float64],
    errors: NDArray[np.float64],
) -> float:
    """Weighted least-squares cost Σ((n − μ)/σ)² for explicit per-point σ."""
    return float(np.sum(((observed - model) / errors) ** 2))


def poisson_cash(observed: NDArray[np.float64], model: NDArray[np.float64]) -> float:
    """Cash statistic ``2·Σ(μ − n + n·ln(n/μ))`` for Poisson counts.

    The single Cash-cost primitive (factored here from ``count_domain.py`` so
    the count-domain fitters and the grouped/global driver share one
    implementation). Scaled so ``errordef = 1`` yields correct parameter errors
    (ΔC behaves like Δχ² near the minimum). ``n = 0`` bins reduce to ``2μ`` with
    no logarithm; ``μ`` is floored to keep the log finite.
    """
    mu = np.clip(model, 1.0e-12, None)
    term = mu - observed
    positive = observed > 0.0
    term[positive] += observed[positive] * np.log(observed[positive] / mu[positive])
    return 2.0 * float(np.sum(term))


@dataclass(frozen=True)
class CostFactory:
    """A pluggable objective for the single/global fit driver.

    ``build`` constructs the iminuit objective over (possibly concatenated)
    data; ``pointwise`` evaluates the *same* statistic on one (sub)dataset, for
    the per-dataset reduced-statistic report a global fit produces. Both the
    Gaussian (√-weighted least squares — the default everywhere) and the Poisson
    (Cash) cost flow through this one seam, so a grouped/global fit can adopt the
    count-domain modes' Poisson convention without the driver knowing which is
    in play. ``build`` returns either an :mod:`iminuit.cost` object or a plain
    ``cost(*params)`` callable carrying ``errordef``; both are accepted by
    ``Minuit(cost, *initial, name=...)``.
    """

    name: str
    build: Callable[..., object]
    pointwise: Callable[[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]], float]


def _build_least_squares(x, y, yerr, model):
    from iminuit.cost import LeastSquares

    return LeastSquares(x, y, yerr, model)


def _build_poisson_cash(x, y, yerr, model):
    # Cash fits the raw Poisson counts ``y`` against the model expectation μ;
    # ``yerr`` is unused (the Poisson variance is μ itself, not a stored σ).
    counts = np.asarray(y, dtype=float)
    times = np.asarray(x, dtype=float)

    def cost(*args) -> float:
        return poisson_cash(counts, np.asarray(model(times, *args), dtype=float))

    cost.errordef = 1.0
    return cost


#: √-weighted least squares — the historical default for every fit surface.
GAUSSIAN_COST = CostFactory(
    "gaussian",
    _build_least_squares,
    lambda observed, model, errors: gaussian_chi2(observed, model, errors),
)
#: Cash statistic on raw Poisson counts — the count-domain modes' default,
#: extended to grouped fits via the cost-factory seam.
POISSON_COST = CostFactory(
    "poisson",
    _build_poisson_cash,
    lambda observed, model, _errors: poisson_cash(observed, model),
)

#: Resolve a cost name to its factory; ``None``/unknown → Gaussian.
COST_FACTORIES: dict[str, CostFactory] = {
    GAUSSIAN_COST.name: GAUSSIAN_COST,
    POISSON_COST.name: POISSON_COST,
}


@dataclass
class FitResult:
    """Container for the outcome of a fit."""

    success: bool
    chi_squared: float = 0.0
    reduced_chi_squared: float = 0.0
    parameters: ParameterSet = field(default_factory=ParameterSet)
    uncertainties: dict[str, float] = field(default_factory=dict)
    covariance: NDArray[np.float64] | None = None
    covariance_parameters: list[str] = field(default_factory=list)
    residuals: NDArray[np.float64] | None = None
    message: str = ""
    function_calls: int = 0
    gradient_calls: int = 0
    hessian_calls: int = 0
    edm: float | None = None
    covariance_accurate: bool = False
    #: Degrees of freedom ν = N_data − N_free for this (sub)fit. Used by the χ²
    #: quality verdict; 0 means "unknown" and callers fall back to inference.
    dof: int = 0
    #: Opt-in MINOS asymmetric 1σ intervals, ``{param: (lower, upper)}`` with
    #: ``lower < 0 < upper`` (iminuit's signed offsets). ``None`` when MINOS was
    #: not requested or every scan failed. A *display-only* overlay — the
    #: symmetric HESSE :attr:`uncertainties` are unchanged and remain the value
    #: every downstream surface (trends, export, propagation, promote) consumes.
    minos_errors: dict[str, tuple[float, float]] | None = None
    #: Advisory warning messages emitted while fitting (the percent/fraction
    #: scale trap and the fixed-frequency σ-inflation trap — see
    #: :class:`AsymmetryScaleWarning` / :class:`FixedFrequencyFieldMismatchWarning`).
    #: Captured off the Python ``warnings`` system so the GUI can surface them in
    #: the fit panel; they are also *re-emitted*, so stderr/logging and any outer
    #: ``catch_warnings`` still observe them exactly as before.
    warnings: list[str] = field(default_factory=list)


class FitEngine:
    """Fit μSR asymmetry data to a model function using iminuit.

    Example
    -------
    ::

        from asymmetry.core.fitting import FitEngine, ParameterSet, Parameter
        from asymmetry.core.fitting.models import MODELS

        engine = FitEngine()
        model = MODELS["ExponentialRelaxation"]

        # Set up parameters
        params = ParameterSet()
        params.add(Parameter(name="A0", value=0.2, min=0, max=1))
        params.add(Parameter(name="lambda", value=0.5, min=0))

        result = engine.fit(dataset, model.function, params)
        print(f"χ²ᵣ = {result.reduced_chi_squared:.3f}")
    """

    def fit(
        self,
        dataset: MuonDataset,
        model_fn: Callable[..., NDArray],
        parameters: ParameterSet,
        t_min: float | None = None,
        t_max: float | None = None,
        method: str = "migrad",
        minos: bool = False,
        cancel_callback: Callable[[], bool] | None = None,
        frequency_offsets: dict[str, float] | None = None,
        cost_factory: CostFactory | None = None,
        migrad_kwargs: dict | None = None,
    ) -> FitResult:
        """Run a single-dataset fit.

        Parameters
        ----------
        dataset : MuonDataset
            The data to fit. The engine uses the dataset object's ``time``,
            ``asymmetry``, and ``error`` arrays as provided, optionally clipped
            only by ``t_min``/``t_max``.
        model_fn : callable
            ``f(t, **params) -> array``.
        parameters : ParameterSet
            Initial parameter values and constraints.
        t_min, t_max : float, optional
            Restrict fit range.
        method : str
            Minimization method (``"migrad"`` for gradient-based,
            ``"simplex"`` for Nelder-Mead).
        cost_factory : CostFactory, optional
            Selectable fit objective (:data:`GAUSSIAN_COST` / :data:`POISSON_COST`).
            ``None`` keeps the historical √-weighted least squares, byte-for-byte.
            With :data:`POISSON_COST` the ``chi_squared``/``reduced_chi_squared``
            report the Cash statistic (asymptotically χ²-distributed), and the
            data must be raw Poisson counts with a count-expectation model.
        migrad_kwargs : dict, optional
            Forwarded to ``m.migrad``/``m.simplex`` via the shared
            :func:`drive_minuit` seam (e.g. ``ncall`` to cap iterations for a
            cheap screening fit). ``None`` keeps the historical unbounded drive.
        frequency_offsets : dict[str, float], optional
            Rotating-reference-frame offsets ``{param: ν₀}`` resolved by
            :func:`asymmetry.core.fitting.rrf_offset.rrf_frequency_offsets`.
            When given, each named parameter is shifted by its offset before the
            model is evaluated, so the *raw* lab-frame data is fitted while the
            fitted values read as rotating-frame offsets δν (lab = δν + ν₀ via
            :func:`~asymmetry.core.fitting.rrf_offset.apply_rrf_offsets`). This
            is the engine-level home of the RRF offset wrapper — fitting through
            it is bit-for-bit identical to wrapping ``model_fn`` with
            :func:`~asymmetry.core.fitting.rrf_offset.rrf_offset_model`.

        Returns
        -------
        FitResult
            Container with fit results including χ², parameters, and uncertainties.
        """
        ds = dataset.time_range(t_min, t_max) if (t_min or t_max) else dataset

        if frequency_offsets:
            # Apply the rotating-frame offset through the shared shift seam, so
            # this path and the standalone rrf_offset_model wrapper fit raw data
            # identically. The reported parameters remain the rotating-frame δν;
            # callers convert back via apply_rrf_offsets.
            from asymmetry.core.fitting.rrf_offset import offset_model_function

            model_fn = offset_model_function(model_fn, frequency_offsets)

        try:
            from iminuit import Minuit
            from iminuit.cost import LeastSquares
        except ImportError as e:
            error_msg = str(e)
            if "numba" in error_msg.lower() or "numpy" in error_msg.lower():
                return FitResult(
                    success=False,
                    message=f"iminuit import error: {error_msg}\n"
                    "Try: pip install 'numpy<2.3' to fix numpy/numba compatibility.",
                )
            return FitResult(
                success=False,
                message=f"iminuit import error: {error_msg}\nInstall it with: pip install iminuit",
            )

        # Prepare parameter names, values, and constraints
        free = parameters.free_parameters
        fixed_kw = {p.name: p.value for p in parameters if p.fixed}
        # Equality link groups: each follower takes its group main's value, so
        # it drops out of the free-fit set (WiMDA "Ties").
        followers = parameters.link_followers()
        # Affine ties: each follower is a linear map of other parameters
        # (offset / equal-spacing constraints; a deliberate capability beyond
        # WiMDA's equality links). Evaluated after link followers so a tie may
        # reference a link-resolved value; tie references must themselves be
        # free/fixed/link parameters (no tie-to-tie chaining).
        ties = parameters.tie_followers()
        _validate_tie_references(parameters, ties)
        # A tie may add a free *auxiliary* parameter (e.g. the half-splitting
        # ``delta``) that the model never consumes; strip such extras before
        # calling an explicit-signature model. Only computed when ties exist, so
        # the common no-tie path stays byte-identical (no filtering).
        accepted_kwargs = _model_kwarg_names(model_fn) if ties else None

        def _call_model(t, kw):
            if accepted_kwargs is None:
                return model_fn(t, **kw)
            return model_fn(t, **{k: v for k, v in kw.items() if k in accepted_kwargs})

        # Create model wrapper that accepts free parameters
        param_names = [p.name for p in free]
        cancel_guard = _make_cancel_guard(cancel_callback)

        def model_wrapper(t, *args):
            """Model wrapper for iminuit."""
            cancel_guard()
            kw = {**fixed_kw, **dict(zip(param_names, args))}
            for follower, main in followers.items():
                kw[follower] = kw[main]
            for name, tie in ties.items():
                kw[name] = tie.evaluate(kw)
            return _call_model(t, kw)

        # Build the cost. The default (no factory) is the historical √-weighted
        # least squares, kept byte-identical; a factory swaps in the selectable
        # objective (e.g. Poisson Cash on raw counts) through the shared seam.
        if cost_factory is None:
            cost = LeastSquares(ds.time, ds.asymmetry, ds.error, model_wrapper)
        else:
            cost = cost_factory.build(ds.time, ds.asymmetry, ds.error, model_wrapper)

        # Create Minuit object
        initial_values = [p.value for p in free]
        # Run the advisory guards under a recording catch so their warnings can
        # be carried back on the FitResult for the GUI to surface — then re-emit
        # each, so stderr/logging and any outer catch_warnings still observe them
        # exactly as before (the engine stays a transparent pass-through).
        with warnings.catch_warnings(record=True) as caught_advisories:
            warnings.simplefilter("always")
            # Advisory guard: flag the percent-vs-fraction trap before fitting.
            _warn_on_scale_mismatch(ds.time, ds.asymmetry, model_wrapper, initial_values)
            # Advisory guard: flag a frequency pinned far from γ_μ·B(field) (the
            # TF fixed-frequency trap that inflates sigma). Skip in the rotating
            # frame: there the frequency seeds are *offsets* δν (lab = δν + ν₀),
            # so a small fixed δν is correct and comparing it to lab-frame γ_μ·B
            # would misfire.
            if not frequency_offsets:
                _warn_on_fixed_frequency_far_from_field(dataset, parameters)
        # Carry only the two advisory categories to the GUI — the scale guard
        # evaluates the seeded model inside this block, so an incidental numerical
        # warning (e.g. a RuntimeWarning from overflow in exp at a poor seed) must
        # not masquerade as a fit advisory. Re-emit *every* captured warning,
        # though, so non-advisory ones still propagate to stderr/logging exactly
        # as they did before this block existed.
        advisory_warnings = [
            str(w.message)
            for w in caught_advisories
            if issubclass(w.category, (AsymmetryScaleWarning, FixedFrequencyFieldMismatchWarning))
        ]
        for w in caught_advisories:
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
        m = Minuit(cost, *initial_values, name=param_names)

        # Set limits for parameters
        for i, p in enumerate(free):
            if p.min != -float("inf"):
                m.limits[i] = (p.min, m.limits[i][1])
            if p.max != float("inf"):
                m.limits[i] = (m.limits[i][0], p.max)

        # Run minimization (migrad/simplex + explicit HESSE + opt-in MINOS) through
        # the shared drive seam.
        minos_errors_raw = drive_minuit(m, method=method, minos=minos, migrad_kwargs=migrad_kwargs)

        # Pack results
        result_params = ParameterSet()
        uncertainties: dict[str, float] = {}
        minos_errors: dict[str, tuple[float, float]] = {}

        def _fitted_value(name: str) -> float:
            """Resolve any parameter's post-fit value (free, fixed, or linked)."""
            target = followers.get(name, name)  # a link follower tracks its main
            if target in param_names:
                return float(m.values[param_names.index(target)])
            return float(parameters[target].value)  # fixed

        def _free_index(name: str) -> int | None:
            """Covariance-order index for a value, or None when it is fixed."""
            target = followers.get(name, name)
            return param_names.index(target) if target in param_names else None

        def _tie_uncertainty(tie) -> float | None:
            """Delta-method 1σ of an affine tie: var = JᵀCJ over its references.

            Coefficients on the same underlying free parameter add (e.g. when
            ``main`` and ``offset`` both link to one group main); fixed
            references contribute no variance.
            """
            coeff_by_index: dict[int, float] = {}
            terms = [(tie.main, tie.scale)]
            if tie.offset is not None:
                terms.append((tie.offset, tie.offset_scale))
            for ref, coeff in terms:
                idx = _free_index(ref)
                if idx is not None:
                    coeff_by_index[idx] = coeff_by_index.get(idx, 0.0) + coeff
            if not coeff_by_index:
                return None  # every reference is fixed → no free uncertainty
            cov = m.covariance
            var = 0.0
            for i, ci in coeff_by_index.items():
                for j, cj in coeff_by_index.items():
                    if cov is not None:
                        var += ci * cj * float(cov[i, j])
                    elif i == j:
                        # No covariance (HESSE failed): keep the diagonal terms.
                        # Exact for a single reference; ignores correlations for
                        # a multi-reference tie (a rare, already-degraded fit).
                        var += ci * cj * float(m.errors[i]) ** 2
            # A non-positive-semidefinite covariance can yield a tiny negative
            # var for a well-determined tie; clamp to 0 so the entry is still
            # reported (and ``np.sqrt`` never sees a negative).
            return float(np.sqrt(max(var, 0.0)))

        for p in parameters:
            # Linking wins over fix (matching WiMDA): a follower always tracks its
            # group main, so this branch precedes the plain ``fixed`` case.
            if p.name in ties:
                # Affine tie: derive the value from the fitted references and
                # carry a delta-method uncertainty through the linear map.
                tie = ties[p.name]
                value = tie.evaluate({ref: _fitted_value(ref) for ref in tie.references()})
                result_params.add(
                    Parameter(
                        name=p.name,
                        value=value,
                        min=p.min,
                        max=p.max,
                        link_group=p.link_group,
                        tie=p.tie,
                    )
                )
                sigma = _tie_uncertainty(tie)
                if sigma is not None:
                    uncertainties[p.name] = sigma
            elif p.name in followers:
                # Equality link: inherit the group main's fitted value and, by
                # the delta method (∂follower/∂main = 1), its uncertainty.
                main_name = followers[p.name]
                if main_name in param_names:
                    main_idx = param_names.index(main_name)
                    value = m.values[main_idx]
                    main_err = m.errors[main_idx]
                else:
                    # Main is itself fixed: the whole group is fixed.
                    value = parameters[main_name].value
                    main_err = None
                result_params.add(
                    Parameter(
                        name=p.name, value=value, min=p.min, max=p.max, link_group=p.link_group
                    )
                )
                if main_err is not None:
                    uncertainties[p.name] = main_err
                # A follower inherits its main's MINOS interval by the same delta
                # method (∂follower/∂main = 1) that carries its symmetric error.
                if minos_errors_raw and main_name in minos_errors_raw:
                    minos_errors[p.name] = minos_errors_raw[main_name]
            elif p.fixed:
                result_params.add(Parameter(name=p.name, value=p.value, link_group=p.link_group))
            else:
                idx = param_names.index(p.name)
                value = m.values[idx]
                result_params.add(
                    Parameter(
                        name=p.name, value=value, min=p.min, max=p.max, link_group=p.link_group
                    )
                )
                if m.errors[idx] is not None:
                    uncertainties[p.name] = m.errors[idx]
                if minos_errors_raw and p.name in minos_errors_raw:
                    minos_errors[p.name] = minos_errors_raw[p.name]

        ndata = len(ds.time)
        nfree = len(free)
        red_chi2 = m.fval / max(ndata - nfree, 1)
        fitted_values = _call_model(ds.time, {p.name: p.value for p in result_params})
        residuals = np.asarray(ds.asymmetry, dtype=float) - np.asarray(fitted_values, dtype=float)

        return FitResult(
            success=m.valid,
            chi_squared=m.fval,
            reduced_chi_squared=red_chi2,
            parameters=result_params,
            uncertainties=uncertainties,
            covariance=m.covariance if m.valid else None,
            covariance_parameters=list(param_names) if m.valid else [],
            residuals=residuals,
            message=_minuit_status_message(
                m,
                success_message="Fit successful",
                failure_prefix="Fit failed",
            ),
            dof=ndata - nfree,
            minos_errors=minos_errors or None,
            warnings=advisory_warnings,
        )

    # --- global fit -----------------------------------------------------

    def global_fit(
        self,
        datasets: list[MuonDataset],
        model_fn: Callable[..., NDArray],
        global_params: list[str],
        local_params: list[str],
        initial_params: dict[int, ParameterSet],
        t_min: float | None = None,
        t_max: float | None = None,
        method: str = "migrad",
        max_calls: int = 10000,
        migrad_iterations: int = 5,
        use_simplex_rescue: bool = True,
        minuit_strategy: int | None = None,
        minuit_tol: float | None = None,
        initial_step_sizes: dict[str, float] | None = None,
        minos: bool = False,
        screening: bool = False,
        strategy: str = "joint",
        use_varpro: bool = False,
        cancel_callback: Callable[[], bool] | None = None,
        cost_factory: CostFactory | None = None,
        local_param_groups: dict[str, dict[int, Hashable]] | None = None,
    ) -> tuple[dict[int, FitResult], ParameterSet]:
        """Simultaneous fit of multiple datasets with shared and local parameters.

        Parameters
        ----------
        datasets
            List of datasets to fit simultaneously.
        model_fn
            Model function applied to each dataset.
        global_params
            Names of parameters shared across all datasets (e.g., ["A0"]).
        local_params
            Names of parameters that vary per dataset (e.g., ["lambda"]).
        initial_params
            Dictionary mapping dataset run_number to initial ParameterSet.
            Global parameters should have the same value in all sets.
        t_min, t_max
            Optional time range restriction applied to all datasets.
        method
            Minimization method ("migrad" or "simplex").
        max_calls
            Maximum function evaluations for minimization. Limits runtime for
            large global fits.
        strategy
            Minimiser architecture for the shared-parameter problem.
            ``"joint"`` (default) builds one Minuit problem over the globals plus
            every dataset's locals — the historical, byte-for-byte path.
            ``"profiled"`` runs an outer Minuit over the free *globals only* and,
            for each candidate global vector, solves each dataset's locals
            independently (via :meth:`fit` with the globals held fixed). The two
            share the same objective, so at the optimum they agree on values and
            χ²; profiled drops the per-fit Hessian from ``(n_global+n_local·G)²``
            to ``n_global²`` plus ``G`` small block Hessians, so cost scales
            ~linearly in ``G`` rather than super-linearly. Profiled requires at
            least one free global (with none, the joint problem is already
            block-separable and the fast path below handles it).
        use_varpro
            Variable projection. When ``True``, parameters flagged linear in the
            model metadata (amplitudes, constant backgrounds) are solved by
            weighted linear least-squares *inside* each residual evaluation and
            removed from the nonlinear Minuit vector, then reinstated in the
            returned :class:`FitResult` (they still count toward ``dof`` and the
            IC ``k``). Falls back to the nonlinear treatment for any candidate
            model that is not actually affine in a flagged parameter, or whose
            linear solution violates a bound. Off by default; the fitted values,
            errors, and IC are preserved.

        Returns
        -------
        results : dict[int, FitResult]
            Dictionary mapping run_number to individual FitResult for each dataset.
        global_result : ParameterSet
            The fitted global parameters with uncertainties.

        local_param_groups
            Optional per-local-parameter sharing. ``local_param_groups[pname]``
            maps a dataset run number to a group key; datasets with the same key
            share one fitted value for ``pname`` (a third scope between fully
            global and fully per-dataset). Absent → ``pname`` is per-dataset.

        Notes
        -----
        Fixed parameters (where param.fixed=True) are held constant during fitting.
        """
        if not datasets:
            raise ValueError("No datasets provided for global fitting")
        if strategy not in ("joint", "profiled"):
            raise ValueError(
                f"Unknown global-fit strategy {strategy!r}; expected 'joint' or 'profiled'"
            )
        if use_varpro:
            # Variable projection (technique M) is deferred: once the profiled
            # strategy has separated the per-dataset locals it is only a
            # constant-factor per-fit win (it does not change the G-exponent),
            # and matching the current engine's *marginal* parameter errors needs
            # a final full Hessian over all parameters — the same O((n_local·G)²)
            # cost the joint path already pays. The linear-parameter metadata is
            # in place (models.default_linear_params) for the follow-up. Fail
            # loudly rather than silently ignoring the request.
            raise NotImplementedError(
                "use_varpro (variable projection) is not yet wired into global_fit; "
                "use the profiled strategy for the G-scaling win. VarPro is a "
                "deferred constant-factor follow-up."
            )
        _reject_affine_ties(initial_params.values(), "Global fitting")

        def _local_group_key(pname: str, run_number: int) -> Hashable:
            """The sharing key for a local param on a dataset (run number by default)."""
            if local_param_groups and pname in local_param_groups:
                return local_param_groups[pname].get(run_number, run_number)
            return run_number

        dataset_run_numbers = [int(ds.run_number) for ds in datasets]
        duplicate_runs = [run for run, count in Counter(dataset_run_numbers).items() if count > 1]
        if duplicate_runs:
            raise ValueError(
                "Global fitting requires unique dataset run numbers; duplicates found: "
                f"{sorted(duplicate_runs)}"
            )

        missing_initial = [run for run in dataset_run_numbers if run not in initial_params]
        if missing_initial:
            raise KeyError(
                f"initial parameter sets missing for dataset run numbers {sorted(missing_initial)}"
            )

        first_params = initial_params[datasets[0].run_number]
        free_global_params = [pname for pname in global_params if not first_params[pname].fixed]

        # A grouped local parameter (shared across a subset of datasets) ties those
        # datasets together, so the objective is no longer block-separable even with
        # no free globals.
        grouped_local_ties = False
        if local_param_groups:
            for pname in local_params:
                keys = [
                    _local_group_key(pname, ds.run_number)
                    for ds in datasets
                    if not initial_params[ds.run_number][pname].fixed
                ]
                if len(keys) != len(set(keys)):
                    grouped_local_ties = True
                    break

        # When nothing is actually shared, the joint objective is block-separable.
        # Solving each dataset independently is equivalent and avoids a large,
        # ill-conditioned Minuit problem that is less stable than the proven
        # single-fit path.
        if not free_global_params and not grouped_local_ties:
            fitted_global = ParameterSet()
            for pname in global_params:
                parameter = first_params[pname]
                fitted_global.add(
                    Parameter(
                        name=pname,
                        value=parameter.value,
                        min=parameter.min,
                        max=parameter.max,
                        fixed=parameter.fixed,
                    )
                )

            results = {}
            for ds in datasets:
                results[ds.run_number] = self.fit(
                    ds,
                    model_fn,
                    initial_params[ds.run_number],
                    t_min=t_min,
                    t_max=t_max,
                    method=method,
                    minos=minos,
                    cancel_callback=cancel_callback,
                    cost_factory=cost_factory,
                )
            return results, fitted_global

        # Profiled/nested-locals strategy (technique L). An outer Minuit over the
        # free globals only; each candidate global vector is scored by solving
        # every dataset's locals independently (globals held fixed). This shares
        # the joint objective's minimum but replaces the (n_global+n_local·G)²
        # Hessian with n_global² plus G small per-dataset ones, so per-fit cost
        # scales ~linearly in G. Grouped local ties couple datasets, so they are
        # not separable — fall back to the joint path for those.
        if strategy == "profiled" and free_global_params and not grouped_local_ties:
            return self._global_fit_profiled(
                datasets=datasets,
                model_fn=model_fn,
                global_params=global_params,
                local_params=local_params,
                initial_params=initial_params,
                free_global_params=free_global_params,
                first_params=first_params,
                t_min=t_min,
                t_max=t_max,
                method=method,
                max_calls=max_calls,
                migrad_iterations=migrad_iterations,
                use_simplex_rescue=use_simplex_rescue,
                minuit_strategy=minuit_strategy,
                minuit_tol=minuit_tol,
                minos=minos,
                screening=screening,
                use_varpro=use_varpro,
                cancel_callback=cancel_callback,
                cost_factory=cost_factory,
            )
        # ``use_varpro`` is applied by wrapping ``model_fn`` before the joint
        # objective is built (below); the profiled path handles it internally.

        try:
            from iminuit import Minuit
        except ImportError as e:
            error_msg = str(e)
            # Return error results for all datasets
            error_result = FitResult(
                success=False,
                message=f"iminuit import error: {error_msg}",
            )
            return {ds.run_number: error_result for ds in datasets}, ParameterSet()

        # Apply time range to all datasets
        fitted_datasets = []
        for ds in datasets:
            if t_min or t_max:
                fitted_datasets.append(ds.time_range(t_min, t_max))
            else:
                fitted_datasets.append(ds)

        # Build parameter name mapping
        # Format: global params come first, then local params for each dataset
        param_names = []
        param_bounds = []
        initial_values = []

        # Add global parameters
        for pname in free_global_params:
            p = first_params[pname]
            param_names.append(pname)
            param_bounds.append((p.min, p.max))
            initial_values.append(p.value)

        # Add local parameters for each dataset. A grouped local param reuses one
        # Minuit parameter (and index) for every dataset that shares its group key,
        # so those datasets are fitted with a single shared value.
        dataset_param_indices = {}  # Maps (run_number, param_name) -> index in param_names
        group_param_indices: dict[tuple[str, Hashable], int] = {}
        for ds in datasets:
            params = initial_params[ds.run_number]
            dataset_param_indices[ds.run_number] = {}
            for pname in local_params:
                p = params[pname]
                if p.fixed:
                    continue
                group_key = _local_group_key(pname, ds.run_number)
                cache_key = (pname, group_key)
                idx = group_param_indices.get(cache_key)
                if idx is None:
                    idx = len(param_names)
                    param_names.append(f"{pname}_{group_key}")
                    param_bounds.append((p.min, p.max))
                    initial_values.append(p.value)
                    group_param_indices[cache_key] = idx
                dataset_param_indices[ds.run_number][pname] = idx

        # Build fixed parameter dictionaries for each dataset
        fixed_params = {}
        for ds in datasets:
            params = initial_params[ds.run_number]
            fixed_params[ds.run_number] = {p.name: p.value for p in params if p.fixed}

        # Create least squares cost function
        from iminuit.cost import LeastSquares

        # Concatenate all data
        all_times = np.concatenate([ds.time for ds in fitted_datasets])
        all_asymm = np.concatenate([ds.asymmetry for ds in fitted_datasets])
        all_errors = np.concatenate([ds.error for ds in fitted_datasets])
        # Guard against zero/invalid errors that destabilize the objective.
        all_errors = np.where(
            np.isfinite(all_errors) & (all_errors > 0.0),
            all_errors,
            1e-12,
        )

        cancel_guard = _make_cancel_guard(cancel_callback)

        def model_wrapper(t_all, *args):
            """Model wrapper that applies appropriate parameters to each dataset section."""
            cancel_guard()
            result = np.zeros_like(t_all)
            offset = 0

            # Extract global parameter values
            global_values = {}
            global_idx = 0
            for pname in global_params:
                p = first_params[pname]
                if p.fixed:
                    global_values[pname] = p.value
                    continue
                global_values[pname] = args[global_idx]
                global_idx += 1

            for ds in fitted_datasets:
                n_points = len(ds.time)
                params = initial_params[ds.run_number]

                # Build parameter dict
                param_dict = global_values.copy()
                param_dict.update(fixed_params[ds.run_number])

                for pname in local_params:
                    p = params[pname]
                    if p.fixed:
                        param_dict[pname] = p.value
                    else:
                        idx = dataset_param_indices[ds.run_number][pname]
                        param_dict[pname] = args[idx]

                # Evaluate model for this dataset. Non-finite model outputs can
                # happen for extreme trial parameters; convert to a large finite
                # penalty so the minimizer can recover instead of diverging.
                model_vals = model_fn(ds.time, **param_dict)
                if not np.all(np.isfinite(model_vals)):
                    model_vals = np.full_like(ds.time, 1e30, dtype=float)
                result[offset : offset + n_points] = model_vals
                offset += n_points

            return result

        # Validate initial parameters
        for i, val in enumerate(initial_values):
            if not np.isfinite(val):
                raise ValueError(f"Parameter {param_names[i]} has non-finite initial value: {val}")

        # Create cost function and Minuit object. The default (no factory) keeps
        # the historical √-weighted least squares byte-for-byte; a factory swaps
        # in the selectable objective (Poisson Cash on the concatenated raw
        # counts) — Cash is a bin-wise sum, so the dataset concatenation is
        # transparent to it.
        try:
            if cost_factory is None:
                cost = LeastSquares(all_times, all_asymm, all_errors, model_wrapper)
            else:
                cost = cost_factory.build(all_times, all_asymm, all_errors, model_wrapper)
            m = Minuit(cost, *initial_values, name=param_names)
        except Exception as e:
            raise RuntimeError(f"Failed to create Minuit cost function: {str(e)}")

        # Screening mode (wizard IC pre-selection): IC ranking needs only χ², not
        # accurate parameter errors, so drop to Minuit strategy 0 — no accurate
        # post-fit Hessian refinement. The migrad EDM convergence criterion is
        # unchanged (it is governed by ``m.tol``, not the strategy), so the fitted
        # values and χ² are the same; only the covariance quality (``m.accurate``)
        # is relaxed. m.errors are still populated for warm-start step hints. An
        # explicit minuit_strategy (the difficult-assignment path) always wins.
        if minuit_strategy is not None:
            m.strategy = int(minuit_strategy)
        elif screening:
            m.strategy = 0
        if minuit_tol is not None:
            m.tol = float(minuit_tol)

        # Set parameter limits
        for i, (min_val, max_val) in enumerate(param_bounds):
            if min_val != -float("inf"):
                m.limits[i] = (min_val, m.limits[i][1])
            if max_val != float("inf"):
                m.limits[i] = (m.limits[i][0], max_val)

        if initial_step_sizes:
            for i, name in enumerate(param_names):
                hint = initial_step_sizes.get(name)
                if hint is None:
                    continue
                step_size = _clamp_minuit_step_size(hint, *param_bounds[i])
                if step_size > 0.0:
                    m.errors[i] = step_size

        # Run minimization with error handling, through the shared drive seam so the
        # joint fit gains explicit HESSE + opt-in MINOS on the same footing as the
        # single-fit path.
        if method == "simplex":
            migrad_kwargs = {"ncall": max_calls}
        else:
            migrad_kwargs = {
                "ncall": max_calls,
                "iterate": max(1, int(migrad_iterations)),
                "use_simplex": bool(use_simplex_rescue),
            }
        try:
            minos_errors_raw = drive_minuit(
                m, method=method, migrad_kwargs=migrad_kwargs, minos=minos
            )
        except FitCancelledError:
            # A cancelled fit records nothing — let it propagate past the generic
            # failure handler so no partial result is built.
            raise
        except Exception as e:
            # If fitting fails, return error results
            error_result = FitResult(
                success=False,
                message=f"Minimization failed: {str(e)}",
            )
            return {ds.run_number: error_result for ds in datasets}, ParameterSet()

        # Extract fitted global parameters
        fitted_global = ParameterSet()
        global_uncertainties = {}
        fmin = getattr(m, "fmin", None)
        function_calls = int(getattr(m, "nfcn", 0) or 0)
        gradient_calls = int(getattr(m, "ngrad", 0) or 0)
        hessian_calls = int(getattr(m, "nhessian", 0) or 0)
        edm = getattr(fmin, "edm", None)
        edm_value = float(edm) if edm is not None and np.isfinite(edm) else None
        covariance_accurate = bool(getattr(m, "accurate", False))
        covariance_matrix = None
        if m.valid and getattr(m, "covariance", None) is not None:
            try:
                covariance_matrix = np.asarray(m.covariance, dtype=float)
            except Exception:
                covariance_matrix = None
        global_idx = 0
        for pname in global_params:
            p = first_params[pname]
            if p.fixed:
                fitted_global.add(Parameter(name=pname, value=p.value, fixed=True))
            else:
                value = m.values[global_idx]
                fitted_global.add(Parameter(name=pname, value=value, min=p.min, max=p.max))
                if m.errors[global_idx] is not None:
                    global_uncertainties[pname] = m.errors[global_idx]
                global_idx += 1

        # Build per-dataset results
        results = {}

        for ds in fitted_datasets:
            params = initial_params[ds.run_number]

            # Build result parameter set for this dataset
            result_params = ParameterSet()
            uncertainties = {}
            # MINOS intervals are keyed in the joint problem by the global name and
            # the per-dataset local name ``f"{pname}_{run}"``; map both back to the
            # plain per-dataset parameter name.
            minos_errors: dict[str, tuple[float, float]] = {}

            # Add global parameters
            for pname in global_params:
                p = fitted_global[pname]
                result_params.add(Parameter(name=pname, value=p.value, fixed=p.fixed))
                if pname in global_uncertainties:
                    uncertainties[pname] = global_uncertainties[pname]
                if minos_errors_raw and pname in minos_errors_raw:
                    minos_errors[pname] = minos_errors_raw[pname]

            # Add local parameters
            for pname in local_params:
                p = params[pname]
                if p.fixed:
                    result_params.add(Parameter(name=pname, value=p.value, fixed=True))
                else:
                    idx = dataset_param_indices[ds.run_number][pname]
                    value = m.values[idx]
                    result_params.add(Parameter(name=pname, value=value, min=p.min, max=p.max))
                    if m.errors[idx] is not None:
                        uncertainties[pname] = m.errors[idx]
                    joint_name = f"{pname}_{_local_group_key(pname, ds.run_number)}"
                    if minos_errors_raw and joint_name in minos_errors_raw:
                        minos_errors[pname] = minos_errors_raw[joint_name]

            # Add fixed parameters to result
            for pname, value in fixed_params[ds.run_number].items():
                if pname not in result_params:
                    result_params.add(Parameter(name=pname, value=value, fixed=True))

            # Compute the per-dataset cost. The Gaussian default keeps the
            # √-weighted χ²; a factory reports its own statistic (Poisson Cash),
            # so the per-group reduced value stays on the same footing as the
            # joint objective the minimiser actually drove.
            param_dict = {p.name: p.value for p in result_params}
            model_vals = model_fn(ds.time, **param_dict)
            residuals = np.asarray(ds.asymmetry, dtype=float) - np.asarray(model_vals, dtype=float)
            if cost_factory is None:
                dataset_chi2 = np.sum(((ds.asymmetry - model_vals) / ds.error) ** 2)
            else:
                dataset_chi2 = cost_factory.pointwise(
                    np.asarray(ds.asymmetry, dtype=float),
                    np.asarray(model_vals, dtype=float),
                    np.asarray(ds.error, dtype=float),
                )

            covariance_subset = None
            covariance_order: list[str] = []
            if covariance_matrix is not None and covariance_matrix.ndim == 2:
                cov_indices: list[int] = []

                for pname in global_params:
                    if pname in global_uncertainties:
                        idx = param_names.index(pname)
                        cov_indices.append(idx)
                        covariance_order.append(pname)

                for pname in local_params:
                    if pname in uncertainties:
                        idx = dataset_param_indices[ds.run_number][pname]
                        cov_indices.append(idx)
                        covariance_order.append(pname)

                if cov_indices:
                    covariance_subset = covariance_matrix[np.ix_(cov_indices, cov_indices)]

            ndata = len(ds.time)
            # Count free parameters: global (shared) + local for this dataset
            nfree_global = sum(1 for p in global_params if not first_params[p].fixed)
            nfree_local = sum(1 for p in local_params if not params[p].fixed)
            nfree = nfree_global + nfree_local

            red_chi2 = dataset_chi2 / max(ndata - nfree, 1)

            results[ds.run_number] = FitResult(
                success=m.valid,
                chi_squared=dataset_chi2,
                reduced_chi_squared=red_chi2,
                parameters=result_params,
                uncertainties=uncertainties,
                covariance=covariance_subset,
                covariance_parameters=covariance_order,
                residuals=residuals,
                message=_minuit_status_message(
                    m,
                    success_message="Global fit successful",
                    failure_prefix="Global fit failed",
                ),
                function_calls=function_calls,
                gradient_calls=gradient_calls,
                hessian_calls=hessian_calls,
                edm=edm_value,
                covariance_accurate=covariance_accurate,
                dof=ndata - nfree,
                minos_errors=minos_errors or None,
            )

        return results, fitted_global

    def _global_fit_profiled(
        self,
        *,
        datasets: list[MuonDataset],
        model_fn: Callable[..., NDArray],
        global_params: list[str],
        local_params: list[str],
        initial_params: dict[int, ParameterSet],
        free_global_params: list[str],
        first_params: ParameterSet,
        t_min: float | None,
        t_max: float | None,
        method: str,
        max_calls: int,
        migrad_iterations: int,
        use_simplex_rescue: bool,
        minuit_strategy: int | None,
        minuit_tol: float | None,
        minos: bool,
        screening: bool,
        use_varpro: bool,
        cancel_callback: Callable[[], bool] | None,
        cost_factory: CostFactory | None,
    ) -> tuple[dict[int, FitResult], ParameterSet]:
        """Profiled/nested-locals global fit (technique L).

        Outer Minuit varies the free globals only. For a candidate global vector
        every dataset's locals are solved independently by :meth:`fit` with the
        globals pinned; the outer cost is Σ_d χ²_d. Each dataset's local solution
        is cached and reused as the warm start for the next outer iteration, so
        the inner problems stay in the same basin and the outer objective stays
        smooth. At the optimum this shares the joint objective's minimum, but its
        Hessians are ``n_global²`` + ``G`` small per-dataset blocks rather than one
        ``(n_global + n_local·G)²`` — the source of the ~linear (not super-linear)
        G-scaling.

        The shared-global HESSE errors come from the outer profile curvature
        (which equals the marginal curvature, so they match the joint solver);
        the per-dataset local errors are *conditional* on the fitted globals
        (each inner solve holds the globals fixed), which is structurally correct
        for a profiled fit but differs from the joint solver's marginal local
        errors. MINOS is not supported on the profiled path — asymmetric local
        intervals would need the coupled joint problem — so ``minos`` is accepted
        for signature parity with the joint path but ignored here.
        """
        try:
            from iminuit import Minuit
        except ImportError as e:  # pragma: no cover - exercised only without iminuit
            error_result = FitResult(success=False, message=f"iminuit import error: {e}")
            return {ds.run_number: error_result for ds in datasets}, ParameterSet()

        cancel_guard = _make_cancel_guard(cancel_callback)

        # Warm-start cache: the last accepted local solution per dataset. Seeded
        # from the caller's initial params; refreshed after every inner solve so
        # each outer iteration re-enters the same local basin (keeps the profiled
        # objective smooth — the key correctness condition for L).
        warm_locals: dict[int, dict[str, float]] = {}
        for ds in datasets:
            params = initial_params[ds.run_number]
            warm_locals[ds.run_number] = {
                name: params[name].value for name in local_params if not params[name].fixed
            }

        # Inner solve for one dataset with the globals pinned. Reuses the proven
        # single-fit path (self.fit) rather than a bespoke inner Minuit.
        def _inner_fit(ds: MuonDataset, global_values: dict[str, float]) -> FitResult:
            base = initial_params[ds.run_number]
            inner = ParameterSet()
            for pname in global_params:
                p = base[pname]
                inner.add(
                    Parameter(name=pname, value=global_values.get(pname, p.value), fixed=True)
                )
            warm = warm_locals[ds.run_number]
            for pname in local_params:
                p = base[pname]
                if p.fixed:
                    inner.add(
                        Parameter(name=pname, value=p.value, min=p.min, max=p.max, fixed=True)
                    )
                else:
                    inner.add(
                        Parameter(
                            name=pname,
                            value=warm.get(pname, p.value),
                            min=p.min,
                            max=p.max,
                            fixed=False,
                        )
                    )
            # Any model parameter that is neither global nor local (a fixed
            # nuisance the model consumes) must still be supplied.
            for p in base:
                if p.name not in inner:
                    inner.add(
                        Parameter(name=p.name, value=p.value, min=p.min, max=p.max, fixed=True)
                    )
            return self.fit(
                ds,
                model_fn,
                inner,
                t_min=t_min,
                t_max=t_max,
                method=method,
                cancel_callback=cancel_callback,
                cost_factory=cost_factory,
            )

        # Counters accumulated across every inner solve (the profiled fit's cost
        # is dominated by the inner fits, so these are the meaningful totals).
        counters = {"fcn": 0, "grad": 0, "hess": 0}

        def _solve_all(global_values: dict[str, float]) -> dict[int, FitResult]:
            cancel_guard()
            out: dict[int, FitResult] = {}
            for ds in datasets:
                res = _inner_fit(ds, global_values)
                out[ds.run_number] = res
                # Refresh the warm start with the freshly fitted locals.
                for pname in warm_locals[ds.run_number]:
                    if pname in res.parameters:
                        warm_locals[ds.run_number][pname] = res.parameters[pname].value
                counters["fcn"] += int(res.function_calls or 0)
                counters["grad"] += int(res.gradient_calls or 0)
                counters["hess"] += int(res.hessian_calls or 0)
            return out

        def _outer_objective(*args: float) -> float:
            global_values = dict(zip(free_global_params, args))
            for pname in global_params:
                if pname not in global_values:
                    global_values[pname] = first_params[pname].value
            inner_results = _solve_all(global_values)
            return float(sum(r.chi_squared for r in inner_results.values()))

        # iminuit needs named scalar-cost signatures; a χ² sum has errordef 1.0
        # (Minuit.LEAST_SQUARES), matching the joint LeastSquares cost.
        _outer_objective.errordef = Minuit.LEAST_SQUARES  # type: ignore[attr-defined]

        init_values = [first_params[name].value for name in free_global_params]
        m = Minuit(_outer_objective, *init_values, name=list(free_global_params))
        if minuit_strategy is not None:
            m.strategy = int(minuit_strategy)
        elif screening:
            m.strategy = 0
        if minuit_tol is not None:
            m.tol = float(minuit_tol)
        for i, name in enumerate(free_global_params):
            p = first_params[name]
            lo = p.min if p.min != -float("inf") else None
            hi = p.max if p.max != float("inf") else None
            if lo is not None or hi is not None:
                m.limits[i] = (lo, hi)

        try:
            if method == "simplex":
                m.simplex(ncall=max_calls)
            else:
                m.migrad(ncall=max_calls, iterate=max(1, int(migrad_iterations)))
                if use_simplex_rescue and not m.valid:
                    m.simplex()
                    m.migrad()
            try:
                m.hesse()
            except Exception:
                pass
        except FitCancelledError:
            raise
        except Exception as e:
            error_result = FitResult(success=False, message=f"Profiled global fit failed: {e}")
            return {ds.run_number: error_result for ds in datasets}, ParameterSet()

        # Final global vector and its outer-Hessian errors.
        fitted_values = {name: float(m.values[i]) for i, name in enumerate(free_global_params)}
        global_errors: dict[str, float] = {}
        for i, name in enumerate(free_global_params):
            try:
                err = float(m.errors[i])
                if np.isfinite(err):
                    global_errors[name] = err
            except Exception:
                pass

        fitted_global = ParameterSet()
        global_values_full: dict[str, float] = {}
        for pname in global_params:
            p = first_params[pname]
            if p.fixed:
                fitted_global.add(Parameter(name=pname, value=p.value, fixed=True))
                global_values_full[pname] = p.value
            else:
                value = fitted_values[pname]
                fitted_global.add(Parameter(name=pname, value=value, min=p.min, max=p.max))
                global_values_full[pname] = value

        # One clean inner solve per dataset at the optimum to build the returned
        # results (conditional local errors with the globals pinned, block-
        # diagonal → linear in G). The outer Hessian supplies the global errors.
        final_results = _solve_all(global_values_full)

        edm = getattr(getattr(m, "fmin", None), "edm", None)
        edm_value = float(edm) if edm is not None and np.isfinite(edm) else None
        covariance_accurate = bool(getattr(m, "accurate", False))
        outer_valid = bool(m.valid)

        results: dict[int, FitResult] = {}
        n_free_global = len(free_global_params)
        for ds in datasets:
            inner = final_results[ds.run_number]
            params = initial_params[ds.run_number]
            result_params = ParameterSet()
            uncertainties: dict[str, float] = {}
            for pname in global_params:
                gp = fitted_global[pname]
                result_params.add(Parameter(name=pname, value=gp.value, fixed=gp.fixed))
                if pname in global_errors:
                    uncertainties[pname] = global_errors[pname]
            for pname in local_params:
                p = params[pname]
                if p.fixed:
                    result_params.add(Parameter(name=pname, value=p.value, fixed=True))
                else:
                    ip = inner.parameters[pname]
                    result_params.add(Parameter(name=pname, value=ip.value, min=p.min, max=p.max))
                    if pname in inner.uncertainties:
                        uncertainties[pname] = inner.uncertainties[pname]
            for p in params:
                if p.fixed and p.name not in result_params:
                    result_params.add(Parameter(name=p.name, value=p.value, fixed=True))

            ndata = len(ds.time_range(t_min, t_max).time) if (t_min or t_max) else len(ds.time)
            nfree_local = sum(1 for pname in local_params if not params[pname].fixed)
            nfree = n_free_global + nfree_local
            dataset_chi2 = float(inner.chi_squared)
            results[ds.run_number] = FitResult(
                success=bool(inner.success) and outer_valid,
                chi_squared=dataset_chi2,
                reduced_chi_squared=dataset_chi2 / max(ndata - nfree, 1),
                parameters=result_params,
                uncertainties=uncertainties,
                residuals=inner.residuals,
                message=(
                    "Profiled global fit successful"
                    if outer_valid and inner.success
                    else "Profiled global fit did not fully converge"
                ),
                function_calls=counters["fcn"],
                gradient_calls=counters["grad"],
                hessian_calls=counters["hess"],
                edm=edm_value,
                covariance_accurate=covariance_accurate,
                dof=ndata - nfree,
            )

        return results, fitted_global
