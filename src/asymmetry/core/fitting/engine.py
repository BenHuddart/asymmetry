"""Fit driver: single-run and global (simultaneous) fitting.

Uses :mod:`iminuit` as the fitting back-end, providing robust minimization
without scipy dependencies (important for Python 3.13+ compatibility).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.parameters import Parameter, ParameterSet


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
    tuning. An explicit ``m.hesse()`` is run after a valid minimisation (migrad's
    EDM-time covariance is approximate; HESSE improves covariance fidelity and gives
    MINOS an accurate starting point).

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

    if run_hesse and getattr(m, "valid", False):
        try:
            m.hesse()
        except Exception:
            # HESSE is a fidelity refinement, not a correctness requirement; a
            # back-end that rejects it leaves the migrad covariance in place.
            pass

    if not minos or not getattr(m, "valid", False):
        return None

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

        Returns
        -------
        FitResult
            Container with fit results including χ², parameters, and uncertainties.
        """
        ds = dataset.time_range(t_min, t_max) if (t_min or t_max) else dataset

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

        # Create model wrapper that accepts free parameters
        param_names = [p.name for p in free]
        cancel_guard = _make_cancel_guard(cancel_callback)

        def model_wrapper(t, *args):
            """Model wrapper for iminuit."""
            cancel_guard()
            kw = {**fixed_kw, **dict(zip(param_names, args))}
            for follower, main in followers.items():
                kw[follower] = kw[main]
            return model_fn(t, **kw)

        # Create least squares cost function
        cost = LeastSquares(ds.time, ds.asymmetry, ds.error, model_wrapper)

        # Create Minuit object
        initial_values = [p.value for p in free]
        m = Minuit(cost, *initial_values, name=param_names)

        # Set limits for parameters
        for i, p in enumerate(free):
            if p.min != -float("inf"):
                m.limits[i] = (p.min, m.limits[i][1])
            if p.max != float("inf"):
                m.limits[i] = (m.limits[i][0], p.max)

        # Run minimization (migrad/simplex + explicit HESSE + opt-in MINOS) through
        # the shared drive seam.
        minos_errors_raw = drive_minuit(m, method=method, minos=minos)

        # Pack results
        result_params = ParameterSet()
        uncertainties: dict[str, float] = {}
        minos_errors: dict[str, tuple[float, float]] = {}

        for p in parameters:
            # Linking wins over fix (matching WiMDA): a follower always tracks its
            # group main, so this branch precedes the plain ``fixed`` case.
            if p.name in followers:
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
        fitted_values = model_fn(ds.time, **{p.name: p.value for p in result_params})
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
        )

    # --- global fit -----------------------------------------------------

    def global_fit(
        self,
        datasets: list[MuonDataset],
        model_fn: Callable[..., NDArray],
        global_params: list[str],
        local_params: list[str],
        initial_params: dict[str, ParameterSet],
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
        cancel_callback: Callable[[], bool] | None = None,
    ) -> tuple[dict[str, FitResult], ParameterSet]:
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

        Returns
        -------
        results : dict[int, FitResult]
            Dictionary mapping run_number to individual FitResult for each dataset.
        global_result : ParameterSet
            The fitted global parameters with uncertainties.

        Notes
        -----
        Fixed parameters (where param.fixed=True) are held constant during fitting.
        """
        if not datasets:
            raise ValueError("No datasets provided for global fitting")

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

        # When nothing is actually shared, the joint objective is block-separable.
        # Solving each dataset independently is equivalent and avoids a large,
        # ill-conditioned Minuit problem that is less stable than the proven
        # single-fit path.
        if not free_global_params:
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
                )
            return results, fitted_global

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

        # Add local parameters for each dataset
        dataset_param_indices = {}  # Maps (run_number, param_name) -> index in param_names
        for ds in datasets:
            params = initial_params[ds.run_number]
            dataset_param_indices[ds.run_number] = {}
            for pname in local_params:
                p = params[pname]
                if not p.fixed:
                    idx = len(param_names)
                    param_names.append(f"{pname}_{ds.run_number}")
                    param_bounds.append((p.min, p.max))
                    initial_values.append(p.value)
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

        # Create cost function and Minuit object
        try:
            cost = LeastSquares(all_times, all_asymm, all_errors, model_wrapper)
            m = Minuit(cost, *initial_values, name=param_names)
        except Exception as e:
            raise RuntimeError(f"Failed to create Minuit cost function: {str(e)}")

        if minuit_strategy is not None:
            m.strategy = int(minuit_strategy)
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
                    joint_name = f"{pname}_{ds.run_number}"
                    if minos_errors_raw and joint_name in minos_errors_raw:
                        minos_errors[pname] = minos_errors_raw[joint_name]

            # Add fixed parameters to result
            for pname, value in fixed_params[ds.run_number].items():
                if pname not in result_params:
                    result_params.add(Parameter(name=pname, value=value, fixed=True))

            # Compute chi-squared for this dataset
            param_dict = {p.name: p.value for p in result_params}
            model_vals = model_fn(ds.time, **param_dict)
            residuals = np.asarray(ds.asymmetry, dtype=float) - np.asarray(model_vals, dtype=float)
            dataset_chi2 = np.sum(((ds.asymmetry - model_vals) / ds.error) ** 2)

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
