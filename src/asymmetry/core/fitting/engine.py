"""Fit driver: single-run and global (simultaneous) fitting.

Uses :mod:`iminuit` as the fitting back-end, providing robust minimization
without scipy dependencies (important for Python 3.13+ compatibility).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.parameters import Parameter, ParameterSet


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


@dataclass
class FitResult:
    """Container for the outcome of a fit."""

    success: bool
    chi_squared: float = 0.0
    reduced_chi_squared: float = 0.0
    parameters: ParameterSet = field(default_factory=ParameterSet)
    uncertainties: dict[str, float] = field(default_factory=dict)
    covariance: NDArray[np.float64] | None = None
    residuals: NDArray[np.float64] | None = None
    message: str = ""
    function_calls: int = 0
    gradient_calls: int = 0
    hessian_calls: int = 0
    edm: float | None = None
    covariance_accurate: bool = False


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

        # Create model wrapper that accepts free parameters
        param_names = [p.name for p in free]

        def model_wrapper(t, *args):
            """Model wrapper for iminuit."""
            kw = {**fixed_kw, **dict(zip(param_names, args))}
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

        # Run minimization
        if method == "simplex":
            m.simplex()
        else:
            m.migrad()

        # Pack results
        result_params = ParameterSet()
        uncertainties: dict[str, float] = {}

        for p in parameters:
            if p.fixed:
                result_params.add(Parameter(name=p.name, value=p.value))
            else:
                idx = param_names.index(p.name)
                value = m.values[idx]
                result_params.add(Parameter(name=p.name, value=value, min=p.min, max=p.max))
                if m.errors[idx] is not None:
                    uncertainties[p.name] = m.errors[idx]

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
            residuals=residuals,
            message=_minuit_status_message(
                m,
                success_message="Fit successful",
                failure_prefix="Fit failed",
            ),
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

        # Get first dataset's parameters as template for global params
        first_params = initial_params[datasets[0].run_number]

        # Add global parameters
        for pname in global_params:
            p = first_params[pname]
            if not p.fixed:
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

        def model_wrapper(t_all, *args):
            """Model wrapper that applies appropriate parameters to each dataset section."""
            result = np.zeros_like(t_all)
            offset = 0

            # Extract global parameter values
            global_values = {}
            global_idx = 0
            for pname in global_params:
                p = first_params[pname]
                if p.fixed:
                    global_values[pname] = p.value
                else:
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

        # Run minimization with error handling
        try:
            if method == "simplex":
                m.simplex(ncall=max_calls)
            else:
                m.migrad(
                    ncall=max_calls,
                    iterate=max(1, int(migrad_iterations)),
                    use_simplex=bool(use_simplex_rescue),
                )
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

            # Add global parameters
            for pname in global_params:
                p = fitted_global[pname]
                result_params.add(Parameter(name=pname, value=p.value, fixed=p.fixed))
                if pname in global_uncertainties:
                    uncertainties[pname] = global_uncertainties[pname]

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

            # Add fixed parameters to result
            for pname, value in fixed_params[ds.run_number].items():
                if pname not in result_params:
                    result_params.add(Parameter(name=pname, value=value, fixed=True))

            # Compute chi-squared for this dataset
            param_dict = {p.name: p.value for p in result_params}
            model_vals = model_fn(ds.time, **param_dict)
            residuals = np.asarray(ds.asymmetry, dtype=float) - np.asarray(model_vals, dtype=float)
            dataset_chi2 = np.sum(((ds.asymmetry - model_vals) / ds.error) ** 2)

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
            )

        return results, fitted_global
