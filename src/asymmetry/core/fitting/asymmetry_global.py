"""Asymmetry-domain global (shared-parameter) fitting.

This module adds a first-class, discoverable entry point for fitting one model
across several **asymmetry traces** simultaneously — sharing chosen "global"
parameters while keeping other "local" parameters per-dataset — working directly
on each dataset's ``.time`` / ``.asymmetry`` / ``.error`` arrays with a Gaussian
(weighted least-squares) cost.

It is a thin wrapper over :meth:`asymmetry.core.fitting.engine.FitEngine.global_fit`,
which already performs the simultaneous least-squares (it concatenates the
asymmetry traces, builds an iminuit ``LeastSquares`` cost, shares the named
globals across datasets, and gives every local parameter an independent copy per
dataset). The count-domain :func:`~asymmetry.core.fitting.grouped_time_domain.fit_grouped_series`
family wraps the *same* engine method with a Cash/Poisson cost on grouped counts;
this module is its asymmetry-domain, least-squares sibling. See
``docs/porting/asymmetry-domain-global-fit/`` for the study and the
asymmetry-vs-count-domain trade-off (when to use which).

The wrapper exists because the bare engine method has two discoverability traps
for the asymmetry-domain user:

* it keys datasets by ``MuonDataset.run_number``, which falls back to
  ``metadata["run_number"]`` defaulting to ``0`` — several runless asymmetry
  datasets silently collide on key ``0`` and the fit raises; and
* it returns a bare ``tuple[dict, ParameterSet]`` rather than a result bundle
  with the shared globals, per-dataset locals, and a combined reduced χ².

:func:`fit_global` removes both: it accepts a plain sequence (or mapping) of
asymmetry datasets, keys them positionally (or by the caller's keys), assigns
throwaway synthetic run numbers internally, and returns a :class:`GlobalFitResult`.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.engine import FitEngine, FitResult
from asymmetry.core.fitting.parameters import ParameterSet


@dataclass
class GlobalFitResult:
    """Result bundle for an asymmetry-domain global (shared-parameter) fit.

    ``dataset_results`` is keyed by the caller's dataset keys (positional indices
    for a sequence input, or the mapping keys for a mapping input). Each
    :class:`~asymmetry.core.fitting.engine.FitResult` carries that dataset's
    fitted globals *and* locals (with uncertainties and its own χ²/dof), so a
    caller can read either the shared physics or a single dataset's nuisances
    from one place.
    """

    success: bool
    #: The shared global parameters with their fitted values.
    global_parameters: ParameterSet
    #: 1σ (HESSE) uncertainties on the free shared globals, ``{name: sigma}``.
    global_uncertainties: dict[str, float] = field(default_factory=dict)
    #: Per-dataset results keyed by the caller's dataset key.
    dataset_results: dict[Hashable, FitResult] = field(default_factory=dict)
    #: Combined cost Σ_d χ²_d across every dataset (Gaussian weighted LSQ).
    chi_squared: float = 0.0
    #: Combined degrees of freedom ΣN_d − N_free_global − Σ_d N_free_local_d.
    dof: int = 0
    #: Combined reduced χ² = :attr:`chi_squared` / max(:attr:`dof`, 1).
    reduced_chi_squared: float = 0.0
    message: str = ""


def _normalize_datasets(
    datasets: Sequence[MuonDataset] | Mapping[Hashable, MuonDataset],
) -> list[tuple[Hashable, MuonDataset]]:
    """Return ``(caller_key, dataset)`` pairs for a sequence or mapping input."""
    if isinstance(datasets, Mapping):
        pairs = list(datasets.items())
    else:
        pairs = list(enumerate(datasets))
    if not pairs:
        raise ValueError("Global fitting requires at least one dataset")
    for key, ds in pairs:
        if not isinstance(ds, MuonDataset):
            raise TypeError(f"Dataset for key {key!r} is not a MuonDataset: {type(ds).__name__}")
    return pairs


def _resolve_initial_params(
    initial_params: ParameterSet | Mapping[Hashable, ParameterSet],
    keys: list[Hashable],
) -> dict[Hashable, ParameterSet]:
    """Map every caller key to a ``ParameterSet``.

    A single :class:`ParameterSet` is broadcast to every dataset (the common case
    — one seed structure shared across the series); a mapping must provide a set
    for every dataset key.
    """
    if isinstance(initial_params, ParameterSet):
        return {key: initial_params for key in keys}
    if isinstance(initial_params, Mapping):
        missing = [key for key in keys if key not in initial_params]
        if missing:
            raise KeyError(f"initial_params is missing entries for dataset keys {missing!r}")
        return {key: initial_params[key] for key in keys}
    raise TypeError(
        "initial_params must be a ParameterSet (broadcast to all datasets) or a "
        f"mapping of dataset key -> ParameterSet, got {type(initial_params).__name__}"
    )


def _validate_parameter_partition(
    global_params: Sequence[str],
    local_params: Sequence[str],
    initial_by_key: dict[Hashable, ParameterSet],
) -> None:
    """Validate the global/local partition against every dataset's parameter set."""
    if not global_params and not local_params:
        raise ValueError("Global fitting needs at least one global or local parameter")

    overlapping = set(global_params) & set(local_params)
    if overlapping:
        raise ValueError(f"Global and local parameters overlap: {sorted(overlapping)}")

    dup_global = _duplicates(global_params)
    if dup_global:
        raise ValueError(f"Duplicate global parameter names: {sorted(dup_global)}")
    dup_local = _duplicates(local_params)
    if dup_local:
        raise ValueError(f"Duplicate local parameter names: {sorted(dup_local)}")

    required = list(global_params) + list(local_params)
    for key, param_set in initial_by_key.items():
        missing = [name for name in required if name not in param_set]
        if missing:
            raise ValueError(
                f"Initial parameters for dataset {key!r} are missing referenced "
                f"global/local names: {missing}"
            )


def _duplicates(names: Sequence[str]) -> set[str]:
    seen: set[str] = set()
    dups: set[str] = set()
    for name in names:
        if name in seen:
            dups.add(name)
        seen.add(name)
    return dups


def _validate_errors(key: Hashable, dataset: MuonDataset) -> None:
    """Reject datasets whose per-point errors are absent, non-finite, or non-positive."""
    error = np.asarray(dataset.error, dtype=float)
    if error.size == 0:
        raise ValueError(f"Dataset {key!r} has no data points to fit")
    if error.shape != np.asarray(dataset.asymmetry, dtype=float).shape:
        raise ValueError(f"Dataset {key!r} error and asymmetry arrays have mismatched shapes")
    if not np.all(np.isfinite(error)) or np.any(error <= 0.0):
        raise ValueError(
            f"Dataset {key!r} has non-finite or non-positive asymmetry errors; "
            "global asymmetry-domain fitting weights each point by 1/σ² and requires "
            "finite, positive σ on every point"
        )


def _count_in_range(dataset: MuonDataset, t_min: float | None, t_max: float | None) -> int:
    """Number of points that survive the optional ``[t_min, t_max]`` clip."""
    clipped = dataset.time_range(t_min, t_max) if (t_min or t_max) else dataset
    return int(len(clipped.time))


def fit_global(
    datasets: Sequence[MuonDataset] | Mapping[Hashable, MuonDataset],
    model_fn: Callable[..., NDArray],
    *,
    global_params: list[str],
    local_params: list[str],
    initial_params: ParameterSet | Mapping[Hashable, ParameterSet],
    t_min: float | None = None,
    t_max: float | None = None,
    method: str = "migrad",
    max_calls: int = 10000,
    minos: bool = False,
    fit_engine: FitEngine | None = None,
    cancel_callback: Callable[[], bool] | None = None,
) -> GlobalFitResult:
    """Fit one model across several asymmetry traces, sharing chosen parameters.

    This is the asymmetry-domain global (shared-parameter) fit: one combined
    iminuit least-squares cost ``Σ_d Σ_i ((A_i − μ_i)/σ_i)²`` over the datasets'
    ``.asymmetry`` traces, with a single shared value per ``global_params`` entry
    and an independent value per dataset for each ``local_params`` entry.

    It wraps :meth:`asymmetry.core.fitting.engine.FitEngine.global_fit` (the
    shared minimiser seam) and adds the asymmetry-domain ergonomics: datasets need
    not carry unique run numbers, ``initial_params`` may be one broadcast seed, and
    the return value is a :class:`GlobalFitResult` with the shared globals,
    per-dataset locals, and a combined reduced χ². For the statistically-faithful
    low-count alternative (Cash/Poisson on grouped counts) use the count-domain
    :func:`~asymmetry.core.fitting.grouped_time_domain.fit_grouped_series`; see
    ``docs/porting/asymmetry-domain-global-fit/`` for when to use which.

    Parameters
    ----------
    datasets
        Asymmetry datasets to fit simultaneously. A sequence is keyed positionally
        (``0..N-1``); a mapping keeps its keys. Results are keyed the same way.
    model_fn
        Model ``f(t, **params) -> array`` returning the asymmetry/polarization for
        the named parameters (e.g. ``CompositeModel.from_expression(expr)
        .to_model_definition().function``).
    global_params
        Names shared across all datasets (one fitted value for the whole series).
    local_params
        Names estimated independently for each dataset.
    initial_params
        Either a single :class:`ParameterSet` broadcast to every dataset, or a
        mapping ``dataset key -> ParameterSet``. Each set must contain every name
        referenced by ``global_params`` and ``local_params`` (plus any fixed
        parameters the model needs). Global seed values and bounds are taken from
        the first dataset's set.
    t_min, t_max
        Optional time-range clip applied to every dataset.
    method
        Minimisation method (``"migrad"`` or ``"simplex"``).
    max_calls
        Maximum cost-function evaluations.
    minos
        Run MINOS for asymmetric intervals on top of the symmetric HESSE errors.
    fit_engine
        Optional :class:`FitEngine` to reuse; a fresh one is created otherwise.
    cancel_callback
        Cooperative cancel probe polled inside the cost function; a truthy return
        raises :class:`~asymmetry.core.fitting.engine.FitCancelledError`.

    Returns
    -------
    GlobalFitResult
        Shared global estimates and uncertainties, per-dataset results (keyed by
        the caller's dataset keys), and the combined reduced χ².

    Notes
    -----
    A single-dataset call behaves like an ordinary single fit: the combined
    reduced χ² reduces exactly to that dataset's reduced χ².
    """
    pairs = _normalize_datasets(datasets)
    keys = [key for key, _ in pairs]
    initial_by_key = _resolve_initial_params(initial_params, keys)

    _validate_parameter_partition(global_params, local_params, initial_by_key)
    for key, ds in pairs:
        _validate_errors(key, ds)

    # The engine keys datasets by run_number and requires those to be unique. The
    # caller's datasets may share (or default to) a run_number, so we wrap each in
    # a throwaway dataset carrying a synthetic, guaranteed-unique run number and
    # map the engine's results back to the caller's keys afterwards. The arrays are
    # shared (not copied): the engine only reads them and clips with time_range.
    synthetic_to_key: dict[int, Hashable] = {}
    engine_datasets: list[MuonDataset] = []
    engine_initial: dict[int, ParameterSet] = {}
    for index, (key, ds) in enumerate(pairs):
        synthetic_run = index
        synthetic_to_key[synthetic_run] = key
        metadata = {**dict(ds.metadata), "run_number": synthetic_run}
        engine_datasets.append(
            MuonDataset(
                time=np.asarray(ds.time, dtype=float),
                asymmetry=np.asarray(ds.asymmetry, dtype=float),
                error=np.asarray(ds.error, dtype=float),
                metadata=metadata,
                run=None,
            )
        )
        engine_initial[synthetic_run] = initial_by_key[key]

    engine = fit_engine or FitEngine()
    internal_results, fitted_global = engine.global_fit(
        engine_datasets,
        model_fn,
        global_params=list(global_params),
        local_params=list(local_params),
        initial_params=engine_initial,
        t_min=t_min,
        t_max=t_max,
        method=method,
        max_calls=max_calls,
        minos=minos,
        cancel_callback=cancel_callback,
    )

    dataset_results: dict[Hashable, FitResult] = {
        synthetic_to_key[run]: result
        for run, result in internal_results.items()
        if run in synthetic_to_key
    }

    # Combined reduced χ². The engine reports each dataset's Gaussian χ² and its
    # own dof, but per-dataset dof subtracts the shared globals from *every*
    # dataset, so summing them would double-count the globals. Compute the
    # combined dof directly instead: total points minus the free globals (counted
    # once) minus the free locals (counted per dataset).
    first_set = engine_initial[0]
    n_free_global = sum(1 for name in global_params if not first_set[name].fixed)

    combined_chi2 = 0.0
    total_points = 0
    n_free_local_total = 0
    for index, (key, ds) in enumerate(pairs):
        result = dataset_results.get(key)
        if result is not None:
            combined_chi2 += float(result.chi_squared)
        total_points += _count_in_range(ds, t_min, t_max)
        param_set = engine_initial[index]
        n_free_local_total += sum(1 for name in local_params if not param_set[name].fixed)

    dof = total_points - n_free_global - n_free_local_total
    reduced_chi2 = combined_chi2 / max(dof, 1)

    global_uncertainties: dict[str, float] = {}
    if dataset_results:
        any_result = next(iter(dataset_results.values()))
        for name in global_params:
            if name in any_result.uncertainties:
                global_uncertainties[name] = float(any_result.uncertainties[name])

    success = bool(dataset_results) and all(r.success for r in dataset_results.values())
    if success:
        message = "Asymmetry-domain global fit successful"
    elif dataset_results:
        failed = [str(key) for key, r in dataset_results.items() if not r.success]
        message = f"Asymmetry-domain global fit failed for datasets: {', '.join(failed)}"
    else:
        message = "Asymmetry-domain global fit produced no results"

    return GlobalFitResult(
        success=success,
        global_parameters=fitted_global,
        global_uncertainties=global_uncertainties,
        dataset_results=dataset_results,
        chi_squared=combined_chi2,
        dof=dof,
        reduced_chi_squared=reduced_chi2,
        message=message,
    )
