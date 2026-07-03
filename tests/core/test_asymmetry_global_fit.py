"""Tests for the asymmetry-domain global (shared-parameter) fit.

Synthetic, deterministic datasets share a known global relaxation rate while
each carries its own local amplitude — the motivating Keren-style case (a rate
shared across fields, amplitude free per field) reduced to an analytic
exponential so the tests need no external corpus.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting import (
    FitEngine,
    GlobalFitResult,
    Parameter,
    ParameterSet,
    fit_global,
)

# Injected ground truth shared across the tests.
TRUE_LAMBDA = 0.75  # shared global relaxation rate (µs⁻¹)
TRUE_AMPS = [0.24, 0.18, 0.30]  # per-dataset local amplitudes
SIGMA = 0.01  # per-point asymmetry error
N_POINTS = 200
T_MAX = 8.0


def _model(t, **params):
    """A_d(t) = amp · exp(−lambda · t)."""
    amp = params["amp"]
    lam = params["lambda"]
    return amp * np.exp(-lam * np.asarray(t, dtype=float))


def _make_dataset(amp: float, seed: int) -> MuonDataset:
    rng = np.random.default_rng(seed)
    time = np.linspace(0.0, T_MAX, N_POINTS)
    clean = amp * np.exp(-TRUE_LAMBDA * time)
    noisy = clean + rng.normal(0.0, SIGMA, size=time.shape)
    error = np.full_like(time, SIGMA)
    return MuonDataset(time=time, asymmetry=noisy, error=error)


def _seed(amp: float = 0.2, lam: float = 0.5, **overrides) -> ParameterSet:
    amp_kw = {"name": "amp", "value": amp, "min": 0.0, "max": 1.0}
    lam_kw = {"name": "lambda", "value": lam, "min": 0.0, "max": 10.0}
    amp_kw.update(overrides.get("amp", {}))
    lam_kw.update(overrides.get("lambda", {}))
    return ParameterSet([Parameter(**amp_kw), Parameter(**lam_kw)])


def _datasets(n: int = 3) -> list[MuonDataset]:
    return [_make_dataset(TRUE_AMPS[i], seed=100 + i) for i in range(n)]


# --- recovery ---------------------------------------------------------------


def test_recovers_injected_global_and_locals():
    datasets = _datasets(3)
    result = fit_global(
        datasets,
        _model,
        global_params=["lambda"],
        local_params=["amp"],
        initial_params=_seed(),
    )

    assert isinstance(result, GlobalFitResult)
    assert result.success
    # Shared global recovered.
    fitted_lambda = result.global_parameters["lambda"].value
    assert fitted_lambda == pytest.approx(TRUE_LAMBDA, abs=0.02)
    assert "lambda" in result.global_uncertainties
    # Per-dataset locals recovered, keyed positionally.
    assert set(result.dataset_results) == {0, 1, 2}
    for key in (0, 1, 2):
        amp = result.dataset_results[key].parameters["amp"].value
        assert amp == pytest.approx(TRUE_AMPS[key], abs=0.02)
        # Every per-dataset result also carries the shared global.
        assert result.dataset_results[key].parameters["lambda"].value == pytest.approx(
            fitted_lambda
        )


def test_combined_reduced_chi_squared_near_one():
    result = fit_global(
        _datasets(3),
        _model,
        global_params=["lambda"],
        local_params=["amp"],
        initial_params=_seed(),
    )
    # 3 datasets × 200 points − 1 global − 3 locals.
    assert result.dof == 3 * N_POINTS - 1 - 3
    assert result.reduced_chi_squared == pytest.approx(1.0, abs=0.25)
    # Combined χ² is the sum of the per-dataset χ².
    per_dataset_sum = sum(r.chi_squared for r in result.dataset_results.values())
    assert result.chi_squared == pytest.approx(per_dataset_sum)


# --- statistical tightening (the scientific justification) ------------------


def test_global_constrains_shared_parameter_more_tightly_than_independent_fits():
    datasets = _datasets(3)
    global_result = fit_global(
        datasets,
        _model,
        global_params=["lambda"],
        local_params=["amp"],
        initial_params=_seed(),
    )
    global_sigma = global_result.global_uncertainties["lambda"]

    engine = FitEngine()
    independent_sigmas = []
    for ds in datasets:
        single = engine.fit(ds, _model, _seed())
        assert single.success
        independent_sigmas.append(single.uncertainties["lambda"])

    # Pooling the data across datasets must constrain the shared rate better than
    # any single-dataset fit can.
    assert global_sigma < min(independent_sigmas)


# --- single dataset behaves like a normal fit -------------------------------


def test_single_dataset_matches_ordinary_fit():
    ds = _make_dataset(TRUE_AMPS[0], seed=7)
    seed = _seed()
    result = fit_global(
        [ds],
        _model,
        global_params=["lambda"],
        local_params=["amp"],
        initial_params=seed,
    )
    single = FitEngine().fit(ds, _model, _seed())

    assert result.success and single.success
    assert result.dataset_results[0].parameters["lambda"].value == pytest.approx(
        single.parameters["lambda"].value, rel=1e-3
    )
    assert result.dataset_results[0].parameters["amp"].value == pytest.approx(
        single.parameters["amp"].value, rel=1e-3
    )
    # Single-dataset combined reduced χ² reduces to the ordinary one.
    assert result.dof == single.dof
    assert result.reduced_chi_squared == pytest.approx(single.reduced_chi_squared, rel=1e-3)


# --- mapping input and broadcast vs per-key seeds ---------------------------


def test_mapping_input_keys_results_by_caller_key():
    datasets = {"B100": _make_dataset(TRUE_AMPS[0], 1), "B200": _make_dataset(TRUE_AMPS[1], 2)}
    per_key = {"B100": _seed(), "B200": _seed()}
    result = fit_global(
        datasets,
        _model,
        global_params=["lambda"],
        local_params=["amp"],
        initial_params=per_key,
    )
    assert result.success
    assert set(result.dataset_results) == {"B100", "B200"}


def test_broadcast_and_per_key_initial_params_agree():
    datasets = _datasets(2)
    broadcast = fit_global(
        datasets,
        _model,
        global_params=["lambda"],
        local_params=["amp"],
        initial_params=_seed(),
    )
    per_key = fit_global(
        datasets,
        _model,
        global_params=["lambda"],
        local_params=["amp"],
        initial_params={0: _seed(), 1: _seed()},
    )
    assert broadcast.global_parameters["lambda"].value == pytest.approx(
        per_key.global_parameters["lambda"].value, rel=1e-6
    )


# --- fixed and bounded parameters respected ---------------------------------


def test_fixed_global_is_held_constant():
    datasets = _datasets(2)
    fixed_value = 0.9
    seed = _seed(lam=fixed_value, **{"lambda": {"fixed": True}})
    result = fit_global(
        datasets,
        _model,
        global_params=["lambda"],
        local_params=["amp"],
        initial_params=seed,
    )
    assert result.global_parameters["lambda"].value == pytest.approx(fixed_value)
    # A fixed global is not counted as a free parameter in the combined dof.
    assert result.dof == 2 * N_POINTS - 0 - 2
    # No uncertainty is reported for a fixed parameter.
    assert "lambda" not in result.global_uncertainties


def test_bounds_are_respected():
    datasets = _datasets(2)
    # Cap lambda well below the true value; the fit must stay at the bound.
    capped = 0.5
    seed = _seed(lam=0.4, **{"lambda": {"max": capped}})
    result = fit_global(
        datasets,
        _model,
        global_params=["lambda"],
        local_params=["amp"],
        initial_params=seed,
    )
    assert result.global_parameters["lambda"].value <= capped + 1e-9


# --- edge cases -------------------------------------------------------------


def test_overlapping_global_and_local_raises():
    with pytest.raises(ValueError, match="overlap"):
        fit_global(
            _datasets(2),
            _model,
            global_params=["lambda"],
            local_params=["lambda"],
            initial_params=_seed(),
        )


def test_missing_referenced_parameter_raises():
    seed = ParameterSet([Parameter(name="amp", value=0.2)])  # no "lambda"
    with pytest.raises(ValueError, match="missing referenced"):
        fit_global(
            _datasets(2),
            _model,
            global_params=["lambda"],
            local_params=["amp"],
            initial_params=seed,
        )


def test_mapping_missing_dataset_key_raises():
    datasets = {"a": _make_dataset(0.2, 1), "b": _make_dataset(0.2, 2)}
    with pytest.raises(KeyError, match="missing entries"):
        fit_global(
            datasets,
            _model,
            global_params=["lambda"],
            local_params=["amp"],
            initial_params={"a": _seed()},  # "b" missing
        )


def test_empty_datasets_raise():
    with pytest.raises(ValueError, match="at least one dataset"):
        fit_global(
            [],
            _model,
            global_params=["lambda"],
            local_params=["amp"],
            initial_params=_seed(),
        )


def test_non_finite_errors_rejected():
    ds = _make_dataset(0.2, 1)
    bad = MuonDataset(
        time=ds.time,
        asymmetry=ds.asymmetry,
        error=np.zeros_like(ds.error),  # non-positive σ
    )
    with pytest.raises(ValueError, match="non-finite or non-positive"):
        fit_global(
            [ds, bad],
            _model,
            global_params=["lambda"],
            local_params=["amp"],
            initial_params=_seed(),
        )
