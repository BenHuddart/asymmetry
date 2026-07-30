"""Tests for :meth:`FitEngine.fit_arrays` — the array-taking fit front door.

The contract these pin: ``fit_arrays`` and ``fit`` share one internal core, so
fitting a bare ``(t, A, σ)`` triple is *exactly* equivalent to fitting a
``MuonDataset`` carrying the same arrays — no "close enough" tolerance. Every
fixture here is synthetic (analytic curve plus seeded pseudo-random noise) with
invented metadata.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.engine import (
    POISSON_COST,
    AsymmetryScaleWarning,
    FitCancelledError,
    FitEngine,
    FixedFrequencyFieldMismatchWarning,
)
from asymmetry.core.fitting.models import MODELS
from asymmetry.core.fitting.parameters import Parameter, ParameterSet


def _synthetic_curve(
    n: int = 120, *, seed: int = 20250730
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A decaying percent-scale asymmetry curve with reproducible noise."""
    rng = np.random.default_rng(seed)
    time = np.linspace(0.05, 8.0, n)
    truth = 22.0 * np.exp(-0.35 * time) + 1.5
    error = np.full_like(time, 0.4)
    asymmetry = truth + rng.normal(0.0, 0.4, size=n)
    return time, asymmetry, error


def _dataset(**metadata) -> MuonDataset:
    time, asymmetry, error = _synthetic_curve()
    meta = {"run_number": 101, "temperature": 5.0}
    meta.update(metadata)
    return MuonDataset(time=time, asymmetry=asymmetry, error=error, metadata=meta)


def _exponential_params() -> ParameterSet:
    params = ParameterSet()
    params.add(Parameter(name="A0", value=20.0, min=0.0, max=100.0))
    params.add(Parameter(name="Lambda", value=0.3, min=0.0))
    params.add(Parameter(name="baseline", value=0.0))
    return params


_MODEL = MODELS["ExponentialRelaxation"].function


def _assert_identical(first, second) -> None:
    """Assert two FitResults agree exactly — values, not tolerances."""
    assert first.success == second.success
    assert first.chi_squared == second.chi_squared
    assert first.reduced_chi_squared == second.reduced_chi_squared
    assert first.dof == second.dof
    assert first.message == second.message
    assert first.function_calls == second.function_calls
    assert [p.name for p in first.parameters] == [p.name for p in second.parameters]
    for a, b in zip(first.parameters, second.parameters, strict=True):
        assert a.value == b.value, f"parameter {a.name} differs"
    assert first.uncertainties.keys() == second.uncertainties.keys()
    for name, value in first.uncertainties.items():
        assert value == second.uncertainties[name], f"uncertainty {name} differs"
    assert first.covariance_parameters == second.covariance_parameters
    if first.covariance is None:
        assert second.covariance is None
    else:
        assert np.array_equal(np.asarray(first.covariance), np.asarray(second.covariance))
    assert np.array_equal(first.residuals, second.residuals)
    assert first.minos_errors == second.minos_errors


# --- equivalence with fit() ------------------------------------------------


def test_fit_arrays_matches_fit_on_the_same_arrays() -> None:
    """The headline equivalence: same arrays in, byte-identical FitResult out."""
    ds = _dataset()
    engine = FitEngine()
    from_dataset = engine.fit(ds, _MODEL, _exponential_params())
    from_arrays = engine.fit_arrays(ds.time, ds.asymmetry, ds.error, _MODEL, _exponential_params())
    assert from_arrays.success
    _assert_identical(from_dataset, from_arrays)


def test_fit_arrays_matches_the_replace_workaround() -> None:
    """The hack this API replaces: dataclasses.replace to smuggle arrays in."""
    base = _dataset()
    time = np.linspace(0.1, 6.0, 90)
    asymmetry = 18.0 * np.exp(-0.5 * time) + 0.75
    error = np.full_like(time, 0.3)
    smuggled = dataclasses.replace(base, time=time, asymmetry=asymmetry, error=error)

    engine = FitEngine()
    _assert_identical(
        engine.fit(smuggled, _MODEL, _exponential_params()),
        engine.fit_arrays(time, asymmetry, error, _MODEL, _exponential_params()),
    )


@pytest.mark.parametrize(
    ("t_min", "t_max"),
    [(None, None), (1.0, None), (None, 4.0), (0.5, 5.5)],
)
def test_fit_arrays_time_window_matches_fit(t_min: float | None, t_max: float | None) -> None:
    """t_min/t_max clip identically to MuonDataset.time_range."""
    ds = _dataset()
    engine = FitEngine()
    _assert_identical(
        engine.fit(ds, _MODEL, _exponential_params(), t_min, t_max),
        engine.fit_arrays(
            ds.time, ds.asymmetry, ds.error, _MODEL, _exponential_params(), t_min, t_max
        ),
    )


def test_fit_arrays_forwards_every_shared_kwarg() -> None:
    """minos, simplex, migrad_kwargs and error_oversampling all route alike."""
    ds = _dataset()
    engine = FitEngine()
    kwargs = dict(
        method="migrad",
        minos=True,
        migrad_kwargs={"ncall": 400},
        error_oversampling=4.0,
    )
    _assert_identical(
        engine.fit(ds, _MODEL, _exponential_params(), 0.2, 6.0, **kwargs),
        engine.fit_arrays(
            ds.time, ds.asymmetry, ds.error, _MODEL, _exponential_params(), 0.2, 6.0, **kwargs
        ),
    )


def test_fit_arrays_honours_a_cost_factory() -> None:
    """The selectable objective seam reaches the array path too."""
    time = np.linspace(0.05, 6.0, 100)
    counts = 900.0 * np.exp(-time / 2.19703) + 12.0
    error = np.sqrt(counts)
    ds = MuonDataset(time=time, asymmetry=counts, error=error, metadata={"run_number": 7})

    params = ParameterSet()
    params.add(Parameter(name="A0", value=800.0, min=0.0))
    params.add(Parameter(name="Lambda", value=0.4, min=0.0))
    params.add(Parameter(name="baseline", value=5.0, min=0.0))

    engine = FitEngine()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AsymmetryScaleWarning)
        _assert_identical(
            engine.fit(ds, _MODEL, params, cost_factory=POISSON_COST),
            engine.fit_arrays(time, counts, error, _MODEL, params, cost_factory=POISSON_COST),
        )


def test_fit_arrays_accepts_plain_sequences() -> None:
    """array_like input is coerced, so a list of floats works from a script."""
    time, asymmetry, error = _synthetic_curve(n=60)
    engine = FitEngine()
    _assert_identical(
        engine.fit_arrays(time, asymmetry, error, _MODEL, _exponential_params()),
        engine.fit_arrays(
            time.tolist(), asymmetry.tolist(), error.tolist(), _MODEL, _exponential_params()
        ),
    )


def test_fit_arrays_does_not_mutate_its_inputs() -> None:
    time, asymmetry, error = _synthetic_curve(n=50)
    originals = (time.copy(), asymmetry.copy(), error.copy())
    FitEngine().fit_arrays(time, asymmetry, error, _MODEL, _exponential_params(), 1.0, 5.0)
    for before, after in zip(originals, (time, asymmetry, error), strict=True):
        assert np.array_equal(before, after)


def test_fit_arrays_supports_cooperative_cancellation() -> None:
    time, asymmetry, error = _synthetic_curve()
    with pytest.raises(FitCancelledError):
        FitEngine().fit_arrays(
            time,
            asymmetry,
            error,
            _MODEL,
            _exponential_params(),
            cancel_callback=lambda: True,
        )


# --- validation guards -----------------------------------------------------


def test_fit_arrays_rejects_length_mismatch() -> None:
    time, asymmetry, error = _synthetic_curve(n=40)
    with pytest.raises(ValueError, match="same length"):
        FitEngine().fit_arrays(time, asymmetry[:-3], error, _MODEL, _exponential_params())
    with pytest.raises(ValueError, match="error has 37"):
        FitEngine().fit_arrays(time, asymmetry, error[:-3], _MODEL, _exponential_params())


def test_fit_arrays_rejects_non_one_dimensional_input() -> None:
    time, asymmetry, error = _synthetic_curve(n=40)
    with pytest.raises(ValueError, match="asymmetry must be a one-dimensional array"):
        FitEngine().fit_arrays(time, asymmetry.reshape(4, 10), error, _MODEL, _exponential_params())


def test_fit_arrays_rejects_empty_input() -> None:
    empty = np.array([], dtype=float)
    with pytest.raises(ValueError, match="nothing to fit"):
        FitEngine().fit_arrays(empty, empty, empty, _MODEL, _exponential_params())


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_fit_arrays_rejects_non_finite_values(bad: float) -> None:
    time, asymmetry, error = _synthetic_curve(n=40)
    asymmetry = asymmetry.copy()
    asymmetry[7] = bad
    with pytest.raises(ValueError, match="asymmetry contains 1 non-finite"):
        FitEngine().fit_arrays(time, asymmetry, error, _MODEL, _exponential_params())


def test_fit_arrays_rejects_an_empty_fit_window() -> None:
    time, asymmetry, error = _synthetic_curve()
    with pytest.raises(ValueError, match="no data points remain in the fit window"):
        FitEngine().fit_arrays(time, asymmetry, error, _MODEL, _exponential_params(), 20.0, 30.0)
    with pytest.raises(ValueError, match="no data points remain in the fit window"):
        FitEngine().fit_arrays(time, asymmetry, error, _MODEL, _exponential_params(), 5.0, 1.0)


# --- advisory guards -------------------------------------------------------


def test_fit_arrays_still_warns_on_a_percent_fraction_scale_mismatch() -> None:
    """The unit trap this API must not hide: fraction data, percent seeds."""
    time, asymmetry, error = _synthetic_curve()
    with pytest.warns(AsymmetryScaleWarning, match="fraction-scale"):
        FitEngine().fit_arrays(
            time,
            asymmetry / 100.0,
            error / 100.0,
            _MODEL,
            _exponential_params(),
        )


def test_fit_arrays_cannot_run_the_field_metadata_guard() -> None:
    """Documented difference: bare arrays carry no ``field`` to compare against."""
    time = np.linspace(0.05, 8.0, 200)
    asymmetry = 20.0 * np.cos(2.0 * np.pi * 6.0 * time) * np.exp(-0.3 * time)
    error = np.full_like(time, 0.5)

    params = ParameterSet()
    params.add(Parameter(name="A0", value=20.0, min=0.0, max=100.0))
    params.add(Parameter(name="frequency", value=6.0, fixed=True))
    params.add(Parameter(name="phase", value=0.0))
    params.add(Parameter(name="Lambda", value=0.3, min=0.0))
    params.add(Parameter(name="baseline", value=0.0))
    model = MODELS["Oscillatory"].function

    ds = MuonDataset(
        time=time,
        asymmetry=asymmetry,
        error=error,
        metadata={"run_number": 202, "field": 400.0},
    )
    # Via the dataset the pin is flagged against gamma_mu * B ...
    with pytest.warns(FixedFrequencyFieldMismatchWarning):
        FitEngine().fit(ds, model, params)
    # ... and via bare arrays it cannot be, because there is no field metadata.
    with warnings.catch_warnings():
        warnings.simplefilter("error", FixedFrequencyFieldMismatchWarning)
        result = FitEngine().fit_arrays(time, asymmetry, error, model, params)
    assert result.success
