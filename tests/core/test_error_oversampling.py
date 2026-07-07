"""Tests for the zero-padding correlated-samples (effective-sample-size) correction.

A zero-padded FFT spectrum is sinc-interpolated: padding by a factor ``s``
densifies the sampled points without adding independent information, so only
``~1/s`` of the points are statistically independent. Fitting or computing
moment uncertainties as if every sample were independent underestimates
uncertainties by ``~√s``. ``error_oversampling=s`` on :meth:`FitEngine.fit`,
:meth:`FitEngine.global_fit`, and :func:`spectrum_moments` applies the standard
correction (WiMDA precedent: ``Analyse.pas:5228`` divides dof by the zero-pad
factor; this implementation additionally scales χ² and the reported errors).

These tests pin the exact relations documented in the engine/moments
docstrings: fitted *values* are untouched, uncertainties scale by ``√s``, χ² by
``1/s``, and dof by the effective (reduced) point count.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.fitting.engine import FitEngine
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.fitting.series import fit_asymmetry_series
from asymmetry.core.fourier import GroupSpectrumConfig, compute_average_group_spectrum
from asymmetry.core.fourier.moments import spectrum_moments

# ── shared synthetic fixtures ────────────────────────────────────────────────

TRUE_A0 = 0.22
TRUE_LAMBDA = 0.6
FREQUENCY_MHZ = 3.0
SIGMA = 0.01
N_POINTS = 200
T_MAX = 6.0


def _model(t, **params):
    """A decaying cosine at a fixed known frequency."""
    a0 = params["A0"]
    rate = params["Lambda"]
    t = np.asarray(t, dtype=float)
    return a0 * np.exp(-rate * t) * np.cos(2.0 * np.pi * FREQUENCY_MHZ * t)


def _dataset(seed: int = 12345, amplitude: float = TRUE_A0) -> MuonDataset:
    rng = np.random.default_rng(seed)
    time = np.linspace(0.0, T_MAX, N_POINTS)
    clean = _model(time, A0=amplitude, Lambda=TRUE_LAMBDA)
    noisy = clean + rng.normal(0.0, SIGMA, size=time.shape)
    error = np.full_like(time, SIGMA)
    return MuonDataset(time=time, asymmetry=noisy, error=error)


def _seed_params(amplitude: float = 0.2, lam: float = 0.5) -> ParameterSet:
    return ParameterSet(
        [
            Parameter(name="A0", value=amplitude, min=0.0, max=1.0),
            Parameter(name="Lambda", value=lam, min=0.0, max=5.0),
        ]
    )


# ── 1. single fit ────────────────────────────────────────────────────────────


def test_single_fit_error_oversampling_scales_errors_and_chi2_not_values():
    engine = FitEngine()
    dataset = _dataset()

    plain = engine.fit(dataset, _model, _seed_params(), error_oversampling=1.0)
    scaled = engine.fit(dataset, _model, _seed_params(), error_oversampling=4.0)

    assert plain.success and scaled.success

    # Fitted values are the raw-χ² minimiser's location; unaffected.
    for name in ("A0", "Lambda"):
        assert scaled.parameters[name].value == pytest.approx(
            plain.parameters[name].value, rel=1e-9
        )

    # Every uncertainty scales by exactly √4 = 2.
    assert set(scaled.uncertainties) == set(plain.uncertainties)
    for name, plain_sigma in plain.uncertainties.items():
        assert scaled.uncertainties[name] == pytest.approx(2.0 * plain_sigma, rel=1e-9)

    ndata = N_POINTS
    nfree = 2
    ndata_eff = max(round(ndata / 4.0), 1)
    expected_dof = max(ndata_eff - nfree, 1)
    assert plain.dof == ndata - nfree  # default path: unclamped, historical formula
    assert scaled.dof == expected_dof

    assert scaled.chi_squared == pytest.approx(plain.chi_squared / 4.0, rel=1e-9)
    assert scaled.reduced_chi_squared == pytest.approx(scaled.chi_squared / expected_dof, rel=1e-9)

    # Covariance scales by s (variance), consistent with the √s error scale.
    assert plain.covariance is not None and scaled.covariance is not None
    assert np.allclose(scaled.covariance, np.asarray(plain.covariance) * 4.0)

    # The correction is disclosed as an advisory on the corrected result only.
    assert not any("zero-padding" in w for w in plain.warnings)
    assert any("zero-padding" in w for w in scaled.warnings)
    assert any("×2" in w for w in scaled.warnings)


def test_single_fit_error_oversampling_default_is_unchanged():
    """error_oversampling defaults to 1.0 — byte-identical to the historical path."""
    engine = FitEngine()
    dataset = _dataset()

    explicit = engine.fit(dataset, _model, _seed_params(), error_oversampling=1.0)
    implicit = engine.fit(dataset, _model, _seed_params())

    assert explicit.chi_squared == implicit.chi_squared
    assert explicit.dof == implicit.dof
    assert explicit.uncertainties == implicit.uncertainties
    assert explicit.warnings == implicit.warnings == []


# ── 2. global fit (joint strategy: free shared global + per-dataset local) ──


def _global_datasets() -> tuple[list[MuonDataset], dict[int, ParameterSet]]:
    ds1 = _dataset(seed=1, amplitude=0.20)
    ds1.metadata["run_number"] = 1
    ds2 = _dataset(seed=2, amplitude=0.28)
    ds2.metadata["run_number"] = 2

    init = {
        1: ParameterSet(
            [
                Parameter(name="A0", value=0.2, min=0.0, max=1.0),
                Parameter(name="Lambda", value=0.5, min=0.0, max=5.0),
            ]
        ),
        2: ParameterSet(
            [
                Parameter(name="A0", value=0.2, min=0.0, max=1.0),
                Parameter(name="Lambda", value=0.5, min=0.0, max=5.0),
            ]
        ),
    }
    return [ds1, ds2], init


def test_global_fit_joint_error_oversampling_scales_per_dataset_stats():
    datasets, init = _global_datasets()
    engine = FitEngine()

    plain_results, plain_global = engine.global_fit(
        datasets,
        _model,
        global_params=["Lambda"],
        local_params=["A0"],
        initial_params=init,
        strategy="joint",
        error_oversampling=1.0,
    )
    scaled_results, scaled_global = engine.global_fit(
        datasets,
        _model,
        global_params=["Lambda"],
        local_params=["A0"],
        initial_params=init,
        strategy="joint",
        error_oversampling=4.0,
    )

    assert plain_global["Lambda"].value == pytest.approx(scaled_global["Lambda"].value, rel=1e-9)

    ndata = N_POINTS
    nfree = 2  # one shared Lambda + one local A0
    ndata_eff = max(round(ndata / 4.0), 1)
    expected_dof = max(ndata_eff - nfree, 1)

    for ds in datasets:
        run = ds.run_number
        plain = plain_results[run]
        scaled = scaled_results[run]

        for name in ("A0", "Lambda"):
            assert scaled.parameters[name].value == pytest.approx(
                plain.parameters[name].value, rel=1e-9
            )
        assert set(scaled.uncertainties) == set(plain.uncertainties)
        for name, plain_sigma in plain.uncertainties.items():
            assert scaled.uncertainties[name] == pytest.approx(2.0 * plain_sigma, rel=1e-9)

        assert plain.dof == ndata - nfree
        assert scaled.dof == expected_dof
        assert scaled.chi_squared == pytest.approx(plain.chi_squared / 4.0, rel=1e-9)
        assert scaled.reduced_chi_squared == pytest.approx(
            scaled.chi_squared / expected_dof, rel=1e-9
        )
        assert not any("zero-padding" in w for w in plain.warnings)
        assert any("zero-padding" in w for w in scaled.warnings)


def test_global_fit_profiled_error_oversampling_scales_per_dataset_stats():
    """The profiled strategy applies the same one-shot post-hoc correction."""
    datasets, init = _global_datasets()
    engine = FitEngine()

    plain_results, _ = engine.global_fit(
        datasets,
        _model,
        global_params=["Lambda"],
        local_params=["A0"],
        initial_params=init,
        strategy="profiled",
        error_oversampling=1.0,
    )
    scaled_results, _ = engine.global_fit(
        datasets,
        _model,
        global_params=["Lambda"],
        local_params=["A0"],
        initial_params=init,
        strategy="profiled",
        error_oversampling=4.0,
    )

    ndata = N_POINTS
    nfree = 2
    ndata_eff = max(round(ndata / 4.0), 1)
    expected_dof = max(ndata_eff - nfree, 1)

    for ds in datasets:
        run = ds.run_number
        plain = plain_results[run]
        scaled = scaled_results[run]

        for name in ("A0", "Lambda"):
            assert scaled.parameters[name].value == pytest.approx(
                plain.parameters[name].value, rel=1e-6
            )
        assert set(scaled.uncertainties) == set(plain.uncertainties)
        for name, plain_sigma in plain.uncertainties.items():
            assert scaled.uncertainties[name] == pytest.approx(2.0 * plain_sigma, rel=1e-6)

        assert plain.dof == ndata - nfree
        assert scaled.dof == expected_dof
        assert scaled.chi_squared == pytest.approx(plain.chi_squared / 4.0, rel=1e-6)
        assert not any("zero-padding" in w for w in plain.warnings)
        assert any("zero-padding" in w for w in scaled.warnings)


# ── 3. fit_asymmetry_series ──────────────────────────────────────────────────


def test_series_fit_error_oversampling_doubles_uncertainties():
    ds1 = _dataset(seed=11, amplitude=0.2)
    ds1.metadata["run_number"] = 1
    ds2 = _dataset(seed=12, amplitude=0.25)
    ds2.metadata["run_number"] = 2
    datasets = [ds1, ds2]
    init = {1: _seed_params(), 2: _seed_params()}

    plain = fit_asymmetry_series(
        datasets,
        _model,
        global_params=[],
        local_params=["A0", "Lambda"],
        initial_params=init,
        error_oversampling=1.0,
    )
    scaled = fit_asymmetry_series(
        datasets,
        _model,
        global_params=[],
        local_params=["A0", "Lambda"],
        initial_params=init,
        error_oversampling=4.0,
    )

    for run in (1, 2):
        plain_result = plain.results[run]
        scaled_result = scaled.results[run]
        assert set(scaled_result.uncertainties) == set(plain_result.uncertainties)
        for name, plain_sigma in plain_result.uncertainties.items():
            assert scaled_result.uncertainties[name] == pytest.approx(2.0 * plain_sigma, rel=1e-9)
        for name in ("A0", "Lambda"):
            assert scaled_result.parameters[name].value == pytest.approx(
                plain_result.parameters[name].value, rel=1e-9
            )


# ── 4. spectrum_moments ──────────────────────────────────────────────────────


def _gaussian(x: np.ndarray, mu: float, sigma: float, amp: float = 1.0) -> np.ndarray:
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


@pytest.mark.parametrize("uncertainty", ["propagate", "bootstrap"])
def test_spectrum_moments_error_oversampling_scales_errors_only(uncertainty: str) -> None:
    x = np.linspace(-60.0, 60.0, 2001)
    amp = _gaussian(x, 0.0, 8.0)
    errors = 0.05 * np.ones_like(x)

    plain = spectrum_moments(
        x,
        amp,
        x_range=None,
        cutoff_fraction=0.05,
        errors=errors,
        uncertainty=uncertainty,
        n_bootstrap=128,
        seed=3,
        error_oversampling=1.0,
    )
    scaled = spectrum_moments(
        x,
        amp,
        x_range=None,
        cutoff_fraction=0.05,
        errors=errors,
        uncertainty=uncertainty,
        n_bootstrap=128,
        seed=3,
        error_oversampling=4.0,
    )

    assert scaled.n_sample == plain.n_sample
    for value_field in (
        "b_pk",
        "b_ave",
        "b_diff",
        "b_rms_mean",
        "b_rms_peak",
        "skewness",
        "skewness_g1",
        "beta",
    ):
        assert getattr(scaled, value_field) == getattr(plain, value_field)

    for err_field in (
        "b_pk_err",
        "b_ave_err",
        "b_diff_err",
        "b_rms_mean_err",
        "b_rms_peak_err",
        "skewness_err",
        "skewness_g1_err",
        "beta_err",
    ):
        plain_err = getattr(plain, err_field)
        scaled_err = getattr(scaled, err_field)
        if math.isfinite(plain_err):
            assert scaled_err == pytest.approx(2.0 * plain_err)
        else:
            assert math.isnan(scaled_err)


# ── 5. spectrum metadata stamps the padding factor ──────────────────────────


def test_spectrum_metadata_records_padding_factor():
    time = np.arange(512, dtype=float) * 0.04
    counts = 4000.0 * np.exp(-time / 2.1969811) * (1.0 + 0.2 * np.cos(2.0 * np.pi * 2.7 * time))
    run = Run(
        run_number=9,
        histograms=[
            Histogram(counts=counts, bin_width=0.04),
            Histogram(counts=counts * 0.9, bin_width=0.04),
        ],
        metadata={"field": 200.0},
        grouping={"groups": {1: [1], 2: [2]}, "deadtime_correction": False},
    )

    default_padding = compute_average_group_spectrum(run, GroupSpectrumConfig(window="none"))
    padded = compute_average_group_spectrum(run, GroupSpectrumConfig(window="none", padding=4))

    assert default_padding.metadata["fourier_padding"] == 1
    assert padded.metadata["fourier_padding"] == 4
