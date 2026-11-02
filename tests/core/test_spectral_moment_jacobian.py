"""Independent references for first-order spectral-moment uncertainties."""

import numpy as np
import pytest

from asymmetry.core.fourier.moments import _moment_jacobian, _moment_scalars, spectrum_moments

_KEYS = ("b_pk", "b_ave", "b_diff", "b_rms_mean", "b_rms_peak", "skewness_g1", "beta")


def _reference(x, y, mask=None, peak=None):
    k = np.argmax(y) if peak is None else peak
    if mask is None:
        mask = y > 0
    a, b, _ = np.polyfit(x[k - 2 : k + 3], y[k - 2 : k + 3], 2)
    vertex = -b / (2 * a)
    weights = y[mask] / y[mask].sum()
    mean = weights @ x[mask]
    d = x[mask] - mean
    m2 = weights @ d**2
    m3 = weights @ d**3
    wp = np.sqrt(weights @ (x[mask] - vertex) ** 2)
    return np.array(
        [vertex, mean, mean - vertex, np.sqrt(m2), wp, m3 / m2**1.5, (mean - vertex) / wp]
    )


def test_propagated_errors_match_independent_reference():
    x = np.arange(7.0)
    y = np.array([1, 2, 5, 8, 6, 3, 2.0])
    result = spectrum_moments(
        x, y, x_range=None, errors=np.full(7, 1e-3), uncertainty="propagate", cutoff_fraction=0
    )
    np.testing.assert_allclose(
        [result.b_ave_err, result.b_rms_mean_err, result.skewness_g1_err],
        [1.97187630464e-4, 1.37772438800e-4, 2.02000561831e-4],
        rtol=1e-8,
    )
    assert np.isnan(result.skewness_err)


@pytest.mark.parametrize("case", ["gaussian", "skewed", "nonuniform", "high-cutoff"])
def test_propagated_errors_match_independent_finite_differences(case):
    if case == "gaussian":
        x = np.linspace(-30, 30, 121)
        y = np.exp(-0.5 * (x / 6) ** 2)
        cutoff = 0.0
    else:
        x = np.linspace(0, 20, 41)
        if case == "nonuniform":
            x = x + 0.06 * np.sin(x)
        y = x * x * np.exp(-x / 2.5) + 0.05
        cutoff = {"nonuniform": 0.1, "high-cutoff": 0.985}.get(case, 0.0)
    sigma = 0.002 * y.max() * (1 + 0.5 * np.cos(x))
    mask = y > cutoff * y.max()
    peak = np.argmax(y)
    step = 1e-6 * y.max()
    jac = np.empty((7, len(y)))
    for i in range(len(y)):
        delta = np.zeros_like(y)
        delta[i] = step
        jac[:, i] = (
            _reference(x, y + delta, mask, peak) - _reference(x, y - delta, mask, peak)
        ) / (2 * step)
    rows = _moment_jacobian(x, y, cutoff, _moment_scalars(x, y, cutoff))
    np.testing.assert_allclose([rows[key] for key in _KEYS], jac, rtol=1e-6, atol=1e-8)
    expected = np.sqrt((jac**2) @ sigma**2)
    result = spectrum_moments(
        x, y, x_range=None, errors=sigma, uncertainty="propagate", cutoff_fraction=cutoff
    )
    actual = [getattr(result, key + "_err") for key in _KEYS]
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-12)


def test_propagated_errors_agree_with_frozen_support_simulation():
    x = np.linspace(0, 20, 41)
    y = x * x * np.exp(-x / 2.5) + 0.05
    sigma = np.full_like(y, 1e-4 * y.max())
    mask = y > 0
    peak = np.argmax(y)
    rng = np.random.default_rng(20261031)
    samples = np.array(
        [_reference(x, draw, mask, peak) for draw in y + rng.normal(size=(20000, len(y))) * sigma]
    )
    result = spectrum_moments(
        x, y, x_range=None, errors=sigma, uncertainty="propagate", cutoff_fraction=0
    )
    np.testing.assert_allclose(
        [getattr(result, key + "_err") for key in _KEYS], samples.std(axis=0, ddof=1), rtol=0.03
    )


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
def test_propagated_errors_are_amplitude_scale_invariant(scale):
    x = np.arange(7.0)
    y = np.array([1, 2, 5, 8, 6, 3, 2.0])
    s = np.full(7, 1e-3)
    a = spectrum_moments(x, y, x_range=None, errors=s, uncertainty="propagate", cutoff_fraction=0)
    b = spectrum_moments(
        x, y * scale, x_range=None, errors=s * scale, uncertainty="propagate", cutoff_fraction=0
    )
    np.testing.assert_allclose(
        [getattr(a, k + "_err") for k in _KEYS], [getattr(b, k + "_err") for k in _KEYS], rtol=1e-12
    )


@pytest.mark.parametrize("y", [[4.0, 3, 2, 1, 1, 1, 1], [1, 2, 3, 3.0001, 3.0002, 0, 0]])
def test_peak_errors_unavailable_when_vertex_unrefined_or_clamped(y):
    result = spectrum_moments(
        np.arange(7.0),
        y,
        x_range=None,
        errors=np.ones(7) * 0.001,
        uncertainty="propagate",
        cutoff_fraction=0,
    )
    for key in ("b_pk", "b_diff", "b_rms_peak", "beta"):
        assert np.isnan(getattr(result, key + "_err"))
    assert np.isfinite(result.b_ave_err)


def test_zero_width_support_has_no_propagated_uncertainty():
    result = spectrum_moments(
        np.arange(5.0),
        [0, 0, 1, 0, 0],
        x_range=None,
        errors=np.ones(5) * 0.001,
        cutoff_fraction=0.0,
        uncertainty="propagate",
    )
    assert all(np.isnan(getattr(result, k + "_err")) for k in _KEYS)


def test_cube_root_skewness_is_not_propagated():
    x = np.linspace(-10, 10, 81)
    y = np.exp(-x * x / 8)
    result = spectrum_moments(
        x,
        y,
        x_range=None,
        cutoff_fraction=0.0,
        errors=np.full_like(x, 0.001),
        uncertainty="propagate",
    )
    assert np.isnan(result.skewness_err)
    assert np.isfinite(result.skewness_g1_err)
