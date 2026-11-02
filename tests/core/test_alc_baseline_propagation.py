"""Independent linear-algebra checks of fitted-baseline propagation."""

from dataclasses import replace

import numpy as np
import pytest

from asymmetry.core.fitting.field_scan import (
    as_composite_model,
    fit_scan_baseline,
    fit_scan_peaks,
    parameter_set_for_model,
)
from asymmetry.core.transform import FieldScan


def scan(x, y, error=1.0):
    return FieldScan(
        x=x,
        value=y,
        error=np.broadcast_to(error, x.shape).copy(),
        run_numbers=list(range(len(x))),
        order_key="field",
        method="integral",
    )


def test_corrected_covariance_matches_linear_algebra():
    x = np.arange(11.0)
    source = scan(x, 0.2 * x + 2)
    result = fit_scan_baseline(source, [(0, 2)])
    design = np.column_stack((x, np.ones_like(x)))
    selected = np.eye(len(x))[:3]
    residual = (
        np.eye(len(x))
        - design @ np.linalg.solve(design[:3].T @ design[:3], design[:3].T) @ selected
    )
    np.testing.assert_allclose(result.correction_operator, residual, atol=1e-7)
    np.testing.assert_allclose(result.corrected_covariance, residual @ residual.T, atol=1e-6)
    np.testing.assert_allclose(
        result.corrected.error[[0, 1, 10]],
        [0.408248290464, 0.816496580928, 6.467869303977],
        rtol=1e-7,
    )


@pytest.mark.parametrize("regions", [[(0, 25), (75, 100)], [(0, 100)], [(0, 35)]])
def test_amplitude_only_peak_matches_closed_form(regions):
    x = np.linspace(0, 100, 61)
    peak = np.exp(-0.5 * ((x - 50) / 4) ** 2)
    source = scan(x, 0.2 + 0.001 * (x - 50) - 0.02 * peak, 0.002)
    baseline = fit_scan_baseline(source, regions)
    model = as_composite_model("GaussianLCR")
    parameters = parameter_set_for_model(model, {"f": -0.02, "B0": 50, "Bwid": 4})
    parameters["B0"].fixed = True
    parameters["Bwid"].fixed = True
    result = fit_scan_peaks(baseline, model, parameters=parameters)
    influence = peak @ baseline.correction_operator / (peak @ peak)
    expected = np.sum((influence * source.error) ** 2)
    assert result.covariance[0] == ["f"]
    assert result.covariance[1][0][0] == pytest.approx(expected, rel=2e-6)
    assert result.uncertainties["f"] == pytest.approx(np.sqrt(expected), rel=2e-6)


def test_fixed_baseline_limit():
    x = np.linspace(0, 100, 61)
    baseline = fit_scan_baseline(
        scan(x, 0.2 - 0.02 * np.exp(-0.5 * ((x - 50) / 4) ** 2), 0.002), [(0, 25), (75, 100)]
    )
    names, matrix = baseline.fit.covariance
    baseline = replace(
        baseline, fit=replace(baseline.fit, covariance=(names, np.zeros_like(matrix).tolist()))
    )
    result = fit_scan_peaks(baseline, "GaussianLCR", initial={"f": -0.02, "B0": 50, "Bwid": 4})
    np.testing.assert_array_equal(result.covariance[1], result.fit.covariance[1])


def test_all_fixed_baseline_raises():
    model = as_composite_model("Constant")
    parameters = parameter_set_for_model(model)
    parameters["c"].fixed = True
    x = np.arange(5.0)
    with pytest.raises(ValueError, match="free parameter"):
        fit_scan_baseline(scan(x, np.ones_like(x)), [(0, 4)], model=model, parameters=parameters)


def test_corrected_errors_match_seeded_repeated_baseline_fits():
    rng = np.random.default_rng(20261031)
    x = np.arange(11.0)
    corrected = []
    for _ in range(400):
        result = fit_scan_baseline(scan(x, 0.2 * x + 2 + rng.normal(size=x.size)), [(0, 2)])
        corrected.append(result.corrected.value)
    observed = np.std(corrected, axis=0, ddof=1)
    np.testing.assert_allclose(observed, result.corrected.error, rtol=0.12)


def test_extrapolated_baseline_coverage():
    rng = np.random.default_rng(20261031)
    x = np.linspace(0, 100, 61)
    truth = 0.2 + 0.001 * (x - 50) - 0.02 * np.exp(-0.5 * ((x - 50) / 4) ** 2)
    estimates, errors, conditional, chi2, dof = [], [], [], [], []
    for _ in range(400):
        baseline = fit_scan_baseline(
            scan(x, truth + rng.normal(0, 0.002, x.size), 0.002), [(0, 35)]
        )
        result = fit_scan_peaks(baseline, "GaussianLCR", initial={"f": -0.02, "B0": 50, "Bwid": 4})
        estimates.append([result.fit.parameters[n].value for n in ("f", "Bwid")])
        errors.append([result.uncertainties[n] for n in ("f", "Bwid")])
        conditional.append(result.fit.uncertainties["Bwid"])
        chi2.append(result.fit.chi_squared)
        dof.append(result.effective_dof)
    estimates, errors = np.asarray(estimates), np.asarray(errors)
    np.testing.assert_allclose(estimates.std(axis=0, ddof=1), np.median(errors, axis=0), rtol=0.12)
    coverage = np.mean(abs(estimates - [-0.02, 4]) <= errors, axis=0)
    assert np.all((coverage >= 0.61) & (coverage <= 0.75))
    assert np.mean(abs(estimates[:, 1] - 4) <= conditional) < 0.55
    assert np.mean(chi2) == pytest.approx(np.mean(dof), rel=0.03)


def test_region_width_and_joint_fit_cross_check():
    from asymmetry.core.fitting.field_scan import fit_scan_model

    x = np.linspace(0, 100, 61)
    source = scan(x, 0.2 + 0.001 * (x - 50) - 0.02 * np.exp(-0.5 * ((x - 50) / 4) ** 2), 0.002)
    results = []
    for regions in ([(0, 12), (88, 100)], [(0, 30), (70, 100)]):
        baseline = fit_scan_baseline(source, regions, model="Cubic")
        results.append(
            fit_scan_peaks(baseline, "GaussianLCR", initial={"f": -0.02, "B0": 50, "Bwid": 4})
        )
    assert results[1].uncertainties["f"] < results[0].uncertainties["f"]
    assert results[1].uncertainties["Bwid"] < results[0].uncertainties["Bwid"]
    linear = fit_scan_baseline(source, [(0, 30), (70, 100)])
    propagated = fit_scan_peaks(linear, "GaussianLCR", initial={"f": -0.02, "B0": 50, "Bwid": 4})
    joint = fit_scan_model(
        source,
        "GaussianLCR + Linear",
        initial={"f": -0.02, "B0": 50, "Bwid": 4, "m": 0.001, "b": 0.15},
    )
    for name in ("f", "B0", "Bwid"):
        assert propagated.uncertainties[name] == pytest.approx(joint.uncertainties[name], rel=0.1)


def test_percent_scale_covariance():
    x = np.linspace(0, 100, 61)
    source = scan(x, 0.2 + 0.001 * (x - 50) - 0.02 * np.exp(-0.5 * ((x - 50) / 4) ** 2), 0.002)
    results = []
    for scale in (1, 100):
        scaled = replace(source, value=source.value * scale, error=source.error * scale)
        baseline = fit_scan_baseline(scaled, [(0, 30), (70, 100)])
        results.append(
            fit_scan_peaks(
                baseline, "GaussianLCR", initial={"f": -0.02 * scale, "B0": 50, "Bwid": 4}
            )
        )
    assert results[1].uncertainties["f"] == pytest.approx(
        100 * results[0].uncertainties["f"], rel=1e-5
    )
    assert results[1].uncertainties["B0"] == pytest.approx(results[0].uncertainties["B0"], rel=1e-5)


def test_fixed_baseline_limit_with_error_floor():
    x = np.linspace(0, 100, 61)
    errors = np.full(x.size, 0.002)
    errors[30] = 1e-6
    baseline = fit_scan_baseline(
        scan(x, 0.2 - 0.02 * np.exp(-0.5 * ((x - 50) / 4) ** 2), errors), [(0, 25), (75, 100)]
    )
    names, matrix = baseline.fit.covariance
    baseline = replace(
        baseline, fit=replace(baseline.fit, covariance=(names, np.zeros_like(matrix).tolist()))
    )
    result = fit_scan_peaks(baseline, "GaussianLCR", initial={"f": -0.02, "B0": 50, "Bwid": 4})
    np.testing.assert_array_equal(result.covariance[1], result.fit.covariance[1])


def test_field_unit_rescaling_preserves_propagated_errors():
    x = np.linspace(0, 100, 61)
    source = scan(x, 0.2 + 0.001 * (x - 50) - 0.02 * np.exp(-0.5 * ((x - 50) / 4) ** 2), 0.002)
    results = []
    for scale in (1, 1e-5):
        scaled = replace(source, x=source.x * scale)
        baseline = fit_scan_baseline(scaled, [(0, 30 * scale), (70 * scale, 100 * scale)])
        results.append(
            fit_scan_peaks(
                baseline, "GaussianLCR", initial={"f": -0.02, "B0": 50 * scale, "Bwid": 4 * scale}
            )
        )
    assert results[1].uncertainties["f"] == pytest.approx(results[0].uncertainties["f"], rel=1e-4)
    assert results[1].uncertainties["B0"] == pytest.approx(
        1e-5 * results[0].uncertainties["B0"], rel=1e-4
    )


@pytest.mark.parametrize(
    "covariance",
    [None, (["m", "b"], [[-1.0, 0.0], [0.0, 1.0]]), (["m", "b"], [[np.nan, 0.0], [0.0, 1.0]])],
)
def test_baseline_propagation_rejects_unavailable_covariance(covariance):
    x = np.arange(11.0)
    baseline = fit_scan_baseline(scan(x, 0.2 * x + 2), [(0, 2)])
    invalid = replace(baseline, fit=replace(baseline.fit, covariance=covariance))
    with pytest.raises(ValueError, match="covariance"):
        _ = invalid.corrected


def test_baseline_result_owns_source_snapshot():
    x = np.arange(11.0)
    source = scan(x, 0.2 * x + 2)
    result = fit_scan_baseline(source, [(0, 2)])
    expected = result.corrected.value.copy()
    source.value[:] = 0
    source.error[:] = 10
    source.x[:] = 1
    np.testing.assert_array_equal(result.corrected.value, expected)


def test_narrow_cubic_baseline_amplitude_coverage():
    rng = np.random.default_rng(20261031)
    x = np.linspace(0, 100, 61)
    truth = 0.2 + 0.001 * (x - 50) - 0.02 * np.exp(-0.5 * ((x - 50) / 4) ** 2)
    amplitudes, errors = [], []
    for _ in range(600):
        baseline = fit_scan_baseline(
            scan(x, truth + rng.normal(0, 0.002, x.size), 0.002),
            [(0, 12), (88, 100)],
            model="Cubic",
        )
        result = fit_scan_peaks(baseline, "GaussianLCR", initial={"f": -0.02, "B0": 50, "Bwid": 4})
        amplitudes.append(result.fit.parameters["f"].value)
        errors.append(result.uncertainties["f"])
    assert np.std(amplitudes, ddof=1) == pytest.approx(np.median(errors), rel=0.12)
    coverage = np.mean(abs(np.asarray(amplitudes) + 0.02) <= errors)
    assert 0.61 <= coverage <= 0.75
