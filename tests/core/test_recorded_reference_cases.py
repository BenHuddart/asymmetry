"""Regression against recorded execution, with separate independent anchors."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.engine import FitEngine
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.fourier.fft import fft_complex_asymmetry
from asymmetry.core.transform.rebin import rebin_counts
from tests.reference_cases import load_reference_case

FIXTURES = Path(__file__).parents[1] / "fixtures"


def test_recorded_constant_fit_against_independent_normal_equations():
    case = load_reference_case(FIXTURES / "reference_cases/constant/reference.json")
    settings = case["settings"]
    data = np.loadtxt(FIXTURES / "reference_cases/constant/owned.dat", skiprows=3, delimiter=",")
    np.testing.assert_array_equal(data, np.array([settings[k] for k in ("x", "y", "sigma")]).T)
    assert settings["selected_indices"] == [0, 1, 2]
    x, y, sigma = data.T
    weights = 1 / sigma**2
    mean = sum(weights * y) / sum(weights)
    error = 1 / np.sqrt(sum(weights))
    chi2 = sum(((y - mean) / sigma) ** 2)
    np.testing.assert_allclose([mean, error, chi2], [40 / 9, 2 / 3, 113 / 9], rtol=1e-14)
    result = FitEngine().fit(
        MuonDataset(x, y, sigma), lambda t, a: np.full_like(t, a), ParameterSet([Parameter("a", 3)])
    )
    assert result.success
    np.testing.assert_allclose(
        [result.parameters["a"].value, result.uncertainties["a"], result.chi_squared],
        [mean, error, chi2],
        atol=1e-5,
        rtol=0,
    )
    for name, output in case["native_output"].items():
        assert output["ndf"] == len(x) - 1
        objective = (
            output["chi_squared_per_ndf"] * output["ndf"]
            if name == "Mantid"
            else output["chi_squared"]
        )
        tolerance = case["tolerances"][name]
        np.testing.assert_allclose(
            [output["value"], output["standard_error"], objective],
            [mean, error, chi2],
            atol=tolerance["atol"],
            rtol=tolerance["rtol"],
        )


@pytest.mark.parametrize("packing,native_groups", [(1, 40), (3, 13), (4, 9)])
def test_recorded_count_packing_and_explicit_terminal_difference(packing, native_groups):
    case = load_reference_case(FIXTURES / "psi_root/reference.json")
    output = case["native_output"]
    native = np.asarray(
        output["summed_counts"] if packing == 1 else output["packed_counts"][str(packing)]
    )
    assert native_groups == case["mapping"]["packed_native_groups"][str(packing)]
    assert native.shape == (native_groups, 3)
    raw = np.arange(20, 60)
    counts = 100 + raw + 300 + 3 * (raw + 3)
    time = (raw - 4) * 0.01
    # Independent sums use the complete selected observation, including its tail.
    groups = len(raw) // packing
    summed = counts[: groups * packing].reshape(-1, packing).sum(axis=1)
    centres = time[: groups * packing].reshape(-1, packing).mean(axis=1)
    actual_time, actual_counts = rebin_counts(time, counts, packing)
    np.testing.assert_allclose(actual_time, centres, rtol=1e-14)
    np.testing.assert_array_equal(actual_counts, summed)
    assert groups - native_groups == (1 if packing == 4 else 0)
    np.testing.assert_array_equal(native[:, 1], summed[:native_groups])
    np.testing.assert_allclose(native[:, 0], centres[:native_groups], rtol=3e-8)
    np.testing.assert_allclose(native[:, 2], np.sqrt(summed[:native_groups]), rtol=5e-6)


def test_recorded_native_manual_phase_and_unrotated_endpoint():
    folder = FIXTURES / "reference_cases/manual-phase"
    case = load_reference_case(folder / "reference.json")
    output = case["native_output"]
    path = folder / output["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == output["sha256"]
    data = np.loadtxt(path, delimiter=",")
    assert data.shape == (output["bins"], 5)
    assert np.all(np.diff(data[:, 0]) > 0)
    zero = data[:, 1] + 1j * data[:, 2]
    phased = data[:, 3] + 1j * data[:, 4]
    assert case["mapping"]["rotation_sign"] == -1
    assert case["mapping"]["phase_unit"] == "degree"
    assert case["mapping"]["rotated_bins"] == [0, len(data) - 1]
    assert case["mapping"]["unrotated_bins"] == [len(data) - 1]
    assert case["mapping"]["normalization_scale"] == 1
    phi = np.deg2rad(case["settings"]["phase_degrees"][1])
    expected = (zero.real * np.cos(phi) + zero.imag * np.sin(phi)) + 1j * (
        zero.imag * np.cos(phi) - zero.real * np.sin(phi)
    )
    bound = case["tolerances"]["rotation"]["rtol"] * (abs(zero) + abs(phased))
    residual = abs(phased - expected)
    # The entire output is accounted for: native final bin is explicitly unrotated.
    assert np.all(residual[:-1] <= bound[:-1])
    assert phased[-1] == zero[-1]
    assert residual[-1] > bound[-1]
    assert np.max(abs(phased - zero)) > 1e6


def test_direct_dft_manual_phase_sign_and_time_origin():
    t = 0.037 + np.arange(32) * 0.01
    y = 17 * np.cos(2 * np.pi * 1.3 * t + 0.41)
    dataset = MuonDataset(t, y, np.ones(32))
    frequencies, actual = fft_complex_asymmetry(
        dataset, phase_degrees=37, subtract_average_signal=False
    )
    # The uncalibrated finite DFT uses samples measured from t[0], including DC.
    expected = np.array(
        [sum(y * np.exp(-2j * np.pi * f * (t - t[0]))) for f in frequencies]
    ) * np.exp(-1j * np.deg2rad(37))
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
