"""Example: fit superconducting sigma(T) models for penetration-depth analysis."""

from __future__ import annotations

import numpy as np

from asymmetry.core.fitting import Parameter, ParameterCompositeModel, ParameterSet
from asymmetry.core.fitting.parameter_models import fit_parameter_model
from asymmetry.core.fitting.sc.constants import sigma_to_lambda_nm
from asymmetry.core.fitting.sc.models import sc_two_gap_ss


def build_synthetic_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create MgB2-like synthetic sigma(T) data from a two-gap model."""
    rng = np.random.default_rng(42)
    temp = np.linspace(1.5, 35.0, 28)

    clean = sc_two_gap_ss(
        temp,
        sigma_0=1.25,
        Tc=36.0,
        gap_ratio_1=1.1,
        gap_ratio_2=2.3,
        weight=0.55,
        sigma_bg=0.03,
    )
    err = np.full_like(temp, 0.015)
    noisy = clean + rng.normal(0.0, err)
    return temp, noisy, err


def fit_single_gap_swave(temp: np.ndarray, sigma: np.ndarray, err: np.ndarray) -> None:
    model = ParameterCompositeModel(["SC_SWave"])
    params = ParameterSet(
        [
            Parameter("sigma_0", value=1.0, min=0.0),
            Parameter("Tc", value=35.0, min=0.0),
            Parameter("gap_ratio", value=1.764, min=0.2, max=5.0),
            Parameter("sigma_bg", value=0.01, min=0.0),
        ]
    )
    result = fit_parameter_model(temp, sigma, err, model, params)

    print("single-gap s-wave:")
    print("  success:", result.success)
    if not result.success:
        print("  message:", result.message)
        return
    print("  reduced chi2:", f"{result.reduced_chi_squared:.3f}")
    for p in result.parameters:
        print(f"  {p.name:10s} = {p.value:.5f}")



def fit_two_gap_ss(temp: np.ndarray, sigma: np.ndarray, err: np.ndarray) -> None:
    model = ParameterCompositeModel(["SC_TwoGap_SS"])
    params = ParameterSet(
        [
            Parameter("sigma_0", value=1.0, min=0.0),
            Parameter("Tc", value=35.0, min=0.0),
            Parameter("gap_ratio_1", value=1.0, min=0.2, max=5.0),
            Parameter("gap_ratio_2", value=2.2, min=0.2, max=5.0),
            Parameter("weight", value=0.5, min=0.0, max=1.0),
            Parameter("sigma_bg", value=0.01, min=0.0),
        ]
    )
    result = fit_parameter_model(temp, sigma, err, model, params)

    print("two-gap s+s:")
    print("  success:", result.success)
    if not result.success:
        print("  message:", result.message)
        return
    print("  reduced chi2:", f"{result.reduced_chi_squared:.3f}")
    values = {p.name: p.value for p in result.parameters}
    for p in result.parameters:
        print(f"  {p.name:10s} = {p.value:.5f}")

    sigma0_sc = max(values["sigma_0"] - values["sigma_bg"], 1e-12)
    lambda0_nm = float(sigma_to_lambda_nm(np.array([sigma0_sc]))[0])
    print("  estimated lambda(0) [nm]:", f"{lambda0_nm:.2f}")



def main() -> None:
    temp, sigma, err = build_synthetic_data()
    fit_single_gap_swave(temp, sigma, err)
    print()
    fit_two_gap_ss(temp, sigma, err)


if __name__ == "__main__":
    main()
