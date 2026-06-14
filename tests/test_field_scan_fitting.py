"""Tests for ALC field-scan baseline + peak fitting (G1 core).

Covers ``asymmetry.core.fitting.field_scan``: the two-step ALC workflow on a
:class:`~asymmetry.core.transform.FieldScan` — fit a baseline over non-resonant
regions and subtract it, then fit a peak (centred ``GaussianLCR``) on the
corrected curve — reusing the existing parameter-model fitting machinery.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.fitting import (
    ScanBaselineResult,
    as_composite_model,
    fit_scan_baseline,
    fit_scan_model,
    parameter_set_for_model,
)
from asymmetry.core.transform import FieldScan

# --- fixtures ----------------------------------------------------------------


def _scan(x, value, error=None) -> FieldScan:
    x = np.asarray(x, dtype=float)
    value = np.asarray(value, dtype=float)
    if error is None:
        error = np.full(x.size, 1e-3)
    return FieldScan(
        x=x,
        value=value,
        error=np.asarray(error, dtype=float),
        run_numbers=list(range(1, x.size + 1)),
        order_key="field",
        method="integral",
        x_label="B (G)",
    )


def _gaussian_lcr(x, f, B0, Bwid):
    return f * np.exp(-0.5 * ((x - B0) / Bwid) ** 2)


def _lorentzian_lcr(x, f, B0, Bwid):
    return f / (1.0 + ((x - B0) / Bwid) ** 2)


# True model: a sloped linear baseline plus a Gaussian dip resonance at B0=100.
_TRUE_M, _TRUE_B = 0.01, 2.0
_TRUE_F, _TRUE_B0, _TRUE_BWID = -0.5, 100.0, 15.0


def _alc_scan() -> FieldScan:
    x = np.linspace(0.0, 200.0, 41)
    baseline = _TRUE_M * x + _TRUE_B
    peak = _gaussian_lcr(x, _TRUE_F, _TRUE_B0, _TRUE_BWID)
    return _scan(x, baseline + peak)


# --- model helpers -----------------------------------------------------------


def test_as_composite_model_accepts_name_expression_and_list():
    assert as_composite_model("Linear").param_names == ["m", "b"]
    assert as_composite_model(["Constant"]).param_names == ["c"]
    combo = as_composite_model("GaussianLCR + Constant")
    assert set(combo.param_names) >= {"f", "B0", "Bwid", "c"}


def test_parameter_set_for_model_uses_defaults_and_overrides():
    model = as_composite_model("Linear")
    params = parameter_set_for_model(model, {"m": 5.0})
    assert params["m"].value == pytest.approx(5.0)
    assert params["b"].value == pytest.approx(0.0)  # default


def test_parameter_set_for_model_rejects_unknown_override():
    model = as_composite_model("Linear")
    with pytest.raises(ValueError, match="Unknown parameter override"):
        parameter_set_for_model(model, {"slope": 5.0})  # typo for "m"


# --- baseline fit + subtract -------------------------------------------------


def test_fit_scan_baseline_recovers_linear_and_corrects():
    scan = _alc_scan()
    # Non-resonant edges, away from the dip at B0=100.
    result = fit_scan_baseline(scan, [(0.0, 40.0), (160.0, 200.0)], model="Linear")

    assert isinstance(result, ScanBaselineResult)
    assert result.success
    assert result.fit.parameters["m"].value == pytest.approx(_TRUE_M, abs=1e-4)
    assert result.fit.parameters["b"].value == pytest.approx(_TRUE_B, abs=2e-2)

    # The corrected curve removes the baseline: flat regions ~ 0, dip preserved.
    corrected = result.corrected
    edge = (corrected.x <= 40.0) | (corrected.x >= 160.0)
    assert np.allclose(corrected.value[edge], 0.0, atol=1e-3)
    centre = int(np.argmin(np.abs(corrected.x - _TRUE_B0)))
    assert corrected.value[centre] == pytest.approx(_TRUE_F, abs=2e-2)


def test_fit_scan_baseline_constant_model():
    x = np.linspace(0.0, 100.0, 21)
    scan = _scan(x, np.full(x.size, 3.0) + _gaussian_lcr(x, -0.4, 50.0, 8.0))
    result = fit_scan_baseline(scan, [(0.0, 20.0), (80.0, 100.0)], model="Constant")
    assert result.success
    assert result.fit.parameters["c"].value == pytest.approx(3.0, abs=1e-3)


# True cubic baseline coefficients (well-conditioned over x in [0, 200]).
_CUBIC = {"c0": 2.0, "c1": 0.01, "c2": -1e-4, "c3": 2e-7}


def _cubic_value(x):
    xx = np.asarray(x, dtype=float)
    return _CUBIC["c0"] + _CUBIC["c1"] * xx + _CUBIC["c2"] * xx**2 + _CUBIC["c3"] * xx**3


def test_cubic_component_evaluates_as_horner_polynomial():
    # The registered Cubic component is the WiMDA/Mantid ALC background.
    from asymmetry.core.fitting.parameter_models import PARAMETER_MODEL_COMPONENTS

    cubic = PARAMETER_MODEL_COMPONENTS["Cubic"]
    assert cubic.param_names == ["c0", "c1", "c2", "c3"]
    x = np.array([0.0, 50.0, 200.0])
    np.testing.assert_allclose(cubic.function(x, **_CUBIC), _cubic_value(x))


def test_fit_scan_baseline_recovers_cubic_and_corrects():
    # A curved/sloping ALC baseline Linear cannot match: a cubic background
    # plus a Gaussian dip resonance at B0=100.
    x = np.linspace(0.0, 200.0, 41)
    scan = _scan(x, _cubic_value(x) + _gaussian_lcr(x, _TRUE_F, _TRUE_B0, _TRUE_BWID))
    result = fit_scan_baseline(scan, [(0.0, 40.0), (160.0, 200.0)], model="Cubic")

    assert result.success
    for name, truth in _CUBIC.items():
        assert result.fit.parameters[name].value == pytest.approx(truth, abs=1e-2, rel=1e-2)

    # The corrected curve removes the cubic baseline: flat regions ~ 0, dip kept.
    corrected = result.corrected
    edge = (corrected.x <= 40.0) | (corrected.x >= 160.0)
    assert np.allclose(corrected.value[edge], 0.0, atol=1e-3)
    centre = int(np.argmin(np.abs(corrected.x - _TRUE_B0)))
    assert corrected.value[centre] == pytest.approx(_TRUE_F, abs=2e-2)


def test_fit_scan_baseline_cubic_rejects_underdetermined():
    # Three points cannot constrain the 4-parameter Cubic baseline.
    scan = _alc_scan()
    with pytest.raises(ValueError, match="at least 4"):
        fit_scan_baseline(scan, [(0.0, 10.0)], model="Cubic")


def test_fit_scan_baseline_preserves_scan_structure():
    scan = _alc_scan()
    result = fit_scan_baseline(scan, [(0.0, 40.0), (160.0, 200.0)])
    corrected = result.corrected
    assert corrected.n_points == scan.n_points
    assert np.array_equal(corrected.x, scan.x)
    assert corrected.run_numbers == scan.run_numbers
    assert corrected.derivative is False
    assert corrected.order_key == scan.order_key
    assert "baseline-subtracted" in corrected.y_label
    assert result.baseline.shape == scan.x.shape


# --- baseline validation -----------------------------------------------------


def test_fit_scan_baseline_requires_regions():
    scan = _alc_scan()
    with pytest.raises(ValueError, match="At least one baseline region"):
        fit_scan_baseline(scan, [])


def test_fit_scan_baseline_rejects_inverted_region():
    scan = _alc_scan()
    with pytest.raises(ValueError, match="x_lo < x_hi"):
        fit_scan_baseline(scan, [(40.0, 10.0)])


def test_fit_scan_baseline_rejects_empty_selection():
    scan = _alc_scan()
    with pytest.raises(ValueError, match="no usable scan points"):
        fit_scan_baseline(scan, [(1000.0, 2000.0)])


def test_fit_scan_baseline_rejects_underdetermined():
    scan = _alc_scan()
    # A single point can't constrain the 2-parameter Linear baseline.
    with pytest.raises(ValueError, match="at least 2"):
        fit_scan_baseline(scan, [(99.0, 101.0)], model="Linear")


def test_fit_scan_baseline_ignores_unusable_points_in_guard():
    # Region selects 3 x-points but two have non-positive / non-finite error, so
    # only 1 usable point remains — must be rejected, not fit to garbage.
    x = np.linspace(0.0, 200.0, 41)
    value = _TRUE_M * x + _TRUE_B
    error = np.full(x.size, 1e-3)
    # Points at x = 0, 5, 10 fall in the region below; spoil two of them.
    error[1] = 0.0
    value[2] = np.nan
    scan = _scan(x, value, error)
    with pytest.raises(ValueError, match="usable"):
        fit_scan_baseline(scan, [(0.0, 10.0)], model="Linear")


def test_fit_scan_baseline_raises_on_failed_fit_no_silent_corruption():
    # All region points have non-positive error -> the fitter has nothing to fit.
    # Must raise, NOT return a corrected scan built from default parameters.
    scan = _alc_scan()
    bad = FieldScan(
        x=scan.x,
        value=scan.value,
        error=np.zeros_like(scan.error),  # error == 0 everywhere
        run_numbers=scan.run_numbers,
        order_key=scan.order_key,
        method=scan.method,
        x_label=scan.x_label,
    )
    with pytest.raises(ValueError, match="no usable scan points"):
        fit_scan_baseline(bad, [(0.0, 40.0), (160.0, 200.0)], model="Linear")


# --- peak fit on the corrected curve -----------------------------------------


def test_fit_scan_model_recovers_resonance_position():
    scan = _alc_scan()
    baseline = fit_scan_baseline(scan, [(0.0, 40.0), (160.0, 200.0)], model="Linear")
    # Fit a centred Gaussian peak; seed near the truth.
    fit = fit_scan_model(
        baseline.corrected,
        "GaussianLCR",
        initial={"f": -0.4, "B0": 95.0, "Bwid": 20.0},
    )
    assert fit.success
    assert fit.parameters["B0"].value == pytest.approx(_TRUE_B0, abs=2.0)
    assert abs(fit.parameters["Bwid"].value) == pytest.approx(_TRUE_BWID, abs=2.0)
    assert fit.parameters["f"].value == pytest.approx(_TRUE_F, abs=5e-2)


def test_fit_scan_model_rejects_parameters_and_initial_together():
    scan = _alc_scan()
    params = parameter_set_for_model(as_composite_model("GaussianLCR"))
    with pytest.raises(ValueError, match="either .* or .* not both"):
        fit_scan_model(scan, "GaussianLCR", parameters=params, initial={"B0": 100.0})


def test_fit_scan_model_recovers_centred_lorentzian():
    # The new centred LorentzianLCR peak (f, B0, Bwid) is fittable like GaussianLCR.
    x = np.linspace(0.0, 200.0, 81)
    scan = _scan(x, _lorentzian_lcr(x, -0.5, 100.0, 12.0))
    fit = fit_scan_model(scan, "LorentzianLCR", initial={"f": -0.4, "B0": 95.0, "Bwid": 18.0})
    assert fit.success
    assert fit.parameters["B0"].value == pytest.approx(100.0, abs=2.0)
    assert abs(fit.parameters["Bwid"].value) == pytest.approx(12.0, abs=2.0)


def test_fit_scan_model_two_peak_composite():
    # Two centred peaks at distinct fields; a 2-component composite recovers both.
    x = np.linspace(0.0, 300.0, 151)
    value = _lorentzian_lcr(x, -0.4, 80.0, 10.0) + _gaussian_lcr(x, -0.3, 210.0, 14.0)
    scan = _scan(x, value)
    fit = fit_scan_model(
        scan,
        ["LorentzianLCR", "GaussianLCR"],
        initial={
            "f_1": -0.35,
            "B0_1": 85.0,
            "Bwid_1": 12.0,
            "f_2": -0.25,
            "B0_2": 205.0,
            "Bwid_2": 16.0,
        },
    )
    assert fit.success
    assert fit.parameters["B0_1"].value == pytest.approx(80.0, abs=3.0)
    assert fit.parameters["B0_2"].value == pytest.approx(210.0, abs=3.0)


def test_composite_component_param_name_single_and_multi():
    # Single peak: bare names; two peaks: index-suffixed by component position.
    single = as_composite_model("GaussianLCR")
    assert single.component_param_name(0, "B0") == "B0"
    pair = as_composite_model(["LorentzianLCR", "GaussianLCR"])
    assert pair.component_param_name(0, "B0") == "B0_1"
    assert pair.component_param_name(1, "B0") == "B0_2"


def test_lcr_components_carry_fwhm_factor():
    from asymmetry.core.fitting.parameter_models import PARAMETER_MODEL_COMPONENTS

    gauss = PARAMETER_MODEL_COMPONENTS["GaussianLCR"].fwhm_factor
    lorentz = PARAMETER_MODEL_COMPONENTS["LorentzianLCR"].fwhm_factor
    assert gauss == pytest.approx(2.0 * np.sqrt(2.0 * np.log(2.0)))  # ≈ 2.3548
    assert lorentz == pytest.approx(2.0)
    assert PARAMETER_MODEL_COMPONENTS["Linear"].fwhm_factor is None


def test_fit_scan_model_respects_x_range():
    scan = _alc_scan()
    fit = fit_scan_model(
        scan,
        "GaussianLCR + Linear",
        initial={"f": -0.4, "B0": 95.0, "Bwid": 20.0, "m": _TRUE_M, "b": _TRUE_B},
        x_min=40.0,
        x_max=160.0,
    )
    assert fit.success
    assert fit.parameters["B0"].value == pytest.approx(_TRUE_B0, abs=3.0)
