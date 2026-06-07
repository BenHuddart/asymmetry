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
    with pytest.raises(ValueError, match="select no scan points"):
        fit_scan_baseline(scan, [(1000.0, 2000.0)])


def test_fit_scan_baseline_rejects_underdetermined():
    scan = _alc_scan()
    # A single point can't constrain the 2-parameter Linear baseline.
    with pytest.raises(ValueError, match="at least 2 point"):
        fit_scan_baseline(scan, [(99.0, 101.0)], model="Linear")


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
