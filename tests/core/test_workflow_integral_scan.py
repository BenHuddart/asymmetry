"""Workflow façade tests for agent-driven integral field scans."""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.transform import FieldScan
from asymmetry.core.workflow.integral_scan import field_scan_payload, fit_integral_scan


def _scan() -> FieldScan:
    x = np.linspace(2000.0, 5000.0, 61)
    peak = -0.04 / (1.0 + ((x - 3500.0) / 180.0) ** 2)
    return FieldScan(
        x=x,
        value=0.18 + peak,
        error=np.full_like(x, 5e-4),
        run_numbers=list(range(100, 161)),
        order_key="field",
        method="integral",
        x_label="B (G)",
    )


def test_field_scan_payload_keeps_units_points_and_run_provenance() -> None:
    data = field_scan_payload(_scan())

    assert data["order_key"] == "field"
    assert data["units"] == "fraction"
    assert data["points"][0]["run"] == 100
    assert data["points"][-1]["x"] == pytest.approx(5000.0)


def test_fit_integral_scan_seeds_an_off_zero_lorentzian_from_the_data() -> None:
    _fitted, result = fit_integral_scan(_scan(), "LorentzianLCR + Constant")

    assert result["success"]
    assert result["parameters"]["B0"] == pytest.approx(3500.0, abs=20.0)
    assert abs(result["parameters"]["Bwid"]) == pytest.approx(180.0, abs=20.0)
    assert result["parameters"]["c"] == pytest.approx(0.18, abs=0.005)


def test_fit_integral_scan_seeds_cubic_on_a_kilogauss_axis() -> None:
    x = np.linspace(2000.0, 5000.0, 31)
    centred = x - 3500.0
    baseline = 0.235 + 2e-6 * centred - 3e-10 * centred**2 + 1e-13 * centred**3
    resonance = -0.065 / (1.0 + ((x - 3100.0) / 190.0) ** 2)
    scan = FieldScan(
        x=x,
        value=baseline + resonance,
        error=np.full_like(x, 4e-4),
        run_numbers=list(range(200, 231)),
        order_key="field",
        method="integral",
        x_label="B (G)",
    )

    _fitted, result = fit_integral_scan(scan, "LorentzianLCR + Cubic")

    assert result["success"] is True
    assert result["parameters"]["B0"] == pytest.approx(3100.0, abs=5.0)
    assert result["parameters"]["Bwid"] == pytest.approx(190.0, rel=0.1)


def test_fit_integral_scan_can_subtract_a_baseline_first() -> None:
    fitted, result = fit_integral_scan(
        _scan(),
        "LorentzianLCR",
        baseline_model="Constant",
        baseline_regions=[(2000.0, 2600.0), (4400.0, 5000.0)],
    )

    assert result["success"]
    assert result["baseline"]["model"] == "Constant"
    assert fitted.y_label.endswith("(baseline-subtracted)")
