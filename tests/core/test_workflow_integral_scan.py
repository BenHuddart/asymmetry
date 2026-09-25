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


def _datasets(folder, runs):
    from asymmetry.core.io import load

    return [load(str(folder / f"SIM{run:08d}.nxs")) for run in runs]


def test_an_integral_scan_takes_the_settings_pair_and_deadtime(workflow_folder) -> None:
    from asymmetry.core.workflow.integral_scan import build_integral_scan
    from asymmetry.core.workflow.reduction import ReductionSettings
    from tests.core.conftest import DEADTIME_RUN, SCAN_RUNS

    datasets = _datasets(workflow_folder, SCAN_RUNS)
    plain = build_integral_scan(datasets, ReductionSettings(), order_key="run")
    swapped = build_integral_scan(datasets, ReductionSettings(pair=("2", "1")), order_key="run")
    assert np.allclose(swapped.value, -plain.value)

    corrected = build_integral_scan(
        datasets, ReductionSettings(deadtime="from_file"), order_key="run"
    )
    changed = {run for run, a, b in zip(plain.run_numbers, plain.value, corrected.value) if a != b}
    assert changed == {DEADTIME_RUN}


def test_a_green_minus_red_scan_differences_each_periods_integral() -> None:
    from asymmetry.core.data.dataset import MuonDataset
    from asymmetry.core.io.periods import period_run
    from asymmetry.core.simulate import BUILTIN_TEMPLATES, PeriodSpec, simulate_two_period_run
    from asymmetry.core.workflow.integral_scan import build_integral_scan
    from asymmetry.core.workflow.reduction import GREEN_MINUS_RED, ReductionSettings

    def relax(t, A=20.0):  # noqa: N803 (A is the conventional asymmetry symbol)
        return A * np.exp(-0.3 * t)

    datasets = []
    for index, (field, red_amplitude) in enumerate([(700.0, 20.0), (800.0, 12.0)]):
        template = BUILTIN_TEMPLATES["ideal_pulsed_fb"].build()
        template.metadata["field"] = field
        run = simulate_two_period_run(
            template,
            [
                PeriodSpec(relax, {"A": red_amplitude}, label="red"),
                PeriodSpec(relax, {"A": 20.0}, label="green"),
            ],
            total_events=4.0e7,
            seed=20 + index,
            run_number=40 + index,
        )
        datasets.append(
            MuonDataset(
                time=np.zeros(1), asymmetry=np.zeros(1), error=np.ones(1), metadata={}, run=run
            )
        )
    settings = ReductionSettings(period=GREEN_MINUS_RED)
    scan = build_integral_scan(datasets, settings, t_min=0.0, t_max=1.0)
    assert list(scan.x) == [700.0, 800.0]
    # Off resonance the RF changes nothing; on it the red amplitude drops by 8 %.
    assert scan.value[0] == pytest.approx(0.0, abs=0.004)
    assert scan.value[1] > 0.05

    # Each point is green's integral less red's, with their errors in quadrature.
    red_only, green_only = (
        build_integral_scan(
            [
                MuonDataset(
                    time=np.zeros(1),
                    asymmetry=np.zeros(1),
                    error=np.ones(1),
                    metadata={},
                    run=period_run(dataset.run, index),
                )
                for dataset in datasets
            ],
            ReductionSettings(),
            t_min=0.0,
            t_max=1.0,
        )
        for index in (0, 1)
    )
    assert np.allclose(scan.value, green_only.value - red_only.value)
    assert np.allclose(scan.error, np.hypot(red_only.error, green_only.error))
    assert scan.run_numbers == [40, 41]
    assert build_integral_scan(datasets, settings, method="differential").n_points == 2


def test_a_single_period_scan_reports_the_source_run_number() -> None:
    """``select_period`` encodes a period's run number as ``run*1000+period``;

    the scan must report the run it was cut from, not that internal key.
    """
    from asymmetry.core.data.dataset import MuonDataset
    from asymmetry.core.io.periods import select_period
    from asymmetry.core.simulate import BUILTIN_TEMPLATES, PeriodSpec, simulate_two_period_run
    from asymmetry.core.workflow.integral_scan import build_integral_scan
    from asymmetry.core.workflow.reduction import ReductionSettings

    def relax(t, A=20.0):  # noqa: N803 (A is the conventional asymmetry symbol)
        return A * np.exp(-0.3 * t)

    datasets = []
    for index, field in enumerate([700.0, 800.0]):
        template = BUILTIN_TEMPLATES["ideal_pulsed_fb"].build()
        template.metadata["field"] = field
        run = simulate_two_period_run(
            template,
            [
                PeriodSpec(relax, {"A": 20.0}, label="red"),
                PeriodSpec(relax, {"A": 15.0}, label="green"),
            ],
            total_events=4.0e7,
            seed=50 + index,
            run_number=500 + index,
        )
        loaded = MuonDataset(
            time=np.zeros(1), asymmetry=np.zeros(1), error=np.ones(1), metadata={}, run=run
        )
        datasets.append(select_period(loaded, "red"))

    settings = ReductionSettings(period="red")
    scan = build_integral_scan(datasets, settings, t_min=0.0, t_max=1.0)
    by_run = build_integral_scan(datasets, settings, t_min=0.0, t_max=1.0, order_key="run")

    assert scan.run_numbers == [500, 501]
    assert by_run.x.tolist() == [500.0, 501.0]


@pytest.mark.parametrize("background", ["Quadratic", "Cubic"])
def test_two_resonances_on_a_kilogauss_background_fit_together(background: str) -> None:
    rng = np.random.default_rng(1)
    x = np.arange(19000.0, 30000.0, 50.0)

    def lorentzian(f, b0, width):
        return f / (1.0 + ((x - b0) / width) ** 2)

    error = np.full_like(x, 1.5e-3)
    value = (
        0.25
        + 2e-6 * (x - 19000.0)
        + 3e-11 * (x - 19000.0) ** 2
        + lorentzian(-0.02, 20800.0, 60.0)
        + lorentzian(-0.012, 27500.0, 150.0)
        + rng.normal(0.0, error)
    )
    scan = FieldScan(
        x=x,
        value=value,
        error=error,
        run_numbers=list(range(x.size)),
        order_key="field",
        method="integral",
        x_label="B (G)",
    )
    _, fit = fit_integral_scan(scan, f"LorentzianLCR + LorentzianLCR + {background}")
    parameters = fit["parameters"]
    assert fit["success"]
    assert fit["reduced_chi_squared"] < 1.5
    assert parameters["B0_1"] == pytest.approx(20800.0, abs=10.0)
    assert parameters["B0_2"] == pytest.approx(27500.0, abs=30.0)
    assert parameters["Bwid_1"] > 0.0 and parameters["Bwid_2"] > 0.0
