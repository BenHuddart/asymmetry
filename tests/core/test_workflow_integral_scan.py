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


def test_shape_candidates_recover_an_interior_dip_without_a_resonance_model() -> None:
    from asymmetry.core.workflow.integral_scan import candidate_dip_windows

    scan = _scan()
    (window,) = candidate_dip_windows(scan)
    assert window["centre"] == pytest.approx(3500)
    assert window["x_min"] < 3500 - 180 < 3500 + 180 < window["x_max"]


def test_shape_candidates_reject_monotonic_recovery_and_sub_error_structure() -> None:
    from dataclasses import replace

    from asymmetry.core.workflow.integral_scan import candidate_dip_windows

    scan = _scan()
    recovery = 0.2 - 0.1 * np.exp(-(scan.x - 2000) / 800)
    assert candidate_dip_windows(replace(scan, value=recovery)) == []
    assert candidate_dip_windows(replace(scan, value=0.2 + 1e-4 * np.sin(scan.x))) == []
    assert candidate_dip_windows(replace(scan, order_key="temperature")) == []


def test_shape_candidates_keep_separated_dips_on_a_sloping_background() -> None:
    from dataclasses import replace

    from asymmetry.core.workflow.integral_scan import candidate_dip_windows

    scan = _scan()
    values = (
        0.18
        + 1e-6 * scan.x
        - 0.02 / (1 + ((scan.x - 2800) / 100) ** 2)
        - 0.03 / (1 + ((scan.x - 4200) / 150) ** 2)
    )
    windows = candidate_dip_windows(replace(scan, value=values))
    assert [w["centre"] for w in windows] == pytest.approx([2800, 4200], abs=50)


def test_fit_integral_scan_seeds_an_off_zero_lorentzian_from_the_data() -> None:
    _fitted, result = fit_integral_scan(_scan(), "LorentzianLCR + Constant")

    assert result["success"]
    assert result["parameters"]["B0"] == pytest.approx(3500.0, abs=20.0)
    assert abs(result["parameters"]["Bwid"]) == pytest.approx(180.0, abs=20.0)
    assert result["parameters"]["c"] == pytest.approx(0.18, abs=0.005)


def test_explicit_width_bounds_recover_a_broad_line_and_keep_the_interval() -> None:
    from dataclasses import replace

    scan = _scan()
    scan = replace(scan, value=0.18 - 0.04 / (1 + ((scan.x - 3500) / 1800) ** 2))
    _, default = fit_integral_scan(scan, "LorentzianLCR + Constant")
    _, expanded = fit_integral_scan(
        scan, "LorentzianLCR + Constant", bounds={"Bwid": (10.0, 2000.0)}
    )
    assert default["parameters"]["Bwid"] <= 1500.01
    assert expanded["success"]
    assert expanded["parameters"]["Bwid"] == pytest.approx(1800, rel=0.01)
    assert expanded["parameters"]["B0"] == pytest.approx(3500, abs=1)
    assert "Bwid" not in expanded["params_at_bound"]
    assert expanded["bounds"] == {"Bwid": [10.0, 2000.0]}


def test_width_search_does_not_truncate_a_partially_covered_line() -> None:
    from dataclasses import replace

    from asymmetry.core.workflow.integral_scan import _held_line

    scan = _scan()
    scan = replace(scan, value=0.18 - 0.04 / (1 + ((scan.x - 3500) / 1000) ** 2))
    _, fit = fit_integral_scan(scan, "LorentzianLCR + Constant")
    assert fit["success"]
    assert fit["parameters"]["Bwid"] == pytest.approx(1000, rel=0.01)
    assert "Bwid" not in fit["params_at_bound"]
    # A fitted width with partial flanks still does not establish an isolated line.
    assert _held_line(scan, {"x_min": 2000, "x_max": 5000}) is None


@pytest.mark.parametrize(
    ("bounds", "initial", "message"),
    [
        ({"missing": (1, 2)}, {}, "Unknown bounded parameter"),
        ({"Bwid": (1, float("inf"))}, {}, "finite"),
        ({"Bwid": (2, 1)}, {}, "LO < HI"),
        ({"Bwid": (-1, 2)}, {}, "positive"),
        ({"Bwid": (10, 100)}, {"Bwid": 180}, "outside its bounds"),
    ],
)
def test_invalid_scan_bounds_are_rejected_at_the_api_boundary(bounds, initial, message) -> None:
    with pytest.raises(ValueError, match=message):
        fit_integral_scan(_scan(), "LorentzianLCR + Constant", bounds=bounds, initial=initial)


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
    assert list(corrected.run_numbers) == [DEADTIME_RUN]
    baseline = dict(zip(plain.run_numbers, plain.value, strict=True))
    assert corrected.value[0] != baseline[DEADTIME_RUN]
    assert {run for run, _reason in corrected.excluded} == set(SCAN_RUNS) - {DEADTIME_RUN}
    assert all("deadtime correction requested" in reason for _run, reason in corrected.excluded)


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


def test_a_differential_pair_with_its_offset_held_recovers_each_resonance() -> None:
    from asymmetry.core.fitting.parameter_models import _lcr_lorentzian_pair

    rng = np.random.default_rng(3)
    x = np.sort(np.concatenate([np.arange(28500.0, 30000.0, 100.0) + d for d in (0, 20, 40)]))
    error = np.full_like(x, 5e-4)
    value = (
        _lcr_lorentzian_pair(x, -0.016, 28938.5, 14.0, 44.4)
        + _lcr_lorentzian_pair(x, -0.017, 29536.3, 14.0, 44.4)
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

    _, fit = fit_integral_scan(
        scan,
        "LorentzianLCRPair + LorentzianLCRPair + Constant",
        fixed={"dB_1": 44.4, "dB_2": 44.4},
    )

    parameters = fit["parameters"]
    assert fit["success"]
    assert sorted([parameters["B0_1"], parameters["B0_2"]]) == pytest.approx(
        [28938.5, 29536.3], abs=2.0
    )
    assert parameters["Bwid_1"] == pytest.approx(14.0, abs=2.0)
    assert parameters["Bwid_2"] == pytest.approx(14.0, abs=2.0)
    assert parameters["dB_1"] == parameters["dB_2"] == 44.4


def test_a_failed_fit_carries_the_engines_reason_and_a_window_per_resonance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    from asymmetry.core.workflow import integral_scan

    # Whether a minimiser converges on noise depends on the platform's
    # arithmetic, so the fit is made to fail: what is tested is the payload.
    real_fit = integral_scan.fit_scan_model

    def failing_fit(*args, **kwargs):
        return replace(
            real_fit(*args, **kwargs), success=False, message="Fit failed: call limit reached"
        )

    monkeypatch.setattr(integral_scan, "fit_scan_model", failing_fit)
    x = np.linspace(0.0, 100.0, 21)
    scan = FieldScan(
        x=x,
        value=np.random.default_rng(0).normal(0.0, 1e-3, x.size),
        error=np.full_like(x, 1e-3),
        run_numbers=list(range(x.size)),
        order_key="field",
        method="integral",
        x_label="B (G)",
    )
    _, fit = fit_integral_scan(scan, "LorentzianLCR + LorentzianLCR + Cubic")
    assert not fit["success"]
    assert fit["message"] == "Fit failed: call limit reached"
    assert [window["parameter"] for window in fit["resonance_windows"]] == ["B0_1", "B0_2"]
    assert fit["initial"] == {}


def test_each_resonance_window_surrounds_its_own_dip() -> None:
    from asymmetry.core.fitting.field_scan import as_composite_model
    from asymmetry.core.workflow.integral_scan import resonance_windows

    x = np.arange(17000.0, 23050.0, 50.0)
    value = (
        0.30
        - 0.015 / (1.0 + ((x - 19500.0) / 200.0) ** 2)
        - 0.008 / (1.0 + ((x - 21450.0) / 110.0) ** 2)
    )
    scan = FieldScan(
        x=x,
        value=value,
        error=np.full_like(x, 4e-4),
        run_numbers=list(range(x.size)),
        order_key="field",
        method="integral",
        x_label="B (G)",
    )
    windows = resonance_windows(scan, as_composite_model("LorentzianLCR + LorentzianLCR"), {})
    # In field order, whichever component the seeding put on each dip.
    assert [window["centre"] for window in windows] == pytest.approx([19500.0, 21450.0], abs=50)
    first, second = windows
    assert first["x_min"] < 19500.0 < first["x_max"] <= second["x_min"] < 21450.0
    assert second["x_max"] > 21450.0
    assert first["x_max"] <= (first["centre"] + second["centre"]) / 2.0


def test_a_successful_fit_has_no_resonance_windows() -> None:
    _, fit = fit_integral_scan(_scan(), "LorentzianLCR + Constant", initial={"B0": 3400.0})
    assert fit["success"]
    assert fit["resonance_windows"] == []
    assert fit["initial"] == {"B0": 3400.0}


def test_only_a_window_holding_a_line_is_named_as_another_dip() -> None:
    from asymmetry.core.transform.integral import FieldScan
    from asymmetry.core.workflow.integral_scan import _held_line

    x = np.linspace(1000.0, 3000.0, 81)
    error = np.full_like(x, 0.0005)
    dip = 0.2 - 0.03 / (1.0 + ((x - 2200.0) / 40.0) ** 2) + 1e-6 * (x - 2000.0)
    # A background rising across the window, as an ALC scan's does near a step.
    step = 0.2 + 0.03 * np.tanh((x - 2200.0) / 400.0)

    def scan(value) -> FieldScan:
        return FieldScan(x, value, error, list(range(x.size)), "field", "integral")

    window = {"x_min": 1900.0, "x_max": 2500.0}
    assert _held_line(scan(dip), window) == pytest.approx(2200.0, abs=10.0)
    assert _held_line(scan(step), window) is None


def test_every_dip_beyond_the_fitted_line_gets_its_own_window() -> None:
    from asymmetry.core.transform.integral import FieldScan

    # A hot ALC scan: a deep line, a smaller resolved one well below it and a
    # third above, on a falling background (corannulene at 420 K, Haiku 2026-10-21).
    x = np.arange(5000.0, 24601.0, 200.0)
    error = np.full_like(x, 0.0004)
    value = 0.2 - 2e-6 * (x - 5000.0)
    for centre, depth in ((7000.0, 0.015), (14800.0, 0.03), (20000.0, 0.012)):
        value = value - depth / (1.0 + ((x - centre) / 400.0) ** 2)
    scan = FieldScan(x, value, error, list(range(x.size)), "field", "integral")

    _, fit = fit_integral_scan(scan, "LorentzianLCR + Linear", x_min=13000.0, x_max=17000.0)
    assert fit["success"]
    dips = [
        centre
        for centre in (7000.0, 14800.0, 20000.0)
        if any(w["x_min"] <= centre <= w["x_max"] for w in fit["next_dip_windows"])
    ]
    assert dips == [7000.0, 14800.0, 20000.0]


def test_a_wide_minimum_reaching_a_deeper_dip_is_not_named_twice(monkeypatch) -> None:
    from asymmetry.core.transform.integral import FieldScan
    from asymmetry.core.workflow import integral_scan

    x = np.arange(5000.0, 30001.0, 200.0)
    error = np.full_like(x, 0.0004)
    value = 0.2 - 0.03 / (1.0 + ((x - 15500.0) / 600.0) ** 2)
    scan = FieldScan(x, value, error, list(range(x.size)), "field", "integral")
    # The cold corannulene scan's two minima: a shallow wide one whose window
    # reaches the deep line and fits it, and the deep line's own.
    wide = {"centre": 10500.0, "x_min": 6600.0, "x_max": 17400.0, "prominence": 0.002}
    deep = {"centre": 15400.0, "x_min": 13900.0, "x_max": 16900.0, "prominence": 0.03}
    monkeypatch.setattr(integral_scan, "candidate_dip_windows", lambda _scan: [wide, deep])
    assert integral_scan._held_line(scan, wide) == pytest.approx(15500.0, abs=10.0)
    assert integral_scan._further_dips(scan, []) == [deep]


def test_a_tf_stamp_on_a_kilogauss_run_that_keeps_its_integral_is_contradicted() -> None:
    from types import SimpleNamespace

    from asymmetry.core.workflow.integral_scan import contradicted_tf_stamps

    def point(run: int, value: float) -> SimpleNamespace:
        return SimpleNamespace(run_number=run, value=value, error=0.0004)

    scan = SimpleNamespace(
        points=[point(1, 0.002), point(2, 0.19), point(3, 0.20), point(4, 0.21), point(5, 0.18)]
    )
    # A kilogauss TF run keeping a large integral is longitudinal; a 100 G
    # calibration run (too few periods to average) and an unstamped run are not judged.
    stamped = {1: 3000.0, 2: 3000.0, 3: 4000.0, 4: 100.0}
    assert contradicted_tf_stamps(scan, stamped) == [2, 3]


def test_stored_scan_restores_units_errors_and_exclusions() -> None:
    from dataclasses import replace

    from asymmetry.core.workflow.integral_scan import field_scan_from_payload

    scan = replace(_scan(), excluded=[(999, "not in period")])
    restored = field_scan_from_payload(field_scan_payload(scan))
    np.testing.assert_array_equal(restored.x, scan.x)
    np.testing.assert_array_equal(restored.value, scan.value)
    np.testing.assert_array_equal(restored.error, scan.error)
    assert restored.units == scan.units
    assert restored.run_numbers == scan.run_numbers
    assert restored.excluded == scan.excluded


def test_baseline_peak_payload_records_propagated_covariance():
    from asymmetry.core.fitting import fit_scan_baseline

    source = _scan()
    regions = [(2000.0, 2600.0), (4400.0, 5000.0)]
    fitted, payload = fit_integral_scan(
        source, "LorentzianLCR", baseline_model="Constant", baseline_regions=regions
    )
    baseline = fit_scan_baseline(source, regions, model="Constant")
    assert payload["uncertainty_basis"] == "baseline_propagated"
    assert payload["conditional_uncertainties"] is not None
    assert payload["covariance"][0] == ["f", "B0", "Bwid"]
    names, matrix = payload["covariance"]
    for name, variance in zip(names, np.diag(matrix), strict=True):
        assert payload["uncertainties"][name] == pytest.approx(np.sqrt(variance))
    np.testing.assert_allclose(fitted.error, baseline.corrected.error)
    assert payload["reduced_chi_squared"] == pytest.approx(
        payload["chi_squared"] / payload["effective_dof"]
    )
    values = payload["parameters"]
    curve = values["f"] / (1 + ((fitted.x - values["B0"]) / values["Bwid"]) ** 2)
    expected = np.sum(((fitted.value - curve) / source.error) ** 2)
    assert payload["chi_squared"] == pytest.approx(expected, rel=1e-9)


def test_joint_fit_payload_names_its_uncertainty_basis():
    _, payload = fit_integral_scan(_scan(), "LorentzianLCR + Linear")
    assert payload["uncertainty_basis"] == "joint_fit"
    assert payload["conditional_uncertainties"] is None
