"""Cooperative exit preserves successful scalar estimators and covariance."""

import numpy as np
import pytest

import asymmetry.core.fitting.field_scan as fs
from asymmetry.core.fitting.engine import FitCancelledError
from asymmetry.core.fitting.parameter_models import ParameterCompositeModel
from asymmetry.core.transform.integral import FieldScan


def scan_control(scale=1.0):
    x = np.linspace(50, 350, 61)
    return FieldScan(
        x=x,
        value=scale * (0.1 - 0.05 * np.exp(-0.5 * ((x - 200) / 20) ** 2)),
        error=np.full_like(x, scale * 0.002),
        run_numbers=list(range(x.size)),
        order_key="field",
        method="independent_scalar",
    )


@pytest.mark.parametrize("kind", ["baseline", "peaks", "rf"])
def test_pre_cancel_raises_original_exception(kind):
    scan = scan_control()
    baseline = fs.fit_scan_baseline(scan, [(50, 125), (275, 350)])
    sentinel = FitCancelledError("owned request stopped")

    def cancel():
        raise sentinel

    with pytest.raises(FitCancelledError) as caught:
        if kind == "baseline":
            fs.fit_scan_baseline(scan, [(50, 125)], cancel_check=cancel)
        elif kind == "peaks":
            fs.fit_scan_peaks(baseline, "GaussianLCR", cancel_check=cancel)
        else:
            fs.fit_rf_resonance(scan, nu_rf=218.5, cancel_check=cancel)
    assert caught.value is sentinel


@pytest.mark.parametrize("stage", ["solve", "jacobian", "baseline_covariance"])
def test_cancel_inside_real_numerical_stage(stage, monkeypatch):
    scan = scan_control()
    baseline = fs.fit_scan_baseline(scan, [(50, 125), (275, 350)])
    reached = False
    sentinel = FitCancelledError(stage)
    original = ParameterCompositeModel.function

    def evaluate(model, *args, **kwargs):
        nonlocal reached
        result = original(model, *args, **kwargs)
        reached = True
        return result

    def poll():
        if reached:
            raise sentinel

    if stage == "solve":
        monkeypatch.setattr(ParameterCompositeModel, "function", evaluate)

        def operation():
            return fs.fit_scan_baseline(scan, [(50, 125)], cancel_check=poll)
    elif stage == "jacobian":
        monkeypatch.setattr(ParameterCompositeModel, "function", evaluate)
        names, matrix = baseline.fit.covariance

        def operation():
            return fs._model_jacobian(
                baseline.model, baseline.fit.parameters, scan.x, names, np.asarray(matrix), poll
            )
    else:
        original_operator = fs.ScanBaselineResult.baseline_correction_operator

        def operator(result, cancel_check=None):
            nonlocal reached
            value = original_operator(result, cancel_check)
            reached = True
            return value

        monkeypatch.setattr(fs.ScanBaselineResult, "baseline_correction_operator", operator)

        def operation():
            return baseline.baseline_corrected_covariance(poll)

    with pytest.raises(FitCancelledError) as caught:
        operation()
    assert reached and caught.value is sentinel


def test_never_cancelled_retains_full_peak_propagation():
    scan = scan_control()
    regions = [(50, 125), (275, 350)]
    baseline = fs.fit_scan_baseline(scan, regions)
    polled = fs.fit_scan_baseline(scan, regions, cancel_check=lambda: None)
    np.testing.assert_array_equal(polled.baseline, baseline.baseline)
    initial = dict(f=-0.05, B0=200, Bwid=20)
    direct = fs.fit_scan_peaks(baseline, "GaussianLCR", initial=initial)
    worker = fs.fit_scan_peaks(polled, "GaussianLCR", initial=initial, cancel_check=lambda: None)
    from dataclasses import asdict

    left, right = asdict(direct.fit), asdict(worker.fit)
    left.pop("parameters")
    right.pop("parameters")
    assert left == right
    assert [(p.name, p.value) for p in direct.fit.parameters] == [
        (p.name, p.value) for p in worker.fit.parameters
    ]
    assert direct.covariance == worker.covariance
    assert direct.effective_dof == worker.effective_dof


def test_cancel_after_first_multistart_discards_candidate(monkeypatch):
    import asymmetry.core.fitting.parameter_models as pm

    scan = scan_control()
    original = pm._run_parameter_model_minuit
    completed = []
    sentinel = FitCancelledError("stop between starts")

    def solve(*args, **kwargs):
        value = original(*args, **kwargs)
        completed.append(value)
        return value

    def poll():
        if completed:
            raise sentinel

    monkeypatch.setattr(pm, "_run_parameter_model_minuit", solve)
    with pytest.raises(FitCancelledError) as caught:
        fs.fit_scan_model(scan, "Linear", extra_starts=3, cancel_check=poll)
    assert len(completed) == 1
    assert caught.value is sentinel


@pytest.mark.parametrize("scale", [1.0, 100.0])
def test_fraction_percent_peak_control(scale):
    scan = scan_control(scale)
    baseline = fs.fit_scan_baseline(scan, [(50, 125), (275, 350)], model="Constant")
    peaks = fs.fit_scan_peaks(
        baseline, "GaussianLCR", initial=dict(f=-0.045 * scale, B0=195, Bwid=22)
    )
    assert baseline.fit.parameters["c"].value == pytest.approx(0.1 * scale, abs=2e-5 * scale)
    assert peaks.fit.parameters["f"].value == pytest.approx(-0.05 * scale, abs=2e-5 * scale)
    assert peaks.fit.parameters["B0"].value == pytest.approx(200, abs=0.01)
    assert peaks.fit.parameters["Bwid"].value == pytest.approx(20, abs=0.02)
    assert peaks.effective_dof > 0
    assert np.linalg.eigvalsh(peaks.covariance[1]).min() >= 0


def test_rf_never_cancelled_retains_seed_solve_and_prediction():
    from asymmetry.core.fitting.muon_proton import rf_resonance_mup

    x = np.linspace(560, 1080, 53)
    y = rf_resonance_mup(x, 515, 124, 218.5, -0.02, 12, -0.02, 12, 0.001)
    scan = FieldScan(
        x=x,
        value=y,
        error=np.full_like(x, 0.001),
        run_numbers=list(range(x.size)),
        order_key="field",
        method="owned_rf_difference",
    )
    direct = fs.fit_rf_resonance(scan, nu_rf=218.5)
    polled = fs.fit_rf_resonance(scan, nu_rf=218.5, cancel_check=lambda: None)
    assert direct.success and polled.success
    assert direct.covariance == polled.covariance
    assert direct.uncertainties == polled.uncertainties
    assert direct.chi_squared == polled.chi_squared
    values = {p.name: p.value for p in direct.parameters}
    assert values == {p.name: p.value for p in polled.parameters}
    assert values["A_mu"] == pytest.approx(515, abs=0.02)
    assert values["A_p"] == pytest.approx(124, abs=0.02)
    model = fs.as_composite_model("RFResonanceMuP")
    np.testing.assert_allclose(model.function(x, **values), y, atol=2e-5, rtol=0)
