import pickle
from dataclasses import replace

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting import spectral_residual
from asymmetry.core.fitting.fit_wizard import fingerprint_spectrum
from asymmetry.core.fitting.likelihood import Objective
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.fitting.peak_detection import analyze_dataset_peaks
from asymmetry.core.fitting.results import FitResult
from asymmetry.core.fourier.diagnostic import (
    DiagnosticFitRecord,
    diagnostic_fit_record,
    prepare_diagnostic_observation,
)
from asymmetry.core.fourier.fft import fft_complex_arrays
from asymmetry.core.transform.preparation import prepare_wizard_dataset
from tests.core.test_binning_entry_points import _run
from tests.core.test_wizard_preparation import observed


@pytest.mark.parametrize("mode", ["variable", "constant_error"])
@pytest.mark.parametrize("consumer", [fingerprint_spectrum, analyze_dataset_peaks])
def test_nonuniform_production_spectrum_matches_native_fixed_control(mode, consumer):
    time = np.arange(1200) * 0.1
    line = 0.18 * np.cos(2 * np.pi * 1.3 * time + 0.23)
    forward = 17000 * np.exp(-time / 40) * (1 + line)
    backward = 11000 * np.exp(-time / 40) * (1 - line)
    run = _run(
        forward, backward, alpha=17000 / 11000, binning_mode=mode, bin0_us=0.14, bin10_us=0.3
    )
    # Keep variable widths bounded while retaining a long enough record.
    run.histograms = [replace(h, good_bin_end=199) for h in run.histograms]
    run.grouping["last_good_bin"] = 199
    dataset = observed(run)
    control = observed(replace(run, grouping=run.grouping | {"binning_mode": "fixed"}))
    before = [a.copy() for a in (dataset.time, dataset.asymmetry, dataset.error)]
    result, fixed = consumer(dataset), consumer(control)
    if consumer is fingerprint_spectrum:
        actual, expected = result.dominant_fft_frequency_mhz, fixed.dominant_fft_frequency_mhz
    else:
        actual = min(result.peaks, key=lambda p: abs(p.frequency_mhz - 1.3)).frequency_mhz
        expected = min(fixed.peaks, key=lambda p: abs(p.frequency_mhz - 1.3)).frequency_mhz
    assert abs(actual - expected) <= 1 / (4 * control.n_points * 0.1)
    for value, original in zip((dataset.time, dataset.asymmetry, dataset.error), before):
        np.testing.assert_array_equal(value, original)


def _selected_counts():
    f = 1000 + np.arange(101.0) ** 2
    b = 700 + 3 * np.arange(111.0)
    run = _run(f, b, binning_mode="variable", bin0_us=0.14, bin10_us=0.56)
    run.histograms = [replace(h, t0_time_us=0.073) for h in run.histograms]
    base = observed(run, merge=2)
    crop = base.time_range(base.time[2], base.time[-2])
    dataset = prepare_wizard_dataset(crop, 3)
    # Closed width thresholds for .14*4**(left/10) cross half-bin values:
    # width1→2 at .4977us, 2→3 at4.1825us,3→4 at6.6096us,4→5 at8.423us.
    # Whole-bin progression reaches raw6,44,68,88 at those changes.
    edges = np.r_[
        np.arange(0, 7),
        np.arange(8, 45, 2),
        np.arange(47, 69, 3),
        np.arange(72, 89, 4),
        93,
        98,
        101,
    ]
    selected = edges[::2][2:-1][::3]
    assert np.ptp(np.diff(dataset.time)) > 0.001
    return dataset, f, b, selected


def test_independent_cropped_merged_native_membership_unequal_banks():
    dataset, f, b, edges = _selected_counts()
    view = prepare_diagnostic_observation(dataset)
    members = np.arange(edges[0], edges[-1])
    expected = 100 * (f[members] - b[members]) / (f[members] + b[members])
    sigma = 200 * np.sqrt(f[members] * b[members] / (f[members] + b[members]) ** 3)
    np.testing.assert_allclose(view.time, (members + 0.5) * 0.1 - 0.073, atol=2e-14)
    np.testing.assert_allclose(view.asymmetry, expected, rtol=1e-13)
    np.testing.assert_allclose(view.error, sigma, rtol=1e-13)
    assert view.time.size == edges[-1] - edges[0]
    assert not view.time.flags.writeable


def test_transient_worker_record_drops_raw_and_crop_does_not_expand():
    dataset, _, _, _ = _selected_counts()
    record = diagnostic_fit_record(dataset)
    assert isinstance(record, DiagnosticFitRecord)
    assert not record.run.histograms and record.histogram_observation is None
    assert diagnostic_fit_record(record) is record
    restored = pickle.loads(pickle.dumps(record))
    np.testing.assert_array_equal(
        restored.diagnostic_observation.time, record.diagnostic_observation.time
    )
    cropped = record.time_range(record.time[1], record.time[-2])
    assert not isinstance(cropped, DiagnosticFitRecord)
    with pytest.raises(ValueError, match="uniform"):
        prepare_diagnostic_observation(cropped)


def test_residual_new_grid_model_evaluation_and_original_statistics(monkeypatch):
    dataset, _, _, _ = _selected_counts()

    def model(t, amplitude, offset, rate):
        return offset + amplitude * np.exp(-rate * t)

    parameters = ParameterSet(
        [Parameter("amplitude", 7.0), Parameter("offset", 3.0), Parameter("rate", 0.2)]
    )
    residual = dataset.asymmetry - model(dataset.time, 7.0, 3.0, 0.2)
    result = FitResult(
        True,
        Objective.GAUSSIAN,
        parameters=parameters,
        residuals=residual.copy(),
        residual_time=dataset.time.copy(),
        residual_sigma=dataset.error.copy(),
        chi_squared=123.0,
        dof=dataset.n_points - 3,
        covariance=np.eye(3),
    )
    saved = (
        result.residuals.copy(),
        result.residual_time.copy(),
        result.residual_sigma.copy(),
        result.covariance.copy(),
    )
    seen = []

    def transform(t, y, error, **kwargs):
        seen.append((t.copy(), y.copy(), error.copy()))
        return np.array([0.0, 1.0, 2.0, 3.0]), np.zeros(4), np.array([0.0, 1.0, 3.0, 1.0])

    monkeypatch.setattr(spectral_residual, "fft_arrays", transform)
    rms, runs, ac, snr = spectral_residual._residual_diagnostics(
        diagnostic_fit_record(dataset), result, model
    )
    expected = prepare_diagnostic_observation(dataset)
    np.testing.assert_array_equal(seen[0][0], expected.time)
    np.testing.assert_allclose(
        seen[0][1], expected.asymmetry - (3 + 7 * np.exp(-0.2 * expected.time)), rtol=1e-14
    )
    np.testing.assert_array_equal(seen[0][2], expected.error)
    assert rms == pytest.approx(np.sqrt(np.mean((residual / dataset.error) ** 2)))
    assert np.isnan(ac)
    assert result.chi_squared == 123 and result.dof == dataset.n_points - 3
    for actual, old in zip(
        (result.residuals, result.residual_time, result.residual_sigma, result.covariance), saved
    ):
        np.testing.assert_array_equal(actual, old)
    assert result.spectral_residual_attribution["observation"] == "separate_fixed_count_reduction"


def test_uniform_curve_direct_dft_anchor_and_preserved_arrays():
    n = 128
    dt = 0.0125
    t = 0.037 + np.arange(n) * dt
    y = 12 * np.cos(2 * np.pi * (11 / (n * dt)) * t + 0.24)
    dataset = MuonDataset(t, y, np.ones(n))
    view = prepare_diagnostic_observation(dataset)
    for actual, old in zip((view.time, view.asymmetry, view.error), (t, y, np.ones(n))):
        np.testing.assert_array_equal(actual, old)
    freq, z = fft_complex_arrays(
        view.time,
        view.asymmetry,
        view.error,
        window="none",
        padding_factor=1,
        subtract_average_signal=False,
    )
    oracle = np.exp(-2j * np.pi * np.outer(freq, np.arange(n) * dt)) @ y
    np.testing.assert_allclose(z, oracle, atol=3e-10, rtol=1e-10)
    assert freq[1 + np.argmax(abs(z[1:]))] == pytest.approx(11 / (n * dt))


def test_sampled_only_detrend_is_not_reinterpreted():
    dataset, _, _, _ = _selected_counts()
    with pytest.raises(ValueError, match="detrend"):
        analyze_dataset_peaks(dataset, detrend_curve=np.ones(dataset.n_points), damped_pass=False)


@pytest.mark.parametrize("consumer", [fingerprint_spectrum, analyze_dataset_peaks])
def test_attached_run_does_not_authorize_custom_nonuniform_curve(consumer):
    dataset, _, _, _ = _selected_counts()
    custom = replace(dataset, histogram_observation=None)
    with pytest.raises(ValueError, match="uniform"):
        consumer(custom)


def test_fixed_diagnostic_keeps_corrections_fractional_timing_and_retained_variance():
    forward = np.arange(100.0, 201)
    backward = np.arange(60.0, 171)
    run = _run(
        forward,
        backward,
        binning_mode="variable",
        bin0_us=0.14,
        bin10_us=0.56,
        first_good_bin=3,
        last_good_bin=98,
        alpha=1.2,
        beta=0.8,
        deadtime_correction=True,
        deadtime_mode="manual",
        dead_time_us=[0.01, 0.02],
        good_frames=1000,
        background_correction=True,
        background_mode="fixed",
        background_fixed_values=[7, 3],
    )
    run.histograms = [replace(h, t0_time_us=0.073) for h in run.histograms]
    dataset = observed(run)
    view = prepare_diagnostic_observation(dataset)
    f = forward[3:99] / (1 - forward[3:99] * 0.01 / 100) - 7
    b = backward[3:99] / (1 - backward[3:99] * 0.02 / 100) - 3
    denominator = 0.8 * f + 1.2 * b
    expected = 100 * (f - 1.2 * b) / denominator
    df = 100 * 1.2 * 1.8 * b / denominator**2
    db = -100 * 1.2 * 1.8 * f / denominator**2
    np.testing.assert_allclose(view.asymmetry, expected)
    np.testing.assert_allclose(view.error, np.sqrt(df**2 * forward[3:99] + db**2 * backward[3:99]))
    np.testing.assert_allclose(view.time, (np.arange(3, 99) + 0.5) * 0.1 - 0.073, atol=2e-14)
    recipe = view.attribution.to_dict()["recipe"]["periods"]["red"]
    assert recipe["alpha"] == 1.2 and recipe["beta"] == 0.8
    assert recipe["binning"]["mode"] == "fixed" and recipe["binning"]["factor"] == 1
    assert recipe["binning"]["source_binning"]["mode"] == "variable"
    np.testing.assert_array_equal(run.histograms[0].counts, forward)


@pytest.mark.parametrize("mode", ["variable", "constant_error"])
def test_workflow_precession_uses_the_native_diagnostic_band(mode):
    from asymmetry.core.fitting.spectral import frequency_mhz_to_field_gauss
    from asymmetry.core.workflow.survey import precession_evidence

    time = np.arange(200) * 0.1
    signal = 0.2 * np.cos(2 * np.pi * 1.3 * time)
    dataset = observed(
        _run(
            10000 * (1 + signal), 10000 * (1 - signal), binning_mode=mode, bin0_us=0.3, bin10_us=0.8
        )
    )
    evidence = precession_evidence(dataset, frequency_mhz_to_field_gauss(1.3))
    assert evidence.state == "larmor"
    assert evidence.frequency_mhz == pytest.approx(1.3, abs=1 / (4 * 200 * 0.1))


def test_native_diagnostic_does_not_fill_invalid_ratios_or_ignore_cancel():
    from asymmetry.core.fitting.engine import FitCancelledError

    f, b = np.full(200, 100.0), np.full(200, 80.0)
    f[10] = b[10] = 0
    dataset = observed(_run(f, b, binning_mode="variable", bin0_us=0.3, bin10_us=0.8))
    with pytest.raises(ValueError, match="undefined ratios"):
        analyze_dataset_peaks(dataset)

    def cancel():
        raise FitCancelledError("cancelled before replay")

    with pytest.raises(FitCancelledError, match="cancelled before replay"):
        prepare_diagnostic_observation(dataset, cancel_check=cancel)


@pytest.mark.parametrize("mode", ["variable", "constant_error"])
def test_actual_global_completion_and_coupled_fit_keep_science_observations(mode):
    import asymmetry.core.fitting.fit_wizard as single
    import asymmetry.core.fitting.global_fit_wizard as global_wizard
    from asymmetry.core.fitting.component_tags import PhysicsClass
    from asymmetry.core.fitting.global_search.trend_objective import SelectionObjective
    from asymmetry.core.fitting.wizard_scope import WizardScope

    datasets, recommendations = [], {}
    for number, rate in ((1, 0.15), (2, 0.25)):
        t = np.arange(200) * 0.1
        y = 0.16 * np.exp(-rate * t) + 0.01
        run = _run(
            100000 * (1 + y), 100000 * (1 - y), binning_mode=mode, bin0_us=0.14, bin10_us=0.56
        )
        run.run_number = number
        dataset = observed(run)
        datasets.append(dataset)
        template = single.build_candidate_templates(single.fingerprint_spectrum(dataset))[0]
        recommendations[number] = single.build_fit_wizard_recommendation_for_templates(
            dataset, (template,), variant_budget=1
        )
    scope = WizardScope(physics=frozenset({PhysicsClass.GENERIC_RELAXATION}))
    portfolio = global_wizard.build_global_fit_wizard_candidate_portfolio(
        datasets, scope=scope, single_fit_recommendations_by_run=recommendations
    )
    portfolio = replace(portfolio, templates=(template,))
    screening = global_wizard.build_global_fit_wizard_screening_recommendation(
        datasets,
        portfolio=portfolio,
        single_fit_recommendations_by_run=recommendations,
        scope=scope,
    )
    assert screening.assessments
    result = global_wizard.build_global_fit_wizard_recommendation(
        datasets,
        portfolio=portfolio,
        single_fit_recommendations_by_run=recommendations,
        selected_template_keys=(template.key,),
        scope=scope,
        objective=SelectionObjective.STATISTICAL,
    )
    assert result.assessments
    assert any(row.is_successful for row in result.assessments)
    for row in result.assessments:
        for number, fit in row.fit_results_by_run.items():
            original = datasets[number - 1]
            assert len(fit.residuals) == original.n_points
            np.testing.assert_array_equal(fit.residual_time, original.time)
            np.testing.assert_array_equal(fit.residual_sigma, original.error)
            if fit.spectral_residual_attribution is not None:
                assert (
                    fit.spectral_residual_attribution["observation"]
                    == "separate_fixed_count_reduction"
                )


def test_series_fingerprints_reuse_the_full_observation_identity(monkeypatch):
    from asymmetry.core.fitting import fingerprint_cache, fit_wizard, global_fit_wizard

    time = np.arange(128) * 0.1
    datasets = [
        MuonDataset(time, np.cos(time) + number, np.ones(128), {"run_number": number})
        for number in range(4)
    ]
    calls = []
    real = fit_wizard.fingerprint_spectrum

    def counting(dataset):
        calls.append(dataset)
        return real(dataset)

    monkeypatch.setattr(fit_wizard, "fingerprint_spectrum", counting)
    fingerprint_cache._SERIES_FINGERPRINT_CACHE.clear()
    first = global_fit_wizard._fingerprint_jump_warnings(datasets)
    assert global_fit_wizard._fingerprint_jump_warnings(datasets) == first
    assert len(calls) == len(datasets)
    # The old moments key missed errors, despite their effect on the SNR window.
    datasets[0].error = np.linspace(1, 10, 128)
    global_fit_wizard._fingerprint_jump_warnings(datasets)
    assert len(calls) == len(datasets) + 1


@pytest.mark.parametrize("mode", ["variable", "constant_error"])
def test_production_oscillatory_seed_keeps_the_measured_native_band(monkeypatch, mode):
    import asymmetry.core.fitting.fit_wizard as single
    from asymmetry.core.fitting.engine import FitEngine

    t = np.arange(200) * 0.1
    line = 0.2 * np.cos(2 * np.pi * 1.3 * t + 0.23)
    dataset = observed(
        _run(
            np.rint(100000 * (1 + line)),
            np.rint(80000 * (1 - line)),
            alpha=1.25,
            binning_mode=mode,
            bin0_us=0.14,
            bin10_us=0.56,
        )
    )
    view = prepare_diagnostic_observation(dataset)
    analysis = analyze_dataset_peaks(dataset, diagnostic=view, damped_pass=False)
    fingerprint = single.fingerprint_spectrum(dataset, diagnostic=view)
    template = next(
        t for t in single.build_wizard_families(fingerprint) if t.key == "oscillatory"
    ).stage1_rep
    calls = []
    original = FitEngine.fit

    def checking(self, actual_dataset, model, parameters, **options):
        frequency = parameters["frequency"]
        calls.append((frequency.value, frequency.min, frequency.max))
        assert frequency.min < 1.3 < frequency.max, (
            "nonuniform science centres fabricated a Nyquist cap below the native measured line"
        )
        return original(self, actual_dataset, model, parameters, **options)

    monkeypatch.setattr(FitEngine, "fit", checking)
    result = single.build_fit_wizard_recommendation_for_templates(
        dataset,
        (template,),
        fingerprint=fingerprint,
        seed_context=single.TemplateSeedContext(peak_analysis=analysis),
        variant_budget=1,
    )
    assert calls and result.assessments


def test_defined_zero_variance_native_ratio_retains_fixed_fft_support():
    t = np.arange(200) * 0.1
    line = 0.18 * np.cos(2 * np.pi * 1.3 * t + 0.21)
    f = np.rint(10000 * (1 + line))
    b = np.rint(10000 * (1 - line))
    b[10] = 0
    run = _run(f, b, binning_mode="variable", bin0_us=0.3, bin10_us=0.8)
    variable = observed(run)
    fixed = observed(replace(run, grouping=run.grouping | {"binning_mode": "fixed"}))
    assert fixed.asymmetry[10] == 100 and fixed.error[10] == 100
    control = analyze_dataset_peaks(fixed, damped_pass=False, burg_check="never")
    actual = analyze_dataset_peaks(variable, damped_pass=False, burg_check="never")
    assert actual.nyquist_mhz == pytest.approx(control.nyquist_mhz)
    view = prepare_diagnostic_observation(variable)
    np.testing.assert_array_equal(view.asymmetry, fixed.asymmetry)
    np.testing.assert_array_equal(view.error, fixed.error)


@pytest.mark.parametrize("exact_t0,crop_first,native_first", [(0.073, 2, 4), (0.373, 1, 2)])
def test_production_damped_seed_refers_to_actual_diagnostic_origin(
    monkeypatch, exact_t0, crop_first, native_first
):
    import asymmetry.core.fitting.fit_wizard as single
    from asymmetry.core.fitting.engine import FitEngine
    from asymmetry.core.fitting.peak_detection import DetectedPeak, PeakAnalysis

    raw_time = (np.arange(101) + 0.5) * 0.1 - exact_t0
    a = (3 + 1 * np.exp(-0.7 * raw_time) * np.cos(2 * np.pi * 1.3 * raw_time + 0.21)) / 100
    run = _run(
        np.rint(100000 * (1 + a)),
        np.rint(100000 * (1 - a)),
        binning_mode="variable",
        bin0_us=0.14,
        bin10_us=0.56,
    )
    run.histograms = [
        replace(h, t0_bin=int(exact_t0 / 0.1), t0_time_us=exact_t0) for h in run.histograms
    ]
    base = observed(run, merge=2)
    cropped = base.time_range(base.time[crop_first], base.time[-2])
    dataset = prepare_wizard_dataset(cropped, 2)
    view = prepare_diagnostic_observation(dataset)
    origin = (native_first + 0.5) * 0.1 - exact_t0
    assert view.time[0] == pytest.approx(origin)
    amplitude = 1 * np.exp(-0.7 * origin)
    phase = 0.21 + 2 * np.pi * 1.3 * origin
    peak = DetectedPeak(
        1.3,
        amplitude,
        10.0,
        0.7 / np.pi,
        0.0,
        "damped_scan",
        damping_rate_per_us=0.7,
        amplitude_percent=amplitude,
        phase_rad=phase,
        delta_chi_squared=50.0,
    )
    analysis = PeakAnalysis(
        (peak,),
        1.0,
        0.1,
        5.0,
        False,
        diagnostic_attribution=view.window_attribution(len(view.time), "hann"),
    )
    fingerprint = single.fingerprint_spectrum(dataset, diagnostic=view)
    template = next(
        f for f in single.build_wizard_families(fingerprint) if f.key == "oscillatory"
    ).stage1_rep
    calls = []
    original = FitEngine.fit

    def checking(self, actual_dataset, model, parameters, **options):
        calls.append(1)
        if len(calls) == 1:
            assert parameters["A_1"].value == pytest.approx(1.0, rel=1e-12), (
                "damped amplitude used science-centre origin rather than measured diagnostic origin"
            )
            assert parameters["phase"].value == pytest.approx(0.21, abs=1e-12), (
                "damped phase used science-centre origin rather than measured diagnostic origin"
            )
        return original(self, actual_dataset, model, parameters, **options)

    monkeypatch.setattr(FitEngine, "fit", checking)
    single.build_fit_wizard_recommendation_for_templates(
        dataset,
        (template,),
        fingerprint=fingerprint,
        seed_context=single.TemplateSeedContext(peak_analysis=analysis),
        variant_budget=1,
    )
    assert calls
