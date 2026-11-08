"""TF calibration retains measured noise after applied deadtime correction."""

from dataclasses import replace

import numpy as np
import pytest

from asymmetry.core.fitting.count_results import CountFitParameter as P
from asymmetry.core.fitting.engine import FitCancelledError
from asymmetry.core.fitting.likelihood import FitInputError, Objective
from asymmetry.core.fitting.tf_calibration import TFCalibrationRequest, fit_tf_calibration
from asymmetry.core.utils.constants import MUON_LIFETIME_US
from tests.core.test_tf_calibration_fit import _raw_tf


def _deadtime_tf(*, fit_beta=False, mode="manual", factor=1, supplied=False):
    """Independent rational mean loss on native counts, with known TF balance."""
    ds = _raw_tf(beta=0.8 if fit_beta else 1, backgrounds=(0, 0))
    frames, coefficient = 100000.0, 0.02
    for h in ds.run.histograms:
        h.counts = np.rint(h.counts / (1 + coefficient * h.counts / (h.bin_width * frames)))
        if supplied:
            h.count_variance = 4 * h.counts
    ds.run.grouping.update(
        deadtime_correction=True,
        deadtime_mode=mode,
        dead_time_us=[coefficient, coefficient],
        deadtime_file_us=[coefficient, coefficient],
        good_frames=frames,
        bunching_factor=factor,
    )
    return ds


@pytest.mark.parametrize("mode", ["manual", "from_file"])
@pytest.mark.parametrize("fit_beta", [False, True])
@pytest.mark.parametrize("factor", [1, 4])
@pytest.mark.parametrize("supplied", [False, True])
def test_corrected_tf_uses_weighted_counts_and_preserves_native_provenance(
    mode, fit_beta, factor, supplied
):
    ds = _deadtime_tf(fit_beta=fit_beta, mode=mode, factor=factor, supplied=supplied)
    snapshots = [h.counts.copy() for h in ds.run.histograms]
    request = TFCalibrationRequest(
        1,
        2,
        fit_beta=fit_beta,
        frequency_mhz=1.3,
        exclude=(2, 3),
        parameters=(P("background", 0, fixed=True), P("background_b", 0, fixed=True)),
    )
    result = fit_tf_calibration(ds, request)
    assert result.ok, result.message
    assert result.fit.objective is Objective.GAUSSIAN
    assert result.alpha == pytest.approx(1.25, abs=2e-4)
    assert result.beta == pytest.approx(0.8 if fit_beta else 1, abs=2e-4)
    assert result.alpha_error > 0
    assert (result.beta_error is not None) == fit_beta
    assert (result.alpha_beta_correlation is not None) == fit_beta
    for domain, original, h in zip(result.fit.domains, snapshots, ds.run.histograms, strict=True):
        np.testing.assert_array_equal(h.counts, original)
        assert domain.deadtime.status == "applied"
        assert domain.deadtime.requested == mode
        assert domain.deadtime.uncertainty == (
            "measured-count variance retained; coefficient and exposure uncertainty excluded"
        )
        assert all(w.last - w.first + 1 == factor for w in domain.raw_windows)
        assert all(not 2 <= t <= 3 for t in domain.time_us)
        assert domain.detector_indices in ((0,), (1,))
    assert result.fit.dof == 2 * result.n_bins_used - len(result.fit.covariance_parameters)


@pytest.mark.parametrize("supplied", [False, True])
@pytest.mark.parametrize("factor", [1, 4])
def test_corrected_fit_matches_independent_weighted_normalisation_and_curvature(supplied, factor):
    ds = _deadtime_tf(factor=factor, supplied=supplied)
    fixed = dict(alpha=1.25, A0=20, frequency=1.3, Lambda=0.15, phase=0.17)
    fixed.update(background=0, background_b=0)
    request = TFCalibrationRequest(
        1, 2, parameters=tuple(P(n, v, fixed=True) for n, v in fixed.items())
    )
    result = fit_tf_calibration(ds, request)
    assert result.ok, result.message
    times = np.asarray(result.fit.domains[0].time_us)
    wave = 0.2 * np.exp(-0.15 * times) * np.cos(2 * np.pi * 1.3 * times + 0.17)
    shapes = [
        np.sqrt(1.25) * np.exp(-times / MUON_LIFETIME_US) * (1 + wave),
        np.exp(-times / MUON_LIFETIME_US) * (1 - wave) / np.sqrt(1.25),
    ]
    numerator, information = 0.0, 0.0
    for h, domain, shape in zip(ds.run.histograms, result.fit.domains, shapes, strict=True):
        measured = h.counts
        corrected = measured / (1 - 0.02 * measured / (h.bin_width * 100000))
        variance = measured if h.count_variance is None else h.count_variance
        counts = np.array([corrected[w.first : w.last + 1].sum() for w in domain.raw_windows])
        noise = np.array([variance[w.first : w.last + 1].sum() for w in domain.raw_windows])
        numerator += np.sum(shape * counts / noise)
        information += np.sum(shape**2 / noise)
    n0 = next(p.value for p in result.fit.parameters if p.name == "N0")
    assert n0 == pytest.approx(numerator / information, rel=1e-6)
    assert result.fit.standard_error("N0") == pytest.approx(1 / np.sqrt(information), rel=1e-4)


def test_missing_optional_file_table_retains_raw_poisson_and_skip_provenance():
    ds = _raw_tf(beta=1)
    ds.run.grouping.update(deadtime_correction=True, deadtime_mode="from_file")
    with pytest.warns(UserWarning, match="File deadtime correction not applied"):
        result = fit_tf_calibration(ds, TFCalibrationRequest(1, 2, frequency_mhz=1.3))
    assert result.ok
    assert result.fit.objective is Objective.POISSON
    assert all(d.deadtime.status == "skipped" for d in result.fit.domains)


@pytest.mark.parametrize("mode", ["manual", "from_file"])
def test_applied_deadtime_does_not_admit_declared_processed_banks(mode):
    ds = _deadtime_tf(mode=mode)
    ds.run.metadata["histogram_labels"] = ["Forward PostPileUp rej.", "Backward"]
    with pytest.raises(FitInputError, match="PostPileUp"):
        fit_tf_calibration(ds, TFCalibrationRequest(1, 2))


def test_skipped_file_correction_does_not_admit_arbitrary_supplied_variance():
    ds = _deadtime_tf(mode="from_file", supplied=True)
    ds.run.grouping.update(deadtime_file_us=[], dead_time_us=[])
    with pytest.warns(UserWarning, match="File deadtime correction not applied"):
        with pytest.raises(FitInputError, match="supplied counting variance"):
            fit_tf_calibration(ds, TFCalibrationRequest(1, 2))


def test_corrected_fit_keeps_automatic_backgrounds_without_subtracting_grouping_levels():
    ds = _deadtime_tf()
    ds.run.grouping.update(
        background_correction=True, background_mode="fixed", background_fixed_values=[7, 11]
    )
    fixed = fit_tf_calibration(ds, TFCalibrationRequest(1, 2, frequency_mhz=1.3))
    ds.run.grouping["background_correction"] = False
    off = fit_tf_calibration(ds, TFCalibrationRequest(1, 2, frequency_mhz=1.3))
    assert fixed.alpha == off.alpha
    assert fixed.fit.covariance == off.fit.covariance
    assert not next(p for p in fixed.fit.parameters if p.name == "background").fixed


def test_corrected_selected_period_uses_its_acquired_exposure():
    red, green = _deadtime_tf(), _deadtime_tf()
    red.run.metadata.update(run_number=7101, source_run_number=7101, period_count=2)
    empty = (np.array([]), np.array([]), np.array([]))
    red.run.grouping.update(
        period_histograms=[red.run.histograms, green.run.histograms],
        period_reduced=[empty, empty],
        period_good_frames=[50000, 100000],
    )
    selected = fit_tf_calibration(red, TFCalibrationRequest(1, 2, period=2, frequency_mhz=1.3))
    direct = fit_tf_calibration(green, TFCalibrationRequest(1, 2, frequency_mhz=1.3))
    assert selected.ok and direct.ok
    assert selected.alpha == direct.alpha
    assert selected.fit.covariance == direct.fit.covariance
    assert selected.fit.domains[0].deadtime.effective["good_frames"] == 100000
    assert selected.period_number == 2


def test_applied_variance_scale_changes_errors_without_changing_the_minimum():
    ds = _deadtime_tf(fit_beta=True)
    request = TFCalibrationRequest(
        1,
        2,
        fit_beta=True,
        frequency_mhz=1.3,
        parameters=(
            P("background", 0, fixed=True),
            P("background_b", 0, fixed=True),
            P("phase", 0.17, fixed=True),
        ),
    )
    first = fit_tf_calibration(ds, request)
    ds.run.histograms = [replace(h, count_variance=4 * h.counts) for h in ds.run.histograms]
    second = fit_tf_calibration(ds, request)
    assert first.ok and second.ok
    assert second.alpha == pytest.approx(first.alpha, abs=1e-7)
    assert second.beta == pytest.approx(first.beta, abs=1e-7)
    assert second.alpha_error == pytest.approx(2 * first.alpha_error, rel=1e-3)
    assert second.beta_error == pytest.approx(2 * first.beta_error, rel=1e-3)


def test_corrected_fit_still_requires_acquired_counts_on_consumed_native_support():
    ds = _deadtime_tf()
    ds.run.histograms[0].counts[10] += 0.5
    with pytest.raises(FitInputError, match="non-negative integer counts"):
        fit_tf_calibration(ds, TFCalibrationRequest(1, 2))
    result = fit_tf_calibration(ds, TFCalibrationRequest(1, 2, exclude=(0.1, 0.2)))
    assert result.ok


def test_saturation_checks_only_observations_consumed_by_the_calibration():
    ds = _deadtime_tf()
    ds.run.histograms[0].counts[10] = 1000000
    with pytest.raises(ValueError, match="saturat|occupancy"):
        fit_tf_calibration(ds, TFCalibrationRequest(1, 2))
    result = fit_tf_calibration(ds, TFCalibrationRequest(1, 2, exclude=(0.1, 0.2)))
    assert result.ok


@pytest.mark.parametrize("boundary", ["start", "objective", "final"])
def test_corrected_fit_cancels_without_publishing_a_result(monkeypatch, boundary):
    from asymmetry.core.fitting import tf_calibration

    calls = 0
    cancelled = boundary == "start"

    def cancel():
        nonlocal calls
        calls += 1
        return cancelled or (boundary == "objective" and calls >= 5)

    if boundary == "final":
        solve = tf_calibration._fit_fb_count_groups

        def cancel_after_fit(*args, **kwargs):
            nonlocal cancelled
            result = solve(*args, **kwargs)
            cancelled = True
            return result

        monkeypatch.setattr(tf_calibration, "_fit_fb_count_groups", cancel_after_fit)
    with pytest.raises(FitCancelledError):
        fit_tf_calibration(_deadtime_tf(), TFCalibrationRequest(1, 2), cancel_callback=cancel)


def test_historical_beta_adapter_inherits_corrected_observations_and_joint_errors():
    from asymmetry.core.fitting.beta_calibration import estimate_beta_detailed

    ds = _deadtime_tf(fit_beta=True)
    direct = fit_tf_calibration(ds, TFCalibrationRequest(1, 2, fit_beta=True))
    adapter = estimate_beta_detailed(ds, 1, 2)
    assert direct.ok and adapter.ok
    assert adapter.method == "count_fit"
    assert adapter.alpha == direct.alpha
    assert adapter.beta == direct.beta
    assert adapter.alpha_error == direct.alpha_error
    assert adapter.beta_error == direct.beta_error
    assert adapter.alpha_beta_correlation == direct.alpha_beta_correlation


@pytest.mark.parametrize("mode", ["manual", "from_file"])
def test_applied_zero_deadtime_keeps_recorded_weighted_objective(mode):
    ds = _raw_tf(beta=1)
    ds.run.grouping.update(
        deadtime_correction=True,
        deadtime_mode=mode,
        dead_time_us=[0, 0],
        deadtime_file_us=[0, 0],
        good_frames=100000,
    )
    result = fit_tf_calibration(ds, TFCalibrationRequest(1, 2, frequency_mhz=1.3))
    assert result.ok
    assert result.fit.objective is Objective.GAUSSIAN
    assert result.fit.domains[0].deadtime.status == "applied"
    assert result.alpha == pytest.approx(1.25, abs=1e-4)
