"""Acquired-source selection, calibration and compact provenance at consumer boundaries."""

import json
from dataclasses import replace

import numpy as np
import pytest

from asymmetry.core.data.combine import coadd_compatibility_error, combine_runs, reduce_combined_run
from asymmetry.core.data.dataset import Histogram
from asymmetry.core.fitting.count_domain import build_count_groups
from asymmetry.core.fitting.count_inputs import count_model_frame_norm
from asymmetry.core.fitting.count_results import count_fit_domain
from asymmetry.core.fitting.likelihood import FitInputError
from asymmetry.core.fitting.tf_calibration import TFCalibrationRequest, _prepare_tf_counts
from asymmetry.core.fourier.fb_input import build_fb_signal_dataset
from asymmetry.core.fourier.group_preparation import prepare_group_counts
from asymmetry.core.fourier.grouped import build_group_signal_dataset
from asymmetry.core.project.calibration import _domain_payload, _restore_domain
from asymmetry.core.project.profiles import (
    AlphaPolicy,
    GroupingProfile,
    ProfileFingerprint,
    resolve_effective_grouping,
)
from asymmetry.core.transform.corrections import BackgroundReference
from asymmetry.core.transform.integral import (
    build_field_scan,
    integrate_run,
    reduce_integral_run,
    reduce_period_integral,
)
from asymmetry.core.transform.preparation import prepare_wizard_dataset
from asymmetry.core.transform.reference_scale import ReferenceScaleRequest
from tests.core.test_acquired_run_preparation import acquired, prepare


def automatic_runs():
    """Different source balances require one ratio of the pooled corrected sums."""
    runs = [acquired(11, (60, 40)), acquired(22, (90, 10), anchor=2)]
    profile = GroupingProfile(
        "shared ratio",
        ProfileFingerprint("", 2),
        groups={1: [1], 2: [2]},
        forward_group=1,
        backward_group=2,
        alpha_policy=AlphaPolicy(mode="per_run_estimate"),
    )
    for run in runs:
        run.grouping = resolve_effective_grouping(profile, run)
    return runs


@pytest.mark.parametrize("explicit_tables", [False, True])
@pytest.mark.parametrize("weight", [1.0, 0.25])
def test_single_active_source_uses_only_its_observed_good_bins(explicit_tables, weight):
    runs = [acquired(1, (120, 80), anchor=2), acquired(2, (10, 90))]
    runs[0].histograms[0].counts[:2] = [-3, 0.5]
    if explicit_tables:
        runs[0].grouping.update(
            detector_t0_bins=[2, 2],
            detector_first_good_bins=[2, 2],
            detector_last_good_bins=[10, 10],
        )
    result = prepare(combine_runs(runs, scales=[weight, 0]))
    np.testing.assert_allclose(result.time, np.arange(9) / 10, atol=1e-14)
    np.testing.assert_allclose(result.asymmetry, 20)
    np.testing.assert_allclose(result.error, 100 * np.sqrt(4 * 120 * 80 / 200**3))


def test_authored_background_pair_and_small_supplied_variance_survive_fourier():
    runs = [acquired(1, (120, 80)), acquired(2, (40, 20))]
    for run in runs:
        run.grouping.update(groups={11: [1], 17: [2]}, forward_group=11, backward_group=17)
        run.histograms = [replace(h, count_variance=np.full(16, 0.04)) for h in run.histograms]
    combined = combine_runs(runs, scales=[0.5, 0.25])
    combined.grouping.update(
        background_correction=True, background_mode="fixed", background_fixed_values=[7, 3]
    )
    result = build_group_signal_dataset(
        combined, 17, center_signal=False, apply_lifetime_correction=False
    )
    np.testing.assert_allclose(result.asymmetry, 42)
    np.testing.assert_allclose(result.error, np.sqrt(0.04 * (0.5**2 + 0.25**2)))


def test_preflight_reports_unequal_detector_tables_without_profile_exception():
    runs = [acquired(1, (120, 80)), acquired(2, (40, 20))]
    runs[1].histograms.append(Histogram(np.ones(16), 0.1))
    runs[1].grouping["effective_detector_t0_bins"] = [0, 0, 0]
    assert coadd_compatibility_error(runs)


@pytest.mark.parametrize("reference_anchor", [0, 4])
@pytest.mark.parametrize("consumer", ["fourier", "asymmetry"])
def test_reference_requires_observed_good_support_without_unused_context(
    monkeypatch, reference_anchor, consumer
):
    combined = combine_runs([acquired(1, (120, 80), anchor=2), acquired(2, (40, 20), anchor=3)])
    combined.grouping.update(
        background_correction=True,
        background_mode="reference_run",
        background_run={"run_number": 3},
    )
    source = acquired(3, (10, 5), anchor=reference_anchor)
    histograms = [replace(h, counts=h.counts[: reference_anchor + 9]) for h in source.histograms]
    reference = BackgroundReference(
        histograms,
        ReferenceScaleRequest("manual", 1).resolve(),
        source_file="synthetic-reference.nxs",
        run_number=3,
    )
    monkeypatch.setattr("asymmetry.core.io.resolve_background_reference", lambda *a, **k: reference)
    if consumer == "fourier":
        result = build_group_signal_dataset(
            combined, 1, center_signal=False, apply_lifetime_correction=False
        )
        expected = 150
    else:
        result = reduce_combined_run(combined, reference=reference)
        expected = 100 * (150 - 95) / 245
    np.testing.assert_allclose(result.time, np.arange(9) / 10, atol=1e-14)
    np.testing.assert_allclose(result.asymmetry, expected)


@pytest.mark.parametrize("consumer", ["fourier", "integral"])
def test_short_reference_covers_requested_time_crop_without_observed_padding(monkeypatch, consumer):
    combined = combine_runs([acquired(1, (120, 80)), acquired(2, (40, 20))])
    combined.grouping.update(
        background_correction=True,
        background_mode="reference_run",
        background_run={"run_number": 3},
    )
    reference = BackgroundReference(
        [Histogram(np.full(4, value), 0.1) for value in (10, 5)],
        ReferenceScaleRequest("manual", 1).resolve(),
        source_file="synthetic-reference.nxs",
        run_number=3,
    )
    monkeypatch.setattr("asymmetry.core.io.resolve_background_reference", lambda *a, **k: reference)
    if consumer == "fourier":
        result = build_group_signal_dataset(
            combined, 1, center_signal=False, apply_lifetime_correction=False, time_range=(0, 0.31)
        )
        np.testing.assert_allclose(result.time, np.arange(4) / 10, atol=1e-14)
        np.testing.assert_allclose(result.asymmetry, 150)
    else:
        assert reduce_period_integral(
            combined, t_max=0.31, reference=reference
        ).value == pytest.approx(55 / 245)


@pytest.mark.parametrize("consumer", ["count", "fourier"])
def test_generic_consumers_pool_period_sums_and_diagnose_asymmetry_differences(consumer):
    runs = [acquired(1, (120, 80)), acquired(2, (40, 20))]
    for run in runs:
        run.grouping.update(
            period_histograms=[
                run.histograms,
                [replace(h, counts=h.counts * 3) for h in run.histograms],
            ],
            period_mode="green_plus_red",
        )
    combined = combine_runs(runs)
    data = reduce_combined_run(combined)
    if consumer == "count":
        values = build_count_groups(data, [1], lifetime_corrected=False)[0].counts
    else:
        values = build_group_signal_dataset(
            combined, 1, center_signal=False, apply_lifetime_correction=False
        ).asymmetry
    np.testing.assert_allclose(values, 640)
    combined.grouping["period_mode"] = "green_minus_red"
    data = reduce_combined_run(combined)
    with pytest.raises((FitInputError, ValueError), match="asymmetry"):
        if consumer == "count":
            build_count_groups(data, [1], lifetime_corrected=False)
        else:
            build_group_signal_dataset(combined, 1)


def test_resolved_integral_reference_reaches_public_wrappers_and_scan():
    combined = combine_runs([acquired(1, (120, 80)), acquired(2, (40, 20))])
    combined.metadata["field"] = 10
    combined.grouping.update(
        background_correction=True,
        background_mode="reference_run",
        background_run={"run_number": 3, "scale_request": {"mode": "manual", "coefficient": 2}},
    )
    reference = BackgroundReference(
        acquired(3, (10, 5)).histograms,
        ReferenceScaleRequest("manual", 2).resolve(),
        source_file="synthetic-reference.nxs",
        run_number=3,
    )
    expected = 50 / 230
    assert integrate_run(combined, reference=reference)[0] == pytest.approx(expected)
    assert reduce_integral_run(combined, reference=reference).value == pytest.approx(expected)
    scan = build_field_scan([combined], reference=reference)
    assert scan.n_points == 1
    assert scan.value[0] == pytest.approx(expected)


@pytest.mark.parametrize("consumer", ["fourier", "integral"])
def test_reference_unknown_noise_outside_requested_window_is_not_consumed(monkeypatch, consumer):
    combined = combine_runs([acquired(1, (120, 80)), acquired(2, (40, 20))])
    combined.grouping.update(
        background_correction=True,
        background_mode="reference_run",
        background_run={"run_number": 3, "scale_request": {"mode": "manual", "coefficient": 1}},
    )
    source = acquired(3, (10, 5))
    source.histograms[0].counts[0] = -1
    reference = BackgroundReference(
        source.histograms,
        ReferenceScaleRequest("manual", 1).resolve(),
        source_file="synthetic-reference.nxs",
        run_number=3,
    )
    monkeypatch.setattr("asymmetry.core.io.resolve_background_reference", lambda *a, **k: reference)
    if consumer == "fourier":
        result = build_group_signal_dataset(
            combined,
            1,
            center_signal=False,
            apply_lifetime_correction=False,
            time_range=(0.2, 0.71),
        )
        np.testing.assert_allclose(result.asymmetry, 150)
    else:
        assert reduce_period_integral(
            combined, t_min=0.2, t_max=0.71, reference=reference
        ).value == pytest.approx(55 / 245)


def test_single_acquisition_model_is_available_after_a_zero_term():
    runs = [acquired(1, (120, 80)), acquired(2, (40, 20))]
    data = reduce_combined_run(combine_runs(runs, scales=[1, 0]))
    group = build_count_groups(data, [1], lifetime_corrected=False)[0]
    assert count_model_frame_norm(data, group) == 10000
    source, groups, _ = _prepare_tf_counts(data, TFCalibrationRequest(1, 2), None)
    assert source.run.source_file == runs[0].source_file
    np.testing.assert_allclose(groups[0].counts, 120)
    with pytest.raises(FitInputError, match="original acquisition"):
        _prepare_tf_counts(
            reduce_combined_run(combine_runs(runs)), TFCalibrationRequest(1, 2), None
        )


def test_fourier_identity_tracks_replaced_active_buffers_and_ignores_zero_sources():
    runs = [acquired(1, (120, 80)), acquired(2, (40, 20))]
    combined = combine_runs(runs, scales=[1, 0])
    prepared = prepare_group_counts(combined, [1])
    runs[1].histograms = [replace(h, counts=h.counts + 7) for h in runs[1].histograms]
    runs[1].grouping["deadtime_file_us"] = [0.2, 0.3]
    prepared.require_request(combined, 1, False, None, None)
    runs[0].histograms = [replace(h, counts=h.counts + 7) for h in runs[0].histograms]
    with pytest.raises(ValueError, match="do not match"):
        prepared.require_request(combined, 1, False, None, None)


def test_count_fit_domain_maps_flat_channels_to_compact_original_period_sources():
    runs = [acquired(11, (120, 80)), acquired(22, (40, 20))]
    for run in runs:
        run.metadata.update(data_period_number=2, dae_period_number=7, large_log=np.arange(1000))
    data = reduce_combined_run(combine_runs(runs, scales=[2, -0.5]))
    group = build_count_groups(data, [1], lifetime_corrected=False)[0]
    domain = count_fit_domain(group, data)
    assert domain.detector_indices == (0, 2)
    assert domain.acquired_sources == (
        (0, 2, "synthetic-11.nxs", 11, 1, 2.0, 2, 7),
        (2, 2, "synthetic-22.nxs", 22, 2, -0.5, 2, 7),
    )
    assert domain.alignment.acquired_detectors
    payload = json.loads(json.dumps(_domain_payload(domain)))
    assert "large_log" not in str(payload)
    assert _restore_domain(payload) == domain
    del payload["acquired_sources"]
    assert _restore_domain(payload).acquired_sources == ()


@pytest.mark.parametrize("reverse", [False, True])
def test_shared_automatic_ratio_is_estimated_once_on_pooled_counts_and_replayed(reverse):
    runs = automatic_runs()
    assert [r.grouping["alpha"] for r in runs] == [1.5, 9]
    assert coadd_compatibility_error(runs) is None
    combined = combine_runs(runs[::-1] if reverse else runs)
    data = reduce_combined_run(combined)
    assert data.run.grouping["alpha"] == pytest.approx(3)
    for result in (data, build_fb_signal_dataset(combined), prepare_wizard_dataset(data, 3)):
        np.testing.assert_allclose(result.asymmetry, 0, atol=1e-13)
    assert reduce_period_integral(combined).value == pytest.approx(0, abs=1e-13)


def test_shared_automatic_ratio_consumes_full_calibration_domain_even_when_output_is_cropped():
    combined = combine_runs(automatic_runs())
    combined.grouping.pop("alpha_policy")
    combined.grouping["alpha"] = 1
    original = combined.grouping["period_inputs"][0]
    original.histograms[0].counts[8] = 0.5
    assert prepare(combined, merge=2).time.size == 4
    assert np.isfinite(reduce_period_integral(combined, t_max=0.3).value)
    combined.grouping["alpha_policy"] = {"mode": "per_run_estimate"}
    with pytest.raises(ValueError, match="unknown independent count noise"):
        prepare(combined, merge=2)
    with pytest.raises(ValueError, match="unknown independent count noise"):
        reduce_period_integral(combined, t_max=0.3)
    assert reduce_period_integral(combined, t_max=0.3, alpha=2).value == pytest.approx(0.2)


def test_explicit_projection_balance_takes_precedence_over_automatic_profile():
    combined = combine_runs(automatic_runs())
    combined.grouping["projections"] = [
        {"label": "FB", "forward_group": 1, "backward_group": 2, "alpha": 2}
    ]
    np.testing.assert_allclose(prepare(combined).asymmetry, 20)


def test_period_difference_uses_shared_red_calibration_for_both_conditions():
    runs = automatic_runs()
    for run in runs:
        run.grouping.update(
            period_histograms=[
                run.histograms,
                [
                    replace(h, counts=np.full(16, v))
                    for h, v in zip(run.histograms, (80, 20), strict=True)
                ],
            ],
            period_mode="green_minus_red",
        )
    combined = combine_runs(runs)
    result = prepare(combined)
    np.testing.assert_allclose(result.asymmetry, 100 / 7)
    assert reduce_period_integral(combined).value == pytest.approx(1 / 7)
