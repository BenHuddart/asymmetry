"""Independent scalar ratio and reused-reference counting controls."""

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.data.ratio_validity import RatioReason
from asymmetry.core.transform.corrections import BackgroundReference
from asymmetry.core.transform.fb_counts import GroupCounts
from asymmetry.core.transform.integral import (
    integrate_asymmetry,
    integrate_curve,
    reduce_period_integral,
)
from asymmetry.core.transform.reference_scale import ReferenceScaleRequest


def shared_reference_run(*, green=(600.0, 400.0), scale=1.0, alpha=1.0, beta=1.0):
    red = [Histogram(np.full(4, x), 0.1) for x in (600.0, 400.0)]
    greens = [Histogram(np.full(4, x), 0.1) for x in green]
    reference = BackgroundReference(
        [Histogram(np.full(4, x), 0.1) for x in (60.0, 40.0)],
        ReferenceScaleRequest("manual", scale).resolve(),
        run_number=13003,
        source_file="reference.nxs",
    )
    grouping = dict(
        groups={1: [1], 2: [2]},
        forward_group=1,
        backward_group=2,
        period_histograms=[red, greens],
        period_count=2,
        period_mode="green_minus_red",
        alpha=alpha,
        beta=beta,
        deadtime_correction=False,
        background_correction=True,
        background_mode="reference_run",
        background_run={
            "source_file": "reference.nxs",
            "run_number": 13003,
            "scale_request": ReferenceScaleRequest("manual", scale).to_dict(),
        },
    )
    return Run(
        13002, red, source_file="sample.nxs", metadata={"period_count": 2}, grouping=grouping
    ), reference


def test_sparse_native_bins_are_summed_before_classification():
    f, b = GroupCounts([0.0, 10.0]), GroupCounts([10.0, 0.0])
    assert integrate_asymmetry(f, b) == pytest.approx((0.0, np.sqrt(0.05)))
    with pytest.raises(ValueError, match="no usable"):
        integrate_asymmetry(f, b, method="differential")


def test_zero_error_supplied_curve_is_omitted_and_reported():
    assert integrate_curve([0.0, 1.0], [0.2, 0.8], [0.1, 0.0]) == pytest.approx((0.2, 0.1))
    result = reduce_period_integral(
        MuonDataset(np.array([0.0, 1.0]), np.array([20.0, 80.0]), np.array([10.0, 0.0])),
        method="differential",
    )
    assert result.observation.report.to_dict() == dict(
        used=1,
        omitted=1,
        undefined_ratio=0,
        zero_variance=1,
    )
    assert result.corrections == {}


@pytest.mark.parametrize(
    "f,b,reason,value",
    [
        (0.0, 0.0, RatioReason.UNDEFINED, np.nan),
        (0.0, 10.0, RatioReason.ZERO_VARIANCE, -1.0),
    ],
)
def test_scalar_never_gives_unusable_ratio_finite_weight(f, b, reason, value):
    actual, error = integrate_asymmetry(GroupCounts([f]), GroupCounts([b]))
    assert np.isnan(actual) if np.isnan(value) else actual == value
    assert np.isnan(error) if reason == RatioReason.UNDEFINED else error == 0.0


@pytest.mark.parametrize(
    "green,scale,alpha,beta",
    [
        ((600.0, 400.0), 1.0, 1.0, 1.0),
        ((700.0, 350.0), 0.6, 1.3, 0.8),
    ],
)
@pytest.mark.parametrize("method", ["integral", "differential"])
def test_reference_responses_combine_before_variance(green, scale, alpha, beta, method):
    run, reference = shared_reference_run(green=green, scale=scale, alpha=alpha, beta=beta)
    result = reduce_period_integral(run, reference=reference, method=method)
    gradients, values = [], []
    for raw in ((600.0, 400.0), green):
        f, b = 4 * (np.array(raw) - scale * np.array([60.0, 40.0]))
        d = beta * f + alpha * b
        gradients.append(alpha * (1 + beta) * np.array([b, -f]) / d**2)
        values.append((f - alpha * b) / d)
    red, grn = gradients
    variance = (
        np.dot(red**2, 4 * np.array([600.0, 400.0]))
        + np.dot(grn**2, 4 * np.array(green))
        + np.dot((scale * (red - grn)) ** 2, 4 * np.array([60.0, 40.0]))
    )
    assert result.value == pytest.approx(values[1] - values[0])
    assert result.error == pytest.approx(np.sqrt(variance), rel=1e-13)
    if green == (600.0, 400.0):
        assert result.error == pytest.approx(0.024343224778007384)


def test_signed_counts_with_supplied_variance_remain_supported():
    f, b = GroupCounts([-2.0, 6.0], [1.0, 1.0]), GroupCounts([3.0, 2.0], [1.0, 1.0])
    assert integrate_asymmetry(f, b) == pytest.approx((-1 / 9, np.sqrt(328) / 81))
    value, error = integrate_asymmetry(f, b, method="differential")
    assert value == pytest.approx(-2.25)
    assert error == pytest.approx(np.sqrt(13.009765625))


def test_pointwise_difference_classifies_combination_before_mean():
    from dataclasses import replace

    run, _ = shared_reference_run()
    red = [Histogram(np.array([10.0, 0.0]), 0.1), Histogram(np.array([0.0, 0.0]), 0.1)]
    green = [Histogram(np.array([6.0, 6.0]), 0.1), Histogram(np.array([4.0, 4.0]), 0.1)]
    run = replace(
        run,
        histograms=red,
        grouping=dict(run.grouping, background_correction=False, period_histograms=[red, green]),
    )
    result = reduce_period_integral(run, method="differential")
    assert result.value == pytest.approx(-0.8)
    assert result.error == pytest.approx(np.sqrt(0.096))
    assert result.observation.report.to_dict() == dict(
        used=1, omitted=1, undefined_ratio=1, zero_variance=0
    )


def test_known_scan_correlation_survives_payload_crop_and_independent_subset():
    import json
    from dataclasses import replace

    from asymmetry.core.fitting.field_scan import fit_scan_baseline, fit_scan_model
    from asymmetry.core.transform.integral import build_field_scan
    from asymmetry.core.workflow.integral_scan import field_scan_from_payload, field_scan_payload

    run, reference = shared_reference_run()
    run = replace(run, grouping=dict(run.grouping, period_mode="red"))
    runs = [
        replace(run, run_number=i, source_file=f"sample-{i}.nxs", metadata={"field": float(i)})
        for i in range(1, 4)
    ]
    scan = build_field_scan(runs, reference=reference)
    assert scan.n_points == 3
    assert scan.correlated_pairs == ((0, 1), (0, 2), (1, 2))
    restored = field_scan_from_payload(
        json.loads(json.dumps(field_scan_payload(scan), allow_nan=False))
    )
    assert restored.observations == scan.observations
    assert restored.correlated_pairs == scan.correlated_pairs
    for candidate in (scan, restored, restored.select(np.array([True, False, True]))):
        with pytest.raises(ValueError, match="reuse counting observations"):
            fit_scan_model(candidate, "Constant")
    assert fit_scan_baseline(restored, [(0.9, 1.1)], model="Constant").success
    assert fit_scan_model(restored, "Constant", x_min=0.9, x_max=1.1).success


def test_defined_zero_variance_and_undefined_points_roundtrip_without_weight():
    import json

    from asymmetry.core.fitting.field_scan import fit_scan_model
    from asymmetry.core.transform.integral import build_field_scan
    from asymmetry.core.workflow.integral_scan import field_scan_from_payload, field_scan_payload
    from tests.core.test_time_integral_asymmetry import _run

    runs = [
        _run([0.0], [0.0], run_number=1),
        _run([0.0], [10.0], run_number=2),
        _run([6.0], [4.0], run_number=3),
    ]
    scan = build_field_scan(runs)
    payload = field_scan_payload(scan)
    assert payload["points"][0]["value"] is None
    assert payload["points"][0]["error"] is None
    scan = field_scan_from_payload(json.loads(json.dumps(payload, allow_nan=False)))
    with pytest.warns(UserWarning, match="used 1, omitted 2"):
        fit = fit_scan_model(scan, "Constant")
    assert fit.n_points == 1
    assert fit.parameters["c"].value == pytest.approx(0.2)


def test_distinct_periods_and_unknown_sources_are_not_falsely_correlated():
    from dataclasses import replace

    from asymmetry.core.transform.integral import build_field_scan
    from tests.core.test_time_integral_asymmetry import _run

    run = _run([60.0, 60.0], [40.0, 40.0])
    runs = [
        replace(
            run,
            run_number=i,
            source_file="acquired.nxs",
            metadata={"source_run_number": 1, "data_period_number": i},
        )
        for i in (1, 2)
    ]
    assert build_field_scan(runs, order_key="run").correlated_pairs == ()
    runs = [
        replace(run, run_number=1, source_file=None),
        replace(run, run_number=2, source_file=None),
    ]
    scan = build_field_scan(runs, order_key="run")
    assert scan.correlated_pairs == ()
    assert scan.source_assumption == "Unknown acquisition identity is assumed distinct."


def test_supplied_shared_background_mean_retains_estimator_overlap():
    from tests.core.test_background_uncertainty import _background_counts

    counts, _ = _background_counts(background_range=(1, 2))
    value, error = integrate_asymmetry(
        counts.forward,
        counts.backward,
        method="differential",
        time=np.arange(6, dtype=float),
        t_min=2.0,
        t_max=5.0,
    )
    # Each selected corrected bank is flat. The mean equals the aggregate,
    # whose raw coefficients are [-2,-1,1,1,1] on bins 1..5.
    expected = 2 * np.sqrt(144**2 * 432 + 184**2 * 352) / 328**2
    assert value == pytest.approx(40 / 328)
    assert error == pytest.approx(expected)


def test_curve_mean_contracts_retained_shared_counting_responses():
    from scipy.sparse import csr_matrix

    from asymmetry.core.transform.counting_noise import (
        CountingNoise,
        CountingObservation,
        Influence,
    )

    source = CountingObservation(np.ones(3), object())
    noise = CountingNoise((Influence(source, csr_matrix([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])),))
    dataset = MuonDataset(
        np.array([0.0, 1.0]), np.array([20.0, 80.0]), np.full(2, np.sqrt(2)), counting_noise=noise
    )
    result = reduce_period_integral(dataset, method="differential")
    assert result.value == pytest.approx(0.5)
    assert result.error == pytest.approx(np.sqrt(1.5) / 100)
    assert result.corrections == {}


def test_canceled_reference_does_not_veto_scan_fit():
    from dataclasses import replace

    from asymmetry.core.transform.integral import build_field_scan

    run, reference = shared_reference_run()
    runs = [
        replace(run, run_number=i, source_file=f"sample-{i}.nxs", metadata={"field": float(i)})
        for i in range(1, 4)
    ]
    scan = build_field_scan(runs, reference=reference)
    assert scan.correlated_pairs == ()
    assert scan.error == pytest.approx(np.full(3, 0.024343224778007384))


def test_disjoint_native_support_does_not_veto_scan_fit():
    from dataclasses import replace

    from asymmetry.core.transform.integral import build_field_scan

    run, reference = shared_reference_run()
    runs = [
        replace(
            run,
            run_number=i + 1,
            metadata={"source_run_number": 13002, "field": float(i)},
            grouping=dict(
                run.grouping, period_mode="red", first_good_bin=2 * i, last_good_bin=2 * i + 1
            ),
        )
        for i in range(2)
    ]
    scan = build_field_scan(runs, reference=reference)
    assert scan.n_points == 2
    assert scan.correlated_pairs == ()


def test_curve_mean_uses_retained_variance_before_bin_selection():
    from scipy.sparse import eye

    from asymmetry.core.transform.counting_noise import (
        CountingNoise,
        CountingObservation,
        Influence,
    )

    source = CountingObservation(np.array([100.0, 0.0]), object())
    dataset = MuonDataset(
        np.array([0.0, 1.0]),
        np.array([20.0, 80.0]),
        np.array([10.0, 10.0]),
        counting_noise=CountingNoise((Influence(source, eye(2, format="csr")),)),
    )
    result = reduce_period_integral(dataset, method="differential")
    assert (result.value, result.error) == pytest.approx((0.2, 0.1))
    assert result.observation.report.zero_variance == 1


def test_repeated_run_labels_do_not_veto_single_point_selection():
    from dataclasses import replace

    from asymmetry.core.fitting.field_scan import fit_scan_model
    from asymmetry.core.transform.integral import build_field_scan

    run, reference = shared_reference_run()
    run = replace(run, grouping=dict(run.grouping, period_mode="red"))
    runs = [
        replace(run, run_number=1, source_file=f"sample{i}.nxs", metadata={"field": float(i)})
        for i in (1, 2)
    ]
    scan = build_field_scan(runs, reference=reference)
    assert scan.correlated_pairs == ((0, 1),)
    assert fit_scan_model(scan, "Constant", x_min=0.9, x_max=1.1).success


def test_counting_capture_preserves_tail_fit_preparation_and_scalar_error():
    from asymmetry.core.transform.fb_counts import linear_combination_variance
    from asymmetry.core.transform.reduce import corrected_grouped_counts
    from tests.core.test_background_modes import BIN_WIDTH_US, N_BINS, T0_BIN, _pulsed_counts
    from tests.core.test_time_integral_asymmetry import _run

    run = _run(
        _pulsed_counts(30000.0, 20.0, seed=3),
        _pulsed_counts(20000.0, 10.0, seed=4),
        t0=T0_BIN,
        first_good=T0_BIN + 1,
        last_good=N_BINS - 1,
        bin_width=BIN_WIDTH_US,
    )
    run.grouping.update(
        background_mode="tail_fit", background_correction=True, deadtime_correction=False
    )
    counts = corrected_grouped_counts(
        histograms=run.histograms,
        grouping=run.grouping,
        forward_idx=[0],
        backward_idx=[1],
        use_deadtime=False,
        deadtime_mode="off",
        use_background=True,
        facility="",
    )
    edges = np.array([counts.window.first, counts.window.last + 1])
    f, b = [g.counts[counts.window.slice].sum() for g in (counts.forward, counts.backward)]
    vf, vb = [linear_combination_variance(g, edges)[0] for g in (counts.forward, counts.backward)]
    expected = 2 * np.sqrt(b * b * vf + f * f * vb) / (f + b) ** 2
    assert reduce_period_integral(run).error == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("method", ["integral", "differential"])
def test_shared_reference_coefficients_cancel_before_large_variance_contraction(method):
    from dataclasses import replace

    run, reference = shared_reference_run()
    reference = replace(
        reference,
        histograms=[
            replace(h, count_variance=np.full(h.n_bins, 1e30)) for h in reference.histograms
        ],
    )
    result = reduce_period_integral(run, reference=reference, method=method)
    # The identical reference response is exactly zero; its supplied variance
    # cannot erase the independent sample contribution through cancellation.
    assert result.error == pytest.approx(0.024343224778007384, rel=1e-13)


def test_supplied_zero_variance_retains_defined_value_without_gaussian_weight():
    from dataclasses import replace

    from tests.core.test_time_integral_asymmetry import _run

    run = _run([6.0, 6.0], [4.0, 4.0])
    run = replace(run, histograms=[replace(h, count_variance=np.zeros(2)) for h in run.histograms])
    result = reduce_period_integral(run)
    assert (result.value, result.error) == pytest.approx((0.2, 0.0))
    assert result.observation.reason == RatioReason.ZERO_VARIANCE
    with pytest.raises(ValueError, match="no usable"):
        reduce_period_integral(run, method="differential")


def test_same_valued_distinct_reference_acquisitions_remain_independent():
    from dataclasses import replace

    from asymmetry.core.transform.integral import build_field_scan
    from asymmetry.core.transform.preparation import histogram_observation

    run, reference = shared_reference_run()
    datasets = []
    for number in (1, 2):
        selected_reference = replace(
            reference, source_file=f"reference-{number}.nxs", run_number=number + 100
        )
        grouping = dict(
            run.grouping,
            period_mode="red",
            background_run={
                "source_file": selected_reference.source_file,
                "run_number": selected_reference.run_number,
                "scale_request": ReferenceScaleRequest("manual", 1.0).to_dict(),
            },
        )
        source = replace(
            run,
            run_number=number,
            source_file=f"sample-{number}.nxs",
            grouping=grouping,
            metadata={"field": float(number)},
        )
        datasets.append(
            MuonDataset(
                np.array([]),
                np.array([]),
                np.array([]),
                run=source,
                histogram_observation=histogram_observation(source, reference=selected_reference),
            )
        )
    scan = build_field_scan(datasets)
    assert scan.n_points == 2
    assert scan.correlated_pairs == ()
    df, db = 2 * 1440 / 3600**2, -2 * 2160 / 3600**2
    assert scan.error == pytest.approx(np.full(2, np.sqrt(df**2 * 2640 + db**2 * 1760)))


@pytest.mark.parametrize("size", [1, 2])
def test_low_rank_counting_responses_combine_before_scalar_classification(size):
    from scipy.sparse import eye

    from asymmetry.core.transform.counting_noise import (
        CountingNoise,
        CountingObservation,
        Influence,
    )

    source = CountingObservation(np.ones(size), object())
    sensitivity, influence = np.zeros(size), np.zeros(size)
    sensitivity[0], influence[0] = 1.0, -1.0 + 1e-8
    noise = CountingNoise(
        (Influence(source, eye(size, format="csr"), ((sensitivity, influence),)),)
    )
    values = np.array([20.0, 80.0])[:size]
    dataset = MuonDataset(
        np.arange(size, dtype=float), values, np.full(size, 10.0), counting_noise=noise
    )
    result = reduce_period_integral(dataset, method="differential")
    response = np.array([1.0 + influence[0], 1.0])[:size] / size
    assert result.value == pytest.approx(float(values.mean()) / 100)
    assert result.error == pytest.approx(np.linalg.norm(response) / 100, rel=1e-7, abs=1e-17)
    assert result.observation.report.used == size
