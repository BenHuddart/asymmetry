"""Independent acquisition-level controls for transverse counting noise."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.transform.corrections import BackgroundReference
from asymmetry.core.transform.projections import reduce_run_projections, reduce_selected_projections
from asymmetry.core.transform.reference_scale import ReferenceScaleRequest
from asymmetry.core.transform.rotating_frame import Provenance, RotatingFrame, rotate_transverse


def shared_run():
    signs = np.tile([[1, 1, -1, -1], [1, -1, 1, -1]], (1, 24))
    rates = 1000 * (1 + 0.6 * signs[0] * signs[1] + 0.1 * signs[0] + 0.05 * signs[1])
    groups = {
        2 * axis + side + 1: (np.flatnonzero(s == sign) + 1).tolist()
        for axis, s in enumerate(signs)
        for side, sign in enumerate((1, -1))
    }
    run = Run(
        run_number=901,
        histograms=[Histogram(np.full(4, r), 0.125) for r in rates],
        grouping={
            "groups": groups,
            "projections": [
                {"label": label, "forward_group": 2 * i + 1, "backward_group": 2 * i + 2}
                for i, label in enumerate(("P_x", "P_y"))
            ],
            "alpha": 1.0,
            "t0_bin": 0,
        },
    )
    return run, signs, rates


def test_shared_counting_noise_rotates_with_the_observations():
    run, signs, rates = shared_run()
    curves = reduce_run_projections(run, 0)
    frame = RotatingFrame.typed_frequency(1.0, 1)
    rotated = rotate_transverse(curves["P_x"], curves["P_y"], frame, weights=(1.0,))
    total = rates.sum()
    jacobian = 100 * (signs * total - (signs @ rates)[:, None]) / total**2
    covariance = (jacobian * rates) @ jacobian.T
    theta = 2 * np.pi * rotated[0].time[1]
    matrix = np.array(
        [
            [np.cos(theta), frame.sense * np.sin(theta)],
            [np.sin(theta), -frame.sense * np.cos(theta)],
        ]
    )
    expected = matrix @ covariance @ matrix.T
    np.testing.assert_allclose([c.error[1] for c in rotated], np.sqrt(np.diag(expected)))
    np.testing.assert_allclose(
        [c.asymmetry[1] for c in rotated], matrix @ (100 * signs @ rates / total)
    )


def small_run(values=None):
    if values is None:
        values = np.array([1800.0, 900.0, 700.0, 1400.0])[:, None] + 23 * np.arange(8)
    run, _, _ = shared_run()
    run.histograms = [Histogram(v.copy(), 0.125, count_variance=1.4 * v) for v in values]
    run.grouping.update(
        groups={1: [1, 2], 2: [3, 4], 3: [1, 3], 4: [2, 4]}, alpha_x=0.73, alpha_y=1.21, beta=1.17
    )
    return run


def independent_rotation(values, frame, *, reference=None, background=False, mode="red", pack=1):
    """Authored observation algebra, with numerical count derivatives as oracle."""
    axes = []
    for alpha, forward, backward in ((0.73, [0, 1], [2, 3]), (1.21, [0, 2], [1, 3])):
        ratios = []
        sets = [values.sum(axis=0)] if mode == "green_plus_red" else list(values)
        for counts in sets:
            f, b = counts[forward].sum(axis=0), counts[backward].sum(axis=0)
            if reference is not None:
                f = f - 0.4 * reference[forward].sum(axis=0)
                b = b - 0.4 * reference[backward].sum(axis=0)
            if background:
                f, b = f - f[:2].mean(), b - b[:2].mean()
            denominator = f + alpha * b
            ratios.append(100 * (f - alpha * b) / denominator)
        axes.append(ratios[-1] - ratios[0] if mode == "green_minus_red" else ratios[0])
    x, y = axes[0], axes[1] / frame.gain
    theta = 2 * np.pi * frame.frequency_mhz * np.arange(x.size) * 0.125 + np.deg2rad(
        frame.rf_phase_deg
    )
    rotated = np.array(
        [
            x * np.cos(theta) + frame.sense * y * np.sin(theta),
            x * np.sin(theta) - frame.sense * y * np.cos(theta),
        ]
    )
    if background:
        rotated = rotated[:, 2:]
    size = rotated.shape[1] // pack
    return rotated[:, : size * pack].reshape(2, size, pack).mean(axis=2)


@pytest.mark.parametrize("sense,phase,gain", [(1, 23.0, 1.7), (-1, -37.0, 0.8)])
@pytest.mark.parametrize(
    "mode,background,pack",
    [
        ("red", False, 1),
        ("red", True, 2),
        ("green_minus_red", False, 2),
        ("green_plus_red", False, 1),
    ],
)
def test_calibrated_reference_and_period_noise_against_independent_differences(
    sense, phase, gain, mode, background, pack
):
    run = small_run()
    values = np.stack(
        [
            [h.counts for h in run.histograms],
            [h.counts * (1.2 + 0.1 * i) for i, h in enumerate(run.histograms)],
        ]
    )
    if mode != "red":
        run.grouping["period_histograms"] = [
            [Histogram(v.copy(), 0.125, count_variance=1.4 * v) for v in period]
            for period in values
        ]
        run.histograms = run.grouping["period_histograms"][0]
    else:
        values = values[:1]
    run.grouping["period_mode"] = mode
    ref_values = np.array([40.0, 80.0, 50.0, 60.0])[:, None] + 3 * np.arange(8)
    reference = BackgroundReference(
        [Histogram(v, 0.125) for v in ref_values],
        ReferenceScaleRequest("manual", 0.4).resolve(),
        run_number=902,
    )
    run.grouping.update(
        background_correction=True, background_mode="range" if background else "reference_run"
    )
    if background:
        run.grouping.update(background_range=[0, 1], first_good_bin=2)
    frame = RotatingFrame.typed_frequency(0.61, 2 if mode != "red" else 1).with_values(
        Provenance.TYPED, sense=sense, rf_phase_deg=phase, gain=gain
    )
    curves = reduce_selected_projections(run, reference=reference)
    weights = (
        (1.0,) if mode == "red" else ((-1.0, 1.0) if mode == "green_minus_red" else (0.5, 0.5))
    )
    actual = [
        c.merge_curve_bins(pack)
        for c in rotate_transverse(curves["P_x"], curves["P_y"], frame, weights=weights)
    ]
    use_ref = None if background else ref_values
    expected = independent_rotation(
        values, frame, reference=use_ref, background=background, mode=mode, pack=pack
    )
    np.testing.assert_allclose([c.asymmetry for c in actual], expected, rtol=1e-12, atol=1e-12)
    buffers = [values] if background else [values, ref_values]
    variances = [1.4 * values] if background else [1.4 * values, ref_values]
    variance = np.zeros_like(expected)
    for array, source_variance in zip(buffers, variances, strict=True):
        for index in np.ndindex(array.shape):
            step = 0.001
            array[index] += step
            plus = independent_rotation(
                values, frame, reference=use_ref, background=background, mode=mode, pack=pack
            )
            array[index] -= 2 * step
            minus = independent_rotation(
                values, frame, reference=use_ref, background=background, mode=mode, pack=pack
            )
            array[index] += step
            variance += ((plus - minus) / (2 * step)) ** 2 * source_variance[index]
    np.testing.assert_allclose([c.error for c in actual], np.sqrt(variance), rtol=2e-8, atol=1e-10)


@pytest.mark.parametrize("coupling", [0.0, 0.6, -0.6])
def test_shared_covariance_sign_cancellation_and_seeded_poisson(coupling):
    run, signs, _ = shared_run()
    rates = 1000 * (1 + coupling * signs[0] * signs[1])
    run.histograms = [Histogram(np.full(4, r), 0.125) for r in rates]
    curves = reduce_run_projections(run, 0)
    covariance = curves["P_x"].counting_noise.covariance(curves["P_y"].counting_noise)
    np.testing.assert_allclose(covariance, 10000 * coupling / rates.sum(), atol=1e-15)
    frame = RotatingFrame.typed_frequency(1.0, 1)
    actual = rotate_transverse(curves["P_x"], curves["P_y"], frame, weights=(1.0,))
    rng = np.random.default_rng(120011)
    draws = rng.poisson(rates, size=(50000, rates.size))
    ratios = 100 * (draws @ signs.T) / draws.sum(axis=1)[:, None]
    theta = 2 * np.pi * actual[0].time[1]
    matrix = np.array(
        [
            [np.cos(theta), frame.sense * np.sin(theta)],
            [np.sin(theta), -frame.sense * np.cos(theta)],
        ]
    )
    # 50,000 draws give about 0.32% relative sigma sampling error; 2% is >6 sigma.
    np.testing.assert_allclose(
        np.std(ratios @ matrix.T, axis=0, ddof=1), [c.error[1] for c in actual], rtol=0.02
    )


def test_disjoint_observations_recover_independent_curve_errors():
    run = small_run()
    run.grouping["groups"] = {1: [1], 2: [2], 3: [3], 4: [4]}
    curves = reduce_selected_projections(run)
    np.testing.assert_array_equal(
        curves["P_x"].counting_noise.covariance(curves["P_y"].counting_noise), 0
    )
    frame = RotatingFrame.typed_frequency(0.4, 1)
    shared = rotate_transverse(curves["P_x"], curves["P_y"], frame, weights=(1.0,))
    plain = rotate_transverse(
        *(replace(c, counting_noise=None) for c in curves.values()), frame, weights=(1.0,)
    )
    for a, b in zip(shared, plain, strict=True):
        np.testing.assert_allclose(a.error, b.error)


def test_nonunit_beta_count_jacobian_uses_fraction_then_percent_once():
    from asymmetry.core.transform.counting_noise import capture_counting_observations
    from asymmetry.core.transform.rebin import BinningSpec
    from asymmetry.core.transform.reduce import (
        corrected_grouped_counts,
        grouped_asymmetry_from_counts,
    )

    run, _ = capture_counting_observations(small_run(), None)
    run.grouping.pop("projections")
    for alpha, beta in ((0.7, 1.3), (1.4, 0.8)):
        counts = corrected_grouped_counts(
            histograms=run.histograms,
            grouping=run.grouping,
            forward_idx=[0, 1],
            backward_idx=[2, 3],
            use_deadtime=False,
            deadtime_mode="off",
            use_background=False,
        )
        reduced = grouped_asymmetry_from_counts(counts, BinningSpec(), alpha, beta, None)
        f, b = counts.forward.counts, counts.backward.counts
        delta = 0.001

        def ratio(x, y):
            return 100 * (x - alpha * y) / (beta * x + alpha * y)

        jf = (ratio(f + delta, b) - ratio(f - delta, b)) / (2 * delta)
        jb = (ratio(f, b + delta) - ratio(f, b - delta)) / (2 * delta)
        expected = (
            jf**2 * counts.forward.independent_variance
            + jb**2 * counts.backward.independent_variance
        )
        np.testing.assert_allclose(reduced.counting_noise.error**2, expected, rtol=2e-8)
        np.testing.assert_allclose(reduced.error**2, expected, rtol=2e-8)


def test_native_offsets_exclusions_and_fixed_deadtime_preserve_retained_variance():
    run = small_run()
    original = np.array([h.counts for h in run.histograms])
    run.histograms = [
        replace(
            h,
            counts=np.r_[np.zeros(i), h.counts],
            count_variance=np.r_[np.zeros(i), h.count_variance],
            t0_bin=i,
        )
        for i, h in enumerate(run.histograms)
    ]
    run.grouping.update(
        t0_bin=0,
        effective_detector_t0_bins=list(range(4)),
        deadtime_correction=True,
        deadtime_mode="manual",
        dead_time_us=[0.01] * 4,
        good_frames=1e6,
    )
    curves = reduce_selected_projections(run)
    corrected = original / (1 - original * 0.01 / (0.125 * 1e6))
    for label, alpha, f, b in (("P_x", 0.73, [0, 1], [2, 3]), ("P_y", 1.21, [0, 2], [1, 3])):
        forward, backward = corrected[f].sum(axis=0), corrected[b].sum(axis=0)
        jac = (
            np.array([100 * alpha * 2 * backward, -100 * alpha * 2 * forward])
            / (forward + alpha * backward) ** 2
        )
        variance = 1.4 * (
            jac[0] ** 2 * original[f].sum(axis=0) + jac[1] ** 2 * original[b].sum(axis=0)
        )
        np.testing.assert_allclose(curves[label].error ** 2, variance)
        np.testing.assert_allclose(curves[label].counting_noise.error ** 2, variance)
        np.testing.assert_allclose(curves[label].time, np.arange(8) * 0.125)
    run.grouping["excluded_detectors"] = [4]
    excluded = reduce_selected_projections(run)
    assert len(excluded["P_x"].counting_noise.terms) == 3


def test_tail_estimator_influence_and_packing_use_acquisition_variance():
    from asymmetry.core.transform.background import fit_tail_background
    from asymmetry.core.transform.counting_noise import capture_counting_observations
    from asymmetry.core.transform.rebin import BinningSpec, fb_binning_edges
    from asymmetry.core.transform.reduce import corrected_grouped_counts

    t = np.arange(160) * 0.125
    values = (
        np.array([20000, 13000, 9000, 15000])[:, None] * np.exp(-t / 2.1969811)
        + np.array([800, 600, 400, 900])[:, None]
    )
    run = small_run(values)
    run.grouping.update(background_correction=True, background_mode="tail_fit", first_good_bin=2)
    captured, _ = capture_counting_observations(run, None)
    for indices in (([0, 1], [2, 3]), ([0, 2], [1, 3])):
        counts = corrected_grouped_counts(
            histograms=captured.histograms,
            grouping=captured.grouping,
            forward_idx=indices[0],
            backward_idx=indices[1],
            use_deadtime=False,
            deadtime_mode="off",
            use_background=True,
        )
        for group, selected in zip((counts.forward, counts.backward), indices, strict=True):
            raw = values[selected].sum(axis=0)
            fit = fit_tail_background(
                raw, bin_width_us=0.125, t0_bin=0, first_good_bin=2, last_good_bin=159
            )
            start, last = fit.window
            shape = np.exp(-(t[start : last + 1] - t[start]) / 2.1969811)
            design = np.column_stack([shape, np.ones(shape.size)])
            influence = np.linalg.inv(design.T @ (design / fit.expected_counts[:, None])) @ (
                design.T / fit.expected_counts
            )
            weights = np.zeros(160)
            weights[start : last + 1] = influence[1]
            transform = np.eye(160) - np.ones((160, 1)) * weights[None, :]
            expected = (transform * (1.4 * raw)) @ transform.T
            edges, _ = fb_binning_edges(counts, BinningSpec(factor=4))
            pack = np.array(
                [
                    np.isin(np.arange(160), np.arange(a, b)).astype(float)
                    for a, b in zip(edges[:-1], edges[1:], strict=True)
                ]
            )
            np.testing.assert_allclose(
                group.noise.transform(csr_matrix(pack)).error ** 2,
                np.diag(pack @ expected @ pack.T),
                rtol=1e-8,
            )


def test_signed_constituent_weights_precede_ratio_and_shared_covariance():
    from asymmetry.core.data.combine import combine_runs

    a, b = small_run(), small_run()
    b.run_number = 902
    b.histograms = [
        replace(h, counts=h.counts * 0.5, count_variance=h.count_variance * 0.5)
        for h in b.histograms
    ]
    combined = combine_runs([a, b], scales=[1.0, 0.3], sign=-1)
    curves = reduce_selected_projections(combined)
    counts = np.array([h.counts for h in a.histograms]) * 0.85
    variance = np.array([h.count_variance for h in a.histograms]) * (1 + 0.3**2 * 0.5)
    derivatives = []
    for alpha, f, b in ((0.73, [0, 1], [2, 3]), (1.21, [0, 2], [1, 3])):
        forward, backward = counts[f].sum(axis=0), counts[b].sum(axis=0)
        d = np.empty_like(counts)
        d[f] = 100 * 2 * alpha * backward / (forward + alpha * backward) ** 2
        d[b] = -100 * 2 * alpha * forward / (forward + alpha * backward) ** 2
        derivatives.append(d)
    expected = np.sum(derivatives[0] * derivatives[1] * variance, axis=0)
    np.testing.assert_allclose(
        curves["P_x"].counting_noise.covariance(curves["P_y"].counting_noise), expected
    )


def test_shared_detector_at_different_retained_raw_bins_is_not_shared_noise():
    curves = reduce_selected_projections(small_run())
    x = curves["P_x"].time_range(0.0, 0.625)
    y = curves["P_y"].time_range(0.125, 0.75)
    np.testing.assert_array_equal(x.counting_noise.covariance(y.counting_noise), 0.0)


def test_separately_prepared_cohorts_do_not_silently_acquire_independence():
    run = small_run()
    x = reduce_selected_projections(run)["P_x"]
    y = reduce_selected_projections(run)["P_y"]
    with pytest.raises(ValueError, match="together as one counting cohort"):
        rotate_transverse(x, y, RotatingFrame.typed_frequency(0.4, 1), weights=(1.0,))


def test_undefined_rotated_ratios_remain_unusable_after_display_packing():
    run = small_run(np.zeros((4, 8)))
    curves = reduce_selected_projections(run)
    rotated = rotate_transverse(
        *curves.values(), RotatingFrame.typed_frequency(0.4, 1), weights=(1.0,)
    )
    for curve in rotated:
        assert curve.ratio_validity.report().undefined_ratio == 8
        packed = curve.merge_curve_bins(2)
        assert packed.ratio_validity.report().undefined_ratio == 4
