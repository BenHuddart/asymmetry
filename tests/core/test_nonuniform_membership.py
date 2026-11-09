"""Whole-bin target selection against independent count and coordinate oracles."""

from dataclasses import replace

import numpy as np
import pytest

from asymmetry.core.transform.fb_counts import BinWindow
from asymmetry.core.transform.rebin import BinningSpec, binned_fb_asymmetry, fb_binning_edges
from asymmetry.core.utils.constants import MUON_LIFETIME_US
from tests.counts import aligned_counts


@pytest.mark.parametrize("mode", ["variable", "constant_error"])
@pytest.mark.parametrize(
    "target,step",
    [
        (0.01, 1),
        (1.0, 1),
        (1.5, 2),
        (2.0, 2),
        (2.5, 3),
        (np.nextafter(1.5, 0), 1),
        (np.nextafter(1.5, np.inf), 2),
        (np.nextafter(2.5, 0), 2),
        (np.nextafter(2.5, np.inf), 3),
    ],
)
def test_half_up_and_adjacent_representable_targets(mode, target, step):
    # Negative left edges clamp growth, so the supplied width is the target.
    spec = BinningSpec(mode=mode, bin0_us=target, bin10_us=target)
    assert spec.edges(10, 1.0, -100.0)[1] == step


@pytest.mark.parametrize("mode", ["variable", "constant_error"])
def test_coarse_control_membership_and_actual_midpoints(mode):
    counts = aligned_counts(
        1000 + 10 * np.arange(20.0),
        700 + 3 * np.arange(20.0),
        bin_width_us=0.1,
        first_good_bin=2,
        last_good_bin=13,
    )
    spec = BinningSpec(mode=mode, bin0_us=0.12, bin10_us=0.25)
    # At left edge 0.55 us the constant-error target first exceeds 0.15 us.
    edges = [2, 3, 4, 5, 6, 8, 10, 12, 14] if mode == "constant_error" else list(range(2, 15))
    actual_edges, _ = fb_binning_edges(counts, spec)
    np.testing.assert_array_equal(actual_edges, edges)
    times, values, errors = binned_fb_asymmetry(counts, binning=spec, alpha=1)
    f = np.array([sum(counts.forward.counts[i:j]) for i, j in zip(edges[:-1], edges[1:])])
    b = np.array([sum(counts.backward.counts[i:j]) for i, j in zip(edges[:-1], edges[1:])])
    np.testing.assert_allclose(
        times, [(i + j - 1) * 0.05 for i, j in zip(edges[:-1], edges[1:])], atol=1e-15
    )
    np.testing.assert_allclose(values, (f - b) / (f + b))
    np.testing.assert_allclose(errors, 2 * np.sqrt(b * b * f + f * f * b) / (f + b) ** 2)


@pytest.mark.parametrize("mode", ["variable", "constant_error"])
@pytest.mark.parametrize("t0", [0.5, 4.5, 4.3])
@pytest.mark.parametrize("first,last", [(0, 18), (3, 16)])
def test_shifted_windows_use_exact_left_edges(mode, t0, first, last):
    width = 0.25
    counts = aligned_counts(
        np.arange(1.0, 21),
        np.arange(21.0, 41),
        common_t0=4,
        bin_width_us=width,
        t0_time_us=t0 * width,
        first_good_bin=first,
        last_good_bin=last,
    )
    spec = BinningSpec(mode=mode, bin0_us=0.3, bin10_us=0.8)
    # Enumerate admissible integer widths and minimise absolute width error.
    expected = [first]
    while expected[-1] <= last:
        left = (expected[-1] - t0) * width
        target = 0.3 * (
            (0.8 / 0.3) ** (max(left, 0) / 10)
            if mode == "variable"
            else np.exp(max(left, 0) / MUON_LIFETIME_US)
        )
        step = min(range(1, 100), key=lambda k: (abs(k * width - target), -k))
        expected.append(min(expected[-1] + step, last + 1))
    edges, _ = fb_binning_edges(counts, spec)
    np.testing.assert_array_equal(edges, expected)
    times, _, _ = binned_fb_asymmetry(counts, binning=spec, alpha=1)
    native = (np.arange(20) + 0.5 - t0) * width
    np.testing.assert_allclose(
        times, [native[i:j].mean() for i, j in zip(expected[:-1], expected[1:])], atol=1e-15
    )
    filtered, _ = fb_binning_edges(counts, spec, time_range=(times[1], times[-2]))
    np.testing.assert_array_equal(filtered, edges[1:-1])
    np.testing.assert_array_equal(counts.forward.counts, np.arange(1.0, 21))


@pytest.mark.parametrize(
    "spec,edges",
    [
        (BinningSpec(factor=3), [0, 3, 6, 9]),
        (BinningSpec(factor=3, merge=2), [0, 6]),
        (BinningSpec(mode="variable", bin0_us=0.3, bin10_us=0.3), [0, 3, 6, 9, 11]),
        (BinningSpec(mode="variable", bin0_us=0.3, bin10_us=0.3, merge=2), [0, 6, 11]),
        (BinningSpec(mode="variable", bin0_us=0.3, bin10_us=0.3, merge=3), [0, 9]),
    ],
)
def test_base_tail_and_additional_merge_support_are_distinct(spec, edges):
    np.testing.assert_array_equal(spec.edges(11, 0.1, 0.0), edges)
    np.testing.assert_array_equal(replace(spec, merge=1).edges(12, 0.1, 0.0), [0, 3, 6, 9, 12])


@pytest.mark.parametrize("mode", ["variable", "constant_error"])
def test_signed_supplied_variance_on_actual_membership(mode):
    counts = aligned_counts(
        [-5, 10, 20, 30, 40],
        [8, 10, 15, 20, 25],
        forward_error=np.sqrt([100, 10, 21, 31, 41]),
        backward_error=np.sqrt([8, 11, 16, 21, 26]),
        bin_width_us=1,
    )
    spec = BinningSpec(mode=mode, bin0_us=1.2, bin10_us=1.2, merge=2)
    # Restrict to pre-t0 times so both laws select single native bins.
    counts = replace(counts, t0_time_us=10.5, window=BinWindow(0, 4))
    time, a, error = binned_fb_asymmetry(counts, binning=spec, alpha=1.2, beta=0.8)
    f, b = np.array([5, 50]), np.array([18, 35])
    vf, vb = np.array([110, 52]), np.array([19, 37])
    den = 0.8 * f + 1.2 * b
    np.testing.assert_allclose(a, (f - 1.2 * b) / den)
    np.testing.assert_allclose(error, 1.2 * 1.8 * np.sqrt(b * b * vf + f * f * vb) / den**2)
    np.testing.assert_allclose(time, [-9.5, -7.5])


@pytest.mark.parametrize("mode", ["variable", "constant_error"])
@pytest.mark.parametrize("merge", [1, 2])
def test_background_overlap_uses_actual_contributing_observations(mode, merge):
    from asymmetry.core.data.dataset import Histogram
    from asymmetry.core.transform.reduce import corrected_grouped_counts

    raw = np.array([[8, 8, 100, 100, 100, 100], [8, 8, 80, 80, 80, 80]], dtype=float)
    counts = corrected_grouped_counts(
        histograms=[Histogram(y, bin_width=0.01, t0_bin=2) for y in raw],
        grouping={
            "background_mode": "range",
            "background_range": [1, 2],
            "first_good_bin": 2,
            "last_good_bin": 5,
        },
        forward_idx=[0],
        backward_idx=[1],
        use_deadtime=False,
        deadtime_mode="off",
        use_background=True,
    )
    spec = BinningSpec(mode=mode, bin0_us=0.012, bin10_us=0.012, merge=merge)
    edges = list(range(2, 7, merge))
    levels = raw[:, 1:3].mean(axis=1)
    sums, variances = [], []
    for i, j in zip(edges[:-1], edges[1:]):
        coefficients = np.zeros(6)
        coefficients[i:j] = 1
        coefficients[1:3] -= (j - i) / 2
        sums.append(raw[:, i:j].sum(axis=1) - (j - i) * levels)
        variances.append((raw * coefficients**2).sum(axis=1))
    f, b = np.array(sums).T
    vf, vb = np.array(variances).T
    time, a, error = binned_fb_asymmetry(counts, binning=spec, alpha=1)
    np.testing.assert_allclose(a, (f - b) / (f + b))
    np.testing.assert_allclose(error, 2 * np.sqrt(b * b * vf + f * f * vb) / (f + b) ** 2)
    np.testing.assert_allclose(
        time, [(i + j) * 0.005 - 0.025 for i, j in zip(edges[:-1], edges[1:])], atol=1e-15
    )


@pytest.mark.parametrize("mode", ["variable", "constant_error"])
def test_target_laws_keep_literal_controls_and_pre_t0_clamp(mode):
    from asymmetry.core.transform.rebin import _target_width_us

    assert _target_width_us(mode, -3, 0.12, 0.25) == 0.12
    assert _target_width_us(mode, 0, 0.12, 0.25) == 0.12
    if mode == "variable":
        assert _target_width_us(mode, 10, 0.12, 0.25) == pytest.approx(0.25, rel=1e-15)
    else:
        assert _target_width_us(mode, MUON_LIFETIME_US, 0.12, 0.25) == pytest.approx(
            0.12 * np.e, rel=1e-15
        )


def test_global_screening_cannot_reuse_a_pre_policy_single_fit_signature():
    import json

    from asymmetry.core.fitting.fit_wizard import single_fit_build_signature
    from asymmetry.core.fitting.wizard_scope import WizardScope
    from asymmetry.core.transform.rebin import BINNING_POLICY

    signature = json.loads(single_fit_build_signature(WizardScope()))
    assert signature.pop("binning_policy", None) == BINNING_POLICY
    assert json.dumps(signature, sort_keys=True) != single_fit_build_signature(WizardScope())


@pytest.mark.parametrize("mode", ["variable", "constant_error"])
@pytest.mark.parametrize(
    "target,step", [(np.nextafter(0.25, 0), 2), (0.25, 3), (np.nextafter(0.25, np.inf), 3)]
)
def test_decimal_native_grid_half_width_boundary(mode, target, step):
    spec = BinningSpec(mode=mode, bin0_us=target, bin10_us=target)
    assert spec.edges(10, 0.1, -10)[1] == step
