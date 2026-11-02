"""Counts-first binning through the public reduction entry points."""

from dataclasses import replace

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.transform.rebin import BinningSpec, binning_spec
from asymmetry.core.transform.reduce import reduce_grouped_asymmetry, reduce_period_asymmetry
from asymmetry.core.workflow.reduction import ReductionSettings, reduce_run


def _run(forward, backward, **options):
    histograms = [
        Histogram(np.asarray(y, dtype=float), 0.1, good_bin_end=len(y) - 1)
        for y in (forward, backward)
    ]
    grouping = {
        "groups": {1: [1], 2: [2]},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.0,
        "first_good_bin": 0,
        "last_good_bin": len(forward) - 1,
        "bin_index_base": 0,
        **options,
    }
    return Run(run_number=1, histograms=histograms, grouping=grouping)


def _direct(run, spec):
    return reduce_grouped_asymmetry(
        histograms=run.histograms,
        grouping=run.grouping,
        forward_idx=[0],
        backward_idx=[1],
        alpha=1.0,
        use_deadtime=False,
        deadtime_mode="off",
        use_background=False,
        binning=spec,
    )


def test_counts_first_anchor_all_entry_points():
    run = _run([90, 1], [10, 9])
    spec = BinningSpec(factor=2)
    direct = _direct(run, spec)
    workflow = reduce_run(run, ReductionSettings(rebin=2), allow_aliasing=True)
    gui = reduce_period_asymmetry(
        run.histograms,
        run.grouping,
        "red",
        binning=spec,
        forward_idx=[0],
        backward_idx=[1],
        alpha=1.0,
        use_deadtime=False,
        deadtime_mode="off",
        use_background=False,
    )
    for result in (direct, workflow, gui):
        np.testing.assert_allclose(result.time, [0.05], rtol=0, atol=1e-15)
        np.testing.assert_allclose(result.asymmetry, [100 * 72 / 110], rtol=1e-12)
        np.testing.assert_allclose(
            result.error, [200 * np.sqrt(19**2 * 91 + 91**2 * 19) / 110**2], rtol=1e-12
        )


@pytest.mark.parametrize(
    "key,value",
    [
        ("binning_mode", "Varaible"),
        *[("bunching_factor", v) for v in ("abc", 0, -3, 2.7, True, None)],
        ("bin0_us", -1),
        ("bin10_us", "x"),
        ("bin10_us", np.nan),
    ],
)
def test_binning_spec_rejects_present_invalid_values(key, value):
    with pytest.raises(ValueError, match="binning mode|bunching factor|bin0_us|bin10_us"):
        binning_spec({key: value})


def test_absent_binning_keys_have_explicit_defaults():
    assert binning_spec({}) == BinningSpec("fixed", 1, 0.08, 0.25)


@pytest.mark.parametrize("mode", ["fixed", "variable", "constant_error"])
def test_merge_drops_only_incomplete_output_groups(mode):
    spec = BinningSpec(mode=mode, factor=3, bin0_us=0.25, bin10_us=0.25)
    edges = spec.edges(11, 0.1, 0)
    np.testing.assert_array_equal(replace(spec, merge=2).edges(11, 0.1, 0), edges[::2])
    if mode == "variable":
        assert edges[-1] == 11


def test_fixed_merge_equals_product_factor():
    run = _run(np.arange(1, 101), np.arange(101, 201))
    a = _direct(run, BinningSpec(factor=4, merge=2))
    b = _direct(run, BinningSpec(factor=8))
    for key in ("time", "asymmetry", "error"):
        np.testing.assert_array_equal(getattr(a, key), getattr(b, key))


def test_factor_one_is_bit_identical():
    run = _run([90, 1, 15], [10, 9, 20])
    direct = _direct(run, BinningSpec())
    workflow = reduce_run(run, ReductionSettings(), allow_aliasing=True)
    for key in ("time", "asymmetry", "error"):
        np.testing.assert_array_equal(getattr(direct, key), getattr(workflow, key))
    assert direct.binning == BinningSpec()


def test_sparse_tail_matches_hand_sums():
    rng = np.random.default_rng(20261101)
    f, b = rng.poisson(2.4, 11), rng.poisson(1.6, 11)
    run = _run(f, b)
    result = reduce_run(run, ReductionSettings(rebin=3), allow_aliasing=True)
    forward = f[:9].reshape(3, 3).sum(axis=1)
    backward = b[:9].reshape(3, 3).sum(axis=1)
    total = forward + backward
    expected = 100 * (forward - backward) / total
    error = 200 * np.sqrt(backward**2 * forward + forward**2 * backward) / total**2
    np.testing.assert_allclose(result.asymmetry, expected, rtol=1e-12)
    np.testing.assert_allclose(result.error, error, rtol=1e-12)
    np.testing.assert_allclose(result.time, [0.1, 0.4, 0.7], atol=1e-15)


def test_counts_first_seeded_simulation():
    from asymmetry.core.transform.rebin import binned_fb_asymmetry
    from tests.counts import aligned_counts

    rng = np.random.default_rng(20261101)
    f, b = rng.poisson(2.4, (4000, 64)), rng.poisson(1.6, (4000, 64))
    _, a, error = binned_fb_asymmetry(
        aligned_counts(f.ravel(), b.ravel()), binning=BinningSpec(factor=8), alpha=1.0
    )
    empirical = a.std(ddof=1)
    assert abs(a.mean() - 0.2) < 3 * empirical / np.sqrt(a.size)
    assert error.mean() == pytest.approx(empirical, rel=0.05)


@pytest.mark.parametrize("mode,sign", [("green_minus_red", -1), ("green_plus_red", 1)])
def test_period_combination_is_counts_first(mode, sign):
    red = _run([90, 1, 30, 2], [10, 9, 20, 8])
    green = _run([2, 80, 10, 2], [8, 10, 30, 8])
    run = _run(
        [1] * 4,
        [1] * 4,
        period_histograms=[red.histograms, green.histograms],
        period_reduced=[{}, {}],
    )
    spec = BinningSpec(merge=2)
    result = reduce_period_asymmetry(
        run.histograms,
        run.grouping,
        mode,
        binning=spec,
        forward_idx=[0],
        backward_idx=[1],
        alpha=1.0,
        use_deadtime=False,
        deadtime_mode="off",
        use_background=False,
    )
    r, g = _direct(red, spec), _direct(green, spec)
    np.testing.assert_array_equal(result.asymmetry, g.asymmetry + sign * r.asymmetry)
    np.testing.assert_array_equal(result.error, np.sqrt(g.error**2 + r.error**2))
    assert result.binning == spec
    assert set(result.corrections) == {"red", "green"}
    if mode == "green_minus_red":
        workflow = reduce_run(run, ReductionSettings(period=mode, rebin=2), allow_aliasing=True)
        for key in ("time", "asymmetry", "error"):
            np.testing.assert_array_equal(getattr(result, key), getattr(workflow, key))


@pytest.mark.parametrize("mismatch", ["length", "time"])
def test_period_grid_mismatch_raises(mismatch):
    from asymmetry.core.io.periods import combine_period_asymmetry

    t = np.arange(4.0)
    other = t[:-1] if mismatch == "length" else t + 0.01
    with pytest.raises(ValueError, match="Period asymmetry.*identical"):
        combine_period_asymmetry(t, t, t, other, other, other, "green_minus_red")


def test_shared_background_through_workflow():
    run = _run(
        [8.0, 8, 100, 100, 100, 100], [8.0, 8, 80, 80, 80, 80], first_good_bin=2, last_good_bin=5
    )
    run.histograms = [replace(hist, t0_bin=2, good_bin_start=2) for hist in run.histograms]
    run.metadata = {"instrument": "GPS", "facility": "PSI"}
    settings = ReductionSettings(background="range", background_range=(0, 1), rebin=4)
    result = reduce_run(run, settings, allow_aliasing=True)
    assert result.error[0] == pytest.approx(100 * 0.04420992805650498, rel=1e-12)


@pytest.mark.parametrize("factor", [2.5, True, 0, -1])
def test_workflow_rejects_invalid_rebin(factor):
    with pytest.raises(ValueError, match="integer"):
        ReductionSettings(rebin=factor)


def test_t0_alignment_uses_the_good_window_before_binning():
    f, b = np.arange(1.0, 201.0), np.arange(201.0, 401.0)
    run = _run(f, b, first_good_bin=120, last_good_bin=130)
    run.histograms = [
        replace(run.histograms[0], t0_bin=100),
        replace(run.histograms[1], t0_bin=110),
    ]
    result = _direct(run, BinningSpec(factor=3))
    forward = f[110:119].reshape(3, 3).sum(axis=1)
    backward = b[120:129].reshape(3, 3).sum(axis=1)
    np.testing.assert_allclose(
        result.asymmetry, 100 * (forward - backward) / (forward + backward), rtol=1e-12
    )
    np.testing.assert_allclose(result.time, [1.1, 1.4, 1.7], rtol=0, atol=1e-14)


def test_count_domain_group_rejects_invalid_saved_binning():
    from asymmetry.core.data.dataset import MuonDataset
    from asymmetry.core.fitting.grouped_time_domain import build_count_group

    run = _run([90, 1], [10, 9], bunching_factor="abc")
    dataset = MuonDataset(time=np.arange(2) * 0.1, asymmetry=np.zeros(2), error=np.ones(2), run=run)
    with pytest.raises(ValueError, match="bunching factor must be an integer"):
        build_count_group(dataset, 1)


@pytest.mark.parametrize("mode", ["variable", "constant_error"])
def test_nonfixed_merge_includes_a_partial_raw_tail(mode):
    run = _run(
        np.arange(1.0, 12.0), np.arange(21.0, 32.0), binning_mode=mode, bin0_us=0.25, bin10_us=0.25
    )
    direct = _direct(run, replace(binning_spec(run.grouping), merge=2))
    workflow = reduce_run(run, ReductionSettings(rebin=2), allow_aliasing=True)
    # Both rules give raw edges [0, 3, 6, 10 or 9, 11]; merging pairs gives [0, 6, 11].
    f = np.array([sum(range(1, 7)), sum(range(7, 12))], dtype=float)
    b = np.array([sum(range(21, 27)), sum(range(27, 32))], dtype=float)
    expected_a = 100 * (f - b) / (f + b)
    expected_error = 200 * np.sqrt(b * b * f + f * f * b) / (f + b) ** 2
    for result in (direct, workflow):
        np.testing.assert_allclose(result.time, [0.25, 0.8], rtol=0, atol=1e-15)
        np.testing.assert_allclose(result.asymmetry, expected_a, rtol=1e-12)
        np.testing.assert_allclose(result.error, expected_error, rtol=1e-12)
