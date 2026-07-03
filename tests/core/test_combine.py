"""Tests for histogram-level run arithmetic (asymmetry.core.data.combine).

Verification-plan of docs/porting/run-arithmetic/verification-plan.md: co-add
distributional identity (pull test), co-subtract zero / √2 errors, two-period
co-add, the negative-count guard, event-weighted metadata, and the F9
chokepoint reuse.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from asymmetry.core.data.combine import (
    CombineError,
    coadd_member_windows,
    combine_runs,
    reduce_combined_run,
    runs_with_dataset_metadata,
)
from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.simulate import (
    PeriodSpec,
    reduce_run_to_dataset,
    simulate_run,
    simulate_two_period_run,
)
from asymmetry.core.utils.constants import PeriodMode

N_BINS = 600
BIN_WIDTH = 0.016
T0_BIN = 20


def _template(*, t0_bins: list[int] | None = None, good_frames: float = 1000.0) -> Run:
    """A 1+1 forward/backward instrument template (empty counts)."""
    if t0_bins is None:
        t0_bins = [T0_BIN, T0_BIN]
    histograms = [
        Histogram(
            counts=np.zeros(N_BINS),
            bin_width=BIN_WIDTH,
            t0_bin=t0,
            good_bin_start=max(t0_bins) + 5,
            good_bin_end=N_BINS - 10,
        )
        for t0 in t0_bins
    ]
    grouping = {
        "groups": {1: [1], 2: [2]},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.0,
        "t0_bin": max(t0_bins),
        "first_good_bin": max(t0_bins) + 5,
        "last_good_bin": N_BINS - 10,
        "good_frames": good_frames,
    }
    return Run(
        run_number=1000,
        histograms=histograms,
        metadata={"title": "Tmpl", "temperature": 5.0, "field": 100.0, "instrument": "EMU"},
        grouping=grouping,
    )


def _cos(t, A=20.0, freq=2.0, sigma=0.2):  # noqa: N803
    return A * np.cos(2.0 * np.pi * freq * t) * np.exp(-0.5 * (sigma * t) ** 2)


# ---------------------------------------------------------------------------
# Compatibility / validation
# ---------------------------------------------------------------------------


def test_requires_two_runs():
    run = simulate_run(_template(), _cos, total_events=1e5, seed=0)
    with pytest.raises(CombineError, match="at least two"):
        combine_runs([run])


def test_rejects_detector_count_mismatch():
    a = simulate_run(_template(), _cos, total_events=1e5, seed=0)
    # A 4-detector template -> different detector count.
    big = _template()
    big.histograms.extend(copy.deepcopy(big.histograms))
    b = simulate_run(big, _cos, total_events=1e5, seed=1)
    with pytest.raises(CombineError, match="detector counts"):
        combine_runs([a, b])


def test_rejects_bin_width_mismatch():
    a = simulate_run(_template(), _cos, total_events=1e5, seed=0)
    b = simulate_run(_template(), _cos, total_events=1e5, seed=1)
    b.histograms[0].bin_width = BIN_WIDTH * 2.0
    with pytest.raises(CombineError, match="bin widths differ"):
        combine_runs([a, b])


def test_reference_subtract_requires_exactly_two():
    runs = [simulate_run(_template(), _cos, total_events=1e5, seed=i) for i in range(3)]
    with pytest.raises(CombineError, match="exactly two"):
        combine_runs(runs, sign=-1)  # default subtract_method="reference"


def test_invalid_subtract_method_rejected():
    runs = [simulate_run(_template(), _cos, total_events=1e5, seed=i) for i in range(2)]
    with pytest.raises(CombineError, match="subtract_method"):
        combine_runs(runs, sign=-1, subtract_method="bogus")


# ---------------------------------------------------------------------------
# Co-add
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Dataset-scalar metadata merge (display/from-log overrides)
# ---------------------------------------------------------------------------


def test_runs_with_dataset_metadata_merges_display_scalars():
    """A dataset's overridden temperature wins over the run's loader value."""
    run = simulate_run(_template(), _cos, total_events=1e5, seed=0)
    # run.metadata["temperature"] == 5.0 (the template); dataset overrides it.
    dataset = MuonDataset(
        time=np.zeros(1),
        asymmetry=np.zeros(1),
        error=np.ones(1),
        metadata={"temperature": 42.0, "field": 250.0},
        run=run,
    )
    merged = runs_with_dataset_metadata([dataset])
    assert len(merged) == 1
    assert merged[0].metadata["temperature"] == 42.0
    assert merged[0].metadata["field"] == 250.0
    # Shares histograms (combine never mutates them) and leaves the source intact.
    assert merged[0].histograms is run.histograms
    assert run.metadata["temperature"] == 5.0


def test_runs_with_dataset_metadata_skips_runless_datasets():
    runless = MuonDataset(
        time=np.zeros(1), asymmetry=np.zeros(1), error=np.ones(1), metadata={}, run=None
    )
    assert runs_with_dataset_metadata([runless]) == []


def test_coadd_event_weights_use_dataset_override_via_helper():
    """Co-add through the helper event-weights on the overridden temperature."""
    run_a = simulate_run(_template(good_frames=1000.0), _cos, total_events=1e5, seed=1)
    run_b = simulate_run(_template(good_frames=1000.0), _cos, total_events=1e5, seed=2)
    ds_a = MuonDataset(np.zeros(1), np.zeros(1), np.ones(1), {"temperature": 10.0}, run=run_a)
    ds_b = MuonDataset(np.zeros(1), np.zeros(1), np.ones(1), {"temperature": 20.0}, run=run_b)
    combined = combine_runs(runs_with_dataset_metadata([ds_a, ds_b]), sign=1)
    # Equal good frames -> mean of the overridden 10 and 20 = 15 (not the
    # template's loader value of 5.0 on both).
    assert combined.metadata["temperature"] == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# In-batch co-add windowing (WiMDA BatchFit Smooth/Bin)
# ---------------------------------------------------------------------------


def test_coadd_windows_off_returns_singletons():
    assert coadd_member_windows(4, mode="off", window=2) == [[0], [1], [2], [3]]
    # window <= 1 is a no-op even in a co-add mode.
    assert coadd_member_windows(3, mode="bin", window=1) == [[0], [1], [2]]


def test_coadd_windows_bin_steps_by_window_dropping_partial():
    # Bin over 2N runs (W=2) -> N non-overlapping pairs; matches WiMDA's
    # i := i + jump + 1 with the "until i + jump > nff" full-window guard.
    assert coadd_member_windows(6, mode="bin", window=2) == [[0, 1], [2, 3], [4, 5]]
    # A trailing partial window is dropped (run index 4 here).
    assert coadd_member_windows(5, mode="bin", window=2) == [[0, 1], [2, 3]]
    assert coadd_member_windows(7, mode="bin", window=3) == [[0, 1, 2], [3, 4, 5]]


def test_coadd_windows_smooth_slides_by_one():
    # Smooth steps by one run (inc(i)); yields n - W + 1 full windows.
    assert coadd_member_windows(5, mode="smooth", window=3) == [
        [0, 1, 2],
        [1, 2, 3],
        [2, 3, 4],
    ]


def test_coadd_windows_empty_when_window_exceeds_members():
    assert coadd_member_windows(2, mode="bin", window=3) == []
    assert coadd_member_windows(2, mode="smooth", window=5) == []


def test_bin_coadd_over_2n_equals_pairwise_combines_exactly():
    """Bin-mode batch over 2N runs == N fits of pairwise-combined runs (exact).

    The grouped-series in-batch co-add partitions members with
    :func:`coadd_member_windows` then co-adds each window via
    :func:`combine_runs`. For Bin W=2 the windows are consecutive pairs, so each
    combined member must be bit-for-bit the pairwise co-add of its two runs.
    """
    n_pairs = 3
    runs = [
        simulate_run(_template(), _cos, total_events=2e5, seed=10 + i) for i in range(2 * n_pairs)
    ]
    windows = coadd_member_windows(len(runs), mode="bin", window=2)
    assert windows == [[0, 1], [2, 3], [4, 5]]
    for window in windows:
        windowed = combine_runs([runs[i] for i in window], sign=1)
        pairwise = combine_runs([runs[window[0]], runs[window[1]]], sign=1)
        for det in range(len(windowed.histograms)):
            np.testing.assert_array_equal(
                windowed.histograms[det].counts,
                pairwise.histograms[det].counts,
            )
            # And exactly the per-bin sum of the two source histograms.
            np.testing.assert_array_equal(
                windowed.histograms[det].counts,
                runs[window[0]].histograms[det].counts + runs[window[1]].histograms[det].counts,
            )


def test_coadd_sums_counts_bin_for_bin():
    a = simulate_run(_template(), _cos, total_events=2e5, seed=1)
    b = simulate_run(_template(), _cos, total_events=2e5, seed=2)
    c = combine_runs([a, b], sign=1, run_number=-1)
    assert len(c.histograms) == 2
    for det in range(2):
        np.testing.assert_array_equal(
            c.histograms[det].counts,
            a.histograms[det].counts + b.histograms[det].counts,
        )


def test_coadd_accumulates_good_frames():
    a = simulate_run(_template(good_frames=1000.0), _cos, total_events=2e5, seed=1)
    b = simulate_run(_template(good_frames=3000.0), _cos, total_events=2e5, seed=2)
    c = combine_runs([a, b], sign=1)
    # The deadtime normaliser reads grouping["good_frames"]; it must be summed.
    assert c.grouping["good_frames"] == pytest.approx(4000.0)
    assert c.metadata["good_frames"] == pytest.approx(4000.0)


def test_coadd_distributionally_identical_to_one_long_run():
    """Pull test: N co-added runs ≡ one N×-events run (verification headline)."""
    template = _template()
    n_runs = 6
    events_each = 4e5
    parts = [
        simulate_run(template, _cos, total_events=events_each, seed=100 + i) for i in range(n_runs)
    ]
    combined = reduce_combined_run(combine_runs(parts, sign=1, run_number=-1))
    single = reduce_run_to_dataset(
        simulate_run(template, _cos, total_events=n_runs * events_each, seed=999)
    )
    n = min(combined.n_points, single.n_points)
    pulls = (combined.asymmetry[:n] - single.asymmetry[:n]) / np.sqrt(
        combined.error[:n] ** 2 + single.error[:n] ** 2
    )
    # Distributionally standard normal: mean ~ 0, std ~ 1.
    assert abs(float(np.mean(pulls))) < 0.15
    assert 0.8 < float(np.std(pulls)) < 1.2


def test_coadd_metadata_event_weighted_with_spread():
    a = _template(good_frames=1000.0)
    a.metadata["temperature"] = 10.0
    a.metadata["field"] = 100.0
    b = _template(good_frames=3000.0)
    b.metadata["temperature"] = 20.0
    b.metadata["field"] = 100.0
    ra = simulate_run(a, _cos, total_events=1e5, seed=1)
    rb = simulate_run(b, _cos, total_events=1e5, seed=2)
    c = combine_runs([ra, rb], sign=1)
    # Event-weighted by good frames: (10*1000 + 20*3000) / 4000 = 17.5.
    assert c.metadata["temperature"] == pytest.approx(17.5)
    assert c.metadata["temperature_spread"] == (10.0, 20.0)
    # Equal field -> no spread recorded only when values differ; equal here.
    assert c.metadata["field"] == pytest.approx(100.0)
    assert c.metadata["combined_from"] == [ra.run_number, rb.run_number]
    assert c.metadata["combination"]["method"] == "coadd"


def test_coadd_aligns_per_detector_t0():
    """Detectors with differing t0 are shifted to a common bin before summing."""
    a = simulate_run(_template(t0_bins=[20, 20]), _cos, total_events=2e5, seed=1)
    b = simulate_run(_template(t0_bins=[24, 24]), _cos, total_events=2e5, seed=2)
    c = combine_runs([a, b], sign=1)
    # Common t0 is the max (24); the combined detector adopts it.
    assert c.histograms[0].t0_bin == 24


# ---------------------------------------------------------------------------
# Co-subtract
# ---------------------------------------------------------------------------


def test_subtract_identical_runs_is_zero():
    """Same seed -> identical counts -> exact zero difference."""
    a = simulate_run(_template(), _cos, total_events=2e5, seed=7)
    b = simulate_run(_template(), _cos, total_events=2e5, seed=7)
    d = combine_runs([a, b], sign=-1, scales=[1.0, 1.0])
    for det in range(2):
        np.testing.assert_array_equal(d.histograms[det].counts, np.zeros(N_BINS))
    assert d.metadata["combination"]["method"] == "subtract_reference"
    assert d.metadata["combination"]["sign"] == -1


def test_subtract_variances_add():
    """Independent draws of equal expected counts: variance ≈ sample + reference."""
    a = simulate_run(_template(), _cos, total_events=3e5, seed=11)
    b = simulate_run(_template(), _cos, total_events=3e5, seed=22)
    d = combine_runs([a, b], sign=-1, scales=[1.0, 1.0])
    var = d.metadata["combination"]["detector_variance"][0]
    expected = a.histograms[0].counts + b.histograms[0].counts
    # Empty pre-t0 bins carry the chokepoint's 1.0 error sentinel (sliced out of
    # the good window); compare the populated region where variance = a + r.
    populated = expected > 0
    np.testing.assert_allclose(var[populated], expected[populated], rtol=1e-12)


def test_subtract_scaled_reference():
    a = simulate_run(_template(), _cos, total_events=3e5, seed=11)
    b = simulate_run(_template(), _cos, total_events=3e5, seed=22)
    scale = 0.5
    d = combine_runs([a, b], sign=-1, scales=[1.0, scale])
    np.testing.assert_allclose(
        d.histograms[0].counts,
        a.histograms[0].counts - scale * b.histograms[0].counts,
        rtol=1e-12,
    )
    var = d.metadata["combination"]["detector_variance"][0]
    expected_var = a.histograms[0].counts + scale * scale * b.histograms[0].counts
    populated = expected_var > 0
    np.testing.assert_allclose(var[populated], expected_var[populated], rtol=1e-12)


def test_subtract_negative_count_guard():
    """A reference exceeding the sample drives bins negative; the guard counts them."""
    a = simulate_run(_template(), _cos, total_events=1e5, seed=1)
    b = simulate_run(_template(), _cos, total_events=1e5, seed=2)
    d = combine_runs([a, b], sign=-1, scales=[1.0, 3.0])  # over-subtract
    assert d.metadata["combination"]["negative_count_bins"] > 0
    ds = reduce_combined_run(d)
    assert np.all(np.isfinite(ds.error))


def test_subtract_reduces_with_propagated_errors():
    a = simulate_run(_template(), _cos, total_events=3e5, seed=11)
    b = simulate_run(_template(), _cos, total_events=3e5, seed=22)
    d = combine_runs([a, b], sign=-1, scales=[1.0, 1.0])
    ds = reduce_combined_run(d)
    assert ds.n_points > 0
    assert np.all(np.isfinite(ds.asymmetry))
    # Difference errors are larger than a single run's Poisson errors.
    single = reduce_run_to_dataset(a)
    n = min(ds.n_points, single.n_points)
    assert float(np.median(ds.error[:n])) > float(np.median(single.error[:n]))


# ---------------------------------------------------------------------------
# Symmetric N-run signed co-subtract
# ---------------------------------------------------------------------------


def test_signed_subtract_two_identical_runs_is_zero_with_propagated_errors():
    """A − A = 0 with variance = 2·A (every term contributes its own Poisson)."""
    a = simulate_run(_template(), _cos, total_events=2e5, seed=7)
    b = simulate_run(_template(), _cos, total_events=2e5, seed=7)
    d = combine_runs([a, b], sign=-1, subtract_method="signed")
    assert d.metadata["combination"]["method"] == "subtract_signed"
    assert "reference_run_number" not in d.metadata["combination"]
    for det in range(2):
        np.testing.assert_array_equal(d.histograms[det].counts, np.zeros(N_BINS))
    var = d.metadata["combination"]["detector_variance"][0]
    expected = a.histograms[0].counts + b.histograms[0].counts  # = 2·A
    populated = expected > 0
    np.testing.assert_allclose(var[populated], expected[populated], rtol=1e-12)


def test_signed_subtract_three_runs_difference_and_variance():
    """runs[0] − runs[1] − runs[2]; variance = c0 + c1 + c2 (all add)."""
    a = simulate_run(_template(), _cos, total_events=3e5, seed=11)
    b = simulate_run(_template(), _cos, total_events=3e5, seed=22)
    c = simulate_run(_template(), _cos, total_events=3e5, seed=33)
    d = combine_runs([a, b, c], sign=-1, subtract_method="signed")
    for det in range(2):
        np.testing.assert_allclose(
            d.histograms[det].counts,
            a.histograms[det].counts - b.histograms[det].counts - c.histograms[det].counts,
            rtol=1e-12,
        )
    var = d.metadata["combination"]["detector_variance"][0]
    expected_var = a.histograms[0].counts + b.histograms[0].counts + c.histograms[0].counts
    populated = expected_var > 0
    np.testing.assert_allclose(var[populated], expected_var[populated], rtol=1e-12)
    assert d.metadata["combined_from"] == [1000, 1000, 1000]
    assert d.metadata["combination"]["scales"] == [1.0, 1.0, 1.0]


def test_two_run_reference_matches_chokepoint_all_bins():
    """The 2-run reference path equals a direct subtract_scaled_counts, all bins.

    Guards the refactor that unified the subtract loop: the sample-plus-one-
    reference case must still produce the chokepoint's difference (exact) and
    variance (the refactor stores the variance directly rather than the old
    ``error*error`` √-round-trip, so they agree to floating-point), including the
    empty-bin 1.0 sentinel — checked on *all* bins, not just populated ones.
    """
    from asymmetry.core.transform.background import subtract_scaled_counts

    a = simulate_run(_template(), _cos, total_events=3e5, seed=11)
    b = simulate_run(_template(), _cos, total_events=3e5, seed=22)
    scale = 0.7
    d = combine_runs([a, b], sign=-1, scales=[1.0, scale])
    for det in range(2):
        diff, error = subtract_scaled_counts(
            a.histograms[det].counts, b.histograms[det].counts, scale
        )
        np.testing.assert_array_equal(d.histograms[det].counts, diff)
        np.testing.assert_allclose(
            d.metadata["combination"]["detector_variance"][det], error * error, rtol=1e-12
        )


def test_signed_subtract_extra_reference_empty_bin_no_sentinel():
    """An empty reference bin adds 0 variance, not the chokepoint's 1.0 sentinel.

    For runs[k>=2] the variance is taken as scaleₖ²·counts directly; reading it
    off ``subtract_scaled_counts``'s error would fold in the empty-bin 1.0
    sentinel for every extra reference, over-stating the variance where the
    reference is zero.
    """
    a = simulate_run(_template(), _cos, total_events=3e5, seed=11)
    b = simulate_run(_template(), _cos, total_events=3e5, seed=22)
    c = simulate_run(_template(), _cos, total_events=3e5, seed=33)
    # Zero out a populated bin in the *third* run (the extra reference, k=2).
    zero_bin = int(N_BINS // 2)
    c.histograms[0].counts[zero_bin] = 0.0
    d = combine_runs([a, b, c], sign=-1, subtract_method="signed")
    var = d.metadata["combination"]["detector_variance"][0]
    # variance at the zeroed bin = a + b + 0 (c contributes 0, no 1.0 sentinel).
    expected = a.histograms[0].counts[zero_bin] + b.histograms[0].counts[zero_bin]
    assert var[zero_bin] == pytest.approx(expected, rel=1e-12)


def test_signed_subtract_respects_per_run_scales():
    a = simulate_run(_template(), _cos, total_events=3e5, seed=11)
    b = simulate_run(_template(), _cos, total_events=3e5, seed=22)
    c = simulate_run(_template(), _cos, total_events=3e5, seed=33)
    d = combine_runs([a, b, c], sign=-1, subtract_method="signed", scales=[1.0, 0.5, 0.25])
    np.testing.assert_allclose(
        d.histograms[0].counts,
        a.histograms[0].counts - 0.5 * b.histograms[0].counts - 0.25 * c.histograms[0].counts,
        rtol=1e-12,
    )
    var = d.metadata["combination"]["detector_variance"][0]
    expected_var = (
        a.histograms[0].counts + 0.25 * b.histograms[0].counts + 0.0625 * c.histograms[0].counts
    )
    populated = expected_var > 0
    np.testing.assert_allclose(var[populated], expected_var[populated], rtol=1e-12)


def test_signed_subtract_reduces_with_finite_errors():
    a = simulate_run(_template(), _cos, total_events=3e5, seed=11)
    b = simulate_run(_template(), _cos, total_events=3e5, seed=22)
    c = simulate_run(_template(), _cos, total_events=3e5, seed=33)
    ds = reduce_combined_run(combine_runs([a, b, c], sign=-1, subtract_method="signed"))
    assert ds.n_points > 0
    assert np.all(np.isfinite(ds.asymmetry))
    assert np.all(np.isfinite(ds.error))


# ---------------------------------------------------------------------------
# Two-period co-add (W12)
# ---------------------------------------------------------------------------


def test_two_period_coadd_sums_period_histograms():
    template = _template()
    periods = [
        PeriodSpec(model=_cos, label="red"),
        PeriodSpec(model=_cos, parameters={"A": 0.0}, scale=0.0, label="green"),
    ]
    a = simulate_two_period_run(
        template, periods, total_events=4e5, seed=1, period_mode=PeriodMode.GREEN_MINUS_RED
    )
    b = simulate_two_period_run(
        template, periods, total_events=4e5, seed=2, period_mode=PeriodMode.GREEN_MINUS_RED
    )
    c = combine_runs([a, b], sign=1, run_number=-1)
    assert "period_histograms" in c.grouping
    assert len(c.grouping["period_histograms"]) == 2
    # Period 0 detector 0 counts are the per-run sum.
    np.testing.assert_array_equal(
        c.grouping["period_histograms"][0][0].counts,
        a.grouping["period_histograms"][0][0].counts + b.grouping["period_histograms"][0][0].counts,
    )
    # Per-period good frames accumulate.
    assert c.grouping["period_good_frames"][0] == pytest.approx(
        a.grouping["period_good_frames"][0] + b.grouping["period_good_frames"][0]
    )
    # period_mode preserved; reduces through the RG path.
    assert c.grouping["period_mode"] == str(PeriodMode.GREEN_MINUS_RED)
    ds = reduce_combined_run(c)
    assert ds.n_points > 0


# ---------------------------------------------------------------------------
# F9: chokepoint reuse
# ---------------------------------------------------------------------------


def test_subtract_routes_through_chokepoint(monkeypatch):
    """Co-subtract must call subtract_scaled_counts, not a parallel impl (F9)."""
    import asymmetry.core.data.combine as combine_mod

    calls = {"n": 0}
    real = combine_mod.subtract_scaled_counts

    def spy(counts, reference, scale):
        calls["n"] += 1
        return real(counts, reference, scale)

    monkeypatch.setattr(combine_mod, "subtract_scaled_counts", spy)
    a = simulate_run(_template(), _cos, total_events=1e5, seed=1)
    b = simulate_run(_template(), _cos, total_events=1e5, seed=2)
    combine_runs([a, b], sign=-1, scales=[1.0, 1.0])
    assert calls["n"] == 2  # one per detector
