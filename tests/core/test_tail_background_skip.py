"""Optional tail estimation preserves observations and skips the whole result."""

import json
import pickle

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram
from asymmetry.core.transform.background import (
    apply_grouped_background_correction,
    fit_tail_background,
)
from asymmetry.core.transform.corrections import CorrectionOutcome, correction_warnings
from asymmetry.core.transform.fb_counts import GroupCounts
from asymmetry.core.transform.rebin import BinningSpec
from asymmetry.core.transform.reduce import reduce_period_asymmetry
from asymmetry.core.utils.constants import MUON_LIFETIME_US


def observations():
    return 100 * np.exp(-np.arange(200) * 0.1 / MUON_LIFETIME_US) + 5


@pytest.mark.parametrize("bad", ["signed", "empty", "short", "singular", "variance"])
def test_unusable_tail_preserves_both_groups_and_supplied_variance(bad):
    counts = observations()
    width = 0.1
    variance = np.abs(counts) + 1
    if bad == "signed":
        counts[-20] = -2
    elif bad == "empty":
        counts[99:] = 0
    elif bad == "short":
        counts, variance = counts[:6], variance[:6]
    elif bad == "singular":
        width = 1e-18
    else:
        variance[:] = 0
    source = GroupCounts(counts, variance)
    other = GroupCounts(observations()[: counts.size], np.ones(counts.size))
    before = counts.copy(), variance.copy()
    result = apply_grouped_background_correction(
        source, other, grouping={"background_mode": "tail_fit"}, t0_bin=0, bin_width_us=width
    )
    assert result.outcome.status == "skipped"
    assert result.outcome.usable and result.outcome.requested == "tail_fit"
    assert result.outcome.reason
    assert result.outcome.effective is None
    assert result.forward is source and result.backward is other
    np.testing.assert_array_equal(counts, before[0])
    np.testing.assert_array_equal(variance, before[1])


def test_signed_observation_is_not_a_clipped_poisson_estimate():
    counts = observations()
    counts[-20] = -2
    result = fit_tail_background(GroupCounts(counts, np.ones(200)), bin_width_us=0.1, t0_bin=0)
    assert not result.ok
    assert "negative" in result.message.lower()


def test_negative_outside_actual_tail_and_zero_background_are_successful():
    counts = 100 * np.exp(-np.arange(200) * 0.1 / MUON_LIFETIME_US)
    counts[0] = -2
    source = GroupCounts(counts, np.ones(200))
    result = apply_grouped_background_correction(
        source, source, grouping={"background_mode": "tail_fit"}, t0_bin=0, bin_width_us=0.1
    )
    assert result.outcome.status == "applied"
    assert result.outcome.effective["values"][0] == pytest.approx(0, abs=1e-12)
    assert result.forward.counts[0] < 0


@pytest.mark.parametrize("bad_periods", [(0,), (1,), (0, 1)])
@pytest.mark.parametrize("mode", ["green_plus_red", "green_minus_red"])
def test_contributing_periods_skip_atomically_with_deadtime(bad_periods, mode):
    periods = []
    for p in range(2):
        arrays = [observations(), observations() * 0.8]
        if p in bad_periods:
            arrays[1][99:] = 0
        periods.append([Histogram(a, 0.1, t0_bin=0, count_variance=np.abs(a) + 1) for a in arrays])
    grouping = {
        "period_histograms": periods,
        "background_mode": "tail_fit",
        "good_frames": 1e6,
        "dead_time_us": [0.01, 0.02],
    }
    kwargs = dict(
        histograms=periods[0],
        grouping=grouping,
        period_mode=mode,
        binning=BinningSpec(),
        forward_idx=[0],
        backward_idx=[1],
        alpha=1,
        use_deadtime=True,
        deadtime_mode="manual",
    )
    expected = reduce_period_asymmetry(**kwargs, use_background=False)
    actual = reduce_period_asymmetry(**kwargs, use_background=True)
    np.testing.assert_array_equal(actual.asymmetry, expected.asymmetry)
    np.testing.assert_array_equal(actual.error, expected.error)
    for record in actual.corrections.values():
        assert record.deadtime.status == "applied"
        assert record.background.status == "skipped"
        assert record.background.requested == "tail_fit"
    assert grouping["background_mode"] == "tail_fit"
    warnings = correction_warnings(actual.metadata_with_alignment({}))
    assert len(warnings) == 1
    assert "background" in warnings[0].lower()
    assert "unsubtracted" in warnings[0].lower()


def test_skip_round_trip_and_warning():
    record = CorrectionOutcome("background", "tail_fit", "skipped", "backward: no tail counts")
    assert CorrectionOutcome.from_dict(json.loads(json.dumps(record.to_dict()))) == record
    assert pickle.loads(pickle.dumps(record)) == record
    assert "backward" in record.warning


def test_successful_subtraction_can_make_negative_values():
    counts = np.full(200, 10.0)
    counts[0] = 0
    result = apply_grouped_background_correction(
        counts, counts, grouping={"background_mode": "tail_fit"}, t0_bin=0, bin_width_us=0.1
    )
    assert result.outcome.status == "applied"
    assert result.forward.counts[0] == pytest.approx(-10)


def test_both_unusable_groups_and_unrepresentable_amplitude_skip():
    for counts, t0 in ((np.zeros(200), 0), (observations(), -100000)):
        result = apply_grouped_background_correction(
            counts, counts, grouping={"background_mode": "tail_fit"}, t0_bin=t0, bin_width_us=0.1
        )
        assert result.outcome.status == "skipped"
        assert "forward" in result.outcome.reason and "backward" in result.outcome.reason
        np.testing.assert_array_equal(result.forward.counts, counts)


def test_skip_coordination_cannot_promote_invalid_input_failure():
    from asymmetry.core.transform.corrections import result_tail_skip

    invalid = CorrectionOutcome("background", "tail_fit", "failed", "invalid bin width")
    skipped = CorrectionOutcome("background", "tail_fit", "skipped", "empty tail")
    assert result_tail_skip({"red": invalid, "green": skipped}) is None
