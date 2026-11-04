"""Automatic policies resolve current counts; stored calibrations remain authored."""

from dataclasses import replace

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.project.profiles import (
    AlphaPolicy,
    BackgroundPolicy,
    BetaPolicy,
    DeadtimePolicy,
    GoodWindowPolicy,
    GroupingProfile,
    ProfileFingerprint,
    T0Policy,
    effective_grouping_for_loaded_run,
    profile_fingerprint_for_run,
    resolve_effective_grouping,
)
from asymmetry.core.transform.asymmetry import estimate_alpha
from asymmetry.core.transform.corrections import BackgroundReference, CorrectionUnavailableError
from asymmetry.core.transform.grouping import InvalidGroupingError, independent_pair_indices
from asymmetry.core.transform.rebin import binning_spec
from asymmetry.core.transform.reduce import reduce_grouped_asymmetry


def policy_run(levels=(10.0, 30.0, 50.0)):
    return Run(
        run_number=123,
        histograms=[Histogram(np.full(20, level), 0.1) for level in levels],
        grouping={"first_good_bin": 0, "last_good_bin": 19},
    )


def automatic_profile():
    return GroupingProfile(
        name="sample",
        fingerprint=ProfileFingerprint("", 3),
        groups={1: [1], 2: [3]},
        alpha_policy=AlphaPolicy(mode="per_run_estimate"),
    )


def test_loaded_run_api_resolves_the_declared_background_reference():
    run = policy_run()
    profile = replace(
        automatic_profile(),
        fingerprint=profile_fingerprint_for_run(run),
        background_policy=BackgroundPolicy(
            mode="reference_run", details={"background_run": {"run_number": 999}}
        ),
    )
    reference = BackgroundReference(policy_run((1, 3, 5)).histograms, scale=2, run_number=999)
    seen = []

    def resolver(grouping):
        seen.append(grouping["background_run"]["run_number"])
        return reference

    grouping = effective_grouping_for_loaded_run([profile], run, reference_resolver=resolver)
    assert grouping["alpha"] == pytest.approx((10 - 2) / (50 - 10))
    assert seen == [999]


@pytest.mark.parametrize("levels", [(10, 30, 0), (0, 30, 50), (10, 30, -1)])
def test_automatic_alpha_rejects_nonpositive_integrals(levels):
    run = policy_run(levels)
    with pytest.raises(ValueError, match="[Aa]lpha"):
        resolve_effective_grouping(automatic_profile(), run)
    # A deliberately fixed unity coefficient remains valid on these same counts.
    fixed = replace(automatic_profile(), alpha_policy=AlphaPolicy(value=1.0))
    assert resolve_effective_grouping(fixed, run)["alpha"] == 1.0


def test_explicit_auto_t0_failure_is_unavailable_but_file_timing_remains_usable():
    run = policy_run((0, 0, 0))
    profile = replace(
        automatic_profile(),
        alpha_policy=AlphaPolicy(value=1.7),
        beta_policy=BetaPolicy(
            mode="calibrated", value=1.0, error=0.02, method="count_fit", source_run=999
        ),
        t0_policy=T0Policy(mode="auto_detect"),
    )
    with pytest.raises(InvalidGroupingError, match="t0") as error:
        resolve_effective_grouping(profile, run)
    assert error.value.grouping["alpha"] == 1.7
    assert error.value.grouping["beta_reference_run"] == 999
    assert resolve_effective_grouping(replace(profile, t0_policy=T0Policy()), run)["alpha"] == 1.7


def test_automatic_alpha_uses_selected_held_period():
    run = policy_run()
    green = policy_run((40, 60, 20))
    run.grouping["period_histograms"] = [run.histograms, green.histograms]
    profile = replace(automatic_profile(), period_mode="green")
    assert resolve_effective_grouping(profile, run)["alpha"] == pytest.approx(2.0)


@pytest.mark.parametrize("previous", ["file", "manual", "estimate"])
@pytest.mark.parametrize(
    "current,expected",
    [("from_file", [0.1, 0.2, 0.3]), ("manual", [0.01, 0.02, 0.03]), ("estimate", [0.04] * 3)],
)
def test_selected_period_keeps_the_requested_deadtime_policy(previous, current, expected):
    run = policy_run()
    run.grouping.update(
        period_histograms=[run.histograms, run.histograms],
        period_dead_time_us=[[0.5] * 3, [0.1, 0.2, 0.3]],
        deadtime_mode=previous,
        dead_time_us=[0.9] * 3,
    )
    profile = replace(
        automatic_profile(),
        period_mode="green",
        alpha_policy=AlphaPolicy(value=1.7),
        deadtime_policy=DeadtimePolicy(mode=current, values=[0.01, 0.02, 0.03], estimated_us=0.04),
    )
    effective = resolve_effective_grouping(profile, run)
    assert effective["dead_time_us"] == expected
    assert profile.deadtime_policy.mode == current


@pytest.mark.parametrize(
    "forward,backward,excluded,expected",
    [
        ([1], [3], [], 0.2),
        ([1, 2], [3], [], 0.8),
        ([3], [1, 2], [], 1.25),
        ([1, 2], [3], [1], 0.6),
    ],
)
def test_current_membership_orientation_and_exclusions_reach_reduction(
    forward, backward, excluded, expected
):
    run = policy_run()
    original = [hist.counts.copy() for hist in run.histograms]
    profile = replace(
        automatic_profile(), groups={1: forward, 2: backward}, excluded_detectors=excluded
    )
    grouping = resolve_effective_grouping(profile, run)
    assert grouping["alpha"] == pytest.approx(expected)
    forward_idx, backward_idx = independent_pair_indices(grouping, n_histograms=3)
    result = reduce_grouped_asymmetry(
        histograms=run.histograms,
        grouping=grouping,
        binning=binning_spec(grouping),
        forward_idx=forward_idx,
        backward_idx=backward_idx,
        alpha=grouping["alpha"],
        use_deadtime=False,
        deadtime_mode="off",
        use_background=False,
    )
    np.testing.assert_allclose(result.asymmetry, 0, atol=1e-12)
    for hist, raw in zip(run.histograms, original):
        np.testing.assert_array_equal(hist.counts, raw)


def test_automatic_alpha_keeps_window_and_declared_background_source():
    run = policy_run()
    run.metadata["facility"] = "PSI"
    # Outside-window counts deliberately have a different F/B balance.
    run.histograms[0].counts[:4] = 400
    reference = BackgroundReference(policy_run((1, 3, 5)).histograms, scale=2, run_number=999)
    declaration = {"run_number": 999, "scale": 2}
    seen = []

    def resolve_reference(grouping):
        seen.append(grouping["background_run"])
        return reference

    profile = replace(
        automatic_profile(),
        groups={1: [1, 2], 2: [3]},
        good_window_policy=GoodWindowPolicy(
            mode="manual", first_offset_bins=4, last_offset_bins=12
        ),
        background_policy=BackgroundPolicy(
            mode="reference_run", details={"background_run": declaration}
        ),
    )
    grouping = resolve_effective_grouping(profile, run, reference_resolver=resolve_reference)
    assert grouping["alpha"] == pytest.approx((10 + 30 - 2 * (1 + 3)) / (50 - 2 * 5))
    assert (grouping["first_good_bin"], grouping["last_good_bin"]) == (4, 12)
    assert seen == [declaration]
    assert grouping["background_run"] == declaration


def test_range_background_is_reevaluated_for_current_groups():
    run = policy_run()
    run.metadata["facility"] = "PSI"
    for hist, pedestal in zip(run.histograms, (1, 2, 3)):
        hist.counts[:5] = pedestal
        hist.t0_bin = 5
        hist.good_bin_start = 6
        hist.good_bin_end = 19
    profile = replace(
        automatic_profile(),
        background_policy=BackgroundPolicy(mode="range", details={"background_range": [0, 2]}),
    )
    for members, expected in (([1], 9 / 47), ([1, 2], 37 / 47)):
        grouping = resolve_effective_grouping(replace(profile, groups={1: members, 2: [3]}), run)
        assert grouping["alpha"] == pytest.approx(expected)
        assert grouping["background_range"] == [0, 2]


@pytest.mark.parametrize("mode", ["fixed", "calibrated"])
@pytest.mark.parametrize("deadtime_mode", ["manual", "from_file", "estimate"])
def test_stored_values_and_provenance_survive_regrouping(mode, deadtime_mode):
    run = policy_run()
    run.grouping["dead_time_us"] = [0.01, 0.02, 0.03]
    alpha = AlphaPolicy(mode=mode, value=1.7, error=0.02, method="general", source_run=999)
    beta = BetaPolicy(mode=mode, value=1.3, error=0.01, method="count_fit", source_run=998)
    profile = replace(
        automatic_profile(),
        alpha_policy=alpha,
        beta_policy=beta,
        t0_policy=T0Policy(mode="manual", offset_bins=1),
        deadtime_policy=DeadtimePolicy(
            mode=deadtime_mode,
            values=[0.04, 0.05, 0.06],
            estimated_us=0.04,
            source_run=997,
            method="stored",
        ),
    )
    before = profile.to_dict()
    for members in ([1], [1, 2]):
        effective = resolve_effective_grouping(replace(profile, groups={1: members, 2: [3]}), run)
        assert effective["alpha"] == 1.7
        assert effective["beta"] == 1.3
        assert effective["t0_bin"] == 1
        if mode == "calibrated":
            assert (
                effective["alpha_method"],
                effective["alpha_reference_run"],
                effective["alpha_error"],
            ) == ("general", 999, 0.02)
            assert (
                effective["beta_method"],
                effective["beta_reference_run"],
                effective["beta_error"],
            ) == ("count_fit", 998, 0.01)
        expected = {
            "from_file": [0.01, 0.02, 0.03],
            "manual": [0.04, 0.05, 0.06],
            "estimate": [0.04] * 3,
        }
        assert effective["dead_time_us"] == expected[deadtime_mode]
    assert profile.to_dict() == before


@pytest.mark.parametrize(
    "forward,backward",
    [([], []), ([0], [1]), ([1], [0]), ([-1], [2]), ([1e308, 1e308], [1, 1]), ([1e308], [1e-308])],
)
def test_ratio_helper_never_supplies_a_default_for_an_unusable_estimate(forward, backward):
    with np.errstate(over="ignore"):
        with pytest.raises(CorrectionUnavailableError, match="alpha"):
            estimate_alpha(np.array(forward), np.array(backward))


def test_profile_keeps_acquired_period_t0_sources():
    run = policy_run()
    run.grouping.update(
        t0_source="conflict",
        period_t0_sources=["conflict", "file"],
        period_histograms=[run.histograms, run.histograms],
    )
    profile = replace(automatic_profile(), period_mode="green")
    grouping = resolve_effective_grouping(profile, run)
    assert grouping["t0_source"] == "file"
    assert grouping["period_t0_sources"] == ["conflict", "file"]
