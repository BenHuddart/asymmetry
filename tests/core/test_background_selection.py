"""Observation membership contracts for native background estimation."""

import numpy as np
import pytest

from asymmetry.core.transform.background import (
    apply_grouped_background_correction,
    available_background_modes,
    fit_tail_background,
    grouped_background_ranges,
    tail_background_window,
)
from asymmetry.core.transform.fb_counts import GroupCounts


@pytest.mark.parametrize(
    "first,last,start", [(40, 99, 70), (40, 98, 69), (40, 40, 40), (40, 41, 41)]
)
def test_tail_uses_inclusive_selected_support(first, last, start):
    assert tail_background_window(100, 0, last, first_good_bin=first) == (start, last)
    assert tail_background_window(100, 0, last, first_good_bin=first, fit_start_bin=42) == (
        42,
        last,
    )


def test_selected_tail_matches_identical_observations_and_supplied_uncertainty():
    raw = 500 * np.exp(-np.arange(100) * 0.1 / 2.1969811) + 12
    raw[60] = -20
    source = GroupCounts(raw, np.abs(raw) + 3)
    selected = fit_tail_background(
        source, bin_width_us=0.1, t0_bin=0, first_good_bin=40, last_good_bin=99
    )
    explicit = fit_tail_background(
        source, bin_width_us=0.1, t0_bin=0, fit_start_bin=70, last_good_bin=99
    )
    assert selected.ok
    assert selected.window == (70, 99)
    assert selected.rate_per_us == pytest.approx(explicit.rate_per_us)
    assert selected.rate_error_per_us == pytest.approx(explicit.rate_error_per_us)
    np.testing.assert_array_equal(selected.count_covariance, explicit.count_covariance)
    raw[80] = -20
    assert not fit_tail_background(
        GroupCounts(raw, np.abs(raw) + 3),
        bin_width_us=0.1,
        t0_bin=0,
        first_good_bin=40,
        last_good_bin=99,
    ).ok


@pytest.mark.parametrize("facility", ["PSI", "TRIUMF", "ISIS", ""])
@pytest.mark.parametrize("ranges", [([1, 71], [2, 63]), ([10, 60], [10, 60])])
def test_explicit_ranges_are_exact_even_when_equal_to_default(facility, ranges):
    raw = np.arange(200.0) ** 2 + 10
    variance = raw + 7
    result = apply_grouped_background_correction(
        GroupCounts(raw, variance),
        GroupCounts(raw + 5, variance * 2),
        grouping={"background_mode": "range", "background_ranges": ranges},
        t0_bin=100,
        bin_width_us=0.005,
        facility=facility,
    )
    assert result.outcome.effective["ranges"] == tuple(map(tuple, ranges))
    for group, bounds, counts, var in zip(
        (result.forward, result.backward),
        ranges,
        (raw, raw + 5),
        (variance, variance * 2),
        strict=True,
    ):
        start, stop = bounds
        selected = slice(start, stop + 1)
        level = np.sum(counts[selected]) / (stop - start + 1)
        np.testing.assert_allclose(group.counts, counts - level)
        assert group.shared[0].variance == pytest.approx(
            np.sum(var[selected]) / (stop - start + 1) ** 2
        )


def test_automatic_defaults_keep_native_period_convention():
    windows = grouped_background_ranges(
        {}, t0_bin=100, acquisition_start_bin=0, length=200, bin_width_us=0.005, facility="PSI"
    )
    assert [(w.first, w.last) for w in windows] == [(10, 57), (10, 57)]


@pytest.mark.parametrize("facility", ["ISIS", "PSI", ""])
def test_range_method_is_available_independent_of_identity(facility):
    assert "range" in available_background_modes(
        metadata={"facility": facility}, source_file="run.nxs"
    )


def test_zero_t0_requires_authored_interval():
    with pytest.raises(ValueError, match="explicit background interval"):
        apply_grouped_background_correction(
            np.ones(100),
            np.ones(100),
            grouping={"background_mode": "range"},
            t0_bin=0,
            bin_width_us=0.1,
        )
    result = apply_grouped_background_correction(
        np.ones(100),
        np.ones(100),
        grouping={"background_range": [0, 0]},
        t0_bin=0,
        bin_width_us=0.1,
    )
    assert result.outcome.status == "applied"


def test_cropped_tail_and_deadtime_dependencies_select_identical_raw_observations():
    from asymmetry.core.data.dataset import Histogram
    from asymmetry.core.fitting.count_corrections import count_background_domain
    from asymmetry.core.transform.fb_counts import BinWindow
    from asymmetry.core.transform.reduce import corrected_grouped_counts

    histograms = [Histogram(np.full(120, level), 0.1, t0) for level, t0 in ((10.0, 2), (8.0, 5))]
    histograms[0].counts[60] = 1000  # Outside tail and the narrower requested output.
    grouping = {
        "groups": {1: [1], 2: [2]},
        "first_good_bin": 43,
        "last_good_bin": 102,
        "background_mode": "tail_fit",
        "dead_time_us": [1.0, 1.0],
        "good_frames": 10000,
    }
    options = dict(
        histograms=histograms,
        grouping=grouping,
        forward_idx=[0],
        backward_idx=[1],
        use_deadtime=True,
        deadtime_mode="manual",
        use_background=True,
        output_window=BinWindow(95, 99),
    )
    corrected = corrected_grouped_counts(**options)
    effective = corrected.corrections.background.effective
    assert effective["ranges"] == ((70, 99), (70, 99))
    assert effective["values"] == pytest.approx((10 / 0.99, 8 / 0.992))
    scope = corrected.corrections.deadtime.effective["scope"]["raw_bin_ranges"]
    assert scope == {"0": ((70, 99),), "1": ((73, 102),)}
    domains = count_background_domain(histograms, grouping, [(1, [0]), (2, [1])], {})
    assert all(
        [(window.first, window.last) for window in windows] == [(70, 99)]
        for _, _, windows in domains
    )
    histograms[0].counts[80] = 1000
    from asymmetry.core.transform.deadtime import DeadtimeCorrectionError

    with pytest.raises(DeadtimeCorrectionError) as error:
        corrected_grouped_counts(**options)
    assert error.value.bin_index == 80


@pytest.mark.parametrize("factor", [1, 4, 7])
def test_explicit_range_selection_precedes_output_binning(factor):
    from asymmetry.core.data.dataset import Histogram
    from asymmetry.core.transform.rebin import BinningSpec
    from asymmetry.core.transform.reduce import corrected_grouped_counts

    raw = np.arange(120.0) ** 2 + 1
    grouping = {"first_good_bin": 80, "last_good_bin": 119, "background_range": [1, 71]}
    corrected = corrected_grouped_counts(
        histograms=[Histogram(raw, 0.005), Histogram(raw + 10, 0.005)],
        grouping=grouping,
        forward_idx=[0],
        backward_idx=[1],
        use_deadtime=False,
        deadtime_mode="off",
        use_background=True,
        facility="PSI",
        output_binning=BinningSpec(factor=factor),
    )
    assert corrected.corrections.background.effective["values"] == pytest.approx(
        (raw[1:72].mean(), raw[1:72].mean() + 10)
    )
    assert corrected.corrections.background.effective["details"]["selection"] == "explicit"


def test_automatic_range_fourier_ignores_cached_levels_and_keeps_uncertainty():
    from asymmetry.core.data.dataset import Histogram, Run
    from asymmetry.core.fourier.group_preparation import prepare_group_counts

    raw = np.arange(200.0) + 10
    run = Run(
        histograms=[Histogram(raw, 0.005, 100), Histogram(raw + 10, 0.005, 100)],
        metadata={"facility": "PSI"},
        grouping={
            "groups": {1: [1], 2: [2]},
            "t0_bin": 100,
            "background_correction": True,
            "background_mode": "range",
            "background_values": [123, 456],
        },
    )
    prepared = prepare_group_counts(run, [1, 2])
    for gid, offset in ((1, 0), (2, 10)):
        result = prepared.backgrounds[gid]
        assert result.outcome.requested == "range"
        assert result.outcome.effective["ranges"] == ((10, 57), (10, 57))
        assert result.outcome.effective["values"] == pytest.approx(
            (raw[10:58].mean() + offset,) * 2
        )
        assert result.forward.shared[0].variance == pytest.approx(
            (raw[10:58] + offset).sum() / 48**2
        )
    assert "background_range" not in run.grouping
    assert "background_ranges" not in run.grouping


@pytest.mark.parametrize("key", ["background_fixed_values", "background_fix", "bkg_fix"])
@pytest.mark.parametrize("pair", [{}, {"forward_group": 1, "backward_group": 2}])
def test_grouped_fixed_levels_follow_authored_pair_routing(key, pair):
    from asymmetry.core.data.dataset import Histogram, Run
    from asymmetry.core.fourier.group_preparation import prepare_group_counts

    run = Run(
        histograms=[Histogram(np.full(20, 10.0), 0.1) for _ in range(2)],
        grouping={
            "groups": {1: [1], 2: [2]},
            **pair,
            "background_correction": True,
            "background_mode": "fixed",
            key: [2.0, 3.0],
        },
    )
    prepared = prepare_group_counts(run, [1, 2])
    np.testing.assert_array_equal(prepared.backgrounds[1].forward.counts, 8)
    np.testing.assert_array_equal(prepared.backgrounds[2].forward.counts, 7)


def test_off_profile_round_trip_preserves_authored_bounds(tmp_path):
    from asymmetry.core.data.dataset import Histogram, Run
    from asymmetry.core.project.profiles import (
        GroupingProfile,
        profile_fingerprint_for_run,
        profile_from_payload,
        resolve_effective_grouping,
    )

    ranges = {"forward": [10, 60], "backward": [11, 61]}
    run = Run(
        histograms=[Histogram(np.ones(100), 0.005, 50) for _ in range(2)],
        grouping={"groups": {1: [1], 2: [2]}},
    )
    profile = profile_from_payload(
        dict(run.grouping, background_correction=False, background_ranges=ranges),
        "saved",
        profile_fingerprint_for_run(run),
    )
    from asymmetry.core.project.schema import CURRENT_SCHEMA_VERSION, load_project, save_project

    project = tmp_path / "interval.asymp"
    save_project(
        {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "datasets": [],
            "grouping_profiles": [profile.to_dict()],
        },
        project,
    )
    restored = GroupingProfile.from_dict(load_project(project)["grouping_profiles"][0])
    assert restored.background_policy.details["background_ranges"] == ranges
    grouping = resolve_effective_grouping(restored, run)
    assert not grouping["background_correction"]
    assert grouping["background_ranges"] == ranges


def test_automatic_profile_does_not_inherit_stale_authored_bounds():
    from asymmetry.core.data.dataset import Histogram, Run
    from asymmetry.core.project.profiles import (
        BackgroundPolicy,
        GroupingProfile,
        ProfileFingerprint,
        resolve_effective_grouping,
    )

    run = Run(
        histograms=[Histogram(np.ones(100), 0.005, 50) for _ in range(2)],
        grouping={"groups": {1: [1], 2: [2]}, "background_range": [0, 0]},
    )
    profile = GroupingProfile(
        "auto",
        ProfileFingerprint("", 2),
        groups={1: [1], 2: [2]},
        background_policy=BackgroundPolicy(mode="range"),
    )
    resolved = resolve_effective_grouping(profile, run)
    assert "background_range" not in resolved


@pytest.mark.parametrize(
    "key",
    [
        "background_forward_range",
        "background_backward_range",
        "background_fixed_values",
        "background_fix",
        "bkg_fix",
    ],
)
def test_background_alias_edits_invalidate_spectral_identity(key):
    from asymmetry.core.data.dataset import Run
    from asymmetry.core.fourier.spectrum import fourier_grouping_digest

    run = Run(grouping={"background_correction": True, "background_mode": "range", key: [1, 4]})
    before = fourier_grouping_digest(run)
    run.grouping[key] = [2, 5]
    assert fourier_grouping_digest(run) != before


def test_automatic_cropped_support_requires_an_explicit_interval():
    with pytest.raises(ValueError, match="explicit background interval"):
        grouped_background_ranges(
            {}, t0_bin=1, acquisition_start_bin=2, length=20, bin_width_us=0.1, facility=""
        )


def test_contributing_periods_use_their_own_resolved_good_support():
    from asymmetry.core.data.dataset import Histogram
    from asymmetry.core.transform.rebin import BinningSpec
    from asymmetry.core.transform.reduce import corrected_period_counts

    red = [Histogram(np.full(100, v), 0.1, good_bin_start=40, good_bin_end=99) for v in (10.0, 8.0)]
    green = [
        Histogram(np.full(100, v), 0.1, good_bin_start=60, good_bin_end=99) for v in (12.0, 9.0)
    ]
    grouping = {
        "groups": {1: [1], 2: [2]},
        "period_histograms": [red, green],
        "background_mode": "tail_fit",
        "good_window_policy": {"mode": "from_file"},
        "detector_t0_bins": [0, 0],
        "detector_first_good_bins": [40, 40],
        "detector_last_good_bins": [99, 99],
    }
    results = corrected_period_counts(
        red,
        grouping,
        "green_plus_red",
        binning=BinningSpec(),
        forward_idx=[0],
        backward_idx=[1],
        use_deadtime=False,
        deadtime_mode="off",
        use_background=True,
    )
    assert results["red"].corrections.background.effective["ranges"] == ((70, 99), (70, 99))
    assert results["green"].corrections.background.effective["ranges"] == ((80, 99), (80, 99))
    assert results["red"].corrections.background.effective["values"] == pytest.approx((10.0, 8.0))
    assert results["green"].corrections.background.effective["values"] == pytest.approx((12.0, 9.0))


@pytest.mark.parametrize("interval", ["0:0", "1:71"])
def test_cli_explicit_interval_cold_warm_results(tmp_path, capsys, interval):
    import json

    from asymmetry import cli
    from asymmetry.core.io.nexus_writer import write_nexus_v1
    from tests.core.test_tail_skip_consumers import tail_run

    run = tail_run(bad=False)
    for histogram in run.histograms:
        histogram.counts = np.rint(histogram.counts)
    folder = tmp_path / "runs"
    folder.mkdir()
    write_nexus_v1(run, folder / "SIM00000042.nxs")
    args = [
        "reduce",
        str(folder),
        "--runs",
        "42",
        "--background",
        f"range:{interval}",
        "--workdir",
        str(tmp_path / "work"),
        "--json",
    ]
    first, last = map(int, interval.split(":"))
    expected = [hist.counts[first : last + 1].mean() for hist in run.histograms]
    for recomputed in (True, False):
        cli.main(args)
        entry = json.loads(capsys.readouterr().out)["entries"][0]
        assert entry["recomputed"] == recomputed
        background = entry["run"]["correction_outcomes"]["red"]["background"]
        assert background["effective"]["ranges"] == [[first, last], [first, last]]
        assert background["effective"]["values"] == pytest.approx(expected)


@pytest.mark.parametrize(
    "details",
    [
        {"background_ranges": {"forward": [1, 7], "backward": [8, 19]}},
        {"background_ranges": [[1, 7], [8, 19]]},
        {"background_forward_range": [1, 7], "background_backward_range": [8, 19]},
    ],
)
def test_grouped_explicit_ranges_follow_default_pair_routing(details):
    from asymmetry.core.data.dataset import Histogram, Run
    from asymmetry.core.fourier.group_preparation import prepare_group_counts

    raw = np.arange(30.0) + 10
    run = Run(
        histograms=[Histogram(raw, 0.005) for _ in range(2)],
        grouping={
            "groups": {1: [1], 2: [2]},
            "background_correction": True,
            "background_mode": "range",
            **details,
        },
    )
    prepared = prepare_group_counts(run, [1, 2])
    assert prepared.backgrounds[1].outcome.effective["values"] == pytest.approx(
        (raw[1:8].mean(),) * 2
    )
    assert prepared.backgrounds[2].outcome.effective["values"] == pytest.approx(
        (raw[8:20].mean(),) * 2
    )


@pytest.mark.parametrize(
    "details", [{"background_range": [10, 20]}, {"background_ranges": {1: [10, 20]}}, {}]
)
def test_range_pair_mapping_changes_values_and_spectral_identity(details):
    from asymmetry.core.data.dataset import Histogram, Run
    from asymmetry.core.fourier.group_preparation import prepare_group_counts
    from asymmetry.core.fourier.spectrum import fourier_grouping_digest

    run = Run(
        histograms=[
            Histogram(np.arange(30.0) + level, 0.1, t0) for level, t0 in ((10, 2), (20, 5), (30, 8))
        ],
        grouping={
            "groups": {1: [1], 2: [2], 3: [3]},
            "forward_group": 1,
            "backward_group": 2,
            "background_correction": True,
            "background_mode": "range",
            **details,
        },
    )
    # An automatic pre-t0 default for these short unequal anchors lacks support;
    # its staleness identity must still record the pair without executing it.
    old_digest = fourier_grouping_digest(run)
    if details:
        first = prepare_group_counts(run, [1]).backgrounds[1].outcome.effective["values"][0]
        assert first == pytest.approx(22.0)
    run.grouping["backward_group"] = 3
    assert fourier_grouping_digest(run) != old_digest
    if details:
        second = prepare_group_counts(run, [1]).backgrounds[1].outcome.effective["values"][0]
        assert second == pytest.approx(19.0)


@pytest.mark.parametrize("explicit", [False, True])
def test_only_automatic_range_identity_depends_on_facility(explicit):
    from asymmetry.core.data.dataset import Run
    from asymmetry.core.fourier.spectrum import fourier_grouping_digest

    run = Run(
        metadata={"facility": "PSI"},
        grouping={"background_correction": True, "background_mode": "range"},
    )
    if explicit:
        run.grouping["background_range"] = [10, 60]
    before = fourier_grouping_digest(run)
    run.metadata["facility"] = ""
    assert (fourier_grouping_digest(run) == before) == explicit
