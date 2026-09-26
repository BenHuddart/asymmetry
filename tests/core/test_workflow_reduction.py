"""Tests for :mod:`asymmetry.core.workflow.reduction`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from asymmetry.core.io import load
from asymmetry.core.simulate import (
    BUILTIN_TEMPLATES,
    PeriodSpec,
    simulate_run,
    simulate_two_period_run,
)
from asymmetry.core.transform.grouping import effective_group_indices, group_names
from asymmetry.core.transform.reduce import (
    correction_flags_from_grouping,
    reduce_grouped_asymmetry,
)
from asymmetry.core.workflow.reduction import (
    GREEN_MINUS_RED,
    ReductionSettings,
    estimate_alpha_for_run,
    reduce_run,
    reduction_source,
    resolve_reduction_grouping,
)
from tests.core.conftest import (
    CALIBRATION_ALPHA,
    CALIBRATION_RUN,
    DEADTIME_RUN,
    DEADTIME_US,
    SCAN_RUNS,
)


def _load_run(folder: Path, run_number: int):
    result = load(str(folder / f"SIM{run_number:08d}.nxs"))
    dataset = result[0] if isinstance(result, list) else result
    return dataset.run


@pytest.fixture(scope="module")
def scan_run(workflow_folder: Path):
    return _load_run(workflow_folder, SCAN_RUNS[0])


def _direct_reduction(run):
    """Reduce straight through the core chokepoint, on the loader's own grouping."""
    grouping = run.grouping
    n_histograms = len(run.histograms)
    flags = correction_flags_from_grouping(grouping)
    return reduce_grouped_asymmetry(
        histograms=run.histograms,
        grouping=grouping,
        forward_idx=effective_group_indices(
            grouping, int(grouping["forward_group"]), n_histograms=n_histograms
        ),
        backward_idx=effective_group_indices(
            grouping, int(grouping["backward_group"]), n_histograms=n_histograms
        ),
        alpha=float(grouping["alpha"]),
        use_deadtime=flags.use_deadtime,
        deadtime_mode=flags.deadtime_mode,
        use_background=flags.use_background,
        metadata=run.metadata,
    )


def test_default_reduction_matches_a_direct_chokepoint_call(scan_run) -> None:
    dataset = reduce_run(scan_run, ReductionSettings())
    direct = _direct_reduction(scan_run)
    assert np.array_equal(dataset.time, direct.time)
    assert np.array_equal(dataset.asymmetry, direct.asymmetry)
    assert np.array_equal(dataset.error, direct.error)


def test_reduced_dataset_keeps_the_runs_metadata_and_run(scan_run) -> None:
    dataset = reduce_run(scan_run, ReductionSettings())
    assert dataset.run is scan_run
    assert dataset.run_number == SCAN_RUNS[0]
    assert dataset.field == pytest.approx(0.0)


def test_alpha_is_applied_to_the_reduction(scan_run) -> None:
    unbalanced = reduce_run(scan_run, ReductionSettings(alpha=1.4, alpha_source="user"))
    default = reduce_run(scan_run, ReductionSettings())
    assert not np.allclose(unbalanced.asymmetry, default.asymmetry)
    grouping = resolve_reduction_grouping(scan_run, ReductionSettings(alpha=1.4))
    assert grouping["alpha"] == pytest.approx(1.4)


def test_rebinning_merges_bins(scan_run) -> None:
    default = reduce_run(scan_run, ReductionSettings())
    rebinned = reduce_run(scan_run, ReductionSettings(rebin=4))
    assert rebinned.n_points == default.n_points // 4
    assert np.mean(rebinned.error) < np.mean(default.error)


def test_time_window_clips_the_curve(scan_run) -> None:
    windowed = reduce_run(scan_run, ReductionSettings(t_min=1.0, t_max=4.0))
    assert windowed.time.min() >= 1.0
    assert windowed.time.max() <= 4.0
    assert windowed.n_points > 0


def test_deadtime_from_file_uses_the_runs_own_values(workflow_folder: Path) -> None:
    run = _load_run(workflow_folder, DEADTIME_RUN)
    grouping = resolve_reduction_grouping(run, ReductionSettings(deadtime="from_file"))
    assert grouping["deadtime_correction"] is True
    assert grouping["deadtime_mode"] == "file"
    assert grouping["dead_time_us"] == pytest.approx([DEADTIME_US] * len(run.histograms))

    corrected = reduce_run(run, ReductionSettings(deadtime="from_file"))
    uncorrected = reduce_run(run, ReductionSettings())
    assert not np.allclose(corrected.asymmetry, uncorrected.asymmetry)


def test_deadtime_off_is_the_default(scan_run) -> None:
    grouping = resolve_reduction_grouping(scan_run, ReductionSettings())
    assert grouping["deadtime_correction"] is False
    assert grouping["deadtime_mode"] == "off"


def test_estimate_alpha_recovers_the_simulated_balance(workflow_folder: Path) -> None:
    run = _load_run(workflow_folder, CALIBRATION_RUN)
    estimate = estimate_alpha_for_run(run, ReductionSettings())
    assert estimate.run_number == CALIBRATION_RUN
    assert estimate.method == "per_run_estimate"
    assert estimate.forward_group == 1
    assert estimate.backward_group == 2
    assert estimate.alpha == pytest.approx(CALIBRATION_ALPHA, rel=0.01)


def test_estimate_alpha_on_a_relaxing_zero_field_run_is_biased(scan_run) -> None:
    # The integral-ratio estimate balances ΣF against ΣB, which only measures
    # the detector efficiencies when the asymmetry averages to zero over the
    # window. A zero-field run relaxing from +20 % never does, so its estimate
    # runs high even though the run was simulated perfectly balanced — this is
    # why alpha is calibrated on a transverse-field run and why the CLI warns
    # when asked for an estimate on anything else.
    assert estimate_alpha_for_run(scan_run, ReductionSettings()).alpha > 1.2


def test_settings_round_trip_through_their_dict() -> None:
    settings = ReductionSettings(
        alpha=1.25,
        alpha_source="estimated:101",
        deadtime="from_file",
        background="range",
        background_range=(10, 80),
        pair=("Up", "Down"),
        t0_offset_bins=-2,
        t_good_offset_bins=5,
        rebin=2,
        t_min=0.5,
        t_max=8.0,
        period=GREEN_MINUS_RED,
    )
    assert ReductionSettings.from_dict(settings.to_dict()) == settings


@pytest.mark.parametrize(
    "kwargs",
    [
        {"deadtime": "estimate"},
        {"background": "fixed"},
        {"background_range": (10, 80)},
        {"background": "range", "background_range": (80, 10)},
        {"pair": ("Up", "up")},
        {"pair": ("Up", "")},
        {"t_good_offset_bins": -1},
        {"rebin": 0},
        {"alpha": 0.0},
        {"t_min": 5.0, "t_max": 1.0},
    ],
)
def test_settings_reject_values_outside_the_vocabulary(kwargs) -> None:
    with pytest.raises(ValueError):
        ReductionSettings(**kwargs)


def test_a_pair_named_or_numbered_swapped_negates_the_asymmetry(scan_run) -> None:
    default = reduce_run(scan_run, ReductionSettings())
    names = group_names(scan_run)
    # Names match case-insensitively, so the upper-cased ones must resolve too.
    by_name = (
        names[scan_run.grouping["backward_group"]].upper(),
        names[scan_run.grouping["forward_group"]].upper(),
    )
    for pair in (("2", "1"), by_name):
        swapped = reduce_run(scan_run, ReductionSettings(pair=pair))
        assert np.allclose(swapped.asymmetry, -default.asymmetry)


def test_an_unknown_group_names_the_groups_there_are(scan_run) -> None:
    with pytest.raises(ValueError, match="no group 'Up'; the groups are 1 "):
        resolve_reduction_grouping(scan_run, ReductionSettings(pair=("Up", "Down")))


def test_alpha_is_estimated_on_the_pair_it_will_balance(workflow_folder: Path) -> None:
    run = _load_run(workflow_folder, CALIBRATION_RUN)
    swapped = estimate_alpha_for_run(run, ReductionSettings(pair=("2", "1")))
    assert (swapped.forward_group, swapped.backward_group) == (2, 1)
    assert swapped.alpha == pytest.approx(1.0 / CALIBRATION_ALPHA, rel=0.01)


def test_t0_and_t_good_offsets_move_the_window(scan_run) -> None:
    file_grouping = resolve_reduction_grouping(scan_run, ReductionSettings())
    shifted = resolve_reduction_grouping(scan_run, ReductionSettings(t0_offset_bins=3))
    assert shifted["t0_bin"] == file_grouping["t0_bin"] + 3
    assert shifted["first_good_bin"] == file_grouping["first_good_bin"] + 3

    offset = resolve_reduction_grouping(
        scan_run, ReductionSettings(t0_offset_bins=3, t_good_offset_bins=10)
    )
    assert offset["first_good_bin"] == offset["t0_bin"] + 10


def _continuous_run(background_per_bin: float):
    run = simulate_run(
        BUILTIN_TEMPLATES["ideal_continuous_fb"].build(),
        lambda t: 20.0 * np.exp(-0.2 * t),
        total_events=2.0e7,
        seed=3,
        background_per_bin=background_per_bin,
        run_number=7,
    )
    # A PSI run: the pre-t0 region a range background averages exists.
    run.metadata["facility"] = "PSI"
    return run


def test_a_range_background_restores_the_diluted_asymmetry() -> None:
    run = _continuous_run(background_per_bin=1000.0)
    diluted = reduce_run(run, ReductionSettings(t_max=1.0))
    for background_range in (None, (100, 900)):
        settings = ReductionSettings(
            background="range", background_range=background_range, t_max=1.0
        )
        grouping = resolve_reduction_grouping(run, settings)
        assert grouping["background_mode"] == "range"
        restored = reduce_run(run, settings)
        assert np.mean(restored.asymmetry[:50]) == pytest.approx(20.0, abs=1.0)
        assert np.mean(restored.asymmetry[:50]) > np.mean(diluted.asymmetry[:50]) + 2.0


def test_a_pulsed_run_has_no_range_background(scan_run) -> None:
    with pytest.raises(ValueError, match="no pre-t0 region"):
        resolve_reduction_grouping(scan_run, ReductionSettings(background="range"))


def test_a_tail_fit_background_is_subtracted_or_the_reduction_says_why() -> None:
    template = BUILTIN_TEMPLATES["ideal_pulsed_fb"].build()
    run = simulate_run(
        template,
        lambda t: 20.0 * np.exp(-0.2 * t),
        total_events=4.0e7,
        seed=5,
        background_per_bin=2.0,
        run_number=8,
    )
    settings = ReductionSettings(background="tail_fit")
    assert resolve_reduction_grouping(run, settings)["background_mode"] == "tail_fit"
    subtracted = reduce_run(run, settings)
    plain = reduce_run(run, ReductionSettings())
    assert np.mean(subtracted.asymmetry[:50]) > np.mean(plain.asymmetry[:50])

    flat = simulate_run(template, lambda t: 0.0 * t, total_events=1.0e3, seed=5, run_number=9)
    with pytest.raises(ValueError, match="tail_fit background could not be subtracted"):
        reduce_run(flat, settings)


def _two_period_run():
    template = BUILTIN_TEMPLATES["ideal_pulsed_fb"].build()

    def relax(t, A=20.0):  # noqa: N803 (A is the conventional asymmetry symbol)
        return A * np.exp(-0.3 * t)

    return simulate_two_period_run(
        template,
        [
            PeriodSpec(relax, {"A": 12.0}, label="red"),
            PeriodSpec(relax, {"A": 20.0}, label="green"),
        ],
        total_events=4.0e7,
        seed=13,
        run_number=31,
    )


def test_green_minus_red_reduces_each_period_under_the_settings() -> None:
    run = _two_period_run()
    settings = ReductionSettings(period=GREEN_MINUS_RED, pair=("2", "1"))
    difference = reduce_run(run, settings)
    # green − red = (20 − 12)·e^(−0.3t), negated by the swapped pair.
    assert np.mean(difference.asymmetry[:40]) == pytest.approx(-8.0, abs=0.8)
    assert difference.run is run


def test_the_difference_needs_two_periods(scan_run) -> None:
    from asymmetry.core.data.dataset import MuonDataset

    single = MuonDataset(
        time=np.zeros(1), asymmetry=np.zeros(1), error=np.ones(1), metadata={}, run=scan_run
    )
    with pytest.raises(ValueError, match="two-period"):
        reduction_source(single, GREEN_MINUS_RED)
    with pytest.raises(ValueError, match="not a two-period"):
        reduce_run(scan_run, ReductionSettings(period=GREEN_MINUS_RED))


def test_estimate_alpha_reports_the_source_run_of_a_period_selection() -> None:
    """A period's own run number is encoded (run*1000+period); the estimate
    names the run it was cut from.
    """
    from asymmetry.core.io.periods import period_run

    run = _two_period_run()
    red = period_run(run, 0)
    estimate = estimate_alpha_for_run(red, ReductionSettings())
    assert estimate.run_number == run.run_number


def test_the_background_error_names_the_source_run_of_a_period_selection() -> None:
    from asymmetry.core.io.periods import period_run

    template = BUILTIN_TEMPLATES["ideal_pulsed_fb"].build()
    flat_run = simulate_two_period_run(
        template,
        [
            PeriodSpec(lambda t: 0.0 * t, {}, label="red"),
            PeriodSpec(lambda t: 0.0 * t, {}, label="green"),
        ],
        total_events=1.0e3,
        seed=5,
        run_number=44,
    )
    red = period_run(flat_run, 0)
    settings = ReductionSettings(background="tail_fit", period="red")
    with pytest.raises(ValueError, match=r"^Run 44: the tail_fit background"):
        reduce_run(red, settings)
