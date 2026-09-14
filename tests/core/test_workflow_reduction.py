"""Tests for :mod:`asymmetry.core.workflow.reduction`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from asymmetry.core.io import load
from asymmetry.core.transform.grouping import effective_group_indices
from asymmetry.core.transform.reduce import (
    correction_flags_from_grouping,
    reduce_grouped_asymmetry,
)
from asymmetry.core.workflow.reduction import (
    ReductionSettings,
    estimate_alpha_for_run,
    reduce_run,
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
    estimate = estimate_alpha_for_run(run)
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
    assert estimate_alpha_for_run(scan_run).alpha > 1.2


def test_settings_round_trip_through_their_dict() -> None:
    settings = ReductionSettings(
        alpha=1.25,
        alpha_source="estimated:101",
        deadtime="from_file",
        rebin=2,
        t_min=0.5,
        t_max=8.0,
    )
    assert ReductionSettings.from_dict(settings.to_dict()) == settings


@pytest.mark.parametrize(
    "kwargs",
    [
        {"deadtime": "estimate"},
        {"background": "range"},
        {"rebin": 0},
        {"alpha": 0.0},
        {"t_min": 5.0, "t_max": 1.0},
    ],
)
def test_settings_reject_values_outside_the_vocabulary(kwargs) -> None:
    with pytest.raises(ValueError):
        ReductionSettings(**kwargs)
