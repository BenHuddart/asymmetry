"""Independent source clocks for count subtraction, using own tagged observations."""

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.fourier.group_preparation import prepare_group_counts
from asymmetry.core.io import resolve_background_reference
from asymmetry.core.transform.corrections import BackgroundReference, reference_observation_metadata
from asymmetry.core.transform.reduce import corrected_grouped_counts


def timing_runs():
    grouping = {
        "background_mode": "reference_run",
        "background_correction": True,
        "groups": {1: [1], 2: [2]},
        "forward_group": 1,
        "backward_group": 2,
    }
    sample = Run(
        run_number=1,
        source_file="sample.nxs",
        grouping=grouping,
        histograms=[Histogram(np.full(32, x), 0.016) for x in (200.0, 150.0)],
    )
    reference = Run(
        run_number=2,
        source_file="reference.nxs",
        histograms=[Histogram(np.arange(64) + x, 0.016, t0_bin=2) for x in (10.0, 20.0)],
    )
    return sample, reference


def reduce_reference(sample, reference, consumer, monkeypatch):
    if consumer == "spectral":
        monkeypatch.setattr(
            "asymmetry.core.io.resolve_background_reference", lambda *a, **k: reference
        )
        return prepare_group_counts(sample, [1], use_deadtime=False).backgrounds[1].forward.counts
    return corrected_grouped_counts(
        histograms=sample.histograms,
        grouping=sample.grouping,
        forward_idx=[0],
        backward_idx=[1],
        use_deadtime=False,
        deadtime_mode="off",
        use_background=True,
        metadata=reference_observation_metadata(sample),
        reference_resolver=lambda _: reference,
    ).forward.counts


@pytest.mark.parametrize("consumer", ["fb", "spectral"])
def test_noop_sample_timing_cannot_shift_reference(consumer, monkeypatch):
    sample, run = timing_runs()
    reference = BackgroundReference(run.histograms, 1.0, run_number=2, source_file=run.source_file)
    expected = 188.0 - np.arange(32)
    np.testing.assert_array_equal(
        reduce_reference(sample, reference, consumer, monkeypatch), expected
    )
    sample.grouping["effective_detector_t0_bins"] = [0, 0]
    np.testing.assert_array_equal(
        reduce_reference(sample, reference, consumer, monkeypatch), expected
    )


@pytest.mark.parametrize("consumer", ["fb", "spectral"])
def test_reference_own_effective_timing_survives_resolution(consumer, monkeypatch):
    sample, run = timing_runs()
    run.grouping["effective_detector_t0_bins"] = [4, 4]
    reference = resolve_background_reference(
        {"run_number": 2, "source_file": run.source_file, "period_number": 1},
        datasets=[MuonDataset(np.array([]), np.array([]), np.array([]), run=run)],
        allow_file=False,
    )
    np.testing.assert_array_equal(
        reduce_reference(sample, reference, consumer, monkeypatch), 186.0 - np.arange(32)
    )


@pytest.mark.parametrize("consumer", ["fb", "spectral"])
def test_subtraction_rejects_incompatible_native_widths(consumer, monkeypatch):
    sample, run = timing_runs()
    for hist in run.histograms:
        hist.bin_width = 0.032
    reference = BackgroundReference(run.histograms, 1.0, run_number=2, source_file=run.source_file)
    with pytest.raises(ValueError, match="compatible.*grid"):
        reduce_reference(sample, reference, consumer, monkeypatch)


@pytest.mark.parametrize("consumer", ["fb", "spectral"])
@pytest.mark.parametrize("steps,accepted", [(1, True), (2, False), (1000, False)])
def test_native_width_precision_is_one_ulp(consumer, steps, accepted, monkeypatch):
    sample, run = timing_runs()
    width = 0.016
    for _ in range(steps):
        width = np.nextafter(width, np.inf)
    for hist in run.histograms:
        hist.bin_width = width
    reference = BackgroundReference(run.histograms, 1.0, run_number=2, source_file=run.source_file)
    if accepted:
        np.testing.assert_array_equal(
            reduce_reference(sample, reference, consumer, monkeypatch), 188.0 - np.arange(32)
        )
    else:
        with pytest.raises(ValueError, match="compatible.*grid"):
            reduce_reference(sample, reference, consumer, monkeypatch)


def test_reference_alignment_retains_cross_source_fractional_residuals():
    from asymmetry.core.transform.grouping import resolve_alignment
    from asymmetry.core.transform.reference_timing import ReferenceTiming

    sample, run = timing_runs()
    # Exact t0 differs from its integer bin centre by +0.2 and +0.4 bins.
    sample.histograms[0].t0_time_us = 0.7 * 0.016
    run.histograms[0].t0_time_us = 2.9 * 0.016
    run.histograms[1].t0_time_us = 2.6 * 0.016
    sample.grouping["effective_detector_t0_bins"] = [3, 3]
    run.grouping["t0_policy"] = {"mode": "manual", "offset_bins": 2}
    reference = BackgroundReference(
        run.histograms,
        1.0,
        timing=ReferenceTiming.from_grouping(run.histograms, run.grouping, run.metadata),
    )
    crop = resolve_alignment(sample.histograms, [0], [1], detector_t0_bins=[3, 3])
    aligned = reference.alignment(sample.histograms, sample.grouping, crop, [0], [1])
    assert aligned.detector_offsets == ((0, -1), (1, -1))
    assert aligned.common_t0_bin == 3
    assert dict(aligned.timing_sources) == {0: "manual", 1: "manual"}
    assert dict(aligned.timing_residuals_us) == pytest.approx({0: 0.2 * 0.016, 1: -0.1 * 0.016})
    assert [h.t0_bin for h in run.histograms] == [2, 2]


@pytest.mark.parametrize(
    "mode,first", [("from_file", 188.0), ("manual", 186.0), ("auto_detect", 184.0)]
)
@pytest.mark.parametrize("consumer", ["fb", "spectral"])
def test_reference_uses_own_timing_policy(mode, first, consumer, monkeypatch):
    sample, run = timing_runs()
    run.metadata["facility"] = "PSI"
    run.grouping["t0_policy"] = {"mode": mode, "offset_bins": 2}
    for hist in run.histograms:
        hist.counts[6] = 1000
    reference = resolve_background_reference(
        {"run_number": 2},
        datasets=[MuonDataset(np.array([]), np.array([]), np.array([]), run=run)],
        allow_file=False,
    )
    result = reduce_reference(sample, reference, consumer, monkeypatch)
    assert result[0] == first if mode != "auto_detect" else result[0] == -800.0
    assert result[1] == first - 1


@pytest.mark.parametrize("consumer", ["fb", "spectral"])
def test_unused_incompatible_reference_spectator_does_not_block(consumer, monkeypatch):
    sample, run = timing_runs()
    run.histograms.append(Histogram(np.ones(3), 0.123, t0_bin=1))
    reference = BackgroundReference(run.histograms, 1.0, run_number=2, source_file=run.source_file)
    np.testing.assert_array_equal(
        reduce_reference(sample, reference, consumer, monkeypatch), 188.0 - np.arange(32)
    )


def test_maxent_input_retains_reference_timing_and_noop_sample_invariance(monkeypatch):
    from asymmetry.core.maxent.engine import MaxEntConfig, build_maxent_input

    sample, run = timing_runs()
    run.grouping["t0_policy"] = {"mode": "manual", "offset_bins": 2}
    reference = resolve_background_reference(
        {"run_number": 2},
        datasets=[MuonDataset(np.array([]), np.array([]), np.array([]), run=run)],
        allow_file=False,
    )
    monkeypatch.setattr("asymmetry.core.io.resolve_background_reference", lambda *a, **k: reference)
    config = MaxEntConfig(n_spectrum_points=32, f_min_mhz=0.2, f_max_mhz=3.0, auto_window=False)
    original = build_maxent_input(sample, config)
    sample.grouping["effective_detector_t0_bins"] = [0, 0]
    unchanged = build_maxent_input(sample, config)
    assert original.metadata["correction_outcomes"] == unchanged.metadata["correction_outcomes"]
    alignment = original.metadata["correction_outcomes"]["1"]["background"]["effective"][
        "reference_alignment"
    ]
    assert alignment["detector_offsets"] == [[0, -4]]
    assert alignment["timing_sources"] == [[0, "manual"]]


@pytest.mark.parametrize("consumer", ["fb", "spectral"])
def test_declared_auto_failure_never_falls_back_to_file(consumer, monkeypatch):
    sample, run = timing_runs()
    run.grouping["t0_policy"] = {"mode": "auto_detect"}
    reference = resolve_background_reference(
        {"run_number": 2},
        datasets=[MuonDataset(np.array([]), np.array([]), np.array([]), run=run)],
        allow_file=False,
    )
    with pytest.raises(ValueError, match="source|strategy"):
        reduce_reference(sample, reference, consumer, monkeypatch)


@pytest.mark.parametrize(
    "field,value",
    [("t0_bin", 3), ("t0_time_us", 0.043), ("bin_width", 0.032), ("counts", np.arange(65) + 10.0)],
)
def test_prepared_reference_grid_edits_invalidate_alignment(field, value, monkeypatch):
    sample, run = timing_runs()
    reference = BackgroundReference(run.histograms, 1.0)
    monkeypatch.setattr("asymmetry.core.io.resolve_background_reference", lambda *a, **k: reference)
    prepared = prepare_group_counts(sample, [1], use_deadtime=False)
    prepared.require_request(sample, 1, False, None, None)
    setattr(run.histograms[0], field, value)
    with pytest.raises(ValueError, match="match.*source"):
        prepared.require_request(sample, 1, False, None, None)


@pytest.mark.parametrize("consumer", ["fb", "spectral"])
def test_missing_reference_detector_is_contextual_error(consumer, monkeypatch):
    sample, run = timing_runs()
    reference = BackgroundReference(run.histograms[:1], 1.0)
    with pytest.raises(ValueError, match="Detector.*2.*not exist"):
        if consumer == "spectral":
            monkeypatch.setattr(
                "asymmetry.core.io.resolve_background_reference", lambda *a, **k: reference
            )
            prepare_group_counts(sample, [1, 2], use_deadtime=False)
        else:
            reduce_reference(sample, reference, consumer, monkeypatch)


def test_unknown_reference_policy_cannot_behave_as_file():
    from asymmetry.core.transform.reference_timing import ReferenceTiming

    with pytest.raises(ValueError, match="known timing policy"):
        ReferenceTiming(mode="typo")


def test_reference_auto_scans_once_for_cohort_and_accepts_cancellation(monkeypatch):
    from asymmetry.core.transform import t0

    sample, run = timing_runs()
    run.metadata["facility"] = "PSI"
    run.grouping["t0_policy"] = {"mode": "auto_detect"}
    for hist in run.histograms:
        hist.counts[6] = 1000
    reference = resolve_background_reference(
        {"run_number": 2},
        datasets=[MuonDataset(np.array([]), np.array([]), np.array([]), run=run)],
        allow_file=False,
    )
    monkeypatch.setattr("asymmetry.core.io.resolve_background_reference", lambda *a, **k: reference)
    searches = []
    original = t0.find_t0_for_run

    def search(*args, **kwargs):
        searches.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(t0, "find_t0_for_run", search)
    checks = []
    prepare_group_counts(
        sample, [1, 2], use_deadtime=False, cancel_check=lambda: checks.append(True)
    )
    assert len(searches) == 1
    assert len(checks) == 2

    class CancelledError(Exception):
        pass

    def cancel():
        raise CancelledError

    with pytest.raises(CancelledError):
        prepare_group_counts(sample, [1, 2], use_deadtime=False, cancel_check=cancel)


def test_automatic_alpha_uses_the_same_reference_timing():
    from asymmetry.core.project.profiles import (
        AlphaPolicy,
        BackgroundPolicy,
        GroupingProfile,
        ProfileFingerprint,
        resolve_effective_grouping,
    )
    from asymmetry.core.transform.reference_timing import ReferenceTiming

    sample, run = timing_runs()
    reference = BackgroundReference(
        run.histograms, 1.0, timing=ReferenceTiming(mode="manual", offset_bins=2)
    )
    profile = GroupingProfile(
        name="sample",
        fingerprint=ProfileFingerprint("", 2),
        groups={1: [1], 2: [2]},
        alpha_policy=AlphaPolicy(mode="per_run_estimate"),
        background_policy=BackgroundPolicy(
            mode="reference_run", details={"background_run": {"run_number": 2}}
        ),
    )
    effective = resolve_effective_grouping(profile, sample, reference_resolver=lambda _: reference)
    assert effective["alpha"] == pytest.approx(
        np.sum(186.0 - np.arange(32)) / np.sum(126.0 - np.arange(32))
    )


@pytest.mark.parametrize("mode", ["auto_detect", "from_file"])
def test_reference_timing_search_edits_outside_output_invalidate_preparation(mode, monkeypatch):
    sample, run = timing_runs()
    sample.histograms = [Histogram(np.full(8, x), 0.016) for x in (200.0, 150.0)]
    run.metadata["facility"] = "PSI"
    run.grouping["t0_policy"] = {"mode": mode}
    if mode == "from_file":
        run.grouping["detector_t0_sources"] = ["missing", "missing"]
    for hist in run.histograms:
        hist.counts[6] = 1000
    reference = resolve_background_reference(
        {"run_number": 2},
        datasets=[MuonDataset(np.array([]), np.array([]), np.array([]), run=run)],
        allow_file=False,
    )
    monkeypatch.setattr(
        "asymmetry.core.io.resolve_background_reference", lambda *args, **kwargs: reference
    )
    prepared = prepare_group_counts(sample, [1], use_deadtime=False)
    assert dict(prepared.layouts[1].reference_alignment.detector_offsets)[0] == -6
    run.histograms[0].counts[16] = 2000
    with pytest.raises(ValueError, match="match.*source"):
        prepared.require_request(sample, 1, False, None, None)
    fresh = prepare_group_counts(sample, [1], use_deadtime=False)
    assert dict(fresh.layouts[1].reference_alignment.detector_offsets)[0] == -16


def test_resolved_reference_snapshot_cannot_supply_unresolved_new_cohort(monkeypatch):
    from asymmetry.core.transform.grouping import resolve_alignment
    from asymmetry.core.transform.reference_timing import ReferenceTiming, ResolvedReferenceTiming

    sample, run = timing_runs()
    sample.histograms.append(Histogram(np.full(32, 200.0), 0.016))
    sample.grouping["groups"][3] = [3]
    counts = np.arange(64) + 30.0
    counts[8] = 1000
    run.histograms.append(Histogram(counts, 0.016, t0_bin=2))
    timing = ReferenceTiming(mode="auto_detect", file_sources=("file",) * 3, pulsed=False)
    grouping = timing.resolve(run.histograms, [[0], [1]])
    reference = BackgroundReference(
        run.histograms,
        1.0,
        timing=timing,
        resolved_timing=ResolvedReferenceTiming.from_grouping(run.histograms, grouping),
    )
    crop = resolve_alignment(sample.histograms, [2])
    with pytest.raises(ValueError, match="selected channels 3"):
        reference.alignment(sample.histograms, sample.grouping, crop, [2])
    monkeypatch.setattr("asymmetry.core.io.resolve_background_reference", lambda *a, **k: reference)
    prepared = prepare_group_counts(sample, [3], use_deadtime=False)
    assert prepared.backgrounds[3].forward.counts[0] == -800.0
