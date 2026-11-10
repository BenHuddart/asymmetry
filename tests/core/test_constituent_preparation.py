"""Acquired correction and observed-bin alignment through public preparation."""

from dataclasses import replace

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.io.periods import combine_mapped_periods, select_period
from asymmetry.core.transform.preparation import prepare_histogram_asymmetry, prepare_wizard_dataset


def acquired(number, forward, backward, *, frames=100000, tau=0.05, t0=0, first=0, last=None):
    values = [np.atleast_1d(v).astype(float) for v in (forward, backward)]
    histograms = [
        Histogram(
            v,
            0.1,
            t0_bin=t0,
            good_bin_start=first,
            good_bin_end=len(v) - 1 if last is None else last,
        )
        for v in values
    ]
    grouping = {
        "groups": {1: [1], 2: [2]},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.0,
        "good_frames": frames,
        "deadtime_file_us": [tau, tau],
        "dead_time_us": [tau, tau],
        "deadtime_correction": True,
        "deadtime_mode": "file",
        "t0_source": "file",
        "detector_t0_bins": [t0, t0],
        "detector_first_good_bins": [first, first],
        "detector_last_good_bins": [h.good_bin_end for h in histograms],
        "good_window_policy": {"mode": "from_file"},
    }
    run = Run(
        number,
        histograms=histograms,
        grouping=grouping,
        metadata={"run_number": number, "source_run_number": 10, "period_number": number},
    )
    return MuonDataset(
        time=np.zeros(1), asymmetry=np.zeros(1), error=np.ones(1), run=run, metadata=run.metadata
    )


def prepare(run, merge=1):
    return prepare_histogram_asymmetry(run, merge=merge, facility=None, reference=None)


@pytest.mark.parametrize(
    "frames,taus", [((100000, 100000), (0.05, 0.05)), ((80000, 150000), (0.02, 0.07))]
)
def test_correct_each_acquisition_before_red_pooling(frames, taus):
    counts = [(80000, 20000), (10000, 90000)]
    sources = [
        acquired(i + 1, *v, frames=f, tau=t)
        for i, (v, f, t) in enumerate(zip(counts, frames, taus, strict=True))
    ]
    mapped = combine_mapped_periods(sources, {1: "red", 2: "red"})
    corrected = np.array(
        [
            np.array(v) / (1 - np.array(v) * t / (0.1 * f))
            for v, f, t in zip(counts, frames, taus, strict=True)
        ]
    ).sum(axis=0)
    expected = 100 * (corrected[0] - corrected[1]) / corrected.sum()
    if frames == (100000, 100000):
        assert expected == pytest.approx(-12.737826507578193)
    assert prepare(mapped.run).asymmetry == pytest.approx([expected])
    assert prepare(select_period(mapped, 1).run).asymmetry == pytest.approx([expected])


def test_structured_constituents_align_own_t0_and_good_bins_in_replay():
    sources = [
        acquired(1, [11, 30, 70, 20, 13, 19], [10] * 6, t0=1, first=1, last=4),
        acquired(2, [3, 5, 30, 70, 20, 7], [10] * 6, t0=2, first=2, last=4),
    ]
    mapped = combine_mapped_periods(sources, {1: "red", 2: "red"})
    raw = replace(mapped.run, grouping=mapped.run.grouping | {"deadtime_correction": False})
    result = prepare(raw)
    expected_f = np.array([60, 140, 40])
    np.testing.assert_allclose(result.asymmetry, 100 * (expected_f - 20) / (expected_f + 20))
    np.testing.assert_allclose(result.time, [0, 0.1, 0.2], atol=1e-15)
    np.testing.assert_allclose(mapped.asymmetry, result.asymmetry)
    replay = prepare_wizard_dataset(mapped, 1)
    np.testing.assert_allclose(replay.asymmetry, result.asymmetry)


@pytest.mark.parametrize("mode", ["file", "manual", "estimate"])
def test_live_correction_policy_uses_each_acquired_exposure(mode):
    sources = [
        acquired(1, [8000] * 8, [2000] * 8, frames=10000, tau=0.01),
        acquired(2, [1000] * 8, [9000] * 8, frames=30000, tau=0.03),
    ]
    mapped = combine_mapped_periods(sources, {1: "red", 2: "red"})
    # Stale acquired manual values must not override current authored settings.
    for original in mapped.run.grouping["period_inputs"]:
        original.grouping.update(deadtime_mode="manual", dead_time_us=[0.09] * 2)
    mapped.run.grouping.update(deadtime_mode=mode, dead_time_us=[0.02, 0.04], alpha=0.7, beta=1.3)
    taus = [[0.01, 0.01], [0.03, 0.03]] if mode == "file" else [[0.02, 0.04]] * 2
    corrected = sum(
        np.array(v) / (1 - np.array(v) * np.array(t) / (0.1 * f))
        for v, t, f in zip([(8000, 2000), (1000, 9000)], taus, [10000, 30000], strict=True)
    )
    f, b = corrected
    result = prepare(mapped.run, 2)
    np.testing.assert_allclose(result.asymmetry, 100 * (f - 0.7 * b) / (1.3 * f + 0.7 * b))
    outcome = result.corrections["red"].deadtime
    assert outcome.status == "applied"
    for member, frames in zip(outcome.effective["constituents"], [10000, 30000], strict=True):
        assert member["effective"]["good_frames"] == frames


def test_missing_required_file_member_skips_every_set_but_ignore_does_not():
    sources = [acquired(i, [8000] * 8, [2000] * 8, frames=10000) for i in range(1, 5)]
    sources[1].run.grouping["deadtime_file_us"] = []
    sources[3].run.histograms[0].counts[:] = 1e10
    mapped = combine_mapped_periods(sources, {1: "red", 2: "red", 3: "green", 4: "ignore"})
    mapped.run.grouping["period_mode"] = "green_minus_red"
    with pytest.warns(UserWarning, match="File deadtime correction not applied"):
        result = prepare(mapped.run)
    np.testing.assert_allclose(result.asymmetry, 0)
    assert {o.deadtime.status for o in result.corrections.values()} == {"skipped"}
    mapped.run.grouping["period_mode"] = "green"
    assert prepare(mapped.run).corrections["green"].deadtime.status == "applied"


def test_required_domains_exclude_saturated_unused_raw_bins_and_ignored_acquisition():
    sources = [
        acquired(1, [1e9, 80, 60, 40, 1e9], [1e9, 20, 30, 40, 1e9], first=1, last=3),
        acquired(2, [1e9, 20, 30, 40, 1e9], [1e9, 80, 60, 40, 1e9], first=1, last=3),
        acquired(3, [1e10] * 5, [1e10] * 5),
    ]
    mapped = combine_mapped_periods(sources, {1: "red", 2: "red", 3: "ignore"})
    assert prepare(mapped.run).time.size == 3
    # A consumed saturated bin remains an error.
    mapped.run.grouping["period_inputs"][0].histograms[0].counts[2] = 1e10
    with pytest.raises(ValueError, match="occupancy|saturat|N·τ"):
        prepare(mapped.run)


def test_background_is_applied_once_after_pooling_with_retained_variance():
    sources = [acquired(i, [20] * 8, [10] * 8) for i in (1, 2)]
    for source, variance in zip(sources, (25, 49), strict=True):
        source.run.histograms = [
            replace(h, count_variance=np.full(8, variance)) for h in source.run.histograms
        ]
    mapped = combine_mapped_periods(sources, {1: "red", 2: "red"})
    mapped.run.grouping.update(
        deadtime_correction=False,
        background_correction=True,
        background_mode="fixed",
        background_fixed_values=[4, 2],
    )
    result = prepare(mapped.run, 2)
    np.testing.assert_allclose(result.asymmetry, 100 / 3)
    # F=72, B=36; independent VF=VB=148 for each packed bin.
    np.testing.assert_allclose(result.error, 200 * np.sqrt(36**2 * 148 + 72**2 * 148) / 108**2)


def test_fractional_residuals_and_selected_green_replay():
    sources = [
        acquired(1, [20] * 8, [10] * 8, t0=1),
        acquired(2, [30] * 8, [10] * 8, t0=2),
        acquired(3, [40] * 8, [10] * 8, t0=3),
    ]
    for source, residual in zip(sources, (0.01, -0.02, 0.03), strict=True):
        source.run.histograms = [
            replace(h, t0_time_us=(h.t0_bin + 0.5) * 0.1 + residual) for h in source.run.histograms
        ]
    mapped = combine_mapped_periods(sources, {1: "red", 2: "green", 3: "green"})
    mapped.run.grouping.update(period_mode="green", deadtime_correction=False)
    direct = prepare(mapped.run)
    selected = select_period(mapped, 2)
    extracted = prepare(selected.run)
    np.testing.assert_array_equal(direct.time, extracted.time)
    np.testing.assert_allclose(direct.asymmetry, extracted.asymmetry)
    assert direct.alignments["green"].timing_spread_us == pytest.approx(0.05)
    replay = prepare_wizard_dataset(selected, 2)
    merged = prepare(
        replace(selected.run, grouping=selected.histogram_observation.source.grouping), 2
    )
    np.testing.assert_allclose(replay.asymmetry, merged.asymmetry)


def test_equal_rates_have_the_pooled_ratio_limit():
    sources = [acquired(i, [60] * 8, [40] * 8, frames=1000) for i in (1, 2, 3)]
    mapped = combine_mapped_periods(sources, {1: "red", 2: "red", 3: "green"})
    mapped.run.grouping["period_mode"] = "green_plus_red"
    result = prepare(mapped.run)
    f, b = 60 / (1 - 60 * 0.05 / 100), 40 / (1 - 40 * 0.05 / 100)
    np.testing.assert_allclose(result.asymmetry, 100 * (f - b) / (f + b))


def test_mapped_preparation_cancels_before_publishing_any_partial_result():
    mapped = combine_mapped_periods(
        [acquired(i, [60] * 8, [40] * 8) for i in (1, 2)], {1: "red", 2: "red"}
    )
    before = [h.counts.copy() for r in mapped.run.grouping["period_inputs"] for h in r.histograms]
    calls = 0

    def cancel():
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError("cancelled constituent preparation")

    with pytest.raises(RuntimeError, match="cancelled constituent"):
        prepare_histogram_asymmetry(
            mapped.run, merge=1, facility=None, reference=None, cancel_check=cancel
        )
    for expected, h in zip(
        before, (h for r in mapped.run.grouping["period_inputs"] for h in r.histograms), strict=True
    ):
        np.testing.assert_array_equal(expected, h.counts)


@pytest.mark.parametrize("t0_mode", ["from_file", "manual", "auto_detect"])
def test_active_profile_resolves_original_clocks_without_held_search_cache(t0_mode, monkeypatch):
    from asymmetry.core.project.profiles import (
        T0Policy,
        profile_fingerprint_for_run,
        profile_from_payload,
        resolve_effective_grouping,
    )
    from asymmetry.core.transform.t0 import find_t0_for_run

    sources = [
        acquired(
            1, [10, 20, 500, 80, 60, 40, 30, 20], [10, 20, 400, 70, 50, 30, 20, 10], t0=2, first=3
        ),
        acquired(
            2, [10, 10, 20, 500, 80, 60, 40, 30], [10, 10, 20, 400, 70, 50, 30, 20], t0=3, first=4
        ),
    ]
    for source in sources:
        source.run.metadata["facility"] = "PSI"
    mapped = combine_mapped_periods(sources, {1: "red", 2: "red"})
    profile = profile_from_payload(
        mapped.run.grouping, "active", profile_fingerprint_for_run(mapped.run)
    )
    profile.t0_policy = T0Policy(mode=t0_mode, offset_bins=1 if t0_mode == "manual" else None)
    calls = []

    def own_search(histograms, metadata, **options):
        calls.append(histograms[0].t0_bin)
        return find_t0_for_run(histograms, metadata, **options)

    monkeypatch.setattr("asymmetry.core.transform.t0.find_t0_for_run", own_search)

    def held_search(*args, **kwargs):
        raise AssertionError("A held set cannot supply original acquired timing")

    grouping = resolve_effective_grouping(profile, mapped.run, timing_search=held_search)
    result = prepare(replace(mapped.run, grouping=grouping))
    assert result.time.size
    assert calls == ([2, 3] if t0_mode == "auto_detect" else [])
    assert result.alignments["red"].acquired_detectors == (
        (0, 1, 0),
        (1, 1, 1),
        (2, 2, 0),
        (3, 2, 1),
    )


@pytest.mark.parametrize("problem", ["width", "empty"])
def test_incompatible_combination_keeps_original_observations_inspectable(problem):
    sources = [acquired(1, [60] * 8, [40] * 8), acquired(2, [30] * 8, [20] * 8)]
    if problem == "width":
        sources[1].run.histograms = [replace(h, bin_width=0.2) for h in sources[1].run.histograms]
    else:
        sources[1].run.histograms = [replace(h, t0_bin=20) for h in sources[1].run.histograms]
    mapped = combine_mapped_periods(sources, {1: "red", 2: "red"})
    assert len(mapped.run.grouping["period_inputs"]) == 2
    assert all(h.counts.size == 8 for source in sources for h in source.run.histograms)
    with pytest.raises(ValueError, match="compatible|support"):
        prepare(mapped.run)


def test_workflow_extracted_green_uses_its_acquired_members():
    from asymmetry.core.workflow.reduction import ReductionSettings, reduce_run

    sources = [
        acquired(1, [60] * 8, [40] * 8),
        acquired(2, [80] * 8, [20] * 8),
        acquired(3, [90] * 8, [10] * 8),
    ]
    mapped = combine_mapped_periods(sources, {1: "red", 2: "green", 3: "green"})
    selected = select_period(mapped, 2)
    output = reduce_run(selected.run, ReductionSettings(deadtime="from_file", background="none"))
    np.testing.assert_allclose(output.asymmetry, prepare(selected.run).asymmetry)
    np.testing.assert_allclose(prepare_wizard_dataset(output, 2).asymmetry, output.asymmetry[:4])


@pytest.mark.parametrize("missing_reference", [False, True])
def test_mapped_reference_uses_own_clock_exposure_and_manual_scale(missing_reference):
    from asymmetry.core.transform.corrections import BackgroundReference
    from asymmetry.core.transform.reference_scale import ReferenceScaleRequest

    sources = [
        acquired(1, [8000] * 16, [2000] * 16, frames=10000, t0=1),
        acquired(2, [1000] * 16, [9000] * 16, frames=20000, t0=2),
    ]
    mapped = combine_mapped_periods(sources, {1: "red", 2: "red"})
    reference = BackgroundReference(
        [Histogram(np.arange(32) + x, 0.1, t0_bin=3) for x in (100, 200)],
        ReferenceScaleRequest("manual", 0.5).resolve(),
        run_number=99,
        period_number=4,
        deadtime_normaliser=40000,
        file_dead_time_us=() if missing_reference else (0.01, 0.02),
    )
    mapped.run.grouping.update(background_correction=True, background_mode="reference_run")
    if missing_reference:
        with pytest.warns(UserWarning, match="File deadtime correction not applied"):
            result = prepare_histogram_asymmetry(
                mapped.run, merge=1, facility=None, reference=reference
            )
    else:
        result = prepare_histogram_asymmetry(
            mapped.run, merge=1, facility=None, reference=reference
        )
    # On the sample's common grid, reference raw bins 2..16 are the matching observations.
    raw = np.arange(2, 17)
    f, b = (
        (9000, 11000)
        if missing_reference
        else sum(
            np.array(v) / (1 - np.array(v) * 0.05 / (0.1 * frames))
            for v, frames in [((8000, 2000), 10000), ((1000, 9000), 20000)]
        )
    )
    ref_f, ref_b = raw + 100, raw + 200
    if not missing_reference:
        ref_f = ref_f / (1 - ref_f * 0.01 / 4000)
        ref_b = ref_b / (1 - ref_b * 0.02 / 4000)
    f, b = f - 0.5 * ref_f, b - 0.5 * ref_b
    np.testing.assert_allclose(result.asymmetry, 100 * (f - b) / (f + b))
    outcome = result.corrections["red"]
    assert outcome.deadtime.status == ("skipped" if missing_reference else "applied")
    assert outcome.background.effective["reference"]["period_number"] == 4
    assert outcome.background.effective["reference"]["scale"] == 0.5


def test_missing_acquired_exposure_cannot_be_hidden_by_an_applied_first_member():
    from asymmetry.core.transform.corrections import CorrectionUnavailableError

    sources = [acquired(1, [8000] * 8, [2000] * 8), acquired(2, [1000] * 8, [9000] * 8)]
    sources[1].run.grouping.pop("good_frames")
    mapped = combine_mapped_periods(sources, {1: "red", 2: "red"})
    with pytest.raises(CorrectionUnavailableError, match="good frames"):
        prepare(mapped.run)


def test_corrected_pooled_ratio_retains_measured_supplied_variance():
    sources = [acquired(1, [80000] * 4, [20000] * 4), acquired(2, [10000] * 4, [90000] * 4)]
    for source, variance in zip(sources, [25, 49], strict=True):
        source.run.histograms = [
            replace(h, count_variance=np.full(4, variance)) for h in source.run.histograms
        ]
    mapped = combine_mapped_periods(sources, {1: "red", 2: "red"})
    f, b = (
        sum(
            np.array(v) / (1 - np.array(v) * 0.05 / 10000) for v in [(80000, 20000), (10000, 90000)]
        )
        * 2
    )
    result = prepare(mapped.run, 2)
    np.testing.assert_allclose(
        result.error, 200 * np.sqrt(b * b * 148 + f * f * 148) / (f + b) ** 2
    )


def test_projection_alpha_quicklook_matches_its_counts_first_replay():
    sources = [acquired(i, [60] * 8, [40] * 8) for i in (1, 2)]
    for source in sources:
        source.run.grouping.update(
            deadtime_correction=False,
            alpha=1.0,
            projections=[
                {"label": "custom", "forward_group": 1, "backward_group": 2, "alpha": 0.7}
            ],
            vector_axis="custom",
        )
    mapped = combine_mapped_periods(sources, {1: "red", 2: "red"})
    np.testing.assert_allclose(mapped.asymmetry, 100 * (60 - 0.7 * 40) / (60 + 0.7 * 40))
    np.testing.assert_allclose(prepare_wizard_dataset(mapped, 2).asymmetry, mapped.asymmetry[:4])


def test_mapped_reference_cannot_reuse_identical_acquisition_with_separate_buffers():
    from copy import deepcopy

    from asymmetry.core.transform.corrections import BackgroundReference
    from asymmetry.core.transform.reference_scale import ReferenceScaleRequest

    sources = [acquired(i, [60] * 8, [40] * 8) for i in (1, 2)]
    for source in sources:
        source.run.source_file = "same-source.nxs"
        source.run.grouping["deadtime_correction"] = False
    mapped = combine_mapped_periods(sources, {1: "red", 2: "red"})
    mapped.run.grouping.update(background_correction=True, background_mode="reference_run")
    reference = BackgroundReference(
        deepcopy(sources[0].run.histograms),
        ReferenceScaleRequest("manual", 1).resolve(),
        run_number=10,
        source_file="same-source.nxs",
        period_number=1,
    )
    with pytest.raises(ValueError, match="identical observations"):
        prepare_histogram_asymmetry(mapped.run, merge=1, facility=None, reference=reference)
    distinct = replace(reference, period_number=3)
    assert (
        prepare_histogram_asymmetry(
            mapped.run, merge=1, facility=None, reference=distinct
        ).time.size
        == 8
    )


def test_alignment_acquired_identity_restores_optional_json_provenance():
    import json
    from dataclasses import asdict

    from asymmetry.core.transform.fb_counts import AlignmentCrop

    alignment = prepare(
        combine_mapped_periods(
            [acquired(i, [60] * 8, [40] * 8) for i in (1, 2)], {1: "red", 2: "red"}
        ).run
    ).alignments["red"]
    payload = json.loads(json.dumps(asdict(alignment)))
    for key in ("timing_residuals_us", "timing_sources"):
        payload[key] = tuple(tuple(v) for v in payload[key])
    assert AlignmentCrop(**payload) == alignment
    payload.pop("acquired_detectors")
    assert AlignmentCrop(**payload).acquired_detectors == ()


@pytest.mark.parametrize("fact", ["clock", "good_bins", "frames", "coefficient", "selection"])
def test_fourier_and_projection_reuse_identify_acquired_preparation(fact):
    from asymmetry.core.fourier.spectrum import fourier_grouping_digest
    from asymmetry.core.transform.projections import reduction_identity

    mapped = combine_mapped_periods(
        [acquired(i, [60] * 8, [40] * 8) for i in (1, 2, 3)], {1: "red", 2: "red", 3: "green"}
    )
    before = (
        fourier_grouping_digest(mapped.run, "fb_asymmetry"),
        reduction_identity(mapped.run.grouping),
    )
    original = mapped.run.grouping["period_inputs"][1]
    if fact == "clock":
        original.histograms[0].t0_bin += 1
    elif fact == "good_bins":
        original.histograms[0].good_bin_start += 1
    elif fact == "frames":
        original.grouping["good_frames"] *= 2
    elif fact == "coefficient":
        original.grouping["deadtime_file_us"] = [0.01, 0.02]
    else:
        mapped.run.grouping["period_mode"] = "green"
    after = (
        fourier_grouping_digest(mapped.run, "fb_asymmetry"),
        reduction_identity(mapped.run.grouping),
    )
    assert all(a != b for a, b in zip(before, after, strict=True))


def test_reuse_metadata_does_not_copy_or_hash_original_count_buffers(monkeypatch):
    from asymmetry.core.fourier.spectrum import fourier_grouping_digest
    from asymmetry.core.transform.projections import reduction_identity

    mapped = combine_mapped_periods(
        [acquired(i, [60] * 8, [40] * 8) for i in (1, 2)], {1: "red", 2: "red"}
    )

    def count_scan(*args, **kwargs):
        raise AssertionError("Grouping identity must not scan original count buffers")

    monkeypatch.setattr("asymmetry.core.transform.projections.np.asarray", count_scan)
    assert reduction_identity(mapped.run.grouping)
    assert fourier_grouping_digest(mapped.run, "fb_asymmetry")


def test_mapping_ordinals_do_not_rename_evidenced_acquired_data_periods():
    sources = [acquired(i, [60] * 8, [40] * 8) for i in (1, 2)]
    for source, data_period, dae_period in zip(sources, (5, 3), (8, 4), strict=True):
        source.run.metadata.update(data_period_number=data_period, dae_period_number=dae_period)
    mapped = combine_mapped_periods(sources, {1: "red", 2: "red"})
    result = prepare(mapped.run)
    assert [
        item["data_period_number"]
        for item in result.corrections["red"].deadtime.effective["constituents"]
    ] == [5, 3]
    assert result.alignments["red"].acquired_detectors == (
        (0, 5, 0),
        (1, 5, 1),
        (2, 3, 0),
        (3, 3, 1),
    )


@pytest.mark.parametrize("metadata", [None, {}])
def test_quicklook_manual_exact_origin_and_support_do_not_depend_on_optional_metadata(metadata):
    from asymmetry.core.io.quick_asymmetry import quick_asymmetry

    histograms = [
        Histogram(np.full(8, value), 0.1, t0_bin=2, t0_time_us=0.25) for value in (60, 40)
    ]
    quick = quick_asymmetry(
        histograms,
        {"groups": {1: [1], 2: [2]}, "t0_policy": {"mode": "manual", "offset_bins": 2}},
        metadata=metadata,
    )
    assert quick.unavailable_reason is None
    assert quick.common_t0 == 4
    assert quick.t0_time_us == pytest.approx(0.45)
    assert quick.file_grouping["t0_time_us"] == pytest.approx(0.45)
    assert (quick.window.first, quick.window.last) == (0, 7)
    np.testing.assert_allclose(quick.time, (np.arange(8) + 0.5) * 0.1 - 0.45)


def test_quicklook_explicit_auto_without_source_evidence_stays_unavailable():
    from asymmetry.core.io.quick_asymmetry import quick_asymmetry

    quick = quick_asymmetry(
        [Histogram(np.arange(8) + value, 0.1) for value in (60, 40)],
        {"groups": {1: [1], 2: [2]}, "t0_policy": {"mode": "auto_detect"}},
    )
    assert quick.unavailable_reason is not None
    assert "known source" in quick.unavailable_reason


def test_mapped_timing_cannot_bypass_the_project_legacy_policy_guard():
    from asymmetry.core.project.profiles import (
        T0Policy,
        profile_fingerprint_for_run,
        profile_from_payload,
        resolve_effective_grouping,
    )

    mapped = combine_mapped_periods(
        [acquired(i, [60] * 8, [40] * 8) for i in (1, 2)], {1: "red", 2: "red"}
    )
    profile = profile_from_payload(
        mapped.run.grouping, "legacy", profile_fingerprint_for_run(mapped.run)
    )
    profile.t0_policy = T0Policy(mode="manual", legacy_value=6)
    with pytest.raises(ValueError, match="pre-v21 absolute manual t0"):
        resolve_effective_grouping(profile, mapped.run)
