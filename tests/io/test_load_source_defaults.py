"""Captured source authority reaches actual file quick-looks before timing search."""

import h5py
import numpy as np
import pytest

from asymmetry.core.instruments.source import SourceAssessment
from asymmetry.core.io import load
from asymmetry.core.io.run_preparation import ReplayState

pytestmark = pytest.mark.io


def source_file(path, *, file_anchor=None):
    pulse = np.array(
        [
            0,
            0,
            0,
            0,
            0,
            10,
            30,
            60,
            90,
            100,
            95,
            90,
            85,
            80,
            75,
            70,
            65,
            60,
            55,
            50,
            45,
            40,
            35,
            30,
        ],
        dtype=float,
    )
    raw = np.vstack([3 * pulse] * 48 + [pulse] * 48)
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("raw_data_1")
        entry.create_dataset("definition", data=b"muonTD")
        entry.create_dataset("run_number", data=42)
        entry.create_dataset("name", data=b"EMU")
        detector = entry.create_group("instrument/detector_1")
        detector.create_dataset("counts", data=raw)
        detector.create_dataset("grouping", data=[1] * 48 + [2] * 48)
        detector.create_dataset("resolution", data=20).attrs["units"] = "ns"
        detector.create_dataset("raw_time", data=np.arange(25) * 0.02).attrs["units"] = "us"
        detector.create_dataset("first_good_time", data=0.2).attrs["units"] = "us"
        detector.create_dataset("last_good_time", data=0.46).attrs["units"] = "us"
        if file_anchor is not None:
            detector.create_dataset("time_zero", data=file_anchor * 0.02).attrs["units"] = "us"
    return raw


def test_fresh_instrument_only_load_uses_profile_source_before_missing_timing_search(tmp_path):
    path = tmp_path / "run.nxs"
    raw = source_file(path)
    dataset = load(path)
    assert dataset.n_points > 0
    assert dataset.run.source_assessment == SourceAssessment(
        True, "known", "profile", "isis.emu", "1"
    )
    assert dataset.run.grouping["t0_bin"] == 7
    np.testing.assert_allclose(dataset.asymmetry, 50.0, rtol=0, atol=1e-12)
    for actual, expected in zip(dataset.run.histograms, raw, strict=True):
        np.testing.assert_array_equal(actual.counts, expected)
    assert "source_pulsed" not in dataset.run.metadata


@pytest.mark.parametrize("captured", [None, SourceAssessment(None, "absent", "project")])
def test_actual_replay_does_not_enable_new_profile_search_before_restoring_unknown(
    tmp_path, captured
):
    path = tmp_path / "run.nxs"
    raw = source_file(path)
    dataset = load(path, replay=ReplayState(source=captured))
    assert dataset.n_points == 0
    assert dataset.run.source_assessment.pulsed is None
    assert dataset.run.source_assessment.origin == (
        "legacy:metadata" if captured is None else "project"
    )
    assert "known source" in dataset.metadata["correction_error"]
    for actual, expected in zip(dataset.run.histograms, raw, strict=True):
        np.testing.assert_array_equal(actual.counts, expected)


@pytest.mark.parametrize("captured", [None, SourceAssessment(None, "absent", "project")])
def test_unknown_replay_source_keeps_usable_file_timing(tmp_path, captured):
    path = tmp_path / "run.nxs"
    source_file(path, file_anchor=2)
    dataset = load(path, replay=ReplayState(source=captured))
    assert dataset.n_points > 0
    assert dataset.run.grouping["t0_bin"] == 2
    assert dataset.run.source_assessment.pulsed is None
    np.testing.assert_allclose(dataset.asymmetry, 50.0, rtol=0, atol=1e-12)


@pytest.mark.parametrize("pulsed", [None, False])
def test_selected_second_period_retains_its_own_capture(tmp_path, pulsed):
    path = tmp_path / "run.nxs"
    raw = source_file(path, file_anchor=2)
    with h5py.File(path, "r+") as handle:
        detector = handle["raw_data_1/instrument/detector_1"]
        del detector["counts"]
        detector.create_dataset("counts", data=np.stack([raw, raw]))
    captured = SourceAssessment(pulsed, "absent" if pulsed is None else "known", "project")
    dataset = load(
        path, period=2, replay=ReplayState(source=captured, run_number=42, period_number=2)
    )
    assert dataset.run.source_assessment == captured
    assert dataset.histogram_observation.source.source_assessment == captured
    assert dataset.run.grouping["period_inputs"][1].source_assessment == captured
    assert dataset.run.metadata["period_number"] == 2
    np.testing.assert_allclose(dataset.asymmetry, 50.0, rtol=0, atol=1e-12)


@pytest.mark.parametrize("number,period", [(42001, 1), (42002, 2)])
def test_encoded_replay_owner_selects_its_actual_two_period_run(tmp_path, number, period):
    path = tmp_path / "run.nxs"
    raw = source_file(path, file_anchor=2)
    with h5py.File(path, "r+") as handle:
        detector = handle["raw_data_1/instrument/detector_1"]
        del detector["counts"]
        detector.create_dataset("counts", data=np.stack([raw, raw]))
    captured = SourceAssessment(False, "known", "project")
    dataset = load(path, replay=ReplayState(source=captured, run_number=number))
    assert dataset.run.run_number == number
    assert dataset.run.metadata["period_number"] == period
    assert dataset.run.source_assessment == captured
    assert dataset.histogram_observation.source.source_assessment == captured


@pytest.mark.parametrize("capture", ["profile", "unknown", "legacy", "legacy_omitted"])
@pytest.mark.parametrize("through_refresh", [False, True])
def test_actual_live_reference_captures_effective_source_before_missing_anchor_search(
    tmp_path, capture, through_refresh
):
    from asymmetry.core.io import resolve_background_reference
    from asymmetry.core.transform.reference_scale import ReferenceExposure

    path = tmp_path / "run.nxs"
    source_file(path)
    dataset = load(path)
    if capture == "unknown":
        dataset.run.source_assessment = SourceAssessment(None, "absent", "project")
    payload = {"source_file": str(path), "run_number": 42}
    if capture == "legacy":
        payload["timing"] = {"t0_policy": {"mode": "from_file"}}
    elif capture == "legacy_omitted":
        payload["timing"] = None
    if through_refresh:
        from asymmetry.core.io.reference_selection import refresh_reference_timing_requests

        grouping = {"background_mode": "reference_run", "background_run": payload}
        refresh_reference_timing_requests([dataset], groupings=[grouping])
    reference = resolve_background_reference(
        payload, datasets=[dataset], allow_file=False, sample_exposure=ReferenceExposure()
    )
    expected = (
        SourceAssessment(None, "absent", "legacy:metadata")
        if capture.startswith("legacy")
        else dataset.run.source_assessment
    )
    assert reference.timing.source == expected
    assert payload["timing"]["source"] == expected.to_dict()
    if capture == "profile":
        resolved = reference.timing.resolve(reference.histograms, ([0], [48]))
        assert resolved["effective_detector_t0_bins"][0] == 7
    else:
        with pytest.raises(ValueError, match="known source"):
            reference.timing.resolve(reference.histograms, ([0], [48]))


@pytest.mark.parametrize("timing", ["auto_detect", "missing", "file"])
@pytest.mark.parametrize("changed_source", [None, False])
@pytest.mark.parametrize("authority", ["capture", "raw_metadata"])
def test_combined_fourier_caches_own_each_consumed_timing_source(
    tmp_path, timing, changed_source, authority
):
    from asymmetry.core.data.combine import combine_runs
    from asymmetry.core.fourier.group_preparation import prepare_group_counts
    from asymmetry.core.fourier.grouping_identity import fourier_grouping_digest

    runs = []
    for number in (42, 43):
        path = tmp_path / f"run{number}.nxs"
        source_file(path, file_anchor=None if timing == "missing" else 2)
        with h5py.File(path, "r+") as handle:
            handle["raw_data_1/run_number"][()] = number
            handle["raw_data_1/instrument/detector_1/last_good_time"][()] = 0.30
        run = load(path).run
        if authority == "raw_metadata":
            run.source_assessment = None
        runs.append(run)
    combined = combine_runs(runs)
    combined.grouping["t0_policy"] = {
        "mode": "auto_detect" if timing == "auto_detect" else "from_file"
    }
    before = fourier_grouping_digest(combined)
    prepared = prepare_group_counts(combined, [1])
    original = combined.grouping["period_inputs"][1]
    if authority == "capture":
        original.source_assessment = SourceAssessment(
            changed_source, "absent" if changed_source is None else "known", "project"
        )
    else:
        original.metadata["source_pulsed"] = changed_source
        if changed_source is None:
            original.metadata["source"] = "unsupported"
    after = fourier_grouping_digest(combined)
    if timing == "file":
        assert before == after
        prepared.require_request(combined, 1, False, None, None)
        assert prepare_group_counts(combined, [1]).reference_t0_bin == prepared.reference_t0_bin
    else:
        assert before != after
        with pytest.raises(ValueError, match="Prepared grouped counts"):
            prepared.require_request(combined, 1, False, None, None)
        if changed_source is None:
            with pytest.raises(ValueError, match="known source"):
                prepare_group_counts(combined, [1])
        else:
            assert prepare_group_counts(combined, [1]).reference_t0_bin != prepared.reference_t0_bin
