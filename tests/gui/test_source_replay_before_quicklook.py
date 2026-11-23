"""Project load purposes carry captured source authority into real file quick-looks."""

from copy import deepcopy

import h5py
import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from asymmetry.core.data.acquisition import SourceOwner
from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.instruments.resolution import InstrumentSelection, RecordedInstrumentChoice
from asymmetry.core.instruments.source import SourceAssessment
from asymmetry.core.io import load_background_run
from asymmetry.core.io.base import BaseLoader, LoaderRegistry
from asymmetry.core.io.run_preparation import ReplayState
from asymmetry.core.project import (
    CURRENT_SCHEMA_VERSION,
    BackgroundPolicy,
    GroupingProfile,
    ProfileFingerprint,
    load_project,
    prepare_project,
    save_project,
    verify_project_sources,
)
from asymmetry.gui.mainwindow import MainWindow
from tests.io.test_load_source_defaults import source_file
from tests.io.test_nexus_grouping_declarations import grouping_file

pytestmark = pytest.mark.gui

UNKNOWN = SourceAssessment(None, "absent", "project")
CONTINUOUS = SourceAssessment(False, "known", "project")


@pytest.fixture
def window(qapp, monkeypatch):
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: pytest.fail(str(args[-1])))
    instance = MainWindow()
    yield instance
    instance._clear_dirty()
    instance.close()


@pytest.fixture
def quick_looks(monkeypatch):
    import asymmetry.core.io.nexus as nexus

    original = nexus.quick_asymmetry
    seen = []

    def observe(histograms, grouping, *, metadata=None, **kwargs):
        result = original(histograms, grouping, metadata=metadata, **kwargs)
        seen.append(
            {
                "run_number": metadata["run_number"],
                "source": metadata["source_assessment"],
                "groups": deepcopy(grouping.get("groups")),
                "pair": (grouping.get("forward_group"), grouping.get("backward_group")),
                "preset": grouping.get("grouping_preset"),
                "points": len(result.asymmetry),
            }
        )
        return result

    monkeypatch.setattr(nexus, "quick_asymmetry", observe)
    return seen


def sample_entry(path, *, number=42, source=None, grouping=None):
    entry = {"run_number": number, "source_file": str(path)}
    if source is not None:
        entry["source_assessment"] = source.to_dict()
    if grouping is not None:
        entry["grouping_overrides"] = deepcopy(grouping)
    return entry


def period_file(path, count=3):
    raw = source_file(path)
    with h5py.File(path, "r+") as handle:
        detector = handle["raw_data_1/instrument/detector_1"]
        del detector["counts"]
        detector.create_dataset(
            "counts", data=np.stack([(index + 1) * raw for index in range(count)])
        )
    return raw


@pytest.mark.parametrize(
    "captured", [None, UNKNOWN, CONTINUOUS, SourceAssessment(True, "known", "project")]
)
def test_project_sample_capture_reaches_decoder_before_quicklook(
    window, tmp_path, quick_looks, captured
):
    path = tmp_path / "sample.nxs"
    raw = source_file(path)
    choice = RecordedInstrumentChoice(
        InstrumentSelection("isis.emu", "nexus96", "old", "project"), ()
    )
    entry = sample_entry(path, source=captured)
    entry["instrument_choice"] = choice.to_dict()
    window.restore_project_state(
        {"schema_version": CURRENT_SCHEMA_VERSION, "datasets": [entry]},
        str(tmp_path / "sample.asymp"),
    )
    expected = captured or SourceAssessment(None, "absent", "legacy:metadata")
    assert [item["source"] for item in quick_looks] == [expected]
    dataset = window._data_browser.get_dataset(42)
    assert dataset is not None
    assert dataset.run.source_assessment == expected
    assert dataset.run.instrument_choice == choice
    assert (dataset.n_points == 0) == (expected.pulsed is None)
    for actual, wanted in zip(dataset.run.histograms, raw, strict=True):
        np.testing.assert_array_equal(actual.counts, wanted)
    assert "source_pulsed" not in dataset.run.metadata


@pytest.mark.parametrize("captured", [None, UNKNOWN])
def test_legacy_or_unknown_project_does_not_apply_new_absent_musr_grouping(
    window, tmp_path, quick_looks, captured
):
    path = tmp_path / "musr.nxs"
    grouping_file(path, "v2")
    custom = {
        "groups": {1: list(range(1, 33)), 2: list(range(33, 65))},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 0.75,
        "scientific_extension": {"retained": 0.93},
    }
    entry = sample_entry(path, source=captured, grouping=custom)
    window.restore_project_state(
        {"schema_version": CURRENT_SCHEMA_VERSION, "datasets": [entry]},
        str(tmp_path / "musr.asymp"),
    )
    assert len(quick_looks) == 1
    assert quick_looks[0]["preset"] is None
    assert quick_looks[0]["pair"] == (1, 2)
    dataset = window._data_browser.get_dataset(42)
    assert dataset is not None
    for key, value in custom.items():
        assert dataset.run.grouping[key] == value
    np.testing.assert_allclose(
        dataset.asymmetry, 100 * (32000 - 0.75 * 96000) / (32000 + 0.75 * 96000), rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("number", [42, 42001])
@pytest.mark.parametrize("periods", [2, 3])
def test_profile_reference_verification_loads_its_sample_with_saved_period_source(
    tmp_path, quick_looks, number, periods
):
    path = tmp_path / "periods.nxs"
    period_file(path, periods)
    profile = GroupingProfile(
        "Saved",
        ProfileFingerprint("isis.emu", 96, "nexus96"),
        background_policy=BackgroundPolicy(
            "reference_run",
            {"background_run": {"source_file": str(path), "run_number": 42, "period_number": 1}},
        ),
    )
    entry = sample_entry(path, number=number, source=CONTINUOUS)
    entry["profile"] = profile.name
    prepared = prepare_project(
        {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "datasets": [entry],
            "grouping_profiles": [profile.to_dict()],
        }
    )
    verified = verify_project_sources(prepared, tmp_path / "verify.asymp")
    assert len(quick_looks) == periods
    for item in quick_looks:
        expected = (
            CONTINUOUS
            if number == 42 or item["run_number"] == number
            else SourceAssessment(None, "absent", "legacy:metadata")
        )
        assert item["source"] == expected
    assert any(f.source.role == "reference" and f.source.owner.active for f in verified.facts)


@pytest.mark.parametrize("periods", [2, 3])
def test_same_file_period_owners_keep_distinct_captures_and_authored_requests(
    window, tmp_path, quick_looks, periods
):
    path = tmp_path / "periods.nxs"
    period_file(path, periods)
    entries = []
    for number, capture, value in ((42001, UNKNOWN, 0.8), (42002, CONTINUOUS, 1.2)):
        entries.append(
            sample_entry(
                path,
                number=number,
                source=capture,
                grouping={
                    "groups": {1: [1], 2: [49]},
                    "forward_group": 1,
                    "backward_group": 2,
                    "alpha": value,
                    "scientific_extension": {"value": value},
                },
            )
        )
    window.restore_project_state(
        {"schema_version": CURRENT_SCHEMA_VERSION, "datasets": entries},
        str(tmp_path / "periods.asymp"),
    )
    assert len(quick_looks) == 2 * periods
    for items, number, capture in (
        (quick_looks[:periods], 42001, UNKNOWN),
        (quick_looks[periods:], 42002, CONTINUOUS),
    ):
        assert {item["run_number"] for item in items} == set(range(42001, 42001 + periods))
        for item in items:
            assert item["source"] == (
                capture
                if item["run_number"] == number
                else SourceAssessment(None, "absent", "legacy:metadata")
            )
    first, second = (window._data_browser.get_dataset(n) for n in (42001, 42002))
    assert first is not None and second is not None
    assert first.run.source_assessment == UNKNOWN
    assert second.run.source_assessment == CONTINUOUS
    first.run.grouping["scientific_extension"]["value"] = 99
    assert second.run.grouping["scientific_extension"] == {"value": 1.2}
    assert first.run is not second.run


def test_equal_replay_states_on_distinct_owners_do_not_share_mutable_runs(
    window, tmp_path, quick_looks
):
    path = tmp_path / "sample.nxs"
    source_file(path, file_anchor=2)
    replay = ReplayState(source=UNKNOWN, run_number=42)
    requests = [(SourceOwner(("datasets", index), True), str(path), replay) for index in (0, 1)]
    loaded = window._load_paths_with_progress(requests)
    assert set(loaded) == set(requests)
    first, second = (loaded[request] for request in requests)
    assert first.run is not second.run
    first.run.grouping["groups"][1].append(96)
    assert 96 not in second.run.grouping["groups"][1]
    assert [item["source"] for item in quick_looks] == [UNKNOWN, UNKNOWN]


def test_same_file_sample_and_reference_preserve_their_independent_captures(
    window, tmp_path, quick_looks
):
    path = tmp_path / "sample.nxs"
    source_file(path)
    entry = sample_entry(path, source=UNKNOWN)
    window.restore_project_state(
        {"schema_version": CURRENT_SCHEMA_VERSION, "datasets": [entry]},
        str(tmp_path / "sample.asymp"),
    )
    sample = window._data_browser.get_dataset(42)
    assert sample is not None
    reference = load_background_run(
        {
            "source_file": str(path),
            "run_number": 42,
            "period_number": 1,
            "timing": {"t0_policy": {"mode": "from_file"}, "source": CONTINUOUS.to_dict()},
        }
    )
    assert [item["source"] for item in quick_looks] == [UNKNOWN, CONTINUOUS]
    assert sample.run.source_assessment == UNKNOWN
    assert reference.run.source_assessment == CONTINUOUS
    reference.run.grouping["groups"][1].append(96)
    assert 96 not in sample.run.grouping["groups"][1]
    assert sample.run is not reference.run


def test_unmatched_multiperiod_owner_is_retained_without_borrowing_a_base_run(
    window, tmp_path, quick_looks
):
    path = tmp_path / "periods.nxs"
    period_file(path)
    entry = sample_entry(path, number=42004, source=CONTINUOUS)
    window.restore_project_state(
        {"schema_version": CURRENT_SCHEMA_VERSION, "datasets": [entry]},
        str(tmp_path / "unmatched.asymp"),
    )
    assert window._data_browser.get_dataset(42004) is None
    assert len(quick_looks) == 3
    assert all(
        item["source"] == SourceAssessment(None, "absent", "legacy:metadata")
        for item in quick_looks
    )
    saved = next(
        row for row in window.collect_project_state()["datasets"] if row["run_number"] == 42004
    )
    assert saved["source_assessment"] == CONTINUOUS.to_dict()
    assert saved["source_file"] == str(path)


def test_two_period_base_owner_keeps_captured_source_for_both_acquisitions(
    window, tmp_path, quick_looks
):
    path = tmp_path / "paired.nxs"
    period_file(path, 2)
    window.restore_project_state(
        {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "datasets": [sample_entry(path, source=UNKNOWN)],
        },
        str(tmp_path / "paired.asymp"),
    )
    dataset = window._data_browser.get_dataset(42)
    assert dataset is not None
    assert dataset.run.run_number == 42
    assert dataset.run.source_assessment == UNKNOWN
    assert [item["source"] for item in quick_looks] == [UNKNOWN, UNKNOWN]
    assert len(dataset.run.grouping["period_inputs"]) == 2
    assert all(run.source_assessment == UNKNOWN for run in dataset.run.grouping["period_inputs"])


@pytest.fixture(params=["filepath", "kwargs"])
def legacy_plugin(monkeypatch, request):
    class LegacyLoader(BaseLoader):
        extensions = [".oldplugin"]
        format_name = "Legacy decoder control"

        def load(self, filepath):
            from pathlib import Path

            metadata = {"run_number": 42, "instrument": "EMU", "source_file": filepath}
            value = Path(filepath).read_text()
            if value != "unknown":
                metadata["source_pulsed"] = value == "pulsed"
            run = Run(
                run_number=42,
                source_file=filepath,
                metadata=deepcopy(metadata),
                histograms=[
                    Histogram(np.arange(8, dtype=float) + i + 100, 0.01, 2, 2, 7) for i in range(96)
                ],
                grouping={
                    "groups": {1: [1], 2: [49]},
                    "forward_group": 1,
                    "backward_group": 2,
                    "t0_bin": 2,
                    "first_good_bin": 2,
                    "last_good_bin": 7,
                },
            )
            forward = np.arange(102, 108, dtype=float)
            backward = np.arange(150, 156, dtype=float)
            total = forward + backward
            return MuonDataset(
                np.arange(6) * 0.01,
                100 * (forward - backward) / total,
                100 * np.sqrt(4 * forward * backward / total**3),
                metadata=metadata,
                run=run,
            )

    class KwargsLegacyLoader(LegacyLoader):
        def load(self, filepath, **kwargs):
            return super().load(filepath)

    loader = LegacyLoader if request.param == "filepath" else KwargsLegacyLoader
    monkeypatch.setattr(
        LoaderRegistry,
        "_loaders",
        dict(LoaderRegistry._loaders, **{".oldplugin": loader}),
    )


@pytest.mark.parametrize("raw_source", ["unknown", "continuous", "pulsed"])
@pytest.mark.parametrize("fresh", [False, True])
def test_actual_legacy_plugin_project_reopens_captured_and_omitted_source(
    window, tmp_path, legacy_plugin, raw_source, fresh
):
    path = tmp_path / "sample.oldplugin"
    path.write_text(raw_source)
    if fresh:
        window._load_files([str(path)])
    else:
        window.restore_project_state(
            {"schema_version": CURRENT_SCHEMA_VERSION, "datasets": [sample_entry(path)]},
            str(tmp_path / "input.asymp"),
        )
    dataset = window._data_browser.get_dataset(42)
    assert dataset is not None
    expected_value = {"unknown": None, "continuous": False, "pulsed": True}[raw_source]
    assert dataset.run.source_assessment.pulsed is expected_value
    assert dataset.run.source_assessment.origin != "profile"
    arrays = [a.copy() for a in (dataset.time, dataset.asymmetry, dataset.error)]
    raw_counts = [h.counts.copy() for h in dataset.run.histograms]
    capture = dataset.run.source_assessment
    project = tmp_path / "saved.asymp"
    save_project(window.collect_project_state(), project)
    state = load_project(project)
    window.restore_project_state(state, str(project))
    reopened = window._data_browser.get_dataset(42)
    assert reopened is not None
    assert reopened.run.source_assessment == capture
    for actual, expected in zip(
        (reopened.time, reopened.asymmetry, reopened.error), arrays, strict=True
    ):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    for histogram, expected in zip(reopened.run.histograms, raw_counts, strict=True):
        np.testing.assert_array_equal(histogram.counts, expected)


def test_missing_source_preserves_captured_unknown_and_custom_request(
    window, tmp_path, monkeypatch
):
    monkeypatch.setattr(QMessageBox, "exec", lambda box: QMessageBox.StandardButton.No)
    path = tmp_path / "missing.nxs"
    custom = {"groups": {7: [1, 2]}, "alpha": 0.73, "scientific_extension": {"retained": [0.91]}}
    entry = sample_entry(path, source=UNKNOWN, grouping=custom)
    window.restore_project_state(
        {"schema_version": CURRENT_SCHEMA_VERSION, "datasets": [entry]},
        str(tmp_path / "missing.asymp"),
    )
    assert window._data_browser.get_dataset(42) is None
    saved = window.collect_project_state()["datasets"][0]
    assert saved["source_assessment"] == UNKNOWN.to_dict()
    assert saved["grouping_overrides"] == custom
    assert saved["source_file"] == str(path)


def test_legacy_plugin_equal_decision_restores_historical_provenance(
    window, tmp_path, legacy_plugin
):
    path = tmp_path / "historical.oldplugin"
    path.write_text("continuous")
    capture = SourceAssessment(False, "known", "profile", "isis.emu", "historical")
    choice = RecordedInstrumentChoice(
        InstrumentSelection("isis.emu", "nexus96", "historical", "project"), ()
    )
    entry = sample_entry(path, source=capture)
    entry["instrument_choice"] = choice.to_dict()
    window.restore_project_state(
        {"schema_version": CURRENT_SCHEMA_VERSION, "datasets": [entry]},
        str(tmp_path / "historical.asymp"),
    )
    dataset = window._data_browser.get_dataset(42)
    assert dataset is not None
    assert dataset.run.source_assessment == capture
    assert dataset.run.instrument_choice == choice
    np.testing.assert_allclose(
        dataset.asymmetry, 100 * (-48) / np.arange(252, 264, 2), rtol=0, atol=1e-12
    )
    saved = window.collect_project_state()["datasets"][0]
    assert saved["source_assessment"] == capture.to_dict()
    assert saved["instrument_choice"] == choice.to_dict()


@pytest.mark.parametrize("contrary", ["source", "identity"])
def test_legacy_plugin_cannot_replay_a_different_scientific_decision(
    window, tmp_path, legacy_plugin, contrary
):
    path = tmp_path / "contrary.oldplugin"
    path.write_text("continuous")
    entry = sample_entry(path, source=SourceAssessment(contrary == "source", "known", "project"))
    if contrary == "identity":
        entry["instrument_choice"] = RecordedInstrumentChoice(
            InstrumentSelection("psi.gps", "bin_combined", "historical", "project"), ()
        ).to_dict()
    window.restore_project_state(
        {"schema_version": CURRENT_SCHEMA_VERSION, "datasets": [entry]},
        str(tmp_path / "contrary.asymp"),
    )
    assert window._data_browser.get_dataset(42) is None
    saved = window.collect_project_state()["datasets"][0]
    assert saved["source_assessment"] == entry["source_assessment"]
    if contrary == "identity":
        assert saved["instrument_choice"] == entry["instrument_choice"]
