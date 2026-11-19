"""MainWindow identity decisions retain authored requests independently of fresh evidence."""

from copy import deepcopy

import numpy as np
import pytest
from PySide6.QtWidgets import QDialog

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.instruments.resolution import (
    InstrumentSelection,
    RecordedInstrumentChoice,
    accept_instrument_choice,
    resolve_instrument,
)
from asymmetry.core.instruments.source import SourceAssessment
from asymmetry.core.project import CURRENT_SCHEMA_VERSION, load_project, save_project
from asymmetry.core.project.profiles import ProfileFingerprint, profile_from_payload
from asymmetry.gui.mainwindow import MainWindow
from asymmetry.gui.windows.instrument_identity_dialog import InstrumentIdentityDialog

pytestmark = pytest.mark.gui


def saved_request():
    return {
        "groups": {1: [1, 3], 7: [2, 4]},
        "group_names": {1: "Saved positive", 7: "Saved negative"},
        "forward_group": 1,
        "backward_group": 7,
        "projections": [
            {"label": "Saved axis", "forward_group": 1, "backward_group": 7, "alpha": 0.93}
        ],
        "excluded_detectors": [8],
        "alpha": 0.93,
        "alpha_error": 0.02,
        "alpha_method": "ratio",
        "alpha_reference_run": 900,
        "alpha_x": 0.94,
        "alpha_x_error": 0.03,
        "alpha_x_reference_run": 901,
        "t0_bin": 0,
        "first_good_bin": 0,
        "last_good_bin": 7,
        "deadtime_mode": "manual",
        "deadtime_correction": True,
        "dead_time_us": [0.001] * 8,
        "background_mode": "reference_run",
        "background_correction": True,
        "background_run": {
            "run_number": 999,
            "timing": {
                "t0_policy": {"mode": "manual", "offset_bins": 2},
                "source": SourceAssessment(None, "absent", "legacy:metadata").to_dict(),
            },
        },
        "scientific_extension": {
            "history": [{"value": 1.23, "uncertainty": 0.04}],
            "note": "retained",
        },
    }


@pytest.fixture
def workflow(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    source = tmp_path / "synthetic-flame.bin"
    source.write_text("synthetic loader input", encoding="utf-8")
    request = saved_request()
    profile = profile_from_payload(
        request, "Saved GPS", ProfileFingerprint("psi.gps", 8, "bin_combined")
    )
    profile.extra["scientific_extension"] = deepcopy(request["scientific_extension"])
    choice = RecordedInstrumentChoice(
        InstrumentSelection("psi.gps", "bin_combined", "1", "legacy"), ()
    )

    def load_source(path):
        metadata = {
            "run_number": 17,
            "facility": "PSI",
            "instrument": "FLAME",
            "source_file": str(source),
        }
        run = Run(
            run_number=17,
            source_file=str(source),
            histograms=[Histogram(np.full(8, 100.0), 0.01, 0, 0, 7) for _ in range(8)],
            metadata=deepcopy(metadata),
            grouping={"groups": {1: [1], 2: [2]}, "first_good_bin": 0, "last_good_bin": 7},
        )
        return MuonDataset(
            np.arange(8) * 0.01, np.zeros(8), np.full(8, 0.01), metadata=metadata, run=run
        )

    monkeypatch.setattr(window, "_load_file", load_source)
    state = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "grouping_profiles": [profile.to_dict()],
        "datasets": [
            {
                "run_number": 17,
                "source_file": str(source),
                "profile": profile.name,
                "instrument_choice": choice.to_dict(),
                "representations": {
                    "time_fb_asymmetry": {
                        "rep_type": "time_fb_asymmetry",
                        "recipe": {},
                        "result_metadata": {
                            "scientific_history": {
                                "fitted_amplitude": 0.13,
                                "conditional_error": 0.02,
                            }
                        },
                    }
                },
            }
        ],
    }
    yield window, state, profile, request, tmp_path
    window._clear_dirty()
    window.close()


def assert_request_retained(grouping, request):
    for key in (
        "groups",
        "group_names",
        "forward_group",
        "backward_group",
        "projections",
        "excluded_detectors",
        "alpha_x",
        "alpha_x_error",
        "alpha_x_reference_run",
        "background_run",
        "scientific_extension",
    ):
        actual, expected = grouping[key], request[key]
        if key in {"groups", "group_names"}:
            actual = {int(group): value for group, value in actual.items()}
        assert actual == expected, key
    if "alpha_policy" in grouping:
        assert grouping["alpha_policy"] == {
            "mode": "calibrated",
            "value": 0.93,
            "error": 0.02,
            "method": "ratio",
            "source_run": 900,
        }
    assert grouping["alpha"] == request["alpha"]
    assert grouping["alpha_error"] == request["alpha_error"]
    assert grouping["alpha_reference_run"] == request["alpha_reference_run"]
    if "deadtime_policy" in grouping:
        assert grouping["deadtime_policy"]["mode"] == "manual"
        assert grouping["deadtime_policy"]["values"] == request["dead_time_us"]
    else:
        assert grouping["dead_time_us"] == request["dead_time_us"]


@pytest.mark.parametrize("released", [False, True])
def test_conflict_restore_resave_and_replay_preserve_request_assignment_and_history(
    workflow, released
):
    window, state, profile, request, tmp_path = workflow
    if released:
        state["datasets"][0]["grouping_overrides"] = deepcopy(request)
    window.restore_project_state(state, str(tmp_path / "input.asymp"))
    dataset = window._data_browser.get_dataset(17)
    assert dataset is not None
    assert resolve_instrument(dataset.run).requires_choice
    assert_request_retained(dataset.run.grouping, request)
    assert window._dataset_assigned_profile_name(dataset) == profile.name
    assert "correction_error" in dataset.metadata
    saved = window.collect_project_state()
    entry = saved["datasets"][0]
    assert entry["profile"] == profile.name
    assert ("grouping_overrides" in entry) is released
    assert entry["instrument_choice"] == state["datasets"][0]["instrument_choice"]
    assert (
        entry["representations"]["time_fb_asymmetry"]["result_metadata"]["scientific_history"]
        == state["datasets"][0]["representations"]["time_fb_asymmetry"]["result_metadata"][
            "scientific_history"
        ]
    )
    path = tmp_path / "saved.asymp"
    save_project(saved, path)
    window.restore_project_state(load_project(path), str(path))
    replayed = window._data_browser.get_dataset(17)
    assert_request_retained(replayed.run.grouping, request)
    assert replayed.run.instrument_choice.to_dict() == entry["instrument_choice"]
    assert window._dataset_assigned_profile_name(replayed) == profile.name


@pytest.mark.parametrize("keep_retained", [False, True])
def test_explicit_choice_only_releases_incompatible_follower_without_calculation(
    workflow, monkeypatch, keep_retained
):
    window, state, profile, request, tmp_path = workflow
    window.restore_project_state(state, str(tmp_path / "input.asymp"))
    dataset = window._data_browser.get_dataset(17)
    before_profile = deepcopy(window._grouping_profiles[0].to_dict())
    before_arrays = (dataset.time.copy(), dataset.asymmetry.copy(), dataset.error.copy())

    def forbidden(*args, **kwargs):
        pytest.fail("Identity confirmation must not resolve a reference or reduce data")

    monkeypatch.setattr(window, "_resolve_background_reference", forbidden)
    monkeypatch.setattr(window, "_apply_grouping_settings_to_dataset", forbidden)
    monkeypatch.setattr(window._reduction_cache, "get_or_compute", forbidden)
    selected = (
        dataset.run.instrument_choice.selection
        if keep_retained
        else InstrumentSelection("psi.flame", "eight_channel", "1", "manual")
    )
    released = window._record_instrument_choice(
        dataset, accept_instrument_choice(dataset.run, selected)
    )
    assert released is (not keep_retained)
    assert not resolve_instrument(dataset.run).requires_choice
    assert_request_retained(dataset.run.grouping, request)
    assert window._dataset_assigned_profile_name(dataset) == profile.name
    assert window._dataset_has_grouping_override(dataset) is (not keep_retained)
    assert window._grouping_profiles[0].to_dict() == before_profile
    for actual, expected in zip(
        (dataset.time, dataset.asymmetry, dataset.error), before_arrays, strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    entry = window.collect_project_state()["datasets"][0]
    assert entry["profile"] == profile.name
    assert ("grouping_overrides" in entry) is (not keep_retained)


def test_cancel_toolbar_identity_leaves_the_complete_live_state_unchanged(workflow, monkeypatch):
    window, state, profile, request, tmp_path = workflow
    window.restore_project_state(state, str(tmp_path / "input.asymp"))
    dataset = window._data_browser.get_dataset(17)
    window._current_dataset = dataset
    window._clear_dirty()
    before = deepcopy(
        (
            dataset.run.grouping,
            dataset.run.metadata,
            dataset.metadata,
            dataset.run.instrument_choice,
            window._grouping_profiles[0].to_dict(),
        )
    )
    opened = []

    def cancel(dialog):
        opened.append(dialog)
        index = next(
            i
            for i in range(dialog._choices.count())
            if isinstance(dialog._choices.itemData(i), InstrumentSelection)
            and dialog._choices.itemData(i).profile_key == "psi.flame"
        )
        dialog._choices.setCurrentIndex(index)
        return QDialog.Rejected

    monkeypatch.setattr(InstrumentIdentityDialog, "exec", cancel)
    window._on_instrument_identity()
    assert len(opened) == 1
    assert (
        dataset.run.grouping,
        dataset.run.metadata,
        dataset.metadata,
        dataset.run.instrument_choice,
        window._grouping_profiles[0].to_dict(),
    ) == before
    assert not window._dirty


def test_fresh_file_retains_template_identity_request_and_fresh_raw_evidence(workflow):
    window, state, profile, request, tmp_path = workflow
    template = window._load_file(state["datasets"][0]["source_file"])
    template.run.run_number = 11
    template.metadata["run_number"] = 11
    template.run.metadata["run_number"] = 11
    template.run.source_file = str(tmp_path / "template.bin")
    template.metadata["source_file"] = template.run.source_file
    template.run.metadata["instrument"] = "GPS"
    template.run.grouping = deepcopy(request)
    template.run.instrument_choice = accept_instrument_choice(
        template.run, InstrumentSelection("psi.gps", "bin_combined", "1", "manual")
    )
    window._data_browser.add_dataset(template)
    assert window._grouping_profiles == []
    window._load_files([state["datasets"][0]["source_file"]])
    fresh = window._data_browser.get_dataset(17)
    assert fresh is not None
    assert fresh.run.metadata["instrument"] == "FLAME"
    assert fresh.run.instrument_choice.selection.identity == ("psi.gps", "bin_combined")
    assert resolve_instrument(fresh.run).requires_choice
    assert_request_retained(fresh.run.grouping, request)
    assert "Instrument identity conflicts" in fresh.metadata["correction_error"]
    assert window._grouping_profiles == []
    assert_request_retained(template.run.grouping, request)


def test_combined_identity_source_and_released_request_survive_save_and_replay(
    workflow, monkeypatch
):
    from pathlib import Path

    from asymmetry.core.instruments.resolution import UnresolvedInstrumentIdentity

    window, state, profile, request, tmp_path = workflow
    original_loader = window._load_file

    def load_constituent(path):
        dataset = original_loader(path)
        number = 18 if Path(path).name == "second-flame.bin" else 17
        dataset.run.run_number = number
        dataset.run.source_file = str(path)
        dataset.metadata.update(run_number=number, source_file=str(path))
        dataset.run.metadata.update(run_number=number, source_file=str(path))
        dataset.run.instrument_choice = accept_instrument_choice(
            dataset.run, InstrumentSelection("psi.flame", "eight_channel", "1", "metadata")
        )
        dataset.run.source_assessment = SourceAssessment(False, "known", "metadata")
        return dataset

    monkeypatch.setattr(window, "_load_file", load_constituent)
    first_path = state["datasets"][0]["source_file"]
    second_path = tmp_path / "second-flame.bin"
    second_path.write_text("second synthetic loader input", encoding="utf-8")
    for path in (first_path, second_path):
        window._data_browser.add_dataset(load_constituent(str(path)))
    combined_number = window._data_browser.add_combined_dataset([17, 18], combined_run_number=-1)
    assert combined_number == -1
    combined = window._data_browser.get_dataset(combined_number)
    assert combined.run.instrument_choice.selection.profile_key == "psi.flame"
    assert combined.run.source_assessment.pulsed is False
    window._grouping_profiles = [profile]
    combined.run.grouping.update(deepcopy(request))
    combined.run.instrument_choice = accept_instrument_choice(
        combined.run, InstrumentSelection("psi.gps", "bin_combined", "1", "manual")
    )
    window._assign_dataset_profile(combined, profile.name)
    captured_unknown = SourceAssessment(None, "absent", "manual")
    combined.run.source_assessment = captured_unknown
    selected = UnresolvedInstrumentIdentity("Authored combined identity")
    assert window._record_instrument_choice(
        combined, accept_instrument_choice(combined.run, selected)
    )
    profile_before = deepcopy(profile.to_dict())
    choice_before = combined.run.instrument_choice.to_dict()
    histograms_before = [histogram.counts.copy() for histogram in combined.run.histograms]
    saved = window.collect_project_state()
    entry = saved["combined_datasets"][0]
    assert entry["instrument_choice"] == choice_before
    assert entry["source_assessment"] == captured_unknown.to_dict()
    assert entry["profile"] == profile.name
    assert_request_retained(entry["grouping_overrides"], request)
    assert all(
        item["instrument_choice"]["selection"]["profile_key"] == "psi.flame"
        and item["source_assessment"]["pulsed"] is False
        for item in saved["datasets"]
    )
    path = tmp_path / "combined.asymp"
    save_project(saved, path)
    window.restore_project_state(load_project(path), str(path))
    replayed = window._data_browser.get_dataset(combined_number)
    assert replayed is not None
    assert replayed.run is not None, replayed.metadata
    assert replayed.run.instrument_choice.to_dict() == choice_before
    assert replayed.run.source_assessment == captured_unknown
    assert_request_retained(replayed.run.grouping, request)
    assert window._dataset_has_grouping_override(replayed)
    assert window._dataset_assigned_profile_name(replayed) == profile.name
    assert window._grouping_profiles[0].to_dict() == profile_before
    for actual, expected in zip(replayed.run.histograms, histograms_before, strict=True):
        np.testing.assert_array_equal(actual.counts, expected)
    second_path.unlink()
    import asymmetry.gui.utils.project_sources as source_ui

    monkeypatch.setattr(source_ui, "locate_source_directory", lambda *args: "")
    window.restore_project_state(load_project(path), str(path))
    unavailable = window._data_browser.get_dataset(combined_number)
    assert unavailable is not None
    assert unavailable.run is None
    assert "unavailable" in unavailable.metadata["correction_error"].lower()
    resaved = window.collect_project_state()
    assert combined_number not in {item["run_number"] for item in resaved["datasets"]}
    assert len(resaved["combined_datasets"]) == 1
    retained = resaved["combined_datasets"][0]
    assert retained["combined_run_number"] == combined_number
    assert retained["source_run_numbers"] == [17, 18]
    assert retained["instrument_choice"] == choice_before
    assert retained["source_assessment"] == captured_unknown.to_dict()
    assert retained["profile"] == profile.name
    assert retained["constituent_sources"] == entry["constituent_sources"]
    assert_request_retained(retained["grouping_overrides"], request)
    assert window._grouping_profiles[0].to_dict() == profile_before
    missing_path = tmp_path / "unavailable-combined.asymp"
    save_project(resaved, missing_path)
    assert load_project(missing_path)["combined_datasets"][0]["instrument_choice"] == choice_before
