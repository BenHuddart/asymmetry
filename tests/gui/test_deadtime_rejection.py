"""Rejected corrections leave live runs intact and do not discard restored runs."""

from copy import deepcopy

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.transform import DeadtimeCorrectionError
from asymmetry.gui.mainwindow import MainWindow
from asymmetry.gui.windows.grouping.dialog import GroupingDialog
from tests._qt_helpers import wait_for

pytestmark = pytest.mark.gui


@pytest.fixture
def dataset(tmp_path):
    source = tmp_path / "run1.nxs"
    source.touch()
    run = Run(
        run_number=1,
        histograms=[
            Histogram(np.array(values), bin_width=1, good_bin_end=2)
            for values in ([5, 10, 5], [4, 4, 4])
        ],
        grouping={
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 2,
            "good_frames": 1,
            "deadtime_correction": False,
        },
        source_file=str(source),
    )
    return MuonDataset(
        np.arange(3.0),
        np.zeros(3),
        np.ones(3),
        metadata={"run_number": 1, "source_file": str(source)},
        run=run,
    )


def test_rejected_apply_leaves_grouping_and_arrays_intact(qapp, dataset):
    window = MainWindow()
    before = deepcopy(dataset.run.grouping)
    arrays = [array.copy() for array in (dataset.time, dataset.asymmetry, dataset.error)]
    payload = before | {
        "deadtime_correction": True,
        "deadtime_mode": "manual",
        "dead_time_us": [0.1, 0.1],
    }
    with pytest.raises(DeadtimeCorrectionError, match="detector 1"):
        window._apply_grouping_settings_to_dataset(dataset, payload)
    assert dataset.run.grouping == before
    for actual, expected in zip(
        (dataset.time, dataset.asymmetry, dataset.error), arrays, strict=True
    ):
        np.testing.assert_array_equal(actual, expected)


def test_project_restore_keeps_a_run_with_rejected_deadtime(qapp, dataset, monkeypatch, tmp_path):
    window = MainWindow()
    fresh = deepcopy(dataset)
    dataset.run.grouping.update(
        deadtime_correction=True, deadtime_mode="manual", dead_time_us=[0.1, 0.1]
    )
    dataset.metadata["grouping_overrides"] = True
    window._data_browser.add_dataset(dataset)
    state = window.collect_project_state()
    restored = MainWindow()
    messages = []
    monkeypatch.setattr(restored, "_load_file", lambda path: fresh)
    monkeypatch.setattr(
        restored._log_panel, "log", lambda message, **kwargs: messages.append(message)
    )
    restored.restore_project_state(state, str(tmp_path / "project.asymp"))
    loaded = restored._data_browser.get_dataset(1)
    assert loaded is fresh
    assert not loaded.run.grouping["deadtime_correction"]
    np.testing.assert_array_equal(loaded.asymmetry, np.zeros(3))
    assert any("Deadtime correction rejected for detector 1" in message for message in messages)


def test_alpha_estimate_surfaces_invalid_deadtime_without_blocking(qapp, dataset, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    del dataset.run.grouping["good_frames"]
    dialog = GroupingDialog([dataset])
    section = dialog._deadtime_section
    section._mode_buttons["manual"].click()
    section._fill_all_spin.setValue(-100)
    section._fill_all_btn.click()
    messages = []
    provider = dialog._calibration_correction_provider()
    assert provider is not None
    assert provider(dataset)["dead_time_us"] == [-0.1, -0.1]
    assert "good_frames" not in provider(dataset)
    monkeypatch.setattr(
        QMessageBox, "warning", lambda parent, title, message: messages.append(message)
    )
    dialog._alpha_section._on_estimate()
    wait_for(lambda: bool(messages), qapp)
    assert "Deadtime correction rejected: good frames" in messages[0]
    dialog._clear_dirty()


def test_summary_uses_first_good_bin_instead_of_a_prompt_peak(qapp, dataset):
    dialog = GroupingDialog([dataset])
    section = dialog._deadtime_section
    section._mode_buttons["manual"].click()
    section._fill_all_spin.setValue(-100)
    section._fill_all_btn.click()
    assert section._summary_label.text() == "Max correction at t=0: 33.3%"
    dialog._clear_dirty()


@pytest.mark.parametrize("mode", ["file", "manual"])
def test_signed_tables_survive_editor_and_profile_roundtrip(qapp, dataset, mode):
    from asymmetry.core.project.profiles import (
        GroupingProfile,
        profile_fingerprint_for_run,
        resolve_effective_grouping,
    )
    from asymmetry.gui.windows.grouping.profile_bridge import profile_from_form_payload

    values = [-0.1, 0.05]
    dataset.run.grouping.update(
        deadtime_correction=True, deadtime_mode=mode, deadtime_method=mode, dead_time_us=values
    )
    dialog = GroupingDialog([dataset])
    assert dialog._deadtime_section._display_values() == values
    profile = profile_from_form_payload(
        dialog._current_grouping_payload(),
        name="Signed calibration",
        fingerprint=profile_fingerprint_for_run(dataset.run),
    )
    restored = GroupingProfile.from_dict(profile.to_dict())
    assert resolve_effective_grouping(restored, dataset.run)["dead_time_us"] == values
