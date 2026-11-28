"""Reopen HAL sources and explicit groupings through the desktop project path."""

import json
from dataclasses import replace

import pytest

from asymmetry.core.instruments.resolution import UnmappedAcquisition, resolve_instrument
from asymmetry.core.io import load
from asymmetry.core.project import CURRENT_SCHEMA_VERSION, save_project
from asymmetry.gui.mainwindow import MainWindow
from asymmetry.gui.windows.grouping.dialog import GroupingDialog
from tests.io.test_psi_hal_grouping import _HAL_FORWARD_LABELS, _HAL_FULL_LABELS
from tests.io.test_psi_loader import _write_psi_mdu

pytestmark = pytest.mark.gui


@pytest.fixture
def window(qapp):
    window = MainWindow()
    yield window
    window._clear_dirty()
    window.close()


@pytest.mark.parametrize("overrides", [{}, {"grouping_overrides": {}}])
def test_source_only_project_keeps_hal_full17_default(window, tmp_path, overrides):
    source = tmp_path / "tdc_hifi_00153.mdu"
    _write_psi_mdu(source, run_number=153, labels=_HAL_FULL_LABELS)
    project = tmp_path / "source_only.asymp"
    save_project(
        {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "datasets": [{"run_number": 153, "source_file": str(source), **overrides}],
        },
        project,
    )

    window._open_project_file(str(project))
    dataset = window._data_browser.get_dataset(153)
    grouping = dataset.run.grouping
    assert grouping["grouping_preset"] == "Per-octant"
    assert grouping["groups"] == {i: [i + 1, i + 9] for i in range(1, 9)}
    assert (grouping["forward_group"], grouping["backward_group"]) == (1, 5)
    assert dataset.n_points > 0


def test_full17_grouping_without_saved_clocks_keeps_native_good_window(window, tmp_path):
    source = tmp_path / "tdc_hifi_00153.mdu"
    _write_psi_mdu(source, run_number=153, labels=_HAL_FULL_LABELS)
    project = tmp_path / "full_grouping.asymp"
    groups = {1: [2, 3], 5: [6]}
    save_project(
        {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "datasets": [
                {
                    "run_number": 153,
                    "source_file": str(source),
                    "grouping_overrides": {
                        "groups": groups,
                        "forward_group": 1,
                        "backward_group": 5,
                    },
                }
            ],
        },
        project,
    )

    window._open_project_file(str(project))
    dataset = window._data_browser.get_dataset(153)
    grouping = dataset.run.grouping
    assert grouping["groups"] == groups
    assert (grouping["t0_bin"], grouping["first_good_bin"], grouping["last_good_bin"]) == (1, 2, 5)
    assert dataset.n_points == 4


@pytest.mark.parametrize(
    "last_detector, preset, acquired, saved_labels",
    [
        (6, None, _HAL_FORWARD_LABELS, True),
        (6, None, _HAL_FORWARD_LABELS, False),
        (14, None, _HAL_FORWARD_LABELS, True),
        (14, "Per-octant", _HAL_FORWARD_LABELS, True),
        (6, None, _HAL_FULL_LABELS, True),
    ],
    ids=[
        "available",
        "legacy-clock-table",
        "unavailable",
        "unavailable-named",
        "conflicting-acquisition",
    ],
)
def test_legacy_project_keeps_explicit_hal_grouping(
    window, tmp_path, last_detector, preset, acquired, saved_labels
):
    source = tmp_path / "tdc_hifi_00686.mdu"
    _write_psi_mdu(source, run_number=686, labels=acquired)
    groups = {"1": [2, 3], "5": [last_detector]}
    grouping = {
        "instrument": "HAL",
        "groups": groups,
        "forward_group": 1,
        "backward_group": 5,
    }
    if saved_labels:
        grouping["histogram_labels"] = [label.decode() for label in _HAL_FORWARD_LABELS]
    else:
        grouping["detector_t0_bins"] = [1] * len(_HAL_FORWARD_LABELS)
    if preset is not None:
        grouping["grouping_preset"] = preset
    project = tmp_path / "hal.asymp"
    project.write_text(
        json.dumps(
            {
                "schema_version": 40,
                "datasets": [
                    {
                        "run_number": 686,
                        "source_file": str(source),
                        "grouping_overrides": grouping,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    window._open_project_file(str(project))
    dataset = window._data_browser.get_dataset(686)
    choice = dataset.run.instrument_choice
    assert choice.selection.identity == ("psi.hal9500", UnmappedAcquisition("incompatible"))
    assert choice.selection.profile_revision == "1"
    assert choice.evidence == (("grouping_overrides.instrument", "HAL"),)
    context = resolve_instrument(dataset.run)
    assert len(dataset.run.histograms) == len(acquired)
    assert {str(gid): members for gid, members in dataset.run.grouping["groups"].items()} == groups
    if acquired == _HAL_FORWARD_LABELS:
        assert dataset.run.grouping.get("grouping_preset") == preset
    diagnostic = (
        "Instrument identity conflicts"
        if acquired == _HAL_FULL_LABELS
        else "14 do not exist"
        if last_detector > len(acquired)
        else None
    )
    assert context.requires_choice == (acquired == _HAL_FULL_LABELS)
    window._on_dataset_selected(dataset.run_number)
    dialog = GroupingDialog([dataset], parent=window)
    try:
        assert dialog._preset_combo.count() == 0
        assert not window._plot_panel._grouping_hint_bar.isVisibleTo(window._plot_panel)
    finally:
        dialog.close()
    if diagnostic is None:
        assert "correction_error" not in dataset.metadata
        assert dataset.n_points == 6
        assert dataset.effective_recipe is not None
    else:
        assert diagnostic in dataset.metadata["correction_error"]
        assert dataset.n_points == dataset.asymmetry.size == dataset.error.size == 0
        assert dataset.effective_recipe is None
        if acquired == _HAL_FORWARD_LABELS:
            assert dataset.histogram_observation is None
        assert dataset.ratio_validity is None


@pytest.mark.parametrize("previous_count", [8, 9], ids=["changed-count", "same-readout"])
def test_captured_unmapped_hal_checks_acquired_evidence(window, tmp_path, previous_count):
    source = tmp_path / "tdc_hifi_00686.mdu"
    _write_psi_mdu(source, run_number=686, labels=_HAL_FORWARD_LABELS[:previous_count])
    choice = load(source).run.instrument_choice
    choice = replace(
        choice,
        selection=replace(
            choice.selection, acquisition=UnmappedAcquisition("incompatible"), profile_revision="1"
        ),
    )
    _write_psi_mdu(source, run_number=686, labels=_HAL_FORWARD_LABELS)
    project = tmp_path / "captured_hal.asymp"
    groups = {1: [2, 3], 5: [6]}
    save_project(
        {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "datasets": [
                {
                    "run_number": 686,
                    "source_file": str(source),
                    "instrument_choice": choice.to_dict(),
                    "grouping_overrides": {
                        "groups": groups,
                        "forward_group": 1,
                        "backward_group": 5,
                    },
                }
            ],
        },
        project,
    )

    window._open_project_file(str(project))
    dataset = window._data_browser.get_dataset(686)
    assert dataset.run.instrument_choice == choice
    assert resolve_instrument(dataset.run).requires_choice == (previous_count == 8)
    assert {int(gid): members for gid, members in dataset.run.grouping["groups"].items()} == groups
    assert "grouping_preset" not in dataset.run.grouping
    if previous_count == 8:
        assert "Instrument identity conflicts" in dataset.metadata["correction_error"]
        assert dataset.n_points == 0
        assert dataset.effective_recipe is None
    else:
        assert "correction_error" not in dataset.metadata
        assert dataset.n_points == 6


def test_psi_project_keeps_decoded_veto_exclusion_and_group_names(window, tmp_path):
    source = tmp_path / "deltat_tdc_gps_00888.mdu"
    labels = (b"MV", b"Forw", b"Back", b"Up", b"Down", b"Righ", b"Left")
    _write_psi_mdu(source, run_number=888, labels=labels)
    groups = {i: [i] for i in range(1, len(labels) + 1)}
    project = tmp_path / "gps.asymp"
    save_project(
        {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "datasets": [
                {
                    "run_number": 888,
                    "source_file": str(source),
                    "grouping_overrides": {
                        "groups": groups,
                        "forward_group": 3,
                        "backward_group": 2,
                    },
                }
            ],
        },
        project,
    )

    window._open_project_file(str(project))
    dataset = window._data_browser.get_dataset(888)
    grouping = dataset.run.grouping
    assert grouping["groups"] == groups
    assert grouping["group_names"] == {i: label.decode() for i, label in enumerate(labels, 1)}
    assert grouping["included_groups"] == {i: i != 1 for i in groups}
    assert "grouping_preset" not in grouping
    assert dataset.n_points > 0
