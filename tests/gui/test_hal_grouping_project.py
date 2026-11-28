"""Replay an explicit HAL grouping against the acquired forward-ring file."""

from dataclasses import replace

import pytest

from asymmetry.core.instruments.resolution import UnmappedAcquisition
from asymmetry.core.io import load
from asymmetry.core.project import CURRENT_SCHEMA_VERSION, save_project
from asymmetry.gui.mainwindow import MainWindow
from tests.io.test_psi_loader import _HAL_FORWARD_LABELS, _write_psi_mdu

pytestmark = pytest.mark.gui


@pytest.mark.parametrize("last_detector", [6, 14], ids=["available", "unavailable"])
def test_project_keeps_explicit_hal_grouping(qapp, tmp_path, last_detector):
    source = tmp_path / "tdc_hifi_00686.mdu"
    _write_psi_mdu(source, run_number=686, labels=_HAL_FORWARD_LABELS)
    choice = load(source).run.instrument_choice
    legacy_choice = replace(
        choice,
        selection=replace(
            choice.selection, acquisition=UnmappedAcquisition("incompatible"), profile_revision="1"
        ),
    )
    groups = {1: [2, 3], 5: [last_detector]}
    project = tmp_path / "hal.asymp"
    save_project(
        {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "datasets": [
                {
                    "run_number": 686,
                    "source_file": str(source),
                    "instrument_choice": legacy_choice.to_dict(),
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

    window = MainWindow()
    try:
        window._open_project_file(str(project))
        dataset = window._data_browser.get_dataset(686)
        assert dataset.run.instrument_choice == legacy_choice
        assert {
            int(gid): members for gid, members in dataset.run.grouping["groups"].items()
        } == groups
        assert "grouping_preset" not in dataset.run.grouping
        if last_detector > len(dataset.run.histograms):
            assert "14 do not exist" in dataset.metadata["correction_error"]
        else:
            assert "correction_error" not in dataset.metadata
    finally:
        window._clear_dirty()
        window.close()
