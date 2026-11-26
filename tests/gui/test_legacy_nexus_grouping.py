"""Import and reopen legacy NeXus grouping through the desktop project path."""

import warnings

import h5py
import numpy as np
import pytest

from asymmetry.core.io.nexus_grouping import NexusGroupingWarning
from asymmetry.core.project import save_project
from asymmetry.core.project.calibration import PreparedGrouping
from asymmetry.core.workflow import apply_grouping_preset
from asymmetry.gui.mainwindow import MainWindow
from tests.io.test_nexus_grouping_declarations import grouping_file, legacy_zero_grouping_file

pytestmark = pytest.mark.gui


@pytest.mark.parametrize("edit_grouping", [False, True])
def test_legacy_grouping_import_and_project_reopen(qapp, tmp_path, edit_grouping):
    source = tmp_path / "synthetic.nxs"
    raw = legacy_zero_grouping_file(source, instrument="HiFi")
    window = MainWindow()
    try:
        with pytest.warns(NexusGroupingWarning):
            window._load_files([str(source)])
        dataset = window._data_browser.get_dataset(42)
        message = "The file’s detector-grouping table contains only zeros. Applied the standard HiFi forward/backward grouping."
        assert message in window._log_panel.to_plain_text()

        if edit_grouping:
            edited = apply_grouping_preset(dataset, "Transverse (Vector)")
            applied, _ = window._apply_grouping_settings_to_dataset(
                dataset, PreparedGrouping(edited.run.grouping)
            )
            assert applied

        grouping = dataset.run.grouping
        project = tmp_path / "legacy.asymp"
        state = window.collect_project_state()
        payload = state["datasets"][0]["grouping_overrides"]
        assert payload["nexus_grouping_declaration"] == {"values": [0] * 64, "available": 1}
        provenance = payload.get("grouping_provenance")
        if edit_grouping:
            assert provenance is None
        else:
            assert provenance["fallback_reason"] == "nexus_v1_all_zero"
            assert provenance["profile_revision"] == "2"
        save_project(state, project)
        window._clear_dirty()
        with warnings.catch_warnings():
            warnings.simplefilter("error", NexusGroupingWarning)
            window._open_project_file(str(project))
        reopened = window._data_browser.get_dataset(42)
        expected_preset = "Transverse (Vector)" if edit_grouping else "Longitudinal"
        assert reopened.run.grouping["grouping_preset"] == expected_preset
        assert reopened.run.grouping["groups"] == grouping["groups"]
        assert reopened.run.grouping.get("grouping_provenance") == provenance
        assert (
            reopened.run.grouping["nexus_grouping_declaration"]
            == payload["nexus_grouping_declaration"]
        )
        assert (
            reopened.effective_recipe.to_dict()["periods"]["red"].get("grouping_provenance")
            == provenance
        )
        np.testing.assert_array_equal([h.counts for h in reopened.run.histograms], raw)
    finally:
        window._clear_dirty()
        window.close()


def test_ad_hoc_grouping_template_does_not_transfer_legacy_file_provenance(qapp, tmp_path):
    legacy = tmp_path / "legacy.nxs"
    declared = tmp_path / "declared.nxs"
    legacy_zero_grouping_file(legacy, instrument="HiFi")
    grouping_file(declared, "v1", instrument="HiFi", table=[1] * 32 + [2] * 32)
    with h5py.File(declared, "r+") as handle:
        handle["run/number"][()] = 43
    window = MainWindow()
    try:
        with pytest.warns(NexusGroupingWarning):
            window._load_files([str(legacy)])
        window._load_files([str(declared)])
        first = window._data_browser.get_dataset(42)
        second = window._data_browser.get_dataset(43)
        assert first.run.grouping["grouping_provenance"]["fallback_reason"] == "nexus_v1_all_zero"
        assert "nexus_grouping_declaration" not in second.run.grouping
        assert "grouping_provenance" not in second.run.grouping
    finally:
        window._clear_dirty()
        window.close()
