"""Preview and Apply use the actual selected F/B observations."""

from copy import deepcopy

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram
from asymmetry.core.transform.deadtime import DeadtimeCorrectionError
from asymmetry.gui.mainwindow import MainWindow
from tests.gui.test_deadtime_rejection import dataset as dataset
from tests.gui.test_grouping_preview_pane import _reduce_preview

pytestmark = pytest.mark.gui


def test_preview_apply_and_window_edit_share_required_scope(qapp, dataset):
    dataset.run.histograms.append(Histogram(np.full(3, 10.0), 1))
    grouping = dataset.run.grouping
    grouping.update(
        deadtime_correction=True,
        deadtime_mode="file",
        dead_time_us=[0.1, 0.1, None],
        deadtime_file_us=[0.1, 0.1, None],
        first_good_bin=2,
        last_good_bin=2,
    )
    window = MainWindow()
    preview = _reduce_preview(dataset)
    assert preview.corrections[0].deadtime.status == "applied"
    window._apply_grouping_settings_to_dataset(dataset, deepcopy(grouping))
    np.testing.assert_allclose(dataset.asymmetry, preview.asymmetry)
    assert "correction_error" not in dataset.metadata
    assert window._get_fit_dataset(dataset) is not None
    before = deepcopy(dataset.run.grouping)
    with pytest.raises(DeadtimeCorrectionError, match="bin 1"):
        window._apply_grouping_settings_to_dataset(
            dataset,
            dict(
                before,
                good_window_policy={
                    "mode": "manual",
                    "first_offset_bins": 0,
                    "last_offset_bins": 2,
                },
            ),
        )
    assert dataset.run.grouping == before
    payload = dict(before, groups={1: [1], 2: [3]})
    with pytest.warns(UserWarning, match="File deadtime correction not applied"):
        window._apply_grouping_settings_to_dataset(dataset, payload)
    assert "correction_error" not in dataset.metadata
    assert all(
        c["deadtime"]["status"] == "skipped"
        for c in dataset.metadata["correction_outcomes"].values()
    )


def test_dialog_can_select_file_policy_for_partial_acquisition_table(qapp, dataset):
    from asymmetry.gui.windows.grouping.dialog import GroupingDialog
    from tests._qt_helpers import wait_for

    dataset.run.histograms.append(Histogram(np.full(3, 10.0), 1))
    dataset.run.grouping.update(
        groups={1: [1], 2: [2], 3: [3]},
        good_frames=100,
        deadtime_correction=False,
        deadtime_file_us=[0.1, 0.1, None],
        first_good_bin=2,
        last_good_bin=2,
    )
    dialog = GroupingDialog([dataset])
    try:
        section = dialog._deadtime_section
        assert section._mode_buttons["file"].isEnabled()
        section._mode_buttons["file"].click()
        pane = dialog._preview_pane
        pane.flush()
        wait_for(
            lambda: (
                pane._last_result is not None
                and pane._last_result.corrections[0].deadtime.status == "applied"
            ),
            qapp,
        )
        assert pane.corrections_complete
        assert dialog._apply_btn.isEnabled()
        dialog._backward_combo.setCurrentIndex(dialog._backward_combo.findData(3))
        pane.flush()
        wait_for(
            lambda: (
                pane._last_result is not None
                and pane._last_result.corrections[0].deadtime.status == "skipped"
            ),
            qapp,
        )
        assert pane.corrections_complete
        assert dialog._apply_btn.isEnabled()
        assert "File deadtime correction not applied" in pane._incomplete_banner.text()
    finally:
        dialog._clear_dirty()
        dialog.close()
