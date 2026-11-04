"""File-good raw windows survive the GUI's profile and project consumers."""

import numpy as np
import pytest

from asymmetry.core.io.nexus import NexusLoader
from asymmetry.core.project import load_project, save_project
from asymmetry.core.simulate import reduce_run_to_dataset
from asymmetry.gui.mainwindow import MainWindow
from asymmetry.gui.windows.grouping.dialog import GroupingDialog
from tests.io.test_nexus_good_times import write_good_times

pytestmark = pytest.mark.gui


def test_physical_file_window_preview_apply_and_project_reopen(qapp, tmp_path):
    source = tmp_path / "good.nxs"
    raw = write_good_times(source)
    dataset = NexusLoader().load(source)
    dialog = GroupingDialog([dataset])
    assert dialog._current_good_window_policy().mode == "from_file"
    assert dialog._last_good_spin.value() == 15  # one-based display
    dialog._clear_dirty()
    dialog.close()
    dialog.deleteLater()

    first = MainWindow()
    first._data_browser.add_dataset(dataset)
    applied, _ = first._apply_grouping_settings_to_dataset(dataset, dataset.run.grouping)
    assert applied
    np.testing.assert_allclose(dataset.time, [0.0226, 0.0426, 0.0626, 0.0826], atol=1e-15)
    project = tmp_path / "good.asymp"
    save_project(first.collect_project_state(), project)
    second = MainWindow()
    second.restore_project_state(load_project(project), str(project))
    restored = second._data_browser.get_dataset(dataset.run_number)
    assert restored is not None
    np.testing.assert_allclose(restored.time, dataset.time, atol=1e-15)
    np.testing.assert_allclose(reduce_run_to_dataset(restored.run).time, dataset.time, atol=1e-15)
    for hist, counts in zip(restored.run.histograms, raw, strict=True):
        assert (hist.good_bin_start, hist.good_bin_end) == (11, 14)
        np.testing.assert_array_equal(hist.counts, counts)
    first._clear_all_state()
    second._clear_all_state()
    first.close()
    second.close()
