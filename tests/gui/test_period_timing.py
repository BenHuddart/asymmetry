"""The t0 preview uses the calibration and counts of the selected held period."""

import pytest

from asymmetry.gui.windows.grouping.dialog import GroupingDialog
from tests.gui.test_grouping_dialog import (
    _set_good_window_mode,
    _t0_line_dataset,
    _wait_for_t0_detection,
)

pytestmark = pytest.mark.gui


@pytest.mark.parametrize("conflicting_period", [0, 1])
@pytest.mark.parametrize("t0_mode", ["from_file", "manual", "auto_detect"])
def test_live_period_switch_selects_t0_calibration_and_detection(qapp, conflicting_period, t0_mode):
    red = _t0_line_dataset(t0_bins=(10, 10), peaks=(10, 10), n_bins=64)
    green = _t0_line_dataset(t0_bins=(20, 20), peaks=(20, 20), n_bins=64)
    sources = ["file", "file"]
    sources[conflicting_period] = "conflict"
    red.run.grouping.update(
        t0_source=sources[0],
        first_good_bin=25,
        period_histograms=[red.run.histograms, green.run.histograms],
        period_t0_sources=sources,
        period_reduced=[(ds.time, ds.asymmetry, ds.error) for ds in (red, green)],
    )
    dialog = GroupingDialog([red])
    dialog._set_t0_mode_combo(t0_mode)
    dialog._on_t0_mode_changed()
    if t0_mode == "manual":
        dialog._t0_spin.setValue(12)
    _set_good_window_mode(dialog, "manual")
    dialog._t_good_offset_spin.setValue(3)
    dialog._last_good_spin.setValue(dialog._t0_spin.value() + 18)
    offset = 2 if t0_mode == "manual" else 0
    for period, mode in [(0, "red"), (1, "green"), (0, "red")]:
        dialog._set_period_mode(mode)
        dialog._refresh_preview()
        _wait_for_t0_detection(dialog)
        assert dialog._file_common_t0_for_preview_run() == (10 if period == 0 else 20)
        anchor = 10 if period == 0 else 20
        assert dialog._t0_spin.value() == anchor + offset
        if t0_mode == "manual":
            assert dialog._current_t0_policy().offset_bins == offset
        assert dialog._t0_assessment_grouping()["t0_source"] == sources[period]
        verdict = dialog._current_t0_verdict()
        assert any("disagrees" in message for message in verdict.messages) == (
            period == conflicting_period
        )
        assert "Δ +0" in dialog._t0_detected_label.text()
        window = dialog._current_good_window_policy()
        assert (window.first_offset_bins, window.last_offset_bins) == (3, 18)
        if t0_mode == "manual" and period == 1:
            offset = 3
            dialog._t0_spin.setValue(anchor + offset)
            assert dialog._current_t0_policy().offset_bins == offset
    dialog._clear_dirty()
    dialog.close()
