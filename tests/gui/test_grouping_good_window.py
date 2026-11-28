"""Tests for the Grouping dialog's good-data window controls."""

from __future__ import annotations

import sys
from dataclasses import replace

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.gui.windows.grouping_dialog import GroupingDialog
from tests.gui.test_grouping_dialog import (
    _invalid_hal_grouping_dataset,
    _set_good_window_mode,
    _wait_for_t0_detection,
    _wait_until,
)

pytestmark = pytest.mark.gui


def test_disjoint_spectator_window_preserves_invalid_membership_editor(qapp: QApplication) -> None:
    dataset = _invalid_hal_grouping_dataset()
    dataset.run.grouping["detector_first_good_bins"][0] = 35
    dataset.run.grouping["detector_last_good_bins"][0] = 39

    dialog = GroupingDialog([dataset])
    pane = dialog._preview_pane
    pane.flush()
    _wait_until(lambda: "Detector(s) 10 do not exist" in pane._incomplete_banner.text())
    assert not dialog._apply_btn.isEnabled()
    assert "File good-data window unavailable" in dialog._good_window_label.text()
    assert dialog._group_table.item(0, 3).flags() & Qt.ItemFlag.ItemIsEditable


def test_edit_to_disjoint_pair_reports_empty_window_without_slot_error(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _invalid_hal_grouping_dataset()
    dataset.run.grouping.update(
        groups={1: [2], 2: [3], 3: [6]},
        group_names={1: "F1", 2: "F2", 3: "F5"},
        grouping_preset=None,
        forward_group=1,
        backward_group=3,
    )
    dataset.run.grouping["detector_first_good_bins"][2] = 36
    dataset.run.grouping["detector_last_good_bins"][2] = 39
    dialog = GroupingDialog([dataset])
    pane = dialog._preview_pane
    pane.flush()
    _wait_until(dialog._apply_btn.isEnabled)
    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda *args: errors.append(args[1]))

    dialog._group_table.item(2, 3).setText("3")
    pane.flush()

    assert not errors
    _wait_until(lambda: "empty good-bin support" in pane._incomplete_banner.text())
    assert not dialog._apply_btn.isEnabled()
    assert "File good-data window unavailable" in dialog._good_window_label.text()


@pytest.mark.parametrize("mode", ["from_file", "manual"])
def test_repair_reseeds_file_timing_after_detection_finishes(qapp: QApplication, mode: str) -> None:
    dataset = _invalid_hal_grouping_dataset()
    dataset.run.histograms[0] = replace(dataset.run.histograms[0], t0_bin=12)
    grouping = dataset.run.grouping
    grouping["t0_bin"] = grouping["detector_t0_bins"][0] = 12
    grouping["detector_first_good_bins"][0] += 4
    grouping["detector_last_good_bins"][0] += 4
    dialog = GroupingDialog([dataset])
    _wait_for_t0_detection(dialog)
    assert dialog._t0_spin.value() == 12
    if mode == "manual":
        dialog._t0_mode_combo.setCurrentIndex(dialog._t0_mode_combo.findData("manual"))
        dialog._t0_spin.setValue(13)

    for row in range(8):
        dialog._group_table.item(row, 3).setText(str(row + 2))
    dialog._preview_pane.flush()
    _wait_until(dialog._apply_btn.isEnabled)

    if mode == "from_file":
        assert dialog._t0_spin.value() == 8
        assert dialog._t_good_offset_spin.value() == 3
        assert dialog._last_good_spin.value() == 32
    else:
        assert dialog._t0_spin.value() == 13


def _good_window_dataset(
    run_number: int = 4800,
    detector_t0: tuple[int, int] = (6, 8),
    detector_first_good: tuple[int, int] = (9, 11),
    detector_last_good: tuple[int, int] = (30, 32),
    n_bins: int = 40,
) -> MuonDataset:
    """A PSI run carrying the per-detector t0 + good-bin tables a file window needs.

    The two detectors are staggered, so the run's file window is the
    *intersection* over the analysis pair in the aligned coordinates of the
    common t0 (``max(detector_t0)``) — not either detector's own window.
    """
    histograms = [
        Histogram(counts=np.full(n_bins, 10.0), bin_width=0.016, t0_bin=t0_bin)
        for t0_bin in detector_t0
    ]
    common_t0 = max(detector_t0)
    run = Run(
        run_number=run_number,
        histograms=histograms,
        metadata={"run_number": run_number, "facility": "PSI"},
        grouping={
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "t0_bin": common_t0,
            "first_good_bin": common_t0,
            "last_good_bin": n_bins - 1,
            "detector_t0_bins": list(detector_t0),
            "detector_first_good_bins": list(detector_first_good),
            "detector_last_good_bins": list(detector_last_good),
        },
    )
    t = np.arange(n_bins, dtype=float) * 0.016
    return MuonDataset(
        time=t,
        asymmetry=np.zeros_like(t),
        error=np.full_like(t, 0.01),
        metadata={"run_number": run_number, "facility": "PSI"},
        run=run,
    )


def _second_good_window_dataset(run_number: int = 4801) -> MuonDataset:
    """A sibling run with a different file t0 *and* a different file window."""
    return _good_window_dataset(
        run_number=run_number,
        detector_t0=(11, 13),
        detector_first_good=(16, 19),
        detector_last_good=(34, 36),
    )


def test_good_window_defaults_to_from_file_with_readonly_spins(qapp: QApplication) -> None:
    """Default: both spins read-only, showing the preview run's own file window."""
    dialog = GroupingDialog([_good_window_dataset()])

    assert dialog._current_good_window_mode() == "from_file"
    assert dialog._draft.good_window_policy.mode == "from_file"
    assert dialog._good_window_mode_combo.isHidden() is False
    assert dialog._t_good_offset_spin.isReadOnly() is True
    assert dialog._last_good_spin.isReadOnly() is True
    # File window over the analysis pair: t0 = 8, first = 8 + max(3, 3) = 11,
    # last = 8 + min(24, 24) = 32.
    assert dialog._t_good_offset_spin.value() == 3
    assert dialog._last_good_spin.value() == 32


def test_good_window_manual_mode_enables_both_spins_and_seeds_from_file(
    qapp: QApplication,
) -> None:
    """Switching to Manual makes both ends editable without moving them."""
    dialog = GroupingDialog([_good_window_dataset()])
    dialog._draft_dirty = False

    _set_good_window_mode(dialog, "manual")

    assert dialog._t_good_offset_spin.isReadOnly() is False
    assert dialog._last_good_spin.isReadOnly() is False
    assert dialog._t_good_offset_spin.value() == 3
    assert dialog._last_good_spin.value() == 32
    assert dialog._draft_dirty is True

    dialog._draft_dirty = False
    dialog._t_good_offset_spin.setValue(7)
    assert dialog._draft_dirty is True

    dialog._sync_draft_from_form()
    policy = dialog._draft.good_window_policy
    assert policy.mode == "manual"
    # Both ends are stored as offsets from the run's effective t0 (8).
    assert policy.first_offset_bins == 7
    assert policy.last_offset_bins == 32 - 8


def test_good_window_payload_carries_the_policy(qapp: QApplication) -> None:
    """The payload includes both the absolute window and its policy."""
    dialog = GroupingDialog([_good_window_dataset()])

    assert dialog._current_grouping_payload()["good_window_policy"] == {"mode": "from_file"}

    _set_good_window_mode(dialog, "manual")
    dialog._t_good_offset_spin.setValue(7)
    dialog._last_good_spin.setValue(30)

    payload = dialog._current_grouping_payload()
    assert payload["good_window_policy"] == {
        "mode": "manual",
        "first_offset_bins": 7,
        "last_offset_bins": 30 - 8,
    }
    # The preview run's absolute window rides along unchanged.
    assert payload["t0_bin"] == 8
    assert payload["t_good_offset"] == 7
    assert payload["first_good_bin"] == 15
    assert payload["last_good_bin"] == 30


def test_from_file_good_window_follows_the_preview_run(qapp: QApplication) -> None:
    """Read-only spins re-derive the file window for whichever run is previewed."""
    dialog = GroupingDialog(
        [_good_window_dataset(4800), _second_good_window_dataset(4801)],
        selected_run_number=4800,
    )
    assert (dialog._t_good_offset_spin.value(), dialog._last_good_spin.value()) == (3, 32)

    dialog._scope_panel.set_current_run(4801)

    # Run 4801: t0 = 13, first = 13 + max(5, 6) = 19, last = 13 + min(23, 23) = 36.
    assert (dialog._t_good_offset_spin.value(), dialog._last_good_spin.value()) == (6, 36)


def test_manual_good_window_keeps_its_offsets_across_a_preview_run_switch(
    qapp: QApplication,
) -> None:
    """One profile keeps its window relative to each run's own t0."""
    dialog = GroupingDialog(
        [_good_window_dataset(4800), _second_good_window_dataset(4801)],
        selected_run_number=4800,
    )
    _set_good_window_mode(dialog, "manual")
    dialog._t_good_offset_spin.setValue(7)
    dialog._last_good_spin.setValue(30)

    dialog._scope_panel.set_current_run(4801)

    assert dialog._current_good_window_mode() == "manual"
    assert dialog._t_good_offset_spin.value() == 7
    # Same offsets, run 4801's t0 of 13: last = 13 + 22 = 35.
    assert dialog._last_good_spin.value() == 13 + 22


def test_manual_good_window_rides_a_manual_t0_edit(qapp: QApplication) -> None:
    """A t0 edit keeps both offsets and moves the absolute Last Good Bin."""
    dialog = GroupingDialog([_good_window_dataset()])
    _set_good_window_mode(dialog, "manual")
    dialog._t_good_offset_spin.setValue(7)
    dialog._last_good_spin.setValue(30)
    dialog._set_t0_mode_combo("manual")
    dialog._on_t0_mode_changed()

    dialog._t0_spin.setValue(12)

    assert dialog._t_good_offset_spin.value() == 7
    assert dialog._last_good_spin.value() == 34
    dialog._sync_draft_from_form()
    policy = dialog._draft.good_window_policy
    assert (policy.first_offset_bins, policy.last_offset_bins) == (7, 22)


def test_good_window_line_shows_the_offset_in_time_and_the_file_values(
    qapp: QApplication,
) -> None:
    """The offset is in µs, with file limits also visible in Manual mode."""
    dialog = GroupingDialog([_good_window_dataset()])

    assert dialog._good_window_label.text() == "≈ 0.048 µs after t0"

    _set_good_window_mode(dialog, "manual")
    dialog._t_good_offset_spin.setValue(7)

    assert dialog._good_window_label.text() == (
        "≈ 0.112 µs after t0 · File: offset 3 · last bin 32"
    )


def test_override_editing_hides_the_good_window_selector(qapp: QApplication) -> None:
    """A released run edits absolute values."""
    dialog = GroupingDialog(
        [_good_window_dataset(4800), _second_good_window_dataset(4801)],
        selected_run_number=4801,
        overridden_run_numbers=[4801],
    )

    assert dialog._editing_target() == 4801
    assert dialog._good_window_mode_combo.isHidden() is True
    assert dialog._t_good_offset_spin.isReadOnly() is False
    assert dialog._last_good_spin.isReadOnly() is False
    assert "good_window_policy" not in dialog._current_grouping_payload()
