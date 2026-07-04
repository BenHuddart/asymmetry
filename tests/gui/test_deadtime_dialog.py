"""Tests for the dedicated deadtime-configuration dialog.

Covers mode gating (off/file/manual/estimate availability + widget enable
state), manual-table editing (direct edit + fill-all), the Cal button
(calibrate-from-reference), the Estimate button (estimate-from-source-run),
the max-correction-at-t=0 summary line, and the returned ``DeadtimePolicy``.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from asymmetry.core.data.dataset import Histogram
from asymmetry.core.project.profiles import DeadtimePolicy
from asymmetry.gui.windows.grouping.deadtime_dialog import (
    DeadtimeDialog,
    DeadtimeSourceRun,
    deadtime_status_text,
)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _source_run(run_number: int = 4001, tau_us: float = 0.02) -> DeadtimeSourceRun:
    """A source run whose early-time counts are consistent with *tau_us*."""
    amplitude = 120.0
    bin_width = 0.01
    num_good_frames = 1000.0
    lifetime_us = 2.1969811
    times = (np.arange(12, dtype=float) + 1.0) * bin_width
    frame_scale = num_good_frames * bin_width
    true_counts = amplitude * np.exp(-times / lifetime_us)
    observed = true_counts * (
        1.0 - (true_counts / frame_scale) * lifetime_us * (1.0 - np.exp(-tau_us / lifetime_us))
    )
    histograms = [Histogram(observed.copy(), bin_width=bin_width) for _ in range(2)]
    return DeadtimeSourceRun(
        run_number=run_number,
        label=f"Run {run_number}",
        histograms=histograms,
        good_frames=num_good_frames,
    )


def _dialog(**overrides) -> DeadtimeDialog:
    defaults = dict(
        n_detectors=2,
        mode="off",
        file_values_us=[],
        manual_values_us=[0.01, 0.01],
        manual_method="manual",
        estimated_us=None,
        source_run=None,
        source_runs=[],
        reference_run_number=None,
        peak_rates_per_us=[],
        bin_width_us=0.01,
        good_frames=1.0,
    )
    defaults.update(overrides)
    return DeadtimeDialog(**defaults)


def test_off_mode_disables_table_editing_and_hides_summary(qapp: QApplication) -> None:
    dlg = _dialog(mode="off")
    assert dlg._current_mode() == "off"
    assert dlg._summary_label.text() == ""
    assert dlg.get_policy() == DeadtimePolicy(mode="off")


def test_file_mode_only_enabled_when_reference_provides_values(qapp: QApplication) -> None:
    dlg = _dialog(mode="off", file_values_us=[])
    assert dlg._mode_buttons["file"].isEnabled() is False

    dlg2 = _dialog(mode="file", file_values_us=[0.011, 0.022])
    assert dlg2._mode_buttons["file"].isEnabled() is True
    assert dlg2._current_mode() == "file"
    assert dlg2._display_values() == pytest.approx([0.011, 0.022])
    policy = dlg2.get_policy()
    assert policy.mode == "from_file"


def test_manual_mode_table_is_editable_and_fill_all_broadcasts(qapp: QApplication) -> None:
    dlg = _dialog(mode="manual", manual_values_us=[0.01, 0.01])

    # Direct per-row edit (ns displayed, µs stored).
    dlg._table.item(0, 1).setText("25.0")
    assert dlg._manual_values_us[0] == pytest.approx(0.025)

    # Fill-all broadcasts a single ns value to every detector.
    dlg._fill_all_spin.setValue(30.0)
    dlg._on_fill_all_clicked()
    assert dlg._manual_values_us == pytest.approx([0.03, 0.03])

    policy = dlg.get_policy()
    assert policy.mode == "manual"
    assert policy.values == pytest.approx([0.03, 0.03])
    assert policy.method == "manual"


def test_calibrate_button_fills_table_from_reference_run(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asymmetry.gui.windows.grouping.deadtime_dialog as deadtime_dialog_module

    monkeypatch.setattr(
        deadtime_dialog_module,
        "calibrate_deadtime_from_histograms",
        lambda *args, **kwargs: [0.011, 0.022],
    )
    run = _source_run(4001)
    dlg = _dialog(
        mode="manual",
        source_runs=[run],
        reference_run_number=4001,
    )
    dlg._on_calibrate_clicked()

    assert dlg._current_mode() == "manual"
    policy = dlg.get_policy()
    assert policy.values == pytest.approx([0.011, 0.022])
    assert policy.method == "calibrate"
    assert policy.source_run == 4001


def test_estimate_button_uses_selected_source_run(qapp: QApplication) -> None:
    reference = _source_run(4101, tau_us=0.02)
    other = _source_run(4102, tau_us=0.04)
    dlg = _dialog(
        mode="estimate",
        source_runs=[reference, other],
        reference_run_number=4101,
    )
    idx = dlg._source_run_combo.findData(4101)
    dlg._source_run_combo.setCurrentIndex(idx)
    dlg._on_estimate_clicked()

    policy = dlg.get_policy()
    assert policy.mode == "estimate"
    assert policy.source_run == 4101
    assert policy.estimated_us == pytest.approx(0.02, rel=1e-2, abs=5e-4)


def test_estimate_mode_without_estimate_blocks_accept(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asymmetry.gui.windows.grouping.deadtime_dialog as deadtime_dialog_module

    warnings: list[str] = []
    monkeypatch.setattr(
        deadtime_dialog_module.QMessageBox,
        "warning",
        lambda *args, **kwargs: warnings.append(str(args[2]) if len(args) > 2 else ""),
    )
    dlg = _dialog(mode="estimate", source_runs=[_source_run(4001)], reference_run_number=4001)
    dlg._on_accept()
    # No estimate has been computed yet, so the dialog must not have accepted.
    assert warnings, "expected a warning dialog for the missing estimate"
    assert dlg.result() != DeadtimeDialog.DialogCode.Accepted


def test_summary_line_reports_max_correction_at_t0(qapp: QApplication) -> None:
    # tau=0.02us, peak rate 2000 counts/us, bin width 0.01us, good_frames 1e6:
    # N = rate * bin_width = 20 counts/frame-scale-bin; correction is tiny but
    # nonzero and the label must report a percentage.
    dlg = _dialog(
        mode="manual",
        manual_values_us=[0.02],
        n_detectors=1,
        peak_rates_per_us=[2000.0],
        bin_width_us=0.01,
        good_frames=1.0e6,
    )
    assert "Max correction at t=0:" in dlg._summary_label.text()

    # A much larger deadtime relative to the frame-normalized rate produces a
    # larger reported correction — the label should react to table edits.
    dlg._fill_all_spin.setValue(500.0)
    dlg._on_fill_all_clicked()
    text_after = dlg._summary_label.text()
    assert "Max correction at t=0:" in text_after


def test_deadtime_status_text_matches_mode() -> None:
    assert deadtime_status_text(DeadtimePolicy(mode="off")) == "Deadtime: off"
    assert deadtime_status_text(DeadtimePolicy(mode="from_file")) == "Deadtime: from file"
    assert "manual" in deadtime_status_text(DeadtimePolicy(mode="manual", values=[0.02, 0.02]))
    assert "estimated" in deadtime_status_text(DeadtimePolicy(mode="estimate", source_run=42))
    assert "42" in deadtime_status_text(DeadtimePolicy(mode="estimate", source_run=42))
