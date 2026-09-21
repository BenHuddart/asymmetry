"""Grouping window: the preview pane's Counts view, background compare focused.

Opens the Grouping window on the same two-profile YBCO TF setup as
:mod:`grouping_window_profile_editor` — "Sample A" (the ★ default, editing
target) and "Sample B", one run assigned to each — then sets the draft's
background mode to **Tail fit**, focuses the background compare, and switches
the shared preview pane to its **Counts** view: the corrected forward/backward
group spectra over the full histogram, with t0, the good window and the
subtracted background level marked, and the without-background ghost (F
solid, B dashed) drawn on top. Companion to :doc:`/reference/detector_grouping`.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtWidgets import QApplication

from ..data import make_ybco_knight_grouped
from ._base import CaptureContext, Scenario, register
from .grouping_window_profile_editor import _renumbered


class GroupingWindowCountsViewScenario(Scenario):
    name = "grouping_window_counts_view"
    description = (
        "Grouping window preview pane in its Counts view, background compare "
        "focused: the corrected F/B spectra with t0, the good window and the "
        "subtracted background level marked, and the without-background ghost."
    )
    size = (1180, 780)

    def capture(self, ctx: CaptureContext) -> Path:  # noqa: D401
        from asymmetry.core.project.profiles import (
            profile_fingerprint_for_run,
            profile_from_payload,
        )
        from asymmetry.gui.windows.grouping.dialog import GroupingDialog

        dataset_a = make_ybco_knight_grouped()
        dataset_b = _renumbered(make_ybco_knight_grouped(seed=202), 7102)

        # Two profiles for the fingerprint, one per sample — the same setup as
        # grouping_window_profile_editor, so the two scenarios read as the same
        # window in two states.
        fingerprint = profile_fingerprint_for_run(dataset_a.run)
        profile_a = profile_from_payload(
            dict(dataset_a.run.grouping), "Sample A", fingerprint, active=True
        )
        profile_b = profile_from_payload(
            dict(dataset_b.run.grouping), "Sample B", fingerprint, active=False
        )

        dialog = GroupingDialog(
            [dataset_a, dataset_b],
            profiles=[profile_a, profile_b],
            assigned_profiles={
                int(dataset_a.run_number): "Sample A",
                int(dataset_b.run_number): "Sample B",
            },
            selected_run_number=int(dataset_a.run_number),
        )
        dialog.resize(*self.size)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        dialog.show()
        _pump_events(500)

        # Tail-fit background on the draft, via the same combo the dialog's own
        # tests drive (tests/gui/test_grouping_dialog.py).
        combo = dialog._background_section._mode_combo
        combo.setCurrentIndex(combo.findData("tail_fit"))
        _pump_events(200)

        # Focus the background compare, then switch the pane to Counts — a
        # redraw of the same result, never a recompute — and flush the
        # debounced preview request the mode change above just queued.
        dialog._set_compare_stage("background")
        dialog._preview_pane.set_view("counts")
        dialog._preview_pane.flush()
        _pump_events(1500)

        pix = dialog.grab()
        out_path = ctx.output_dir / f"{self.name}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not pix.save(str(out_path), "PNG"):
            raise RuntimeError(f"Failed to save screenshot to {out_path}")

        # Setting the background mode dirties the draft, so close() would hit
        # the dialog's "Discard changes" guard (which hangs headless) — tear
        # the workers down directly instead.
        dialog._teardown_workers()
        dialog.deleteLater()
        _pump_events(40)
        return out_path


def _pump_events(milliseconds: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(int(milliseconds), loop.quit)
    loop.exec()
    QApplication.processEvents()


register(GroupingWindowCountsViewScenario())
