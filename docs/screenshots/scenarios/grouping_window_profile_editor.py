"""Grouping window: profile editor with its live asymmetry preview pane.

Opens the Grouping window on two synthesised YBCO TF runs (four detector
histograms grouped one-per-group) with **two grouping profiles in concurrent
use** — "Sample A" (the ★ default, editing target) and "Sample B", one run
assigned to each — showing the profile selector with its default marker, the
"Default for new runs" checkbox, the scope panel's ``follows <profile>``
chips and Assign-to control, and the debounced live forward/backward
asymmetry preview together. Companion to :doc:`/reference/detector_grouping`
and :doc:`/reference/grouping_calibration`.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtWidgets import QApplication

from ..data import make_ybco_knight_grouped
from ._base import CaptureContext, Scenario, register


def _renumbered(dataset, run_number: int):
    """Give a synthesised dataset a distinct run number (all three records)."""
    dataset.run.run_number = int(run_number)
    dataset.run.metadata["run_number"] = int(run_number)
    dataset.metadata["run_number"] = int(run_number)
    return dataset


class GroupingWindowProfileEditorScenario(Scenario):
    name = "grouping_window_profile_editor"
    description = "Grouping window profile editor, with the live asymmetry preview pane."
    size = (1180, 780)

    def capture(self, ctx: CaptureContext) -> Path:  # noqa: D401
        from asymmetry.core.project.profiles import (
            AlphaPolicy,
            profile_fingerprint_for_run,
            profile_from_payload,
        )
        from asymmetry.gui.windows.grouping.dialog import GroupingDialog

        dataset_a = make_ybco_knight_grouped()
        dataset_b = _renumbered(make_ybco_knight_grouped(seed=202), 7102)

        # Two profiles for the fingerprint, one per sample: "Sample A" is the
        # ★ default and the editing target; run 7102 is assigned to "Sample B",
        # so the scope panel shows both chip states.
        fingerprint = profile_fingerprint_for_run(dataset_a.run)
        profile_a = profile_from_payload(
            dict(dataset_a.run.grouping), "Sample A", fingerprint, active=True
        )
        profile_b = profile_from_payload(
            dict(dataset_b.run.grouping), "Sample B", fingerprint, active=False
        )
        profile_a.alpha_policy = AlphaPolicy(mode="per_run_estimate")

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
        from .grouping_window_good_window_row import _pump_until

        dialog._preview_pane.flush()
        _pump_until(lambda: dialog._preview_pane.corrections_complete)
        if not dialog._preview_pane.corrections_complete:
            raise RuntimeError("Automatic alpha preview did not complete")
        if dialog._alpha_spin.value() != round(
            dialog._preview_pane._last_result.alpha, dialog._alpha_spin.decimals()
        ):
            raise RuntimeError("Alpha control must display the completed preview coefficient")

        pix = dialog.grab()
        out_path = ctx.output_dir / f"{self.name}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not pix.save(str(out_path), "PNG"):
            raise RuntimeError(f"Failed to save screenshot to {out_path}")

        dialog.close()
        dialog.deleteLater()
        _pump_events(40)
        return out_path


def _pump_events(milliseconds: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(int(milliseconds), loop.quit)
    loop.exec()
    QApplication.processEvents()


register(GroupingWindowProfileEditorScenario())
