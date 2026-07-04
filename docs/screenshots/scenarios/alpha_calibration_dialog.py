"""Alpha calibration dialog: TF run auto-suggestion and before/after preview.

Opens the alpha calibration dialog directly on a synthesised YBCO
transverse-field run. The run's title ("YBCO TF 200G...") and its 200 G
field both satisfy the weak-transverse-field calibration heuristic
(:func:`asymmetry.core.data.calibration.classify_tf_calibration_run`), so it
is highlighted and pre-selected in the calibration-run dropdown exactly as it
would be for a real TF calibration run. Estimating alpha then draws the
before (alpha = 1) / after (fitted alpha) asymmetry preview. Companion to
:doc:`/reference/detector_grouping` and :doc:`/reference/grouping_calibration`.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtWidgets import QApplication

from ..data import make_ybco_knight_grouped
from ._base import CaptureContext, Scenario, register


class AlphaCalibrationDialogScenario(Scenario):
    name = "alpha_calibration_dialog"
    description = (
        "Alpha calibration dialog with a highlighted TF candidate run and a before/after preview."
    )
    size = (760, 640)

    def capture(self, ctx: CaptureContext) -> Path:  # noqa: D401
        from asymmetry.gui.windows.grouping.alpha_calibration_dialog import (
            AlphaCalibrationDialog,
        )

        dataset = make_ybco_knight_grouped()
        grouping = dataset.run.grouping

        dialog = AlphaCalibrationDialog(
            [dataset],
            groups=grouping["groups"],
            group_names=grouping.get("group_names"),
            forward_group=grouping["forward_group"],
            backward_group=grouping["backward_group"],
            selected_run_number=int(dataset.run_number),
        )
        dialog.resize(*self.size)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        dialog.show()
        _pump_events(150)

        # Estimate is a synchronous, pure-core reduction (no worker thread),
        # so a single button click is enough to populate the before/after
        # preview and the result label before the grab.
        estimate_btn = getattr(dialog, "_estimate_btn", None)
        if estimate_btn is not None:
            estimate_btn.click()
            _pump_events(150)

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


register(AlphaCalibrationDialogScenario())
