"""Get Info provenance from a serialized synthetic legacy LEM ROOT acquisition.

The same production loader used by the GUI supplies numeric timing, scalar
metadata, two histogram identities, and retained terminal bins. This capture
exercises the calibration boundary without asserting physical t0 accuracy.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtWidgets import QApplication

from asymmetry.core.io import load

from ._base import CaptureContext, Scenario, _optimize_png, register


class RunInfoProvenanceScenario(Scenario):
    name = "run_info_provenance"
    description = (
        "Run Info dialog showing the preserved provenance of a loaded legacy LEM ROOT run."
    )
    size = (780, 660)

    def capture(self, ctx: CaptureContext) -> Path:  # noqa: D401
        from asymmetry.gui.windows.run_info_dialog import RunInfoDialog

        fixture = Path(__file__).parents[3] / "tests/fixtures/psi_root/legacy-lem.root"
        dataset = load(str(fixture))
        assert dataset.run.histograms[0].bin_width == 0.01
        assert dataset.run.grouping["detector_t0_bins"] == [4, 7]
        # Logical synthetic provenance keeps workstation paths out of the image.
        dataset.metadata["source_file"] = fixture.name
        dataset.run.source_file = fixture.name

        dialog = RunInfoDialog(dataset)
        dialog.resize(*self.size)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        dialog.show()
        # The total-counts cells are backfilled from a QTimer.singleShot(0, ...)
        # deferred sum (see RunInfoDialog._fill_total_counts), so pump the event
        # loop until they land before grabbing — otherwise the capture shows the
        # transient "computing…" placeholder.
        _pump_events(200)

        pix = dialog.grab()
        out_path = ctx.output_dir / f"{self.name}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not pix.save(str(out_path), "PNG"):
            raise RuntimeError(f"Failed to save screenshot to {out_path}")
        _optimize_png(out_path)

        dialog.close()
        dialog.deleteLater()
        _pump_events(40)
        return out_path


def _pump_events(milliseconds: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(int(milliseconds), loop.quit)
    loop.exec()
    QApplication.processEvents()


register(RunInfoProvenanceScenario())
