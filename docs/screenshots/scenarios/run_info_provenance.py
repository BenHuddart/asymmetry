"""Run Info dialog: loaded-run provenance made explicit.

Companion screenshot to :doc:`/reference/loading_data` (the "Accessing
metadata" and "Reference provenance" sections). Loading a raw run preserves
its experiment provenance, and the **Get Info** dialog is where Asymmetry
surfaces it. This scenario opens :class:`RunInfoDialog` on the grouped YBCO
transverse-field archetype
(:func:`docs.screenshots.data.make_ybco_knight_grouped`) — a run backed by a
full :class:`~asymmetry.core.data.dataset.Run` with four detector histograms —
so the **Run Parameters** table shows the metadata a loader extracts and keeps
explicit: instrument, title, comment, temperature, field, the number of
detector histograms, bin count, bin width, and total counts.

The run header fields (instrument, comment, start/end, field direction) are
plausible ISIS/MUSR values fed through the real dialog code path — the same
metadata dict a NeXus loader would populate — with fixed timestamp strings so
the capture is byte-stable (no wall clock). Temperature (100 K, just above
YBCO's Tc=90 K) and field (200 G transverse) are consistent with the
underlying synthetic run.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtWidgets import QApplication

from ..data import make_ybco_knight_grouped
from ._base import CaptureContext, Scenario, _optimize_png, register


class RunInfoProvenanceScenario(Scenario):
    name = "run_info_provenance"
    description = (
        "Run Info dialog showing the preserved provenance of a loaded YBCO TF run."
    )
    size = (780, 660)

    def capture(self, ctx: CaptureContext) -> Path:  # noqa: D401
        from asymmetry.gui.windows.run_info_dialog import RunInfoDialog

        dataset = make_ybco_knight_grouped()
        # Enrich the run header the way a real NeXus loader would: these are the
        # provenance fields the dialog renders (meta.get("instrument"),
        # "comment", "started"/"stopped", "field_direction",
        # "detector_orientation"). Fixed strings — no wall clock — keep the
        # capture byte-identical across runs.
        dataset.metadata.update(
            {
                "instrument": "MUSR",
                "comment": "YBCO single crystal, TF 200 G, field perpendicular to c-axis",
                "started": "2024-03-14 09:12:41",
                "stopped": "2024-03-14 11:47:03",
                "field_direction": "Transverse",
                "detector_orientation": "Transverse",
            }
        )

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
