"""Finite signed coefficients in the Grouping window's deadtime card."""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget

from ..data import make_ybco_knight_grouped
from ._base import CaptureContext, Scenario, _optimize_png, register


class SignedDeadtimeScenario(Scenario):
    name = "signed_deadtime"
    description = "Manual deadtime table with finite negative and positive coefficients."
    size = (620, 320)

    def capture(self, ctx: CaptureContext) -> Path:
        from asymmetry.gui.windows.grouping.deadtime_section import DeadtimeSectionWidget

        run = make_ybco_knight_grouped().run
        width = run.histograms[0].bin_width
        section = DeadtimeSectionWidget()
        section.configure(
            n_detectors=4,
            mode="manual",
            file_values_us=[],
            manual_values_us=[-0.02, 0.01, -0.005, 0.015],
            manual_method="manual",
            estimated_us=None,
            source_run=None,
            source_runs=[],
            reference_run_number=None,
            peak_rates_per_us=[float(hist.counts[hist.t0_bin]) / width for hist in run.histograms],
            bin_width_us=width,
            good_frames=100000000,
        )
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(section)
        container.resize(*self.size)
        container.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        container.show()
        QApplication.processEvents()
        path = ctx.output_dir / f"{self.name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not container.grab().save(str(path), "PNG"):
            raise RuntimeError("Failed to capture signed deadtime card")
        _optimize_png(path)
        container.close()
        container.deleteLater()
        return path


register(SignedDeadtimeScenario())
