"""A missing reference keeps its background request visible in the preview."""

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.gui.windows.grouping.dialog import GroupingDialog

from ._base import CaptureContext, Scenario, _optimize_png, register
from .grouping_window_good_window_row import _pump_events, _pump_until


class IncompletePreviewScenario(Scenario):
    name = "grouping_incomplete_preview"
    description = "Missing background reference and disabled Apply."
    size = (1180, 780)
    complete = False
    reference_width = None
    count_levels = (120.0, 80.0)
    correction_settings = {
        "background_correction": True,
        "background_mode": "reference_run",
        "background_run": {"run_number": 9999, "period_number": 2},
    }

    def capture(self, ctx: CaptureContext) -> Path:
        # Counts are deterministic Poisson means per acquisition bin; time is in µs.
        time = np.arange(200) * 0.02
        run = Run(
            run_number=8102,
            histograms=[
                Histogram(np.full(200, level), bin_width=0.02) for level in self.count_levels
            ],
            grouping={
                "instrument": "generic_layout",
                "groups": {1: [1], 2: [2]},
                "forward_group": 1,
                "backward_group": 2,
                "alpha": 1.0,
                "t0_bin": 0,
                "first_good_bin": 0,
                "last_good_bin": 199,
                "deadtime_correction": False,
                **self.correction_settings,
            },
        )
        dataset = MuonDataset(
            time,
            np.full(200, 20.0),
            np.ones(200),
            metadata={"run_number": 8102, "title": self.description + " (synthetic)"},
            run=run,
        )
        datasets = [dataset]
        if self.reference_width is not None:
            reference = Run(
                run_number=9999,
                grouping={"instrument": "generic_layout"},
                histograms=[
                    Histogram(np.full(400, level), self.reference_width) for level in (10.0, 7.0)
                ],
            )
            datasets.append(MuonDataset(time, np.zeros(200), np.ones(200), run=reference))
        dialog = GroupingDialog(datasets)
        dialog.resize(*self.size)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        dialog.show()
        if self.name == "grouping_file_deadtime_warning":
            section = dialog._deadtime_section
            if not section._mode_buttons["file"].isEnabled():
                raise RuntimeError("From file must remain selectable for a warned skip")
            section._mode_buttons["off"].click()
            section._mode_buttons["file"].click()
        pane = dialog._preview_pane
        _pump_events(80)
        pane.flush()
        _pump_until(
            lambda: (
                pane._last_result is not None
                and pane._last_result.generation == pane._generation
                and pane._pending is None
                and not pane._debounce.isActive()
                and pane._tasks.active_count == 0
                and pane.corrections_complete == self.complete
                and dialog._apply_btn.isEnabled() == self.complete
            )
        )
        if (
            pane.corrections_complete != self.complete
            or dialog._apply_btn.isEnabled() != self.complete
        ):
            raise RuntimeError("Correction preview acceptance does not match its outcome")
        if not pane._incomplete_banner.text():
            raise RuntimeError("Correction warning must remain visible")
        if self.reference_width is not None and "compatible" not in pane._incomplete_banner.text():
            raise RuntimeError(
                f"Incompatible reference grid must explain the incomplete preview: {pane._incomplete_banner.text()}"
            )
        output = ctx.output_dir / f"{self.name}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        if not dialog.grab().save(str(output), "PNG"):
            raise RuntimeError("Could not save incomplete preview screenshot")
        _optimize_png(output)
        dialog._clear_dirty()
        dialog.close()
        dialog.deleteLater()
        _pump_events(40)
        return output


class ReferenceGridScenario(IncompletePreviewScenario):
    name = "grouping_reference_grid"
    description = "Incompatible reference native grid and disabled Apply."
    reference_width = 0.04
    correction_settings = {
        "background_correction": True,
        "background_mode": "reference_run",
        "background_run": {"run_number": 9999, "period_number": 1},
    }


register(ReferenceGridScenario())


class FileDeadtimeWarningScenario(IncompletePreviewScenario):
    name = "grouping_file_deadtime_warning"
    description = "Missing file deadtime calibration with enabled Apply."
    complete = True
    correction_settings = {"deadtime_correction": True, "deadtime_mode": "file", "dead_time_us": []}


class TailBackgroundWarningScenario(IncompletePreviewScenario):
    name = "grouping_tail_background_warning"
    description = "Unusable automatic tail with visible warning and enabled Apply."
    complete = True
    count_levels = (120.0, 0.0)
    correction_settings = {"background_correction": True, "background_mode": "tail_fit"}


register(TailBackgroundWarningScenario())
register(IncompletePreviewScenario())
register(FileDeadtimeWarningScenario())


class ExplicitBackgroundRangeScenario(Scenario):
    name = "grouping_explicit_background_range"
    description = "Separate authored background intervals in the grouping editor."
    size = (1180, 780)

    def capture(self, ctx: CaptureContext) -> Path:
        from ..data import make_ybco_knight_grouped

        dataset = make_ybco_knight_grouped()
        dataset.run.grouping.update(
            background_correction=True,
            background_mode="range",
            background_ranges=[[10, 60], [11, 61]],
        )
        dialog = GroupingDialog([dataset])
        dialog.resize(*self.size)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        dialog.show()
        pane = dialog._preview_pane
        pane.flush()
        _pump_until(
            lambda: (
                pane._last_result is not None and pane._last_result.generation == pane._generation
            )
        )
        if not pane.corrections_complete:
            raise RuntimeError("Explicit background interval preview did not complete")
        output = ctx.output_dir / f"{self.name}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        if not dialog._background_card.grab().save(str(output), "PNG"):
            raise RuntimeError("Could not save background interval controls")
        _optimize_png(output)
        dialog._clear_dirty()
        dialog.close()
        dialog.deleteLater()
        _pump_events(40)
        return output


register(ExplicitBackgroundRangeScenario())
