"""Global Fit Wizard — Running page mid-screening.

Freezes the wizard's **Running** state part-way through a screening pass: the
streaming decision trail shows the first steps marked done (green check) and
the current step active, above the expanded *Live log* that captures every
progress message inline. No fit runs — the trail and log are driven directly
through the same primitives the live worker uses (``stream_placeholders`` /
``activate_step`` / ``set_status`` and the log panel), which is a faithful
snapshot of a real mid-stream state.

Uses the four-field Ag LF decoupling series so the header chips match the
Setup and Result captures.
"""

from __future__ import annotations

from PySide6.QtWidgets import QWidget

from ..data import make_ag_lf_decoupling
from ._base import Scenario, register, _process_events_for


class GlobalFitWizardRunningScenario(Scenario):
    name = "global_fit_wizard_running"
    description = (
        "Global Fit Wizard Running page mid-screening — streaming decision "
        "trail with the Live log expanded."
    )
    size = (1180, 760)

    def build(self) -> QWidget:
        from asymmetry.gui.windows.global_fit_wizard_window import (
            GlobalFitWizardWindow,
            _screening_placeholder_steps,
        )

        datasets = make_ag_lf_decoupling(fields_g=(0.0, 15.0, 50.0, 100.0))
        window = GlobalFitWizardWindow()
        window.set_analysis_context(datasets)
        _process_events_for(milliseconds=80)

        # Enter the Running page and drive the trail to a realistic mid-screening
        # frame: series conditions read (done), candidate families chosen (done),
        # per-run screening currently active. These are the exact primitives the
        # live worker's progress callback drives; activate_step marks earlier
        # steps done.
        window._running_header_label.setText("Screening the series…")
        window._running_trail.stream_placeholders(_screening_placeholder_steps())
        window._running_trail.activate_step("screening")
        window._running_trail.set_status(
            "Running per-run single-fit screening (2 of 4 runs)…"
        )

        # Populate and reveal the Live log with representative progress lines.
        window._log_section.setExpanded(True)
        for line in (
            "Preparing consolidated candidate portfolio for 4 datasets.",
            "Preparing per-dataset single-fit wizard tables for the shared portfolio.",
            "Running phase-1 single-fit screening on run 5201 (0 G).",
            "Running phase-1 single-fit screening on run 5202 (15 G).",
        ):
            window._log_panel.log(line)

        window._stack.setCurrentIndex(1)  # _PAGE_RUNNING
        _process_events_for(milliseconds=150)
        return window


register(GlobalFitWizardRunningScenario())
