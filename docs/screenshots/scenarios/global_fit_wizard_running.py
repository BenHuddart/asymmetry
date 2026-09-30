"""Global Fit Wizard — screening in progress inside the Screen step.

Freezes the stepper part-way through a screening pass. Screen is marked
running ("Screening…") and shows the run's progress block: the header with
Cancel, the streaming decision trail (the first steps done, per-run screening
active) and the expanded *Live log*. No worker runs. The scenario starts the
run the way ``_begin_run`` does, minus the worker, and feeds representative
core progress messages through the window's own ``_on_progress``. The same
prefix table as a live run lights the trail, so this is a faithful snapshot of
a real mid-stream state.

Uses the four-field Ag LF decoupling series so the header chips match the
other Global Fit Wizard captures.
"""

from __future__ import annotations

from PySide6.QtWidgets import QWidget

from ..data import make_ag_lf_decoupling
from ._base import Scenario, _process_events_for, register

#: The direction row's "Longitudinal" button.
_LONGITUDINAL = 1

#: Core progress messages as a screening run emits them, up to run 5202.
_PROGRESS = (
    "Preparing consolidated candidate portfolio for 4 datasets.",
    "Preparing per-dataset single-fit wizard tables for the shared portfolio.",
    "Running phase-1 single-fit screening on run 5201 (0 G).",
    "Running phase-1 single-fit screening on run 5202 (15 G).",
)


class GlobalFitWizardRunningScenario(Scenario):
    name = "global_fit_wizard_running"
    description = (
        "Global Fit Wizard Screen step mid-screening — the stepper marks Screen "
        "running, above the streaming decision trail and the Live log."
    )
    size = (1180, 560)

    def build(self) -> QWidget:
        from asymmetry.gui.windows.global_fit_wizard_window import (
            _RUN_MODES,
            GlobalFitWizardWindow,
        )

        datasets = make_ag_lf_decoupling(fields_g=(0.0, 15.0, 50.0, 100.0))
        window = GlobalFitWizardWindow()
        window.set_analysis_context(datasets)
        # Answered the way global_fit_wizard_setup answers it.
        window._picker._direction_group.button(_LONGITUDINAL).click()
        _process_events_for(milliseconds=80)

        # _begin_run without _run_analysis: the progress block moves into the
        # Screen view and the window goes busy, so the stepper marks Screen running.
        run = _RUN_MODES["screening"]
        window._analysis_mode = "screening"
        window._status_label.setText(
            "Building the single-fit screening table in the background. "
            "The main window stays responsive while the shared candidate portfolio is screened."
        )
        window._run_progress.start(run.header, run.placeholders)
        window._run_progress.trail.set_status(run.opening_status)
        window._run_progress.append_log("Starting screening for 4 datasets.")
        window._host_progress(run.step)
        window._run_progress.show()
        window._show_step(run.step)
        window._set_busy(True)
        for message in _PROGRESS:
            window._on_progress(0, 0, message)
        window._run_progress._log_section.setExpanded(True)
        _process_events_for(milliseconds=150)
        return window


register(GlobalFitWizardRunningScenario())
