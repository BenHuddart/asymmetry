"""Own bounded scalar-scan fits and publish only complete applicable executions."""

from copy import deepcopy
from functools import partial

import numpy as np
from PySide6.QtCore import QObject

from asymmetry.core.fitting.field_scan import (
    as_composite_model,
    fit_rf_resonance,
    fit_scan_baseline,
    fit_scan_peaks,
)
from asymmetry.core.fitting.scan_history import ScanFitRecord, restore_scan, scan_support
from asymmetry.core.project.json_values import encode_non_finite
from asymmetry.gui.scan.fit_history import fit_settings, publish_fit
from asymmetry.gui.tasks import TaskRunner, check_task_cancelled

TITLES = {"baseline": "Baseline", "peaks": "Peaks", "rf": "RF resonance"}


class ScanFitJob(QObject):
    """Detached scientific input with QObject-bound delivery to its owning controller."""

    def __init__(self, controller, kind, scan, settings, model, initial, baseline, baseline_index):
        super().__init__(controller)
        self.controller = controller
        self.kind, self.scan, self.settings = kind, scan, settings
        self.model, self.initial = model, initial
        self.baseline, self.baseline_index = baseline, baseline_index
        window = controller.window
        self.project = window._project_model
        self.series = self.project.batch(window._alc_scan_series_id)
        self.points = window._alc_scan_points
        self.history = window._alc_scan_view.fit_history
        self.rf_mode = window._alc_rf_mode
        self.construction = window._scan_controller.request_identity()
        self.baseline_execution = window._alc_baseline
        self.runner = TaskRunner(self)
        self.worker = None

    def compute(self, worker):
        poll = partial(check_task_cancelled, worker)
        poll()
        worker.progress.emit(0, 0, f"Fitting {TITLES[self.kind].lower()}…")
        kwargs = {}
        if self.kind == "baseline":
            result = fit_scan_baseline(
                self.scan, self.settings["regions"], model=self.model, cancel_check=poll
            )
            fit = result.fit
            curve = result.evaluate_baseline(poll)
        else:
            if self.kind == "peaks":
                result = fit_scan_peaks(
                    self.baseline, self.model, initial=self.initial, cancel_check=poll
                )
                fit = result.fit
                kwargs = dict(
                    baseline_index=self.baseline_index,
                    covariance=result.covariance,
                    effective_dof=result.effective_dof,
                )
            else:
                fit = fit_rf_resonance(
                    self.scan,
                    nu_rf=self.settings["rf_nu"],
                    a_mu=self.settings["rf_a_mu"],
                    a_p=self.settings["rf_a_p"],
                    cancel_check=poll,
                )
            if not fit.success:
                raise ValueError(f"Fit did not converge: {fit.message}")
            poll()
            worker.progress.emit(0, 0, "Sampling fitted prediction…")
            values = {name: fit.parameters[name].value for name in self.model.param_names}
            curve = np.asarray(self.model.function(self.scan.x, **values))
            if self.kind == "peaks":
                poll()
                curve = self.baseline.evaluate_baseline(poll) + curve
        poll()
        record = ScanFitRecord.capture(
            self.kind,
            self.scan,
            self.settings,
            self.model,
            fit,
            curve,
            request=self.settings,
            **kwargs,
        )
        poll()
        return record

    def finished(self, record):
        self.controller.terminal(self, record=record)

    def error(self, message):
        self.controller.terminal(self, message=message)

    def cancelled(self):
        self.controller.terminal(self)

    def progress(self, current, total, message):
        if self.controller.current is self and not self.worker.is_cancelled():
            self.controller.controls.set_progress(current, total, message)
            if total <= 0:
                self.controller.controls.set_indeterminate(True)


class ScanFitController(QObject):
    """One running solve and at most one replacement request; no concurrent heavy fits."""

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.current = None
        self.pending = None

    @property
    def controls(self):
        return self.window._alc_scan_view.fit_controls

    @property
    def idle(self):
        return self.current is None and self.pending is None

    def invalidate(self, kind=None):
        affected = {"baseline", "peaks"} if kind == "baseline" else {kind}
        if kind is None:
            affected = set(TITLES)
        if self.pending is not None and self.pending.kind in affected:
            self.pending.deleteLater()
            self.pending = None
        if self.current is not None and self.current.kind in affected:
            self.current.worker.cancel()
            self.controls.progress_label.setText("Stopping fit…")

    def stop(self):
        self.invalidate()

    def reset(self):
        """Cancel the departing project without leaving progress in its replacement."""
        self.stop()
        self.controls.button.hide()
        self.controls.progress_bar.hide()
        self.controls.progress_label.clear()
        self.controls.progress_label.hide()

    def shutdown(self):
        self.stop()
        if self.current is not None:
            job, self.current = self.current, None
            job.runner.shutdown()
            job.deleteLater()

    def applicable(self, job):
        window, view = self.window, self.window._alc_scan_view
        if (
            job.project is not window._project_model
            or job.series is not window._project_model.batch(window._alc_scan_series_id)
            or job.points is not window._alc_scan_points
            or job.history is not view.fit_history
            or job.rf_mode != window._alc_rf_mode
            or job.construction != window._scan_controller.request_identity()
            or job.settings != fit_settings(view, job.kind)
        ):
            return False
        if job.kind == "peaks" and (
            job.baseline_index != view.fit_history.active.get("baseline")
            or self.window._alc_baseline is not job.baseline_execution
        ):
            return False
        scan = window._alc_display_scan(view.x_key())
        return scan is not None and encode_non_finite(scan_support(scan)) == encode_non_finite(
            scan_support(job.scan)
        )

    def launch(self, job):
        self.current = job
        self.controls.button.show()
        self.controls.progress_label.show()
        self.controls.progress_label.setText(f"Fitting {TITLES[job.kind].lower()}…")
        self.controls.progress_bar.show()
        self.controls.set_indeterminate(True)
        job.worker = job.runner.start(
            job.compute,
            on_finished=job.finished,
            on_error=job.error,
            on_cancelled=job.cancelled,
            on_progress=job.progress,
        )

    def terminal(self, job, *, record=None, message=None):
        if self.current is not job:
            return
        applicable = self.applicable(job)
        current = not job.worker.is_cancelled() and applicable
        self.current = None
        # The callable has returned; join its native thread before starting its replacement.
        job.runner.shutdown()
        if current and record is not None:
            publish_fit(self.window, record)
            self.window._sync_alc_series_extra()
            self.window._mark_dirty()
            self.window._log_panel.log(f"Fitted {TITLES[job.kind].lower()}.")
        elif current and message is not None:
            self.window._alc_notify(TITLES[job.kind], f"Could not fit: {message}", warning=True)
        job.deleteLater()
        pending, self.pending = self.pending, None
        if pending is not None and self.applicable(pending):
            self.launch(pending)
        else:
            if pending is not None:
                pending.deleteLater()
            if job.history is not self.window._alc_scan_view.fit_history:
                return
            self.controls.button.hide()
            self.controls.progress_bar.hide()
            self.controls.progress_label.setVisible(applicable)
            if not applicable:
                return
            self.controls.progress_label.setText(
                "Fit complete."
                if current and record
                else "Fit stopped."
                if message is None
                else "Fit failed."
            )

    def request(self, kind):
        window, view = self.window, self.window._alc_scan_view
        title = TITLES[kind]
        if kind == "peaks" and (
            window._alc_baseline is None or "baseline" not in view.fit_history.active
        ):
            window._alc_notify(
                title, "Fit a baseline first; peaks are fitted on the corrected curve."
            )
            return
        if kind == "rf" and not window._alc_rf_mode:
            window._alc_notify(
                title,
                "Enable 'RF resonance (Green − Red)' and Build Scan first — the RF model fits the period-difference observable.",
            )
            return
        if view.derivative_enabled():
            message = (
                "Turn off dA/dB to fit a baseline on the scan."
                if kind == "baseline"
                else "Turn off dA/dB to fit the RF resonance."
            )
            window._alc_notify(title, message)
            return
        if kind == "rf" and view.x_key() != "field":
            window._alc_notify(title, "Switch the x-axis to B (G) — the RF model fits vs field.")
            return
        scan = window._alc_display_scan(view.x_key())
        if scan is None:
            window._alc_notify(title, window._alc_too_few_points_message())
            return
        settings = deepcopy(fit_settings(view, kind))
        baseline, baseline_index, initial = None, None, None
        if kind == "baseline":
            if not settings["regions"]:
                window._alc_notify(
                    title,
                    "Add at least one non-resonant region (start/end) to define the baseline.",
                )
                return
            model = as_composite_model(settings["baseline_model"])
        elif kind == "peaks":
            baseline_index = view.fit_history.active.get("baseline")
            try:
                specs = view.peak_specs()
            except ValueError as exc:
                window._alc_notify(title, str(exc))
                return
            if not specs:
                window._alc_notify(title, "Add at least one peak (+ Gaussian / + Lorentzian).")
                return
            baseline = view.fit_history.records[baseline_index].baseline_result()
            model = as_composite_model([spec["component"] for spec in specs])
            initial = {
                model.component_param_name(i, local): spec[local]
                for i, spec in enumerate(specs)
                for local in ("f", "B0", "Bwid")
            }
        else:
            model = as_composite_model("RFResonanceMuP")
        scan = restore_scan(scan_support(scan))
        job = ScanFitJob(self, kind, scan, settings, model, initial, baseline, baseline_index)
        if self.current is None:
            self.launch(job)
        else:
            self.stop()
            self.pending = job
            self.controls.progress_label.setText("Stopping fit; replacement queued…")
