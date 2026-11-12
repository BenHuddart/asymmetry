"""Cancellable scan construction and publication for the integral-scan surface."""

from __future__ import annotations

from dataclasses import replace
from functools import partial

from PySide6.QtCore import QEventLoop, QObject, Qt
from PySide6.QtWidgets import QProgressDialog

from asymmetry.core.io.periods import build_rf_difference_scan
from asymmetry.core.representation import RepresentationType
from asymmetry.core.representation.series import FitSeries
from asymmetry.core.transform.integral import _resolve_run, build_field_scan
from asymmetry.core.utils.perf import perf_timer
from asymmetry.gui.tasks import TaskRunner, check_task_cancelled


class ScanJob(QObject):
    """One immutable request, GUI-thread delivery, and bounded worker lifetime."""

    def __init__(self, controller, datasets, bounds, rf_mode):
        super().__init__(controller)
        self.datasets, self.bounds, self.rf_mode = datasets, bounds, rf_mode
        self.runner = TaskRunner(self)
        self.loop = QEventLoop(self)
        self.dialog = QProgressDialog(
            "Building count-integral scan…", "Cancel", 0, len(datasets), controller.window
        )
        self.dialog.setWindowTitle("Integral scan")
        self.dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.dialog.setMinimumDuration(0)
        self.dialog.setAutoClose(False)
        self.result = None
        self.message = None
        self.worker = None
        self.dialog.canceled.connect(self.cancel)

    def compute(self, worker):
        builder = build_rf_difference_scan if self.rf_mode else build_field_scan
        # Retained observations own read-only arrays and resolved settings.
        with perf_timer("core.scan.construct", n_runs=len(self.datasets)):
            return builder(
                self.datasets,
                t_min=self.bounds[0],
                t_max=self.bounds[1],
                order_key="run",
                cancel_check=partial(check_task_cancelled, worker),
                progress_callback=worker.progress.emit,
            )

    def progress(self, current, total, message):
        if self.loop.isRunning():
            self.dialog.setMaximum(total)
            self.dialog.setValue(current)
            self.dialog.setLabelText(message)

    def finished(self, result):
        self.result = result
        self.loop.quit()

    def error(self, message):
        self.message = message
        self.loop.quit()

    def cancelled(self):
        self.loop.quit()

    def cancel(self):
        if self.worker is not None:
            self.worker.cancel()

    def run(self):
        self.worker = self.runner.start(
            self.compute,
            on_finished=self.finished,
            on_error=self.error,
            on_cancelled=self.cancelled,
            on_progress=self.progress,
        )
        self.loop.exec()
        cancelled = self.worker is None or self.worker.is_cancelled()
        self.runner.shutdown()
        self.dialog.close()
        self.dialog.deleteLater()
        return None if cancelled else self.result


class ScanController(QObject):
    """Snapshot requests and retain the previous scan until a current build succeeds."""

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.generation = 0
        self.jobs = []

    def invalidate(self):
        self.generation += 1
        for job in self.jobs:
            job.cancel()

    def shutdown(self):
        self.invalidate()
        for job in self.jobs:
            job.worker = None
            job.runner.shutdown()
            job.loop.quit()

    def estimator(self):
        series = self.window._project_model.batch(self.window._alc_scan_series_id)
        return series.extra["estimator"] if series is not None else "legacy_unspecified"

    def request_identity(self):
        window = self.window
        return (
            window._project_model,
            tuple(
                (
                    id(d),
                    id(d.histogram_observation),
                    tuple(
                        (
                            _resolve_run(d).metadata
                            if d.run is not None or d.histogram_observation is not None
                            else d.metadata
                        ).get(key)
                        for key in ("field", "temperature")
                    ),
                )
                for d in window._fit_panel.batch_datasets()
            ),
            window._plot_panel.get_fit_range(),
            window._alc_fit_panel.rf_difference_enabled(),
            window._active_representation_type(),
        )

    def build(self):
        window = self.window
        if window._active_representation_type() != RepresentationType.TIME_FB_ASYMMETRY:
            window._guidance_message(
                "Integral scan",
                "Integral-scan (ALC) mode applies to the Forward-Backward asymmetry only.",
            )
            return
        datasets = window._fit_panel.batch_datasets()
        if len(datasets) < 2:
            window._guidance_message(
                "Integral scan", "Select at least two runs to build a field scan."
            )
            return
        self.invalidate()
        generation, identity = self.generation, self.request_identity()
        t_min, t_max = window._plot_panel.get_fit_range()
        rf_mode = window._alc_fit_panel.rf_difference_enabled()
        datasets = tuple(
            replace(
                d,
                metadata=dict(d.metadata),
                run=_resolve_run(d) if d.histogram_observation is not None else d.run,
            )
            for d in datasets
        )
        job = ScanJob(self, datasets, (t_min, t_max), rf_mode)
        self.jobs.append(job)
        scan = job.run()
        self.jobs.remove(job)
        message = job.message
        job.deleteLater()
        if generation != self.generation or identity != self.request_identity():
            return
        if message is not None:
            window._guidance_message(
                "Integral scan", f"Could not build the scan: {message}", warning=True
            )
            return
        if scan is None:
            return
        if scan.n_points < 2:
            # A failed rebuild must retain the displayed scan's RF mode.
            window._guidance_message(
                "Integral scan",
                "The scan has too few usable points to plot. "
                + "; ".join(f"{n}: {reason}" for n, reason in scan.excluded),
                warning=True,
            )
            return
        window._alc_rf_mode = rf_mode

        # Build-time drops surface in the scan's provenance line (and the log).
        window._alc_build_excluded = list(scan.excluded)
        if scan.excluded:
            dropped = ", ".join(f"{run} ({reason})" for run, reason in scan.excluded)
            window._log_panel.log(f"Integral scan: excluded {len(scan.excluded)} run(s): {dropped}")
        # Run-keyed exclusions survive rebuilds over the same selection.
        window._alc_excluded_runs &= {int(run) for run in scan.run_numbers}

        # Native integral values are fractions; display conversion happens later.
        runs_by_number = {int(d.run.run_number): d.run for d in datasets if d.run is not None}
        window._alc_scan_points = [
            window._alc_scan_point(
                run_number,
                value,
                error,
                runs_by_number.get(int(run_number)),
                {
                    label: record.to_dict()
                    for label, record in scan.correction_outcomes.get(run_number, {}).items()
                },
                scan.observations[index].to_dict() if scan.observations else None,
                index,
            )
            for index, (run_number, value, error) in enumerate(
                zip(scan.run_numbers, scan.value, scan.error)
            )
        ]

        # FitSeries stores scan values in percent; native integrals are fractions.
        quantity = window._SCAN_QUANTITY
        results_by_run = {
            int(run_number): {
                "success": True,
                "scan_index": index,
                "observation": scan.observations[index].to_dict() if scan.observations else None,
                "correction_outcomes": {
                    label: record.to_dict()
                    for label, record in scan.correction_outcomes.get(run_number, {}).items()
                },
                "parameters": {quantity: float(value) * 100.0},
                "uncertainties": {quantity: float(error) * 100.0},
                "chi_squared": 0.0,
                "reduced_chi_squared": 0.0,
            }
            for index, (run_number, value, error) in enumerate(
                zip(scan.run_numbers, scan.value, scan.error)
            )
        }
        scan_kind = "RF scan" if rf_mode else "Integral scan"
        series = FitSeries(
            f"scan-{window._next_scan_index}",
            RepresentationType.TIME_FB_ASYMMETRY,
            label=f"{scan_kind} {window._next_scan_index}",
            member_run_numbers=list(scan.run_numbers),
            order_key="run",  # built/sorted by run; the view picks the display axis
            canonical_model=None,
            param_roles={},
            extra={
                "estimator": scan.method,
                "time_window_us": [t_min, t_max],
                "correlated_pairs": [list(pair) for pair in scan.correlated_pairs],
            },
            results_by_run=results_by_run,
        )
        window._next_scan_index += 1
        series.sort_members(runs_by_number)
        # One scan series owns the active integration window.
        if window._alc_scan_series_id and window._project_model.batch(window._alc_scan_series_id):
            window._project_model.remove_batch(window._alc_scan_series_id)
        window._project_model.add_batch(series)
        window._alc_scan_series_id = series.batch_id
        # A successful scan changes the persistent project.
        window._mark_dirty()

        # Regions/peaks use the old x units; a new scan requires new selections.
        window._alc_scan_view.clear_analysis()
        window._render_alc_scan()
        window._log_panel.log(f"Built integral scan '{series.label}' ({scan.n_points} points).")

        # Parameters holds the scan's Baseline/Peaks/RF controls.
        window._show_panel("fit_parameters")
