"""Modeless presentation of a captured conditional simulation/refit experiment."""

from __future__ import annotations

import copy

import numpy as np
from PySide6.QtCore import QObject
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from asymmetry.core.pull_diagnostic import PullDistribution, Refit, run_pull_request
from asymmetry.core.pull_refit import CapturedPullFit
from asymmetry.core.pull_request import PullRequest
from asymmetry.gui.tasks import TaskRunner, TaskWorker
from asymmetry.gui.widgets.mpl_canvas import create_canvas


class _PullDelivery(QObject):
    """Bind queued terminals to their worker, including after close and recovery."""

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.worker: TaskWorker | None = None

    def finished(self, result):
        if self.worker is self.window._worker:
            self.window._on_diagnostic_finished(result)

    def error(self, message):
        if self.worker is self.window._worker:
            self.window._on_diagnostic_error(message)

    def progress(self, current, total, message):
        if self.worker is self.window._worker:
            self.window._on_diagnostic_progress(current, total, message)


class PullDiagnosticWindow(QDialog):
    """Owned captured inputs, worker cancellation and full-denominator histograms."""

    def __init__(
        self,
        *,
        request: PullRequest | CapturedPullFit,
        model=None,
        refit: Refit | None = None,
        parent=None,
    ):
        super().__init__(parent)
        if isinstance(request, PullRequest) and (model is None or refit is None):
            raise ValueError("A prepared pull request requires its supplied model and refit.")
        self.setWindowTitle("Pull Distribution Diagnostic")
        self.resize(850, 650)
        self._request = request
        self._model = copy.deepcopy(model)
        self._refit = copy.deepcopy(refit)
        self._last_result: PullDistribution | None = None
        self._tasks = TaskRunner(self)
        self._worker: TaskWorker | None = None
        layout = QVBoxLayout(self)
        hint = QLabel(
            "Conditional ideal Poisson simulation at fitted plug-in values. "
            "Deadtime distortion and deadtime/background correction are omitted. "
            "Grouping, timing, alpha and flat background are fixed. "
            "Mean and sample width describe the usable subset; they do not certify bias or coverage."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        config = request.to_dict() if isinstance(request, PullRequest) else request.presentation()
        details = QLabel(
            f"Captured source run {config['source_context']['run_number']}; "
            f"fit window {config['time_range']}; symmetric reported errors. "
            "Refits optimise laboratory parameters; source frequency offsets are omitted. "
            "Reference SEs assume independent N(0,1) pulls."
        )
        details.setWordWrap(True)
        layout.addWidget(details)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Seeds:"))
        self._seeds_spin = QSpinBox()
        self._seeds_spin.setRange(1, 2000)
        self._seeds_spin.setValue(config["n_seeds"])
        controls.addWidget(self._seeds_spin)
        controls.addWidget(QLabel("First seed:"))
        self._seed_start_spin = QSpinBox()
        self._seed_start_spin.setRange(0, 2**31 - 1)
        self._seed_start_spin.setValue(config["seed_start"])
        controls.addWidget(self._seed_start_spin)
        self._run_button = QPushButton("Run")
        self._run_button.clicked.connect(self._on_run)
        controls.addWidget(self._run_button)
        self._cancel_button = QPushButton("Cancel")
        self._cancel_button.clicked.connect(self._on_cancel)
        self._cancel_button.setEnabled(False)
        controls.addWidget(self._cancel_button)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        controls.addWidget(self._progress, stretch=1)
        layout.addLayout(controls)
        self._figure, self._canvas = create_canvas(figsize=(8, 3.2))
        layout.addWidget(self._canvas, stretch=1)
        self._verdict_label = QLabel("Run the conditional simulation to inspect parameter pulls.")
        self._verdict_label.setWordWrap(True)
        layout.addWidget(self._verdict_label)
        self._button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self._button_box.rejected.connect(self.reject)
        layout.addWidget(self._button_box)

    def _on_cancel(self):
        if self._worker is not None:
            self._worker.cancel()
        self._cancel_button.setEnabled(False)
        self._verdict_label.setText("Stopping at a cooperative boundary…")

    def reject(self):
        self._finish_run()
        self._tasks.shutdown()
        super().reject()

    def closeEvent(self, event):  # noqa: N802
        self.reject()
        event.accept()

    def _on_run(self):
        if self._worker is not None:
            return
        source = self._request
        n_seeds, seed_start = self._seeds_spin.value(), self._seed_start_spin.value()
        model, refit = self._model, self._refit
        self._run_button.setEnabled(False)
        self._cancel_button.setEnabled(True)
        self._progress.setValue(0)
        self._verdict_label.setText("Running…")

        def task(worker):
            if isinstance(source, CapturedPullFit):
                request, engine_refit = source.prepare()
                generating_model, fitting = engine_refit.model, engine_refit
            else:
                request, generating_model, fitting = source, model, refit
            config = request.to_dict()
            config.update(n_seeds=n_seeds, seed_start=seed_start)
            request = PullRequest.from_dict(config)
            return run_pull_request(
                request,
                generating_model,
                fitting,
                progress=lambda done, total: worker.progress.emit(done, total, ""),
                should_continue=lambda: not worker.is_cancelled(),
            )

        delivery = _PullDelivery(self)
        self._worker = self._tasks.start(
            task,
            on_finished=delivery.finished,
            on_error=delivery.error,
            on_progress=delivery.progress,
        )
        delivery.worker = self._worker

    def _on_diagnostic_progress(self, current, total, _message):
        self._progress.setValue(round(100 * current / total))

    def _finish_run(self):
        self._worker = None
        self._run_button.setEnabled(True)
        self._cancel_button.setEnabled(False)

    def _on_diagnostic_finished(self, result):
        self._last_result = result
        self._finish_run()
        self._plot(result)
        self._verdict_label.setText(result.verdict())

    def _on_diagnostic_error(self, message):
        self._finish_run()
        self._verdict_label.setText(f"Diagnostic failed: {message}")

    def _plot(self, result):
        self._figure.clear()
        tracked = [p for p in result.parameters.values() if p.n]
        if tracked:
            axes = self._figure.subplots(1, len(tracked), squeeze=False)[0]
            for ax, pull in zip(axes, tracked, strict=True):
                # Automatic edges include all tails; density uses all usable pulls.
                scale = max(1.0, float(np.max(np.abs(pull.pulls))))
                ax.hist(pull.pulls / scale, bins=20, density=True, alpha=0.6, color="#4477aa")
                grid = np.linspace(-4.0, 4.0, 400)
                ax.plot(
                    grid / scale,
                    scale * np.exp(-0.5 * grid**2) / np.sqrt(2 * np.pi),
                    color="#cc3311",
                    label="N(0,1)",
                )
                ax.set_title(f"{pull.name}: mean {pull.mean:.2g}\nSD {pull.width:.2g}; N={pull.n}")
                ax.set_xlabel(f"pull / {scale:g}" if scale != 1 else "pull")
                ax.grid(alpha=0.3)
            axes[0].set_ylabel("density")
        self._canvas.draw_idle()

    @property
    def last_result(self):
        return self._last_result
