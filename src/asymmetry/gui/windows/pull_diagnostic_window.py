"""Pull-distribution diagnostic window (launched from a completed single fit).

Re-simulates the fitted run many times at matched statistics, refits each, and
histograms the parameter pulls (θ̂ − θ_true)/σ_θ̂ against the N(0, 1) null. A
mean at zero means the fit is unbiased; a width at one means the reported
errors are honest. The heavy lifting is :func:`asymmetry.core.pull_diagnostic`;
this window only collects the inputs, drives the run on a
:class:`~asymmetry.gui.tasks.TaskRunner` worker thread with a progress bar and
draws the histograms and verdict.

The run can take up to 2000 simulate+refit iterations — potentially minutes —
so the "Run" button starts it on a background thread via ``TaskRunner`` instead
of blocking the GUI thread. ``run_pull_distribution`` accepts a
``progress(done, total)`` callback and a ``should_continue()`` poll instead of
raising a cancel exception, so the worker task adapts them to
``worker.progress.emit`` / ``worker.is_cancelled`` and cancellation surfaces as
a normal (shorter) result on the ``finished`` signal rather than ``cancelled``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
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

from asymmetry.core.data.dataset import Run
from asymmetry.core.pull_diagnostic import PullDistribution, Refit, run_pull_distribution
from asymmetry.gui.tasks import TaskRunner, TaskWorker


class PullDiagnosticWindow(QDialog):
    """Modeless diagnostic: pull histograms with an N(0, 1) overlay."""

    def __init__(
        self,
        *,
        template: Run,
        model: Any,
        parameters: Mapping[str, float],
        refit: Refit,
        track: Sequence[str],
        total_events: float,
        background_per_bin: float = 0.0,
        time_range: tuple[float | None, float | None] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Pull Distribution Diagnostic")
        self.resize(720, 460)

        self._template = template
        self._model = model
        self._parameters = dict(parameters)
        self._refit = refit
        self._track = list(track)
        self._total_events = float(total_events)
        self._background_per_bin = float(background_per_bin)
        self._time_range = time_range
        self._last_result: PullDistribution | None = None
        self._cancelled = False
        self._running = False
        self._cancel_requested = False
        self._tasks = TaskRunner(self)
        self._worker: TaskWorker | None = None

        layout = QVBoxLayout(self)

        hint = QLabel(
            "Re-simulates this fit's run at matched statistics, refits each "
            "synthetic copy, and histograms the pulls (θ̂ − θ_true)/σ. Honest "
            "errors give a width of 1; below 1 is over-estimated, above 1 "
            "under-estimated."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Seeds:"))
        self._seeds_spin = QSpinBox()
        self._seeds_spin.setRange(10, 2000)
        self._seeds_spin.setValue(200)
        self._seeds_spin.setSingleStep(50)
        controls.addWidget(self._seeds_spin)
        self._run_button = QPushButton("Run")
        self._run_button.clicked.connect(self._on_run)
        controls.addWidget(self._run_button)
        self._cancel_button = QPushButton("Cancel")
        self._cancel_button.clicked.connect(self._on_cancel)
        self._cancel_button.setEnabled(False)
        controls.addWidget(self._cancel_button)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        controls.addWidget(self._progress, stretch=1)
        layout.addLayout(controls)

        self._canvas = None
        self._figure = None
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure

            self._figure = Figure(figsize=(7.0, 3.2))
            self._canvas = FigureCanvas(self._figure)
            layout.addWidget(self._canvas, stretch=1)
        except Exception:
            layout.addWidget(QLabel("matplotlib unavailable — histograms disabled."))

        self._verdict_label = QLabel("Run the diagnostic to assess error calibration.")
        self._verdict_label.setWordWrap(True)
        layout.addWidget(self._verdict_label)

        self._button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self._button_box.rejected.connect(self.reject)
        self._button_box.accepted.connect(self.accept)
        layout.addWidget(self._button_box)

    # ------------------------------------------------------------------

    def run_diagnostic(self, n_seeds: int) -> PullDistribution:
        """Run the pull distribution for ``n_seeds`` synchronously and store/return it.

        This is a blocking helper for scripting/tests — it runs on the calling
        thread and does not touch the progress bar or button states. The "Run"
        button never calls this; it starts the same underlying
        :func:`~asymmetry.core.pull_diagnostic.run_pull_distribution` call on a
        :class:`~asymmetry.gui.tasks.TaskRunner` worker via :meth:`_on_run`
        instead, so the GUI thread stays responsive.

        Polls the ``self._cancelled`` flag before each seed (set by
        :meth:`reject`), so a direct caller can still request early
        termination.
        """
        result = run_pull_distribution(
            self._template,
            self._model,
            self._parameters,
            self._refit,
            total_events=self._total_events,
            n_seeds=n_seeds,
            track=self._track,
            background_per_bin=self._background_per_bin,
            time_range=self._time_range,
            should_continue=lambda: not self._cancelled,
        )
        self._last_result = result
        return result

    def _on_cancel(self) -> None:
        self._cancelled = True
        self._cancel_requested = True
        if self._worker is not None:
            self._worker.cancel()
        self._cancel_button.setEnabled(False)

    def reject(self) -> None:
        # Closing mid-run cancels the worker instead of leaving it running
        # against a dismissed window, then waits for it to unwind before the
        # dialog actually goes away. The Close button routes here directly
        # (QDialogButtonBox.rejected → self.reject) without going through
        # closeEvent, so the shutdown must happen here, not only there.
        self._cancelled = True
        if self._worker is not None:
            self._worker.cancel()
        self._tasks.shutdown()
        super().reject()

    def closeEvent(self, event: object) -> None:  # noqa: N802 — Qt override
        # Belt-and-suspenders for a native window-close (title bar / Escape)
        # that reaches closeEvent without going through reject()'s explicit
        # button wiring; shutdown() is idempotent so this never double-blocks.
        if self._worker is not None:
            self._worker.cancel()
        self._tasks.shutdown()
        super().closeEvent(event)

    def _on_run(self) -> None:
        if self._running:
            return
        n_seeds = self._seeds_spin.value()
        # Snapshot every input the worker touches into plain locals before
        # starting the thread — the worker callable must never read back
        # through self, only the arguments and its own progress/cancel hooks.
        template = self._template
        model = self._model
        parameters = dict(self._parameters)
        refit = self._refit
        track = list(self._track)
        total_events = self._total_events
        background_per_bin = self._background_per_bin
        time_range = self._time_range

        self._cancelled = False
        self._cancel_requested = False
        self._running = True
        self._run_button.setEnabled(False)
        self._cancel_button.setEnabled(True)
        self._progress.setValue(0)
        self._verdict_label.setText("Running…")

        def task(worker: TaskWorker) -> PullDistribution:
            # run_pull_distribution has no dedicated cancel exception: it
            # polls should_continue() and, when it returns False, simply
            # breaks the loop and returns a normal (shorter) PullDistribution
            # instead of raising — so cancellation surfaces on `finished`,
            # never on `cancelled`.
            return run_pull_distribution(
                template,
                model,
                parameters,
                refit,
                total_events=total_events,
                n_seeds=n_seeds,
                track=track,
                background_per_bin=background_per_bin,
                time_range=time_range,
                progress=lambda done, total: worker.progress.emit(done, total, ""),
                should_continue=lambda: not worker.is_cancelled(),
            )

        self._worker = self._tasks.start(
            task,
            on_finished=self._on_diagnostic_finished,
            on_error=self._on_diagnostic_error,
            on_progress=self._on_diagnostic_progress,
        )

    def _on_diagnostic_progress(self, current: int, total: int, _message: str) -> None:
        if total > 0:
            self._progress.setValue(int(round(100 * current / total)))

    def _finish_run(self) -> None:
        self._worker = None
        self._running = False
        self._run_button.setEnabled(True)
        self._cancel_button.setEnabled(False)

    def _on_diagnostic_finished(self, result: PullDistribution) -> None:
        self._last_result = result
        self._finish_run()
        self._plot(result)
        verdict = result.verdict()
        if self._cancel_requested:
            verdict = f"Cancelled after {result.n_seeds} seed(s) — {verdict}"
        self._verdict_label.setText(verdict)

    def _on_diagnostic_error(self, message: str) -> None:
        self._finish_run()
        self._verdict_label.setText(f"Diagnostic failed: {message}")

    def _plot(self, result: PullDistribution) -> None:
        if self._figure is None or self._canvas is None:
            return
        self._figure.clear()
        tracked = [name for name in self._track if result.parameters[name].n > 1]
        if not tracked:
            self._canvas.draw_idle()
            return
        grid = np.linspace(-4.0, 4.0, 200)
        normal = np.exp(-0.5 * grid**2) / np.sqrt(2.0 * np.pi)
        axes = self._figure.subplots(1, len(tracked), squeeze=False)[0]
        for ax, name in zip(axes, tracked, strict=True):
            pull = result.parameters[name]
            ax.hist(
                pull.pulls, bins=20, range=(-4.0, 4.0), density=True, alpha=0.6, color="#4477aa"
            )
            ax.plot(grid, normal, color="#cc3311", lw=1.6, label="N(0, 1)")
            ax.set_title(f"{name}\nwidth {pull.width:.2f}({pull.width_uncertainty * 100:.0f})")
            ax.set_xlabel("pull")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("density")
        self._figure.tight_layout()
        self._canvas.draw_idle()

    @property
    def last_result(self) -> PullDistribution | None:
        return self._last_result


def make_engine_refit(
    model: Any,
    parameter_template: Any,
    *,
    t_min: float | None = None,
    t_max: float | None = None,
) -> Refit:
    """Build a :data:`Refit` that fits ``model`` with the fitting engine.

    ``parameter_template`` is a :class:`ParameterSet`-like object that is
    deep-copied for each fit (so start values, bounds and fixed flags are
    reused). Importing the engine here keeps it out of the core diagnostic and
    the import-time cost off the GUI's critical path.
    """
    import copy as _copy

    from asymmetry.core.fitting.engine import FitEngine

    model_fn = model.function if hasattr(model, "function") else model
    engine = FitEngine()

    def refit(dataset) -> tuple[dict[str, float], dict[str, float]] | None:
        params = _copy.deepcopy(parameter_template)
        try:
            result = engine.fit(dataset, model_fn, params, t_min=t_min, t_max=t_max)
        except Exception:
            return None
        if not result.success:
            return None
        values = {p.name: p.value for p in result.parameters}
        return values, dict(result.uncertainties)

    return refit
