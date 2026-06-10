"""Pull-distribution diagnostic window (launched from a completed single fit).

Re-simulates the fitted run many times at matched statistics, refits each, and
histograms the parameter pulls (θ̂ − θ_true)/σ_θ̂ against the N(0, 1) null. A
mean at zero means the fit is unbiased; a width at one means the reported
errors are honest. The heavy lifting is :func:`asymmetry.core.pull_diagnostic`;
this window only collects the inputs, drives the loop with a progress bar and
draws the histograms and verdict.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from PySide6.QtWidgets import (
    QApplication,
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
        """Run the pull distribution for ``n_seeds`` and store/return it.

        Polls a cancel flag before each seed (set by the Cancel button or by
        closing the window mid-run), so the loop can stop early instead of
        freezing the UI until all seeds finish.
        """

        def progress(done: int, total: int) -> None:
            self._progress.setValue(int(round(100 * done / total)))
            QApplication.processEvents()

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
            progress=progress,
            should_continue=lambda: not self._cancelled,
        )
        self._last_result = result
        return result

    def _on_cancel(self) -> None:
        self._cancelled = True

    def reject(self) -> None:
        # Closing mid-run aborts the loop instead of leaving it running against
        # a dismissed window.
        self._cancelled = True
        super().reject()

    def _on_run(self) -> None:
        self._cancelled = False
        self._running = True
        self._run_button.setEnabled(False)
        self._cancel_button.setEnabled(True)
        # The Close button must not dismiss the window mid-loop (processEvents
        # would deliver the reject while the run is still touching widgets).
        self._button_box.setEnabled(False)
        self._progress.setValue(0)
        try:
            result = self.run_diagnostic(self._seeds_spin.value())
        finally:
            self._running = False
            self._run_button.setEnabled(True)
            self._cancel_button.setEnabled(False)
            self._button_box.setEnabled(True)
        self._plot(result)
        self._verdict_label.setText(result.verdict())

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
