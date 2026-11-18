"""Detached spectral-moment inputs and cancellable GUI task ownership."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QObject

from asymmetry.core.fourier.moments import (
    MomentsCancelledError,
    moments_trend_row,
    spectrum_moments,
)
from asymmetry.core.fourier.units import FieldUnit, convert
from asymmetry.gui.tasks import TaskRunner, check_task_cancelled


@dataclass(frozen=True)
class MomentRecipe:
    unit: FieldUnit
    range_mhz: tuple[float, float] | None
    cutoff_fraction: float
    mode: str
    seed: int = 0

    @classmethod
    def from_widget(cls, widget, rep_type):
        return cls(widget.unit(), widget.range_mhz(), widget.cutoff_fraction(), rep_type.value)

    def as_dict(self):
        return {
            "unit": self.unit.value,
            "range_mhz": None if self.range_mhz is None else list(self.range_mhz),
            "cutoff_fraction": self.cutoff_fraction,
            "rep_type": self.mode,
        }


def _same_spectrum_samples(left, right):
    return all(
        a is b if a is None or b is None else np.array_equal(a, b, equal_nan=True)
        for a, b in zip(left, right, strict=True)
    )


def spectrum_error_oversampling(dataset) -> float:
    """Read the spectrum's recorded padding heuristic at the GUI boundary."""
    padding = dataset.metadata.get("fourier_padding", 1) or 1
    try:
        padding = float(padding)
    except (ValueError, TypeError):
        padding = 1.0
    return padding if padding > 1 else 1.0


@dataclass(frozen=True, eq=False)
class MomentInput:
    source: tuple
    recipe: MomentRecipe
    frequency: np.ndarray
    amplitude: np.ndarray
    errors: np.ndarray | None
    oversampling: float
    run_number: int
    run_label: str
    field: float | None
    temperature: float | None

    @classmethod
    def capture(cls, dataset, accessor, recipe):
        frequency, amplitude, errors = accessor[:3]
        arrays = [
            np.array(a, dtype=float, copy=True) if a is not None else None
            for a in (frequency, amplitude, errors)
        ]
        for array in arrays:
            if array is not None:
                array.setflags(write=False)
        oversampling = spectrum_error_oversampling(dataset)
        source = (id(dataset), id(dataset.time), id(dataset.asymmetry), id(dataset.error))
        return cls(
            source,
            recipe,
            *arrays,
            oversampling,
            int(dataset.run_number),
            str(dataset.metadata.get("run_label") or dataset.run_number),
            dataset.field,
            dataset.temperature,
        )

    def same_spectrum(self, other):
        return (
            self.source == other.source
            and self.recipe == other.recipe
            and self.oversampling == other.oversampling
            and _same_spectrum_samples(
                (self.frequency, self.amplitude, self.errors),
                (other.frequency, other.amplitude, other.errors),
            )
        )

    def matches(self, dataset, recipe):
        source = (id(dataset), id(dataset.time), id(dataset.asymmetry), id(dataset.error))
        errors = dataset.error
        if errors is not None and errors.shape != dataset.time.shape:
            errors = None
        return (
            source == self.source
            and recipe == self.recipe
            and spectrum_error_oversampling(dataset) == self.oversampling
            and _same_spectrum_samples(
                (self.frequency, self.amplitude, self.errors),
                (dataset.time, dataset.asymmetry, errors),
            )
        )

    def compute(self, *, cheap=False, cancel_callback=None):
        recipe = self.recipe
        x = convert(self.frequency, FieldUnit.MHZ, recipe.unit)
        x_range = (
            None
            if recipe.range_mhz is None
            else tuple(
                float(convert(value, FieldUnit.MHZ, recipe.unit)) for value in recipe.range_mhz
            )
        )
        return spectrum_moments(
            x,
            self.amplitude,
            x_range=x_range,
            cutoff_fraction=recipe.cutoff_fraction,
            errors=self.errors,
            uncertainty="none" if cheap or self.errors is None else "bootstrap",
            unit=recipe.unit.value,
            mode=recipe.mode,
            seed=recipe.seed,
            error_oversampling=self.oversampling,
            cancel_callback=cancel_callback,
        )


class MomentReadout(QObject):
    """One full readout at a time, with one latest pending spectrum."""

    def __init__(self, parent, tasks: TaskRunner):
        super().__init__(parent)
        self._tasks = tasks
        self._worker = None
        self._running: MomentInput | None = None
        self._wanted: MomentInput | None = None

    def invalidate(self):
        self._wanted = None
        if self._worker is not None:
            self._worker.cancel()

    def request(self, request: MomentInput) -> bool:
        if self._wanted is not None and self._wanted.same_spectrum(request):
            return False
        self._wanted = request
        if self._worker is not None:
            self._worker.cancel()
        else:
            self._start()

        return True

    def _start(self):
        self._running = self._wanted
        request = self._running
        if request is None:
            return
        self._worker = self._tasks.start(
            lambda worker: request.compute(cancel_callback=worker.is_cancelled),
            on_finished=self._finished,
            on_error=self._error,
            on_cancelled=self._cancelled,
            cancel_exceptions=(MomentsCancelledError,),
        )

    def _finished(self, result):
        request = self._running
        if request is self._wanted:
            self.parent()._show_moment_result(request, result)
        self._settle()

    def _error(self, message):
        if self._running is self._wanted:
            self.parent()._show_moment_result(self._running, self._running.compute(cheap=True))
            self.parent()._set_fourier_status(f"Spectral moments: {message}")
            self._wanted = None
        self._settle()

    def _cancelled(self):
        self._settle()

    def _settle(self):
        previous = self._running
        self._worker = None
        self._running = None
        if self._wanted is not None and self._wanted is not previous:
            self._start()


@dataclass(frozen=True, eq=False)
class MomentSelection:
    runs: tuple[int, ...]
    recipe: MomentRecipe
    project: object


class MomentBatch(QObject):
    """Own preparation intent and publish a complete selected-run result once."""

    def __init__(self, parent, tasks):
        super().__init__(parent)
        self._tasks = tasks
        self._intent: MomentSelection | None = None
        self._worker = None
        self._running: MomentSelection | None = None
        self._pending: tuple | None = None

    def invalidate(self):
        self._intent = None
        self._pending = None
        if self._worker is not None:
            self._worker.cancel()

    def begin(self, runs, recipe, project):
        if self._intent is not None and (
            self._intent.runs == tuple(runs)
            and self._intent.recipe == recipe
            and self._intent.project is project
        ):
            return None
        self.invalidate()
        self._intent = MomentSelection(tuple(runs), recipe, project)
        return self._intent

    def is_current(self, intent):
        return (
            intent is not None
            and intent is self._intent
            and intent.project is self.parent()._project_model
        )

    def prepared(self, intent, inputs, skipped):
        if not self.is_current(intent):
            return
        self._pending = (intent, tuple(inputs), tuple(skipped))
        if self._worker is None:
            self._start()

    def _start(self):
        intent, inputs, skipped = self._pending
        self._pending = None
        self._running = intent

        def calculate(worker):
            results = {}
            excluded = list(skipped)
            for request in inputs:
                check_task_cancelled(worker)
                moments = request.compute(cancel_callback=worker.is_cancelled)
                if moments.n_sample == 0:
                    excluded.append(request.run_number)
                    continue
                results[request.run_number] = moments_trend_row(
                    moments,
                    run_number=request.run_number,
                    run_label=request.run_label,
                    field=request.field,
                    temperature=request.temperature,
                )
            check_task_cancelled(worker)
            return inputs, results, excluded

        self._worker = self._tasks.start(
            calculate,
            on_finished=self._finished,
            on_error=self._error,
            on_cancelled=self._cancelled,
            cancel_exceptions=(MomentsCancelledError,),
        )

    def _finished(self, result):
        if self.is_current(self._running):
            self.parent()._publish_moment_batch(self._running, *result)
        self._settle()

    def _error(self, message):
        if self.is_current(self._running):
            self.parent()._set_fourier_status(f"Spectral moments: {message}")
        self._settle()

    def _cancelled(self):
        self._settle()

    def _settle(self):
        if self._running is self._intent:
            self._intent = None
        self._worker = None
        self._running = None
        if self._pending is not None:
            self._start()
