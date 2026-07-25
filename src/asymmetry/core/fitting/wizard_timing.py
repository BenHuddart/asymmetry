"""Standard timing / progress instrumentation for the fit wizards.

Why this exists
---------------
A caller driving either wizard headless has to answer one operational question
that the recommendation object could not previously answer: *is this run slow,
or is it hung?*  The two demand opposite responses — wait, or kill — and from
outside they look identical.  Downstream analyses ended up reading the child
process tree's cumulative CPU out of ``/proc`` by hand to tell them apart.

That is a library gap, not a caller's job.  Every wizard entry point that
already accepts an ``instrumentation`` dict now populates a **timing block**
under the ``"timing"`` key with wall-clock and CPU totals plus a per-stage
breakdown, and can emit a structured :class:`WizardStageProgress` event as each
stage starts and finishes.  A caller can then implement a timeout on *progress*
("no stage event for N seconds") rather than on total runtime, and can report a
stall as a stall — with the CPU number that distinguishes it from slow work.

CPU accounting
--------------
:func:`os.times` gives user+system time for this process *and for reaped
children*.  The wizards' heavy stages run in a :class:`ProcessPoolExecutor`
whose workers are reaped at pool shutdown, so a stage that opens and closes its
own pool has its workers' CPU attributed to it.  Workers still alive when a
stage closes are not counted, which biases the number *down* — the safe
direction, since it can only make live work look more idle, never the reverse.
On platforms where child times are unavailable (Windows reports zero), the block
still carries wall-clock and this process's own CPU.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

__all__ = [
    "TIMING_KEY",
    "StageTimings",
    "WizardStageProgress",
    "record_stage_timing",
    "stage_timer",
    "timing_block",
]

#: Key under which the timing block is stored in an ``instrumentation`` dict.
TIMING_KEY = "timing"


@dataclass(frozen=True)
class WizardStageProgress:
    """One structured progress event from a wizard stage.

    Emitted when a stage starts, whenever it completes a unit of work, and when
    it ends.  ``items_done``/``items_total`` are ``None`` for stages that are not
    item-wise.
    """

    #: Stable stage identifier, e.g. ``"screening.single_fit_tables"``.
    stage: str
    #: Human-readable message — the same text the plain progress callback gets.
    message: str
    #: ``"start"``, ``"item"`` or ``"end"``.
    event: str
    items_done: int | None = None
    items_total: int | None = None
    #: Seconds since this stage started.
    elapsed_seconds: float = 0.0
    #: CPU seconds (this process + reaped children) burned since it started.
    cpu_seconds: float = 0.0

    @property
    def cpu_cores(self) -> float:
        """Average cores busy over the stage so far (``cpu / elapsed``)."""
        return self.cpu_seconds / self.elapsed_seconds if self.elapsed_seconds > 0.0 else 0.0

    @property
    def fraction_done(self) -> float | None:
        if not self.items_total:
            return None
        return float(self.items_done or 0) / float(self.items_total)


@dataclass
class StageTimings:
    """Accumulating wall-clock/CPU record for one wizard run."""

    started_wall: float = field(default_factory=time.monotonic)
    started_cpu: float = 0.0
    stages: list[dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.started_cpu = _cpu_seconds()

    def as_block(self) -> dict[str, object]:
        elapsed = max(time.monotonic() - self.started_wall, 0.0)
        cpu = max(_cpu_seconds() - self.started_cpu, 0.0)
        return {
            "elapsed_seconds": elapsed,
            "cpu_seconds": cpu,
            "cpu_cores": (cpu / elapsed) if elapsed > 0.0 else 0.0,
            "stages": list(self.stages),
        }


def _cpu_seconds() -> float:
    """User+system CPU for this process and its reaped children, in seconds."""
    times = os.times()
    return float(times.user + times.system + times.children_user + times.children_system)


def timing_block(instrumentation: dict[str, object] | None) -> dict[str, object] | None:
    """Return the timing block of an instrumentation dict, if it has one."""
    if instrumentation is None:
        return None
    block = instrumentation.get(TIMING_KEY)
    return block if isinstance(block, dict) else None


def record_stage_timing(
    instrumentation: dict[str, object] | None,
    *,
    stage: str,
    elapsed_seconds: float,
    cpu_seconds: float,
    items_total: int | None = None,
) -> None:
    """Append one finished stage to the instrumentation dict's timing block.

    Creates the block (and its running totals) on first use, so a caller only
    ever has to pass the same ``instrumentation`` dict it already passes.
    """
    if instrumentation is None:
        return
    block = instrumentation.get(TIMING_KEY)
    if not isinstance(block, dict):
        block = {"elapsed_seconds": 0.0, "cpu_seconds": 0.0, "cpu_cores": 0.0, "stages": []}
        instrumentation[TIMING_KEY] = block
    stages = block.setdefault("stages", [])
    if isinstance(stages, list):
        stages.append(
            {
                "stage": stage,
                "elapsed_seconds": float(elapsed_seconds),
                "cpu_seconds": float(cpu_seconds),
                "cpu_cores": (
                    float(cpu_seconds) / float(elapsed_seconds) if elapsed_seconds > 0.0 else 0.0
                ),
                "items_total": items_total,
            }
        )
    total_elapsed = float(block.get("elapsed_seconds", 0.0)) + float(elapsed_seconds)
    total_cpu = float(block.get("cpu_seconds", 0.0)) + float(cpu_seconds)
    block["elapsed_seconds"] = total_elapsed
    block["cpu_seconds"] = total_cpu
    block["cpu_cores"] = (total_cpu / total_elapsed) if total_elapsed > 0.0 else 0.0


@contextmanager
def stage_timer(
    instrumentation: dict[str, object] | None,
    stage: str,
    *,
    items_total: int | None = None,
    stage_callback: Callable[[WizardStageProgress], None] | None = None,
    message: str = "",
) -> Iterator[Callable[[int, str], None]]:
    """Time one wizard stage, emitting structured start/item/end events.

    Yields an ``advance(items_done, message)`` callable so an item-wise stage can
    report progress as it drains.  On exit — including on an exception, so a
    cancelled or crashed stage still leaves a record — the stage is appended to
    the instrumentation timing block and an ``"end"`` event is emitted.
    """
    start_wall = time.monotonic()
    start_cpu = _cpu_seconds()

    def _emit(event: str, items_done: int | None, text: str) -> None:
        if stage_callback is None:
            return
        stage_callback(
            WizardStageProgress(
                stage=stage,
                message=text or message or stage,
                event=event,
                items_done=items_done,
                items_total=items_total,
                elapsed_seconds=max(time.monotonic() - start_wall, 0.0),
                cpu_seconds=max(_cpu_seconds() - start_cpu, 0.0),
            )
        )

    _emit("start", 0 if items_total is not None else None, message)

    def advance(items_done: int, text: str = "") -> None:
        _emit("item", items_done, text)

    try:
        yield advance
    finally:
        elapsed = max(time.monotonic() - start_wall, 0.0)
        cpu = max(_cpu_seconds() - start_cpu, 0.0)
        record_stage_timing(
            instrumentation,
            stage=stage,
            elapsed_seconds=elapsed,
            cpu_seconds=cpu,
            items_total=items_total,
        )
        _emit("end", items_total, message)
