"""Prepare effective instrument/source authority before a loader's quick-look."""

from dataclasses import dataclass, replace

from asymmetry.core.data.dataset import Run
from asymmetry.core.instruments.resolution import (
    InstrumentSelection,
    RecordedInstrumentChoice,
    resolve_instrument,
)
from asymmetry.core.instruments.source import SourceAssessment, assess_source, source_for_run


@dataclass(frozen=True)
class ReplayState:
    """One load purpose's retained authority; an omitted source is legacy replay."""

    instrument_choice: RecordedInstrumentChoice | None = None
    source: SourceAssessment | None = None
    run_number: int | None = None
    period_number: int | None = None

    def __post_init__(self):
        if self.instrument_choice is not None and not isinstance(
            self.instrument_choice, RecordedInstrumentChoice
        ):
            raise ValueError("Replay state requires a typed recorded instrument choice")
        if self.source is not None and not isinstance(self.source, SourceAssessment):
            raise ValueError("Replay state requires a typed captured source")
        if self.run_number is not None and type(self.run_number) is not int:
            raise ValueError("Replay run number must be an integer")
        if self.period_number is not None and (
            type(self.period_number) is not int or self.period_number < 1
        ):
            raise ValueError("Replay acquired period must be a positive integer")

    def selects(self, run: Run) -> bool:
        from asymmetry.core.io.reference_selection import reference_selection

        selected = reference_selection(run)
        exact = self.run_number == run.run_number
        if (
            exact
            and selected["run_number"] != run.run_number
            and self.period_number is not None
            and self.period_number != selected["period_number"]
        ):
            raise ValueError("Replay encoded run and acquired period selectors contradict")
        return (
            self.run_number is None or self.run_number in {run.run_number, selected["run_number"]}
        ) and (self.period_number is None or self.period_number == selected["period_number"])


def prepare_loaded_run(run: Run, replay: ReplayState | None = None) -> Run:
    """Capture fresh authority or replay a retained selection before scientific preparation."""
    if replay is not None:
        from asymmetry.core.project.grouping_request import restore_instrument_state

        selected = replay.selects(run)
        return restore_instrument_state(
            run,
            replay.instrument_choice if selected else None,
            replay.source if selected else None,
        )
    context = resolve_instrument(run)
    choice = context.recorded
    if choice is None and context.selection is not None:
        choice = RecordedInstrumentChoice(context.selection, context.observed.evidence)
    return replace(run, instrument_choice=choice, source_assessment=source_for_run(run))


def prepare_legacy_loaded_run(run: Run, replay: ReplayState | None) -> None:
    """Restore no-op captures after an opaque decoder without reinterpreting its arrays."""
    source = run.source_assessment
    if source is None:
        source = assess_source(run.metadata)
        if replay is not None:
            source = replace(source, origin="legacy:" + source.origin)
    if replay is not None and replay.selects(run):
        context = resolve_instrument(run)
        requested = replay.instrument_choice
        current = context.recorded.selection if context.recorded is not None else context.selection
        selected = requested.selection if requested is not None else current
        same_identity = (
            selected.identity == current.identity
            if isinstance(selected, InstrumentSelection)
            and isinstance(current, InstrumentSelection)
            else selected == current
        )
        if (replay.source is not None and replay.source.pulsed is not source.pulsed) or (
            requested is not None and (context.requires_choice or not same_identity)
        ):
            raise ValueError(
                "The selected loader cannot honor changed captured replay authority after "
                "decoding; use a loader supporting replay before quick-look or open history only"
            )
        if replay.source is not None:
            source = replay.source
        if requested is not None:
            run.instrument_choice = requested
    run.source_assessment = source
    for original in run.grouping.get("period_inputs", ()):
        prepare_legacy_loaded_run(original, replay)
