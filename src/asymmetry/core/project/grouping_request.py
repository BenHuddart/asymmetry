"""Data-only project transport for a run's complete authored grouping request."""

from dataclasses import replace

from asymmetry.core.data.dataset import Run
from asymmetry.core.instruments.defaults import retained_grouping_provenance
from asymmetry.core.instruments.resolution import RecordedInstrumentChoice, resolve_instrument
from asymmetry.core.instruments.source import assess_source, source_for_run
from asymmetry.core.project.profiles import _PER_RUN_FACT_KEYS, grouping_request_payload
from asymmetry.core.transform.rebin import binning_spec
from asymmetry.core.transform.reduce import correction_flags_from_grouping


def persistent_grouping_for_run(run: Run) -> dict | None:
    """Retain request extensions and timing coordinates without resolving a correction.

    Raw held-period buffers stay with the acquisition; file deadtime is read from
    the current file rather than promoted to a manual override by saving.
    """
    grouping = run.grouping
    if not grouping:
        return None
    payload = grouping_request_payload(grouping)
    t0 = int(grouping.get("t0_bin", run.histograms[0].t0_bin if run.histograms else 0))
    offset = int(grouping.get("t_good_offset", int(grouping.get("first_good_bin", t0)) - t0))
    mode = correction_flags_from_grouping(grouping).deadtime_mode
    payload.update(
        t0_bin=t0,
        t_good_offset=offset,
        first_good_bin=int(grouping.get("first_good_bin", t0 + offset)),
        last_good_bin=int(grouping.get("last_good_bin", 0)),
        bin_index_base=1 if int(grouping.get("bin_index_base", 0)) == 1 else 0,
        bunching_factor=binning_spec(grouping).factor,
        deadtime_mode=mode,
        deadtime_correction=bool(grouping.get("deadtime_correction", False)),
        background_correction=bool(grouping.get("background_correction", False)),
    )
    if mode == "from_file":
        payload.pop("dead_time_us", None)
    return payload


def instrument_state_for_run(run: Run) -> dict:
    """Serialize the selected identity/source independently of the scientific grouping."""
    context = resolve_instrument(run)
    choice = context.recorded
    if choice is None and context.selection is not None:
        choice = RecordedInstrumentChoice(context.selection, context.observed.evidence)
    return {
        **({"instrument_choice": choice.to_dict()} if choice is not None else {}),
        "source_assessment": source_for_run(run).to_dict(),
    }


def restore_instrument_state(run: Run, choice, source) -> Run:
    """An omitted legacy source uses raw hints; a captured unknown stays unknown."""
    if source is None:
        inferred = assess_source(run.metadata)
        source = replace(inferred, origin="legacy:" + inferred.origin)
    return replace(run, instrument_choice=choice, source_assessment=source)


def merge_grouping_request(current: dict, requested: dict) -> dict:
    """Keep acquired evidence ahead of saved tables while retaining request extensions."""
    incoming = grouping_request_payload(requested)
    incoming.pop("nexus_grouping_declaration", None)
    for key in _PER_RUN_FACT_KEYS:
        if key in current:
            incoming.pop(key, None)
    merged = dict(current) | incoming
    if "grouping_provenance" not in incoming:
        merged.pop("grouping_provenance", None)
        merged.update(retained_grouping_provenance(merged, current))
    return merged
