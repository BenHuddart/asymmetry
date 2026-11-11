"""Bounded acquired identity checks for a requested independent count sum."""

from dataclasses import replace

from asymmetry.core.instrument import detector_label_key
from asymmetry.core.transform.corrections import (
    independent_observations,
    reference_observation_metadata,
)


def combination_members(run):
    """Borrow active originals, keeping authored sets separate from data periods."""
    originals = run.grouping.get("period_inputs")
    if originals is not None:
        selected = (
            {"red", "green"}
            if run.grouping.get("period_histograms")
            or run.grouping.get("period_mode") in {"green_plus_red", "green_minus_red"}
            else {run.grouping.get("period_mode", "red")}
        )
        mapping = run.grouping["period_mapping"]
        return [
            original
            for number, (original, coefficient) in enumerate(
                zip(
                    originals,
                    run.grouping.get("constituent_coefficients", [1.0] * len(originals)),
                    strict=True,
                ),
                1,
            )
            if coefficient != 0 and mapping.get(str(number), mapping.get(number)) in selected
        ]
    periods = run.grouping.get("period_histograms")
    if periods:
        return [
            replace(run, histograms=histograms, metadata=dict(run.metadata, data_period_number=i))
            for i, histograms in enumerate(periods, 1)
        ]
    return [run]


def detector_roles(run):
    """Recognise established orientation labels; cosmetic/partial tables stay unknown.

    Histogram ordinals (including MusrRoot bank numbers) have no shared physical
    namespace. Only complete orientation tables support this bounded comparison.
    """
    labels = run.grouping.get("histogram_labels", run.metadata.get("histogram_labels", []))
    keys = tuple(detector_label_key(label) for label in labels)
    roles = {"forward", "backward", "up", "down", "left", "right"}
    return keys if len(keys) == len(run.histograms) and set(keys) <= roles else ()


def declared_period_labels(run):
    """Use only complete, distinct declared data-slot captions for implicit pairing."""
    declaration = run.metadata.get("period_declaration", {})
    slots = declaration.get("data_dae_slots", [])
    labels = str(declaration.get("labels", "")).split(";")
    if (
        not slots
        or len(labels) != len(declaration.get("types", []))
        or any(not 1 <= slot <= len(labels) for slot in slots)
    ):
        return ()
    selected = tuple(labels[slot - 1].strip().casefold() for slot in slots)
    return selected if all(selected) and len(set(selected)) == len(selected) else ()


def correspondence_error(runs):
    """Reject demonstrated conflicts/reuse, leaving absent facts as recorded assumptions."""
    members = [combination_members(run) for run in runs]
    known = [(run, detector_roles(run)) for group in members for run in group]
    known = [(run, roles) for run, roles in known if roles]
    if known:
        base, expected = known[0]
        for run, roles in known[1:]:
            if roles != expected:
                return (
                    f"Known detector correspondence conflicts between runs {base.run_number} "
                    f"and {run.run_number}: {expected} versus {roles}. "
                    "Select acquisitions with corresponding detector order; grouping cannot "
                    "remap physical counters for count addition."
                )
    implicit = []
    for run in runs:
        if not run.grouping.get("period_histograms"):
            continue
        originals = run.grouping.get("period_inputs")
        if originals is None:
            implicit.append(declared_period_labels(run))
        else:
            weights = run.grouping.get("constituent_coefficients", (1.0,) * len(originals))
            implicit.extend(
                declared_period_labels(originals[number - 1])
                for number in run.grouping.get("implicit_period_inputs", ())
                if weights[number - 1] != 0
            )
    for labels in implicit:
        if labels and any(
            other and set(labels) == set(other) and labels != other for other in implicit
        ):
            return (
                "Known declared period order conflicts with implicit counterpart pairing; "
                "select corresponding periods explicitly or use Map periods…."
            )
    observations = [run for group in members for run in group]
    for i, run in enumerate(observations):
        if any(
            not independent_observations(
                run.histograms,
                reference_observation_metadata(run),
                other.histograms,
                reference_observation_metadata(other),
            )
            for other in observations[:i]
        ):
            return "Run arithmetic requires independent observations; a source is reused"
    return None
