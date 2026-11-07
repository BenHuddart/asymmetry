"""Convert released reference declarations using acquired source facts at restore."""

from asymmetry.core.data.dataset import Run
from asymmetry.core.io.periods import (
    encode_period_run_number,
    reference_mapping_requires_replay,
    source_run_of,
)


def migrate_reference_declarations(value):
    """Mark historical keys for fact-checked conversion when acquisition data is available."""
    mapped = {
        (ds.get("source_file"), ds.get("run_number")): ds["grouping_overrides"]["period_mapping"]
        for ds in value.get("datasets", [])
        if isinstance(ds, dict)
        and isinstance(ds.get("grouping_overrides"), dict)
        and ds["grouping_overrides"].get("period_mapping")
    }

    def visit(item):
        if isinstance(item, list):
            return [visit(child) for child in item]
        if not isinstance(item, dict):
            return item
        result = {key: visit(child) for key, child in item.items()}
        payload = result.get("background_run")
        if isinstance(payload, dict) and "period_number" not in payload:
            number = payload.pop("run_number", None)
            payload["legacy_run_number"] = number
            mapping = mapped.get((payload.get("source_file"), number))
            if reference_mapping_requires_replay(mapping):
                payload["period_mapping"] = mapping
        return result

    return visit(value)


def migrate_reference_timing_declarations(value):
    """Released requests defer timing to their source; sample timing is never reference intent."""
    if isinstance(value, list):
        return [migrate_reference_timing_declarations(child) for child in value]
    if not isinstance(value, dict):
        return value
    result = {key: migrate_reference_timing_declarations(child) for key, child in value.items()}
    if isinstance(result.get("background_run"), dict):
        result["background_run"].setdefault("timing", None)
    return result


def restore_reference_selection(payload: dict, run: Run) -> dict:
    """Decode only keys established by this source's number and acquired periods."""
    if "legacy_run_number" not in payload:
        return payload
    result = dict(payload)
    number = result.pop("legacy_run_number")
    source = source_run_of(run)
    count = len(run.grouping.get("period_histograms") or []) or int(
        run.metadata.get("period_count", 1)
    )
    if number is None or int(number) == source:
        return result | {"run_number": source, "period_number": 1}
    for period in range(1, count + 1):
        if int(number) == encode_period_run_number(source, period):
            return result | {"run_number": source, "period_number": period}
    return result | {"run_number": number, "period_number": 1}


def complete_reference_restore(payload: dict, run: Run) -> None:
    """Replace a pending historical key once the matching acquisition is restored."""
    if "legacy_run_number" in payload:
        selected = restore_reference_selection(payload, run)
        payload.clear()
        payload.update(selected)


def migrate_reference_scale_declarations(value):
    """Separate historical fallback snapshots from explicit authored scale intent."""
    from math import isfinite

    if isinstance(value, list):
        return [migrate_reference_scale_declarations(child) for child in value]
    if not isinstance(value, dict):
        return value
    result = {key: migrate_reference_scale_declarations(child) for key, child in value.items()}
    payload = result.get("background_run")
    if isinstance(payload, dict) and "scale_request" not in payload:
        fields = ("scale", "good_frames_sample", "good_frames_reference")
        historical = {key: payload.pop(key) for key in fields if key in payload}
        request = {"mode": "automatic"}
        if set(historical) == {"scale"} and not isinstance(historical["scale"], bool):
            try:
                coefficient = float(historical["scale"])
            except (TypeError, ValueError, OverflowError):
                coefficient = float("nan")
            if isfinite(coefficient) and coefficient > 0:
                request = {"mode": "manual", "coefficient": coefficient}
        payload["scale_request"] = request
        if historical:
            payload["historical_scale"] = historical | {"exposure_origin": "unverified"}
    return result
