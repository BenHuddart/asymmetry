"""Permanent correction-request conversion for released project payloads."""

from __future__ import annotations

from asymmetry.core.transform.background import fixed_background_values

_LEGACY_MODES = {
    "": "off",
    "file": "from_file",
    "from_file": "from_file",
    "manual": "manual",
    "load": "manual",
    "value": "manual",
    "calibrate": "manual",
    "count_fit": "manual",
    "maxent_fit": "manual",
    "estimate": "estimate",
    "off": "off",
    "none": "off",
}


def correction_request(payload: dict) -> None:
    """Keep released precedence, separating coefficient origin from request mode."""
    for key in ("background_fixed_values", "background_fix", "bkg_fix"):
        if payload.get(key) is not None:
            payload["background_fixed_values"] = list(
                fixed_background_values({"background_fixed_values": payload[key]})
            )
            break
    for key in ("background_fix", "bkg_fix"):
        payload.pop(key, None)
    has_deadtime = any(
        key in payload for key in ("deadtime_mode", "deadtime_method", "deadtime_loaded_us")
    ) or payload.get("deadtime_correction", False)
    if not has_deadtime:
        return
    raw_mode = (
        str(
            payload.get(
                "deadtime_mode",
                payload.get(
                    "deadtime_method", "file" if payload.get("deadtime_correction") else "off"
                ),
            )
        )
        .strip()
        .lower()
    )
    mode = _LEGACY_MODES.get(raw_mode, raw_mode)
    method = str(payload.get("deadtime_method") or mode).strip().lower()
    # File coefficients were borrowed only when neither the request nor its
    # recorded method denoted an authored table. Freeze that distinction once.
    if "deadtime_file_us" not in payload and mode not in ("manual", "estimate"):
        if _LEGACY_MODES.get(method, method) in ("from_file", "off"):
            payload["deadtime_file_us"] = payload.get(
                "dead_time_us", payload.get("deadtime_loaded_us", [])
            )
        else:
            payload["deadtime_file_us"] = []
    if "dead_time_us" not in payload and "deadtime_loaded_us" in payload:
        payload["dead_time_us"] = payload["deadtime_loaded_us"]
    payload.pop("deadtime_loaded_us", None)
    payload["deadtime_mode"] = mode
    # This is estimator/table provenance, including historical spellings.
    if mode in ("manual", "estimate"):
        payload.setdefault("deadtime_method", method)


def project_corrections(data: dict) -> None:
    """Visit correction settings, never frozen execution/calibration records."""
    for dataset in data.get("datasets", []):
        if not isinstance(dataset, dict):
            continue
        for key in ("grouping", "grouping_overrides"):
            if isinstance(dataset.get(key), dict):
                correction_request(dataset[key])
        for source in dataset.get("sources", []):
            if isinstance(source, dict):
                for key in ("grouping", "grouping_overrides"):
                    if isinstance(source.get(key), dict):
                        correction_request(source[key])
    for profile in data.get("grouping_profiles", []):
        if not isinstance(profile, dict):
            continue
        policy = profile.get("background_policy", {})
        if isinstance(policy, dict) and isinstance(policy.get("details"), dict):
            correction_request(policy["details"])
