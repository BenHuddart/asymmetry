"""Translate historical toolbar settings to F–B Auto/Hold policies."""

from math import isfinite

VIEW_MODE_COUNT = 3


def migrate_display_bunching(data: dict) -> dict:
    """Preserve recorded bin widths without applying toolbar grouping factors twice."""
    migrated = dict(data, schema_version=29)
    views = data.get("view_modes_state")
    if not isinstance(views, dict) or not isinstance(views.get("modes"), list):
        return migrated
    plot = data.get("plot_state")
    panel_factor = plot.get("bunch_factor", 1) if isinstance(plot, dict) else 1
    active = views.get("active_index", 0)
    migrated["view_modes_state"] = dict(
        views,
        modes=[
            dict(mode, bunch_factor=panel_factor if index == active else 1)
            if isinstance(mode, dict)
            else mode
            for index, mode in enumerate(views["modes"])
        ],
    )
    return migrated


def _legacy_factor(value) -> int:
    try:
        return max(1, min(1000, int(value)))
    except (ValueError, TypeError, OverflowError):
        return 1


def _legacy_axis(mode: dict, axis: str) -> dict | None:
    keys = (f"{axis}_min", f"{axis}_max")
    if not any(key in mode for key in keys):
        return None
    defaults = (0.0, 10.0) if axis == "x" else (-30.0, 30.0)
    try:
        held = sorted(float(mode.get(key, default)) for key, default in zip(keys, defaults))
    except (ValueError, TypeError, OverflowError):
        return None
    return {"auto": False, "held": held, "quantity": None} if all(map(isfinite, held)) else None


def migrate_saved_views(data: dict) -> None:
    """Recover numeric slots and prefer the saved live F–B policy for the active slot."""
    plot = data.get("plot_state") or {}
    views = data.get("view_modes_state")
    if views is None:
        views = {}
    if not views and not plot:
        return
    for panel in (plot, plot.get("frequency_plot_state") or {}):
        if "bunch_factor" in panel:
            panel["bunch_factor"] = _legacy_factor(panel["bunch_factor"])

    try:
        active = max(0, min(VIEW_MODE_COUNT - 1, int(views.get("active_index", 0))))
    except (ValueError, TypeError, OverflowError):
        active = 0
    modes = views.get("modes", [])
    if not isinstance(modes, list):
        raise TypeError("view_modes_state.modes must be a list")
    workspace = plot.get("workspace_state") or {}
    panel_axes = (plot.get("axis_limits") or {}).get("axes", {})
    is_fb = (
        plot.get("time_view_mode", "fb_asymmetry") == "fb_asymmetry"
        and workspace.get("active_view") != "reconstruction"
    )
    fb_axes = (
        panel_axes
        if is_fb
        else {
            key: entry
            for key, entry in panel_axes.items()
            if key == "x" or str(entry.get("quantity")).startswith("fb_asymmetry|")
        }
    )
    slots: list[dict | None] = [None] * VIEW_MODE_COUNT
    for index in range(VIEW_MODE_COUNT):
        if index >= len(modes) and index != active:
            continue
        mode = modes[index] if index < len(modes) else {}
        if not isinstance(mode, dict):
            raise TypeError(f"view_modes_state.modes[{index}] must be a mapping")
        fallback_factor = plot.get("bunch_factor", 1) if index == active else 1
        factor = _legacy_factor(mode.get("bunch_factor", fallback_factor))
        axes = {
            axis: limits for axis in ("x", "y") if (limits := _legacy_axis(mode, axis)) is not None
        }
        if index == active:
            if not is_fb:
                # Counts and reconstruction limits do not measure F–B asymmetry.
                axes.pop("y", None)
            axes.update(fb_axes)
        elif factor == 1 and mode == {
            "bunch_factor": 1,
            "x_min": 0.0,
            "x_max": 10.0,
            "y_min": -30.0,
            "y_max": 30.0,
        }:
            continue
        if index == active and not modes and not fb_axes and factor == 1:
            continue
        slots[index] = {"bunch_factor": factor, "axis_limits": {"axes": axes}}

    data["view_modes_state"] = {"active_index": active, "modes": slots}
