"""Restore a saved fit overlay for the observations available on its projection."""

from asymmetry.core.fitting.fit_curves import dense_fit_curve
from asymmetry.core.fitting.input_attribution import saved_fit_observations
from asymmetry.core.fitting.model_comparison import saved_fit_curve_bounds, saved_fit_model
from asymmetry.gui.panels.plot_panel import SINGLE_FIT_ID


def restore_saved_fit_overlay(panel, record, projection, axis_key, slot, name, fit_id):
    """Update a cached alternate only when its drawable extent changes."""
    if not panel._has_mpl:
        return
    run_number = record.run_number
    displayed = panel._fit_curves_by_key.get((run_number, axis_key, fit_id))
    drawable = saved_fit_model(slot)
    bounds = None
    if drawable is not None:
        model, values = drawable
        dataset = record
        may_widen = _recorded_extent_can_widen(slot)
        open_ended = slot.fit_range is None or None in slot.fit_range.values()
        if axis_key != projection and (may_widen or open_ended):
            targets = panel._vector_subplot_datasets.get(axis_key, ())
            plotted = next((target for target in targets if target.run_number == run_number), None)
            dataset = panel.get_analysis_dataset(plotted)
        observations = saved_fit_observations(slot, dataset) if may_widen else None
        bounds = saved_fit_curve_bounds(model, slot, dataset or record, observations)
    if bounds is None:
        if displayed is not None or fit_id == SINGLE_FIT_ID:
            panel.clear_run_fit(run_number, fit_id)
        return
    if displayed is not None and fit_id != SINGLE_FIT_ID:
        x = displayed[0]
        lower, upper = model.time_response.curve_times(*bounds, 2)
        if x.size and (x[0], x[-1]) == (lower, upper):
            return
    x, y = dense_fit_curve(model, values, *bounds)
    payload = (x, y, name, (), None, model.formula_string(), axis_key)
    panel.set_global_fits({run_number: payload}, fit_id=fit_id)


def _recorded_extent_can_widen(slot):
    """Only hash observations when a recorded centre could extend a literal bound."""
    recorded = (slot.result or {}).get("input_attribution") or {}
    lower, upper = recorded.get("min"), recorded.get("max")
    window = slot.fit_range
    return window is not None and (
        (lower is not None and window["min"] is not None and lower < window["min"])
        or (upper is not None and window["max"] is not None and upper > window["max"])
    )
